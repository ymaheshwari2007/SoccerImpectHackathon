"""
Play scoring and PyTorch Geometric graph construction.

This is where football events become machine learning inputs. Each possession
(a sequence of events while one team has the ball) gets turned into a graph:

  - Each event (pass, dribble, shot, ...) becomes a NODE with 6 features
  - Sequential events are connected by EDGES (event i → event i+1)
  - We also add "synthetic" skip-edges (i → i+2, i → i+3) so the GAT can
    look a couple steps ahead — like giving it peripheral vision

Each graph also gets a SCORE label: a number reflecting how dangerous/valuable
the possession was. This is what the GAT learns to predict.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from ..config import FieldConfig, GraphConfig, ScoringConfig


def get_attacking_goal_x(
    team_id: str, period_id: int, home_team_id: str, field: FieldConfig,
) -> float:
    """Which end of the pitch is this team attacking?

    In the secondspectrum coordinate system, the home team attacks the
    positive-x goal in the first half, then they swap. This returns the
    x-coordinate of the goal the possessing team is trying to score in.
    """
    is_home = str(team_id) == str(home_team_id)
    attacks_positive = (is_home and period_id == 1) or (not is_home and period_id == 2)
    return field.field_half if attacks_positive else -field.field_half


def score_play(
    play_clean: pd.DataFrame,
    attacking_goal_x: float,
    field: FieldConfig,
    scoring: ScoringConfig,
) -> float:
    """Score a possession: how dangerous was it?

    The score combines four things:
      1. DISPLACEMENT: did the ball move closer to goal? (normalised to [-1, 1])
      2. ADVANCEMENT: how deep into the opponent's half did it get? ([0, 1])
      3. TIME DECAY: quick attacks score higher via sigmoid (a 3-second counter
         attack is worth more than a 30-second sideways passing sequence)
      4. END BONUS: flat reward/penalty for how the play ended (goal > shot > turnover)

    Final score = (displacement × advancement × sigmoid) + end_bonus
    """
    if len(play_clean) == 0:
        return 0.0

    first, last = play_clean.iloc[0], play_clean.iloc[-1]
    x_s, y_s = float(first["coordinates_x"]), float(first["coordinates_y"])
    x_e, y_e = float(last["end_coordinates_x"]), float(last["end_coordinates_y"])

    # How much closer to goal did the ball get? (normalised by max pitch distance)
    d_s = ((x_s - attacking_goal_x) ** 2 + y_s ** 2) ** 0.5
    d_e = ((x_e - attacking_goal_x) ** 2 + y_e ** 2) ** 0.5
    disp = min(1.0, (d_s - d_e) / field.max_field_dist)

    # How far into the opponent's half? 0 = own goal, 1 = opponent goal
    sign = 1.0 if attacking_goal_x > 0 else -1.0
    adv = scoring.advancement_weight * max(0.0, min(1.0, x_e * sign / field.field_half))

    # Sigmoid of play duration — clamped to prevent math.exp overflow
    # (some plays have timestamps hundreds of seconds apart due to stoppages)
    t = float(last["timestamp_sec"] - first["timestamp_sec"])
    exp_arg = max(-500.0, min(500.0, -(t - scoring.sigmoid_shift)))
    sig = 1.0 / (1.0 + math.exp(exp_arg))

    mult = disp * adv * sig

    # What happened at the end of the play?
    lt = str(last["event_type"])
    lr = str(last["result"]).upper()
    if lt == "SHOT":
        bonus = scoring.goal_bonus if "GOAL" in lr else scoring.shot_bonus
    elif "OUT" in lr:
        bonus = scoring.out_of_bounds_penalty
    elif lt == "INTERCEPTION":
        bonus = scoring.intercept_penalty
    else:
        bonus = 0.0

    return mult + bonus


def build_graph(
    play: pd.DataFrame,
    home_team_id: str,
    graph_cfg: GraphConfig,
    field_cfg: FieldConfig,
    scoring_cfg: ScoringConfig,
) -> Data | None:
    """Turn one possession into a PyG graph. Returns None if the data is too messy.

    The graph looks like this:

        Node features (6 per node):
          [player_id_enc, coord_x, coord_y, event_type_enc, t_relative, under_pressure]

        Edge attributes (7 per edge):
          [edge_type, pass_type_enc, end_x, end_y, distance, result_enc, success]
          edge_type: 1.0 = real sequential edge, 2.0 = synthetic skip-connection

    We also attach metadata to the graph object (player_ids, team_id, etc)
    so the synergy accumulator can trace back which players were involved.
    """
    timestamps = play["timestamp_sec"].to_numpy()
    durations = np.append(np.diff(timestamps), 0.0)

    # Skip possessions with too much missing video data
    nv_mask = (play["event_type"] == "GENERIC:NO_VIDEO").to_numpy()
    if nv_mask.any() and durations[nv_mask].sum() > graph_cfg.no_video_threshold:
        return None

    # Remove NO_VIDEO events and bail if too few real events remain
    pc = play[~nv_mask].reset_index(drop=True)
    n = len(pc)
    if n < 2:
        return None

    t0 = pc["timestamp_sec"].iloc[0]

    # Build node features
    # Each row in the possession becomes a node
    feats = []
    for _, r in pc.iterrows():
        feats.append([
            float(r["player_id_enc"]),
            float(r["coordinates_x"]),
            float(r["coordinates_y"]),
            float(r["event_type_enc"]),
            float(r["timestamp_sec"] - t0),      # time relative to start of play
            float(r["is_under_pressure"]),
        ])
    x = torch.tensor(feats, dtype=torch.float)

    # Build edges
    # Real edges: event i → event i+1 (the natural sequence)
    # Synthetic edges: event i → event i+2, i+3, ... (skip-connections)
    src, dst, attrs = [], [], []
    REAL, SYNTH = 1.0, 2.0

    for i in range(n):
        r = pc.iloc[i]
        x1, y1 = r["coordinates_x"], r["coordinates_y"]
        x2, y2 = r["end_coordinates_x"], r["end_coordinates_y"]
        dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        real_attr = [REAL, float(r["pass_type_enc"]), float(x2), float(y2),
                     float(dist), float(r["result_enc"]), float(r["success"])]

        # Connect to next event (real edge)
        if i + 1 < n:
            src.append(i); dst.append(i + 1); attrs.append(real_attr)

        # Synthetic skip-connections (distance decays as 1/k)
        for k in range(2, 2 + graph_cfg.n_synthetic_edges):
            if i + k < n:
                src.append(i); dst.append(i + k)
                attrs.append([SYNTH, 0.0, 0.0, 0.0, 1.0 / k, 0.0, 0.0])

    ei = torch.tensor([src, dst], dtype=torch.long)
    ea = torch.tensor(attrs, dtype=torch.float)

    # Compute play score (this is the label the GAT will learn)
    goal_x = get_attacking_goal_x(
        str(pc.iloc[0]["team_id"]), int(pc.iloc[0]["period_id"]),
        home_team_id, field_cfg,
    )
    y = torch.tensor([score_play(pc, goal_x, field_cfg, scoring_cfg)], dtype=torch.float)

    # Pack everything into a PyG Data object
    g = Data(x=x, edge_index=ei, edge_attr=ea, y=y)

    # Attach metadata for the synergy stage (not used by the GAT itself)
    g.player_ids = pc["player_id"].tolist()
    g.team_id = str(pc.iloc[0]["team_id"])
    g.period_id = int(pc.iloc[0]["period_id"])
    g.t_start = float(pc.iloc[0]["timestamp_sec"])
    return g
