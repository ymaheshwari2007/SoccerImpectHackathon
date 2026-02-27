"""
Synergy accumulation and interaction matrix construction.

After the GAT scores every possession, we need to answer: "which PLAYERS
played well together?" The GAT only scores possessions (whole sequences),
so we attribute that score to every pair of players who appeared in the same
possession.

Think of it like movie credits — if a movie is great, everyone in it gets
credit. But a player who appeared in one scene (1-hop edge) gets more credit
than someone who was loosely connected (synthetic skip-edge).

The end product is an N×N interaction matrix where:
  - Diagonal [i,i]: average score when player i participated
  - Off-diagonal [i,j]: average score when players i and j appeared together,
    weighted by how directly they interacted (edge hop weight)
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from torch_geometric.data import Data

from ..config import IndividualStatsConfig
from ..data.fetch import MatchContext


@dataclass
class SynergyAccumulator:
    """Running totals that build up across all matches.

    We keep separate sums and counts so we can compute averages at the end.
    This is mutable — each match adds to the running totals.
    """
    # Off-diagonal: how well did each pair play together?
    score_sum: dict = field(default_factory=lambda: defaultdict(float))   # (pid_a, pid_b) → sum of scores
    pair_count: dict = field(default_factory=lambda: defaultdict(float))  # (pid_a, pid_b) → number of co-appearances

    # Diagonal: how well did each player play individually?
    diag_sum: dict = field(default_factory=lambda: defaultdict(float))    # pid → sum of scores
    diag_count: dict = field(default_factory=lambda: defaultdict(float))  # pid → number of possessions

    # Player metadata (accumulated across matches)
    player_names: dict[str, str] = field(default_factory=dict)
    player_positions: dict[str, str] = field(default_factory=dict)
    player_groups: dict[str, str] = field(default_factory=dict)

    # Individual stats — raw event counts per player (goals, shots, tackles, etc)
    player_stats: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    # How many plays each player appeared in (for appearance bonus)
    player_appearances: dict[str, int] = field(default_factory=lambda: defaultdict(int))


def merge_match_context(acc: SynergyAccumulator, ctx: MatchContext) -> None:
    """Pull player names/positions from a match into the accumulator.

    Later matches overwrite earlier ones — that's fine because a player's
    name doesn't change, and we want the most recent position if it updated.
    """
    acc.player_names.update(ctx.player_names)
    acc.player_positions.update(ctx.player_positions)
    acc.player_groups.update(ctx.player_groups)


def accumulate_match_synergy(
    acc: SynergyAccumulator,
    graphs: list[Data],
    predictions: list[float],
    ctx: MatchContext,
) -> None:
    """Add one match's worth of synergy data to the running totals.

    Only looks at possessions by the target team (we don't care how well
    the opponent's players synergise). Also filters to roster players only —
    opponent players appear in duel events and we don't want them polluting
    the matrix.

    For each possession:
      - Every roster player who touched the ball gets diagonal credit
      - Every PAIR of roster players connected by an edge gets off-diagonal
        credit, weighted by hop distance (direct pass = weight 1.0,
        synthetic 2-hop = weight from edge attribute)
    """
    roster = ctx.roster
    target_tid = ctx.target_team_id

    for g, ps in zip(graphs, predictions):
        is_own_possession = (g.team_id == target_tid)

        # Tally individual event stats from ALL possessions (both teams)
        # because defensive stats (saves, interceptions, blocks) happen during
        # opponent possessions — if we only count own-team possessions, goalkeepers
        # get zero saves and defenders get almost no interceptions.
        if hasattr(g, "event_data"):
            for ev in g.event_data:
                pid = ev["player_id"]
                if pid == "UNKNOWN" or pid not in roster:
                    continue
                etype = ev["event_type"]
                result = ev["result"].upper()
                success = ev["success"]

                if etype == "SHOT":
                    acc.player_stats[pid]["total_shots"] += 1
                    if "GOAL" in result:
                        acc.player_stats[pid]["goals"] += 1
                        acc.player_stats[pid]["shots_on_target"] += 1
                    elif "SAVED" in result:
                        acc.player_stats[pid]["shots_on_target"] += 1
                elif etype == "PASS":
                    if success:
                        acc.player_stats[pid]["successful_passes"] += 1
                    if success and ev["end_x"] > 35.0:
                        acc.player_stats[pid]["key_passes"] += 1
                    # Progressive pass: moves ball ≥10m toward opponent goal
                    start_x = ev.get("coord_x", 0.0) or 0.0
                    end_x = ev.get("end_x", 0.0) or 0.0
                    if success and (end_x - start_x) >= 10.0:
                        acc.player_stats[pid]["progressive_passes"] += 1
                elif etype == "CARRY":
                    acc.player_stats[pid]["carries"] += 1
                    # Progressive carry: moves ball ≥5m toward opponent goal
                    start_x = ev.get("coord_x", 0.0) or 0.0
                    end_x = ev.get("end_x", 0.0) or 0.0
                    if (end_x - start_x) >= 5.0:
                        acc.player_stats[pid]["progressive_carries"] += 1
                    # Carry into final third (x > 17.5 in secondspectrum = last third)
                    if start_x <= 17.5 and end_x > 17.5:
                        acc.player_stats[pid]["carries_into_final_third"] += 1
                elif etype == "DUEL":
                    if result == "WON":
                        acc.player_stats[pid]["successful_duels"] += 1
                    acc.player_stats[pid]["total_duels"] += 1
                elif etype == "INTERCEPTION":
                    acc.player_stats[pid]["interceptions"] += 1
                elif etype == "CLEARANCE":
                    acc.player_stats[pid]["clearances"] += 1
                elif "BLOCK" in etype:
                    acc.player_stats[pid]["blocks"] += 1
                elif etype == "RECOVERY":
                    acc.player_stats[pid]["recoveries"] += 1
                elif etype == "GOALKEEPER":
                    gk_type = str(ev.get("goalkeeper_type", "NONE")).upper()
                    if gk_type == "SAVE":
                        acc.player_stats[pid]["saves"] += 1

        # Synergy scoring only for own-team possessions
        if not is_own_possession:
            continue

        # Diagonal: every roster player in this possession gets credit
        unique_pids = {p for p in g.player_ids if p != "UNKNOWN" and p in roster}
        for pid in unique_pids:
            acc.diag_sum[pid] += ps
            acc.diag_count[pid] += 1.0
            acc.player_appearances[pid] += 1

        # Off-diagonal: find pairs of roster players connected by edges
        # Use the max weight per pair (a pair might share multiple edges)
        pair_w: dict[tuple[str, str], float] = {}
        for e in range(g.edge_index.shape[1]):
            sp = g.player_ids[g.edge_index[0, e].item()]
            dp = g.player_ids[g.edge_index[1, e].item()]

            # Skip self-loops, unknowns, and opponent players
            if sp == "UNKNOWN" or dp == "UNKNOWN" or sp == dp:
                continue
            if sp not in roster or dp not in roster:
                continue

            # Canonical pair ordering so (A,B) and (B,A) map to the same entry
            pair = (min(sp, dp), max(sp, dp))

            # Weight: 1.0 for real edges (type=1.0), otherwise use the distance feature
            w = 1.0 if g.edge_attr[e, 0].item() == 1.0 else g.edge_attr[e, 4].item()
            pair_w[pair] = max(pair_w.get(pair, 0.0), w)

        for (pi, pj), w in pair_w.items():
            acc.score_sum[(pi, pj)] += ps * w
            acc.pair_count[(pi, pj)] += 1.0


@dataclass
class InteractionMatrix:
    """The N×N synergy matrix — this is what the MILP optimizer uses.

    matrix[i,j] tells you: on average, how good were possessions where
    players i and j both appeared? Higher = better synergy.
    """
    matrix: np.ndarray
    player_ids: list[str]
    player_names: list[str]
    dataframe: pd.DataFrame       # same data but with player names as row/col labels


def compute_psi(
    acc: SynergyAccumulator,
    qualified: list[str],
    stats_cfg: IndividualStatsConfig,
) -> dict[str, float]:
    """Position-Specific Skill Index for each qualified player.

    The idea: a striker's value comes from goals and shots on target,
    a defender's from tackles and interceptions. We z-score normalise
    each stat within the qualified pool, then dot-product with the
    position-group-specific weights. This gives one number per player
    that says "how good are they at what their position is supposed to do?"
    """
    weight_map = {
        "Attacker": stats_cfg.attacker_weights,
        "Midfielder": stats_cfg.midfielder_weights,
        "Defender": stats_cfg.defender_weights,
        "Goalkeeper": stats_cfg.goalkeeper_weights,
    }

    # Collect all stat names used across any position group
    all_stat_names = set()
    for w in weight_map.values():
        all_stat_names.update(w.keys())

    # Build per-appearance rate vectors (not raw counts — a player with 50 apps
    # and 14 progressive carries is better than 249 apps and 44 carries)
    raw = {}
    for s in all_stat_names:
        raw[s] = np.array([
            float(acc.player_stats[pid].get(s, 0)) / max(acc.player_appearances.get(pid, 1), 1)
            for pid in qualified
        ])

    # Z-score each stat across the qualified pool
    z_scored = {}
    for s, vals in raw.items():
        mu, sigma = vals.mean(), vals.std()
        z_scored[s] = (vals - mu) / sigma if sigma > 0 else np.zeros_like(vals)

    # Dot-product with position-group weights to get PSI per player
    psi = {}
    for idx, pid in enumerate(qualified):
        group = acc.player_groups.get(pid, "Midfielder")  # default to mid if unknown

        # Goalkeepers get a special PSI: save rate (saves / appearances)
        # Z-scoring saves across 20 outfielders + 2 GKs is meaningless —
        # just directly measure how many saves per appearance the GK makes.
        if group == "Goalkeeper":
            saves = float(acc.player_stats[pid].get("saves", 0))
            apps = float(acc.player_appearances.get(pid, 1))
            save_rate = saves / max(apps, 1.0)
            # Scale so it's comparable to outfielder PSI range
            psi[pid] = save_rate * 10.0
        else:
            weights = weight_map.get(group, stats_cfg.midfielder_weights)
            score = sum(weights.get(s, 0.0) * z_scored[s][idx] for s in weights)
            psi[pid] = score

    return psi


def compute_appearance_bonus(
    acc: SynergyAccumulator,
    qualified: list[str],
) -> dict[str, float]:
    """Logarithmic appearance bonus — rewards players who actually play regularly.

    Uses log(1 + appearances) so the first 100 appearances matter a lot more
    than going from 400 to 500. Normalised to [0, 1] range.
    """
    raw = {pid: math.log(1 + acc.player_appearances.get(pid, 0)) for pid in qualified}
    max_val = max(raw.values()) if raw else 1.0
    if max_val == 0:
        max_val = 1.0
    return {pid: v / max_val for pid, v in raw.items()}


def build_interaction_matrix(
    acc: SynergyAccumulator,
    min_appearances: int = 150,
    stats_cfg: IndividualStatsConfig | None = None,
) -> InteractionMatrix:
    """Build the final N×N matrix from accumulated synergy stats.

    Only includes players with enough appearances (min_appearances) to get
    reliable estimates. A player with 10 appearances would have super noisy
    averages, so we exclude them.

    Diagonal: w_gat * norm_GAT + w_psi * norm_PSI + w_app * norm_App
              (all components min-max normalised to [0,1] before weighting)
    Off-diagonal: avg hop-weighted pair score for (i, j)
    """
    # Filter to players with enough data
    qualified = sorted(pid for pid, c in acc.diag_count.items() if c >= min_appearances)
    n = len(qualified)
    pid_to_idx = {pid: i for i, pid in enumerate(qualified)}

    # Compute individual stat bonuses if config provided
    if stats_cfg is not None:
        psi = compute_psi(acc, qualified, stats_cfg)
        app_bonus = compute_appearance_bonus(acc, qualified)
        w_gat = stats_cfg.w_gat
        w_psi = stats_cfg.w_psi
        w_app = stats_cfg.w_app
    else:
        psi = {pid: 0.0 for pid in qualified}
        app_bonus = {pid: 0.0 for pid in qualified}
        w_gat, w_psi, w_app = 1.0, 0.0, 0.0

    matrix = np.zeros((n, n), dtype=np.float64)

    # Compute raw GAT averages
    gat_raw = np.array([acc.diag_sum[pid] / acc.diag_count[pid] for pid in qualified])

    # Min-max normalise GAT to [0, 1]
    gat_min, gat_max = gat_raw.min(), gat_raw.max()
    norm_gat = (gat_raw - gat_min) / (gat_max - gat_min) if gat_max > gat_min else np.zeros_like(gat_raw)

    # Min-max normalise PSI to [0, 1]
    psi_raw = np.array([psi.get(pid, 0.0) for pid in qualified])
    psi_min, psi_max = psi_raw.min(), psi_raw.max()
    norm_psi = (psi_raw - psi_min) / (psi_max - psi_min) if psi_max > psi_min else np.zeros_like(psi_raw)

    # app_bonus is already [0, 1] — no normalisation needed

    # Fill diagonal: weighted average of normalised components
    for idx, pid in enumerate(qualified):
        i = pid_to_idx[pid]
        matrix[i, i] = (w_gat * norm_gat[idx]
                         + w_psi * norm_psi[idx]
                         + w_app * app_bonus.get(pid, 0.0))

    # Fill off-diagonal: average pairwise synergy (symmetric)
    for (pi, pj), s in acc.score_sum.items():
        if pi in pid_to_idx and pj in pid_to_idx:
            i, j = pid_to_idx[pi], pid_to_idx[pj]
            c = acc.pair_count[(pi, pj)]
            matrix[i, j] = s / c
            matrix[j, i] = s / c

    names = [acc.player_names.get(pid, pid) for pid in qualified]
    df = pd.DataFrame(matrix, index=names, columns=names)

    return InteractionMatrix(
        matrix=matrix, player_ids=qualified, player_names=names, dataframe=df,
    )
