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

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from torch_geometric.data import Data

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
        # Skip opponent possessions
        if g.team_id != target_tid:
            continue

        # Diagonal: every roster player in this possession gets credit
        unique_pids = {p for p in g.player_ids if p != "UNKNOWN" and p in roster}
        for pid in unique_pids:
            acc.diag_sum[pid] += ps
            acc.diag_count[pid] += 1.0

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


def build_interaction_matrix(
    acc: SynergyAccumulator,
    min_appearances: int = 150,
) -> InteractionMatrix:
    """Build the final N×N matrix from accumulated synergy stats.

    Only includes players with enough appearances (min_appearances) to get
    reliable estimates. A player with 10 appearances would have super noisy
    averages, so we exclude them.

    Diagonal: avg score when player i participated
    Off-diagonal: avg hop-weighted pair score for (i, j)
    """
    # Filter to players with enough data
    qualified = sorted(pid for pid, c in acc.diag_count.items() if c >= min_appearances)
    n = len(qualified)
    pid_to_idx = {pid: i for i, pid in enumerate(qualified)}

    matrix = np.zeros((n, n), dtype=np.float64)

    # Fill diagonal: average individual performance
    for pid in qualified:
        i = pid_to_idx[pid]
        matrix[i, i] = acc.diag_sum[pid] / acc.diag_count[pid]

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
