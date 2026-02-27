"""
Synergy accumulation and interaction matrix construction.

After the GAT scores every possession, we need to answer: "which PLAYERS
played well together?" The GAT only scores possessions (whole sequences),
so we attribute that score to every pair of players who appeared in the same
possession.

The interaction matrix uses involvement-weighted averages:
  - Diagonal [i,i]: diag_sum[pid] / field_plays[pid]
      = avg score in plays pid touched the ball
        × P(pid touched the ball in a play | pid was on the field)
  - Off-diagonal [i,j]: score_sum[(pi,pj)] / field_pairs[(pi,pj)]
      = avg hop-weighted score when pi–pj had an edge
        × P(pi–pj edge existed in a play | both were on the field)

This penalises passengers and rewards players who are genuinely involved.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from torch_geometric.data import Data

from ..data.fetch import MatchContext, get_lineup_at_time


@dataclass
class SynergyAccumulator:
    """Running totals that build up across all matches.

    We keep separate sums and counts so we can compute averages at the end.
    This is mutable — each match adds to the running totals.
    """
    # Off-diagonal: how well did each pair play together?
    score_sum: dict = field(default_factory=lambda: defaultdict(float))   # (pid_a, pid_b) → sum of hop-weighted scores
    pair_count: dict = field(default_factory=lambda: defaultdict(float))  # (pid_a, pid_b) → number of co-edge plays

    # Diagonal: how well did each player play individually?
    diag_sum: dict = field(default_factory=lambda: defaultdict(float))    # pid → sum of scores when pid touched ball
    diag_count: dict = field(default_factory=lambda: defaultdict(float))  # pid → number of own-team possessions pid appeared in

    # Player metadata (accumulated across matches)
    player_names: dict[str, str] = field(default_factory=dict)
    player_positions: dict[str, str] = field(default_factory=dict)
    player_groups: dict[str, str] = field(default_factory=dict)

    # Lineup-based denominators (incremented for every possession, both teams)
    field_plays: dict = field(default_factory=lambda: defaultdict(int))
    # pid → total possessions while pid was on the field

    field_pairs: dict = field(default_factory=lambda: defaultdict(int))
    # (min_pid, max_pid) → total possessions while BOTH were on the field


def merge_match_context(acc: SynergyAccumulator, ctx: MatchContext) -> None:
    """Pull player names/positions from a match into the accumulator."""
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

    For every possession (both teams):
      1. Get the target team's lineup at that moment
      2. Increment field_plays for each roster player in the lineup
      3. Increment field_pairs for each pair of roster players in the lineup

    For own-team possessions only:
      4. Increment diag_sum/diag_count for each player who touched the ball
      5. Increment score_sum/pair_count for each edge-connected player pair
    """
    roster = ctx.roster
    target_tid = ctx.target_team_id

    for g, ps in zip(graphs, predictions):
        # ── Lineup-based field_plays and field_pairs (both teams' possessions) ──
        lineup = get_lineup_at_time(ctx, target_tid, g.period_id, g.t_start)
        on_field = [pid for pid in lineup if pid in roster]

        for pid in on_field:
            acc.field_plays[pid] += 1

        for i in range(len(on_field)):
            for j in range(i + 1, len(on_field)):
                pair = (min(on_field[i], on_field[j]), max(on_field[i], on_field[j]))
                acc.field_pairs[pair] += 1

        # ── Synergy scoring only for own-team possessions ──
        if g.team_id != target_tid:
            continue

        # Diagonal: every roster player who touched the ball gets credit
        unique_pids = {p for p in g.player_ids if p != "UNKNOWN" and p not in ("TERMINAL",) and p in roster}
        for pid in unique_pids:
            acc.diag_sum[pid] += ps
            acc.diag_count[pid] += 1.0

        # Off-diagonal: roster player pairs connected by graph edges
        # Use max hop weight per pair (a pair may share multiple edges)
        pair_w: dict[tuple[str, str], float] = {}
        for e in range(g.edge_index.shape[1]):
            sp = g.player_ids[g.edge_index[0, e].item()]
            dp = g.player_ids[g.edge_index[1, e].item()]

            if sp in ("UNKNOWN", "TERMINAL") or dp in ("UNKNOWN", "TERMINAL") or sp == dp:
                continue
            if sp not in roster or dp not in roster:
                continue

            pair = (min(sp, dp), max(sp, dp))
            w = 1.0 if g.edge_attr[e, 0].item() == 1.0 else g.edge_attr[e, 4].item()
            pair_w[pair] = max(pair_w.get(pair, 0.0), w)

        for (pi, pj), w in pair_w.items():
            acc.score_sum[(pi, pj)] += ps * w
            acc.pair_count[(pi, pj)] += 1.0


@dataclass
class InteractionMatrix:
    """The N×N synergy matrix — this is what the MILP optimizer uses.

    matrix[i,i] = diag_sum / field_plays  (individual involvement-weighted score)
    matrix[i,j] = score_sum / field_pairs (pairwise involvement-weighted score)
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

    Only includes players with at least min_appearances possessions where they
    actually touched the ball (diag_count threshold).

    Diagonal: diag_sum[pid] / field_plays[pid]
    Off-diagonal: score_sum[(pi,pj)] / field_pairs[(pi,pj)]
    """
    qualified = sorted(pid for pid, c in acc.diag_count.items() if c >= min_appearances)
    n = len(qualified)
    pid_to_idx = {pid: i for i, pid in enumerate(qualified)}

    matrix = np.zeros((n, n), dtype=np.float64)

    # Diagonal: involvement-weighted average GAT score
    for pid in qualified:
        i = pid_to_idx[pid]
        fp = acc.field_plays.get(pid, 0)
        matrix[i, i] = acc.diag_sum[pid] / fp if fp > 0 else 0.0

    # Off-diagonal: pair involvement-weighted average hop score (symmetric)
    for (pi, pj), s in acc.score_sum.items():
        if pi in pid_to_idx and pj in pid_to_idx:
            i, j = pid_to_idx[pi], pid_to_idx[pj]
            fp = acc.field_pairs.get((pi, pj), 0)
            val = s / fp if fp > 0 else 0.0
            matrix[i, j] = val
            matrix[j, i] = val

    names = [acc.player_names.get(pid, pid) for pid in qualified]
    df = pd.DataFrame(matrix, index=names, columns=names)

    return InteractionMatrix(
        matrix=matrix, player_ids=qualified, player_names=names, dataframe=df,
    )
