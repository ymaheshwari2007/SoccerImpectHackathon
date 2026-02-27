"""
Two-pass graph collection across multiple matches.

Why two passes?
  Pass 1: Load every match, clean up the data, and remember what values we saw.
  Pass 2: Now that we know ALL possible player IDs, event types, etc, we can
           fit the label encoders once and apply them consistently to every match.

This is what makes pooled training work — without it, "player #42" could mean
different people in different matches, and the GAT would learn garbage.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from torch_geometric.data import Data

from ..config import PipelineConfig
from ..data.fetch import MatchContext, extract_match_context, load_match_events
from ..data.possession import segment_possessions
from ..data.preprocess import (
    GlobalEncoders,
    apply_encoders,
    collect_unique_values,
    fill_nan_columns,
    fit_global_encoders,
)
from .features import build_graph


@dataclass
class MatchGraphs:
    """All the graphs from one match, bundled with its context metadata."""
    match_id: int
    context: MatchContext
    graphs: list[Data] = field(default_factory=list)


def build_all_match_graphs(
    match_ids: list[int],
    config: PipelineConfig,
    verbose: bool = True,
) -> tuple[list[MatchGraphs], GlobalEncoders]:
    """The heart of data preparation: load matches → fit encoders → build graphs.

    Pass 1 — Load and preprocess:
      For each match, download the events, clean NaNs, parse timestamps.
      Store everything in memory (~70MB for 34 Bundesliga matches).

    Pass 2 — Encode and build:
      Fit ONE set of label encoders on the union of all matches.
      Then encode each match, segment into possessions, and build graphs.

    Returns (list of MatchGraphs, fitted GlobalEncoders).
    """
    import pandas as pd  # local import to keep top-level lightweight

    # Pass 1: Load and preprocess all matches
    raw_data: list[tuple[int, object, MatchContext, pd.DataFrame]] = []

    for i, mid in enumerate(match_ids, 1):
        if verbose:
            print(f"  [{i}/{len(match_ids)}] Loading match {mid}...")
        try:
            dataset, events_df = load_match_events(mid, config.competition_id)
        except Exception as exc:
            if verbose:
                print(f"    ✗ Failed to load: {exc}")
            continue

        ctx = extract_match_context(dataset, mid, config.team_name, config.squad)
        if ctx.target_team_id is None:
            if verbose:
                print(f"    ✗ Could not identify {config.team_name}")
            continue

        df = fill_nan_columns(events_df)
        raw_data.append((mid, dataset, ctx, df))

    if verbose:
        print(f"  Pass 1 complete: {len(raw_data)} matches loaded")

    # Fit global encoders on the union of all match data
    all_dfs = [item[3] for item in raw_data]
    unique_vals = collect_unique_values(all_dfs)
    encoders = fit_global_encoders(unique_vals)

    if verbose:
        print(f"  Global encoders fit: {len(encoders.player.classes_)} players, "
              f"{len(encoders.event_type.classes_)} event types")

    # Pass 2: Encode and build graphs
    results: list[MatchGraphs] = []

    for mid, dataset, ctx, df in raw_data:
        df_enc = apply_encoders(df, encoders)
        possessions = segment_possessions(df_enc, config.graph.no_video_threshold)

        graphs = []
        for play in possessions:
            g = build_graph(play, ctx.home_team_id,
                            config.graph, config.pitch, config.scoring)
            if g is not None:
                graphs.append(g)

        mg = MatchGraphs(match_id=mid, context=ctx, graphs=graphs)
        results.append(mg)

        if verbose:
            target_count = sum(1 for g in graphs if g.team_id == ctx.target_team_id)
            print(f"  [{mid}] {len(graphs)} graphs ({target_count} {config.team_name})")

    if verbose:
        total = sum(len(mg.graphs) for mg in results)
        print(f"  Pass 2 complete: {total} total graphs from {len(results)} matches")

    return results, encoders


def pool_all_graphs(match_graphs_list: list[MatchGraphs]) -> list[Data]:
    """Flatten all graphs from all matches into one big list for training.

    The GAT doesn't care which match a graph came from — it just needs
    a pile of (graph, score) pairs to learn from.
    """
    return [g for mg in match_graphs_list for g in mg.graphs]
