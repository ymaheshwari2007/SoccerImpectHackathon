"""
The main orchestrator — runs the full pipeline from raw data to optimal XI.

7 stages:
  1. Fetch match list from IMPECT and filter to the target team
  2. Two-pass graph building (load all → fit global encoders → build graphs)
  3. Pool all graphs and train one shared PlayGAT
  4. Run inference on per-match graphs to get predicted scores
  5. Accumulate synergy stats (which player pairs played well together?)
  6. Build the N×N interaction matrix
  7. Solve MILP for the optimal starting XI

Each stage prints progress if verbose=True (the default).
"""
from __future__ import annotations

from dataclasses import dataclass

from .analysis.optimizer import OptimalSquad, solve_optimal_squad
from .analysis.synergy import (
    InteractionMatrix,
    SynergyAccumulator,
    accumulate_match_synergy,
    build_interaction_matrix,
    merge_match_context,
)
from .config import PipelineConfig
from .data.fetch import fetch_match_list, get_team_match_ids
from .data.preprocess import GlobalEncoders
from .graph.dataset import MatchGraphs, build_all_match_graphs, pool_all_graphs
from .model.inference import predict_scores
from .model.train import TrainResult, train_gat


@dataclass
class PipelineResult:
    """Everything the pipeline produces — keep this around for analysis/debugging."""
    match_graphs: list[MatchGraphs]
    encoders: GlobalEncoders
    train_result: TrainResult
    interaction: InteractionMatrix
    optimal_squad: OptimalSquad


def run_pipeline(
    config: PipelineConfig | None = None,
    verbose: bool = True,
) -> PipelineResult:
    """Run the full 7-stage pipeline. This is the main entry point.

    Pass a custom PipelineConfig to change anything (team, formation, epochs, etc).
    Default config is FC Bayern München, all available matches, 50 epochs.
    """
    if config is None:
        config = PipelineConfig()

    # Stage 1: Fetch match list
    if verbose:
        print("=" * 70)
        print(f"STAGE 1: Fetching {config.team_name} matches "
              f"(competition {config.competition_id})")
        print("=" * 70)

    matches_df = fetch_match_list(config.competition_id)
    match_ids = get_team_match_ids(matches_df, config.team_name, config.max_matches)
    if verbose:
        print(f"  Found {len(match_ids)} matches\n")

    # Stage 2: Build graphs (the big two-pass step)
    if verbose:
        print("=" * 70)
        print("STAGE 2: Loading matches & building graphs (two-pass)")
        print("=" * 70)

    match_graphs_list, encoders = build_all_match_graphs(match_ids, config, verbose)

    # Stage 3: Pool everything and train one GAT
    if verbose:
        print("\n" + "=" * 70)
        print("STAGE 3: Training pooled PlayGAT")
        print("=" * 70)

    all_graphs = pool_all_graphs(match_graphs_list)
    if verbose:
        print(f"  Total pooled graphs: {len(all_graphs)}")
    train_result = train_gat(all_graphs, config.training, verbose)

    # Stage 4 + 5: Score possessions and accumulate synergy
    # We do these together per match to avoid storing all predictions in memory
    if verbose:
        print("\n" + "=" * 70)
        print("STAGE 4–5: Inference & synergy accumulation")
        print("=" * 70)

    accumulator = SynergyAccumulator()
    for mg in match_graphs_list:
        merge_match_context(accumulator, mg.context)
        preds = predict_scores(train_result.model, mg.graphs)
        accumulate_match_synergy(accumulator, mg.graphs, preds, mg.context)
        if verbose:
            target_graphs = sum(1 for g in mg.graphs if g.team_id == mg.context.target_team_id)
            print(f"  Match {mg.match_id}: {target_graphs} target-team graphs processed")

    # Stage 6: Build the interaction matrix
    if verbose:
        print("\n" + "=" * 70)
        print("STAGE 6: Building interaction matrix")
        print("=" * 70)

    interaction = build_interaction_matrix(
        accumulator, config.squad.min_appearances,
    )
    if verbose:
        n = len(interaction.player_ids)
        print(f"  Qualified players (>= {config.squad.min_appearances} apps): {n}")
        print(f"  Matrix shape: {interaction.matrix.shape}")
        print(f"  Value range: [{interaction.matrix.min():.4f}, {interaction.matrix.max():.4f}]")

    # Stage 7: Solve for the best XI
    if verbose:
        print("\n" + "=" * 70)
        print("STAGE 7: Solving MILP for optimal starting XI")
        print("=" * 70)

    optimal = solve_optimal_squad(interaction, accumulator, config.squad, verbose)

    if verbose:
        _print_result(optimal, accumulator, config)

    return PipelineResult(
        match_graphs=match_graphs_list,
        encoders=encoders,
        train_result=train_result,
        interaction=interaction,
        optimal_squad=optimal,
    )


def _print_result(
    optimal: OptimalSquad,
    acc: SynergyAccumulator,
    config: PipelineConfig,
) -> None:
    """Pretty-print the final result."""
    if optimal.status != "OPTIMAL":
        print(f"\n  ✗ MILP failed (status={optimal.status})")
        # Show which players have Unknown positions — usually means we need
        # to add them to position_overrides in SquadConfig
        unknowns = [pid for pid in acc.player_groups if acc.player_groups[pid] == "Unknown"]
        if unknowns:
            print("  Unknown positions (add to position_overrides):")
            for pid in unknowns:
                print(f"    '{acc.player_names.get(pid, pid)}': '???',")
        return

    print(f"\n  ✓ OPTIMAL STARTING XI (synergy = {optimal.objective_value:.6f})")
    print(f"  Formation: {config.squad.formation}\n")

    for grp in config.squad.group_order:
        for pid in optimal.by_group.get(grp, []):
            name = acc.player_names.get(pid, pid)
            pos = acc.player_positions.get(pid, "?")
            apps = int(acc.diag_count.get(pid, 0))
            print(f"  [{grp:>11}]  {name:30s}  ({pos:>4s})  apps={apps}")

    print(f"\n  Bench:")
    for pid in optimal.bench_ids:
        name = acc.player_names.get(pid, pid)
        pos = acc.player_positions.get(pid, "?")
        grp = acc.player_groups.get(pid, "?")
        apps = int(acc.diag_count.get(pid, 0))
        print(f"    {name:30s}  ({pos:>4s}, {grp:12s})  apps={apps}")
