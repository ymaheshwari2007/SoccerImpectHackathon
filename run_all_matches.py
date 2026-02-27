"""
Entry point: run the full GAT → synergy → optimal XI pipeline.

This is just a thin wrapper around src.pipeline.run_pipeline() with
command-line arguments for easy tweaking.

Usage:
    python run_all_matches.py                   # all matches, defaults
    python run_all_matches.py --max-matches 5   # quick test with 5 matches
    python run_all_matches.py --epochs 100      # more training
    python run_all_matches.py --team "Borussia Dortmund"  # different team
"""
import argparse
import warnings

from src.config import PipelineConfig
from src.pipeline import run_pipeline

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description="GAT → Synergy → MILP pipeline")
    parser.add_argument("-max-matches", type=int, default=None,
                        help="Limit number of matches (default: all)")
    parser.add_argument("--min-appearances", type=int, default=150,
                        help="Minimum player appearances to qualify (default: 150)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs (default: 50)")
    parser.add_argument("--team", type=str, default="FC Bayern München",
                        help="Target team name")
    args = parser.parse_args()

    from src.config import SquadConfig, TrainingConfig

    config = PipelineConfig(
        team_name=args.team,
        max_matches=args.max_matches,
        training=TrainingConfig(n_epochs=args.epochs),
        squad=SquadConfig(min_appearances=args.min_appearances),
    )

    run_pipeline(config)


if __name__ == "__main__":
    main()
