"""
Entry point: run the full GAT → synergy → optimal XI pipeline.

Edit the settings below to change team, formation, training, etc.
Then just run: python run_all_matches.py
"""
import warnings

from src.config import PipelineConfig, SquadConfig, TrainingConfig
from src.pipeline import run_pipeline

warnings.filterwarnings("ignore")

# ─── SETTINGS (edit these) ───────────────────────────────────────────
TEAM_NAME       = "FC Bayern München"
MAX_MATCHES     = None    # None = use all available matches
EPOCHS          = 50
MIN_APPEARANCES = 150

# Formation — must add up to 10 (+ 1 GK = 11)
ATTACKERS   = 3
MIDFIELDERS = 3
DEFENDERS   = 4
# ─────────────────────────────────────────────────────────────────────


def main():
    formation = {
        "Attacker": ATTACKERS,
        "Midfielder": MIDFIELDERS,
        "Defender": DEFENDERS,
        "Goalkeeper": 1,
    }
    total = sum(formation.values())
    assert total == 11, f"Formation must add up to 11, got {total}"

    config = PipelineConfig(
        team_name=TEAM_NAME,
        max_matches=MAX_MATCHES,
        training=TrainingConfig(n_epochs=EPOCHS),
        squad=SquadConfig(min_appearances=MIN_APPEARANCES, formation=formation),
    )

    run_pipeline(config)


if __name__ == "__main__":
    main()
