"""
Entry point: run the full GAT → synergy → optimal XI pipeline.
Edit the settings below, then run: python run_all_matches.py
"""
import warnings

from src.config import PipelineConfig, SquadConfig, TrainingConfig
from src.pipeline import run_pipeline

warnings.filterwarnings("ignore")

# ─── SETTINGS (edit these) ───────────────────────────────────────────
TEAM_NAME       = "FC Bayern München"
EPOCHS          = 50
MIN_APPEARANCES = 150

# Formation — set to None for auto (synergy-driven, soft bounds)
#           — or set ATK, MID, DEF counts (must add up to 10, +1 GK = 11)
ATTACKERS   = None   # e.g. 3
MIDFIELDERS = None   # e.g. 4
DEFENDERS   = None   # e.g. 3
# ─────────────────────────────────────────────────────────────────────


def main():
    use_fixed = all(v is not None for v in [ATTACKERS, MIDFIELDERS, DEFENDERS])

    squad_kwargs = {"min_appearances": MIN_APPEARANCES}

    if use_fixed:
        formation = {
            "Attacker": ATTACKERS,
            "Midfielder": MIDFIELDERS,
            "Defender": DEFENDERS,
            "Goalkeeper": 1,
        }
        total = sum(formation.values())
        assert total == 11, f"Formation must add up to 11, got {total}"
        squad_kwargs["formation"] = formation
        squad_kwargs["use_fixed_formation"] = True
        print(f"Team: {TEAM_NAME}")
        print(f"Formation: {ATTACKERS}-{MIDFIELDERS}-{DEFENDERS} (+ 1 GK)\n")
    else:
        print(f"Team: {TEAM_NAME}")
        print(f"Formation: auto (synergy-driven, soft bounds)\n")

    config = PipelineConfig(
        team_name=TEAM_NAME,
        training=TrainingConfig(n_epochs=EPOCHS),
        squad=SquadConfig(**squad_kwargs),
    )

    run_pipeline(config)


if __name__ == "__main__":
    main()
