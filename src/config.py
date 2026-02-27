"""
All the knobs and settings for the pipeline live here.

Instead of magic numbers scattered across 15 files, everything is in one place.
Each config is a frozen dataclass — once created, you can't accidentally mutate it
mid-pipeline. To tweak something, create a new config with the value you want:

    config = PipelineConfig(training=TrainingConfig(n_epochs=100))
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FieldConfig:
    """Pitch dimensions in metres.

    These match the 'secondspectrum' coordinate system that kloppy gives us.
    Origin is at center circle, x runs along the length, y across the width.
    So the goals sit at x = ±52.5.
    """
    field_half: float = 52.5       # center to goal line
    field_length: float = 105.0    # goal line to goal line
    field_width: float = 68.0      # sideline to sideline

    @property
    def max_field_dist(self) -> float:
        """Longest possible distance on the pitch (corner to far-side center).
        Used to normalise displacement scores to [-1, 1]."""
        return (self.field_length ** 2 + (self.field_width / 2) ** 2) ** 0.5


@dataclass(frozen=True)
class ScoringConfig:
    """How we score each possession (the label our GAT learns to predict).

    The score tries to capture "how dangerous was this possession?" by combining:
    - How far did the ball move toward goal? (displacement × advancement)
    - How long was the play? (sigmoid decay — quick attacks score higher)
    - Did it end in a shot or goal? (bonus/penalty)
    """
    sigmoid_shift: float = 7.5          # plays shorter than ~7.5s get boosted
    advancement_weight: float = 1.0     # weight for "how far into opponent half"
    goal_bonus: float = 1.0             # flat bonus if the play ended in a goal
    shot_bonus: float = 0.3             # smaller bonus for a shot (no goal)
    out_of_bounds_penalty: float = -0.1   # lost possession out of bounds
    intercept_penalty: float = -0.2       # opponent intercepted


@dataclass(frozen=True)
class GraphConfig:
    """Controls how we turn a possession sequence into a PyG graph.

    no_video_threshold: if a possession has more than this many seconds of
        NO_VIDEO events, we throw it out (the data is too gappy to trust).
    n_synthetic_edges: we add "skip connections" so the GAT can see 2-3 events
        ahead, not just the immediate next action. Think of it like giving the
        model peripheral vision.
    """
    no_video_threshold: float = 3.0
    n_synthetic_edges: int = 2


@dataclass(frozen=True)
class TrainingConfig:
    """GAT training hyperparameters.

    These are pretty standard for a small GNN. The model is lightweight
    (~30K params) so it trains fast — the bottleneck is data loading, not GPU.
    """
    seed: int = 42
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    n_epochs: int = 50
    train_split: float = 0.8       # 80% train, 20% val
    hidden: int = 32               # hidden dim per GAT layer
    heads: int = 4                 # multi-head attention in layer 1
    dropout: float = 0.3
    log_every: int = 10            # print loss every N epochs


@dataclass(frozen=True)
class SquadConfig:
    """Everything the MILP optimizer needs to pick the best XI.

    position_groups: maps broad roles (Attacker, Midfielder, ...) to specific
        position codes from the IMPECT data. We need this because the optimizer
        enforces formation constraints (e.g. exactly 3 attackers in a 4-3-3).

    position_overrides: some players don't have a starting_position in the data
        (subs, late arrivals, etc). We hardcode their positions here so they
        don't get dropped or classified as "Unknown".

    min_appearances: players with fewer participations than this get excluded
        from the synergy matrix. Keeps the matrix manageable and avoids noisy
        estimates from tiny sample sizes.
    """
    min_appearances: int = 150

    position_groups: dict[str, list[str]] = field(default_factory=lambda: {
        "Attacker":   ["ST", "LW", "RW", "CF", "SS"],
        "Midfielder": ["CDM", "LDM", "RDM", "CAM", "LAM", "RAM",
                        "CM", "LCM", "RCM", "LM", "RM"],
        "Defender":   ["CB", "LCB", "RCB", "LWB", "RWB", "LB", "RB"],
        "Goalkeeper":  ["GK"],
    })

    group_order: list[str] = field(default_factory=lambda: [
        "Attacker", "Midfielder", "Defender", "Goalkeeper",
    ])

    # 4-2-3-1 expressed as broad groups (3 ATK + 4 MID + 3 DEF + 1 GK = 11)
    formation: dict[str, int] = field(default_factory=lambda: {
        "Attacker": 3, "Midfielder": 4, "Defender": 3, "Goalkeeper": 1,
    })

    # Manual position fixes for players missing starting_position in the data
    position_overrides: dict[str, str] = field(default_factory=lambda: {
        "Thomas Müller": "CAM", "Eric-Maxim Choupo-Moting": "ST",
        "Benjamin Pavard": "RCB", "Konrad Laimer": "RDM",
        "Matthijs de Ligt": "CB", "Ryan Gravenberch": "CDM",
        "Frans Krätzig": "LWB", "Tom Ritzy Hülsmann": "GK",
        "Mathys Tel": "LW", "Aleksandar Pavlovic": "LDM",
        "Bryan Zaragoza": "LW", "Josip Stanisic": "RWB",
        "Bouna Sarr": "RWB", "Paul Wanner": "CAM",
        "Arijon Ibrahimovic": "LW", "Gabriel Vidovic": "CAM",
        "Chris Richards": "CB", "Sven Ulreich": "GK",
        "Daniel Peretz": "GK", "Joshua Zirkzee": "ST",
        "Eric Dier": "CB", "Raphael Guerreiro": "LWB",
        "Sacha Boey": "RWB", "Noussair Mazraoui": "RWB",
    })

    @property
    def code_to_group(self) -> dict[str, str]:
        """Reverse lookup: position code → group name (e.g. 'ST' → 'Attacker')."""
        return {code: grp for grp, codes in self.position_groups.items() for code in codes}


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level config — this is what you pass to run_pipeline().

    Bundles all the sub-configs together. Defaults are set for FC Bayern München
    in the 2023/24 Bundesliga (competition_id=743). Change team_name and
    competition_id to analyze a different team/league.
    """
    competition_id: int = 743                  # Bundesliga 2023/24
    team_name: str = "FC Bayern München"
    max_matches: int | None = None             # None = use all available matches
    pitch: FieldConfig = field(default_factory=FieldConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    squad: SquadConfig = field(default_factory=SquadConfig)
