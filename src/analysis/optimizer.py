"""
MILP squad optimization — pick the best starting XI using integer programming.

Given the N×N synergy matrix, we want to choose 11 players that maximise
total pairwise synergy PLUS individual quality.

Only constraint: exactly 1 goalkeeper, total 11 players. No formation constraints —
let the synergy matrix decide the team shape.

This is a quadratic binary problem (we're multiplying binary variables x_i * x_j),
but we linearise it using McCormick envelopes so Gurobi can solve it as a
standard MILP.

Gurobi solves this in under a second for ~20 players.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import gurobipy as gp
from gurobipy import GRB

from ..config import SquadConfig
from .synergy import InteractionMatrix, SynergyAccumulator


@dataclass
class OptimalSquad:
    """The output of the optimizer."""
    selected_ids: list[str] = field(default_factory=list)   # the chosen 11
    objective_value: float = 0.0                            # total synergy score
    status: str = "NOT_SOLVED"                              # "OPTIMAL" if it worked
    by_group: dict[str, list[str]] = field(default_factory=dict)  # grouped by position
    bench_ids: list[str] = field(default_factory=list)      # everyone else


def solve_optimal_squad(
    interaction: InteractionMatrix,
    acc: SynergyAccumulator,
    squad_cfg: SquadConfig,
    verbose: bool = True,
) -> OptimalSquad:
    """Solve for the optimal starting XI — pure synergy, no formation constraints.

    Decision variables:
      x_i ∈ {0,1} — is player i selected?
      y_ij ∈ [0,1] — linearised product of x_i and x_j (McCormick)

    Objective: maximise Σ S[i,i]*x_i + Σ S[i,j]*y_ij
      (individual quality + pairwise synergy)

    Constraints:
      - Exactly 1 goalkeeper
      - Exactly 11 total
      - McCormick: y_ij ≤ x_i, y_ij ≤ x_j, y_ij ≥ x_i + x_j - 1
    """
    pids = interaction.player_ids
    S = interaction.matrix
    n = len(pids)

    mdl = gp.Model("optimal_squad")
    mdl.setParam("OutputFlag", 0)

    x = mdl.addVars(n, vtype=GRB.BINARY, name="x")

    y = {}
    for i in range(n):
        for j in range(i + 1, n):
            y[i, j] = mdl.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"y_{i}_{j}")

    # Objective: individual quality + pairwise synergy
    obj = gp.LinExpr()
    for i in range(n):
        obj += S[i, i] * x[i]
    for i in range(n):
        for j in range(i + 1, n):
            sij = (S[i, j] + S[j, i]) / 2.0
            obj += sij * y[i, j]
    mdl.setObjective(obj, GRB.MAXIMIZE)

    # McCormick constraints
    for i in range(n):
        for j in range(i + 1, n):
            mdl.addConstr(y[i, j] <= x[i])
            mdl.addConstr(y[i, j] <= x[j])
            mdl.addConstr(y[i, j] >= x[i] + x[j] - 1)

    if squad_cfg.use_fixed_formation:
        # Fixed formation: exact counts per group (e.g. 3-4-3)
        for grp, count in squad_cfg.formation.items():
            indices = [k for k in range(n) if acc.player_groups.get(pids[k]) == grp]
            if verbose:
                status = "OK" if len(indices) >= count else "SHORTAGE"
                print(f"  {grp}: need {count}, have {len(indices)}  [{status}]")
            mdl.addConstr(gp.quicksum(x[k] for k in indices) == count, name=f"pos_{grp}")
    else:
        # Soft bounds: let synergy drive the shape
        min_per_group = {"Attacker": 2, "Midfielder": 2, "Defender": 3, "Goalkeeper": 1}
        max_per_group = {"Attacker": 4, "Midfielder": 5, "Defender": 5, "Goalkeeper": 1}

        for grp in squad_cfg.group_order:
            indices = [k for k in range(n) if acc.player_groups.get(pids[k]) == grp]
            if verbose:
                print(f"  {grp}: {len(indices)} available")
            lo = min_per_group.get(grp, 0)
            hi = max_per_group.get(grp, 10)
            if lo > 0:
                mdl.addConstr(gp.quicksum(x[k] for k in indices) >= lo, name=f"min_{grp}")
            if hi < 10:
                mdl.addConstr(gp.quicksum(x[k] for k in indices) <= hi, name=f"max_{grp}")

        gk_indices = [k for k in range(n) if acc.player_groups.get(pids[k]) == "Goalkeeper"]
        mdl.addConstr(gp.quicksum(x[k] for k in gk_indices) == 1, name="one_gk")

    # Total = 11
    mdl.addConstr(gp.quicksum(x[i] for i in range(n)) == 11, name="total")

    mdl.optimize()

    result = OptimalSquad()

    if mdl.Status == GRB.OPTIMAL:
        result.status = "OPTIMAL"
        result.objective_value = mdl.ObjVal
        result.selected_ids = [pids[i] for i in range(n) if x[i].X > 0.5]

        for grp in squad_cfg.group_order:
            result.by_group[grp] = [
                pid for pid in result.selected_ids
                if acc.player_groups.get(pid) == grp
            ]

        selected_set = set(result.selected_ids)
        result.bench_ids = [pid for pid in pids if pid not in selected_set]
    else:
        result.status = f"GUROBI_STATUS_{mdl.Status}"

    return result
