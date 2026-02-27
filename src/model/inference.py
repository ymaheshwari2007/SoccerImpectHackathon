"""
Model inference — run the trained GAT on graphs to get predicted scores.

These predicted scores are what feed into the synergy accumulator.
A high score means the model thinks that possession was dangerous/valuable.
"""
from __future__ import annotations

import torch
from torch_geometric.data import Data

from .gat import PlayGAT


def predict_scores(model: PlayGAT, graphs: list[Data]) -> list[float]:
    """Run the trained GAT on each graph and return one predicted score per graph.

    We process graphs one at a time (not batched) because we need to keep
    track of which graph corresponds to which possession. The graphs are
    tiny so this is fast enough.
    """
    model.eval()
    scores: list[float] = []
    with torch.no_grad():
        for g in graphs:
            # Create a fake batch vector (all zeros = "all nodes belong to graph 0")
            batch = torch.zeros(g.num_nodes, dtype=torch.long)
            pred = model(g.x, g.edge_index, g.edge_attr, batch)
            scores.append(pred.item())
    return scores
