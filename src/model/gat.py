"""
PlayGAT — the Graph Attention Network that scores possessions.

This is a fairly small GNN (~30K parameters). It takes a possession graph
(where nodes are events and edges connect sequential/nearby events) and
predicts a single number: how "dangerous" that possession was.

Architecture:
  Layer 1: GATConv with 4 attention heads (each head learns to focus on
           different aspects — maybe one head cares about pass distance,
           another about pressure). Outputs get concatenated → 128 dims.
  Layer 2: GATConv with 1 head that merges everything down to 32 dims.
  Readout: Global mean pool (average all node embeddings) → Linear → scalar.

Why GAT specifically? Because attention weights tell us which events in
a possession mattered most. Regular GCN would just average neighbors equally.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class PlayGAT(nn.Module):
    """Two-layer GAT that predicts a scalar play score from a possession graph."""

    NODE_DIM = 7   # features per node (player_id_enc, x, y, event_type_enc, t_rel, pressure, outcome_type)
    EDGE_DIM = 7   # features per edge (type, pass_type, end_x, end_y, dist, result, success)

    def __init__(self, hidden: int = 32, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout

        # Layer 1: multi-head attention (concat=True → output is hidden*heads = 128)
        self.conv1 = GATConv(
            self.NODE_DIM, hidden, heads=heads, concat=True,
            edge_dim=self.EDGE_DIM, dropout=dropout, add_self_loops=False,
        )
        self.bn1 = nn.BatchNorm1d(hidden * heads)

        # Layer 2: single head to compress back down (concat=False → output is hidden = 32)
        self.conv2 = GATConv(
            hidden * heads, hidden, heads=1, concat=False,
            edge_dim=self.EDGE_DIM, dropout=dropout, add_self_loops=False,
        )

        # Final linear layer: 32 → 1 scalar prediction
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        """Standard forward pass for training/inference with batched graphs."""
        x = F.elu(self.bn1(self.conv1(x, edge_index, edge_attr=edge_attr)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index, edge_attr=edge_attr))
        x = global_mean_pool(x, batch)   # average all node embeddings per graph
        return self.head(x).squeeze(-1)   # one score per graph in the batch

    def forward_with_attention(self, data):
        """Single-graph forward that also returns attention weights.

        Useful for interpretability — you can see which edges (event pairs)
        the model paid most attention to when scoring a possession.
        Returns (prediction, layer1_attention, layer2_attention).
        """
        x, ei, ea = data.x, data.edge_index, data.edge_attr
        batch = torch.zeros(x.size(0), dtype=torch.long)

        x, (_, attn1) = self.conv1(x, ei, edge_attr=ea, return_attention_weights=True)
        x = F.elu(self.bn1(x))
        x, (_, attn2) = self.conv2(x, ei, edge_attr=ea, return_attention_weights=True)
        x = F.elu(x)
        x = global_mean_pool(x, batch)
        return self.head(x).squeeze(-1), attn1, attn2
