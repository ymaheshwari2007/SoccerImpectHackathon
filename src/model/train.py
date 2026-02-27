"""
Pooled GAT training — one model learns from ALL matches at once.

The old approach trained 34 separate models (one per match, ~160 graphs each).
Now we pool everything into ~5,400 graphs and train a single model.

Benefits:
  - Faster (1 training loop instead of 34)
  - Better generalisation (more diverse training data)
  - The model sees the same player across different matches, so it can learn
    player-specific patterns rather than memorising match-specific quirks
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from ..config import TrainingConfig
from .gat import PlayGAT


@dataclass
class TrainResult:
    """What you get back after training: the model + loss history."""
    model: PlayGAT
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)


def train_gat(
    graphs: list[Data],
    config: TrainingConfig,
    verbose: bool = True,
) -> TrainResult:
    """Train one PlayGAT on all the pooled graphs.

    Does an 80/20 train/val split (shuffled with a fixed seed for
    reproducibility), then runs a standard training loop with MSE loss.

    Nothing fancy — the model is small enough that it trains in ~30s on CPU.
    """
    torch.manual_seed(config.seed)

    # Shuffle and split into train/val
    perm = torch.randperm(
        len(graphs), generator=torch.Generator().manual_seed(config.seed),
    ).tolist()
    n_train = int(config.train_split * len(graphs))

    train_graphs = [graphs[i] for i in perm[:n_train]]
    val_graphs = [graphs[i] for i in perm[n_train:]]

    train_loader = DataLoader(train_graphs, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=config.batch_size, shuffle=False)

    if verbose:
        print(f"  Train: {len(train_graphs)} graphs  |  Val: {len(val_graphs)} graphs")

    model = PlayGAT(hidden=config.hidden, heads=config.heads, dropout=config.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.MSELoss()

    result = TrainResult(model=model)

    for epoch in range(1, config.n_epochs + 1):
        # Training
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(pred, batch.y.squeeze(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.num_graphs
        avg_train = epoch_loss / len(train_graphs)
        result.train_losses.append(avg_train)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                val_loss += criterion(pred, batch.y.squeeze(-1)).item() * batch.num_graphs
        avg_val = val_loss / len(val_graphs)
        result.val_losses.append(avg_val)

        if verbose and (epoch % config.log_every == 0 or epoch == 1):
            print(f"  Epoch {epoch:>3}/{config.n_epochs}  |  "
                  f"train MSE: {avg_train:.6f}  |  val MSE: {avg_val:.6f}")

    if verbose:
        print("  Training complete.")

    return result
