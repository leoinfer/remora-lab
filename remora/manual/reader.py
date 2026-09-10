from __future__ import annotations

import torch
from torch import nn


class LearnedDependencyReader(nn.Module):
    """Trainable graph-query reader; it never has authority to mutate the ledger."""

    def __init__(self, n_modules: int, hidden_dim: int = 64):
        super().__init__()
        self.n_modules = n_modules
        graph_dim = n_modules * n_modules
        self.net = nn.Sequential(
            nn.Linear(graph_dim + n_modules, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_modules),
        )

    def forward(self, adjacency: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        x = torch.cat([adjacency.flatten(start_dim=1), query], dim=-1)
        return self.net(x)

    def predict(self, adjacency: torch.Tensor, query: torch.Tensor, threshold: float = 0.0) -> list[int]:
        with torch.no_grad():
            scores = self(adjacency, query).squeeze(0)
        return [i for i, x in enumerate(scores.tolist()) if x > threshold]
