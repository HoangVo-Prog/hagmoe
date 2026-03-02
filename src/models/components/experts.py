from __future__ import annotations

import torch
import torch.nn as nn


class FFNExpert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float, act_fn: nn.Module):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.act = act_fn
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x
