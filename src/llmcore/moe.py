from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    return nn.SiLU()


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float, activation: str = "silu", use_bias: bool = False) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=use_bias)
        self.w2 = nn.Linear(d_ff, d_model, bias=use_bias)
        self.w3 = nn.Linear(d_model, d_ff, bias=use_bias)
        self.activation = make_activation(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(self.activation(self.w1(x)) * self.w3(x)))


class MoEFeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float,
        num_experts: int,
        top_k: int,
        activation: str = "silu",
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=use_bias)
        self.experts = nn.ModuleList([FeedForward(d_model, d_ff, dropout=dropout, activation=activation, use_bias=use_bias) for _ in range(num_experts)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.router(x)
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_idx = torch.topk(probs, k=min(self.top_k, self.num_experts), dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-2)
        selected = torch.zeros_like(probs)
        selected.scatter_(-1, topk_idx, topk_probs)
        output = torch.sum(expert_outputs * selected.unsqueeze(-1), dim=-2)

        importance = probs.mean(dim=(0, 1))
        load = selected.mean(dim=(0, 1))
        aux_loss = self.num_experts * torch.sum(importance * load)
        return self.dropout(output), aux_loss
