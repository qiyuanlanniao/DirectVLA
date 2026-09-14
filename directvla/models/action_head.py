from __future__ import annotations

import torch
from torch import nn

from .components.utils import MLP
from .configuration import ActionHeadConfig


class StateProjection(nn.Module):
    def __init__(self, config: ActionHeadConfig, hidden_dim: int) -> None:
        super().__init__()
        self.num_tokens = int(config.num_state_tokens)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(config.state_dim),
            nn.Linear(config.state_dim, config.state_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.state_hidden_dim, self.num_tokens * self.hidden_dim),
        )
        self.position = nn.Parameter(torch.randn(1, self.num_tokens, self.hidden_dim) * 0.02)
        self.output_norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim == 3:
            state = state[:, -1]
        if state.ndim != 2:
            raise ValueError(f"state must be [B,D] or [B,T,D], got {tuple(state.shape)}")
        tokens = self.net(state).view(state.shape[0], self.num_tokens, self.hidden_dim)
        return self.output_norm(tokens + self.position.to(device=tokens.device, dtype=tokens.dtype))


class ACTDecoder(nn.Module):
    def __init__(self, config: ActionHeadConfig, hidden_dim: int, nheads: int, dim_feedforward: int) -> None:
        super().__init__()
        self.horizon = int(config.horizon)
        self.action_queries = nn.Embedding(self.horizon, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_layers)
        self.action_projection = MLP(hidden_dim, config.mlp_hidden_dim, config.action_dim, 3)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        queries = self.action_queries.weight.unsqueeze(0).expand(memory.shape[0], -1, -1)
        hidden = self.decoder(tgt=queries, memory=memory)
        return torch.tanh(self.action_projection(hidden))


class TurboVLAActionHead(nn.Module):
    def __init__(self, config: ActionHeadConfig, hidden_dim: int, nheads: int, dim_feedforward: int) -> None:
        super().__init__()
        self.state_projection = StateProjection(config, hidden_dim)
        self.decoder = ACTDecoder(config, hidden_dim, nheads, dim_feedforward)

    def forward(self, vision_language_tokens: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        state = state.to(device=vision_language_tokens.device, dtype=vision_language_tokens.dtype)
        state_tokens = self.state_projection(state)
        return self.decoder(torch.cat([vision_language_tokens, state_tokens], dim=1))
