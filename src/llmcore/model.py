from __future__ import annotations

from dataclasses import asdict
import warnings

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .attention import Attention, RMSNorm
from .config import ModelConfig
from .moe import FeedForward, MoEFeedForward


class ReasoningCell(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden = max(1, int(config.d_ff * config.reasoning_hidden_mult))
        self.norm = RMSNorm(config.d_model) if config.use_rms_norm else nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config.d_model, hidden, dropout=config.dropout, activation=config.activation, use_bias=config.use_bias)
        self.scale = config.reasoning_residual_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.ffn(self.norm(x))


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_index: int, shared_reasoner: ReasoningCell | None = None) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.use_moe = config.moe_every_n_layers > 0 and (layer_index + 1) % config.moe_every_n_layers == 0
        self.reasoning_steps = max(0, config.reasoning_steps)
        self.norm1 = RMSNorm(config.d_model) if config.use_rms_norm else nn.LayerNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model) if config.use_rms_norm else nn.LayerNorm(config.d_model)
        self.attn = Attention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len,
            rope_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
            rope_scaling_factor=config.rope_scaling_factor,
            backend=config.attention_backend,
            use_bias=config.use_bias,
        )
        d_ff = int(config.d_ff * config.moe_hidden_mult)
        if self.use_moe:
            self.ffn = MoEFeedForward(
                d_model=config.d_model,
                d_ff=d_ff,
                dropout=config.dropout,
                num_experts=config.moe_num_experts,
                top_k=config.moe_top_k,
                activation=config.activation,
                use_bias=config.use_bias,
            )
        else:
            self.ffn = FeedForward(config.d_model, d_ff, dropout=config.dropout, activation=config.activation, use_bias=config.use_bias)
        self.reasoner = shared_reasoner if shared_reasoner is not None else ReasoningCell(config)

    def forward(self, x: torch.Tensor, position_offset: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = x
        x = self.norm1(x)
        x = self.attn(x, position_offset=position_offset)
        x = residual + x

        residual = x
        x = self.norm2(x)
        if self.use_moe:
            ffn_out, aux_loss = self.ffn(x)
        else:
            ffn_out = self.ffn(x)
            aux_loss = x.new_zeros(())
        x = residual + ffn_out

        reasoning_loss = x.new_zeros(())
        if self.reasoning_steps > 0:
            for _ in range(self.reasoning_steps):
                prev_x = x
                x = self.reasoner(x)
                reasoning_loss = reasoning_loss + (x - prev_x).pow(2).mean()
            reasoning_loss = reasoning_loss / self.reasoning_steps
        return x, aux_loss, reasoning_loss


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.checkpointing_enabled = bool(config.use_checkpointing and config.reasoning_steps == 0 and config.moe_every_n_layers == 0)
        if config.use_checkpointing and not self.checkpointing_enabled:
            warnings.warn(
                "Activation checkpointing is disabled for this model configuration because MoE and/or reasoning steps are enabled. "
                "The current full-block checkpoint path is not stable with dynamic routing.",
                stacklevel=2,
            )
        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        shared_reasoner = ReasoningCell(config) if config.reasoning_share_parameters else None
        self.layers = nn.ModuleList([DecoderBlock(config, index, shared_reasoner=shared_reasoner) for index in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model) if config.use_rms_norm else nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embeddings.weight

    def forward(self, input_ids: torch.Tensor, position_offset: int = 0) -> dict[str, torch.Tensor]:
        x = self.token_embeddings(input_ids)
        reasoning_losses = []
        reasoning_losses = []
        aux_losses = []
        reasoning_losses = []
        for layer in self.layers:
            if self.training and self.checkpointing_enabled:
                x, aux_loss, reasoning_loss = checkpoint(lambda hidden: layer(hidden, position_offset=position_offset), x, use_reentrant=False)
            else:
                x, aux_loss, reasoning_loss = layer(x, position_offset=position_offset)
            aux_losses.append(aux_loss)
            reasoning_losses.append(reasoning_loss)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        moe_aux_loss = torch.stack(aux_losses).sum() if aux_losses else logits.new_zeros(())
        reasoning_aux_loss = torch.stack(reasoning_losses).mean() if reasoning_losses else logits.new_zeros(())
        return {"logits": logits, "moe_aux_loss": moe_aux_loss, "reasoning_aux_loss": reasoning_aux_loss}

    def estimate_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def to_config_dict(self) -> dict:
        return asdict(self.config)
