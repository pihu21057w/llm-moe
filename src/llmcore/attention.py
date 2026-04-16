from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func
except Exception:  # pragma: no cover - optional dependency
    flash_attn_func = None

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except Exception:  # pragma: no cover - fallback for older torch builds
    SDPBackend = None
    sdpa_kernel = None


def _repeat_kv(x: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return x
    return x.repeat_interleave(repeats, dim=1)


def _scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dropout_p: float, causal: bool) -> torch.Tensor:
    if q.is_cuda and sdpa_kernel is not None and SDPBackend is not None:
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=causal)
    return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=causal)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * norm * self.weight


@dataclass(slots=True)
class RotaryCache:
    cos: torch.Tensor
    sin: torch.Tensor


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10_000.0, scaling_type: str = "linear", scaling_factor: float = 1.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_type = scaling_type
        self.scaling_factor = scaling_factor
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache: dict[tuple[int, torch.device, torch.dtype], RotaryCache] = {}

    def _scaled_positions(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device, dtype=dtype)
        if self.scaling_type == "linear" and self.scaling_factor != 1.0:
            positions = positions / self.scaling_factor
        elif self.scaling_type == "ntk" and self.scaling_factor != 1.0:
            positions = positions / max(self.scaling_factor, 1e-6)
        return positions

    def get_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> RotaryCache:
        key = (seq_len, device, dtype)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        positions = self._scaled_positions(seq_len, device=device, dtype=dtype)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq.to(device=device, dtype=dtype))
        emb = torch.cat([freqs, freqs], dim=-1)
        cache = RotaryCache(cos=emb.cos()[None, None, :, :], sin=emb.sin()[None, None, :, :])
        self._cache[key] = cache
        return cache

    def apply(self, q: torch.Tensor, k: torch.Tensor, position_offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(-2)
        cache = self.get_cache(seq_len + position_offset, device=q.device, dtype=q.dtype)
        cos = cache.cos[:, :, position_offset : position_offset + seq_len, :]
        sin = cache.sin[:, :, position_offset : position_offset + seq_len, :]
        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)
        return q_rot, k_rot


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class LinearAttention(nn.Module):
    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True) -> torch.Tensor:
        if not causal:
            return _scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p if self.training else 0.0, causal=False)

        q_phi = F.elu(q) + 1.0
        k_phi = F.elu(k) + 1.0
        batch, heads, seq_len, dim = q_phi.shape
        value_dim = v.size(-1)
        kv_state = torch.zeros(batch, heads, dim, value_dim, device=q.device, dtype=q.dtype)
        k_state = torch.zeros(batch, heads, dim, device=q.device, dtype=q.dtype)
        outputs = []
        for index in range(seq_len):
            key_t = k_phi[:, :, index, :]
            value_t = v[:, :, index, :]
            kv_state = kv_state + torch.einsum("bhd,bhe->bhde", key_t, value_t)
            k_state = k_state + key_t
            numerator = torch.einsum("bhd,bhde->bhe", q_phi[:, :, index, :], kv_state)
            denominator = torch.einsum("bhd,bhd->bh", q_phi[:, :, index, :], k_state).unsqueeze(-1).clamp_min(1e-6)
            outputs.append((numerator / denominator).unsqueeze(2))
        return self.dropout(torch.cat(outputs, dim=2))


class Attention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        dropout: float,
        max_seq_len: int,
        rope_theta: float,
        rope_scaling: str,
        rope_scaling_factor: float,
        backend: str,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.head_dim = d_model // n_heads
        self.kv_repeats = n_heads // n_kv_heads
        self.backend = backend
        self.q_proj = nn.Linear(d_model, d_model, bias=use_bias)
        kv_dim = n_kv_heads * self.head_dim
        self.k_proj = nn.Linear(d_model, kv_dim, bias=use_bias)
        self.v_proj = nn.Linear(d_model, kv_dim, bias=use_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.dropout = nn.Dropout(dropout)
        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len, base=rope_theta, scaling_type=rope_scaling, scaling_factor=rope_scaling_factor)
        self.linear_attention = LinearAttention(dropout=dropout)

    def forward(self, x: torch.Tensor, causal: bool = True, position_offset: int = 0) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.rotary.apply(q, k, position_offset=position_offset)
        k = _repeat_kv(k, self.kv_repeats)
        v = _repeat_kv(v, self.kv_repeats)

        use_flash = self.backend in {"auto", "flash"} and flash_attn_func is not None and x.is_cuda and torch.cuda.get_device_capability(x.device)[0] >= 8
        if self.backend == "linear":
            attn = self.linear_attention(q, k, v, causal=causal)
        elif use_flash:
            attn = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=self.dropout.p if self.training else 0.0, causal=causal)
            attn = attn.transpose(1, 2)
        elif self.backend == "auto" and x.is_cuda and torch.cuda.get_device_capability(x.device)[0] < 8:
            attn = self.linear_attention(q, k, v, causal=causal)
        else:
            attn = _scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p if self.training else 0.0, causal=causal)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out_proj(self.dropout(attn))
