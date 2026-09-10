from __future__ import annotations

import math
import os

import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, use_sdpa: bool | None = None):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must divide evenly across attention heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = dropout
        if use_sdpa is None:
            # AOTriton SDPA is faster on this RDNA4 setup only when explicitly
            # enabled. Keep the portable/manual path as the safe AMD default;
            # NVIDIA/CPU and opted-in ROCm use the fused implementation.
            amd_experimental_enabled = os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL") == "1"
            self.use_sdpa = not (torch.version.hip is not None and not amd_experimental_enabled)
        else:
            self.use_sdpa = use_sdpa

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        if self.use_sdpa:
            out = nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            causal = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
            probs = torch.softmax(scores, dim=-1)
            probs = nn.functional.dropout(probs, p=self.dropout, training=self.training)
            out = probs @ v
        out = out.transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.out(out)
