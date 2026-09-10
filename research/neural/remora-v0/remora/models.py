from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .config import ModelConfig
from .language_bus import SharedLanguageBus
from .modules import CausalSelfAttention, FastPlasticAdapter, GatedDeltaState, ModularExperts


class RemoraBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, bus: SharedLanguageBus):
        super().__init__()
        self.norm = nn.LayerNorm(cfg.d_model)
        self.attention = CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.recurrent = GatedDeltaState(cfg.d_model, cfg.bus_dim)
        self.experts = ModularExperts(cfg.bus_dim, cfg.d_ff, cfg.n_experts)
        self.plastic = FastPlasticAdapter(cfg.d_model, cfg.adapter_dim)
        self.bus = bus
        self.branch_scale = nn.Parameter(torch.full((4,), 0.25))

    def forward(
        self, x: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        h = self.norm(x)
        packet = self.bus(h, producer="remora-block")
        attn = self.attention(h)
        recurrent, new_state = self.recurrent(h, state)
        expert_latent, route_weights = self.experts(packet.latent)
        expert = self.bus.decode(expert_latent)
        plastic = self.plastic(h)
        y = x + (
            self.branch_scale[0] * attn
            + self.branch_scale[1] * recurrent
            + self.branch_scale[2] * expert
            + self.branch_scale[3] * plastic
        )
        aux = {
            "route_weights": route_weights,
            "bus_confidence": packet.confidence,
            "state": new_state,
        }
        return y, new_state, aux


class RemoraModel(nn.Module):
    model_type = "remora-v0"

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position_embedding = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.bus = SharedLanguageBus(cfg.d_model, cfg.bus_dim, version="language-v1")
        self.blocks = nn.ModuleList([RemoraBlock(cfg, self.bus) for _ in range(cfg.n_layers)])
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)
        # The plastic island is an explicit no-op at initialization. The generic
        # initializer above visits all Linear layers, so restore that contract.
        for block in self.blocks:
            nn.init.zeros_(block.plastic.up.weight)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        state: list[torch.Tensor | None] | None = None,
        return_aux: bool = False,
    ):
        b, t = input_ids.shape
        if t > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {t} exceeds {self.cfg.max_seq_len}")
        positions = torch.arange(t, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        if state is None:
            state = [None] * len(self.blocks)
        new_states = []
        route_weights = []
        confidences = []
        for block, block_state in zip(self.blocks, state):
            x, new_state, aux = block(x, block_state)
            new_states.append(new_state)
            route_weights.append(aux["route_weights"])
            confidences.append(aux["bus_confidence"])
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        aux = {
            "route_weights": route_weights,
            "bus_confidence": confidences,
            "state": new_states,
        }
        if return_aux:
            return logits, loss, aux
        return logits, loss

    def replace_expert(self, layer_index: int, expert_index: int, expert: nn.Module) -> nn.Module:
        return self.blocks[layer_index].experts.replace_expert(expert_index, expert)


class BaselineBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.ff(self.norm2(x))


class BaselineModel(nn.Module):
    model_type = "monolithic-transformer-control"

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position_embedding = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(
            [BaselineBlock(cfg.d_model, cfg.n_heads, cfg.baseline_d_ff, cfg.dropout) for _ in range(cfg.n_layers)]
        )
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None, return_aux: bool = False):
        b, t = input_ids.shape
        if t > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {t} exceeds {self.cfg.max_seq_len}")
        positions = torch.arange(t, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        if return_aux:
            return logits, loss, {}
        return logits, loss


def build_model(kind: str, cfg: ModelConfig) -> nn.Module:
    if kind == "remora":
        return RemoraModel(cfg)
    if kind == "baseline":
        return BaselineModel(cfg)
    raise ValueError(f"unknown model kind: {kind}")
