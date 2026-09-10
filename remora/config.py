from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    vocab_size: int = 128
    max_seq_len: int = 96
    d_model: int = 192
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 384
    bus_dim: int = 96
    n_experts: int = 2
    adapter_dim: int = 24
    dropout: float = 0.0
    # Chosen so the default tiny baseline is parameter-matched to Remora-v0;
    # the matching calculation is recorded in the training result.
    baseline_d_ff: int = 672
    tie_embeddings: bool = True

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        with Path(path).open() as f:
            return cls(**json.load(f))

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        with Path(path).open("w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
            f.write("\n")
