from __future__ import annotations

import json
from pathlib import Path


class ByteTokenizer:
    """A deterministic ASCII byte tokenizer with no pretrained vocabulary."""

    def __init__(self, vocab_size: int = 128):
        if vocab_size < 128:
            raise ValueError("vocab_size must cover ASCII for the synthetic corpus")
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        values = list(text.encode("ascii"))
        if any(v >= self.vocab_size for v in values):
            raise ValueError("text contains a byte outside the configured vocabulary")
        return values

    def decode(self, ids) -> str:
        return bytes(int(i) for i in ids).decode("ascii", errors="replace")

    def save(self, path: str | Path) -> None:
        with Path(path).open("w") as f:
            json.dump({"type": "ascii-byte", "vocab_size": self.vocab_size}, f)
            f.write("\n")

    @classmethod
    def load(cls, path: str | Path) -> "ByteTokenizer":
        with Path(path).open() as f:
            return cls(**json.load(f))
