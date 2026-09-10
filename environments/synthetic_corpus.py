from __future__ import annotations

import random


_WORDS = (
    "anchor", "basin", "copper", "delta", "ember", "fathom", "granite",
    "harbor", "island", "juniper", "kernel", "lantern", "marble", "notion",
    "orbit", "pebble", "quartz", "ripple", "signal", "thimble",
)


def build_language_corpus(n_lines: int = 5000, seed: int = 0) -> str:
    """Create a deterministic, leakage-free text/code/math training stream."""
    rng = random.Random(seed)
    lines: list[str] = []
    for i in range(n_lines):
        a = rng.randrange(0, 100)
        b = rng.randrange(0, 100)
        p = rng.randrange(0, 2)
        q = rng.randrange(0, 2)
        w1, w2, w3 = rng.sample(_WORDS, 3)
        lines.append(f"text: {w1} {w2} {w3}; index={i % 17}; parity={(a + b) % 2}\n")
        lines.append(f"math: {a}+{b}={a + b}; {a}*{p}={a * p}; mod7={a % 7}\n")
        lines.append(f"logic: p={p}; q={q}; both={p & q}; either={p | q}; xor={p ^ q}\n")
        lines.append(f"code: x={a}; y={b}; return=x+y={a + b}\n")
        lines.append(
            f"world: source=inherited; z=0; condition=ordinary; outcome=Y; confidence={70 + i % 20}\n"
        )
        if i % 5 == 0:
            lines.append("reason: [alpha [beta]] -> compositional; copy=REMORA\n")
    return "".join(lines)


def build_target_corpus(n_lines: int = 800, seed: int = 100) -> str:
    """A related but distribution-shifted stream for local adaptation tests."""
    rng = random.Random(seed)
    lines: list[str] = []
    for i in range(n_lines):
        a = rng.randrange(100, 200)
        b = rng.randrange(0, 50)
        lines.append(f"target: z=1; x={a}; y={b}; answer={a - b}; route=special\n")
        lines.append(f"target-code: input={a}; transform={a // 3}; checksum={(a + b) % 11}\n")
    return "".join(lines)


def build_heldout_corpus(n_lines: int = 1000, seed: int = 999) -> str:
    return build_language_corpus(n_lines=n_lines, seed=seed)
