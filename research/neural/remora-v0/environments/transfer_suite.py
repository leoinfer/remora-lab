from __future__ import annotations

"""Fixed, local-only text/code/math inputs for the next Remora milestone.

The suite deliberately keeps two kinds of evidence separate:

* Wikitext-2 and repository source files are real local language streams.
* Math/code prompt-response examples are generated from known functions so
  exact held-out correctness has an external oracle.

No personal dataset is read by this module. The returned metadata pins source
paths, hashes, seeds, and the ASCII sanitization rule so a run can be audited
without copying the source corpus into Git.
"""

import hashlib
import random
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from remora.data import encode_stream
from remora.tokenizer import ByteTokenizer


@dataclass(frozen=True)
class TaskExample:
    task_id: str
    domain: str
    prompt: str
    response: str
    lineage_key: str

    @property
    def text(self) -> str:
        return f"{self.prompt}{self.response}\n"


@dataclass
class TransferSuite:
    train_streams: dict[str, torch.Tensor]
    valid_streams: dict[str, torch.Tensor]
    test_streams: dict[str, torch.Tensor]
    code_task_train: list[TaskExample]
    code_task_valid: list[TaskExample]
    code_task_shifted: list[TaskExample]
    math_train: list[TaskExample]
    math_valid: list[TaskExample]
    math_shifted: list[TaskExample]
    parity_train: list[TaskExample]
    parity_valid: list[TaskExample]
    parity_shifted: list[TaskExample]
    metadata: dict


def ascii_sanitize(text: str) -> str:
    """Make arbitrary local text representable by the v0 ASCII tokenizer."""

    normalized = "".join(
        character
        for character in unicodedata.normalize("NFKD", text)
        if unicodedata.category(character) != "Mn"
    )
    encoded = normalized.encode("ascii", errors="replace").decode("ascii")
    return "".join(character for character in encoded if character in "\n\r\t" or ord(character) >= 32)


def _read_text(path: Path, max_chars: int | None = None) -> str:
    text = ascii_sanitize(path.read_text(encoding="utf-8", errors="replace"))
    if max_chars is not None:
        text = text[:max_chars]
    if not text:
        raise ValueError(f"empty text source: {path}")
    return text


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _stream(text: str, tokenizer: ByteTokenizer) -> torch.Tensor:
    result = encode_stream(text, tokenizer)
    if result.numel() < 128:
        raise ValueError("transfer stream is too short for the configured sequence length")
    return result


def _join_files(paths: Iterable[Path], root: Path) -> tuple[str, list[dict]]:
    chunks: list[str] = []
    manifest: list[dict] = []
    for path in sorted({path.resolve() for path in paths}):
        if not path.is_file():
            continue
        relative = str(path.relative_to(root.resolve()))
        try:
            text = _read_text(path)
        except ValueError:
            # Empty package markers are valid source files but contribute no
            # language signal and should not invalidate an otherwise usable
            # code split.
            continue
        chunks.append(f"\n# FILE {relative}\n{text}\n")
        manifest.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    if not chunks:
        raise ValueError("no source files found for the code split")
    return "".join(chunks), manifest


def _unique_pairs(count: int, seed: int, low: int, high: int) -> list[tuple[int, int]]:
    capacity = max(high - low, 0) ** 2
    if count < 0 or count > capacity:
        raise ValueError(f"cannot draw {count} unique pairs from range [{low}, {high}) with capacity {capacity}")
    rng = random.Random(seed)
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    while len(pairs) < count:
        pair = (rng.randrange(low, high), rng.randrange(low, high))
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    return pairs


def build_code_task_examples(
    count: int,
    seed: int,
    *,
    low: int,
    high: int,
    shifted: bool = False,
) -> list[TaskExample]:
    examples = []
    for index, (x, y) in enumerate(_unique_pairs(count, seed, low, high)):
        if shifted:
            prompt = f"fn add(a={x}; b={y}) -> "
            lineage = "verified-code-task-v2-shifted"
        else:
            prompt = f"python:add x={x}; y={y}; return="
            lineage = "verified-code-task-v1"
        examples.append(
            TaskExample(
                task_id=f"code-{seed}-{index:04d}",
                domain="code",
                prompt=prompt,
                response=f"{x + y:03d}",
                lineage_key=lineage,
            )
        )
    return examples


def build_math_examples(
    count: int,
    seed: int,
    *,
    low: int,
    high: int,
    shifted: bool = False,
) -> list[TaskExample]:
    examples = []
    for index, (a, b) in enumerate(_unique_pairs(count, seed, low, high)):
        operation = "add" if index % 2 == 0 else "mul"
        answer = a + b if operation == "add" else a * b
        if shifted:
            prompt = f"calculate[{operation}]({a},{b}) => "
            lineage = "verified-math-task-v2-shifted"
        else:
            prompt = f"math-v1 op={operation}; a={a}; b={b}; answer="
            lineage = "verified-math-task-v1"
        examples.append(
            TaskExample(
                task_id=f"math-{seed}-{index:04d}",
                domain="math",
                prompt=prompt,
                response=f"{answer:03d}",
                lineage_key=lineage,
            )
        )
    return examples


def build_parity_examples(
    count: int,
    seed: int,
    *,
    low: int,
    high: int,
    shifted: bool = False,
) -> list[TaskExample]:
    examples = []
    for index, (a, b) in enumerate(_unique_pairs(count, seed, low, high)):
        if shifted:
            prompt = f"calculate[parity]({a},{b}) => "
            lineage = "verified-parity-task-v2-shifted"
        else:
            prompt = f"math-parity-v1 a={a}; b={b}; answer="
            lineage = "verified-parity-task-v1"
        examples.append(
            TaskExample(
                task_id=f"parity-{seed}-{index:04d}",
                domain="math",
                prompt=prompt,
                response=str((a + b) % 2),
                lineage_key=lineage,
            )
        )
    return examples


def task_stream(examples: Iterable[TaskExample], tokenizer: ByteTokenizer | None = None) -> torch.Tensor:
    tokenizer = tokenizer or ByteTokenizer()
    text = "".join(example.text for example in examples)
    if not text:
        raise ValueError("task split is empty")
    return _stream(text, tokenizer)


def build_transfer_suite(
    repo_root: str | Path,
    wiki_root: str | Path,
    *,
    tokenizer: ByteTokenizer | None = None,
    wiki_max_chars: int | None = None,
    code_task_count: int = 256,
    math_task_count: int = 256,
) -> TransferSuite:
    """Build pinned streams without reading any model or personal dataset."""

    tokenizer = tokenizer or ByteTokenizer()
    repo_root = Path(repo_root).expanduser().resolve()
    wiki_root = Path(wiki_root).expanduser().resolve()
    wiki_paths = {
        "train": wiki_root / "wiki.train.raw",
        "valid": wiki_root / "wiki.valid.raw",
        "test": wiki_root / "wiki.test.raw",
    }
    for path in wiki_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    wiki_text = {split: _read_text(path, wiki_max_chars) for split, path in wiki_paths.items()}

    code_train_paths = list((repo_root / "remora").rglob("*.py")) + [repo_root / "train" / "train.py"]
    code_valid_paths = list((repo_root / "experiments").glob("*.py")) + list((repo_root / "tests").glob("*.py"))
    code_train_text, code_train_manifest = _join_files(code_train_paths, repo_root)
    code_valid_text, code_valid_manifest = _join_files(code_valid_paths, repo_root)

    code_task_train = build_code_task_examples(code_task_count, 901, low=0, high=20)
    code_task_valid = build_code_task_examples(96, 902, low=20, high=40)
    code_task_shifted = build_code_task_examples(96, 903, low=40, high=60, shifted=True)
    math_train = build_math_examples(math_task_count, 911, low=0, high=20)
    math_valid = build_math_examples(96, 912, low=20, high=40)
    math_shifted = build_math_examples(96, 913, low=40, high=60, shifted=True)
    # The parity response stays one bit, but the input range must still hold
    # the default 256 unique training pairs. Keep all three ranges disjoint so
    # exact transfer is not accidentally tested on duplicate pairs.
    parity_train = build_parity_examples(math_task_count, 921, low=0, high=32)
    parity_valid = build_parity_examples(96, 922, low=32, high=48)
    parity_shifted = build_parity_examples(96, 923, low=48, high=64, shifted=True)

    # Keep the byte construction explicit: real source remains identifiable in
    # metadata while verified code tasks share the same training stream.
    code_train_combined = code_train_text + "\n" + "".join(example.text for example in code_task_train)
    code_valid_combined = code_valid_text + "\n" + "".join(example.text for example in code_task_valid)
    code_test_combined = "".join(example.text for example in code_task_shifted)

    metadata = {
        "schema": "remora-v1-transfer-suite",
        "tokenizer": {"type": "ascii-byte", "vocab_size": tokenizer.vocab_size},
        "sanitization": "NFKD then ASCII replacement; control bytes except whitespace removed",
        "wiki": {
            "root": str(wiki_root),
            "splits": {
                split: {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                    "sanitized_chars": len(wiki_text[split]),
                }
                for split, path in wiki_paths.items()
            },
        },
        "code": {
            "train_files": code_train_manifest,
            "valid_files": code_valid_manifest,
            "split_policy": "remora and train source for train; experiments and tests for validation",
        },
        "verified_tasks": {
            "code": {
                "train_count": len(code_task_train),
                "valid_count": len(code_task_valid),
                "shifted_count": len(code_task_shifted),
                "train_seed": 901,
                "valid_seed": 902,
                "shifted_seed": 903,
            },
            "math": {
                "train_count": len(math_train),
                "valid_count": len(math_valid),
                "shifted_count": len(math_shifted),
                "train_seed": 911,
                "valid_seed": 912,
                "shifted_seed": 913,
            },
            "parity": {
                "train_count": len(parity_train),
                "valid_count": len(parity_valid),
                "shifted_count": len(parity_shifted),
                "train_seed": 921,
                "valid_seed": 922,
                "shifted_seed": 923,
            },
        },
    }
    metadata["stream_sha256"] = {
        "text_train": _sha256_text(wiki_text["train"]),
        "text_valid": _sha256_text(wiki_text["valid"]),
        "text_test": _sha256_text(wiki_text["test"]),
        "code_train": _sha256_text(code_train_combined),
        "code_valid": _sha256_text(code_valid_combined),
        "code_test": _sha256_text(code_test_combined),
        "math_train": _sha256_text("".join(example.text for example in math_train)),
        "math_valid": _sha256_text("".join(example.text for example in math_valid)),
        "math_shifted": _sha256_text("".join(example.text for example in math_shifted)),
        "parity_train": _sha256_text("".join(example.text for example in parity_train)),
        "parity_valid": _sha256_text("".join(example.text for example in parity_valid)),
        "parity_shifted": _sha256_text("".join(example.text for example in parity_shifted)),
    }

    return TransferSuite(
        train_streams={
            "text": _stream(wiki_text["train"], tokenizer),
            "code": _stream(code_train_combined, tokenizer),
            "math": task_stream(math_train, tokenizer),
            "parity": task_stream(parity_train, tokenizer),
        },
        valid_streams={
            "text": _stream(wiki_text["valid"], tokenizer),
            "code": _stream(code_valid_combined, tokenizer),
            "math": task_stream(math_valid, tokenizer),
            "parity": task_stream(parity_valid, tokenizer),
        },
        test_streams={
            "text": _stream(wiki_text["test"], tokenizer),
            "code": _stream(code_test_combined, tokenizer),
            "math": task_stream(math_shifted, tokenizer),
            "parity": task_stream(parity_shifted, tokenizer),
        },
        code_task_train=code_task_train,
        code_task_valid=code_task_valid,
        code_task_shifted=code_task_shifted,
        math_train=math_train,
        math_valid=math_valid,
        math_shifted=math_shifted,
        parity_train=parity_train,
        parity_valid=parity_valid,
        parity_shifted=parity_shifted,
        metadata=metadata,
    )
