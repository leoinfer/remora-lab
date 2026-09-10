from __future__ import annotations

"""Deterministic multi-lifetime concept/interface curriculum.

Each stage has a known external oracle.  The underlying operation is reused
across several deliberately different surface protocols, while later stages
introduce genuinely new operations.  Every split uses disjoint operand pairs
and carries a lineage key so duplicate/correlated evidence can be audited.
"""

import hashlib
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class LifetimeExample:
    task_id: str
    stage_id: str
    concept_id: str
    interface_id: str
    prompt: str
    response: str
    lineage_key: str
    evidence_cluster: str

    @property
    def text(self) -> str:
        return f"{self.prompt}{self.response}\n"


@dataclass(frozen=True)
class LifetimeTask:
    stage_id: str
    concept_id: str
    primary_interface: str
    train: tuple[LifetimeExample, ...]
    valid: tuple[LifetimeExample, ...]
    shifted: tuple[LifetimeExample, ...]
    unseen: tuple[LifetimeExample, ...]
    threshold: float
    description: str


def _unique_pairs(count: int, seed: int, low: int, high: int) -> list[tuple[int, int]]:
    capacity = max(high - low, 0) ** 2
    if count < 0 or count > capacity:
        raise ValueError(
            f"cannot draw {count} unique pairs from range [{low}, {high}) with capacity {capacity}"
        )
    rng = random.Random(seed)
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    while len(pairs) < count:
        pair = (rng.randrange(low, high), rng.randrange(low, high))
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    return pairs


def _answer(concept_id: str, a: int, b: int) -> str:
    if concept_id == "parity":
        return str((a + b) % 2)
    if concept_id == "mod3":
        return str((a + b) % 3)
    if concept_id == "compare":
        return str(int(a > b))
    raise ValueError(f"unknown lifetime concept: {concept_id}")


def _prompt(concept_id: str, interface_id: str, a: int, b: int) -> str:
    if concept_id == "parity":
        if interface_id == "primary":
            return f"parity-v1 x={a}; y={b}; result="
        if interface_id == "shifted":
            return f"The sum of {a} and {b} has parity: "
        if interface_id == "unseen":
            return f"x:{a} y:{b} | parity(sum) -> "
    elif concept_id == "mod3":
        if interface_id == "primary":
            return f"residue3-v1 left={a}; right={b}; class="
        if interface_id == "shifted":
            return f"reduce modulo three: ({a} plus {b}) = "
        if interface_id == "unseen":
            return f"r3[{a}|{b}] => "
    elif concept_id == "compare":
        if interface_id == "primary":
            return f"compare-v1 a={a}; b={b}; a_greater="
        if interface_id == "shifted":
            return f"Is {a} larger than {b}? answer "
        if interface_id == "unseen":
            return f"gt? first:{a} second:{b} -> "
    raise ValueError(f"unknown concept/interface: {concept_id}/{interface_id}")


def _make_split(
    *,
    stage_id: str,
    concept_id: str,
    interface_id: str,
    pairs: list[tuple[int, int]],
    split: str,
    evidence_cluster: str,
) -> tuple[LifetimeExample, ...]:
    return tuple(
        LifetimeExample(
            task_id=f"{stage_id}-{split}-{index:04d}",
            stage_id=stage_id,
            concept_id=concept_id,
            interface_id=interface_id,
            prompt=_prompt(concept_id, interface_id, a, b),
            response=_answer(concept_id, a, b),
            lineage_key=f"{stage_id}:{split}:pair:{a}:{b}",
            evidence_cluster=evidence_cluster,
        )
        for index, (a, b) in enumerate(pairs)
    )


def _split_hash(examples: tuple[LifetimeExample, ...]) -> str:
    digest = hashlib.sha256()
    for example in examples:
        digest.update(
            (
                f"{example.task_id}\0{example.prompt}\0{example.response}\0"
                f"{example.lineage_key}\0{example.evidence_cluster}\n"
            ).encode("ascii")
        )
    return digest.hexdigest()


def build_lifetime_curriculum(
    *,
    train_count: int = 64,
    eval_count: int = 32,
    seed: int = 1701,
) -> tuple[tuple[LifetimeTask, ...], dict]:
    """Build five ordered lifetimes: three parity interfaces and two new concepts."""

    if train_count <= 0 or eval_count <= 0:
        raise ValueError("train_count and eval_count must be positive")
    specs = (
        ("T1", "parity", "primary", "parity-v1", "first inherited-compatible parity protocol"),
        ("T2", "parity", "primary", "parity-natural-v1", "parity under natural-language surface form"),
        ("T3", "parity", "primary", "parity-symbolic-v1", "parity under compact symbolic surface form"),
        ("T4", "mod3", "primary", "mod3-v1", "new modulo-three concept"),
        ("T5", "compare", "primary", "compare-v1", "new ordered-comparison concept"),
    )
    tasks: list[LifetimeTask] = []
    metadata_tasks: list[dict] = []
    for index, (stage_id, concept_id, _primary, interface_label, description) in enumerate(specs):
        # Stage-specific ranges prevent exact operand reuse across lifetimes.
        base = index * 48
        train_pairs = _unique_pairs(train_count, seed + index * 11, base, base + 16)
        valid_pairs = _unique_pairs(eval_count, seed + index * 11 + 1, base + 16, base + 24)
        shifted_pairs = _unique_pairs(eval_count, seed + index * 11 + 2, base + 24, base + 32)
        unseen_pairs = _unique_pairs(eval_count, seed + index * 11 + 3, base + 32, base + 40)
        # The surface IDs distinguish the semantic protocol from the task's
        # primary label.  T2/T3 use a different primary formatter below.
        if stage_id == "T2":
            primary_interface = "natural"
        elif stage_id == "T3":
            primary_interface = "symbolic"
        else:
            primary_interface = "primary"
        # The formatter is concept-specific; map the repeated parity stages to
        # deliberately different interfaces without changing the oracle.
        def stage_prompt(concept: str, interface: str, a: int, b: int) -> str:
            if concept == "parity" and stage_id == "T2" and interface == "primary":
                return f"The parity of the sum of a={a} and b={b} is "
            if concept == "parity" and stage_id == "T3" and interface == "primary":
                return f"P[ a={a} ; b={b} ] -> "
            return _prompt(concept, interface, a, b)

        def split(interface: str, pairs: list[tuple[int, int]], split_name: str, cluster: str):
            return tuple(
                LifetimeExample(
                    task_id=f"{stage_id}-{split_name}-{j:04d}",
                    stage_id=stage_id,
                    concept_id=concept_id,
                    interface_id=interface,
                    prompt=stage_prompt(concept_id, interface, a, b),
                    response=_answer(concept_id, a, b),
                    lineage_key=f"{stage_id}:{split_name}:pair:{a}:{b}",
                    evidence_cluster=cluster,
                )
                for j, (a, b) in enumerate(pairs)
            )

        train = split("primary", train_pairs, "train", f"{stage_id}-train-cluster")
        valid = split("primary", valid_pairs, "valid", f"{stage_id}-valid-cluster")
        shifted = split("shifted", shifted_pairs, "shifted", f"{stage_id}-shifted-cluster")
        unseen = split("unseen", unseen_pairs, "unseen", f"{stage_id}-unseen-cluster")
        threshold = 0.75 if concept_id in {"parity", "compare"} else 0.60
        task = LifetimeTask(
            stage_id=stage_id,
            concept_id=concept_id,
            primary_interface=interface_label,
            train=train,
            valid=valid,
            shifted=shifted,
            unseen=unseen,
            threshold=threshold,
            description=description,
        )
        tasks.append(task)
        metadata_tasks.append(
            {
                "stage_id": stage_id,
                "concept_id": concept_id,
                "primary_interface": interface_label,
                "description": description,
                "threshold": threshold,
                "splits": {
                    name: {
                        "count": len(getattr(task, name)),
                        "sha256": _split_hash(getattr(task, name)),
                        "interface_ids": sorted({example.interface_id for example in getattr(task, name)}),
                        "evidence_clusters": sorted({example.evidence_cluster for example in getattr(task, name)}),
                    }
                    for name in ("train", "valid", "shifted", "unseen")
                },
            }
        )
    metadata = {
        "schema": "remora-v1-lifetime-curriculum",
        "seed": seed,
        "train_count": train_count,
        "eval_count": eval_count,
        "task_order": [task.stage_id for task in tasks],
        "concept_order": [task.concept_id for task in tasks],
        "no_exact_operand_reuse_across_stages": True,
        "oracle": "external deterministic parity, modulo-three, and comparison functions",
        "tasks": metadata_tasks,
    }
    overall = hashlib.sha256()
    for item in metadata_tasks:
        overall.update(repr(item).encode("ascii"))
    metadata["curriculum_sha256"] = overall.hexdigest()
    return tuple(tasks), metadata


def all_examples(task: LifetimeTask) -> tuple[LifetimeExample, ...]:
    return task.train + task.valid + task.shifted + task.unseen
