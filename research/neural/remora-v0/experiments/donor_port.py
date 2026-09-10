from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from remora.donors.port import TeacherPortAdapter, teacher_logit_distillation_loss
from remora.ledger import record_experiment
from remora.utils import changed_parameter_stats, choose_device, count_parameters, parameter_snapshot, set_seed, write_json


class FrozenTeacher(nn.Module):
    """Small deliberately foreign representation used before a real donor runtime."""

    def __init__(self, input_dim: int = 16, hidden_dim: int = 48, classes: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 32), nn.Tanh(), nn.Linear(32, hidden_dim), nn.Tanh())
        self.head = nn.Linear(hidden_dim, classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        return features, self.head(features)


def _dataset(seed: int, n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n, 16, generator=generator, device=device)
    # Four quadrants provide a known external evaluator, not a donor-generated label.
    labels = (x[:, 0] > 0).long() + 2 * (x[:, 1] > 0).long()
    return x, labels


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(-1) == labels).float().mean())


def run(seed: int = 67, device_name: str = "cpu", output: str | Path | None = None) -> dict:
    set_seed(seed)
    device = choose_device(device_name)
    train_x, train_y = _dataset(seed, 768, device)
    test_x, test_y = _dataset(seed + 1, 768, device)

    teacher = FrozenTeacher().to(device)
    teacher_optimizer = torch.optim.AdamW(teacher.parameters(), lr=0.02)
    teacher_history = []
    for _ in range(240):
        teacher_optimizer.zero_grad(set_to_none=True)
        _, logits = teacher(train_x)
        loss = nn.functional.cross_entropy(logits, train_y)
        loss.backward()
        teacher_optimizer.step()
        teacher_history.append(float(loss.detach()))
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    teacher.eval()
    with torch.no_grad():
        train_features, train_logits = teacher(train_x)
        test_features, test_logits = teacher(test_x)
        teacher_test_accuracy = _accuracy(test_logits, test_y)

    # Candidate: only the port and its small output head are trainable.
    port = TeacherPortAdapter(teacher_dim=48, bus_dim=16, bottleneck=12).to(device)
    candidate_head = nn.Linear(16, 4).to(device)
    candidate = nn.ModuleDict({"port": port, "head": candidate_head})
    before_candidate = parameter_snapshot(candidate)
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=0.03)
    candidate_history = []
    for _ in range(240):
        optimizer.zero_grad(set_to_none=True)
        train_packet = port(train_features.unsqueeze(1), producer="synthetic-teacher")
        student_logits = candidate_head(train_packet.latent[:, 0])
        loss = teacher_logit_distillation_loss(student_logits, train_logits, temperature=2.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.0)
        optimizer.step()
        candidate_history.append(float(loss.detach()))
    with torch.no_grad():
        packet = port(test_features.unsqueeze(1), producer="synthetic-teacher")
        candidate_logits = candidate_head(packet.latent[:, 0])

    # Same-width raw-input control; its labels are never supplied to the donor port.
    raw_control = nn.Sequential(nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 4)).to(device)
    raw_optimizer = torch.optim.AdamW(raw_control.parameters(), lr=0.03)
    raw_history = []
    for _ in range(240):
        raw_optimizer.zero_grad(set_to_none=True)
        raw_loss = nn.functional.cross_entropy(raw_control(train_x), train_y)
        raw_loss.backward()
        raw_optimizer.step()
        raw_history.append(float(raw_loss.detach()))
    with torch.no_grad():
        raw_accuracy = _accuracy(raw_control(test_x), test_y)

    result = {
        "schema": "remora-v0-donor-port-result",
        "seed": seed,
        "device": str(device),
        "mode": "MECHANISM_ONLY_SYNTHETIC_FROZEN_TEACHER",
        "donor": {"id": "synthetic-teacher-v1", "hidden_dim": 48, "frozen_during_candidate": True},
        "port": port.interface_signature(),
        "teacher": {
            "train_loss_first_last": [teacher_history[0], teacher_history[-1]],
            "test_accuracy": teacher_test_accuracy,
            "parameters": count_parameters(teacher),
        },
        "candidate": {
            "distillation_loss_first_last": [candidate_history[0], candidate_history[-1]],
            "test_accuracy": _accuracy(candidate_logits, test_y),
            "parameters": count_parameters(candidate),
            "changed_parameters": changed_parameter_stats(before_candidate, candidate),
            "external_evaluator": "quadrant labels from held-out inputs",
        },
        "raw_input_control": {
            "test_accuracy": raw_accuracy,
            "parameters": count_parameters(raw_control),
            "train_loss_first_last": [raw_history[0], raw_history[-1]],
        },
        "promotion": {
            "state": "CANDIDATE",
            "decision_authority": "external held-out evaluator",
            "promoted": False,
            "reason": "port mechanism is validated, but no real open-weight donor was used",
        },
        "interpretation": "MEASURED MECHANISM TEST: a frozen foreign representation can be compressed through donor-port-v1 into a trainable small head; this is not Qwen transfer evidence.",
    }
    if output:
        write_json(output, result)
    record_experiment(
        ROOT,
        "DONOR-PORT-001",
        "A frozen donor representation with a different hidden width can enter the common bus through a trainable, replaceable port without updating the donor.",
        "Train a TeacherPortAdapter(48 -> 16) and small head against frozen teacher logits; compare to a same-width raw-input control under an external quadrant-label evaluator.",
        "The candidate reaches high held-out accuracy, donor parameters remain frozen, and only the port/head update while the result remains explicitly CANDIDATE.",
        "The port cannot train, donor parameters receive gradients, held-out accuracy is no better than an untrained head, or the candidate is silently promoted by its own score.",
        "python -m experiments.donor_port",
        seed,
        {k: v for k, v in result.items() if k not in {"promotion"}},
        result["interpretation"],
        "Repeat with a real local donor runtime, response-verifier controls, and a tokenizer/position mismatch before considering promotion.",
        hardware={"device": str(device), "mode": result["mode"]},
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--device", default="cpu", choices=["cpu", "auto", "cuda"])
    parser.add_argument("--output", default=str(ROOT / "results" / "donor-port.json"))
    args = parser.parse_args()
    print(json.dumps(run(args.seed, args.device, args.output), indent=2))


if __name__ == "__main__":
    main()
