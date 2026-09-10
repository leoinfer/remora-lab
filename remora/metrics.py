from __future__ import annotations

import math
import time

import torch

from .data import contiguous_batches, sample_batch
from .utils import count_parameters


@torch.no_grad()
def evaluate_stream(model, stream: torch.Tensor, batch_size: int, seq_len: int, device: torch.device) -> dict:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    started = time.perf_counter()
    for x, y in contiguous_batches(stream, batch_size, seq_len, device):
        _, loss = model(x, y)
        n = y.numel()
        total_loss += float(loss) * n
        total_tokens += n
    elapsed = time.perf_counter() - started
    if was_training:
        model.train()
    mean_loss = total_loss / max(total_tokens, 1)
    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(20.0, mean_loss)),
        "tokens": total_tokens,
        "wall_seconds": elapsed,
        "tokens_per_second": total_tokens / max(elapsed, 1e-9),
    }


def train_step(model, stream, batch_size, seq_len, device, generator, optimizer) -> tuple[float, float]:
    started = time.perf_counter()
    model.train()
    x, y = sample_batch(stream, batch_size, seq_len, device, generator)
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    optimizer.step()
    return float(loss.detach()), time.perf_counter() - started, grad_norm


def model_summary(model) -> dict:
    return {"parameters": count_parameters(model), "trainable_parameters": count_parameters(model, True)}
