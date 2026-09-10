from __future__ import annotations

import torch

from .tokenizer import ByteTokenizer


def encode_stream(text: str, tokenizer: ByteTokenizer | None = None) -> torch.Tensor:
    tokenizer = tokenizer or ByteTokenizer()
    return torch.tensor(tokenizer.encode(text), dtype=torch.long)


def sample_batch(
    stream: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if stream.numel() <= seq_len + 1:
        raise ValueError("stream is shorter than sequence length")
    starts = torch.randint(
        0, stream.numel() - seq_len - 1, (batch_size,), generator=generator
    )
    x = torch.stack([stream[int(s) : int(s) + seq_len] for s in starts])
    y = torch.stack([stream[int(s) + 1 : int(s) + seq_len + 1] for s in starts])
    return x.to(device), y.to(device)


def contiguous_batches(
    stream: torch.Tensor, batch_size: int, seq_len: int, device: torch.device
):
    usable = (stream.numel() - 1) // seq_len * seq_len
    x = stream[:usable].view(-1, seq_len)
    y = stream[1 : usable + 1].view(-1, seq_len)
    for start in range(0, x.size(0), batch_size):
        yield x[start : start + batch_size].to(device), y[start : start + batch_size].to(device)
