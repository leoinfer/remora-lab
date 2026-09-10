from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environments.synthetic_corpus import build_heldout_corpus
from remora.data import encode_stream
from remora.metrics import evaluate_stream
from remora.models import build_model
from remora.utils import choose_device, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    device = choose_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    from remora.config import ModelConfig

    cfg = ModelConfig(**ckpt["config"])
    model = build_model(ckpt["model_type"].replace("-v0", "").replace("monolithic-transformer-control", "baseline"), cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    stream = encode_stream(build_heldout_corpus(1000, seed=ckpt["seed"] + 1000))
    result = evaluate_stream(model, stream, args.batch_size, min(96, cfg.max_seq_len), device)
    print(result)


if __name__ == "__main__":
    main()
