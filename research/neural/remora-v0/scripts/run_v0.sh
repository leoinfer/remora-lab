#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${REMORA_PYTHON_BIN:-/home/leo/.venvs/remora-rocm10/bin/python}"
LOCK_PATH="/tmp/remora-v0-gpu.lock"

cd "$ROOT_DIR"
exec 9>"$LOCK_PATH"
if ! flock -n 9; then
  echo "GPU lock is held: $LOCK_PATH" >&2
  exit 75
fi

"$PYTHON_BIN" -m train.train \
  --model remora --steps 400 --batch-size 32 --seq-len 96 \
  --seed 7 --run-id remora-v0-scratch-seed7 --device auto

"$PYTHON_BIN" -m train.train \
  --model baseline --steps 400 --batch-size 32 --seq-len 96 \
  --seed 7 --run-id baseline-v0-scratch-seed7 --device auto

"$PYTHON_BIN" -m experiments.run_experiments \
  --quick \
  --remora-checkpoint checkpoints/remora-v0-scratch-seed7.pt \
  --baseline-checkpoint checkpoints/baseline-v0-scratch-seed7.pt
