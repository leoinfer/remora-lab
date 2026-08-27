#!/usr/bin/env bash
set -euo pipefail

command -v cargo >/dev/null
command -v rustc >/dev/null

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

cargo metadata --no-deps --format-version 1 >/dev/null
printf 'REPRO_ENV PASS: cargo=%s rustc=%s\n' "$(cargo --version)" "$(rustc --version)"
