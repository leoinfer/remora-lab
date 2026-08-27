#!/usr/bin/env bash
set -u

# HISTORICAL REFERENCE ONLY.
#
# This is a sanitized transcription of the command that generated the
# recovered receipt. It is not part of HAR, is not called by the public Rust
# runner, and must never become a production dependency. The referenced legacy
# executable and model are intentionally not included in this repository.
# In the historical llama-bench CLI, each -p value is n_prompt: logical
# prefill rows. It is not a shader or workgroup-width selector.

legacy_bin="${R4X_LEGACY_BENCH_BIN:?set R4X_LEGACY_BENCH_BIN to an approved historical binary}"
model="${R4X_MODEL_PATH:?set R4X_MODEL_PATH to the identity-matched local model}"
output_dir="${R4X_WIDTH_OUTPUT_DIR:-width-output}"
mkdir -p "$output_dir"

sample_clocks() {
  (
    while :; do
      echo "T $(date +%s.%N)" >> "$1"
      rocm-smi --showclocks --showpower --showtemp 2>/dev/null \
        | grep -E 'Clocks|Power|Temperature|gfx|sclk|GPU' >> "$1"
      sleep 2
    done
  ) &
  sampler_pid=$!
}

run_bench() {
  tag="$1"
  ubatch="$2"
  prompts="$3"
  (
    sample_clocks "$output_dir/clocks_${tag}.log"
    RADV_PERFTEST=nogttspill GGML_VK_ALLOW_GRAPHICS_QUEUE=1 R4X_TRACE=1 \
      "$legacy_bin/llama-bench" -m "$model" -p "$prompts" -n 0 -r 3 -ub "$ubatch" \
      -ctk f16 -ctv f16 -fa 1 -t 8 -o json \
      > "$output_dir/bench_${tag}.json" \
      2> "$output_dir/trace_${tag}.log"
    kill "$sampler_pid" 2>/dev/null || true
  )
}

run_bench ub512 512 "64,128,256,384,512,768,1024,1536,2048"
run_bench ub4096 4096 "64,128,256,384,512,768,1024,1536,2048"
