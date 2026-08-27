# R4X width sweep

This lane is the recovered, sanitized record of the 2026-08-25 R4X-D32A
full-model prefill width campaign. The authoritative clean receipt was found
locally and is published as [`sanitized_receipt.json`](sanitized_receipt.json).
The raw files and model weights remain outside the public repository.

The measured quantity is **kernel/prefill diagnostic rows per second**. It is
not generation tokens per second. At `ubatch=512`, the controlled sweep peaked
at approximately 700 kernel rows/s in the tested range. It must not be
reported as “R4X does 700 t/s”.

## Reproduction boundary

Run the public, Rust-only validation lane with:

```sh
./repro/r4x/width-sweep/run_width_sweep.sh
```

That command runs the clean-room Rust D32A known-answer test used by this
repository and exits successfully only when the public geometry, nibble order,
scale rule, and extrema gate pass. It does not load the excluded model or
invoke the historical benchmark binary.

The historical throughput receipt is therefore classified
`HISTORICAL_RECONSTRUCTION_AVAILABLE`, not `FULLY_REPRODUCIBLE`: the exact
2026-08-25 executable was built from historical C++/GGML/Vulkan/CMake source at
commit `5181a189ed65ad98b069c6dbdd20d486091d7d60`. That foreign implementation
is not imported into the HAR production path. A future exact rerun requires a
Rust-only R4X model loader/executor and a public model artifact or an approved
download manifest. The public runner is deliberately fail-closed about this
boundary; it cannot fall back to a foreign inference backend.

The sanitized historical command transcription is retained outside `har/` in
[`research/archival/r4x/width-sweep/historical_command.sh`](../../../research/archival/r4x/width-sweep/historical_command.sh).
It is reference evidence only and is not called by the public runner.

## Exact historical target and configuration

| Field | Recovered value |
| --- | --- |
| Model target | Qwen3.8-27B transformed R4X-D32A Q4_0/MTP GGUF artifact |
| Model identity | SHA-256 `ead567f63dc1b1f774fccc6000385b62d5e57d1ac26cb26831cf8592cca049b4` |
| Model file / payload | 15,388,062,592 file bytes; 15,377,076,224 loader-reported packed payload bytes |
| Model structure | GGUF V3; 866 tensors; 506 R4X-D32A tensors; 360 F32 tensors |
| GPU | AMD Radeon RX 9060 XT, RDNA4 `gfx1200`, Vulkan/RADV |
| Reference phenotype | [`RX9060XT16-NITRO-GFX1200-RADV-2026.08.27-v1`](../../../HARDWARE_PROFILE.md) |
| Source build | `5181a189ed65ad98b069c6dbdd20d486091d7d60`, build number `10188` |
| Batch / ubatch | `n_batch=2048`; clean series `n_ubatch=512` |
| Threads | 8; CPU mask `0x0`; strict CPU mode off; poll 50 |
| KV | `ctk=f16`, `ctv=f16`; no KV offload |
| Placement | all layers on Vulkan; layer split; main GPU 0; mmap load |
| Flash attention | enabled |
| Generation | `n_gen=0`; no generation throughput was measured |
| Repetitions | 3 measured repetitions per clean cell |
| Warm-up | default warm-up enabled; one prompt warm-up per parameter instance |
| Timing | host `high_resolution_clock`; each prompt call synchronizes the backend before the interval ends |
| Historical environment | `RADV_PERFTEST=nogttspill`, `GGML_VK_ALLOW_GRAPHICS_QUEUE=1`, `R4X_TRACE=1` |

The receipt itself records RADV GFX1200 but not a per-run Mesa version, clock
lock, or per-cell telemetry file. The public phenotype records Mesa RADV
26.2.1 and the configured board controls; those are explicitly not promoted to
an exact sustained clock for this historical sweep. The companion summary
reported a 3,202 MHz core sample for the broader width-roofline analysis, but
it is not phase-addressable enough to serve as the exact clock context for each
cell.

## What “width” means here

The recovered script passes `-p 64,128,...` to `llama-bench`. In that program,
`-p` is `n_prompt`, and each prompt is processed in batches of at most
`n_batch`. Thus `W` is the logical number of prefill rows supplied to the
model. For `ubatch=512`, widths above 512 are processed as multiple internal
microbatches and the reported value is the aggregate rows/s for the whole
prompt.

This is not a shader local size, subgroup width, workgroup width, or compile-
time tile parameter. The separate `test-r4x-widem` correctness source used in
the historical tree is a matrix-shape probe; it is not the source of the
full-model `llama-bench -p W` numbers published here.

## Metric and gate

For each repetition, the historical benchmark recorded a host elapsed time
`t_ns` around the prompt decode and computed:

```text
sample_rows_per_s = 1e9 * (n_prompt + n_gen) / t_ns
```

Because `n_gen=0`, this is `W / (t_ns / 1e9)`. `avg_ts` is the mean of the
three per-repetition rates, and `stddev_ts` is their sample standard deviation.
The timing source is a host clock, not a GPU timestamp query.

The process-level gate was a successful `llama_decode` return followed by
backend synchronization for each prompt call. The width receipt did not carry
a per-output known-answer comparison. Consequently, it is a performance
receipt, not a standalone numerical-correctness certificate. The separate
public Rust D32A KAT supplies the public format gate; the historical companion
report also recorded zero dequantization events for its native-path trace, but
that raw trace was not preserved in this lane.

## Status boundaries

- `ubatch=512`: complete clean nine-width series, W64 through W2048.
- `ubatch=4096`: the file contains complete prefix observations through W1536,
  then the W2048 run aborts during Vulkan queue submission and GDB text is
  appended to the JSON file. The series is `MALFORMED`; its prefix cells are
  retained as individually complete observations but are not a clean comparison
  series.
- W4096 kernel/prefill width: no authoritative run was found. The recovered
  script requests only W64 through W2048. Other local documents contain W4096
  predictions or preregistration, not a W4096 measurement.
- `ubatch=4096` is not W4096. The former is the microbatch setting; the latter
  would be a prompt-row width. They are kept separate in [`matrix.csv`](matrix.csv).
- W32 and other ubatch values are `NOT_RUN` in this recovered series. Related
  isolated-operator and preregistration surfaces are linked in
  [`analysis.md`](analysis.md) but are not merged into these full-model rows.

See [`analysis.md`](analysis.md) for the peak, delta, plateau, degradation, and
sample-variance discussion. See [`manifest.json`](manifest.json) for the
machine-readable provenance and claim boundary.
