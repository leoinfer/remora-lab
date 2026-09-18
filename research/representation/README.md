# Representation research

How should 125B parameters of MoE expert weights be stored on a machine with
16 GB of VRAM, 32 GB of RAM, and an NVMe device? This family records what has
actually been measured and what has been decided so far.

**Evidence class of everything below:** `MEASURED`, tensor- and
activation-level, on a bounded panel. **Not** capability evidence: no
end-to-end quality, benchmark-retention, or whole-model result is implied, and
the teacher-KL probe for these panels was never captured (`BLOCKED`).

## The two metrics, defined

Numbers in this file are easy to misread, so the two metrics are stated
explicitly. The distinction matters, because some published rows quote one and
some quote the other.

**Weight relRMS** compares a reconstructed weight tensor against the BF16
authority tensor:

```text
rel_rms(a, b) = sqrt(mean((a - b)^2)) / sqrt(mean(b^2)),  b = W_BF16
```

**Local / activation error** is a forward-pass output comparison on captured
MoE inputs, not a weight metric:

```text
relative_output_error(teacher, candidate) = ||teacher - candidate|| / ||teacher||
```

where `teacher` is the BF16 weights' output. Inputs are real captured routed-MoE
activations (a 2048-token calibration pass over all 48 layers, capture point =
the hidden tensor the routed experts consume, CPU executor, 2026-09-16), with a
separate 512-token holdout pass. For `gate` and `up` the input is the captured
hidden; for `down` the input is reconstructed as `silu(gate(x)) * up(x)`. Rows
are filtered to slices where the expert was actually routed.

Panel slices are 9 = 3 targets × 3 roles, on targets deliberately chosen as one
T2-positive, one T2-negative, and one T2-neutral expert. Per-slice payload is
1,638,400 values, so byte rates are exact:

```text
1.75 bpw  =   358,400 B / 1,638,400 values   (28 B per 128-value group)
2.25 bpw  =   460,800 B / 1,638,400 values   (Q2_0 style, block 64)
3.50 bpw  =   716,800 B / 1,638,400 values   (two planes)
```

## Measured bases

Both metrics are given where both were measured, because they are not
interchangeable and the difference has already caused one published
mislabelling.

| Base representation | Bits/weight | Weight relRMS | Activation error |
| --- | --- | --- | --- |
| Raw T1 ternary (absmax group-128) | ~1.75 | ~0.776 | ~0.778 (0.781 heavy-tailed input) |
| LS-reencoded T1, identical bytes | ~1.75 | ~0.609 | ~0.611 (0.607 heavy-tailed input) |
| Q2_0 donor scaffold (block 64) | ~2.25 | ~0.459 (27 slices) / ~0.488 (panel) | ~0.444 |
| T1 + dense ternary T2 plane | ~3.50 | ~0.438 on the measured slice | — |

Three things follow directly from the table:

1. **The LS re-encode is a large free win.** The same ternary pattern at the
   same byte cost drops weight relRMS from ~0.776 to ~0.609 by fitting a single
   group scale per group instead of using the absmax scale. Nothing about the
   container changed.
2. **A 4× weight-relRMS gap to the Q2 donor costs 0.5 bits per weight.** T1 is
   ~1.69× worse than the Q2 donor in weight relRMS for 0.5 bpw less.
3. **Dense T1+T2 is not compelling.** Spending two full planes to reach ~3.5 bpw
   buys a weight relRMS that is only marginally better than the Q2 donor at
   ~2.25 bpw. The extra plane is not paying for itself.

Caveat on the donor row: the donor figures come from decoding the real Q2_0
bytes of the external GGUF and scoring them on real captured activations. They
are *not* a re-encode: a fresh Q2_0 re-encode of the same BF16 source at the
same 2.25 bpw measures ~0.723 weight relRMS, which is a different and much
worse artifact. Do not conflate the donor's stored bytes with a same-rate
re-encode.

## Equal-added-byte correction panel

The more useful question is what a *fixed byte budget* buys. This panel holds
the base fixed, adds exactly 358,400 B per role to every candidate, and measures
the fraction of activation error removed. The equal-byte constraint was
verified in-panel (maximum deviation 0.56% of budget against a 1% tolerance).

Over the **donor base**:

| Correction format | Activation error removed |
| --- | --- |
| Dense ternary T2 residual plane (the published artifact) | ~20.78% |
| Uniform 2-bit / 4-level residual codebook | ~47.66% |
| Rotated 2-bit codebook (two FWHT-128 stages) | ~47.78% |

Over the **T1 base**:

| Correction format | Activation error removed |
| --- | --- |
| Dense ternary T2 residual plane | ~46.0% |
| LS-scale ternary re-encode at the same bytes | ~62.9% |

Conclusions that follow:

- **A 2-bit residual codebook removed about 2.3× more error than dense ternary
  T2** for the same added bytes over the donor base.
- **Rotation contributed almost nothing** on top of the 2-bit codebook
  (47.66% → 47.78%). The gain lives in the codebook, not the rotation.
- **At equal bytes over T1, LS-scale ternary re-encoding beat dense ternary T2**
  (62.9% vs 46.0%). The cheap calibration change outperformed adding a
  structured residual plane.

So correction sidecars remain useful, but T2 is only one candidate correction
format. Ternary should not be assumed to be the universal base.

## Current representation thesis

```text
BF16 authority
   -> heterogeneous base representation
      + optional additive correction
```

Candidate palette, to be selected per layer, per role, and possibly per expert
by measured value per byte:

- LS-T1 (the calibrated ternary base);
- Q2-class, Q3-class, Q4-class, and higher precision where justified;
- a 2-bit residual codebook as the default correction sidecar;
- dense ternary T2 only where it measurably wins its byte budget;
- selective high-precision islands when they pass a value-per-byte test.

There is no single global answer, and this file deliberately does not claim one.

## High-precision islands, and the boundary that comes with them

A bounded residual-island experiment on layer 0 / expert 283 stored a Q8_0
residual plane over all three roles and measured weight relRMS:

| State | Weight relRMS |
| --- | --- |
| T1 base, layer 0 / expert 283 | ~0.779 |
| T1 + Q8_0 residual island | ~0.0032 |

Payload: 5,222,400 B total, 1,740,800 B per expert-role. Per role the gain is
237.5× (gate), 242.0× (up), and 245.7× (down).

This proves the additive island mechanism can recover near-BF16 local fidelity
on a selected expert. It does **not** make the representation efficient:
T1 + a Q8_0 island costs roughly 10.25 bits per weight, which is 4.86× the
equal-byte budget used by the panel above, and a stronger base codec reaches
better fidelity for far fewer bytes. The island result is therefore published
as mechanism evidence — optional high-fidelity corrective payloads work — and
explicitly not as the final representation.

## What is not decided, and what is not measured

Not decided:

- which base representation wins globally (evidence supports heterogeneity,
  not a single winner);
- whether any per-expert allocation policy survives on a wider panel than
  9 slices;
- how much of this survives contact with the full 48-block bank.

Not measured:

- full BF16 capability retention;
- benchmark retention (AutomationBench, Artificial Analysis, or any other);
- end-to-end quality of the final V2 bank;
- teacher-KL for these panel candidates — the probe is recorded as
  `BLOCKED_NOT_CAPTURED`, so every row in this file is a fidelity measurement,
  not a capability measurement.

## Receipts

The underlying receipts are sanitized under
[`repro/flash-next/representation-panel/`](../../repro/flash-next/representation-panel/).
Weight and activation payloads, the BF16 source tree, and the model files
themselves are not part of this repository.

Two defects found while publishing this material are recorded rather than
quietly fixed, because both are the kind of error this repository exists to
catch:

- one receipt records a per-role T1 payload as 1,433,600 B; the correct value is
  358,400 B (a `float32` byte-count unit error). All other receipts and the
  panel use 358,400 B.
- an earlier prose summary printed the donor's *activation* error (~0.444) in a
  **weight relRMS** column. The donor's weight relRMS is ~0.459–0.488. Ratios
  derived from the mislabelled cell (for example "T1 is 1.77× worse than the
  donor per weight") should be ~1.6–1.7× when computed against weight relRMS.
