# R4X logical-prefill-row sweep analysis

The clean comparison is the `ubatch=512` slice in
[`sanitized_receipt.json`](sanitized_receipt.json). The companion
`ubatch=4096` attempt is shown only as a malformed-prefix diagnostic; it is not
used to select the peak.

## Clean logical-prefill-row table

| Logical prefill rows W (`llama-bench -p W`) | Mean rows/s | Sample stddev rows/s | Delta from W64 |
| ---: | ---: | ---: | ---: |
| 64 | 441.056190 | 0.369598 | 0.000000 |
| 128 | 571.452541 | 1.560446 | +130.396351 |
| 256 | 661.587407 | 6.572781 | +220.531217 |
| 384 | 687.749345 | 0.168341 | +246.693155 |
| 512 | **699.677849** | 0.331047 | **+258.621659** |
| 768 | 685.977024 | 0.387761 | +244.920834 |
| 1024 | 697.425575 | 0.289948 | +256.369385 |
| 1536 | 696.165057 | 0.155833 | +255.108867 |
| 2048 | 694.516269 | 0.088739 | +253.460079 |

Peak logical prefill-row request: **W512** at **699.677849 logical prefill rows/s**.

Relative to W64, the peak delta is **+258.621659 rows/s**, or
**+58.636896%**. This is a diagnostic prefill rate, not a generation rate.

## Plateau and degradation

The practical plateau is W512, W1024, W1536, and W2048. Across those four
points, the range is only 5.161580 rows/s, 0.743191% of the W512 peak. The
W768 point is a local dip of 1.958% from W512 and returns near the plateau at
W1024. The clean W2048 value is 5.161580 rows/s (0.737708%) below W512.

The data supports “saturation around W512 in this tested range”; it does not
support a sharp W512 optimum or a universal hardware roof. Differences within
the sub-percent plateau should not be overinterpreted, especially because the
receipt has only three repetitions per point and the timing is host-wall based.

## Repeated-measurement variance

The largest within-cell sample standard deviation is W256 at 6.572781 rows/s
(0.994% of its mean). The plateau cells have much smaller within-cell sample
standard deviations: 0.331047, 0.289948, 0.155833, and 0.088739 rows/s for
W512, W1024, W1536, and W2048 respectively. These are descriptive sample
statistics, not confidence intervals and not evidence that the plateau
differences are causally meaningful.

## The malformed ubatch=4096 attempt

The complete prefix observations are retained in `results.csv` and the
receipt. They are:

| Logical prefill rows W | rows/s | sample stddev rows/s |
| ---: | ---: | ---: |
| 64 | 442.027216 | 0.982925 |
| 128 | 573.336955 | 3.174368 |
| 256 | 660.773200 | 4.028288 |
| 384 | 677.965799 | 1.066125 |
| 512 | 692.900523 | 0.157620 |
| 768 | 683.102544 | 0.152077 |
| 1024 | 696.671809 | 0.156760 |
| 1536 | 668.741235 | 0.612971 |

The next W2048 run aborted during Vulkan queue submission; GDB output was
appended directly after the final JSON object, so the file is not valid JSON.
That makes the overall `ubatch=4096` series `MALFORMED`, and the failed W2048
cell is `DEVICE_LOST`/invalid. The prefix is useful for archaeology but is not
a clean ubatch comparison.

## Related work kept separate

The search found other W32/W64 and W4096 references, including isolated
operator surfaces, preregistration, and modelled predictions. They do not share
the exact full-model receipt contract above. The main examples are the
controlled isolated-operator surface and the preregistered W4096 prediction;
neither supplies a valid full-model W4096 throughput receipt. They therefore
remain separate research evidence and are not inserted into the clean matrix.

The matrix in [`matrix.csv`](matrix.csv) records every requested width/ubatch
combination in the audit scope without filling missing cells with estimates.

## Interpretation

The defensible statement is:

> At ubatch 512, the controlled R4X native `llama-bench -p W` campaign peaked
> around 700 logical prefill diagnostic rows/s at W512, with an approximately flat
> W512–W2048 region in this receipt.

It is not defensible to convert this table into an autoregressive generation
tokens/s claim. The exact throughput rerun remains a Rust migration task; the
historical result is preserved as evidence and its foreign implementation is
not an active HAR alternative.
