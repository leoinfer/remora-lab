# Qwen GatedDeltaNet core transplant

This tranche tests whether trained Qwen numerical structure can be reused by
an aged Remora checkpoint without relearning the donor computation. It does
not claim broad language transfer.

## Selected organ

The bounded scan selected Qwen3.8-Flash-Next layer 17, linear-attention value
head 10 (key head 3). The smallest tested closed subgraph is:

`q/k/v + beta/decay projections + causal qkv convolution + 128x128 delta recurrence`

The Qwen `z` gate, RMS normalization, and output projection are excluded as
co-adapted outer machinery. The extraction reads only the selected slices of
`model-00051-of-00131.safetensors`; the 360 GB source is never materialized.

## Preservation and accounting

The full extracted head contains 1,645,186 parameters (3,290,372 BF16
payload bytes). The compact Remora core contains 38,594 parameters:

| category | parameters |
| --- | ---: |
| source selected payload | 1,645,186 |
| source core kept unchanged (conv/scalars) | 1,538 |
| source core projections entering analytic conversion | 988,160 |
| converted resident projections | 37,056 |
| excluded non-core source tensors | 655,488 |
| discarded by rank-96 width conversion | 951,104 |
| resident compact core | 38,594 |
| rank-4 learned repair port | 2,432 |

The core-only organ matches the converted full-head recurrent path exactly on
the recorded probe (`max_absolute_error=0`, relative L2 `0`). A second
native-width wrapped experiment retains 989,698 source core parameters
unchanged and matches the compact core with relative L2 about `5.9e-7`; the
tradeoff is a 491,520-parameter frozen input port and 1,571,328 modeled
MAC/token socket cost.

## Measured results

On the donor-matched delayed associative recall task inside the real aged
`blocks.1.plastic` path, the compact frozen donor core scored `0.5703 ± 0.0547`
at zero repair steps over seeds 7, 19, and 31. The random frozen core scored
`0.2865 ± 0.0325`; the paired donor-minus-random difference was
`0.2839 ± 0.0226`. The fresh fully trainable same-mechanism core (38,594
trainable parameters) did not reach the fixed 0.50 threshold within 64 steps
on any seed. These are conditional pathway measurements because the queries
were analytically synthesized toward the selected donor key.

The shifted-interface control failed: compact actual accuracy was
`0.2266 ± 0.0475`, near four-way chance. Rank-4 and rank-96 shifted port
repair also failed to reach 0.50, and rank-96 sometimes favored the random
core. This is evidence against the current bus/socket abstraction, not
evidence that the donor weights are useless.

On ordinary aged-checkpoint text/code evaluation, inserting the compact core
raised loss by roughly `0.066`/`0.048` on average (text/code) versus the aged
checkpoint; actual-vs-zero insertion was nearly unchanged. No broad language
capability gain was measured.

Original Qwen pretraining compute, energy, and an economic compute-avoidance
ratio remain **UNMEASURED**. Zero-gradient assimilation is measured for the
bounded pathway only; it must not be described as reclaiming the unknown Qwen
training bill.

## Reproduction

From the repository root, with the source path available:

```bash
python -m unittest discover -v

flock -n /tmp/remora-v0-gpu.lock env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.donor_gdn_remora_pathway \
  --source /home/leo/models/Qwen3.8-Flash-Next-BF16-source \
  --seeds 7,19,31 --repair-steps 0,1,2,4,8,16,32,64 \
  --output results/qwen-neural-gdn-remora-path-v6.json \
  --experiment-id QWEN-DONOR-GDN-REMORA-PATH-005

flock -n /tmp/remora-v0-gpu.lock env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.donor_gdn_interface_repair \
  --source /home/leo/models/Qwen3.8-Flash-Next-BF16-source \
  --seeds 7,19,31 --repair-ranks 4,96 --repair-steps 0,1,2,4,8,16,32,64 \
  --output results/qwen-neural-gdn-interface-repair-v2.json \
  --experiment-id QWEN-DONOR-GDN-INTERFACE-REPAIR-002

flock -n /tmp/remora-v0-gpu.lock env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.donor_gdn_native_core \
  --source /home/leo/models/Qwen3.8-Flash-Next-BF16-source \
  --seeds 7,19,31 --output results/qwen-neural-gdn-native-core-v2.json \
  --experiment-id QWEN-DONOR-GDN-NATIVE-CORE-002

python -m experiments.analyze_qwen_gdn_tranche \
  --path-result results/qwen-neural-gdn-remora-path-v6.json \
  --interface-result results/qwen-neural-gdn-interface-repair-v2.json \
  --native-result results/qwen-neural-gdn-native-core-v2.json \
  --output results/qwen-neural-gdn-tranche-analysis-v2.json
```

The current promotion state is conditional/candidate-only. The next donor
experiment should use a non-donor-conditioned held-out capability or train a
transplant-tolerant semantic bus. Remora should remain at approximately 1.7M
parameters until that failure is addressed.
