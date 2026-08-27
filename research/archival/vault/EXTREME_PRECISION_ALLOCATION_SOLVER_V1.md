# R4X Extreme Allocation Solver V1

Marginal-utility descent under weight-space cost model v0.
Eligible: 177 linears, 24.56 GiB bf16; anchor = uniform D32A at 4.500 bpw (eligible subset).

| max bpw | rel cost | moves | B1 | T2 | I3 | D32A | I6 | I8 |
|---|---|---|---|---|---|---|---|---|
| 4.5 | 1.0000 | 0 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| 4.0 | 3.1878 | 69 | 0.00 | 0.00 | 0.50 | 0.50 | 0.00 | 0.00 |
| 3.5 | 5.4443 | 177 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| 3.25 | 9.2820 | 187 | 0.12 | 0.00 | 0.88 | 0.00 | 0.00 | 0.00 |
| 3.0 | 13.1931 | 225 | 0.25 | 0.00 | 0.75 | 0.00 | 0.00 | 0.00 |
| 2.75 | 17.0749 | 267 | 0.37 | 0.00 | 0.63 | 0.00 | 0.00 | 0.00 |
| 2.5 | 20.9416 | 317 | 0.50 | 0.00 | 0.50 | 0.00 | 0.00 | 0.00 |
| 2.25 | 24.8862 | 368 | 0.62 | 0.01 | 0.37 | 0.00 | 0.00 | 0.00 |
| 2.0 | 28.8189 | 396 | 0.68 | 0.10 | 0.22 | 0.00 | 0.00 | 0.00 |
| 1.9 | 30.3796 | 411 | 0.80 | 0.00 | 0.20 | 0.00 | 0.00 | 0.00 |

Interpretation: rel cost is the model-weighted squared-error vs the
D32A anchor under unit-variance inputs. Empirical KL/PPL receipts refine
this curve; QAT rescue targets come from soft-cliff regions of it.
