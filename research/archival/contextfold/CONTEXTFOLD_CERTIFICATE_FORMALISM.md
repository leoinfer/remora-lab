# Delta-certified context elimination formalism

Let a block's verified logit upper bound be `u_b` and a verified lower bound on
the global maximum be `m_lower`.  Under a declared target exponential
implementation, an exact-zero certificate is available if

```
exp_target(u_b - m_lower) == 0
```

with the inequality evaluated conservatively away from the implementation's
boundary.  Then every allowed logit in the block has zero target weight, so
`l_b=0` and `o_b=0` under that contract.

A stronger local finite-precision condition can bound `Delta_l` and each
absolute output contribution `Delta_o`.  If, for the fixed reduction order and
round-to-nearest arithmetic,

```
Round(l_acc + Delta_l) == Round(l_acc)
Round(o_acc + Delta_o) == Round(o_acc)
```

for every allowed contribution, the block is invariant at that accumulator
boundary.  The prototype uses a conservative half-ulp sufficient condition.

## Required proof inputs

- norm bounds for q and every key/value in the block;
- exact scale, mask, bias, RoPE/position and logit cap;
- target exp/flush-to-zero behavior;
- accumulator precision, rounding mode, FMA/reduction order;
- value norm/component bounds and downstream output/state rounding;
- proof that the lower global maximum remains valid as other blocks arrive.

A next-token equality, attention-mass threshold, or learned predictor is not a
certificate.  If the certificate cannot compose through the recurrent/output
authority boundary, it is advisory only.  The CPU prototype passes local tests
but reports `authority_boundary_composable=false`.
