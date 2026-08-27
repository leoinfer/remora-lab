# Key-first / value-late model

Split execution loads/scans exact keys first, then materializes values only for
surviving blocks:

```
C_split = C_K + p_survive C_V + O_split
C_ordinary = C_KV
```

The CPU sweep uses half of the F16 KV byte model for K and V and varies
`p_survive`, block size, RAM bandwidth, and PCIe bandwidth.  At `p_survive=1`,
the extra pass/bookkeeping loses.  At lower `p_survive` it can win only as a
hypothesis; omitting a value block is approximate unless a finite-precision
certificate proves output/state invariance.  The default concentration-derived
survival values are synthetic sensitivities, not attention traces.

The exact-lane verdict is therefore narrow: key-first is a useful measurement
plan and perhaps a certified-zero optimization, not an unverified retrieval
mechanism.
