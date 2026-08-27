---
title: Progressive Future Materialization
type: architecture
status: rejected-pfm-a-deferred-pfm-b
family: PFM
frozen-model-path: PFM-A only
training: PFM-B only
---

# PFM: Progressive Future Materialization

PFM asks whether future computation can begin before the next token is
committed and be progressively refined across logical memory tiers:

```text
future hypothesis -> execution plan -> partially prepared computation
                  -> exact verified state

SKETCH -> BLUEPRINT -> BUILD
```

The names are logical, not hardware contracts. On the current machine the
candidate mapping is usually `NVMe -> RAM -> VRAM`; another phenotype may map
them to cold/warm unified memory or GPU2/GPU1. PFM attacks cold out-of-core
weight/expert movement, but it does not assume that movement is the only
bottleneck.

> **Current verdict:** `REJECT PFM-A AS A DISTINCT MECHANISM; DEFER PFM-B`.
> The fair oracle fails in the established project-parameter envelope. PFM-A's
> state/accounting overlay remains static-only; PFM-B is a later training
> conjecture with no evidence or authorization.

## 1. Scope and non-claims

PFM-A is frozen-model compatible and may use draft/MTP output, n-gram or route
prediction, recent acceptance history, LayerPack plans, and exact path-bound
candidate work. The frozen target remains authoritative, default candidate
width is one, and width two is permitted only under a positive full-cost
expected-value test.

PFM does **not** assume that arbitrary future hidden states or KV states can be
predicted cheaply or corrected cheaply. Exact recurrent/KV/hidden state depends
on the precise token path. A candidate-specific state is not reusable after a
different token is committed unless its dependency and prefix predicates pass.
Cheap residual correction is unestablished and must be measured, not assumed.
Every miss, stale artifact, failed promotion or failed certificate falls back
to the exact target path.

PFM-B is static/formal only until PFM-A oracle economics and future-state
compressibility evidence justify explicit training authorization.

## 2. Exact authoritative state

Let the committed history at step `t` be:

\[
h_t=(y_1,\ldots,y_t).
\]

The frozen target conditional is:

\[
p_\theta(y_{t+1}\mid h_t).
\]

The complete continuation authority is:

\[
X_t=(h_t,K_t,R_t,G_t,M_t,E_t),
\]

where:

- `K_t` is the exact KV cache;
- `R_t` is recurrent or architecture-specific state;
- `G_t` is sampler and RNG state;
- `M_t` is model, build, backend and runtime identity;
- `E_t` is the execution epoch and namespace;
- `h_t` is the exact committed prefix.

The authoritative transition is:

\[
X_{t+1}=F_\theta(X_t,y_{t+1}).
\]

For candidate continuation
`c=(\hat y_{t+1},\ldots,\hat y_{t+k})`, the path-specific state is:

\[
X_{t+k}^{(c)}=F_\theta^{(k)}(X_t,c).
\]

It is valid for continuation only if the committed future actually begins
with `c`. There is no generic exact future state independent of the future
token sequence.

## 3. Artifact classes

### P0 — planning artifact

P0 contains no model state that can be committed. Examples are predicted
LayerPack order, expert IDs, file offsets, read batches, deadlines, queue
assignments, expected bandwidth, candidate paths and placement plans. A P0
artifact may be wrong without changing exactness, but it can still incur
measurable waste or contention.

### P1 — path-exact artifact

For candidate `c`, P1 may contain exact partial activations, candidate KV or
recurrent state, exact target logits, streamed layer outputs, or verifier state.
It is reusable only when the candidate prefix and all dependencies match:

\[
c\preceq h_{\mathrm{new}}\quad\land\quad D(a)=D(X_{\mathrm{current}}).
\]

Here `c \preceq h_new` means that `c` is a prefix of the newly committed
continuation from the same authority boundary; it is not a token-set or
same-final-logit test.

### P2 — approximate predictive artifact

P2 includes low-rank hidden sketches, predicted KV deltas, semantic route
representations, approximate expert sequences or correction vectors. In strict
mode it is advisory only. It may guide prefetch, branch selection, promotion
and scheduling, but it may not become authoritative state without an exact
correction or valid certificate.

## 4. Future Handoff Packet

Every artifact is represented by:

\[
a=(I_a,D_a,c_a,H_a,\tau_a,f_a,p_a,s_a,\rho_a,R_a,C_a,\Gamma_a),
\]

with:

- `I_a`: unique artifact identity;
- `D_a`: dependency and provenance set;
- `c_a`: candidate path or future region;
- `H_a`: predicted token horizon until use;
- `\tau_a`: current physical tier;
- `f_a`: fidelity class P0/P1/P2;
- `p_a`: estimated probability of use;
- `s_a`: estimated probability of staleness;
- `\rho_a`: remaining exact-work fraction;
- `R_a`: resource demand vector;
- `C_a`: measured or explicitly unknown cost estimates;
- `\Gamma_a`: expiry and validity conditions.

The resource vector is:

\[
R_a=(b_{NVMe},b_{RAM},b_{PCIe},b_{VRAM},c_{CPU},c_{GPU},m_{RAM},m_{VRAM}).
\]

The dependency identity must include, as applicable:

```text
model hash
runtime/build/backend hash
kernel or graph generation
prefix hash and position range
sampler/RNG epoch
KV generation and recurrent-state generation
branch ID
execution epoch/namespace
authority root
```

Missing dependency fields are `UNKNOWN`, never zero or an implicit match.

## 5. Artifact state machine

The normal state machine is:

```text
CREATED -> SKETCH -> BLUEPRINT -> BUILD -> VERIFIED -> COMMITTED
```

Exceptional transitions are:

```text
SKETCH   -> EXPIRED
BLUEPRINT-> DEMOTED
BUILD    -> REJECTED
REJECTED -> RECLAIMED
ANY      -> INVALIDATED
```

Transition rules:

1. `CREATED -> SKETCH` requires a unique identity, candidate/region, dependency
   root, evidence class and expiry policy.
2. `SKETCH -> BLUEPRINT` requires a declared resource estimate and a feasible
   parent plan. It may stage P0/P2 information and does not authorize state
   commitment.
3. `BLUEPRINT -> BUILD` requires positive slack, available background budget,
   valid dependencies and no violation of the exact-path reserve. Unknown
   capacity, deadline or dependency fields block promotion.
4. `BUILD -> VERIFIED` requires the authoritative target verifier, exact
   candidate/output/state checks and a matching epoch/fence boundary.
5. `VERIFIED -> COMMITTED` is atomic with the accepted prefix and current
   authority state. A verified artifact for a rejected or stale branch cannot
   be committed.
6. `SKETCH -> EXPIRED` occurs when its deadline or validity window closes.
7. `BLUEPRINT -> DEMOTED` retains only artifacts whose class and dependencies
   permit lower-tier reuse; it does not preserve an invalid P1 state as exact.
8. `BUILD -> REJECTED` occurs on target disagreement, dependency mismatch,
   failed resource reservation, fence failure or expiry.
9. `REJECTED -> RECLAIMED` may salvage only independently valid shared P0/P1
   work. Branch-specific state is not silently relabeled as shared work.
10. `ANY -> INVALIDATED` occurs on model/build/backend/graph/state/sampler/
    epoch mismatch, source mutation, expiry, failed certificate or authority
    reset. Invalidated artifacts cannot be promoted.

## 6. Deadline and slack

Let `H_a` be predicted tokens until use, `r_t` the current exact committed rate,
and `T_ready(a)` the remaining measured time to make the artifact executable,
including queueing and promotion. Then:

\[
T_{available}(a)=\frac{H_a}{r_t},
\qquad
\sigma_a=T_{available}(a)-T_{ready}(a).
\]

Interpretation:

- `\sigma_a \gg 0`: leave the artifact in a distant tier;
- `\sigma_a > 0` and shrinking: begin promotion;
- `\sigma_a \approx 0`: promote immediately only if the reserve remains safe;
- `\sigma_a < 0`: it cannot arrive on time; cancel, downgrade or exact-fallback.

Uncertain `H_a`, `r_t` or `T_ready` must use a conservative interval or block
promotion. Missing values are not treated as zero slack.

## 7. Cost and expected value

The primary reported metrics remain exact committed tokens/s, exact committed
tokens/joule, physical bytes per exact committed token, tier traffic, queueing,
contention, promotion hit rate, wasted preparation and peak state memory.

For policy comparison only, define:

\[
\mathcal C=\lambda_TT+\lambda_EE+\lambda_BB+\lambda_MM,
\]

where `T` is exposed latency, `E` joules, `B` physical bytes moved and `M`
memory opportunity cost. The weights are an explicitly named operating-mode
choice; they never replace the primary metrics.

Let `C_cold(a)` be the measured cost of producing the required exact state with
no useful preparation. Let `C_remain(a)` be the remaining exact cost after
preparation and promotion:

\[
\rho_a=\frac{C_{remain}(a)}{C_{cold}(a)}.
\]

The valid interesting range is `0\le\rho_a\le1`. A measured `\rho_a>1` means
preparation increased remaining work and fails the value gate; it must not be
clamped away.

Gross potential saving is:

\[
G_a=C_{cold}(a)-C_{remain}(a)=(1-\rho_a)C_{cold}(a).
\]

All preparation costs are charged:

\[
C_{prep-total}=C_{create}+C_{refine}+C_{promote}+C_{hold}
+C_{verify}+C_{contention}+C_{queue}+C_{disposal}.
\]

Unknown terms remain unknown and block a promotion claim. Expected net value is:

\[
V(a)=p_a(1-s_a)G_a-C_{prep-total}(a).
\]

An artifact may be created, retained or promoted only when:

\[
V(a)>0\quad\land\quad\sigma_a>0,
\]

subject to the exact-path reserve and artifact validity rules. The promotion
special case is checked directly as:

\[
p_{used}C_{saved}>C_{promotion}+C_{wasted\ promotion}.
\]

A high token probability does not override high movement, state, verification
or contention cost.

## 8. Critical-path interference and resource selection

For resource `j` with measured capacity `\beta_j`, authoritative critical-path
load `L_j^{critical}` and safety reserve `R_j^{safety}`, available background
capacity is:

\[
B_j^{background}=\max(0,\beta_j-L_j^{critical}-R_j^{safety}).
\]

For active artifact set `A`:

\[
\sum_{a\in A}R_{a,j}\le B_j^{background}
\]

must hold for NVMe, RAM, PCIe, GPU compute, CPU compute, VRAM and system RAM.
The selection problem is:

\[
\max_{x_a\in\{0,1\}}\sum_a x_aV(a)
\]

subject to resource constraints and dependency constraints:

\[
x_{child}\le x_{parent}.
\]

Greedy or receding-horizon approximations are allowed only if they preserve
hard reserve, dependency and fail-closed constraints. A background action that
slows the exact path can receive no free-overlap credit.

## 9. LayerPack and full roofline conditions

For LayerPack `l` with `W_l` bytes and effective weight bandwidth
`\beta_{W,l}`:

\[
F_l=\frac{W_l}{\beta_{W,l}}.
\]

For `N` independently prepared candidate positions with arithmetic/KV/activation
cost `C_l` per position:

\[
T_l^{sequential}\approx NF_l+NC_l,
\]

while a layer-stationary schedule is approximately:

\[
T_l^{wavefront}\approx F_l+NC_l+O_l(N),
\]

where `O_l(N)` includes scheduling, branch metadata, additional KV traffic,
synchronization and candidate-state management. The movement-amortization
break-even is:

\[
(N-1)F_l>O_l(N).
\]

It is not enough that resident weights reduce logical bytes; state, traffic,
contention, exposed latency and exact committed-token denominators must also
close.

For a completed candidate block with work `B_j` on each resource and capacity
`\beta_j`, precedence critical path `CP` gives:

\[
T\ge\max\left(CP,\max_j\frac{B_j}{\beta_j}\right).
\]

If `A` exact target-valid tokens are committed:

\[
R_{exact}=\frac{A}{T}
\le
\frac{A}{\max(CP,\max_j B_j/\beta_j)}.
\]

PFM succeeds only if it reduces this denominator per exact committed token.

## 10. Candidate geometry

Let `\mathcal G_t` be the candidate set. The default is:

\[
|\mathcal G_t|=1.
\]

A short second branch `b` is opened only when its full expected value is positive:

\[
p_b(1-s_b)(1-\rho_b)F_{avoided}>C_b,
\]

where `C_b` includes draft and target work, KV/activation allocation, promotion,
verification, scheduling, rollback and opportunity cost. High entropy alone is
not a branch justification.

## 11. Exactness and reuse invariants

Strict PFM must preserve the target conditional distribution:

\[
P_{PFM}(y_{t+1}\mid h_t)=p_\theta(y_{t+1}\mid h_t).
\]

At every token:

1. the target distribution is computed or exactly reconstructed;
2. the authoritative sampler uses the matching `G_t` state;
3. prepared state is reused only when its dependency certificate matches;
4. otherwise full exact fallback executes;
5. rejected suffix state cannot mutate the committed prefix;
6. all artifact writes are epoch/fence ordered and auditable.

Under those conditions, induction yields:

\[
P_{PFM}(y_{1:T})=
\prod_{t=0}^{T-1}p_\theta(y_{t+1}\mid y_{\le t}).
\]

The reuse predicate is:

\[
Valid(a,X_t)=
[D(a)=D(X_t)]
\land[c_a\preceq h_{committed\ future}]
\land[\Gamma_a\text{ unexpired}].
\]

If `Valid(a,X_t)=0`, the artifact cannot become authoritative. For stochastic
sampling, equality means the same target distribution and sampler semantics,
not necessarily the same sampled token; for deterministic greedy controls,
token identity can be compared under the fixed sampler state.

## 12. PFM-A frozen-model algorithm

For each authoritative step:

1. Observe exact `X_t`.
2. Generate candidate set `C_t`, default width one; allow width two only after
   positive full-cost EV.
3. Create P0 Future Handoff Packets containing prefix identity, candidates,
   predicted LayerPacks/experts, offsets, deadlines, resources and expiry.
4. Estimate `p_use`, `p_stale`, `rho`, preparation/promotion/verification/
   contention costs and slack.
5. Solve bounded resource allocation under the exact-path reserve.
6. At NVMe, prepare manifests, group physical reads, stage likely data and
   preserve provenance.
7. At RAM, materialize metadata, exact path-bound inputs and verifier buffers.
8. At VRAM/active pool, promote imminent LayerPacks and retain rollback state;
   speculative work cannot starve current exact work.
9. Let the authoritative target verify the candidate sequence.
10. Commit only the exact accepted prefix and corresponding state.
11. Reclaim only valid shared artifacts; demote reusable plans; expire invalid
    path-specific state.
12. Update measured costs and the Waste Ledger.

PFM-A may use exact path-bound work, but it must not pretend that a predicted
future hidden/KV state is exact merely because the token guess is likely.

## 13. PFM-B trainable future-state conjecture

A later trainable model may emit:

\[
z_t=G_\phi(X_t),
\qquad
z_t^{(k+1)}=R_{\phi,k}(z_t^{(k)},m_k),
\]

and for future candidate `c`:

\[
\widehat X_{t+|c|}=C_\phi(z_t^{(K)},c).
\]

The target state remains:

\[
X_{t+|c|}^{(c)}=F_\theta^{(|c|)}(X_t,c).
\]

State error is:

\[
\epsilon_c=d(\widehat X_{t+|c|},X_{t+|c|}^{(c)}),
\]

but the practical quantity is correction fraction:

\[
\rho_c=\frac{C_{exact-correction}}{C_{full\ target\ pass}}.
\]

PFM-B is interesting only when `rho_c << 1` for a meaningful, calibrated
fraction of likely futures. A possible objective is:

\[
\mathcal L=\mathcal L_{LM}+\alpha\mathcal L_{state}
+\beta\mathcal L_{correction}+\gamma\mathcal L_{calibration}
+\delta\mathcal L_{rank}+\eta\mathcal L_{hardware}.
\]

The terms cover language modeling, state reconstruction, exact-correction
fraction, calibration of `p_use`/`rho`, low-rank future-state residuals and
hardware cost. This objective is a conjectural training design, not an
authorization to train or modify the frozen model.

## 14. Oracle experiment

The first PFM experiment is an offline oracle upper-bound simulation. It is
allowed perfect future tokens, future LayerPack/expert routes, use/staleness
knowledge, residual cost and measured hardware capacities. This is an
impossible upper bound, not a performance result.

Compare under equal model, prompt, output authority, hardware, memory limits,
context length, draft budget and measurement boundaries:

1. sequential out-of-core target;
2. predictive weight/expert prefetch;
3. ordinary speculative decoding;
4. SpecExec-style multi-position verification;
5. PFM-A width one;
6. PFM-A width two;
7. PFM-A oracle;
8. PFM-B hypothetical residual fractions.

Record per exact committed token:

```text
exact committed tokens/s and tokens/J
physical weight, NVMe, RAM, PCIe and KV bytes
critical-path milliseconds
queueing and contention
wasted preparation
promotion hit rate
peak state memory
remaining-work fraction rho
```

PFM-A oracle is an upper bound. If it cannot beat the strongest baseline after
all preparation, promotion, state, verification, contention and opportunity
costs, ordinary PFM-A is rejected or severely scoped.

## 15. Gates and kill conditions

### PFM-0 — oracle viability

Pass only if the oracle beats the strongest baseline by at least 10% higher
exact committed tokens/s with no meaningful energy regression, or at least 10%
lower joules per exact committed token with no meaningful throughput regression.
The allowed meaning of “meaningful regression” must be pre-registered before
replay; no favorable metric may be selected after seeing results.

### PFM-1 — contention safety

When background work fails to produce a useful artifact:

\[
\Delta T_{critical\ path}\le2\%.
\]

A controller that regularly slows current exact work beyond this is unsafe.

### PFM-2 — promotion value

Across matched traces require:

\[
p_{used\ after\ promotion}C_{saved}
>
C_{promotion}+C_{wasted\ promotion}.
\]

### PFM-3 — residual viability

For future-state preparation, the minimum interesting threshold is:

\[
median(\rho)<0.5,
\]

with `median(rho)<0.25` a stronger target. If `rho≈1`, the preparation has
not avoided meaningful exact work. The threshold is not a proof of exactness.

### PFM-4 — baseline superiority

PFM-A must beat sequential offload, predictive prefetch, ordinary speculation
and multi-position verification under equal conditions. All exactness and
measurement gates must pass before any speed or energy claim is promoted.

Kill or scope down PFM-A if the oracle loses, width two is almost never positive
EV, promotion hit rate is low, KV/activation storage dominates, prepared states
still need nearly a full pass, SpecExec captures the benefit, prediction cost
eats the saving, contention destroys overlap, or exact provenance cannot be
maintained.

Kill PFM-B if future deltas are not compressible, low-rank structure disappears,
correction remains near a full pass, quality degrades, or `p_use`/`rho`
calibration is unreliable. Ordinary MTP and latent-token baselines remain
valid competitors.

## 16. Integration with REMORA

PFM is a coordination family above existing mechanisms, not a replacement:

```text
PHASE                  possible future outcomes
TBEH                   horizon/candidate depth
PFM                    progressive materialization and promotion
REMORA Flow            resource and tier scheduling
RSSO                   multi-position LayerPack execution
Persistent Expert Atlas stable expert identity
ExpertPack             reversible physical layout
REMORA Verify          authoritative boundary and rollback
Causal-Closure Cache   legal artifact reuse
ENSM                   artifact namespace/versioning
REMORA Reclaim         valid rejected-work salvage
Waste Ledger           useful/wasted accounting
Maintenance Setpoint   exact-path reserve
Shadow-Price Scheduler dynamic resource pricing
```

The combined loop is:

```text
predict future -> create handoff -> price handoff -> refine -> promote
-> share traversal -> exact verify -> commit -> reclaim or expire
```

## 17. Manifest classification

| Field | Classification |
|---|---|
| Name | Progressive Future Materialization |
| Family | PFM-A / PFM-B |
| Physical bottleneck | cold out-of-core weight and expert movement |
| Dense applicability | yes, PFM-A frozen-host only |
| MoE applicability | yes, with model-specific route/state adapters |
| Hybrid applicability | yes |
| Strict exact mode | yes, target verification plus fallback |
| Frozen-host compatible | PFM-A only |
| Training required | PFM-B |
| Current evidence | none; formal specification only |
| Current verdict | REJECT PFM-A AS DISTINCT MECHANISM; DEFER PFM-B |
| Live implementation | PFM-A hot path rejected; PFM-B training deferred; static overlay only |
| Static work | formal model, schemas, simulator, oracle planner, adversarial tests |
| Priority | below RSSO and accepted-token roofline; above unbounded speculation |

## 18. Dependencies and evidence boundary

PFM-A depends on B0 repeatability, F1/F10 cost ledgers, F4/F6/F8/F9/F12/F14
state/resource checks, valid model-specific traces and exact target authority.
PFM-B additionally depends on future-state compressibility, correction-cost and
calibration evidence plus explicit training authorization.

No current artifact proves PFM economics, future-state compressibility, cheap
residual correction, exact multi-position execution or runtime benefit.
