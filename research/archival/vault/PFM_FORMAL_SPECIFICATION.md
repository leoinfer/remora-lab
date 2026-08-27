---
title: PFM Formal Specification
type: formal-specification
status: provisional-oracle-pending
family: PFM-A / PFM-B
authority: canonical-static-ingestion
---

# Progressive Future Materialization — formal specification

This specification separates two tracks:

- **PFM-A — Frozen-Model Temporal Staging:** a systems/control proposal for
  preparing path-bound work before it becomes authoritative.
- **PFM-B — Trainable Future-State Materialization:** a model-architecture
  proposal for learning compact future representations and cheap exact
  correction.

This document is a formal and static record. It is not a runtime certificate.
The latest CPU/static investigation rejects PFM-A as a distinct incremental
mechanism in its evidence/project-parameter envelope and defers PFM-B. No PFM
GPU, inference, training or production-hot-path result is claimed.

## 1. Authority and central hypothesis

The committed token history at position `t` is

\[
h_t=(y_1,\ldots,y_t).
\]

A frozen target defines

\[
p_\theta(y_{t+1}\mid h_t).
\]

PFM asks whether likely future computation can be prepared before it becomes
authoritative and progressively refined through logical fidelity stages:

```text
SKETCH -> BLUEPRINT -> BUILD -> VERIFIED -> COMMITTED
```

The stages are logical, not permanent hardware names. `NVMe -> RAM -> VRAM`
is one possible physical mapping on this machine, but a future implementation
must measure the actually exposed movement path.

The narrow PFM-A claim is:

> A typed, dependency-closed, deadline-aware preparation packet can reduce
> exact committed-token cost beyond the strongest equal-information baseline
> when its exact path-bound saving exceeds all preparation, promotion, holding,
> queueing, contention and disposal costs.

The narrow PFM-B claim is:

> A model trained to expose a compact candidate-conditioned future
> representation can reduce the exact correction fraction for likely futures
> while preserving target quality and calibrating use probability and residual
> cost.

Neither claim is established by this specification.

## 2. Exact authoritative state

Define the complete exact state at `t` as

\[
X_t=(h_t,K_t,R_t,G_t,M_t,E_t),
\]

where:

- `h_t` is the exact committed token prefix;
- `K_t` is exact KV state;
- `R_t` is recurrent or architecture-specific state;
- `G_t` is sampler and RNG state;
- `M_t` is model, runtime, build, backend and kernel identity;
- `E_t` is execution epoch and namespace.

The exact target transition is

\[
X_{t+1}=F_\theta(X_t,y_{t+1}).
\]

For candidate sequence

\[
c=(\hat y_{t+1},\ldots,\hat y_{t+k}),
\]

the candidate-specific state is

\[
X_{t+k}^{(c)}=F_\theta^k(X_t,c).
\]

There is no generic exact future state independent of the future token path
for an ordinary frozen autoregressive transformer. Candidate probability is
not a state certificate, equal final logits are not a dependency certificate,
and token-set equality is not path equality.

## 3. Artifact classes and handoff packet

PFM uses three artifact classes:

- **P0 — planning:** candidate tokens, LayerPack order, expert IDs, offsets,
  read plans, deadlines, placement plans and probability estimates. P0 is
  advisory and can never be committed as model state.
- **P1 — path-exact:** exact candidate KV, partial activations, verifier state,
  logits or recurrent state. P1 is usable only when the committed path and
  every dependency match.
- **P2 — approximate predictive:** low-rank sketches, approximate KV deltas,
  semantic route sketches or predicted correction vectors. P2 is advisory in
  strict mode and requires exact reconstruction, correction or a valid
  certificate before any authoritative use.

Every packet is

\[
a=(I_a,D_a,c_a,H_a,\tau_a,f_a,p_a,s_a,\rho_a,R_a,C_a,\Gamma_a),
\]

where `I_a` is identity, `D_a` dependencies/provenance, `c_a` candidate path,
`H_a` horizon, `\tau_a` physical tier, `f_a` fidelity class, `p_a` use
probability, `s_a` staleness probability, `\rho_a` remaining exact-work
fraction, `R_a` resource demand, `C_a` cost estimates and `\Gamma_a` expiry
and validity conditions.

The resource vector is

\[
R_a=(b_{NVMe},b_{RAM},b_{PCIe},b_{VRAM},c_{CPU},c_{GPU},m_{RAM},m_{VRAM}).
\]

The dependency identity must include, where applicable:

```text
model hash; runtime/build/backend hash; kernel/graph generation;
prefix hash and position range; sampler/RNG epoch; KV and recurrent-state
generations; branch ID; execution epoch/namespace; authority root
```

Unknown fields are `UNKNOWN`, never zero and never an implicit match.

## 4. Lifecycle and exactness rules

The normal lifecycle is:

```text
CREATED -> SKETCH -> BLUEPRINT -> BUILD -> VERIFIED -> COMMITTED
```

Allowed exceptional transitions are `SKETCH -> EXPIRED`, `BLUEPRINT ->
DEMOTED`, `BUILD -> REJECTED`, `REJECTED -> RECLAIMED` only for independently
valid shared work, and `ANY -> INVALIDATED` on dependency, epoch, expiry,
source, certificate or authority failure.

`BLUEPRINT -> BUILD` requires positive slack, complete dependencies, an
available background reserve and no critical-path safety violation.
`BUILD -> VERIFIED` requires target-authoritative exact output/state checks and
matching epoch/fence boundaries. `VERIFIED -> COMMITTED` is atomic with the
accepted prefix. Rejected-suffix state is never silently relabeled as shared
state.

The strict-mode invariant is

\[
P_{PFM}(y_{t+1}\mid h_t)=p_\theta(y_{t+1}\mid h_t).
\]

For a candidate artifact:

\[
Valid(a,X_t)=dependency\_match(a,X_t)\land
path\_match(a,committed\_future)\land\neg expired(a).
\]

If `Valid` is false, the artifact is advisory/rejected and exact target work
continues. By induction, a legal strict-mode run preserves

\[
P_{PFM}(y_{1:T})=\prod_{t=0}^{T-1}p_\theta(y_{t+1}\mid y_{\le t}).
\]

## 5. Deadline, value and resource equations

Let `H_a` be tokens until predicted use, `r_t` current exact committed tokens
per second, and `T_ready(a)` the measured remaining creation, queueing and
promotion time. Then

\[
T_{available}(a)=H_a/r_t,\qquad
\sigma_a=H_a/r_t-T_{ready}(a).
\]

Positive slack is necessary, not sufficient. Large positive slack permits a
distant tier; shrinking positive slack triggers promotion; near-zero slack
requires immediate promotion only if reserves remain safe; negative slack
cancels, downgrades or falls back to exact work.

Let `C_cold(a)` be cold exact-state cost and `C_remain(a)` remaining exact
cost after preparation. Define

\[
\rho_a=C_{remain}(a)/C_{cold}(a),\quad 0\le\rho_a\le1,
\]

\[
G_a=(1-\rho_a)C_{cold}(a),
\]

and

\[
\begin{aligned}
C_{prep\_total}(a)={}&C_{create}+C_{refine}+C_{promote}+C_{hold}\\
&+C_{verify}+C_{contention}+C_{queue}+C_{disposal}.
\end{aligned}
\]

Expected value is

\[
V(a)=p_a(1-s_a)(1-\rho_a)C_{cold}(a)-C_{prep\_total}(a).
\]

Creation, retention or promotion is legal only when `V(a)>0` and `\sigma_a>0`.

For each variable the controller must record the following operational
metadata:

| Variable | Units | Measurement/estimation | Update rule | Decision boundary |
|---|---|---|---|---|
| `H_a` | exact committed tokens | held-out predictor or exact trace | update after each authoritative token | unknown blocks promotion |
| `r_t` | exact tokens/s | monotonic wall-clock around fixed contract | rolling conservative lower bound | zero/unknown blocks promotion |
| `T_ready` | seconds | timed create/refine/queue/promote path | exponentially weighted upper bound | negative slack cancels |
| `p_a`, `s_a` | probability in `[0,1]` | held-out use/stale events with calibration | update by versioned trace window | uncalibrated values are unknown |
| `C_cold`, `C_remain` | seconds, joules or resource units | paired exact baseline and certified residual | update only from matched runs | `rho` outside `[0,1]` rejects |
| `R_a` | bytes, bytes/s, CPU/GPU units | telemetry or declared oracle input | update per packet and tier | unknown demand blocks selection |
| `V(a)` | same cost unit as `C` | complete value expression | recompute on every dependency change | must be positive |

Background capacity for resource dimension `j` is

\[
B_{background,j}=\max(0,Capacity_j-CriticalLoad_j-SafetyReserve_j).
\]

For selected packets `A`,

\[
\sum_{a\in A}x_aR_{a,j}\le B_{background,j},\qquad x_a\in\{0,1\}.
\]

The controller objective may be written as

\[
\max_{x_a\in\{0,1\}}\sum_a x_aV(a)
\]

subject to the resource constraints above and the dependency constraint

\[
x_{child}\le x_{parent}.
\]

Child packets cannot be selected without their parent plan. A greedy or
receding-horizon controller is acceptable only after comparison with an oracle
on the same packet trace; an exact integer solver is not required.

## 6. Physical LayerPack boundary

For LayerPack `l` with physical bytes `W_l` and exposed movement bandwidth
`\beta_{W,l}`,

\[
F_l=W_l/\beta_{W,l}.
\]

For `K` separately executed positions and a layer-stationary wavefront:

\[
T_{sequential,l}\approx KF_l+KC_l,
\]

\[
T_{wavefront,l}\approx F_l+KC_l+O_l(K),
\]

where `O_l(K)` includes scheduling, KV/activation traffic,
synchronization, branch metadata and rollback. The physical break-even is

\[
(K-1)F_l>O_l(K).
\]

Logical work reduction or a renamed packet does not satisfy this condition.
`W_l`, bytes moved and all timing terms must be exposed physical measurements
or explicitly labelled oracle inputs.

## 7. PFM-A — frozen-model track

PFM-A may schedule draft/MTP predictions, n-gram predictions, route/expert
predictions, predicted LayerPacks, deadline-aware prefetch, staged transfers
and exact path-bound speculative state. It must not assume arbitrary future
hidden/KV state is cheaply correctable, that independent paths reconverge, that
background work is free, or that NVMe matters without exposed traffic.

Current status: `PROVISIONAL; ORACLE_PENDING; LIVE TEST BLOCKED BY B0`.
The latest static oracle rejects PFM-A as a distinct incremental execution
mechanism in the established evidence envelope. Its typed state/accounting
overlay remains reusable as a static specification for existing prefetch and
verification lines, not as an authorized hot path.

Required baselines are sequential out-of-core inference, OS page cache and
readahead, predictive weight/expert prefetch, ordinary speculative decoding,
strong multi-position target verification, SpecExec-style offload-aware
speculation, and RSSO/layer-stationary execution without PFM scheduling.

## 8. PFM-B — trainable track

PFM-B proposes

\[
z_t=G_\phi(X_t),
\]

progressive refinement

\[
z_t^{(k+1)}=R_{\phi,k}(z_t^k,m_k),
\]

and candidate-conditioned reconstruction

\[
\hat X_{t+|c|}=C_\phi(z_t^K,c).
\]

The exact target state remains

\[
X_{t+|c|}^{c}=F_\theta^{|c|}(X_t,c),
\]

with state error

\[
\epsilon_c=d(\hat X_{t+|c|},X_{t+|c|}^{c}).
\]

The decisive practical quantity is exact correction fraction

\[
\rho_c=C_{exact\ correction}/C_{cold\ target\ pass}.
\]

PFM-B is interesting only when `rho_c << 1` for a meaningful, calibrated
fraction of likely futures without material target-quality loss.

The provisional training objective is

\[
L=L_{LM}+\alpha L_{state}+\beta L_{correction}+\gamma L_{calibration}
 +\delta L_{rank}+\eta L_{hardware},
\]

with

\[
L_{LM}=-\sum_t\log p_\theta(y_{t+1}\mid h_t),
\]

\[
L_{state}=d(C_\phi(z_t,c),X_{t+|c|}^{c}),
\quad
L_{correction}=C_{required\ exact\ correction}/C_{full\ target\ computation},
\]

\[
L_{calibration}=(\hat p_{use}-p_{use})^2+(\hat\rho-\rho)^2.
\]

For future-state delta matrix `D=[\Delta X^1,\ldots,\Delta X^B]`,

\[
L_{rank}=\sum_{i>r}\sigma_i(D)^2,
\]

and the provisional hardware term is

\[
L_{hardware}=\lambda_BB+\lambda_EE+\lambda_TT+\lambda_MM.
\]

These are hypotheses and training targets, not validated results. PFM-B status
is `PROVISIONAL; STATIC_ONLY; LONG-TERM CONJECTURE; TRAINING NOT AUTHORIZED`.

## 9. Gate and kill contract

PFM-0 oracle viability requires at least 10% exact committed-throughput gain
with no meaningful energy regression, or 10% lower joules per exact committed
token with no meaningful throughput regression.

PFM-1 requires unused preparation to increase critical-path latency by no more
than 2%. PFM-2 requires

\[
p_{used\ after\ promotion}C_{saved}>
C_{promotion}+C_{wasted\ promotion}.
\]

PFM-3 requires median `rho<0.5` to be minimally interesting and `<0.25` as a
strong target. PFM-4 requires superiority to predictive prefetch, ordinary
speculation and strong multi-position verification.

Kill or severely scope PFM-A if perfect-information oracle superiority fails,
contention exceeds the safety bound, width two is almost never positive,
promotion misses dominate, KV/activation cost dominates, residual correction
approaches a full pass, ordinary verification captures the reuse, prediction
cost consumes the saving, queueing destroys overlap or exact provenance cannot
be maintained.

Kill PFM-B if future deltas are not compressible, low-rank structure is not
stable, correction remains near one full pass, quality degrades, use/residual
calibration fails, or it loses to MTP, latent-token and span-generation
baselines.

## 10. Relationships and authority boundary

PFM does not absorb or rename adjacent ideas:

- PHASE proposes possible future outcomes;
- TBEH selects a useful speculative horizon;
- PFM controls progressive preparation and promotion;
- REMORA Flow schedules resources;
- RSSO performs physical LayerPack reuse;
- Persistent Expert Atlas supplies stable expert identity;
- REMORA Verify protects exact authority;
- Causal-Closure Cache governs legal reuse;
- ENSM versions state and artifact namespaces;
- REMORA Reclaim salvages independently reusable rejected work;
- Waste Ledger measures useful and wasted preparation.

PFM-A may feed or schedule these lines, but it is not a replacement for them.

## 11. Complete variable and measurement dictionary

The equations above are contracts only when every symbol has a declared unit,
measurement or estimation method, update rule and decision boundary. The
following dictionary is normative for a future static checker and deliberately
labels unmeasured quantities as `UNKNOWN` rather than assigning zero.

| Symbol or field | Units / domain | Measurement or estimation method | Update rule | Decision boundary |
|---|---|---|---|---|
| `t` | integer token position | authoritative committed-token counter | increment only on commit | position must belong to the current epoch |
| `y_i`, `\hat y_i` | token ID | tokenizer/model output under the declared sampler | append only to the candidate or committed path that produced it | candidate IDs never certify state equality |
| `h_t` | ordered token-ID sequence | hash of the committed prefix plus position range | replace only at an authoritative commit | prefix hash mismatch invalidates P1 |
| `p_\theta` | probability distribution | frozen target logits and declared sampling semantics | recompute at each authoritative boundary | strict PFM must equal the target distribution |
| `F_\theta` | exact state transition | target implementation/model/runtime identity | version with `M_t` and `E_t` | identity mismatch blocks reuse |
| `X_t` | typed exact-state record | digest/structured certificate for all six state components | advance only after target commit | incomplete state is `UNKNOWN`, not exact |
| `K_t` | exact KV bytes/tensors | KV generation, tensor metadata and digest | increment on valid target update | wrong generation invalidates P1 |
| `R_t` | architecture-specific state bytes/tensors | recurrent-state generation and digest | update with the target transition | recurrent mismatch invalidates P1 |
| `G_t` | sampler/RNG state | sampler configuration, seed/stream and RNG epoch | advance with sampling | sampler mismatch blocks authoritative reuse |
| `M_t` | identity tuple | model, runtime, build, backend and kernel hashes | change on any artifact/build mutation | any identity change invalidates artifacts |
| `E_t` | execution epoch/namespace | monotonic epoch and namespace registry | bump on reset, rollback or authority change | cross-epoch artifacts cannot commit |
| `c` | ordered candidate-token sequence | candidate generator/draft trace | replace on a new branch | only a matching committed prefix permits P1 reuse |
| `a` | packet record | schema validation and provenance hash | append event-sourced lifecycle records | missing packet fields block promotion |
| `I_a` | unique identifier | cryptographic/random ID with collision check | immutable after creation | duplicate ID is a hard failure |
| `D_a` | dependency/provenance set | transitive dependency hash and leaf register | recompute after any dependency event | exact match is required for P1 |
| `H_a` | exact committed tokens until use | held-out predictor or exact trace | update after each authoritative token | unknown horizon blocks promotion |
| `\tau_a` | logical/physical tier label | placement event and tier inventory | update at each transfer | tier label is not permanently tied to a device |
| `f_a` | categorical `P0`, `P1` or `P2` | packet-class validator | immutable for the artifact | P0/P2 cannot become authority without exact work/certificate |
| `p_a`, `s_a` | calibrated probabilities in `[0,1]` | held-out use/staleness events | versioned rolling calibration window | uncalibrated values are `UNKNOWN` |
| `\rho_a` | dimensionless fraction | paired cold and remaining exact-cost ledger | update only from matched traces | reject outside `[0,1]`; near one is non-interesting |
| `R_a` | bytes, bytes/s, CPU/GPU units, memory bytes | per-tier telemetry or explicit oracle inputs | refresh on packet/tier change | unknown demand blocks selection |
| `C_a` | seconds, joules or one declared scalar cost unit | timed/resource/energy ledger | recompute after every dependency or queue event | cost terms cannot be silently omitted |
| `\Gamma_a` | expiry timestamps, epochs and predicates | validity/expiry checker | close on deadline, reset or certificate failure | expired artifacts cannot promote |
| `T_{available}` | seconds | `H_a/r_t` using a conservative `r_t` | recompute per authoritative token | only positive slack can be promoted |
| `r_t` | exact committed tokens/s | monotonic wall clock around the fixed output contract | conservative rolling lower bound | zero/unknown rate blocks promotion |
| `T_{ready}` | seconds | timed create/refine/queue/promote path | conservative upper bound from matched traces | if it exceeds availability, cancel/downgrade |
| `\sigma_a` | seconds | `T_available-T_ready` | recompute on horizon/rate/queue changes | `\sigma_a>0` is necessary, not sufficient |
| `C_{cold}`, `C_{remain}` | declared cost units | matched no-preparation and prepared exact paths | update only with same model/output contract | `C_{cold}>0`; otherwise `\rho` is undefined |
| `G_a` | declared cost units | `C_{cold}-C_{remain}` | recompute with residual ledger | no gross saving claim without a physical witness |
| `C_{prep\_total}` and each term | declared cost units | create/refine/promote/hold/verify/contention/queue/disposal ledger | charge every observed event; unknown remains unknown | omitted cost blocks positive-value promotion |
| `V(a)` | declared cost units | full expected-value expression | recompute when any input changes | create/retain/promote only if `V(a)>0` and `\sigma_a>0` |
| `Capacity_j`, `CriticalLoad_j`, `SafetyReserve_j` | resource-specific bytes, bytes/s, compute or memory units | hardware phenotype/telemetry or explicit oracle input | refresh per workload and epoch | unknown capacity is not background budget |
| `B_{background,j}` | resource-specific units | max-capacity subtraction | recompute at each critical-load change | selected demand must not exceed it |
| `x_a` | binary decision | controller decision log | update at scheduling boundary | child cannot be selected without parent |
| `W_l` | physical LayerPack bytes | source-range manifest and measured exposed transfer | immutable for a pack identity | logical bytes do not substitute for `W_l` |
| `\beta_{W,l}` | physical bytes/s | paired transfer measurement at the relevant tier | update by phenotype/queue regime | unknown bandwidth blocks break-even |
| `F_l` | seconds | `W_l/\beta_{W,l}` | recompute with bytes/bandwidth | movement amortization must be positive |
| `K` | integer positions | target-batch/verification contract | sweep only pre-registered widths | exact state/rollback must cover every position |
| `C_l` | seconds or declared per-position cost | matched layer compute/KV/activation timing | update from same trace | compute terms cannot be omitted |
| `O_l(K)` | seconds or declared overhead units | scheduling, KV, activation, sync, branch and rollback ledger | charge per width | require `(K-1)F_l>O_l(K)` |
| `z_t`, `z_t^k` | representation-dependent tensor/bytes | encoder/refinement output on held-out exact states | update during training/refinement | representation is advisory until exact correction |
| `G_\phi`, `R_{\phi,k}`, `C_\phi` | trainable maps/parameters | training checkpoint and code hash | version with the training run | no current training authorization |
| `m_k` | refinement metadata | packet/refinement trace | update at each stage | missing metadata prevents reconstruction claims |
| `\widehat X`, `X^c` | exact-state tensor/bytes | candidate reconstruction versus teacher-forced target | compare on held-out futures | approximate state cannot commit |
| `\epsilon_c` | declared state-distance units | explicit metric `d` on matched state tensors | recompute per candidate/depth | error alone is insufficient without correction cost |
| `\rho_c` | dimensionless exact-correction fraction | exact correction cost divided by cold target pass | summarize distribution by held-out split | median `<0.5` minimally interesting; near one kills value |
| `L_*` | normalized training-loss units | declared loss implementation and normalization | update per training batch/epoch | provisional objective only |
| `\alpha,\beta,\gamma,\delta,\eta` | dimensionless loss weights | pre-registered hyperparameter configuration | immutable within a comparison | tuning cannot use held-out gate data |
| `D`, `\Delta X^b`, `\sigma_i(D)`, `r` | state units, singular-value units, integer rank | exact future-state tensor batch and SVD | recompute per depth/horizon/split | rank claim requires residual and stability evidence |
| `B,E,T,M` in `L_{hardware}` | bytes, joules, seconds, memory/opportunity units | hardware ledger with declared normalization | update per matched run | hardware loss is not evidence of a speed result |
| `\lambda_B,\lambda_E,\lambda_T,\lambda_M` | inverse normalized units | pre-registered operating-mode weights | fixed per comparison | primary bytes/time/energy metrics remain separately reported |

### Decision-boundary convention

A variable with no source, units, calibration, or update rule is not a zero;
it is `UNKNOWN`. Unknown values block promotion, exact authority and positive
value claims. Simulated or hypothetical values are tagged in the manifest and
may only support sensitivity analysis or an upper-bound oracle.
