//! Model contract for the serving scheduler.
//!
//! A [`BatchStepModel`] is the narrow seam between the scheduler and any
//! concrete inference backend (native Vulkan kernels, native CPU models, or
//! a toy model for tests). The scheduler only ever asks
//! for one thing: *given the current hidden state and the next token, run
//! one forward step and return the next hidden state plus the logits*.
//!
//! The batched form exists so the backend can amortize weight reads across
//! every sequence in the batch in a single kernel invocation — the
//! continuous-batching mechanism that avoids rereading weights for each
//! sequence. On a bandwidth-bound RDNA4 card this is
//! the difference between reading the weights once per batch and once per
//! sequence per token.

/// Hidden-state vector type (the scheduler-owned "KV" abstraction for the
/// serving layer; a real backend replaces it with paged KV slots).
pub type Hidden = Vec<f32>;

/// Logits row for one sequence (one row of the vocabulary).
pub type Logits = Vec<f32>;

/// One model invocation over a batch of sequences.
///
/// `inputs[i]` is `(hidden, next_token)` for sequence `i`; the result
/// `outputs[i]` describes one step.  The scheduler guarantees call-count
/// semantics: exactly one invocation per sequence per step.
pub trait BatchStepModel {
    fn batch_step(&self, inputs: &[(Hidden, u32)]) -> Vec<StepOutcome>;

    /// Chunked prefill: consume several prompt tokens in one call.
    ///
    /// `inputs[i]` is `(hidden, tokens)`; the result is the per-token
    /// hidden trajectory AND per-token logits (one entry per consumed
    /// token — the KV/state the prefix graph needs for mid-chunk reuse,
    /// and the predictions the speculation verifier needs at every
    /// drafted position).  `logits[i]` predicts position `i+1`.
    ///
    /// The default implementation folds `batch_step` one token at a time
    /// — correct for every model, just not batched.  A real backend
    /// overrides this with the batched prefill (GEMM) path; the
    /// scheduler's differential contract holds either way.
    fn prefill_batch(&self, inputs: &[(Hidden, Vec<u32>)]) -> Vec<(Vec<Hidden>, Vec<Logits>)> {
        inputs
            .iter()
            .map(|(h, tokens)| {
                let mut hidden = h.clone();
                let mut states = Vec::with_capacity(tokens.len());
                let mut logits = Vec::with_capacity(tokens.len());
                for &t in tokens {
                    let out = self.batch_step(&[(hidden, t)]);
                    let out = &out[0];
                    states.push(out.next.clone());
                    logits.push(out.logits.clone());
                    hidden = out.next.clone();
                }
                (states, logits)
            })
            .collect()
    }

    /// Hidden state a fresh sequence starts from (no tokens consumed).
    fn initial_hidden(&self) -> Hidden;

    /// The EOS token id; sequences sampling this token finish.
    fn eos(&self) -> u32;

    /// Weight bytes the backend reads per sequence row per step — the
    /// bandwidth figure the batch amortizes.  Used for honest accounting in
    /// reports and benchmarks (never an official performance claim).
    fn weight_bytes_per_row(&self) -> u64;
}

/// Outcome of one batched step for one sequence.
///
/// The scheduler's step contract:
/// - the sequence's stream grows by `drafts.len() + 1` tokens this step:
///   the accepted drafts (each recorded in the prefix graph with the
///   state `draft_states[j]` at its position), then the token sampled
///   from `logits` (whose state is `next`);
/// - `next` is the hidden state at the position of the sampled token —
///   exactly the state the next step feeds as input with that token;
/// - for a plain decode step (no speculation), `drafts` is empty and
///   `consumed_state == next` (the state after the input token).
#[derive(Clone, Debug)]
pub struct StepOutcome {
    /// State at the position of the token sampled from `logits` (the
    /// next step's input state).
    pub next: Hidden,
    /// Logits predicting the next token (the one the scheduler samples).
    pub logits: Logits,
    /// Accepted speculative drafts consumed this step, in order (empty
    /// for a plain decode step).
    pub drafts: Vec<u32>,
    /// The state at each draft's position: `draft_states[j]` is the state
    /// AFTER consuming `drafts[j]` (the prefix graph records it on the
    /// draft's edge, like any consumed token).
    pub draft_states: Vec<Hidden>,
    /// The state after consuming the INPUT token (recorded on the input
    /// token's edge).  Equals `next` for a plain decode step.
    pub consumed_state: Hidden,
}

impl StepOutcome {
    /// A plain single-token step (no speculation).
    pub fn plain(next: Hidden, logits: Logits) -> Self {
        Self {
            consumed_state: next.clone(),
            next,
            logits,
            drafts: Vec::new(),
            draft_states: Vec::new(),
        }
    }
}

impl BatchStepModel for Box<dyn BatchStepModel> {
    fn batch_step(&self, inputs: &[(Hidden, u32)]) -> Vec<StepOutcome> {
        self.as_ref().batch_step(inputs)
    }
    fn prefill_batch(&self, inputs: &[(Hidden, Vec<u32>)]) -> Vec<(Vec<Hidden>, Vec<Logits>)> {
        self.as_ref().prefill_batch(inputs)
    }
    fn initial_hidden(&self) -> Hidden {
        self.as_ref().initial_hidden()
    }
    fn eos(&self) -> u32 {
        self.as_ref().eos()
    }
    fn weight_bytes_per_row(&self) -> u64 {
        self.as_ref().weight_bytes_per_row()
    }
}
