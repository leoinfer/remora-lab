//! Continuous-batching decode scheduler with prefix reuse.
//!
//! This serving layer runs **one batched model
//! invocation across all live sequences**, so weight bytes are read once
//! per batch instead of once per sequence — a common high-throughput serving
//! mechanism. On top of that, every consumed token is
//! registered in a [`har_kv::PrefixGraph`] radix tree with its hidden
//! state (the scheduler's "KV residency" abstraction); a new request whose
//! prompt shares a prefix with anything already served skips that prefill
//! work entirely and resumes from the cached state — the prefix-reuse
//! effect can be observed and counted by the scheduler's telemetry.
//!
//! Correctness contract: the scheduler is a *pure orchestration layer*.
//! Given the same [`BatchStepModel`] and the same prompts, the token
//! streams it produces are identical whether sequences run one at a time,
//! interleaved in a batch, or after a prefix-matching warm-up.  Tests
//! enforce this differentially.

use crate::adapter::{BatchStepModel, Hidden, Logits};
use crate::telemetry::{depth_bucket, PrefixTelemetry};
use har_kv::{
    KVPageId, KVPageRecord, PrefixGraph, PrefixGraphError, PrefixIdentity, PrefixNodeId,
    StableDigest,
};
use std::collections::{HashMap, VecDeque};

/// Stable identity roots for the scheduler's prefix graph.  A real runtime
/// binds these to the actual model/tokenizer/rope fingerprints; the serving
/// layer keeps them fixed per `ServeConfig`.
#[derive(Clone, Debug)]
pub struct IdentityRoots {
    pub model: String,
    pub tokenizer: String,
    pub rope: String,
    pub authority: String,
}

impl Default for IdentityRoots {
    fn default() -> Self {
        Self {
            model: "har-serve-model".into(),
            tokenizer: "har-serve-tokenizer".into(),
            rope: "har-serve-rope".into(),
            authority: "har-serve-authority".into(),
        }
    }
}

/// Eviction policy for the residency pool.
///
/// `Lru` (default) evicts the least-recently-used state first.  `TrunkFirst`
/// (H-06 value-weighted, runtime form) evicts the deepest unpinned leaves
/// first, keeping the shallow shared prefix — the part of the radix tree
/// that future requests reuse (their reuse potential is proportional to
/// the subtree they serve, which is highest near the root).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum EvictionPolicy {
    #[default]
    Lru,
    TrunkFirst,
}

/// Scheduler configuration.
#[derive(Clone, Debug)]
pub struct ServeConfig {
    /// Maximum number of sequences processed per step (the batch width).
    pub max_batch: usize,
    /// KV page granularity: a page record is attached to the prefix graph
    /// every `page_size` consumed tokens.
    pub page_size: usize,
    /// Layer span recorded on attached KV pages (0..kv_layers).
    pub kv_layers: u32,
    /// Head span recorded on attached KV pages (0..kv_heads).
    pub kv_heads: u32,
    /// Identity roots for the prefix graph.
    pub identity: IdentityRoots,
    /// KV codec tag recorded on pages.
    pub kv_type: String,
    /// Physical KV residency budget (bytes): cached per-node hidden states
    /// beyond this are evicted LRU.  `None` = unbounded (no eviction).
    /// This is the paged-KV pool the prefix graph's logical records map
    /// onto — the vLLM/SGLang "KV pool" counterpart in the serving layer.
    pub max_cache_bytes: Option<u64>,
    /// GDN state-pool bound (bytes): the *recurrent* state reservation per
    /// running request (SGLang cookbook: on hybrid-GDN models this bounds
    /// concurrency before the KV pool does).  Admission refuses requests
    /// beyond `max_live_state_bytes` (they stay queued).
    pub max_live_state_bytes: Option<u64>,
    /// Pin the state of every node on a live sequence's path so eviction
    /// never loses the reuse a running request still enables.
    pub pin_live_nodes: bool,
    /// Chunked prefill: how many prompt tokens a sequence consumes per
    /// step in one model call.  1 = token-at-a-time (default, exact
    /// current behavior).  Larger chunks amortize prefill as GEMM and —
    /// per the SGLang cookbook (2048 on hybrid-GDN models) — stop long
    /// prompts from stalling decode: decode sequences still run every
    /// step, between prefill chunks.
    pub prefill_chunk: usize,
    /// Residency eviction policy (H-06 trunk-first value weighting).
    pub eviction_policy: EvictionPolicy,
}

impl Default for ServeConfig {
    fn default() -> Self {
        Self {
            max_batch: 4,
            page_size: 8,
            kv_layers: 8,
            kv_heads: 4,
            identity: IdentityRoots::default(),
            kv_type: "q8_0".into(),
            max_cache_bytes: None,
            max_live_state_bytes: None,
            pin_live_nodes: true,
            prefill_chunk: 1,
            eviction_policy: EvictionPolicy::Lru,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SequenceId(pub u64);

impl std::fmt::Display for SequenceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "seq-{:04}", self.0)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SequenceState {
    Running,
    Finished,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ServeError {
    EmptyPrompt,
    UnknownSequence,
}

impl std::fmt::Display for ServeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyPrompt => write!(f, "prompt must contain at least one token"),
            Self::UnknownSequence => write!(f, "unknown sequence id"),
        }
    }
}
impl std::error::Error for ServeError {}

/// One admitted sequence.
#[derive(Clone, Debug)]
pub struct Sequence {
    pub id: SequenceId,
    /// Full requested token stream (prompt followed by generated tokens as
    /// they are sampled).
    stream: Vec<u32>,
    /// Number of `stream` tokens whose hidden state is already computed
    /// (either by this scheduler or inherited from a matched prefix).
    consumed: usize,
    /// Tokens inherited from a matched prefix at admission (0 = cold).
    pub prefix_depth: usize,
    /// Maximum generated tokens; the sequence finishes when reached.
    pub max_new: usize,
    pub state: SequenceState,
    /// Current hidden state (position `consumed`).
    hidden: Hidden,
    /// Cached logits predicting position `consumed` (Some only when a
    /// prefix match supplied them without a model step).
    pending_logits: Option<Logits>,
    /// Current prefix-graph node (depth == `consumed`).
    node: Option<PrefixNodeId>,
    /// Model invocations spent on this sequence (accounting only).
    pub work_rows: u64,
    /// Generated token count sampled so far.
    pub generated: usize,
}

impl Sequence {
    /// Tokens of the original prompt (everything before generated).
    fn prompt(&self) -> &[u32] {
        &self.stream[..self.stream.len() - self.generated]
    }
    fn prompt_len(&self) -> usize {
        self.prompt().len()
    }
    pub fn stream(&self) -> &[u32] {
        &self.stream
    }
    pub fn finished(&self) -> bool {
        self.state == SequenceState::Finished
    }
}

/// Classification of one sequence's work in a step.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Phase {
    /// Consuming a prompt token.
    Prefill,
    /// Consuming a previously generated token.
    Decode,
}

/// Per-step outcome, one entry per active sequence (in sequence order).
#[derive(Clone, Debug)]
pub struct SequenceStepOutcome {
    pub id: SequenceId,
    pub phase: Phase,
    pub token_consumed: u32,
    /// How many tokens this sequence consumed this step (1 for decode;
    /// `prefill_chunk` for a full prefill chunk).
    pub tokens_consumed: usize,
    pub token_sampled: Option<u32>,
    pub finished: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StepKind {
    /// At least one sequence did prefill work.
    Prefill,
    /// Only decode work happened.
    Decode,
    /// No sequences were live.
    Idle,
}

/// Result of one scheduler step.
#[derive(Clone, Debug)]
pub struct StepReport {
    pub step_index: u64,
    pub kind: StepKind,
    /// Sequences processed this step.
    pub active: usize,
    /// Model invocation rows (== active; the batched call).
    pub rows: usize,
    pub prefill_rows: usize,
    pub decode_rows: usize,
    /// Tokens sampled this step (across all sequences).
    pub tokens_sampled: usize,
    /// Sequences that finished this step.
    pub finished: usize,
    /// Weight bytes the backend reads for this step's batch — the
    /// bandwidth model: a batched kernel invocation touches the weight set
    /// once (rows share the read), so this is one per-row weight set, not
    /// rows × per-row.  Divided by `tokens_sampled` this is the honest
    /// amortized bytes/token figure continuous batching buys.
    pub weight_bytes: u64,
    /// Prefix-graph nodes (tracked by the scheduler).
    pub graph_nodes: usize,
    /// KV pages attached so far.
    pub kv_pages: usize,
    /// Physical KV residency bytes currently cached.
    pub cache_bytes: u64,
    /// Cumulative LRU evictions from the residency pool.
    pub evictions: u64,
    pub outcomes: Vec<SequenceStepOutcome>,
}

impl StepReport {
    pub fn weight_bytes_per_token(&self) -> f64 {
        if self.tokens_sampled == 0 {
            return 0.0;
        }
        self.weight_bytes as f64 / self.tokens_sampled as f64
    }
}

/// Greedy argmax with lowest-index tie break (deterministic).
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0u32;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i as u32;
        }
    }
    best
}

/// Physical residency entry for one prefix node: the hidden state at that
/// depth plus the logits predicting the next position (when the node's
/// full stream was consumed).  This is the "KV page in VRAM" abstraction —
/// bounded by `ServeConfig::max_cache_bytes` and evicted LRU.
#[derive(Clone, Debug)]
struct CachedState {
    hidden: Hidden,
    logits: Option<Logits>,
    /// Step index of the last read or write (LRU clock).
    last_access: u64,
    /// Number of live sequences whose path pins this node.
    pinned: u32,
}

impl CachedState {
    fn bytes(&self) -> u64 {
        let hidden = self.hidden.len() * 4;
        let logits = self.logits.as_ref().map_or(0, |l| l.len() * 4);
        (hidden + logits) as u64
    }
}

/// The continuous-batching scheduler.
pub struct ServeScheduler<M: BatchStepModel> {
    cfg: ServeConfig,
    model: M,
    graph: PrefixGraph,
    root: PrefixNodeId,
    /// Physical KV residency: hidden state + predicting logits at each
    /// graph node (the scheduler's "KV resident in VRAM" abstraction).
    /// The graph is the append-only *logical* record; this map is the
    /// *physical* pool — bounded and evictable.
    state_cache: HashMap<PrefixNodeId, CachedState>,
    sequences: Vec<Sequence>,
    queue: VecDeque<Sequence>,
    next_id: u64,
    step_index: u64,
    graph_nodes: usize,
    kv_pages: usize,
    cache_bytes: u64,
    evictions: u64,
    telemetry: PrefixTelemetry,
}

impl<M: BatchStepModel> ServeScheduler<M> {
    pub fn new(model: M, cfg: ServeConfig) -> Self {
        let mut graph = PrefixGraph::new();
        let root_identity = identity_for(&cfg, &[]);
        let root = graph.insert_root(root_identity).expect("fresh graph root");
        Self {
            cfg,
            model,
            graph,
            root: root.id.clone(),
            state_cache: HashMap::new(),
            sequences: Vec::new(),
            queue: VecDeque::new(),
            next_id: 0,
            step_index: 0,
            graph_nodes: 1,
            kv_pages: 0,
            cache_bytes: 0,
            evictions: 0,
            telemetry: PrefixTelemetry::default(),
        }
    }

    /// Number of sequences currently admitted and running.
    pub fn live(&self) -> usize {
        self.sequences.iter().filter(|s| !s.finished()).count()
    }
    pub fn queued(&self) -> usize {
        self.queue.len()
    }
    pub fn graph_nodes(&self) -> usize {
        self.graph_nodes
    }
    pub fn kv_pages(&self) -> usize {
        self.kv_pages
    }
    pub fn cache_bytes(&self) -> u64 {
        self.cache_bytes
    }
    pub fn evictions(&self) -> u64 {
        self.evictions
    }
    /// Prefix reuse / eviction telemetry (read-only).
    pub fn telemetry(&self) -> &PrefixTelemetry {
        &self.telemetry
    }

    /// The underlying model (e.g. to read speculation telemetry from a
    /// `SpeculativeModel`).
    pub fn model(&self) -> &M {
        &self.model
    }
    /// Mutable model access (auto-calibration hot-swaps the policy).
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }
    pub fn step_index(&self) -> u64 {
        self.step_index
    }
    pub fn sequence(&self, id: SequenceId) -> Result<&Sequence, ServeError> {
        self.sequences
            .iter()
            .find(|s| s.id == id)
            .ok_or(ServeError::UnknownSequence)
    }
    pub fn stream_of(&self, id: SequenceId) -> Result<Vec<u32>, ServeError> {
        Ok(self.sequence(id)?.stream().to_vec())
    }
    pub fn work_rows_of(&self, id: SequenceId) -> Result<u64, ServeError> {
        Ok(self.sequence(id)?.work_rows)
    }
    pub fn graph(&self) -> &PrefixGraph {
        &self.graph
    }

    /// Probe: longest prefix of `prompt` currently resident in the graph.
    /// Returns (node id, depth).  Tests use this to assert reuse depth.
    pub fn longest_prefix_probe(&self, prompt: &[u32]) -> (PrefixNodeId, usize) {
        self.longest_prefix(prompt)
    }

    /// Queue a request.  The prompt is prefilled (minus any matched prefix)
    /// and then up to `max_new` tokens are generated greedily.
    pub fn submit(&mut self, prompt: &[u32], max_new: usize) -> Result<SequenceId, ServeError> {
        if prompt.is_empty() {
            return Err(ServeError::EmptyPrompt);
        }
        let id = SequenceId(self.next_id);
        self.next_id += 1;
        self.queue.push_back(Sequence {
            id,
            stream: prompt.to_vec(),
            consumed: 0,
            prefix_depth: 0,
            max_new,
            state: SequenceState::Running,
            hidden: self.model.initial_hidden(),
            pending_logits: None,
            node: None,
            work_rows: 0,
            generated: 0,
        });
        Ok(id)
    }

    /// Match the longest prefix of `prompt` that is BOTH in the graph AND
    /// physically resident: the walk stops at the first node whose state
    /// was evicted, so the matched node's state is always loadable (the
    /// deepest resident node, not the deepest logical node).  This is
    /// what makes trunk-first eviction meaningful: with the shallow
    /// shared prefix kept, later requests resume from it instead of
    /// falling all the way back to cold.
    fn longest_prefix(&self, prompt: &[u32]) -> (PrefixNodeId, usize) {
        let mut node = self.root.clone();
        let mut depth = 0usize;
        for token in prompt {
            let next = self
                .graph
                .node(&node)
                .and_then(|n| n.children.get(token).cloned());
            match next {
                Some(next) => {
                    if self.state_cache.contains_key(&next) {
                        node = next;
                        depth += 1;
                    } else {
                        break; // logical edge exists, physical state gone
                    }
                }
                None => break,
            }
        }
        (node, depth)
    }

    /// Admit queued sequences while the batch has room.  Prefix-matched
    /// requests resume from the cached hidden state instead of re-prefilling.
    ///
    /// Two admission gates, both config-gated:
    /// - `max_batch` (batch width) and
    /// - `max_live_state_bytes` — the GDN state-pool bound (SGLang
    ///   cookbook): each running request reserves its recurrent state, so
    ///   this can bound concurrency before the KV pool does.  Requests
    ///   beyond the bound stay queued.
    ///
    /// A cache miss at the matched node (state evicted) falls back to a
    /// full cold prefill from the root — the honest "not resident" path.
    fn admit(&mut self) {
        let hidden_bytes = self.model.initial_hidden().len() * 4;
        loop {
            let running = self.sequences.iter().filter(|s| !s.finished()).count();
            if running >= self.cfg.max_batch {
                break;
            }
            if let Some(budget) = self.cfg.max_live_state_bytes {
                let reserved = (running + 1) as u64 * hidden_bytes as u64;
                if reserved > budget {
                    break;
                }
            }
            let Some(mut seq) = self.queue.pop_front() else {
                break;
            };
            let (node, depth) = self.longest_prefix(seq.prompt());
            let resident = self.state_cache.contains_key(&node);
            self.telemetry.admissions += 1;
            if resident {
                self.telemetry.cache_hits += 1;
                self.telemetry.hit_depth_total += depth as u64;
                self.telemetry.hit_depth_buckets[depth_bucket(depth)] += 1;
                self.telemetry.reuse_rows_saved += depth as u64;
                if depth == seq.prompt_len() {
                    self.telemetry.full_prompt_hits += 1;
                }
                seq.prefix_depth = depth;
                seq.consumed = depth;
                seq.node = Some(node.clone());
                let state = self.state_cache.get_mut(&node).expect("checked");
                state.last_access = self.step_index;
                seq.hidden = state.hidden.clone();
                // Cached logits may only seed the first sample when the
                // WHOLE prompt is resident; with a partial match they
                // predict a mid-prompt position and must not be used.
                seq.pending_logits = (depth == seq.prompt_len())
                    .then(|| state.logits.clone())
                    .flatten();
                if self.cfg.pin_live_nodes {
                    self.pin_path(&node);
                }
            } else {
                // Cache miss: the logical prefix exists but its physical
                // state was evicted — resume cold from the root.
                self.telemetry.cache_misses += 1;
                seq.prefix_depth = 0;
                seq.consumed = 0;
                seq.node = Some(self.root.clone());
                seq.hidden = self.model.initial_hidden();
                seq.pending_logits = None;
            }
            self.sequences.push(seq);
        }
    }

    /// Pin every cached state on the path `node → root` (walk via parent
    /// links) so eviction skips them while a live sequence holds them.
    /// Called once per sequence at admission; balanced by the single
    /// `unpin_path` at finish (extensions pin only the new node, so every
    /// node on a live path is pinned exactly once per sequence).
    fn pin_path(&mut self, node: &PrefixNodeId) {
        let mut current = Some(node.clone());
        while let Some(id) = current {
            if let Some(state) = self.state_cache.get_mut(&id) {
                state.pinned += 1;
                state.last_access = self.step_index;
            }
            current = self.graph.node(&id).and_then(|n| n.parent.clone());
        }
    }

    /// Pin exactly one node (a sequence's current node after an extend).
    fn pin_node(&mut self, node: &PrefixNodeId) {
        if let Some(state) = self.state_cache.get_mut(node) {
            state.pinned += 1;
            state.last_access = self.step_index;
        }
    }

    /// Release one pin on every cached state on the path `node → root`.
    /// A sequence calls this exactly once, when it finishes.
    fn unpin_path(&mut self, node: &PrefixNodeId) {
        let mut current = Some(node.clone());
        while let Some(id) = current {
            if let Some(state) = self.state_cache.get_mut(&id) {
                state.pinned = state.pinned.saturating_sub(1);
            }
            current = self.graph.node(&id).and_then(|n| n.parent.clone());
        }
    }

    /// Evict LRU unpinned states until the cache fits the budget.  The
    /// logical graph is untouched — eviction only drops physical state.
    fn evict_to_budget(&mut self) {
        let Some(budget) = self.cfg.max_cache_bytes else {
            return;
        };
        while self.cache_bytes > budget {
            let victim = self
                .state_cache
                .iter()
                .filter(|(_, s)| s.pinned == 0)
                .min_by_key(|(id, s)| match self.cfg.eviction_policy {
                    // LRU: evict the least-recently-used first.
                    EvictionPolicy::Lru => (s.last_access, 0u64),
                    // Trunk-first: evict the deepest leaf first (min depth
                    // with max age), keeping the shared shallow prefix.
                    EvictionPolicy::TrunkFirst => {
                        let depth = self.graph.node(id).map(|n| n.depth as u64).unwrap_or(0);
                        (u64::MAX - depth, u64::MAX - s.last_access)
                    }
                })
                .map(|(id, s)| (id.clone(), s.last_access));
            let Some((victim, age_at_eviction)) = victim else {
                // Everything is pinned: nothing evictable, pool over
                // budget — the two pin-pressure signals.
                self.telemetry.pinned_evictions_skipped += 1;
                self.telemetry.over_budget_steps += 1;
                break;
            };
            let removed = self.state_cache.remove(&victim).expect("victim present");
            self.cache_bytes = self.cache_bytes.saturating_sub(removed.bytes());
            self.evictions += 1;
            self.telemetry.evictions += 1;
            self.telemetry.eviction_age_total += self.step_index.saturating_sub(age_at_eviction);
        }
    }

    /// Run one continuous-batch step and return its report.
    pub fn step(&mut self) -> StepReport {
        self.step_index += 1;
        self.admit();

        let mut report = StepReport {
            step_index: self.step_index,
            kind: StepKind::Idle,
            active: 0,
            rows: 0,
            prefill_rows: 0,
            decode_rows: 0,
            tokens_sampled: 0,
            finished: 0,
            weight_bytes: 0,
            graph_nodes: self.graph_nodes,
            kv_pages: self.kv_pages,
            cache_bytes: self.cache_bytes,
            evictions: self.evictions,
            outcomes: Vec::new(),
        };

        // Admission-time sampling: a request whose whole prompt is already
        // resident samples its first generated token from the cached logits
        // with zero model invocations.
        let mut tokens_sampled_at_admission = 0usize;
        let mut finished_at_sampling: Vec<usize> = Vec::new();
        for (i, seq) in self
            .sequences
            .iter_mut()
            .enumerate()
            .filter(|(_, s)| !s.finished())
        {
            if seq.consumed == seq.prompt_len() && seq.pending_logits.is_some() {
                if seq.max_new > 0 {
                    let logits = seq.pending_logits.take().expect("checked above");
                    let sampled = sample_greedy(self.model.eos(), seq, &logits);
                    if sampled.is_some() {
                        tokens_sampled_at_admission += 1;
                    }
                } else {
                    seq.state = SequenceState::Finished;
                    finished_at_sampling.push(i);
                }
            }
        }
        report.tokens_sampled = tokens_sampled_at_admission;

        let running_indices: Vec<usize> = (0..self.sequences.len())
            .filter(|&i| !self.sequences[i].finished())
            .collect();
        if running_indices.is_empty() {
            // Nothing ran this step: release pins sampled-finish took
            // (max_new == 0) and sweep the pool before reporting idle.
            for &i in &finished_at_sampling {
                let node = self.sequences[i].node.clone().expect("node");
                self.unpin_path(&node);
            }
            self.evict_to_budget();
            report.cache_bytes = self.cache_bytes;
            report.evictions = self.evictions;
            return report;
        }

        // Partition running sequences into prefill (chunked) and decode.
        // Prefill and decode are different kernel shapes (GEMM vs GEMV) —
        // they run as separate model calls per step, exactly as in
        // vLLM/SGLang — and decode always runs every step, between
        // prefill chunks, so long prompts cannot stall generation.
        let chunk = self.cfg.prefill_chunk.max(1);
        let mut prefill_indices: Vec<usize> = Vec::new();
        let mut decode_indices: Vec<usize> = Vec::new();
        for &i in &running_indices {
            let seq = &self.sequences[i];
            if seq.consumed < seq.prompt_len() {
                prefill_indices.push(i);
            } else {
                decode_indices.push(i);
            }
        }

        let prefill_inputs: Vec<(Hidden, Vec<u32>)> = prefill_indices
            .iter()
            .map(|&i| {
                let seq = &self.sequences[i];
                let start = seq.consumed;
                let end = (start + chunk).min(seq.prompt_len());
                (seq.hidden.clone(), seq.prompt()[start..end].to_vec())
            })
            .collect();
        let prefill_outputs = self.model.prefill_batch(&prefill_inputs);

        let decode_inputs: Vec<(Hidden, u32)> = decode_indices
            .iter()
            .map(|&i| {
                let seq = &self.sequences[i];
                (seq.hidden.clone(), seq.stream[seq.stream.len() - 1])
            })
            .collect();
        let decode_outputs = if decode_inputs.is_empty() {
            Vec::new()
        } else {
            self.model.batch_step(&decode_inputs)
        };

        let rows = prefill_inputs.iter().map(|(_, t)| t.len()).sum::<usize>() + decode_inputs.len();
        report.active = running_indices.len();
        report.rows = rows;
        if !prefill_indices.is_empty() {
            report.weight_bytes += self.model.weight_bytes_per_row();
        }
        if !decode_indices.is_empty() {
            report.weight_bytes += self.model.weight_bytes_per_row();
        }
        report.kind = if !prefill_indices.is_empty() {
            StepKind::Prefill
        } else {
            StepKind::Decode
        };

        let mut finished_this_step: Vec<usize> = Vec::new();

        // --- Chunked prefill: consume up to `chunk` tokens per step. ---
        for (k, &i) in prefill_indices.iter().enumerate() {
            let (tokens, parent0, identities, base_consumed) = {
                let seq = &self.sequences[i];
                let consumed = seq.consumed;
                let n = prefill_inputs[k].1.len();
                let identities: Vec<PrefixIdentity> = (1..=n)
                    .map(|j| identity_for(&self.cfg, &seq.stream[..consumed + j]))
                    .collect();
                (
                    prefill_inputs[k].1.clone(),
                    seq.node.clone().expect("admitted sequence has a node"),
                    identities,
                    consumed,
                )
            };
            let (states, logits) = &prefill_outputs[k];
            let n = tokens.len();
            report.prefill_rows += n;
            let final_logits = logits.last().expect("non-empty chunk");

            // Extend the graph one edge per consumed token, recording the
            // per-token hidden trajectory (mid-chunk reuse stays exact).
            let mut node = parent0;
            let mut consumed = base_consumed;
            for (j, &t) in tokens.iter().enumerate() {
                consumed += 1;
                let (new_node, created) =
                    extend_or_follow(&mut self.graph, &node, t, identities[j].clone(), consumed);
                if created {
                    self.graph_nodes += 1;
                    let state = CachedState {
                        hidden: states[j].clone(),
                        logits: if j + 1 == n {
                            Some(final_logits.clone())
                        } else {
                            None
                        },
                        last_access: self.step_index,
                        pinned: 0,
                    };
                    self.cache_bytes += state.bytes();
                    self.state_cache.insert(new_node.clone(), state);
                }
                attach_page_if_boundary(
                    &mut self.graph,
                    &self.cfg,
                    &identities[j],
                    &new_node,
                    consumed,
                    &mut self.kv_pages,
                );
                if self.cfg.pin_live_nodes {
                    self.pin_node(&new_node);
                }
                node = new_node;
            }

            let (sampled_here, finished_here) = {
                let seq = &mut self.sequences[i];
                seq.hidden = states.last().expect("non-empty chunk").clone();
                seq.work_rows += n as u64;
                seq.node = Some(node.clone());
                seq.consumed = consumed;
                let mut sampled_here = None;
                let mut finished_here = false;
                if consumed == seq.stream.len() {
                    if seq.generated < seq.max_new {
                        sampled_here = sample_greedy(self.model.eos(), seq, final_logits);
                    } else {
                        seq.state = SequenceState::Finished;
                        finished_here = true;
                    }
                }
                (sampled_here, finished_here)
            };
            if sampled_here.is_some() {
                report.tokens_sampled += 1;
            }
            if finished_here {
                report.finished += 1;
                finished_this_step.push(i);
            }
            report.outcomes.push(SequenceStepOutcome {
                id: self.sequences[i].id,
                phase: Phase::Prefill,
                token_consumed: *tokens.last().expect("non-empty chunk"),
                tokens_consumed: n,
                token_sampled: sampled_here,
                finished: self.sequences[i].finished(),
            });
        }

        // --- Decode: one model step per sequence (GEMV path). ---
        //
        // A step may consume more than one token: the accepted
        // speculative drafts enter the stream (each recorded in the
        // prefix graph with the packed state at its position), then the
        // sampled token.  Plain decode is the drafts.len() == 0 case.
        for (k, &i) in decode_indices.iter().enumerate() {
            let token_in = decode_inputs[k].1;
            report.decode_rows += 1;
            let out = &decode_outputs[k];
            let eos = self.model.eos();

            let (_new_node, token_sampled_here, finished_here) = {
                let seq = &mut self.sequences[i];
                seq.hidden = out.next.clone();
                seq.work_rows += 1;

                let mut node = seq.node.clone().expect("admitted sequence has a node");
                let mut consumed = seq.consumed;

                // 1. Input token edge (state after consuming it).
                consumed += 1;
                let identity = identity_for(&self.cfg, &seq.stream[..consumed]);
                let (node2, created) =
                    extend_or_follow(&mut self.graph, &node, token_in, identity.clone(), consumed);
                if created {
                    self.graph_nodes += 1;
                    let state = CachedState {
                        hidden: out.consumed_state.clone(),
                        logits: Some(out.logits.clone()),
                        last_access: self.step_index,
                        pinned: 0,
                    };
                    self.cache_bytes += state.bytes();
                    self.state_cache.insert(node2.clone(), state);
                }
                attach_page_if_boundary(
                    &mut self.graph,
                    &self.cfg,
                    &identity,
                    &node2,
                    consumed,
                    &mut self.kv_pages,
                );
                if self.cfg.pin_live_nodes {
                    if let Some(state) = self.state_cache.get_mut(&node2) {
                        state.pinned += 1;
                        state.last_access = self.step_index;
                    }
                }
                node = node2;

                // 2. Accepted drafts: each becomes a stream token with
                //    its own graph edge and per-position state.  An EOS
                //    draft ends the sequence (EOS itself is not pushed,
                //    matching plain sampling).
                let mut finished = false;
                for (j, &dt) in out.drafts.iter().enumerate() {
                    if finished || dt == eos || seq.generated >= seq.max_new {
                        finished = true;
                        break;
                    }
                    seq.stream.push(dt);
                    seq.generated += 1;
                    consumed += 1;
                    let identity = identity_for(&self.cfg, &seq.stream[..consumed]);
                    let (node3, created) =
                        extend_or_follow(&mut self.graph, &node, dt, identity.clone(), consumed);
                    if created {
                        self.graph_nodes += 1;
                        let state = CachedState {
                            hidden: out.draft_states[j].clone(),
                            logits: None,
                            last_access: self.step_index,
                            pinned: 0,
                        };
                        self.cache_bytes += state.bytes();
                        self.state_cache.insert(node3.clone(), state);
                    }
                    attach_page_if_boundary(
                        &mut self.graph,
                        &self.cfg,
                        &identity,
                        &node3,
                        consumed,
                        &mut self.kv_pages,
                    );
                    if self.cfg.pin_live_nodes {
                        if let Some(state) = self.state_cache.get_mut(&node3) {
                            state.pinned += 1;
                            state.last_access = self.step_index;
                        }
                    }
                    node = node3;
                }

                seq.node = Some(node.clone());
                seq.consumed = consumed;

                // 3. Sample when the consumed tokens caught up with the
                //    stream (the sampled token is the pending one; its
                //    edge is created on the next step).
                let mut sampled_here = None;
                let mut finished_here = false;
                if finished || seq.generated >= seq.max_new {
                    seq.state = SequenceState::Finished;
                    finished_here = true;
                } else if consumed == seq.stream.len() {
                    sampled_here = sample_greedy(eos, seq, &out.logits);
                    if sampled_here.is_none() {
                        finished_here = true;
                    }
                }
                seq.consumed = consumed;
                (node, sampled_here, finished_here)
            };

            if token_sampled_here.is_some() {
                report.tokens_sampled += 1;
            }
            if finished_here {
                report.finished += 1;
                finished_this_step.push(i);
            }
            report.outcomes.push(SequenceStepOutcome {
                id: self.sequences[i].id,
                phase: Phase::Decode,
                token_consumed: token_in,
                tokens_consumed: 1,
                token_sampled: token_sampled_here,
                finished: self.sequences[i].finished(),
            });
        }

        // Release pins held by sequences that finished this step, then
        // sweep the residency pool back under budget.
        for &i in finished_this_step.iter().chain(finished_at_sampling.iter()) {
            let node = self.sequences[i].node.clone().expect("node");
            self.unpin_path(&node);
        }
        self.evict_to_budget();

        report.graph_nodes = self.graph_nodes;
        report.kv_pages = self.kv_pages;
        report.cache_bytes = self.cache_bytes;
        report.evictions = self.evictions;
        report
    }

    /// Run steps until the queue and all sequences drain; returns every
    /// non-idle report.  Primarily a test convenience.
    pub fn run_to_idle(&mut self) -> Vec<StepReport> {
        let mut reports = Vec::new();
        loop {
            let report = self.step();
            if report.kind == StepKind::Idle {
                return reports;
            }
            reports.push(report);
        }
    }

    /// Aggregate generated streams of all sequences (deterministic order).
    pub fn all_streams(&self) -> Vec<(SequenceId, Vec<u32>)> {
        self.sequences
            .iter()
            .map(|s| (s.id, s.stream().to_vec()))
            .collect()
    }
}

/// Extend the prefix graph with one consumed token; on a duplicate edge
/// (shared prefix already stored) follow the existing child.  Returns the
/// node id and whether a new node was created.
fn extend_or_follow(
    graph: &mut PrefixGraph,
    parent: &PrefixNodeId,
    token: u32,
    identity: PrefixIdentity,
    consumed: usize,
) -> (PrefixNodeId, bool) {
    match graph.extend(parent, token, identity) {
        Ok((node_id, _edge)) => (node_id, true),
        Err(PrefixGraphError::DuplicateEdge) => {
            let child = graph
                .node(parent)
                .expect("parent exists")
                .children
                .get(&token)
                .expect("duplicate edge implies child exists")
                .clone();
            (child, false)
        }
        Err(e) => panic!("prefix graph extend failed at depth {consumed}: {e}"),
    }
}

/// Sample greedily into the sequence; returns the sampled token, or None
/// when EOS terminated the stream (EOS itself is not pushed).
fn sample_greedy(eos: u32, seq: &mut Sequence, logits: &[f32]) -> Option<u32> {
    let token = argmax(logits);
    seq.generated += 1;
    if token == eos {
        seq.state = SequenceState::Finished;
        None
    } else {
        seq.stream.push(token);
        Some(token)
    }
}

fn identity_for(cfg: &ServeConfig, tokens: &[u32]) -> PrefixIdentity {
    PrefixIdentity {
        model_root: StableDigest::from_text(&cfg.identity.model),
        tokenizer_root: StableDigest::from_text(&cfg.identity.tokenizer),
        token_sequence: tokens.to_vec(),
        rope_config_root: StableDigest::from_text(&cfg.identity.rope),
        layer_start: 0,
        layer_end: cfg.kv_layers,
        head_start: 0,
        head_end: cfg.kv_heads,
        kv_type: cfg.kv_type.clone(),
        codec_version: "har-serve-v0".into(),
        runtime_generation: 1,
        authority_state_root: StableDigest::from_text(&cfg.identity.authority),
    }
}

/// Attach a KV page record every `page_size` consumed tokens.
fn attach_page_if_boundary(
    graph: &mut PrefixGraph,
    cfg: &ServeConfig,
    identity: &PrefixIdentity,
    node: &PrefixNodeId,
    consumed: usize,
    kv_pages: &mut usize,
) {
    let ps = cfg.page_size;
    if consumed % ps != 0 || consumed < ps {
        return;
    }
    let closure = identity.closure(graph.generation(), 0);
    let page = KVPageRecord::new(
        KVPageId {
            prefix_node_id: node.clone(),
            ordinal: (consumed / ps) as u32 - 1,
            token_start: (consumed - ps) as u64,
            token_end: consumed as u64,
            layer_start: 0,
            layer_end: cfg.kv_layers,
            head_start: 0,
            head_end: cfg.kv_heads,
            logical_generation: 1,
        },
        closure,
    );
    graph
        .attach_page(page)
        .unwrap_or_else(|e| panic!("KV page attach failed at depth {consumed}: {e}"));
    *kv_pages += 1;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seq_with(stream: Vec<u32>, max_new: usize) -> Sequence {
        Sequence {
            id: SequenceId(0),
            stream,
            consumed: 0,
            prefix_depth: 0,
            max_new,
            state: SequenceState::Running,
            hidden: Vec::new(),
            pending_logits: None,
            node: None,
            work_rows: 0,
            generated: 0,
        }
    }

    #[test]
    fn argmax_picks_highest_with_lowest_index_tiebreak() {
        assert_eq!(argmax(&[1.0, 5.0, 3.0]), 1);
        assert_eq!(argmax(&[4.0, 4.0, 4.0]), 0);
        assert_eq!(argmax(&[-1.0, -0.5, -2.0]), 1);
    }

    #[test]
    fn sample_greedy_pushes_token_and_counts_generation() {
        let mut seq = seq_with(vec![1, 2], 4);
        let sampled = sample_greedy(9, &mut seq, &[0.0, 0.0, 10.0]);
        assert_eq!(sampled, Some(2));
        assert_eq!(seq.generated, 1);
        assert_eq!(seq.stream(), &[1, 2, 2]);
        assert!(!seq.finished());
    }

    #[test]
    fn sample_greedy_eos_terminates_without_pushing() {
        let mut seq = seq_with(vec![1, 2], 4);
        let sampled = sample_greedy(
            9,
            &mut seq,
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 99.0],
        );
        assert_eq!(sampled, None);
        assert_eq!(seq.generated, 1);
        assert_eq!(seq.stream(), &[1, 2], "EOS is not pushed");
        assert!(seq.finished());
    }
}
