//! HAR serving layer: continuous-batching decode scheduler with prefix
//! reuse.
//!
//! What this is: the orchestration that a from-scratch RDNA4 runtime needs
//! to reduce end-to-end token time. Two mechanisms are central here:
//!
//! 1. **Continuous batching** — every decode step runs one batched model
//!    invocation across all live sequences, so each weight byte read serves
//!    the whole batch instead of one sequence.
//! 2. **Prefix reuse** — every consumed token is registered in a
//!    [`har_kv::PrefixGraph`] radix tree together with its hidden state;
//!    a request whose prompt shares a prefix with previously served work
//!    skips that prefill entirely.
//!
//! The scheduler is backend-agnostic through [`adapter::BatchStepModel`]
//! (native Vulkan kernels, native CPU models, or the deterministic
//! [`toy::ToyModel`] used by tests). Its correctness
//! contract is differential: batched, interleaved, and prefix-resumed runs
//! must produce the exact same token streams as isolated runs.

pub mod adapter;
pub mod attention;
pub mod dense;
pub mod gguf;
pub mod http;
pub mod moe;
pub mod q40;
pub mod q4k;
pub mod scheduler;
pub mod server;
pub mod speculation;
pub mod telemetry;
pub mod tokenizer;
pub mod toy;
#[cfg(feature = "vulkan")]
pub mod vulkan;

pub use adapter::{BatchStepModel, Hidden, Logits, StepOutcome};
pub use attention::{FlashDecodeParams, HEAD_DIM, KV_ROW_BYTES, Q8_BLOCK_BYTES};
pub use dense::{dense_kv_len, DenseConfig, DenseModel};
pub use moe::{MoEConfig, MoEModel, MoETelemetry};
pub use q40::{Q40Model, Q40_BLOCKS_PER_ROW, Q40_BLOCK_BYTES, Q40_BLOCK_VALUES};
pub use q4k::{Q4KModel, Q4K_BLOCK_BYTES, Q4K_BLOCK_VALUES};
pub use scheduler::{
    argmax, EvictionPolicy, IdentityRoots, Phase, Sequence, SequenceId, SequenceState,
    SequenceStepOutcome, ServeConfig, ServeError, ServeScheduler, StepKind, StepReport,
};
pub use server::{parse_args, BackendKind, ServerConfig, SpecType};
pub use speculation::{tier_cap, SpecConfig, SpecTelemetry, SpeculativeModel};
pub use telemetry::{depth_bucket, PrefixTelemetry};
pub use tokenizer::Tokenizer;
pub use toy::{ToyConfig, ToyModel};
