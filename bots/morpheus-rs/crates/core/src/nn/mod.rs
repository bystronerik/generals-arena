//! Weights in, eleven heads out: everything defined by the trained artifact
//! rather than by the rules.
//!
//! `tensor` is the 49-plane input contract and the `BeliefSummary` that feeds
//! it, `network` the forward pass, `gemm` the kernels under it, `safetensors`
//! the container the weights arrive in, and `inference` the session that
//! holds them.
//!
//! Two of the crate's three non-zero parity tolerances live here (`net` at
//! 1e-5, measured 4.05e-6), because this is where the floats are. M3's
//! per-stage timings are measured inside `network.rs` and proved sensitive to
//! edits elsewhere in the crate under whole-crate LTO, which is why that file
//! is not split (docs/bots/morpheus-rs/refactor-plan.md §3).

pub mod gemm;
pub mod inference;
pub mod network;
pub mod safetensors;
pub mod tensor;
