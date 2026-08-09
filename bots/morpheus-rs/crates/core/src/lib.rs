//! Morpheus-rs core: everything the bot binary needs that is not `main`.
//!
//! The Rust sibling of `bots/morpheus/`, developed clean-room against the
//! Python bot as a frozen decision oracle (docs/bots/morpheus-rs/rewrite-plan.md).
//! M1 lands the deterministic board layer: the wire protocol, the board state,
//! the exact competition transition, the action codec and legal masks, and
//! fogged observation emission. Everything here is checked bit-for-bit against
//! the Python oracle over the recorded corpus (§5, tier 1).
//!
//! Single-threaded by construction. One dedicated core is a competition
//! constraint, not a tuning choice, so nothing here may spawn a thread.

pub mod action;
pub mod belief;
pub mod deployment;
pub mod evaluator;
pub mod gemm;
pub mod hashing;
pub mod inference;
pub mod json;
pub mod matrix;
pub mod memory;
pub mod network;
pub mod observe;
pub mod parity;
pub mod particle_summary;
pub mod proposal;
pub mod recovery;
pub mod reservoir;
pub mod rng;
pub mod runtime;
pub mod safetensors;
pub mod search;
pub mod sha256;
pub mod state;
pub mod symmetry;
pub mod tactics;
pub mod tensor;
pub mod transition;
pub mod tree;
pub mod wire;
