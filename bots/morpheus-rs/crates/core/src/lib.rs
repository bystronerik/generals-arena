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
pub mod memory;
pub mod observe;
pub mod parity;
pub mod state;
pub mod transition;
pub mod wire;
