//! Morpheus-rs core: everything the bot binary needs that is not `main`.
//!
//! The Rust sibling of `bots/morpheus/`, developed clean-room against the
//! Python bot as a frozen decision oracle (docs/bots/morpheus-rs/rewrite-plan.md).
//! At M0.5 this is a walking skeleton: the wire protocol and nothing else, so
//! the submission path can be proven before any real porting starts.
//!
//! Single-threaded by construction. One dedicated core is a competition
//! constraint, not a tuning choice, so nothing here may spawn a thread.

pub mod wire;
