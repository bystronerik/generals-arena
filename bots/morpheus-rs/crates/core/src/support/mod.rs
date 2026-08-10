//! Primitives the crate implements itself because it has no dependencies.
//!
//! Both of these exist only because the dependency budget declined `rand` and
//! `sha2` (docs/bots/morpheus-rs/rewrite-plan.md §1). Keeping them in one
//! directory makes that cost one visible line in the tree rather than two
//! files scattered among the game layers.
//!
//! `rng` is not a general-purpose generator: it reproduces NumPy's PCG64 and
//! `Generator` bit-for-bit, because the belief layer's draws are compared
//! against the Python oracle's.

pub mod rng;
pub mod sha256;
