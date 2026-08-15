//! Play mask, prior shaping, hard rules, and the four planners.
//!
//! Port of `bots/morpheus/tactics.py` — 2,283 lines and the largest single
//! surface in the rewrite, with no speed story to motivate it (rewrite-plan §7
//! calls it "pure translation, little gain, highest port effort per line").
//! It is here because the decision depends on it and the shipped binary cannot
//! call Python.
//!
//! The rules are the Python's; every constant in [`params`] carries the
//! measurement or the field-observed loss that set it, copied from the source
//! it was written in. Nothing here is retuned — M5 ports behaviour, M7 tunes.
//!
//! ## The directory
//!
//! This file is a module list and its re-exports, so `crate::tactics::<item>`
//! resolves for every rule wherever it now lives.
//!
//! The five surfaces the parity harness already separates each get a file:
//! [`play_mask`], [`candidates`], [`shaping`], [`constrain`], and the four
//! planners the `planners` surface covers — [`seek`], [`castle`], [`defense`]
//! and [`kill`]. Those are four files rather than a `planners/` directory
//! because each is a named subsystem in the rewrite plan and a third level of
//! nesting buys nothing at ~140 lines apiece.
//!
//! The rest is what they share: [`params`] is the tuning surface, [`geometry`]
//! the action decoding, [`pathing`] the distance fields, [`query`] the
//! read-only board questions, [`weights`] the scalar terms, and
//! [`oscillation`] the two-cell shuffle rule.

pub mod candidates;
pub mod castle;
pub mod constrain;
pub mod defense;
pub mod geometry;
pub mod kill;
pub mod oscillation;
pub mod params;
pub mod pathing;
pub mod play_mask;
pub mod query;
pub mod seek;
pub mod shaping;
pub mod weights;

pub use candidates::*;
pub use castle::*;
pub use constrain::*;
pub use defense::*;
pub use geometry::*;
pub use kill::*;
pub use oscillation::*;
pub use params::*;
pub use pathing::*;
pub use play_mask::*;
pub use query::*;
pub use seek::*;
pub use shaping::*;
pub use weights::*;
