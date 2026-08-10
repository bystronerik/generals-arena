//! The deterministic game layer: the competition rules and nothing else.
//!
//! `state` holds the grids, `transition` is the exact competition step,
//! `action` the codec and the legal masks, `observe` the fogged view a seat
//! receives, `memory` what a seat is allowed to remember of what it has seen,
//! `symmetry` the board's dihedral group, and `hashing` the state digest the
//! search and the belief filter key on.
//!
//! This is the layer whose parity tolerance is zero: every surface here is
//! compared bit-for-bit against the Python oracle, there are no floats in the
//! transition, and nothing here knows about a policy, a particle or a clock.
//! It is also upstream of everything else, which is why `mutation_check.py`
//! runs the whole surface set for a mutation in this directory.

pub mod action;
pub mod hashing;
pub mod memory;
pub mod observe;
pub mod state;
pub mod symmetry;
pub mod transition;
