//! Morpheus-rs core: everything the bot binary needs that is not `main`.
//!
//! The Rust sibling of `bots/morpheus/`, developed clean-room against the
//! Python bot as a frozen decision oracle (docs/bots/morpheus-rs/rewrite-plan.md).
//! Everything here is checked against the Python oracle over the recorded
//! corpus (§5) — bit-for-bit on every surface but two.
//!
//! # Layering
//!
//! The modules form a DAG and are listed below in dependency order: a module
//! may name the ones above it and must not name the ones below it.
//!
//! ```text
//! support  io  ->  board  ->  nn  ->  belief  ->  tactics  ->  search  ->  runtime
//! ```
//!
//! with `parity` above everything, because it is the harness half of the
//! binary and never plays. Nothing enforces this but review — a crate split
//! would, and costs more than it is worth here
//! (docs/bots/morpheus-rs/refactor-plan.md §1).
//!
//! Single-threaded by construction. One dedicated core is a competition
//! constraint, not a tuning choice, so nothing here may spawn a thread.

pub mod io;
pub mod support;

pub mod board;

pub mod nn;

pub mod belief;

pub mod tactics;

pub mod search;

pub mod deployment;
pub mod parity;
pub mod runtime;
pub mod telemetry;
