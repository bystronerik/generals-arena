//! What both tactics stand on.
//!
//! Nothing here decides anything, and nothing here belongs to attacking or to
//! defending: [`reach`] answers *how far*, [`frame`] answers what a single
//! frame states about the board and about the army it does not show. Each
//! question has exactly one implementation, here, so the two tactics cannot
//! drift into answering it differently — and the proof search reuses both
//! rather than growing its own (U3).
//!
//! Kept deliberately small. A helper only one tactic needs belongs in that
//! tactic's file, where its reasoning sits beside its use; a helper that
//! encodes a *rule* belongs here, because two callers reading the rule two
//! ways is the bug this module exists to prevent.

pub mod frame;
pub mod reach;

#[cfg(test)]
pub mod fixtures;

pub use frame::hidden_army;
pub use reach::Reach;
