//! Exact search over the rules: a forward model, and a prover on top of it.
//!
//! The layer sits between `board` and `tactics` and names neither the network
//! nor the tactics that drive it. It answers one kind of question — *can this
//! be forced?* — and its only two answers are a move and silence.
//!
//! [`sim`] is the transition function, transcribed from the engine. [`minimax`]
//! is the AND/OR search that plays both sides of it to a fixed horizon and
//! returns a move only when the goal survives every reply.
//!
//! Everything a proof needs that a frame does not state — the hidden-army
//! bound, the range window, the deathtouch threshold — arrives as an argument.
//! Those are rules the tactics layer already owns one implementation of, and
//! one rule read two ways is the failure that layer exists to prevent; so this
//! module consumes them and never re-derives them.

pub mod minimax;
pub mod sim;

#[cfg(test)]
pub mod fixtures;
