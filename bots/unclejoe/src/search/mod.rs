//! Exact search over the rules: a forward model, a prover on top of it, and
//! the one path that runs the model backwards.
//!
//! The layer sits between `board` and `tactics` and names neither the network
//! nor the tactics that drive it.
//!
//! [`sim`] is the transition function, transcribed from the engine. [`minimax`]
//! is the AND/OR search that plays both sides of it to a fixed horizon and
//! returns a move only when the goal survives every reply — its only two
//! answers are a move and silence. [`afterstate`] is the other direction: it
//! advances one candidate and renders the position back into the frame joe's
//! pipeline reads, so something above this layer can ask the network what that
//! position is worth.
//!
//! Everything a proof needs that a frame does not state — the hidden-army
//! bound, the range window, the deathtouch threshold — arrives as an argument.
//! Those are rules the tactics layer already owns one implementation of, and
//! one rule read two ways is the failure that layer exists to prevent; so this
//! module consumes them and never re-derives them.

pub mod afterstate;
pub mod minimax;
pub mod sim;

#[cfg(test)]
pub mod fixtures;
