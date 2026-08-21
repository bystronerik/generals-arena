//! The two things the process reads from outside itself.
//!
//! `wire` is the judge's stdio protocol, agent side; `json` is the format the
//! artifact manifest and the safetensors header are written in. Both are
//! format codecs with no model and no game knowledge — nothing here knows
//! what a general is or what a channel means.
//!
//! Neither is a parity surface: the corpus feeds `parity` an integer stream,
//! not wire text, and `tests/test_wire_replay.py` covers `wire` against a
//! recorded game instead. Both are ported from morpheus-rs, which wrote them
//! for the same two inputs.

pub mod json;
pub mod wire;
