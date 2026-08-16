//! Frame in, action index out: everything that speaks in board coordinates.
//!
//! `obs` is the port of the joe bot's observation pipeline — raw tensor,
//! build cost, masks, the 39-channel augment, normalize — and `action` is the
//! codec over the 10-channel head that turns a flat logit index back into an
//! engine action.
//!
//! **joe never simulates the board.** There is no `transition` here and there
//! should not be one: the competition engine runs the game and sends a fogged
//! frame per turn, so this layer only ever reads what it is given.
//!
//! This is the layer whose parity tolerance is **zero**. The `raw`, `cost`,
//! `mask` and `obs` surfaces are compared bit-for-bit against the Python/XLA
//! oracle, because everything here is integer logic plus f32 adds, subtracts,
//! multiplies and `RECIP_*` multiplies mirrored site by site in the oracle's
//! op order (port-plan §4). The one transcendental, `log1p` on channel 21,
//! is swept exhaustively rather than assumed — see `crate::xla_math`.

pub mod action;
pub mod obs;
