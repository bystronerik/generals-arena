//! Joe's frozen network, copied here byte-for-byte and never edited.
//!
//! Every file under `src/` except this one is a **byte-identical copy** of the
//! file at the same path in `bots/joe-rs/src/`. That is not tidiness, it is
//! the parity argument: joe-rs already carries a JAX-oracle corpus proving its
//! forward pass matches joe to pinned relative bounds, and
//! `tests/test_joe_source_fanout.py` asserts the copies are equal to the byte.
//! Equal copies plus a green joe-rs corpus means this crate's forward pass is
//! joe's forward pass, without a second corpus
//! (docs/bots/morpheus-rs/joe-net-plan.md §8.2).
//!
//! **So nothing in this crate may be "improved".** A fix belongs in joe-rs
//! first and is then re-copied down to both of joe-rs's downstream forks. An
//! edit made here instead turns the digest test red and names the file.
//!
//! # Why a separate crate rather than files under `crates/core/src/`
//!
//! Byte-identity includes the imports. `board/obs.rs` opens with
//! `use crate::io::wire::{Observation, OWNER_ME, ...}` and joe's `Observation`
//! carries `i32` grids where morpheus's carries `u8`; `nn/net.rs` reaches for
//! `crate::io::json` and `crate::xla_math`. Dropping these files into
//! `morpheus-joe-core` would resolve those paths to morpheus's own modules,
//! which are different types with the same names. A crate boundary gives joe's
//! files joe's `crate::` root and leaves morpheus's alone, at the cost of one
//! widening adapter at the seam — which N2 writes, and which
//! docs/bots/morpheus-rs/joe-net-plan.md §1.6 priced as a widening loop and
//! nothing else.
//!
//! The plan's §8.5 assumed these files would sit under `crates/core/src/`; the
//! mutation-check rows it specifies move to `crates/joenet/src/` instead.
//! Nothing else about §8 changes.
//!
//! # What is here and what is not
//!
//! `board::obs` is the observation pipeline (raw tensor, build cost, masks,
//! the 39-channel augment, normalize), `nn::net` the HistoryTransformer
//! forward pass, `nn::gemm` the kernel under it, `nn::safetensors` and
//! `io::json` the container the weights arrive in, `xla_math` the two
//! transcendentals. `io::wire` and `board::action` come along because
//! `board/mod.rs` and `io/mod.rs` name them and those files are copies too;
//! **morpheus keeps its own wire and its own action codec**, and this crate's
//! copies are not on the play path.

pub mod io;

pub mod board;

pub mod nn;

pub mod xla_math;
