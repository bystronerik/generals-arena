//! What is left of the network layer after joe's net replaced morpheus's.
//!
//! Everything that *is* the network — the forward pass, the GEMM kernels under
//! it, the safetensors container, the observation pipeline that feeds it —
//! moved to the `joenet` crate at N1, where it is a byte-identical copy of
//! `bots/joe-rs/src/` and may not be edited (joe-net-plan §8.2). Deleted
//! outright with morpheus's own 249,316-parameter CNN: `network.rs` (eleven
//! heads over a 49-plane input), `tensor.rs` (that input contract and the
//! `BeliefSummary` feeding its eight belief planes), and `inference.rs` (the
//! manifest guardrails, whose job `joenet::nn::net::Net::load` now does).
//!
//! What stays is [`head`]: the arithmetic *between* the network and the
//! search, which is action-space logic rather than learned weights and which
//! §7.2 keeps unchanged across the port.
//!
//! This directory therefore no longer holds any float the parity harness has a
//! tolerance for. The two non-zero tolerances that used to live here went with
//! the Torch oracle; joe's forward is proved transitively instead, by
//! byte-identity against a corpus joe-rs already carries (§8.2).

pub mod head;
