//! Weights in, masked logits and a value out: everything defined by the
//! trained artifact rather than by the frame.
//!
//! `net` is the HistoryTransformer forward pass, `gemm` the one hot kernel
//! under it, `safetensors` the container the weights arrive in. The loader
//! refuses to start on a schema-tag, name, shape or dtype mismatch, which is
//! the guardrail the Python side gets from `tree_deserialise_leaves`.
//!
//! This is where the floats are, and so where the only non-zero parity
//! tolerances live: the `forward` surface is bounded rather than bit-exact
//! (`tests/test_parity.py` — logits 8.0e-6 rel, bins 1.6e-5 rel, value 5.0e-6
//! abs, worst measured 4.726e-6 / 8.483e-6 / 2.682e-6). The gap is the GEMM's
//! summation order and one `mul_add` per term against the oracle's XLA
//! lowering; op order and float width mirror it site by site everywhere else.

pub mod gemm;
pub mod net;
pub mod safetensors;
