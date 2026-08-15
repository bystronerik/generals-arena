//! One file per group of parity kinds; `super::run` dispatches into them.
//!
//! Each function reads its own case off the integer stream and appends its
//! answer to `out`. None of them plays.

pub mod belief;
pub mod board;
pub mod net;
pub mod numpy;
pub mod search;
pub mod tactics;
