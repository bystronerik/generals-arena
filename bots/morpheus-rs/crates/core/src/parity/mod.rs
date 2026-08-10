//! Parity subcommands: run one ported surface over recorded cases.
//!
//! The Rust half of the tier-1 harness in rewrite-plan §5. `pytest` under
//! `bots/morpheus-rs/tests/` builds cases from the M0 corpus, runs the Python
//! oracle on them, invokes the binary on the same cases, and compares.
//!
//! **Format: a flat stream of integers, not JSON.** The plan says "canonical
//! JSON", and this is a deliberate, narrow deviation. Rust's standard library
//! has no JSON, so the choice was a dependency in the *shipped* binary or two
//! hundred lines of hand-rolled parser — for a machine-to-machine channel
//! whose entire payload is integers. The 10,000-file unpacked limit is the
//! binding sandbox constraint (§1), the crate is zero-dependency to protect
//! it, and integers compare bit-exactly with no float formatting to argue
//! about. Reading is the same `parse_ints` the wire protocol already uses.
//!
//! Every layout below is positional and shared with `tests/parity_cases.py`.
//! Lengths are self-describing: each case starts with the dimensions that
//! determine how many integers follow, so a truncated stream fails loudly at
//! the point of truncation instead of silently shifting every later case.
//!
//! ## The directory
//!
//! [`ints`] is the stream reader, [`codec`] every layout the surfaces share,
//! [`bench`] the belief benchmark, and [`surfaces`] one file per group of
//! kinds. This file is `run()`: one line per kind, and the `Ctx` that carries
//! the loaded artifact across a subcommand's cases.

pub mod bench;
pub mod codec;
pub mod ints;
pub mod surfaces;

pub use bench::bench_belief;

use std::io::{BufRead, Write};

use crate::parity::ints::{itoa, Ints};

/// What one subcommand carries across its cases.
///
/// Only `net` and `decide` need the artifact and loading it costs a few
/// milliseconds, so it is filled on first use and then reused — which is what
/// the `Option<Session>` outside the case loop used to do.
pub struct Ctx {
    pub net: Option<crate::nn::inference::Session>,
}

pub fn run<R: BufRead, W: Write>(kind: &str, reader: &mut R, writer: &mut W) -> Result<(), String> {
    let mut ints = Ints::read_all(reader)?;
    let cases = ints.n()?;
    let mut out: Vec<i64> = Vec::new();
    let mut ctx = Ctx { net: None };

    for case in 0..cases {
        out.clear();
        match kind {
            "transition" => surfaces::board::transition(&mut ints, &mut out, &mut ctx)?,
            "order" => surfaces::board::order(&mut ints, &mut out, &mut ctx)?,
            "observe" => surfaces::board::observe(&mut ints, &mut out, &mut ctx)?,
            "mask" => surfaces::board::mask(&mut ints, &mut out, &mut ctx)?,
            "cost" => surfaces::board::cost(&mut ints, &mut out, &mut ctx)?,
            "memory" => surfaces::board::memory(&mut ints, &mut out, &mut ctx)?,
            "hash" => surfaces::board::hash(&mut ints, &mut out, &mut ctx)?,
            "tensor" => surfaces::board::tensor(&mut ints, &mut out, &mut ctx)?,
            "symmetry" => surfaces::board::symmetry(&mut ints, &mut out, &mut ctx)?,
            "net" => surfaces::net::net(&mut ints, &mut out, &mut ctx)?,
            "prior" => surfaces::net::prior(&mut ints, &mut out, &mut ctx)?,
            "toplegal" => surfaces::net::toplegal(&mut ints, &mut out, &mut ctx)?,
            "npsum" => surfaces::numpy::npsum(&mut ints, &mut out, &mut ctx)?,
            "argsort" => surfaces::numpy::argsort(&mut ints, &mut out, &mut ctx)?,
            "initbelief" => surfaces::belief::initbelief(&mut ints, &mut out, &mut ctx)?,
            "summary" => surfaces::belief::summary(&mut ints, &mut out, &mut ctx)?,
            "propose" => surfaces::belief::propose(&mut ints, &mut out, &mut ctx)?,
            "filter" => surfaces::belief::filter(&mut ints, &mut out, &mut ctx)?,
            "rejuvenate" => surfaces::belief::rejuvenate(&mut ints, &mut out, &mut ctx)?,
            "maxent" => surfaces::belief::maxent(&mut ints, &mut out, &mut ctx)?,
            "reservoir" => surfaces::belief::reservoir(&mut ints, &mut out, &mut ctx)?,
            "playmask" => surfaces::tactics::playmask(&mut ints, &mut out, &mut ctx)?,
            "shaping" => surfaces::tactics::shaping(&mut ints, &mut out, &mut ctx)?,
            "candidates" => surfaces::tactics::candidates(&mut ints, &mut out, &mut ctx)?,
            "planners" => surfaces::tactics::planners(&mut ints, &mut out, &mut ctx)?,
            "constrain" => surfaces::tactics::constrain(&mut ints, &mut out, &mut ctx)?,
            "matrix" => surfaces::search::matrix(&mut ints, &mut out, &mut ctx)?,
            "decide" => surfaces::search::decide(&mut ints, &mut out, &mut ctx)?,
            "search" => surfaces::search::search(&mut ints, &mut out, &mut ctx)?,
            "evict" => surfaces::search::evict(&mut ints, &mut out, &mut ctx)?,
            "runtime" => surfaces::search::runtime(&mut ints, &mut out, &mut ctx)?,
            other => return Err(format!("unknown parity kind {other:?}")),
        }

        let mut line = String::with_capacity(out.len() * 4);
        for (i, value) in out.iter().enumerate() {
            if i > 0 {
                line.push(' ');
            }
            line.push_str(itoa(*value).as_str());
        }
        writeln!(writer, "{line}").map_err(|e| format!("case {case}: {e}"))?;
    }
    writer.flush().map_err(|e| format!("flush: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn tiny_state_ints() -> String {
        // 1x2 board, time 0, winner -1, generals at (0,0) and (0,1).
        let mut s = String::from("1 2 0 -1 0 0 0 1 ");
        s.push_str("5 3 "); // armies
        s.push_str("1 0 "); // own0
        s.push_str("0 1 "); // own1
        s.push_str("0 0 "); // neutral
        s.push_str("1 1 "); // generals
        s.push_str("0 0 "); // castles
        s.push_str("0 0 "); // mountains
        s.push_str("1 1 "); // passable
        s
    }

    #[test]
    fn a_transition_case_round_trips_through_the_stream() {
        let input = format!("1 {} 1 0 0 0 0 1 0 0 0 0", tiny_state_ints());
        let mut out = Vec::new();
        run("transition", &mut Cursor::new(input), &mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        let values: Vec<i64> = text
            .split_ascii_whitespace()
            .map(|t| t.parse().unwrap())
            .collect();
        // h w time winner + 4 general coords + 8 planes of 2 cells + 7 info
        assert_eq!(values.len(), 4 + 4 + 8 * 2 + 7);
        assert_eq!(values[0], 1);
        assert_eq!(values[1], 2);
        assert_eq!(values[2], 1, "a double pass still advances the clock");
    }

    #[test]
    fn a_truncated_stream_fails_loudly() {
        let mut out = Vec::new();
        let err = run("transition", &mut Cursor::new("1 1 2 0"), &mut out).unwrap_err();
        assert!(err.contains("ended after"), "{err}");
    }

    #[test]
    fn an_unknown_kind_is_refused() {
        let mut out = Vec::new();
        assert!(run("wat", &mut Cursor::new("1"), &mut out).is_err());
    }
}
