//! `morpheus-rs` — the stdio seat.
//!
//! M0.5 walking skeleton: read the handshake, read every frame, reply pass.
//! It exists to prove the submission path end to end — offline build, file
//! count, binary compatibility with the judge's sandbox — before any of the
//! real port is written (rewrite-plan §4, M0.5).
//!
//! Two behaviours here are not placeholders and should survive the port:
//!
//! * **A panic becomes a pass.** The judge forfeits a game on a crash or an
//!   early exit, but only charges one fault out of fifty for a bad reply
//!   (RULES.md §08). So the decision runs inside `catch_unwind` and a panicking
//!   turn costs a fault, not the match.
//! * **EOF is a clean exit.** The engine ends a game by closing stdin. Any
//!   other treatment turns a normal finish into a forfeit.

use std::io::{self, BufWriter, Write};
use std::panic::{self, AssertUnwindSafe};

use morpheus_core::wire::{
    read_handshake, read_observation, write_action, Action, Observation, PASS,
};

/// The decision. Pass, for now — M1 gives it legal moves and a policy.
fn decide(_obs: &Observation) -> Action {
    PASS
}

fn main() {
    // Panic messages go to stderr, which the harness captures; the process
    // must not die on one. Keep the default hook so a real bug is still
    // visible in a match log rather than silently becoming a pass.
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let stdout = io::stdout();
    let mut writer = BufWriter::new(stdout.lock());

    let handshake = match read_handshake(&mut reader) {
        Ok(Some(handshake)) => handshake,
        // No handshake at all: the harness spawned us and went away. Nothing
        // was played, so there is nothing to report.
        Ok(None) => return,
        Err(err) => {
            eprintln!("[morpheus-rs] bad handshake: {err}");
            std::process::exit(1);
        }
    };

    let mut obs = Observation::with_dims(handshake.h, handshake.w);
    let mut line = String::new();

    loop {
        match read_observation(&mut reader, &mut obs, &mut line) {
            Ok(true) => {}
            Ok(false) => break, // stdin closed: game over
            Err(err) => {
                // A frame we cannot parse is not a frame we can answer. Pass
                // and keep the seat alive; the alternative is a forfeit.
                eprintln!("[morpheus-rs] {err}");
                let _ = write_action(&mut writer, PASS);
                continue;
            }
        }

        let action = panic::catch_unwind(AssertUnwindSafe(|| decide(&obs))).unwrap_or_else(|_| {
            eprintln!("[morpheus-rs] decide panicked on turn {}; passing", obs.turn);
            PASS
        });

        if write_action(&mut writer, action).is_err() {
            break; // engine hung up while we were replying
        }
    }

    let _ = writer.flush();
}
