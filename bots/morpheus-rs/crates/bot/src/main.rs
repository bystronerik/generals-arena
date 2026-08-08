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

use morpheus_core::action::{decode_action, legal_mask, PASS_INDEX};
use morpheus_core::memory::VisibleMemory;
use morpheus_core::wire::{
    read_handshake, read_observation, write_action, Action, Observation, PASS,
};

/// The decision: the first legal non-pass move, else pass.
///
/// M1 asks for "legal prior-free moves" — enough to prove the ported mask
/// drives a real game end to end. There is no strategy here on purpose: the
/// prior, the belief, and the search arrive at M3–M5, and a hand-written
/// heuristic in the meantime would be code nobody intends to keep and a
/// tempting thing to compare against.
///
/// The mask is built against an *empty* memory, which makes builds unreachable
/// (they require a cell proven plain) and costs nothing else — memory's update
/// rule lands in M2.
fn decide(obs: &Observation, memory: &VisibleMemory) -> Action {
    let mask = legal_mask(obs, memory, None);
    for index in 0..PASS_INDEX {
        if mask[index] {
            if let Some(action) = decode_action(index) {
                return Action {
                    pass: action[0] as u8,
                    row: action[1] as u16,
                    col: action[2] as u16,
                    dir: action[3] as u8,
                    split: action[4] as u8,
                };
            }
        }
    }
    PASS
}

/// `morpheus-rs parity <kind>`: run one ported surface over recorded cases.
///
/// Lives in the shipped binary rather than a separate harness because the
/// thing under test has to be the thing that plays — a parity build with its
/// own feature flags could pass while the submitted binary diverges.
fn run_parity(kind: &str) -> ! {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut writer = BufWriter::new(stdout.lock());
    match morpheus_core::parity::run(kind, &mut stdin.lock(), &mut writer) {
        Ok(()) => std::process::exit(0),
        Err(err) => {
            eprintln!("[morpheus-rs] parity {kind}: {err}");
            std::process::exit(2);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("parity") {
        match args.get(1) {
            Some(kind) => run_parity(kind),
            None => {
                eprintln!("[morpheus-rs] usage: morpheus-rs parity <transition|observe|mask|cost>");
                std::process::exit(2);
            }
        }
    }

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
    let memory = VisibleMemory::empty(handshake.h, handshake.w);
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

        let action = panic::catch_unwind(AssertUnwindSafe(|| decide(&obs, &memory))).unwrap_or_else(|_| {
            eprintln!("[morpheus-rs] decide panicked on turn {}; passing", obs.turn);
            PASS
        });

        if write_action(&mut writer, action).is_err() {
            break; // engine hung up while we were replying
        }
    }

    let _ = writer.flush();
}
