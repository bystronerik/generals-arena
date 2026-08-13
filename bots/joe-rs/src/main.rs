//! `joe-rs` — the greedy policy of the frozen joe network, in Rust.
//!
//! Wire mode (no arguments): the competition stdio seat. Per turn: parse the
//! frame, rebuild the 14-channel raw tensor, build cost + masks, augment to
//! 39 channels with the persistent state, one float32 forward pass, greedy
//! argmax — exactly what the Python sibling's `agent.py` does. No search, no
//! sampling; joe is deterministic at play time.
//!
//! `joe-rs parity <surface>` runs one ported surface over fixture cases
//! (see `parity.rs`); `pytest` in `tests/` drives it against the Python
//! oracle's recordings.
//!
//! Two behaviours inherited from the morpheus-rs seat:
//! * **A panic becomes a pass** — the judge forfeits a crash but charges a
//!   bad reply as one fault out of fifty, so the decision runs inside
//!   `catch_unwind`.
//! * **EOF is a clean exit** — the engine ends a game by closing stdin.

mod action;
mod net;
mod obs;
mod parity;
mod wire;
mod xla_math;

use std::io::{self, BufWriter};
use std::panic::{self, AssertUnwindSafe};
use std::path::PathBuf;
use std::time::Instant;

use action::{argmax, decode_action};
use net::Net;
use obs::{
    augment_obs, build_cost_from_raw, compute_build_mask_from_raw, compute_valid_move_mask,
    frame_to_raw, normalize_observations, prepare_action_mask, AugScratch, AugState, CELLS,
    N_ACTION_CHANNELS, N_CHANNELS, PAD, TEMPORAL_WINDOW,
};
use wire::{read_handshake, read_observation, write_action, Action, Observation, PASS};

/// Where `model.safetensors` + `manifest.json` live. `run.sh` exports
/// `JOE_RS_ARTIFACT`; the exe-relative fallback covers running the binary by
/// hand from `target/release/`.
pub(crate) fn artifact_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("JOE_RS_ARTIFACT") {
        return PathBuf::from(dir);
    }
    if let Ok(exe) = std::env::current_exe() {
        // target/release/joe-rs -> <bot dir>/artifact
        if let Some(bot_dir) = exe.parent().and_then(|p| p.parent()).and_then(|p| p.parent()) {
            let candidate = bot_dir.join("artifact");
            if candidate.is_dir() {
                return candidate;
            }
        }
    }
    PathBuf::from("artifact")
}

/// The playing seat: network plus the per-game state and preallocated
/// buffers. Load and warmup happen in `new`, inside the first-move grace.
struct Seat {
    net: Net,
    h: usize,
    w: usize,
    state: AugState,
    next_state: AugState,
    scratch: AugScratch,
    raw: Vec<f32>,
    cost: Vec<i32>,
    move_mask: Vec<bool>,
    build_mask: Vec<bool>,
    aug: Vec<f32>,
    penalties: Vec<f32>,
    temporal: Vec<f32>,
}

impl Seat {
    fn new(h: usize, w: usize) -> Result<Self, String> {
        if h > PAD || w > PAD {
            return Err(format!("board {h}x{w} exceeds the net's pad_to {PAD}"));
        }
        let load = Instant::now();
        let net = Net::load(&artifact_dir())?;
        let load_ms = load.elapsed().as_secs_f64() * 1e3;

        let seat = Self {
            net,
            h,
            w,
            state: AugState::zeros(),
            next_state: AugState::zeros(),
            scratch: AugScratch::new(),
            raw: Vec::new(),
            cost: Vec::new(),
            move_mask: Vec::new(),
            build_mask: Vec::new(),
            aug: vec![0.0; N_CHANNELS * CELLS],
            penalties: vec![0.0; N_ACTION_CHANNELS * CELLS],
            temporal: vec![0.0; 2 * TEMPORAL_WINDOW],
        };

        // Warmup: one forward on zeros primes the allocator and page cache.
        // The forward is stateless, so nothing to reset afterwards.
        let warm = Instant::now();
        seat.net
            .forward(&seat.aug, &seat.penalties, &seat.temporal)
            .map_err(|e| format!("warmup forward: {e}"))?;
        let warmup_ms = warm.elapsed().as_secs_f64() * 1e3;
        eprintln!("[joe-rs] load {load_ms:.1} ms, warmup {warmup_ms:.1} ms");
        Ok(seat)
    }

    /// The full per-move path. Mirrors `agent.py::step` + the pass clamp in
    /// `agent.py::act`.
    fn act(&mut self, obs: &Observation) -> Result<Action, String> {
        frame_to_raw(obs, &mut self.raw);
        build_cost_from_raw(&self.raw, self.h, self.w, &mut self.cost);
        compute_valid_move_mask(&self.raw, self.h, self.w, &mut self.move_mask);
        compute_build_mask_from_raw(&self.raw, self.h, self.w, &self.cost, &mut self.build_mask);
        augment_obs(
            &self.raw,
            self.h,
            self.w,
            &self.cost,
            &self.state,
            &mut self.next_state,
            &mut self.scratch,
            &mut self.aug,
        );
        std::mem::swap(&mut self.state, &mut self.next_state);
        self.temporal[..TEMPORAL_WINDOW].copy_from_slice(&self.state.opponent_army_history);
        self.temporal[TEMPORAL_WINDOW..].copy_from_slice(&self.state.opponent_land_history);
        normalize_observations(&mut self.aug);
        prepare_action_mask(
            &self.move_mask,
            &self.build_mask,
            self.h,
            self.w,
            &mut self.penalties,
        );
        let fwd = self.net.forward(&self.aug, &self.penalties, &self.temporal)?;
        let a = decode_action(argmax(&fwd.logits));
        // Free eval telemetry (port-plan §5): v1 play ignores the value.
        eprintln!("[joe-rs] turn {} value {:+.4}", obs.turn, fwd.value);
        if a.pass_field == 1 {
            // The pass channel's argmax cell can sit in the pad region; the
            // engine ignores row/col on a pass, but keep the reply in bounds.
            return Ok(PASS);
        }
        Ok(Action {
            pass: a.pass_field as u8,
            row: a.row as u16,
            col: a.col as u16,
            dir: a.dir as u8,
            split: a.is_half as u8,
        })
    }
}

fn wire_main() -> Result<(), String> {
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let stdout = io::stdout();
    let mut writer = BufWriter::new(stdout.lock());

    let handshake = match read_handshake(&mut reader).map_err(|e| e.to_string())? {
        Some(hs) => hs,
        None => return Ok(()), // engine went away before the game started
    };

    // `JOE_RS_PASS=1` keeps the J1 walking skeleton reachable: same wire
    // loop, pass every turn, no artifact needed.
    let pass_only = std::env::var("JOE_RS_PASS").is_ok_and(|v| v == "1");
    let mut seat = if pass_only {
        None
    } else {
        Some(Seat::new(handshake.h, handshake.w)?)
    };

    let mut obs = Observation::with_dims(handshake.h, handshake.w);
    let mut line = String::new();
    let mut scratch = Vec::new();
    loop {
        match read_observation(&mut reader, &mut obs, &mut line, &mut scratch) {
            Ok(true) => {}
            Ok(false) => return Ok(()), // EOF: game over
            Err(e) => return Err(e.to_string()),
        }
        let action = match seat.as_mut() {
            None => PASS,
            Some(seat) => {
                match panic::catch_unwind(AssertUnwindSafe(|| seat.act(&obs))) {
                    Ok(Ok(action)) => action,
                    Ok(Err(e)) => {
                        eprintln!("[joe-rs] turn {} error: {e}; passing", obs.turn);
                        PASS
                    }
                    Err(_) => {
                        eprintln!("[joe-rs] turn {} panicked; passing", obs.turn);
                        PASS
                    }
                }
            }
        };
        write_action(&mut writer, action).map_err(|e| e.to_string())?;
    }
}

/// `joe-rs bench < game.in.log`: replay a recorded wire log through the full
/// per-move path — frame parse, obs pipeline, forward pass, reply encode —
/// and report per-turn latency percentiles (port-plan §5's budget check;
/// the reply goes to a sink instead of the engine).
fn bench_main() -> Result<(), String> {
    use std::io::Read;

    let mut text = String::new();
    io::stdin().read_to_string(&mut text).map_err(|e| e.to_string())?;
    let mut reader = std::io::Cursor::new(text.as_bytes());

    let handshake = read_handshake(&mut reader)
        .map_err(|e| e.to_string())?
        .ok_or("empty input")?;
    let load = Instant::now();
    let mut seat = Seat::new(handshake.h, handshake.w)?;
    let startup_ms = load.elapsed().as_secs_f64() * 1e3;

    let mut obs = Observation::with_dims(handshake.h, handshake.w);
    let mut line = String::new();
    let mut scratch = Vec::new();
    let mut sink = io::sink();
    let mut times_ms: Vec<f64> = Vec::new();
    loop {
        let t0 = Instant::now();
        match read_observation(&mut reader, &mut obs, &mut line, &mut scratch) {
            Ok(true) => {}
            Ok(false) => break,
            Err(e) => return Err(e.to_string()),
        }
        let action = seat.act(&obs)?;
        write_action(&mut sink, action).map_err(|e| e.to_string())?;
        times_ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }

    times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f64| times_ms[((times_ms.len() as f64 - 1.0) * p) as usize];
    println!(
        "turns {} startup_ms {startup_ms:.1} p50 {:.2} p90 {:.2} p99 {:.2} max {:.2}",
        times_ms.len(),
        pct(0.50),
        pct(0.90),
        pct(0.99),
        pct(1.0),
    );
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let result = match args.get(1).map(String::as_str) {
        None => wire_main(),
        Some("bench") => bench_main(),
        Some("parity") => match args.get(2) {
            Some(surface) => parity::run(surface),
            None => Err("usage: joe-rs parity <surface>".into()),
        },
        Some(other) => Err(format!("unknown subcommand {other:?}")),
    };
    if let Err(e) = result {
        eprintln!("[joe-rs] fatal: {e}");
        std::process::exit(1);
    }
}
