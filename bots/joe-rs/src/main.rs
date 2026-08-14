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
//! Behaviours inherited from the morpheus-rs seat. All of them follow from one
//! asymmetry in RULES.md §08: the judge **forfeits a game on a crash or an
//! early exit**, but charges a late, missing, or malformed reply as one fault
//! out of fifty. So nothing between the handshake and EOF may leave the
//! process.
//! * **A panic becomes a pass** — the decision runs inside `catch_unwind`.
//! * **A seat that will not load becomes a pass** — every turn, with the
//!   reason on stderr. It costs the game, not the match.
//! * **An unparseable frame becomes a pass** — a frame we cannot answer is
//!   still a frame we must reply to.
//! * **EOF is a clean exit** — the engine ends a game by closing stdin, and a
//!   write that fails means it hung up first.
//!
//! Only a malformed *handshake* still exits non-zero: no game had started, so
//! there is nothing to forfeit. `selfcheck` exists because of this policy —
//! once a broken artifact plays on instead of dying, intake is the last moment
//! where failing is cheaper than playing.

mod action;
mod gemm;
mod json;
mod net;
mod obs;
mod parity;
mod safetensors;
mod wire;
mod xla_math;

use std::io::{self, BufWriter, Write};
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
use wire::{
    read_handshake, read_observation, write_action, Action, Observation, PASS, TYPE_FOG,
    TYPE_GENERAL, TYPE_PLAIN,
};

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
    // A seat that will not construct cannot play, but it must not exit: the
    // judge forfeits the whole match on an early exit, where passing every
    // turn costs only this game. `selfcheck` is what catches the same failure
    // at intake, where dying is the cheaper answer.
    let mut seat = if pass_only {
        None
    } else {
        match Seat::new(handshake.h, handshake.w) {
            Ok(seat) => Some(seat),
            Err(e) => {
                eprintln!("[joe-rs] cannot start the seat, passing every turn: {e}");
                None
            }
        }
    };

    let mut obs = Observation::with_dims(handshake.h, handshake.w);
    let mut line = String::new();
    let mut scratch = Vec::new();
    // Every parse error consumes at least one line, so a desynced stream walks
    // to EOF rather than spinning. A *read* error need not consume anything,
    // and a persistent one would spin this loop hot on the single core we are
    // given — so consecutive failures are capped. The cap is above the 50-fault
    // forfeit threshold on purpose: by the time it trips, the game is lost
    // anyway and a clean exit is all that is left to get right.
    let mut consecutive_read_errors = 0;
    loop {
        match read_observation(&mut reader, &mut obs, &mut line, &mut scratch) {
            Ok(true) => consecutive_read_errors = 0,
            Ok(false) => break, // EOF: game over
            Err(e) => {
                // A frame we cannot parse is not a frame we can answer, but it
                // is still a frame we must reply to.
                eprintln!("[joe-rs] {e}; passing");
                consecutive_read_errors += 1;
                if consecutive_read_errors > 64 || write_action(&mut writer, PASS).is_err() {
                    break;
                }
                continue;
            }
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
        if write_action(&mut writer, action).is_err() {
            break; // engine hung up while we were replying
        }
    }
    let _ = writer.flush();
    Ok(())
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

/// One handshake and one frame where passing is the wrong answer: our general
/// sits mid-board on plentiful army with four empty plains around it, and the
/// rest of the board is fog.
///
/// Built as wire text and fed back through the real parser rather than
/// constructed as an `Observation`, so `selfcheck` exercises `read_handshake`
/// and `read_observation` too. It mirrors
/// `tools/package_submission.py::_smoke_frame`; the two must agree on what a
/// healthy seat answers, because the packager smoke-tests with one and
/// `build.sh` runs the other.
fn selfcheck_input() -> String {
    let (h, w, army) = (PAD, PAD, 40);
    let (gr, gc) = (h / 2, w / 2);
    let mut types = vec![vec![TYPE_FOG; w]; h];
    let mut owners = vec![vec![0; w]; h];
    let mut armies = vec![vec![0; w]; h];
    types[gr][gc] = TYPE_GENERAL;
    owners[gr][gc] = 1; // us
    armies[gr][gc] = army;
    for (dr, dc) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
        types[(gr as i32 + dr) as usize][(gc as i32 + dc) as usize] = TYPE_PLAIN;
    }
    let grid = |g: &Vec<Vec<i32>>| {
        g.iter()
            .map(|row| {
                row.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" ")
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    format!(
        "0 {h} {w}\n50 1 {army} 1 1\n{}\n{}\n{}\n",
        grid(&types),
        grid(&owners),
        grid(&armies)
    )
}

/// `joe-rs selfcheck`: prove at intake what the seat will no longer prove at
/// play time.
///
/// Every failure checked here is **silent during a game by design**. A missing
/// artifact, a manifest the loader rejects, a build that lost
/// `target-cpu=x86-64-v3` — none of them stop the process, because RULES.md
/// §08 forfeits the match on an early exit and charges one fault out of fifty
/// for a bad reply. So the seat degrades to passing and nothing downstream can
/// tell a healthy bot from a broken one: both answer well-formed actions.
///
/// Intake is the one moment where failing loudly is the better trade. A
/// rejected submission costs a resubmission; a degraded one costs every rated
/// game it plays, with nothing in a match log to say why. `build.sh` runs this
/// and aborts on a non-zero exit.
fn run_selfcheck() -> ! {
    let mut failures: Vec<String> = Vec::new();
    let say = |key: &str, value: String| println!("{key} {value}");

    // Reported first: it is the one fact that turns a correct binary into a
    // far slower one, and it is invisible on the macOS host that packages the
    // zip. Building from the wrong working directory silently drops the flag,
    // because cargo finds `.cargo/config.toml` by walking up from the cwd.
    // `hardware_fma` is morpheus-rs's name for the same fact, and the shared
    // packager reads that key by name when it builds the smoke note.
    #[cfg(target_arch = "x86_64")]
    {
        say("hardware_fma", cfg!(target_feature = "fma").to_string());
        say("hardware_avx2", cfg!(target_feature = "avx2").to_string());
        if !cfg!(target_feature = "fma") || !cfg!(target_feature = "avx2") {
            failures.push(
                "this x86_64 build has no AVX2/FMA: cargo did not read \
                 .cargo/config.toml (target-cpu=x86-64-v3). Build from the crate \
                 root, as build.sh does."
                    .into(),
            );
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    say("hardware_fma", "n/a (not x86_64)".to_string());

    let dir = artifact_dir();
    say("artifact_dir", dir.display().to_string());
    // Provenance only — the digest is *verified* by the packager against the
    // file it puts in the zip. Reported here so a judge-side log and a row in
    // data/bot_versions/joe-rs.json can be compared without a rebuild.
    match std::fs::read_to_string(dir.join("manifest.json"))
        .map_err(|e| e.to_string())
        .and_then(|t| crate::json::parse(&t))
    {
        Ok(m) => {
            for key in ["safetensors_sha256", "tensor_schema"] {
                say(key, m.get(key).and_then(|v| v.as_str()).unwrap_or("").to_string());
            }
            let checkpoint = m.get("checkpoint");
            say(
                "checkpoint",
                checkpoint
                    .and_then(|c| c.get("run_name"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
            );
            say(
                "checkpoint_step",
                checkpoint
                    .and_then(|c| c.get("global_step"))
                    .and_then(|v| v.as_i64())
                    .map_or_else(|| "null".to_string(), |v| v.to_string()),
            );
        }
        Err(e) => failures.push(format!("manifest.json: {e}")),
    }

    // The playing path itself, not a reconstruction of it: same constructor,
    // same warmup, same parser, same `act`.
    let began = Instant::now();
    match Seat::new(PAD, PAD) {
        Ok(mut seat) => {
            say("startup_ms", format!("{:.3}", began.elapsed().as_secs_f64() * 1e3));
            let text = selfcheck_input();
            let mut reader = std::io::Cursor::new(text.as_bytes());
            let mut obs = Observation::with_dims(PAD, PAD);
            let mut line = String::new();
            let mut scratch = Vec::new();
            match read_handshake(&mut reader)
                .map_err(|e| e.to_string())
                .and_then(|_| {
                    read_observation(&mut reader, &mut obs, &mut line, &mut scratch)
                        .map_err(|e| e.to_string())
                }) {
                Ok(true) => {
                    let decide = Instant::now();
                    match seat.act(&obs) {
                        Ok(a) => {
                            say("decide_ms", format!("{:.3}", decide.elapsed().as_secs_f64() * 1e3));
                            say(
                                "decision",
                                format!("{} {} {} {} {}", a.pass, a.row, a.col, a.dir, a.split),
                            );
                            // A degraded seat answers every frame with a skip.
                            // On this position — a general on 40 army with four
                            // empty neighbours — a skip is not a decision this
                            // bot makes.
                            if a.pass == 1 {
                                failures.push(
                                    "the seat skipped a turn with an obvious move available, \
                                     which is what a bot that failed to start looks like"
                                        .into(),
                                );
                            }
                        }
                        Err(e) => failures.push(format!("act: {e}")),
                    }
                }
                Ok(false) => failures.push("selfcheck frame ended early".into()),
                Err(e) => failures.push(format!("selfcheck frame: {e}")),
            }
        }
        Err(e) => failures.push(format!("seat: {e}")),
    }

    for failure in &failures {
        eprintln!("[joe-rs] selfcheck: {failure}");
    }
    println!("selfcheck {}", if failures.is_empty() { "ok" } else { "FAILED" });
    std::process::exit(if failures.is_empty() { 0 } else { 1 });
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let result = match args.get(1).map(String::as_str) {
        None => wire_main(),
        Some("selfcheck") => run_selfcheck(),
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
