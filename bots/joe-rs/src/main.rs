//! `joe-rs` — the greedy policy of the frozen joe network, in Rust.
//!
//! Wire mode (no arguments): the competition stdio seat. Per turn: parse the
//! frame, rebuild the 14-channel raw tensor, build cost + masks, augment to
//! 39 channels with the persistent state, one float32 forward pass, then
//! deterministic Gumbel selection over the masked logits (`board::select`,
//! selection-plan S1): an exact sample of `softmax(logits / T)` whose noise
//! is hashed from the frame, so the bot stays a pure function of the game.
//! `JOE_RS_TEMPERATURE` sets T (default 1 — the distribution training
//! sampled; 0 restores the plain argmax). The Python sibling takes the plain
//! argmax; the divergence is deliberate
//! (docs/bots/joe-rs/selection-plan.md).
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
//!
//! # Layering
//!
//! The modules form a DAG and are listed below in dependency order: a module
//! may name the ones above it and must not name the ones below it.
//!
//! ```text
//! xla_math  io  ->  board  ->  nn  ->  parity   main
//! ```
//!
//! `xla_math` is the floor — the f32 sites where XLA does not compute what
//! naive Rust computes — and `io` the formats, neither of which knows what a
//! channel is. `parity` sits above the stack because it is the harness half
//! of the binary and never plays, and this file above that because it is the
//! composition root. Nothing enforces the rule but review; a crate split
//! would, and costs more than it is worth at this size
//! (docs/bots/joe-rs/refactor-plan.md §1).
//!
//! Single-threaded by construction. One dedicated core is a competition
//! constraint, not a tuning choice, so nothing here may spawn a thread.

mod board;
mod io;
mod nn;
mod parity;
mod xla_math;

use std::io::{self as stdio, BufWriter, Write};
use std::panic::{self, AssertUnwindSafe};
use std::path::PathBuf;
use std::time::Instant;

use crate::board::action::decode_action;
use crate::board::obs::{
    augment_obs, build_cost_from_raw, compute_build_mask_from_raw, compute_valid_move_mask,
    frame_to_raw, normalize_observations, prepare_action_mask, AugScratch, AugState, CELLS,
    N_ACTION_CHANNELS, N_CHANNELS, PAD, TEMPORAL_WINDOW,
};
use crate::board::select::{board_digest, select_action};
use crate::io::wire::{
    read_handshake, read_observation, write_action, Action, Observation, PASS, TYPE_FOG,
    TYPE_GENERAL, TYPE_PLAIN,
};
use crate::nn::net::Net;

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

// ------------------------------------------------------- the stage breakdown

/// The stages `bench --stages` splits the per-move path into, in run order.
///
/// The split exists for the joe-net port
/// (`docs/bots/morpheus-rs/joe-net-plan.md` N0.1), which has to know three
/// things the whole-move figure in `docs/bots/joe-rs/latency.md` cannot answer:
/// what the forward alone costs, what `augment_obs` alone costs, and what the
/// mask work costs — because that port keeps the first two per network call and
/// drops the third.
///
/// `normalize` carries the `mem::swap` and the two 512-wide ring-buffer copies
/// as well as `normalize_observations`; `parse` and part of `decode` are timed
/// by the harness rather than by `act_staged`, because they sit outside it.
const STAGE_NAMES: [&str; 8] = [
    "parse",
    "raw",
    "mask",
    "augment",
    "normalize",
    "forward",
    "decode",
    "stderr",
];
const S_PARSE: usize = 0;
const S_RAW: usize = 1;
const S_MASK: usize = 2;
const S_AUGMENT: usize = 3;
const S_NORMALIZE: usize = 4;
const S_FORWARD: usize = 5;
const S_DECODE: usize = 6;
const S_STDERR: usize = 7;

/// Where `act_staged` reports a stage boundary.
trait StageClock {
    /// Charge the time since the last mark to `stage`. Accumulates, so one
    /// stage may occupy more than one span of the order.
    fn mark(&mut self, stage: usize);
}

/// The wire path's clock: no clock at all. Monomorphization deletes the marks.
struct NoStages;

impl StageClock for NoStages {
    #[inline(always)]
    fn mark(&mut self, _stage: usize) {}
}

/// `bench --stages`'s clock: one turn's accumulators, plus the running sample.
///
/// Eight `Instant::now()` calls per turn cost around 200 ns against a ~21 ms
/// move, which is nothing at the whole-move scale but is a real fraction of
/// `decode` and `stderr` — read those two as upper bounds.
struct StageTimes {
    last: Instant,
    turn_ms: [f64; STAGE_NAMES.len()],
    samples: Vec<[f64; STAGE_NAMES.len()]>,
}

impl StageTimes {
    fn new() -> Self {
        Self {
            last: Instant::now(),
            turn_ms: [0.0; STAGE_NAMES.len()],
            samples: Vec::new(),
        }
    }

    /// Start a turn: zero the accumulators and reset the reference instant.
    fn begin_turn(&mut self) {
        self.turn_ms = [0.0; STAGE_NAMES.len()];
        self.last = Instant::now();
    }

    fn end_turn(&mut self) {
        self.samples.push(self.turn_ms);
    }
}

impl StageClock for StageTimes {
    #[inline]
    fn mark(&mut self, stage: usize) {
        let now = Instant::now();
        self.turn_ms[stage] += now.duration_since(self.last).as_secs_f64() * 1e3;
        self.last = now;
    }
}

/// The playing seat: network plus the per-game state and preallocated
/// buffers. Load and warmup happen in `new`, inside the first-move grace.
struct Seat {
    net: Net,
    h: usize,
    w: usize,
    /// Selection temperature (selection-plan S1). 1 replays the distribution
    /// PPO trained under; 0 or less restores the plain argmax.
    temperature: f32,
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

        // The S1 knob, read once — a per-turn getenv would put a syscall on
        // the move path. A malformed value is loud but not fatal: a seat
        // that refuses to construct forfeits by passing every turn, which
        // costs more than playing the default temperature does.
        let temperature = match std::env::var("JOE_RS_TEMPERATURE") {
            Err(_) => 1.0,
            Ok(raw) => match raw.trim().parse::<f32>() {
                Ok(t) if t.is_finite() => t,
                _ => {
                    eprintln!(
                        "[joe-rs] JOE_RS_TEMPERATURE {raw:?} is not a finite number; using 1"
                    );
                    1.0
                }
            },
        };

        let seat = Self {
            net,
            h,
            w,
            temperature,
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
        eprintln!(
            "[joe-rs] load {load_ms:.1} ms, warmup {warmup_ms:.1} ms, temperature {}",
            seat.temperature
        );
        Ok(seat)
    }

    /// The full per-move path. Mirrors `agent.py::step` + the pass clamp in
    /// `agent.py::act`.
    fn act(&mut self, obs: &Observation) -> Result<Action, String> {
        self.act_staged(obs, &mut NoStages)
    }

    /// `act`, with the stage boundaries `bench --stages` measures.
    ///
    /// The wire path calls it through `act` with [`NoStages`], whose `mark` is
    /// an empty inlined method, so the marks cost nothing where they are not
    /// wanted and the measured path stays the played path. Two code paths would
    /// drift, and a drifted stage split is a measurement that describes a
    /// program nobody runs.
    fn act_staged<C: StageClock>(
        &mut self,
        obs: &Observation,
        clock: &mut C,
    ) -> Result<Action, String> {
        frame_to_raw(obs, &mut self.raw);
        build_cost_from_raw(&self.raw, self.h, self.w, &mut self.cost);
        clock.mark(S_RAW);
        compute_valid_move_mask(&self.raw, self.h, self.w, &mut self.move_mask);
        compute_build_mask_from_raw(&self.raw, self.h, self.w, &self.cost, &mut self.build_mask);
        clock.mark(S_MASK);
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
        clock.mark(S_AUGMENT);
        std::mem::swap(&mut self.state, &mut self.next_state);
        self.temporal[..TEMPORAL_WINDOW].copy_from_slice(&self.state.opponent_army_history);
        self.temporal[TEMPORAL_WINDOW..].copy_from_slice(&self.state.opponent_land_history);
        normalize_observations(&mut self.aug);
        clock.mark(S_NORMALIZE);
        prepare_action_mask(
            &self.move_mask,
            &self.build_mask,
            self.h,
            self.w,
            &mut self.penalties,
        );
        // The second half of joe's mask work, and the reason `mark` accumulates
        // rather than assigns: one stage, two places in the order.
        clock.mark(S_MASK);
        let fwd = self.net.forward(&self.aug, &self.penalties, &self.temporal)?;
        clock.mark(S_FORWARD);
        // Selection (plan S1) joins `decode`: both turn the network's answer
        // into the reply, and together they still cost microseconds.
        let idx = select_action(
            &fwd.logits,
            &self.penalties,
            obs.turn,
            board_digest(obs),
            self.temperature,
        );
        let a = decode_action(idx);
        clock.mark(S_DECODE);
        // Free eval telemetry (port-plan §5): v1 play ignores the value.
        // Timed on its own because it is an unbuffered write per turn, and it
        // is this bot's habit rather than anything the net or the port needs.
        eprintln!("[joe-rs] turn {} value {:+.4}", obs.turn, fwd.value);
        clock.mark(S_STDERR);
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
    let stdin = stdio::stdin();
    let mut reader = stdin.lock();
    let stdout = stdio::stdout();
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
///
/// `joe-rs bench --stages` replays the same log and reports the same total,
/// split per [`STAGE_NAMES`]: the table goes to stderr for reading, one JSON
/// object to stdout for recording. The plain form still prints exactly what it
/// printed before, so the numbers in `docs/bots/joe-rs/latency.md` stay
/// comparable to anything measured after this.
fn bench_main(stages: bool) -> Result<(), String> {
    use std::io::Read;

    let mut text = String::new();
    stdio::stdin().read_to_string(&mut text).map_err(|e| e.to_string())?;
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
    let mut sink = stdio::sink();
    let mut times_ms: Vec<f64> = Vec::new();
    let mut clock = StageTimes::new();
    loop {
        let t0 = Instant::now();
        clock.begin_turn();
        match read_observation(&mut reader, &mut obs, &mut line, &mut scratch) {
            Ok(true) => {}
            Ok(false) => break,
            Err(e) => return Err(e.to_string()),
        }
        let action = if stages {
            clock.mark(S_PARSE);
            seat.act_staged(&obs, &mut clock)?
        } else {
            seat.act(&obs)?
        };
        write_action(&mut sink, action).map_err(|e| e.to_string())?;
        if stages {
            // The reply encode joins `decode`: both turn the network's answer
            // into bytes, and neither is separately actionable.
            clock.mark(S_DECODE);
            clock.end_turn();
        }
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
    if stages {
        report_stages(&clock.samples, &times_ms, startup_ms);
    }
    Ok(())
}

/// Per-stage percentiles: a table on stderr, one JSON object on stdout.
///
/// Each stage is sorted on its own, so the columns do not sum down the p99
/// row — the turn with the slow forward is rarely the turn with the slow
/// parse. Only the `mean` column adds up, and it is the one to add up.
fn report_stages(samples: &[[f64; STAGE_NAMES.len()]], total_ms: &[f64], startup_ms: f64) {
    if samples.is_empty() {
        eprintln!("[joe-rs] no staged turns to report");
        return;
    }
    let n = samples.len();
    let pct = |sorted: &[f64], p: f64| sorted[((n as f64 - 1.0) * p) as usize];

    let mut rows: Vec<(&str, f64, [f64; 4])> = Vec::with_capacity(STAGE_NAMES.len() + 1);
    for (at, name) in STAGE_NAMES.iter().enumerate() {
        let mut column: Vec<f64> = samples.iter().map(|turn| turn[at]).collect();
        let mean = column.iter().sum::<f64>() / n as f64;
        column.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let quantiles = [
            pct(&column, 0.50),
            pct(&column, 0.90),
            pct(&column, 0.99),
            pct(&column, 1.0),
        ];
        rows.push((name, mean, quantiles));
    }
    // `total` is the harness's own per-turn timing, not the sum of the stages,
    // so the gap between it and the column of means is the unaccounted time.
    let total_mean = total_ms.iter().sum::<f64>() / n as f64;
    rows.push((
        "total",
        total_mean,
        [
            pct(total_ms, 0.50),
            pct(total_ms, 0.90),
            pct(total_ms, 0.99),
            pct(total_ms, 1.0),
        ],
    ));

    eprintln!("[joe-rs] stages over {n} turns, startup {startup_ms:.1} ms");
    eprintln!(
        "[joe-rs] {:<10} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "stage", "mean", "p50", "p90", "p99", "max"
    );
    for (name, mean, q) in &rows {
        eprintln!(
            "[joe-rs] {name:<10} {mean:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>9.3}",
            q[0], q[1], q[2], q[3]
        );
    }

    let mut json = format!("{{\"turns\":{n},\"startup_ms\":{startup_ms:.3},\"stages\":{{");
    for (i, (name, mean, q)) in rows.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        json.push_str(&format!(
            "\"{name}\":{{\"mean\":{mean:.4},\"p50\":{:.4},\"p90\":{:.4},\"p99\":{:.4},\"max\":{:.4}}}",
            q[0], q[1], q[2], q[3]
        ));
    }
    json.push_str("}}");
    println!("{json}");
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
        .and_then(|t| crate::io::json::parse(&t))
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
            // The selection knob, so a judge-side log or a round arm's
            // registration can be checked without reading the environment.
            say("temperature", format!("{}", seat.temperature));
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Spin until the monotonic clock has visibly moved.
    ///
    /// Not a sleep: the assertions below are about whether two marks land in
    /// separate accumulators, and a platform whose `Instant` resolution makes
    /// two adjacent marks read zero would pass the test by accident.
    fn spin(ms: f64) {
        let t0 = Instant::now();
        while t0.elapsed().as_secs_f64() * 1e3 < ms {
            std::hint::spin_loop();
        }
    }

    /// `mask` is charged from two places in the order, and every stage number
    /// N0 reads is wrong if the second charge overwrites the first.
    #[test]
    fn a_stage_charged_twice_in_one_turn_accumulates() {
        let mut clock = StageTimes::new();
        clock.begin_turn();
        spin(1.0);
        clock.mark(S_MASK);
        let first = clock.turn_ms[S_MASK];
        assert!(first >= 1.0);
        spin(1.0);
        clock.mark(S_FORWARD);
        spin(1.0);
        clock.mark(S_MASK);
        assert!(clock.turn_ms[S_MASK] >= first + 1.0);
        assert!(clock.turn_ms[S_FORWARD] >= 1.0);
        clock.end_turn();

        // And a new turn starts from zero, or every sample is cumulative.
        clock.begin_turn();
        assert_eq!(clock.turn_ms[S_MASK], 0.0);
        assert_eq!(clock.samples.len(), 1);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let result = match args.get(1).map(String::as_str) {
        None => wire_main(),
        Some("selfcheck") => run_selfcheck(),
        Some("bench") => bench_main(args.get(2).map(String::as_str) == Some("--stages")),
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
