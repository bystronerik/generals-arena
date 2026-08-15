//! `morpheus-joe` — the stdio seat.
//!
//! morpheus's search stack over joe's frozen network
//! (docs/bots/morpheus-rs/joe-net-plan.md). Forked from `bots/morpheus-rs/`,
//! which this work never edits — the N6 round needs current morpheus-rs, this
//! candidate, and joe-rs playing in the **same** rating round, and a baseline
//! that exists only in git history cannot be an arm (§0).
//!
//! **At N1 the net is loaded but not consulted.** `Seat::new` resolves the
//! artifact, verifies its digest, schema-checks it and warms one forward — so
//! the artifact wiring, the refusal path and the startup budget are all real —
//! and then plays through the network-free `ShapedUniformEvaluator`. The
//! observation bridge is N2 and the remap plus value decode are N3. A seat
//! that finishes matches while deciding badly is exactly what this phase's
//! gate asks for; a rating from it would mean nothing.
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
use std::time::Instant;

mod artifact;

use joenet::board::obs::{CELLS, N_ACTION_CHANNELS, N_CHANNELS, TEMPORAL_WINDOW};
use joenet::nn::net::Net;
use morpheus_joe_core::belief::Action5;
use morpheus_joe_core::runtime::deployment::{
    default_deployment_path, deployment_candidates, load_deployment, try_load_deployment,
    DeploymentConfig, EvaluatorKind,
};
use morpheus_joe_core::search::evaluator::ShapedUniformEvaluator;
use morpheus_joe_core::support::rng::{SharedRng, SmallRng};
use morpheus_joe_core::runtime::{Clock, FakeClock, MonotonicClock, RuntimeController};
use morpheus_joe_core::search::SearchEvaluator;
use morpheus_joe_core::io::wire::{
    read_handshake, read_observation, write_action, Action, Observation, PASS,
};

/// The playing seat: a deployment-configured runtime over the exported net.
///
/// The Python's `Agent.__init__` equivalent. Load and warm-up happen here, on
/// the caller's clock, before the first frame is answered — the first move's
/// grace window is what pays for it, and M3 measured the two halves at 3.9 ms
/// and 21.4 ms standalone against 8.5 s of grace.
struct Seat {
    controller: RuntimeController,
    evaluator: Box<dyn SearchEvaluator>,
    /// Joe's frozen net: loaded, schema-checked, warmed, and **not consulted**.
    ///
    /// Held rather than dropped so the 34 MB of weights stay resident and the
    /// startup and memory numbers this phase reports are the ones the real bot
    /// will pay. N2 hands it the bridged `AugState`; until then the evaluator
    /// beside it decides everything.
    #[allow(dead_code)]
    net: Option<Net>,
    /// `(load, warmup, init)` in ms — plan §11's fourth success criterion, and
    /// the only place it can be measured: the grace window is spent here.
    startup_ms: (f64, f64, f64),
    /// The knobs this seat is playing, for the trace's first line. M7 sweeps
    /// configurations, and a trace that does not say which one produced it is
    /// a measurement waiting to be misfiled.
    config_json: String,
}

impl Seat {
    fn new(player_id: usize, h: usize, w: usize) -> Result<Self, String> {
        let begin = Instant::now();
        let deployment: DeploymentConfig = try_load_deployment();
        let mut load_ms = 0.0;
        let mut warmup_ms = 0.0;
        // Both arms decide through the same network-free evaluator at N1. The
        // `Network` arm still pays for the artifact — resolve, verify, load,
        // warm — because that is what this phase's gate is about, and because
        // a bot that only discovers a bad artifact at N2 has spent a phase
        // proving nothing.
        let mut net = None;
        if deployment.evaluator == EvaluatorKind::Network {
            let load = Instant::now();
            let loaded = artifact::load(&artifact::default_artifact_dir()?)?;
            load_ms = load.elapsed().as_secs_f64() * 1e3;
            // One forward on zeros, as joe-rs warms: it primes the allocator
            // and touches every page of the weights, and the forward is
            // stateless so there is nothing to reset. `warmup_batch_shapes`
            // is ignored on purpose — it described a batch axis neither engine
            // has, and thirteen warmup forwards at ~21 ms each would spend a
            // quarter-second of the grace window to prime the same pages once.
            let warm = Instant::now();
            let aug = vec![0f32; N_CHANNELS * CELLS];
            let penalties = vec![0f32; N_ACTION_CHANNELS * CELLS];
            let temporal = vec![0f32; 2 * TEMPORAL_WINDOW];
            loaded
                .forward(&aug, &penalties, &temporal)
                .map_err(|e| format!("warmup forward: {e}"))?;
            warmup_ms = warm.elapsed().as_secs_f64() * 1e3;
            net = Some(loaded);
        }
        let evaluator: Box<dyn SearchEvaluator> = Box::new(ShapedUniformEvaluator {
            knobs: deployment.shaping_knobs(),
            ..Default::default()
        });

        let rng = SharedRng::new(Box::new(SmallRng::seed_from_u64(0)));
        // A charged clock is only meaningful with forecasts to charge, and a
        // spike must be impossible to run by accident: both come from
        // `deployment.json`, both are absent from every rated file, and both
        // are announced on stderr when present.
        let spiking = deployment.fixed_forecasts_ms.is_some();
        let charged = spiking && deployment.charge_fixed_forecasts;
        let clock: Box<dyn Clock> = if charged {
            Box::new(FakeClock::default())
        } else {
            Box::new(MonotonicClock::default())
        };
        let mut controller = RuntimeController::new(
            player_id,
            h,
            w,
            deployment.to_runtime_config(),
            clock,
            rng,
        );
        if let Some(fixed) = deployment.fixed_forecasts_ms.clone() {
            eprintln!(
                "[morpheus-joe] SPIKE: forecasting {fixed:?} instead of measuring, \
                 charged={charged}. This is not a playing configuration and its \
                 move_ms is virtual time."
            );
            controller.fixed_forecasts_ms = Some(fixed);
            controller.charge_fixed_forecasts = charged;
        }
        let total_ms = begin.elapsed().as_secs_f64() * 1e3;
        check_single_threaded()?;
        let config_json = format!(
            "{{\"n_particles\":{},\"target_simulations\":{},\"min_simulations\":{},\
             \"pending_leaf_batch\":{},\"search_depth\":{},\
             \"widen_freeze_below\":{},\"max_forward_equivalents\":{},\
             \"normal_deadline_ms\":{},\"reserve_ms\":{},\"admission_guard_ms\":{},\
             \"spike\":{},\"charged\":{},\"net_consulted\":false}}",
            deployment.n_particles,
            deployment.target_simulations,
            deployment.min_simulations,
            deployment.pending_leaf_batch,
            deployment.search_depth,
            deployment.widen_freeze_below,
            deployment.max_forward_equivalents,
            deployment.normal_deadline_ms,
            deployment.reserve_ms,
            deployment.admission_guard_ms,
            spiking,
            charged,
        );
        Ok(Self {
            controller,
            evaluator,
            net,
            startup_ms: (load_ms, warmup_ms, (total_ms - load_ms - warmup_ms).max(0.0)),
            config_json,
        })
    }

    fn act(&mut self, obs: &Observation) -> Action {
        wire(self.controller.decide(self.evaluator.as_mut(), obs))
    }
}

/// Plan §10's thread-pinning invariant, checked after warmup.
///
/// The bot is single-threaded by construction and every latency number in this
/// project assumes it; a dependency that quietly started a pool would make all
/// of them measurements of a different machine. `Err` here degrades the seat to
/// passing rather than exiting, which is the same trade the loader makes: the
/// judge forfeits a game on an early exit, so a refusal that costs the match is
/// a worse answer than one that costs the game and says why.
fn check_single_threaded() -> Result<(), String> {
    match morpheus_joe_core::runtime::telemetry::thread_count() {
        Some(n) if n > 1 => Err(format!(
            "{n} threads after warmup; this bot is single-threaded by \
             construction and its deadline knobs are calibrated for one core"
        )),
        // `None` is "this platform did not say" — macOS has no /proc — and is
        // deliberately not a pass. The judge runs Linux, which does say.
        _ => Ok(()),
    }
}

/// Whether `f32::mul_add` compiles to an instruction rather than a libm call.
///
/// morpheus-rs exported this as a constant from its own `gemm.rs`. Joe's
/// `gemm.rs` is a byte-identical copy of joe-rs's and carries no such
/// constant, and §8.2 forbids adding one, so the same `cfg!` it would have
/// evaluated is evaluated here. On a non-x86 host the question does not
/// arise — `mul_add` is an instruction on aarch64 — and `true` is the honest
/// answer rather than a skipped check.
fn has_hardware_fma() -> bool {
    !cfg!(target_arch = "x86_64") || cfg!(target_feature = "fma")
}

fn wire(action: Action5) -> Action {
    Action {
        pass: action[0] as u8,
        row: action[1] as u16,
        col: action[2] as u16,
        dir: action[3] as u8,
        split: action[4] as u8,
    }
}

/// `morpheus-joe parity <kind>`: run one ported surface over recorded cases.
///
/// Lives in the shipped binary rather than a separate harness because the
/// thing under test has to be the thing that plays — a parity build with its
/// own feature flags could pass while the submitted binary diverges.
fn run_parity(kind: &str) -> ! {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut writer = BufWriter::new(stdout.lock());
    match morpheus_joe_core::parity::run(kind, &mut stdin.lock(), &mut writer) {
        Ok(()) => std::process::exit(0),
        Err(err) => {
            eprintln!("[morpheus-joe] parity {kind}: {err}");
            std::process::exit(2);
        }
    }
}

/// `morpheus-joe bench [iters]`: how long one forward of joe's net costs here.
///
/// The M3 shoot-out's descendant, and a much shorter program than the one it
/// replaces. That version timed three head sets against three batch sizes,
/// because morpheus's graph had eleven heads behind a `Heads` switch and the
/// question was which entry point the search should pay for. Joe's forward has
/// no switch and no batch axis — one call returns 4,410 logits, a scalar and
/// 128 bins — so there is one number to report.
///
/// It is deliberately **not** the authority on that number. N0 measured the
/// forward at 20.81 ms p50 / 23.01 ms p99 on one x86 core through joe-rs's
/// `bench --stages`, over a real game's frames rather than a deterministic
/// noise stream, and the whole budget in §5 rests on that measurement. This
/// exists to catch a build that is wrong by an order of magnitude — most
/// likely one compiled without the hardware FMA, which costs ~49x — before it
/// plays a rated game.
fn run_bench(iters: usize) -> ! {
    let dir = match artifact::default_artifact_dir() {
        Ok(dir) => dir,
        Err(err) => {
            eprintln!("[morpheus-joe] {err}");
            std::process::exit(2);
        }
    };
    let load = Instant::now();
    let net = match artifact::load(&dir) {
        Ok(net) => net,
        Err(err) => {
            eprintln!("[morpheus-joe] {err}");
            std::process::exit(2);
        }
    };
    println!("load_ms {:.3}", load.elapsed().as_secs_f64() * 1e3);
    // Print the one build flag that can change this measurement by 49x. A
    // benchmark that does not say what it compiled is a benchmark that can be
    // wrong twice.
    println!("hardware_fma {}", has_hardware_fma());

    // The same deterministic stream the candle and tract spikes used, so an
    // engine change is timed on identical inputs rather than on each
    // implementation's idea of "random".
    let mut seed: u32 = 12345;
    let mut next = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        ((seed >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0
    };
    let aug: Vec<f32> = (0..N_CHANNELS * CELLS).map(|_| next()).collect();
    let temporal: Vec<f32> = (0..2 * TEMPORAL_WINDOW).map(|_| next()).collect();
    // Zeros, because that is what the play path passes: joe's penalties are
    // added to the flat logits after unpatchify and the trunk never sees them,
    // so this bot masks with morpheus's `play_mask` instead and never builds
    // joe's (joe-net-plan §6.2).
    let penalties = vec![0f32; N_ACTION_CHANNELS * CELLS];

    let warm = Instant::now();
    for _ in 0..5 {
        std::hint::black_box(net.forward(&aug, &penalties, &temporal).unwrap());
    }
    println!("warmup_ms {:.3}", warm.elapsed().as_secs_f64() * 1e3);

    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        std::hint::black_box(net.forward(&aug, &penalties, &temporal).unwrap());
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pick = |q: f64| times[((q * times.len() as f64).ceil() as usize).max(1) - 1];
    println!(
        "forward p50 {:.3} p99 {:.3} min {:.3}",
        pick(0.50),
        pick(0.99),
        times[0]
    );
    std::process::exit(0);
}

/// `morpheus-joe selfcheck`: refuse to be a bot that only looks like one.
///
/// Every failure this checks for is silent at play time by design. A missing
/// artifact, a `deployment.json` that did not resolve, a build without a
/// hardware FMA — none of them stop the process, because the judge forfeits a
/// game on an early exit and RULES.md §08 charges only one fault out of fifty
/// for a bad reply. So the seat degrades to passing, the protocol stays
/// well-formed, and nothing downstream can tell the difference: the M0.5-era
/// smoke test scored a bot with its weights deleted as "2 well-formed
/// actions", byte-identical to a healthy one.
///
/// Intake is the one moment where failing loudly is the better trade — a
/// rejected submission costs a resubmission, a degraded one costs every rated
/// game it plays. `build.sh` runs this and aborts on a non-zero exit.
fn run_selfcheck() -> ! {
    let mut failures: Vec<String> = Vec::new();
    let say = |key: &str, value: String| println!("{key} {value}");

    // Reported before anything else: it is the one fact that changes a
    // correct binary into a 49x-too-slow one, and M3 shipped it wrong once.
    say("hardware_fma", has_hardware_fma().to_string());
    say("hardware_avx2", cfg!(target_feature = "avx2").to_string());
    if cfg!(target_arch = "x86_64") && !(has_hardware_fma() && cfg!(target_feature = "avx2")) {
        failures.push(
            "this x86_64 build has no AVX2/FMA: every `f32::mul_add` in the inference \
             kernels becomes a libm `fmaf()` call, measured at 277 ms per forward \
             against 5.6 ms on morpheus's smaller graph. cargo did not read \
             .cargo/config.toml (target-cpu=x86-64-v3) — build from the crate root, \
             as tools/submission/build.sh does."
                .into(),
        );
    }

    // The config is resolved here rather than trusted to `Seat::new`, which
    // falls back to the Part 07 placeholders and plays a *different bot* —
    // four times the particles, twice the search depth, a 125 ms deadline
    // against 140 — with one line on stderr to say so.
    match default_deployment_path() {
        Some(path) => {
            say("deployment_path", path.display().to_string());
            if let Err(err) = load_deployment(&path) {
                failures.push(format!("{path:?} does not parse: {err}"));
            }
        }
        None => failures.push(format!(
            "no deployment.json among {:?}; the bot would play placeholder knobs",
            deployment_candidates()
        )),
    }

    match artifact::default_artifact_dir() {
        Ok(dir) => {
            say("artifact_dir", dir.display().to_string());
            match artifact::provenance(&dir) {
                Ok(p) => {
                    say("safetensors_sha256", p.safetensors_sha256);
                    say("tensor_schema", p.tensor_schema);
                    say("checkpoint", p.checkpoint_run);
                    say("checkpoint_step", p.checkpoint_step);
                }
                Err(err) => failures.push(format!("manifest: {err}")),
            }
            // Reported so a judge-side log can be compared against the row in
            // `data/bot_versions/morpheus-joe.json` without a rebuild, and so
            // a stale fan-out is visible here as well as in the test.
            if let Err(err) = artifact::load(&dir) {
                failures.push(format!("artifact: {err}"));
            }
        }
        Err(err) => failures.push(format!("artifact: {err}")),
    }

    // The playing path itself, not a reconstruction of it: same constructor,
    // same warmup, same thread-count invariant.
    match Seat::new(0, 21, 21) {
        Ok(mut seat) => {
            let (load_ms, warmup_ms, init_ms) = seat.startup_ms;
            say("load_ms", format!("{load_ms:.3}"));
            say("warmup_ms", format!("{warmup_ms:.3}"));
            say("init_ms", format!("{init_ms:.3}"));
            say("config", seat.config_json.clone());

            let obs = selfcheck_frame();
            let began = Instant::now();
            let action = seat.act(&obs);
            let decide_ms = began.elapsed().as_secs_f64() * 1e3;
            say(
                "decision",
                format!(
                    "{} {} {} {} {}",
                    action.pass, action.row, action.col, action.dir, action.split
                ),
            );
            say("decide_ms", format!("{decide_ms:.3}"));
            // A degraded seat answers every frame with `1 0 0 0 0`. On this
            // position — a general on thirteen army with four empty
            // neighbours — a skip is not a decision this bot makes, and that
            // holds at N1: the prior is flat, but the tactics layer and the
            // hard rules still run and still prefer a move to a skip.
            if action.pass == 1 {
                failures.push(
                    "the seat skipped a turn with an obvious move available, which is \
                     what a bot that failed to start looks like"
                        .into(),
                );
            }
        }
        Err(err) => failures.push(format!("seat: {err}")),
    }

    for failure in &failures {
        eprintln!("[morpheus-joe] selfcheck: {failure}");
    }
    println!("selfcheck {}", if failures.is_empty() { "ok" } else { "FAILED" });
    std::process::exit(if failures.is_empty() { 0 } else { 1 });
}

/// A mid-opening frame with one unambiguous move in it.
///
/// Hand-built rather than replayed: the parity corpus lives in the repo and
/// the whole point of this check is to run where the repo does not. Thirteen
/// army on the general is below the 35 a castle costs, so a build is an
/// illegal action and a skip is the only wrong answer available.
fn selfcheck_frame() -> Observation {
    use morpheus_joe_core::io::wire::{OWNER_ME, TYPE_FOG, TYPE_GENERAL, TYPE_PLAIN};

    let mut obs = Observation::with_dims(21, 21);
    obs.type_grid.fill(TYPE_FOG);
    obs.turn = 24;
    obs.my_land = 1;
    obs.my_army = 13;
    obs.opp_land = 1;
    obs.opp_army = 13;
    for row in 9..=11 {
        for col in 9..=11 {
            let cell = obs.idx(row, col);
            obs.type_grid[cell] = TYPE_PLAIN;
        }
    }
    let general = obs.idx(10, 10);
    obs.type_grid[general] = TYPE_GENERAL;
    obs.owner_grid[general] = OWNER_ME;
    obs.army_grid[general] = 13;
    obs
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("selfcheck") {
        run_selfcheck();
    }
    if args.first().map(String::as_str) == Some("parity") {
        match args.get(1) {
            Some(kind) => run_parity(kind),
            None => {
                eprintln!(
                    "[morpheus-joe] usage: morpheus-joe parity <transition|observe|mask|cost|...>"
                );
                std::process::exit(2);
            }
        }
    }
    if args.first().map(String::as_str) == Some("bench-belief") {
        let iters = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20);
        let stdin = io::stdin();
        let stdout = io::stdout();
        let mut writer = BufWriter::new(stdout.lock());
        match morpheus_joe_core::parity::bench_belief(&mut stdin.lock(), &mut writer, iters) {
            Ok(()) => std::process::exit(0),
            Err(err) => {
                eprintln!("[morpheus-joe] bench-belief: {err}");
                std::process::exit(2);
            }
        }
    }
    if args.first().map(String::as_str) == Some("bench") {
        let iters = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(200);
        run_bench(iters);
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
            eprintln!("[morpheus-joe] bad handshake: {err}");
            std::process::exit(1);
        }
    };

    let mut obs = Observation::with_dims(handshake.h, handshake.w);
    let mut line = String::new();

    // A seat that will not construct cannot play, and the judge forfeits a
    // game on an early exit — so a load failure degrades to passing every turn
    // with the reason on stderr, which costs the game but not the match.
    let mut seat = match Seat::new(
        handshake.player_id as usize,
        handshake.h,
        handshake.w,
    ) {
        Ok(seat) => Some(seat),
        Err(err) => {
            eprintln!("[morpheus-joe] cannot start the seat, passing every turn: {err}");
            None
        }
    };

    // Armed by MORPHEUS_JOE_TRACE only, and buffered until the game ends: this
    // bot's own probe, since `arena.instrument.runner` can only trace an Agent
    // it constructs in-process. See `telemetry.rs`.
    let mut trace = morpheus_joe_core::runtime::telemetry::Trace::from_env();
    if let (Some(trace), Some(seat)) = (trace.as_mut(), seat.as_ref()) {
        trace.set_header(seat.config_json.clone());
    }
    let mut turn: i64 = 0;

    loop {
        match read_observation(&mut reader, &mut obs, &mut line) {
            Ok(true) => {}
            Ok(false) => break, // stdin closed: game over
            Err(err) => {
                // A frame we cannot parse is not a frame we can answer. Pass
                // and keep the seat alive; the alternative is a forfeit.
                eprintln!("[morpheus-joe] {err}");
                let _ = write_action(&mut writer, PASS);
                continue;
            }
        }

        let action = match seat.as_mut() {
            Some(seat) => panic::catch_unwind(AssertUnwindSafe(|| seat.act(&obs)))
                .unwrap_or_else(|_| {
                    eprintln!("[morpheus-joe] decide panicked on turn {}; passing", obs.turn);
                    PASS
                }),
            None => PASS,
        };

        if write_action(&mut writer, action).is_err() {
            break; // engine hung up while we were replying
        }

        // After the reply is flushed, outside every clock the seat reports.
        turn += 1;
        if let (Some(trace), Some(seat)) = (trace.as_mut(), seat.as_ref()) {
            let startup = if turn == 1 { Some(seat.startup_ms) } else { None };
            trace.record(turn, &seat.controller.metrics, startup);
        }
    }

    let _ = writer.flush();
    if let Some(trace) = trace.as_ref() {
        trace.flush();
    }
}
