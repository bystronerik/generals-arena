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
use std::time::Instant;

use morpheus_core::belief::Action5;
use morpheus_core::deployment::{try_load_deployment, DeploymentConfig, EvaluatorKind};
use morpheus_core::evaluator::{NetworkEvaluator, ShapedUniformEvaluator};
use morpheus_core::inference::Session;
use morpheus_core::network::Heads;
use morpheus_core::rng::{SharedRng, SmallRng};
use morpheus_core::runtime::{MonotonicClock, RuntimeController};
use morpheus_core::search::SearchEvaluator;
use morpheus_core::wire::{
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
    /// `(load, warmup, init)` in ms — plan §11's fourth success criterion, and
    /// the only place it can be measured: the grace window is spent here.
    startup_ms: (f64, f64, f64),
}

impl Seat {
    fn new(player_id: usize, h: usize, w: usize) -> Result<Self, String> {
        let begin = Instant::now();
        let deployment: DeploymentConfig = try_load_deployment();
        let mut load_ms = 0.0;
        let mut warmup_ms = 0.0;
        let evaluator: Box<dyn SearchEvaluator> = match deployment.evaluator {
            EvaluatorKind::Uniform => Box::new(ShapedUniformEvaluator {
                knobs: deployment.shaping_knobs(),
                ..Default::default()
            }),
            EvaluatorKind::Network => {
                let load = Instant::now();
                let mut session = Session::load_default()?;
                load_ms = load.elapsed().as_secs_f64() * 1e3;
                // Exercise the configured shapes, not the export's fixed set.
                // The engine has no batch axis, so "shape n" is n forwards.
                let shapes = if deployment.warmup_batch_shapes.is_empty() {
                    vec![1, 4, 64]
                } else {
                    deployment.warmup_batch_shapes.clone()
                };
                let warm = Instant::now();
                let x = vec![0f32; morpheus_core::network::IN_CHANNELS * 441];
                for batch in shapes {
                    for _ in 0..batch {
                        session.forward(&x, Heads::Policy);
                        session.forward(&x, Heads::PolicyWdl);
                    }
                }
                warmup_ms = warm.elapsed().as_secs_f64() * 1e3;
                Box::new(NetworkEvaluator::new(session, deployment.shaping_knobs()))
            }
        };

        let rng = SharedRng::new(Box::new(SmallRng::seed_from_u64(0)));
        let mut controller = RuntimeController::new(
            player_id,
            h,
            w,
            deployment.to_runtime_config(),
            Box::new(MonotonicClock::default()),
            rng,
        );
        controller.use_policy_proposal = deployment.use_policy_proposal;
        let total_ms = begin.elapsed().as_secs_f64() * 1e3;
        check_single_threaded()?;
        Ok(Self {
            controller,
            evaluator,
            startup_ms: (load_ms, warmup_ms, (total_ms - load_ms - warmup_ms).max(0.0)),
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
    match morpheus_core::telemetry::thread_count() {
        Some(n) if n > 1 => Err(format!(
            "{n} threads after warmup; this bot is single-threaded by \
             construction and its deadline knobs are calibrated for one core"
        )),
        // `None` is "this platform did not say" — macOS has no /proc — and is
        // deliberately not a pass. The judge runs Linux, which does say.
        _ => Ok(()),
    }
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

/// `morpheus-rs bench [iters]`: the inference half of the M3 shoot-out.
///
/// Batch here is a loop, not a tensor dimension, so "batch 4" is four
/// sequential forwards timed as one unit — which is exactly what the search
/// pays when it evaluates a leaf batch, and therefore the number that belongs
/// beside a batched engine's batch-4 figure.
fn run_bench(iters: usize) -> ! {
    let load = Instant::now();
    let mut session = match Session::load_default() {
        Ok(session) => session,
        Err(err) => {
            eprintln!("[morpheus-rs] {err}");
            std::process::exit(2);
        }
    };
    let load_ms = load.elapsed().as_secs_f64() * 1e3;
    let warm = Instant::now();
    session.warmup();
    println!("load_ms {load_ms:.3}");
    // Print the one build flag that can change this measurement by 49x. A
    // benchmark that does not say what it compiled is a benchmark that can be
    // wrong twice.
    println!("hardware_fma {}", morpheus_core::gemm::HAS_HARDWARE_FMA);
    println!("warmup_ms {:.3}", warm.elapsed().as_secs_f64() * 1e3);

    // The same deterministic stream the candle and tract spikes used, so the
    // three engines are timed on identical inputs rather than on each
    // implementation's idea of "random".
    let mut seed: u32 = 12345;
    let mut next = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        ((seed >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0
    };

    let probe: Vec<f32> = (0..49 * 441).map(|_| next()).collect();
    for (name, ms) in session.network.profile_forward(&probe, 50) {
        println!("stage {name} {ms:.4}");
    }

    for &heads in &[Heads::Policy, Heads::PolicyWdl, Heads::All] {
        let label = match heads {
            Heads::Policy => "policy",
            Heads::PolicyWdl => "policy_wdl",
            Heads::All => "full",
        };
        for &batch in &[1usize, 4, 8] {
            let inputs: Vec<Vec<f32>> = (0..batch)
                .map(|_| (0..49 * 441).map(|_| next()).collect())
                .collect();
            for _ in 0..5 {
                for x in &inputs {
                    std::hint::black_box(session.forward(x, heads));
                }
            }
            let mut times = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t = Instant::now();
                for x in &inputs {
                    std::hint::black_box(session.forward(x, heads));
                }
                times.push(t.elapsed().as_secs_f64() * 1e3);
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let pick = |q: f64| times[((q * times.len() as f64).ceil() as usize).max(1) - 1];
            println!(
                "{label} batch {batch} p50 {:.3} p99 {:.3} min {:.3}",
                pick(0.50),
                pick(0.99),
                times[0]
            );
        }
    }
    std::process::exit(0);
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
    if args.first().map(String::as_str) == Some("bench-belief") {
        let iters = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20);
        let stdin = io::stdin();
        let stdout = io::stdout();
        let mut writer = BufWriter::new(stdout.lock());
        match morpheus_core::parity::bench_belief(&mut stdin.lock(), &mut writer, iters) {
            Ok(()) => std::process::exit(0),
            Err(err) => {
                eprintln!("[morpheus-rs] bench-belief: {err}");
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
            eprintln!("[morpheus-rs] bad handshake: {err}");
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
            eprintln!("[morpheus-rs] cannot start the seat, passing every turn: {err}");
            None
        }
    };

    // Armed by MORPHEUS_RS_TRACE only, and buffered until the game ends: this
    // bot's own probe, since `arena.instrument.runner` can only trace an Agent
    // it constructs in-process. See `telemetry.rs`.
    let mut trace = morpheus_core::telemetry::Trace::from_env();
    let mut turn: i64 = 0;

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

        let action = match seat.as_mut() {
            Some(seat) => panic::catch_unwind(AssertUnwindSafe(|| seat.act(&obs)))
                .unwrap_or_else(|_| {
                    eprintln!("[morpheus-rs] decide panicked on turn {}; passing", obs.turn);
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
