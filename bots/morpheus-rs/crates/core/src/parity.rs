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

use std::io::{BufRead, Write};
use std::rc::Rc;

use crate::board::action::{
    decode_action, encode_action, legal_mask, live_build_cost, N_ACTIONS, PAD,
};
use crate::belief::{
    ess, ess_fraction, filter_step, Action5, BeliefConfig, BeliefState, HistoryFrame, Particle,
};
use crate::board::hashing::{
    child_edge_key, enemy_info_hash_prehashed, info_state_key_prehashed, memory_digest,
    observation_payload, roll_history_digest,
};
use crate::search::matrix;
use crate::board::memory::{update_memory, VisibleMemory};
use crate::nn::network;
use crate::board::observe::emit_observation;
use crate::belief::summary::summarize_belief;
use crate::belief::proposal::{
    propose_enemy_actions, singleton_probs, softmax_masked, top_legal_actions, uniform_legal_probs,
    ProposalTelemetry,
};
use crate::belief::recovery::{maximum_entropy_reconstruction, recover_belief, rejuvenate};
use crate::belief::reservoir::ParticleReservoir;
use crate::support::rng::{argsort_desc_numpy, npsum, Method, RecordedDraw, Replay};
use crate::board::state::GameState;
use crate::board::symmetry;
use crate::tactics;
use crate::nn::tensor::{build_tensor, BeliefSummary, ARMY_SCALE};
use crate::board::transition::{determine_move_order, transition, Actions};
use crate::io::wire::Observation;

/// Positional reader over the whitespace-separated integer stream.
pub struct Ints {
    values: Vec<i64>,
    at: usize,
}

impl Ints {
    pub fn read_all<R: BufRead>(reader: &mut R) -> Result<Self, String> {
        let mut text = String::new();
        reader
            .read_to_string(&mut text)
            .map_err(|e| format!("reading cases: {e}"))?;
        let mut values = Vec::new();
        for token in text.split_ascii_whitespace() {
            values.push(
                token
                    .parse::<i64>()
                    .map_err(|_| format!("not an integer: {token:?}"))?,
            );
        }
        Ok(Self { values, at: 0 })
    }

    pub fn next(&mut self) -> Result<i64, String> {
        let value = *self
            .values
            .get(self.at)
            .ok_or_else(|| format!("case stream ended after {} integers", self.at))?;
        self.at += 1;
        Ok(value)
    }

    fn n(&mut self) -> Result<usize, String> {
        Ok(self.next()? as usize)
    }

    fn ints(&mut self, count: usize) -> Result<Vec<i32>, String> {
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            out.push(self.next()? as i32);
        }
        Ok(out)
    }

    /// Floats travel as their raw `f32` bit patterns, as integers.
    ///
    /// Not as decimal text: a float that round-trips through formatting is a
    /// float whose last bits depend on two languages agreeing about printing,
    /// which is exactly the argument tier-2 parity is trying not to have. Bit
    /// patterns make the channel lossless and leave the tolerance decision
    /// entirely to the comparison.
    fn floats(&mut self, count: usize) -> Result<Vec<f32>, String> {
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            out.push(f32::from_bits(self.next()? as u32));
        }
        Ok(out)
    }

    /// f64 the same way `floats` reads f32: raw bit patterns, as integers.
    fn f64s(&mut self, count: usize) -> Result<Vec<f64>, String> {
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            out.push(f64::from_bits(self.next()? as u64));
        }
        Ok(out)
    }

    fn action(&mut self) -> Result<[i32; 5], String> {
        Ok([
            self.next()? as i32,
            self.next()? as i32,
            self.next()? as i32,
            self.next()? as i32,
            self.next()? as i32,
        ])
    }
}

fn fill_bools(dst: &mut [bool], src: &[i32]) {
    for (i, v) in src.iter().enumerate() {
        dst[i] = *v != 0;
    }
}

/// `h w time winner gp[4] armies[n] own0[n] own1[n] neutral[n] generals[n]
/// castles[n] mountains[n] passable[n]`
fn read_state(ints: &mut Ints) -> Result<GameState, String> {
    let h = ints.n()?;
    let w = ints.n()?;
    let mut state = GameState::empty(h, w);
    state.time = ints.next()? as i32;
    state.winner = ints.next()? as i32;
    for seat in 0..2 {
        state.general_positions[seat] = [ints.next()? as i32, ints.next()? as i32];
    }
    let n = h * w;
    let armies = ints.ints(n)?;
    state.armies[..n].copy_from_slice(&armies);
    let planes = ints.ints(n * 7)?;
    fill_bools(&mut state.ownership[0], &planes[0..n]);
    fill_bools(&mut state.ownership[1], &planes[n..2 * n]);
    fill_bools(&mut state.ownership_neutral, &planes[2 * n..3 * n]);
    fill_bools(&mut state.generals, &planes[3 * n..4 * n]);
    fill_bools(&mut state.castles, &planes[4 * n..5 * n]);
    fill_bools(&mut state.mountains, &planes[5 * n..6 * n]);
    fill_bools(&mut state.passable, &planes[6 * n..7 * n]);
    Ok(state)
}

fn write_state(out: &mut Vec<i64>, state: &GameState) {
    let n = state.cells();
    out.push(state.h as i64);
    out.push(state.w as i64);
    out.push(state.time as i64);
    out.push(state.winner as i64);
    for seat in 0..2 {
        out.push(state.general_positions[seat][0] as i64);
        out.push(state.general_positions[seat][1] as i64);
    }
    out.extend(state.armies[..n].iter().map(|&v| v as i64));
    for plane in [
        &state.ownership[0][..n],
        &state.ownership[1][..n],
        &state.ownership_neutral[..n],
        &state.generals[..n],
        &state.castles[..n],
        &state.mountains[..n],
        &state.passable[..n],
    ] {
        out.extend(plane.iter().map(|&v| v as i64));
    }
}

/// `h w turn my_land my_army opp_land opp_army type[n] owner[n] army[n]`
fn read_observation(ints: &mut Ints) -> Result<Observation, String> {
    let h = ints.n()?;
    let w = ints.n()?;
    let mut obs = Observation::with_dims(h, w);
    obs.turn = ints.next()? as i32;
    obs.my_land = ints.next()? as i32;
    obs.my_army = ints.next()? as i32;
    obs.opp_land = ints.next()? as i32;
    obs.opp_army = ints.next()? as i32;
    let n = h * w;
    for i in 0..n {
        obs.type_grid[i] = ints.next()? as u8;
    }
    for i in 0..n {
        obs.owner_grid[i] = ints.next()? as u8;
    }
    for i in 0..n {
        obs.army_grid[i] = ints.next()? as i32;
    }
    Ok(obs)
}

fn write_observation(out: &mut Vec<i64>, obs: &Observation) {
    let n = obs.h * obs.w;
    out.extend([
        obs.h as i64,
        obs.w as i64,
        obs.turn as i64,
        obs.my_land as i64,
        obs.my_army as i64,
        obs.opp_land as i64,
        obs.opp_army as i64,
    ]);
    out.extend(obs.type_grid[..n].iter().map(|&v| v as i64));
    out.extend(obs.owner_grid[..n].iter().map(|&v| v as i64));
    out.extend(obs.army_grid[..n].iter().map(|&v| v as i64));
}

/// `h w` then the eleven planes, in `VisibleMemory` field order.
fn read_memory(ints: &mut Ints) -> Result<VisibleMemory, String> {
    let h = ints.n()?;
    let w = ints.n()?;
    let n = h * w;
    let mut memory = VisibleMemory::empty(h, w);
    let planes = ints.ints(n * 11)?;
    fill_bools(&mut memory.known_mountain, &planes[0..n]);
    fill_bools(&mut memory.known_passable_base, &planes[n..2 * n]);
    fill_bools(&mut memory.known_castle, &planes[2 * n..3 * n]);
    fill_bools(&mut memory.own_general, &planes[3 * n..4 * n]);
    fill_bools(&mut memory.known_enemy_general, &planes[4 * n..5 * n]);
    fill_bools(&mut memory.ever_visible, &planes[5 * n..6 * n]);
    memory.last_seen_turn[..n].copy_from_slice(&planes[6 * n..7 * n]);
    memory.remembered_owner[..n].copy_from_slice(&planes[7 * n..8 * n]);
    memory.remembered_army[..n].copy_from_slice(&planes[8 * n..9 * n]);
    fill_bools(&mut memory.remembered_was_castle, &planes[9 * n..10 * n]);
    memory.remembered_castle_owner[..n].copy_from_slice(&planes[10 * n..11 * n]);
    Ok(memory)
}

fn write_memory(out: &mut Vec<i64>, memory: &VisibleMemory) {
    let n = memory.cells();
    out.push(memory.h as i64);
    out.push(memory.w as i64);
    for plane in [
        &memory.known_mountain[..n],
        &memory.known_passable_base[..n],
        &memory.known_castle[..n],
        &memory.own_general[..n],
        &memory.known_enemy_general[..n],
        &memory.ever_visible[..n],
    ] {
        out.extend(plane.iter().map(|&v| v as i64));
    }
    for plane in [
        &memory.last_seen_turn[..n],
        &memory.remembered_owner[..n],
        &memory.remembered_army[..n],
    ] {
        out.extend(plane.iter().map(|&v| v as i64));
    }
    out.extend(memory.remembered_was_castle[..n].iter().map(|&v| v as i64));
    out.extend(memory.remembered_castle_owner[..n].iter().map(|&v| v as i64));
}

/// Run one parity kind end to end over stdin, writing one line per case.
/// f32 out as its raw bit pattern, the same lossless channel `Ints::floats`
/// reads in.
fn push_f32(out: &mut Vec<i64>, values: &[f32]) {
    out.extend(values.iter().map(|v| v.to_bits() as i64));
}

/// f64 the same way. `to_bits()` is a `u64`, so the top bit rides as a
/// negative `i64` and the Python side reinterprets — the stream is a bit
/// channel, not a number channel.
fn push_f64(out: &mut Vec<i64>, values: &[f64]) {
    out.extend(values.iter().map(|v| v.to_bits() as i64));
}

// ------------------------------------------------------------------ M4 I/O

/// `n_draws` then, per draw: `code a b size replace weighted count values…`
///
/// `size` rides as `-1` for NumPy's scalar form, which is a *different* draw
/// from `size=1` and has to stay distinguishable — `Replay` refuses the swap,
/// and that refusal is most of what makes draw-site order a checked contract
/// rather than a hope (rewrite-plan §5).
fn read_draws(ints: &mut Ints) -> Result<Vec<RecordedDraw>, String> {
    let count = ints.n()?;
    let mut draws = Vec::with_capacity(count);
    for _ in 0..count {
        let code = ints.next()?;
        let method = Method::from_code(code).ok_or_else(|| format!("bad draw method {code}"))?;
        let a = ints.next()?;
        let b = ints.next()?;
        let raw_size = ints.next()?;
        let size = if raw_size < 0 {
            None
        } else {
            Some(raw_size as usize)
        };
        let replace = ints.next()? != 0;
        let weighted = ints.next()? != 0;
        let values = ints.n()?;
        let mut draw = RecordedDraw {
            method,
            a,
            b,
            size,
            replace,
            weighted,
            ints: Vec::new(),
            floats: Vec::new(),
        };
        match method {
            Method::Random => {
                for _ in 0..values {
                    draw.floats.push(f64::from_bits(ints.next()? as u64));
                }
            }
            _ => {
                for _ in 0..values {
                    draw.ints.push(ints.next()?);
                }
            }
        }
        draws.push(draw);
    }
    Ok(draws)
}

/// `weight has_prev action5 n_history state memory [oldest_state pairs…]`
///
/// Histories arrive as their **action pairs plus the oldest state**, exactly
/// as the corpus stores them (parity-corpus.md): eight particles with an
/// eight-deep history is two thirds of a megabyte of boards otherwise. The
/// intermediate states — and every frame's `observation_after` — are rebuilt
/// here through the transition kernel. That is not a shortcut around the
/// check; it *is* the check, because both sides rebuild from the same recipe
/// and a transition that disagreed anywhere would show up as a rejuvenation
/// that accepted a different set of histories.
fn read_particle(ints: &mut Ints, seat: usize) -> Result<Particle, String> {
    let weight = f64::from_bits(ints.next()? as u64);
    let has_prev = ints.next()? != 0;
    let prev_action = ints.action()?;
    let n_history = ints.n()?;
    let state = Rc::new(read_state(ints)?);
    let memory = Rc::new(read_memory(ints)?);

    let mut history = Vec::with_capacity(n_history);
    if n_history > 0 {
        let mut current = Rc::new(read_state(ints)?);
        let pairs: Vec<(Action5, Action5)> = (0..n_history)
            .map(|_| Ok((ints.action()?, ints.action()?)))
            .collect::<Result<_, String>>()?;
        for (my_action, enemy_action) in pairs {
            let mut actions: Actions = [[1, 0, 0, 0, 0]; 2];
            actions[seat] = my_action;
            actions[1 - seat] = enemy_action;
            let (next, _) = transition(&current, &actions);
            let next = Rc::new(next);
            history.push(Rc::new(HistoryFrame {
                state: Rc::clone(&current),
                my_action,
                enemy_action,
                observation_after: Rc::new(emit_observation(&next, seat)),
            }));
            current = next;
        }
    }

    Ok(Particle {
        state,
        weight,
        enemy_memory: memory,
        enemy_prev_action: if has_prev { Some(prev_action) } else { None },
        history,
    })
}

/// `seat collapsed n_particles_config n` then `n` particles.
fn read_belief(ints: &mut Ints) -> Result<BeliefState, String> {
    let seat = ints.n()?;
    let collapsed = ints.next()? != 0;
    let n_particles = ints.n()?;
    let count = ints.n()?;
    let mut particles = Vec::with_capacity(count);
    for _ in 0..count {
        particles.push(read_particle(ints, seat)?);
    }
    Ok(BeliefState {
        seat,
        particles,
        config: BeliefConfig {
            n_particles,
            ..Default::default()
        },
        collapsed,
    })
}

/// The answer side: `seat collapsed n` then per particle `weight has_prev
/// action5 history_len [per frame: my_action5 enemy_action5 time] state
/// memory`.
///
/// Frames go out as their action pair and the timestep of the state they sit
/// on, not as whole boards — the boards are a function of inputs both sides
/// already agree on. Emitting only the *length* was the first design and was
/// wrong: `_append_history` drops from the old end when the window overflows,
/// and swapping that for a truncation keeps the length identical while keeping
/// the wrong eight frames. Mutation testing found exactly that.
fn write_belief(out: &mut Vec<i64>, belief: &BeliefState) {
    out.push(belief.seat as i64);
    out.push(belief.collapsed as i64);
    out.push(belief.n() as i64);
    for particle in &belief.particles {
        out.push(particle.weight.to_bits() as i64);
        let prev = particle.enemy_prev_action.unwrap_or([1, 0, 0, 0, 0]);
        out.push(particle.enemy_prev_action.is_some() as i64);
        out.extend(prev.iter().map(|&v| v as i64));
        out.push(particle.history.len() as i64);
        for frame in &particle.history {
            out.extend(frame.my_action.iter().map(|&v| v as i64));
            out.extend(frame.enemy_action.iter().map(|&v| v as i64));
            out.push(frame.state.time as i64);
        }
        write_state(out, &particle.state);
        write_memory(out, &particle.enemy_memory);
    }
}

// ------------------------------------------------------------------ M5 I/O

/// `seat n` then per particle `weight_bits gr gc`.
///
/// A *lite* belief: the only thing the tactical layer reads out of one is
/// `believed_enemy_general`, which needs each particle's enemy general cell and
/// its weight and nothing else. Sending whole boards here would multiply the
/// stream by three orders of magnitude to check the same function. The full
/// belief still rides the `decide` surface, where the tensor needs it.
fn read_belief_lite(ints: &mut Ints) -> Result<BeliefState, String> {
    let seat = ints.n()?;
    let count = ints.n()?;
    let mut particles = Vec::with_capacity(count);
    for _ in 0..count {
        let weight = f64::from_bits(ints.next()? as u64);
        let gr = ints.next()? as i32;
        let gc = ints.next()? as i32;
        let mut state = GameState::empty(1, 1);
        state.general_positions[1 - seat] = [gr, gc];
        particles.push(Particle::new(
            Rc::new(state),
            weight,
            Rc::new(VisibleMemory::empty(1, 1)),
        ));
    }
    Ok(BeliefState {
        seat,
        particles,
        config: BeliefConfig::default(),
        collapsed: false,
    })
}

/// `has_prev action5 n_recent action5…`
fn read_action_history(ints: &mut Ints) -> Result<(Option<Action5>, Vec<Action5>), String> {
    let has_prev = ints.next()? != 0;
    let prev = ints.action()?;
    let count = ints.n()?;
    let mut recent = Vec::with_capacity(count);
    for _ in 0..count {
        recent.push(ints.action()?);
    }
    Ok((if has_prev { Some(prev) } else { None }, recent))
}

fn push_cell(out: &mut Vec<i64>, cell: Option<(usize, usize)>) {
    match cell {
        Some((r, c)) => out.extend([1, r as i64, c as i64]),
        None => out.extend([0, -1, -1]),
    }
}

fn push_action(out: &mut Vec<i64>, action: Option<Action5>) {
    match action {
        Some(action) => {
            out.push(1);
            out.extend(action.iter().map(|&v| v as i64));
        }
        None => out.extend([0, -1, -1, -1, -1, -1]),
    }
}

/// `morpheus-rs bench-belief`: time the belief update on recorded beliefs.
///
/// The same two calls the runtime charges to `belief_proposal` and
/// `particle_transitions`, on the same inputs the Python is timed on, so M4's
/// exit-gate ratio is a like-for-like measurement rather than a live-play
/// number compared against a microbenchmark. The RNG here is `SmallRng`, not
/// `Replay`: this is the *playing* path, and replaying a recorded stream would
/// time a vector read instead of a sample.
///
/// The timed unit is what `runtime.py` charges, not what is convenient: the
/// controller's `particle_transitions` block is `filter_step` **and**, when
/// nothing survived, `recover_belief`. Timing only the filter would report a
/// happy-path number against a baseline whose p99 is dominated by recovery.
/// Each case therefore carries several target observations — one the belief
/// can explain and one it cannot — and both are timed.
///
/// Input per case: a belief, `my_action`, a `VisibleMemory` for the recovery
/// path, then `n_targets` observations. Output: `case propose_ns` followed by
/// one `update_ns survivors` pair per target, minimum over `iters` repetitions.
/// The minimum because a per-case p99 over a handful of runs measures the
/// machine's scheduling noise; the distribution that matters is the one
/// *across* beliefs.
pub fn bench_belief<R: BufRead, W: Write>(
    reader: &mut R,
    writer: &mut W,
    iters: usize,
) -> Result<(), String> {
    use std::time::Instant;

    let mut ints = Ints::read_all(reader)?;
    let cases = ints.n()?;
    let mut rng = crate::support::rng::SmallRng::seed_from_u64(0x5eed);

    for case in 0..cases {
        let belief = read_belief(&mut ints)?;
        let my_action = ints.action()?;
        let memory = read_memory(&mut ints)?;
        let targets = ints.n()?;
        let mut observations = Vec::with_capacity(targets);
        for _ in 0..targets {
            observations.push(read_observation(&mut ints)?);
        }

        // One untimed pass so the first measurement is not paying for cold
        // pages in the freshly-read belief.
        let warm = propose_enemy_actions(&belief, &mut rng, None, 8, None);
        std::hint::black_box(warm.len());

        let mut propose_ns = u128::MAX;
        for _ in 0..iters.max(1) {
            let t0 = Instant::now();
            let actions = propose_enemy_actions(&belief, &mut rng, None, 8, None);
            propose_ns = propose_ns.min(t0.elapsed().as_nanos());
            std::hint::black_box(actions.len());
        }
        let actions = propose_enemy_actions(&belief, &mut rng, None, 8, None);

        let mut line = format!("{case} {propose_ns}");
        for real_obs in &observations {
            let mut update_ns = u128::MAX;
            let mut survivors = 0usize;
            for _ in 0..iters.max(1) {
                let t = Instant::now();
                let mut next = filter_step(&belief, my_action, real_obs, &actions, &mut rng)?;
                if !next.particles.iter().any(|p| p.weight > 0.0) {
                    next = recover_belief(&belief, real_obs, &memory, &mut rng, None);
                }
                update_ns = update_ns.min(t.elapsed().as_nanos());
                survivors = next.particles.iter().filter(|p| p.weight > 0.0).count();
                std::hint::black_box(next.n());
            }
            line.push_str(&format!(" {update_ns} {survivors}"));
        }
        writeln!(writer, "{line}").map_err(|e| format!("case {case}: {e}"))?;
    }
    writer.flush().map_err(|e| format!("flush: {e}"))?;
    Ok(())
}

/// A scripted prior with a leaf value that depends on the leaf.
///
/// M6's correction to the `search` surface, and the second half of the reason
/// it could not see a child. `ScriptedEvaluator` returns one constant value for
/// every leaf, and a constant value makes the whole enemy mixture *unobservable*
/// — every `q` entry is the same number, so the weights it is averaged with
/// cannot change the result. The enemy-hash cache, the reservoir weights and
/// the marginal aggregation were all invisible for that reason alone.
///
/// The value stays a deterministic function of the leaf's own observation
/// payload — an integer sum reduced to `[-1, 1]` — so it varies without
/// reintroducing what the scripted evaluator exists to keep out: no network, no
/// softmax, and the same double on both sides for the same board.
struct VaryingEvaluator {
    inner: crate::search::ScriptedEvaluator,
}

impl crate::search::SearchEvaluator for VaryingEvaluator {
    fn evaluate(
        &mut self,
        obs: &Observation,
        memory: &VisibleMemory,
        belief: &crate::belief::BeliefState,
        from_root: bool,
        shape: bool,
    ) -> (Vec<f64>, f64) {
        let (prior, _) = self.inner.evaluate(obs, memory, belief, from_root, shape);
        let payload = observation_payload(obs);
        let sum: u64 = payload.iter().map(|&b| b as u64).sum();
        let value = ((sum % 2001) as f64 - 1000.0) / 1000.0;
        (prior, value)
    }
}

pub fn run<R: BufRead, W: Write>(kind: &str, reader: &mut R, writer: &mut W) -> Result<(), String> {
    let mut ints = Ints::read_all(reader)?;
    let cases = ints.n()?;
    let mut out: Vec<i64> = Vec::new();
    // Loading the artifact costs a few milliseconds and only the `net` kind
    // needs it, so it is paid on first use rather than on every subcommand.
    let mut net_session: Option<crate::nn::inference::Session> = None;

    for case in 0..cases {
        out.clear();
        match kind {
            // state + both actions -> next state + info
            "transition" => {
                let state = read_state(&mut ints)?;
                let actions: Actions = [ints.action()?, ints.action()?];
                let (next, info) = transition(&state, &actions);
                write_state(&mut out, &next);
                out.extend([
                    info.army[0],
                    info.army[1],
                    info.land[0],
                    info.land[1],
                    info.is_done as i64,
                    info.winner as i64,
                    info.time as i64,
                ]);
            }
            // state + both actions -> which seat resolves first
            //
            // Its own surface rather than a detail of `transition`, because
            // move order is mostly *unobservable* downstream: when the two
            // moves do not interact, either order gives the same board. A
            // mutation that broke the NumPy index wrap here survived the
            // end-to-end transition check over hundreds of recorded positions
            // and is caught immediately by this one.
            "order" => {
                let state = read_state(&mut ints)?;
                let actions: Actions = [ints.action()?, ints.action()?];
                out.push(determine_move_order(&state, &actions) as i64);
            }
            // state + seat -> the fogged observation that seat receives
            "observe" => {
                let state = read_state(&mut ints)?;
                let seat = ints.n()?;
                write_observation(&mut out, &emit_observation(&state, seat));
            }
            // observation + memory -> the 3970-long legal mask
            "mask" => {
                let obs = read_observation(&mut ints)?;
                let memory = read_memory(&mut ints)?;
                let mask = legal_mask(&obs, &memory, None);
                out.reserve(N_ACTIONS);
                out.extend(mask.iter().map(|&v| v as i64));
            }
            // observation + memory -> the live build-cost grid
            "cost" => {
                let obs = read_observation(&mut ints)?;
                let memory = read_memory(&mut ints)?;
                let cost = live_build_cost(&obs, &memory);
                out.extend(cost[..obs.h * obs.w].iter().map(|&v| v as i64));
            }
            // observation + memory -> memory after folding the observation in
            "memory" => {
                let obs = read_observation(&mut ints)?;
                let memory = read_memory(&mut ints)?;
                write_memory(&mut out, &update_memory(&memory, &obs));
            }
            // observation + memory + prev-history digest + action -> every
            // digest the search keys on
            "hash" => {
                let obs = read_observation(&mut ints)?;
                let memory = read_memory(&mut ints)?;
                let mut prev = [0u8; 32];
                for slot in prev.iter_mut() {
                    *slot = ints.next()? as u8;
                }
                let action = ints.action()?;
                let payload = observation_payload(&obs);
                let mem_digest = memory_digest(&memory);
                for digest in [
                    mem_digest,
                    info_state_key_prehashed(obs.turn, &mem_digest, &payload, &prev),
                    enemy_info_hash_prehashed(&payload, &mem_digest),
                    child_edge_key(action, &payload),
                    roll_history_digest(&prev, action, &payload),
                ] {
                    out.extend(digest.iter().map(|&b| b as i64));
                }
            }
            // observation + memory + belief summary + previous action -> the
            // 49x21x21 tensor, as f32 bit patterns
            "tensor" => {
                let obs = read_observation(&mut ints)?;
                let memory = read_memory(&mut ints)?;
                let n = obs.h * obs.w;
                let belief = BeliefSummary {
                    enemy_owner: ints.floats(n)?,
                    enemy_army_mean: ints.floats(n)?,
                    enemy_army_std: ints.floats(n)?,
                    enemy_general: ints.floats(n)?,
                    enemy_castle_owner: ints.floats(n)?,
                    enemy_visibility: ints.floats(n)?,
                    ess_fraction: ints.floats(1)?[0],
                };
                let has_prev = ints.next()? != 0;
                let action = ints.action()?;
                let tensor = build_tensor(
                    &obs,
                    &memory,
                    Some(&belief),
                    if has_prev { Some(action) } else { None },
                    ARMY_SCALE,
                );
                out.extend(tensor.iter().map(|&v| v.to_bits() as i64));
            }
            // symmetry index -> the coordinate map, direction map, and the
            // permutation it induces on the 3970 policy logits
            "symmetry" => {
                let which = ints.n()?;
                let sym = symmetry::ALL[which % symmetry::ALL.len()];
                for r in 0..PAD {
                    for c in 0..PAD {
                        let (nr, nc) = sym.transform_rc(r, c);
                        out.push(nr as i64);
                        out.push(nc as i64);
                    }
                }
                for d in 0..4 {
                    out.push(sym.transform_dir(d) as i64);
                }
                // Where each logit index lands. A permutation compares as
                // integers, so this stays tier 1 even though it serves the
                // float tensor path.
                for index in 0..N_ACTIONS {
                    let action = decode_action(index).unwrap_or([1, 0, 0, 0, 0]);
                    out.push(encode_action(symmetry::transform_action(action, sym)) as i64);
                }
            }
            // a 49×441 tensor -> every head, through all three entry points
            //
            // Running `Policy`, `PolicyWdl`, and `All` on the same tensor and
            // emitting all three answers costs two extra forwards per case and
            // checks something no single entry point can: that the switch in
            // `forward_into` returns early without changing what it already
            // wrote. The Python side runs its three separate TorchScript
            // modules, so the comparison is entry point against entry point.
            "net" => {
                let session = net_session.get_or_insert_with(|| {
                    crate::nn::inference::Session::load_default()
                        .unwrap_or_else(|e| panic!("loading the artifact: {e}"))
                });
                let tensor = ints.floats(network::IN_CHANNELS * network::CELLS)?;

                let policy_only = session.forward(&tensor, network::Heads::Policy).clone();
                push_f32(&mut out, &policy_only.policy);
                push_f32(&mut out, &[policy_only.pass_logit]);

                let pw = session.forward(&tensor, network::Heads::PolicyWdl).clone();
                push_f32(&mut out, &pw.policy);
                push_f32(&mut out, &[pw.pass_logit]);
                push_f32(&mut out, &pw.wdl_logits);

                let all = session.forward(&tensor, network::Heads::All);
                push_f32(&mut out, &all.policy);
                push_f32(&mut out, &[all.pass_logit]);
                push_f32(&mut out, &all.wdl_logits);
                push_f32(&mut out, &all.hidden_owner);
                push_f32(&mut out, &all.enemy_army_bins);
                push_f32(&mut out, &all.enemy_general);
                push_f32(&mut out, &all.hidden_castle);
                push_f32(
                    &mut out,
                    &[
                        all.land_margin,
                        all.army_margin,
                        all.castle_margin,
                        all.turns_to_termination,
                    ],
                );
            }
            // logits + mask + WDL -> the legal-normalized prior and the backup
            // value, in f64
            //
            // The arithmetic between the network and the search, split out
            // from the network itself so a divergence lands on one of them.
            // It is also the only surface that checks the flat policy layout
            // end to end: a channel-major/row-major swap leaves every head
            // bit-identical and moves every prior mass to the wrong action.
            "prior" => {
                let logits = ints.floats(network::N_ACTIONS)?;
                let mask_ints = ints.ints(network::N_ACTIONS)?;
                let mask: Vec<bool> = mask_ints.iter().map(|v| *v != 0).collect();
                let wdl = ints.floats(3)?;
                let from_root = ints.next()? != 0;
                let prior = network::legal_normalized_policy(&logits, &mask);
                push_f64(&mut out, &prior);
                push_f64(
                    &mut out,
                    &[network::backup_value([wdl[0], wdl[1], wdl[2]], from_root)],
                );
            }
            // logits + mask + k -> the masked softmax, the singleton fallback,
            // and the ranked candidate list
            //
            // The corner of the proposal M4 ports but nothing else reaches.
            // `softmax_masked` runs only under the policy proposal, which
            // `deployment.json` ships turned **off**; and `top_legal_actions`
            // is otherwise exercised through recovery on a *uniform*
            // distribution, where every legal action ties and the check is
            // therefore only about NumPy's tie-breaking. Feeding real logits
            // here checks the ranking itself, and covers the branch that would
            // have to work before M7 could re-qualify `use_policy_proposal`.
            "toplegal" => {
                let logits = ints.f64s(N_ACTIONS)?;
                let mask_ints = ints.ints(N_ACTIONS)?;
                let mask: Vec<bool> = mask_ints.iter().map(|v| *v != 0).collect();
                let k = ints.n()?;

                let probs = softmax_masked(&logits, &mask);
                let nonzero: Vec<usize> = probs
                    .iter()
                    .enumerate()
                    .filter(|(_, &p)| p != 0.0)
                    .map(|(i, _)| i)
                    .collect();
                out.push(nonzero.len() as i64);
                for index in &nonzero {
                    out.push(*index as i64);
                    out.push(probs[*index].to_bits() as i64);
                }
                let singles = singleton_probs(&mask);
                out.push(singles.iter().position(|&p| p != 0.0).unwrap_or(0) as i64);

                let top = top_legal_actions(&probs, &mask, k);
                out.push(top.len() as i64);
                for action in &top {
                    out.extend(action.iter().map(|&v| v as i64));
                }
            }
            // a vector of f64 -> its NumPy pairwise sum
            //
            // A one-line surface guarding a two-line function, because the
            // function is not portable by construction: `np.sum` reduces in
            // eight interleaved lanes, and a host whose NumPy vectorizes that
            // reduction differently would change the ESS, which would change
            // whether a resample happened, which would desynchronize the
            // replay stream three calls later. Checking the sum directly turns
            // that into one named failure instead.
            "npsum" => {
                let count = ints.n()?;
                let values = ints.f64s(count)?;
                push_f64(&mut out, &[npsum(&values)]);
            }
            // a vector of f64 -> `np.argsort(-scores)`
            //
            // `top_legal_actions` ranks a distribution that, on the deployed
            // uniform proposal, is entirely ties — so the recovery candidate
            // order is decided by NumPy's introsort internals and by nothing
            // else. See `rng::argsort_desc_numpy` for why this surface is
            // host-conditional and why that is the oracle's property, not the
            // port's.
            "argsort" => {
                let count = ints.n()?;
                let values = ints.f64s(count)?;
                // Length first: an empty permutation is a legitimate answer,
                // and a bare empty line would be dropped as blank by the
                // harness's line reader rather than compared.
                out.push(count as i64);
                out.extend(argsort_desc_numpy(&values).iter().map(|&i| i as i64));
            }
            // first-frame observation + seat + config + draws -> the initial
            // particle set, or a refusal
            //
            // The corpus cannot reach this by replay: it records beliefs that
            // already exist, never the frame that created one. Mutation
            // testing found the consequence — deleting the minimum-separation
            // rule and deleting the "never a cell we can see" rule both
            // survived every other surface, because nothing called
            // `legal_enemy_general_candidates` at all.
            "initbelief" => {
                let obs = read_observation(&mut ints)?;
                let seat = ints.n()?;
                let n_particles = ints.n()?;
                let min_general_distance = ints.next()? as i32;
                let mut rng = Replay::new(read_draws(&mut ints)?);
                let config = BeliefConfig {
                    n_particles,
                    min_general_distance,
                    ..Default::default()
                };
                // The candidate set is emitted alongside the belief: it is the
                // thing the prior rules actually decide, and a belief sampled
                // from it can agree by luck when the support does not.
                match crate::belief::legal_enemy_general_candidates(&obs, min_general_distance) {
                    Ok(candidates) => {
                        out.push(candidates.len() as i64);
                        for (r, c) in &candidates {
                            out.push(*r as i64);
                            out.push(*c as i64);
                        }
                    }
                    Err(_) => out.push(-1),
                }
                match crate::belief::initialize_belief(&obs, seat, &mut rng, config) {
                    Ok(belief) => {
                        out.push(1);
                        write_belief(&mut out, &belief);
                    }
                    Err(_) => out.push(0),
                }
                out.push(rng.consumed() as i64);
            }
            // a belief -> the six belief planes, plus ESS in full precision
            //
            // The planes narrow to f32 exactly where the Python's `.astype`
            // does; `ess` and `ess_fraction` ride out in f64 so a precision
            // bug in the aggregation cannot hide behind the narrowing.
            "summary" => {
                let belief = read_belief(&mut ints)?;
                let summary = summarize_belief(&belief);
                for plane in [
                    &summary.enemy_owner,
                    &summary.enemy_army_mean,
                    &summary.enemy_army_std,
                    &summary.enemy_general,
                    &summary.enemy_castle_owner,
                    &summary.enemy_visibility,
                ] {
                    out.push(plane.len() as i64);
                    push_f32(&mut out, plane);
                }
                push_f32(&mut out, &[summary.ess_fraction]);
                push_f64(&mut out, &[ess(&belief.weights()), ess_fraction(&belief)]);
            }
            // a belief + a recorded draw stream -> each particle's proposal
            // distribution and the action sampled from it
            //
            // Both halves, deliberately. `Replay` hands back the oracle's
            // index whatever distribution this side computed, so checking only
            // the sampled actions would pass even if every probability were
            // wrong. The distribution rides out **sparsely** — a uniform mask
            // has a few hundred non-zeros out of 3,970 — which keeps the
            // channel honest without making it enormous.
            "propose" => {
                let belief = read_belief(&mut ints)?;
                let mut rng = Replay::new(read_draws(&mut ints)?);
                let mut telemetry = ProposalTelemetry::default();

                for particle in &belief.particles {
                    let enemy_obs = emit_observation(&particle.state, belief.enemy_seat());
                    let enemy_mem = update_memory(&particle.enemy_memory, &enemy_obs);
                    let probs = uniform_legal_probs(&enemy_obs, &enemy_mem);
                    let nonzero: Vec<usize> = probs
                        .iter()
                        .enumerate()
                        .filter(|(_, &p)| p != 0.0)
                        .map(|(i, _)| i)
                        .collect();
                    out.push(nonzero.len() as i64);
                    for index in &nonzero {
                        out.push(*index as i64);
                        out.push(probs[*index].to_bits() as i64);
                    }
                }

                let actions =
                    propose_enemy_actions(&belief, &mut rng, None, 8, Some(&mut telemetry));
                for action in &actions {
                    out.extend(action.iter().map(|&v| v as i64));
                }
                out.extend([
                    telemetry.n_particles as i64,
                    telemetry.n_singleton_particles as i64,
                    telemetry.n_unique_info_keys as i64,
                    telemetry.n_unique_policy_inputs as i64,
                    telemetry.n_policy_batches as i64,
                    rng.consumed() as i64,
                ]);
            }
            // belief + my action + the real frame + one enemy action per
            // particle + draws -> the filtered belief
            "filter" => {
                let belief = read_belief(&mut ints)?;
                let my_action = ints.action()?;
                let real_obs = read_observation(&mut ints)?;
                let mut enemy_actions = Vec::with_capacity(belief.n());
                for _ in 0..belief.n() {
                    enemy_actions.push(ints.action()?);
                }
                let mut rng = Replay::new(read_draws(&mut ints)?);
                let next = filter_step(&belief, my_action, &real_obs, &enemy_actions, &mut rng)?;
                write_belief(&mut out, &next);
                out.push(rng.consumed() as i64);
            }
            // a belief carrying histories + draws -> the rejuvenated belief
            "rejuvenate" => {
                let belief = read_belief(&mut ints)?;
                let mut rng = Replay::new(read_draws(&mut ints)?);
                write_belief(&mut out, &rejuvenate(&belief, &mut rng, None));
                out.push(rng.consumed() as i64);
            }
            // observation + seat + memory + particle count + draws -> the
            // maximum-entropy reconstruction, or a refusal
            //
            // The refusal is part of the contract: the Python raises
            // `ValueError` on an inconsistent frame and `recover_belief`
            // catches it to keep the last valid belief. A port that
            // reconstructed something anyway would look correct here and be
            // wrong in play, so the `ok` flag is compared before the belief.
            "maxent" => {
                let obs = read_observation(&mut ints)?;
                let seat = ints.n()?;
                let memory = read_memory(&mut ints)?;
                let n_particles = ints.n()?;
                let mut rng = Replay::new(read_draws(&mut ints)?);
                let config = BeliefConfig {
                    n_particles,
                    ..Default::default()
                };
                match maximum_entropy_reconstruction(&obs, seat, &memory, &mut rng, config) {
                    Ok(belief) => {
                        out.push(1);
                        write_belief(&mut out, &belief);
                    }
                    Err(_) => out.push(0),
                }
                out.push(rng.consumed() as i64);
            }
            // capacity + arrivals + a belief + draws -> the reservoir after a
            // fixed script: admit everything twice, sample once, then replace
            // from the belief
            //
            // The script is fixed rather than corpus-driven because nothing
            // the reservoir does depends on which board is in a particle. What
            // it depends on is arrival *count* against capacity, which the
            // double pass crosses, and the fixed-seed resample in
            // `replace_from_belief` — the one place in the belief layer where
            // the Python's generator is a local, and where play-time answers
            // legitimately differ (see `reservoir.rs`).
            "reservoir" => {
                let capacity = ints.n()?;
                let seat = ints.n()?;
                let arrivals = ints.n()?;
                let mut incoming = Vec::with_capacity(arrivals);
                for _ in 0..arrivals {
                    incoming.push(read_particle(&mut ints, seat)?);
                }
                let belief = read_belief(&mut ints)?;
                let mut rng = Replay::new(read_draws(&mut ints)?);

                let mut reservoir = ParticleReservoir::new(capacity);
                for _ in 0..2 {
                    for particle in &incoming {
                        reservoir.admit(particle, &mut rng);
                    }
                }
                out.extend([
                    reservoir.admitted_count as i64,
                    reservoir.version as i64,
                    reservoir.n() as i64,
                ]);
                for particle in &reservoir.particles {
                    out.push(particle.weight.to_bits() as i64);
                    write_state(&mut out, &particle.state);
                }

                match reservoir.sample(&mut rng) {
                    Ok(particle) => {
                        out.push(1);
                        write_state(&mut out, &particle.state);
                    }
                    Err(_) => out.push(0),
                }

                reservoir.replace_from_belief(&belief, &mut rng);
                out.extend([
                    reservoir.admitted_count as i64,
                    reservoir.version as i64,
                    reservoir.n() as i64,
                ]);
                for particle in &reservoir.particles {
                    out.push(particle.weight.to_bits() as i64);
                    write_state(&mut out, &particle.state);
                }
                out.push(rng.consumed() as i64);
            }
            // observation + memory -> the play mask, plus the two sub-rules
            // that shape it
            //
            // Emitted alongside the mask because both are *subtractive*: the
            // garrison floor and the castle anchor each clear bits and then
            // withdraw entirely if that would leave no non-pass action, so a
            // port that never applied either would produce a mask identical to
            // `legal_mask` on most frames and differ only where it matters.
            // The counts make "how many bits did each rule remove" a number.
            "playmask" => {
                let obs = read_observation(&mut ints)?;
                let memory = read_memory(&mut ints)?;
                let mask = tactics::play_mask(&obs, &memory, None);
                let base = legal_mask(&obs, &memory, None);
                out.extend(mask.iter().map(|&v| v as i64));
                out.push(
                    base.iter()
                        .zip(mask.iter())
                        .filter(|(&b, &m)| b && !m)
                        .count() as i64,
                );
                out.push(tactics::enemy_is_visible(&obs, &memory) as i64);
                let own_total: i64 = (0..obs.h * obs.w)
                    .filter(|&i| obs.owner_grid[i] as i32 == 1)
                    .map(|i| obs.army_grid[i] as i64)
                    .sum();
                out.push(tactics::garrison_floor(own_total));
                out.push(tactics::max_threat_arrival(&obs, &memory));
            }
            // observation + memory + lite belief + previous action + a network
            // prior -> the heuristic scores and the blended prior
            //
            // rewrite-plan §5 budgets 1e-9 for these in f64. Both sides are in
            // f64 throughout and they agree to the **last bit**, so that is
            // what the harness enforces, following M2's precedent: a tolerance
            // nothing approaches is a check that cannot fail.
            "shaping" => {
                let obs = read_observation(&mut ints)?;
                let memory = read_memory(&mut ints)?;
                let belief = read_belief_lite(&mut ints)?;
                let (prev, _recent) = read_action_history(&mut ints)?;
                let prior = ints.f64s(N_ACTIONS)?;
                let lam = ints.f64s(1)?[0];
                let log_clip = ints.f64s(1)?[0];
                let floor_frac = ints.f64s(1)?[0];
                // The mask is recomputed on both sides rather than sent: it is
                // 3,970 integers a case, and `playmask` already proves it.
                let mask = tactics::play_mask(&obs, &memory, None);
                let scores = tactics::heuristic_action_scores(
                    &obs,
                    &memory,
                    &mask,
                    Some(&belief),
                    prev,
                );
                push_f64(&mut out, &scores);
                push_f64(
                    &mut out,
                    &tactics::blend_prior(&prior, &scores, &mask, lam, log_clip, floor_frac, 0.0),
                );
            }
            // observation + memory + prior + widening limit -> the mandatory
            // list and the ordered candidate list
            "candidates" => {
                let obs = read_observation(&mut ints)?;
                let memory = read_memory(&mut ints)?;
                let prior = ints.f64s(N_ACTIONS)?;
                let limit = ints.n()?;
                let use_play_mask = ints.next()? != 0;
                let mask = if use_play_mask {
                    tactics::play_mask(&obs, &memory, None)
                } else {
                    legal_mask(&obs, &memory, None)
                };
                let mandatory = tactics::mandatory_action_indices(&obs, &memory, &mask);
                out.push(mandatory.len() as i64);
                out.extend(mandatory.iter().map(|&i| i as i64));
                let candidates =
                    tactics::policy_ordered_candidates(&prior, &mask, &mandatory, limit);
                out.push(candidates.len() as i64);
                out.extend(candidates.iter().map(|&i| i as i64));
            }
            // observation + memory + lite belief -> every planner's answer
            //
            // One surface for the whole battery because they share their
            // expensive input (a BFS field) and because a planner that returns
            // `None` everywhere is the failure mode worth catching: each answer
            // rides with an explicit present/absent flag rather than a sentinel
            // that could be confused with a real cell.
            "planners" => {
                let obs = read_observation(&mut ints)?;
                let memory = read_memory(&mut ints)?;
                let belief = read_belief_lite(&mut ints)?;

                push_cell(&mut out, tactics::own_general_cell(&obs, &memory));
                push_cell(&mut out, tactics::known_enemy_general_cell(&obs, &memory));
                push_cell(&mut out, tactics::king_cell(&obs, None));
                push_cell(&mut out, tactics::believed_enemy_general(Some(&belief)));
                push_cell(&mut out, tactics::enemy_seek_target(&obs, &memory, Some(&belief)));
                let goals = tactics::seek_goals(&obs, &memory, Some(&belief));
                out.push(goals.len() as i64);
                for (r, c) in &goals {
                    out.extend([*r as i64, *c as i64]);
                }
                let exclude = tactics::own_general_cell(&obs, &memory);
                push_cell(
                    &mut out,
                    tactics::wave_assembly_cell(&obs, &memory, Some(&belief), exclude),
                );
                let (share, max_own, total) = tactics::army_concentration(&obs, exclude);
                push_f64(&mut out, &[share]);
                out.extend([max_own, total]);
                out.push(tactics::structure_idle_army(&obs, &memory));
                out.push(tactics::max_threat_arrival(&obs, &memory));

                let site = tactics::castle_build_site(&obs, &memory);
                push_cell(&mut out, site);
                match site {
                    Some(site) => {
                        push_action(&mut out, tactics::castle_tithe_move(&obs, &memory, site))
                    }
                    None => push_action(&mut out, None),
                }

                let threat = tactics::general_threat(&obs, &memory);
                match threat {
                    Some((cell, d)) => {
                        out.extend([1, cell.0 as i64, cell.1 as i64, d as i64]);
                        push_action(
                            &mut out,
                            tactics::defend_general_move(&obs, &memory, (cell, d)),
                        );
                    }
                    None => {
                        out.extend([0, -1, -1, -1]);
                        push_action(&mut out, None);
                    }
                }

                match tactics::kill_plan(&obs, &memory) {
                    Some((steps, margin, first)) => {
                        out.extend([1, steps as i64, margin]);
                        push_action(&mut out, Some(first));
                    }
                    None => {
                        out.extend([0, -1, -1]);
                        push_action(&mut out, None);
                    }
                }

                // The two whole-board fields the scoring path rides on.
                let reveal = tactics::reveal_count_grid(&obs);
                out.extend(reveal.iter().copied());
                let goal_cells: Vec<(usize, usize)> = goals.clone();
                let field = tactics::path_distance_field(&obs, &goal_cells);
                out.extend(field.dist.iter().map(|&v| v as i64));
            }
            // observation + memory + a chosen action + history + an optional
            // prior -> the action the hard rules commit
            //
            // The belief is deliberately absent: `constrain_nn_action` takes one
            // and never reads it (see `tactics.rs`), which contradicts
            // rewrite-plan §5's "final-state wrinkle". Passing one here would
            // paper over that.
            "constrain" => {
                let obs = read_observation(&mut ints)?;
                let memory = read_memory(&mut ints)?;
                let chosen = ints.action()?;
                let (prev, recent) = read_action_history(&mut ints)?;
                let has_prior = ints.next()? != 0;
                let prior = ints.f64s(N_ACTIONS)?;
                let action = tactics::constrain_nn_action(
                    &obs,
                    &memory,
                    chosen,
                    prev,
                    &recent,
                    if has_prior { Some(&prior) } else { None },
                    None,
                );
                out.extend(action.iter().map(|&v| v as i64));
                // The oscillation predicate on the chosen move, both ways: with
                // the precomputed reveal grid the redirect loop uses, and with
                // the whole-board dilation the Python calls. They must agree.
                let reveal = tactics::reveal_count_grid(&obs);
                out.push(
                    tactics::blocks_oscillation(chosen, prev, &obs, &recent, false, None) as i64,
                );
                out.push(tactics::blocks_oscillation(
                    chosen,
                    prev,
                    &obs,
                    &recent,
                    false,
                    Some(&reveal),
                ) as i64);
            }
            // strategy vectors and one joint matrix -> the whole regret cycle
            //
            // The one surface in the port with a **tolerance on an integer-free
            // path**, and the reason is the oracle: NumPy sends `@` on f64 to
            // BLAS, whose reduction order is the vendor's. See `matrix.rs`.
            "matrix" => {
                let n_a = ints.n()?;
                let n_b = ints.n()?;
                let n_hashes = ints.n()?;
                let n = ints.next()?;
                let regrets = ints.f64s(n_a)?;
                let prior = ints.f64s(n_a)?;
                let avg_strategy = ints.f64s(n_a)?;
                let marginal_visits = ints.f64s(n_a)?;
                let first_play = ints.f64s(1)?[0];
                let enemy_weights = ints.f64s(n_hashes)?;
                let mut enemy_regrets = Vec::with_capacity(n_hashes);
                let mut enemy_priors = Vec::with_capacity(n_hashes);
                let mut visits = Vec::with_capacity(n_hashes);
                let mut q = Vec::with_capacity(n_hashes);
                for _ in 0..n_hashes {
                    enemy_regrets.push(ints.f64s(n_b)?);
                    enemy_priors.push(ints.f64s(n_b)?);
                    visits.push(ints.f64s(n_a * n_b)?);
                    q.push(ints.f64s(n_a * n_b)?);
                }
                let a_idx = ints.n()?;
                let b_idx = ints.n()?;
                let leaf_value = ints.f64s(1)?[0];

                out.extend([
                    matrix::self_widening_limit(n) as i64,
                    matrix::enemy_widening_limit(n) as i64,
                ]);
                push_f64(&mut out, &[matrix::exploration_epsilon(n)]);

                let sigma_self = matrix::mixed_strategy(&regrets, &prior, n);
                push_f64(&mut out, &sigma_self);
                let mut enemy_sigmas = Vec::with_capacity(n_hashes);
                let mut q_eff_list = Vec::with_capacity(n_hashes);
                for h in 0..n_hashes {
                    let sigma_b = matrix::mixed_strategy(&enemy_regrets[h], &enemy_priors[h], n);
                    push_f64(&mut out, &sigma_b);
                    let q_eff = matrix::effective_q(&visits[h], &q[h], first_play);
                    push_f64(&mut out, &q_eff);
                    enemy_sigmas.push(sigma_b);
                    q_eff_list.push(q_eff);
                }
                let (u_self, v) = matrix::aggregate_self_utilities(
                    &sigma_self,
                    &enemy_weights,
                    &enemy_sigmas,
                    &q_eff_list,
                );
                push_f64(&mut out, &u_self);
                push_f64(&mut out, &[v]);
                let (u_h, u_enemy, v_h) =
                    matrix::matrix_utilities(&sigma_self, &enemy_sigmas[0], &q_eff_list[0]);
                push_f64(&mut out, &u_h);
                push_f64(&mut out, &u_enemy);
                push_f64(&mut out, &[v_h]);
                push_f64(
                    &mut out,
                    &matrix::regret_plus_update(&regrets, &u_self, v, true),
                );
                push_f64(
                    &mut out,
                    &matrix::regret_plus_update(&enemy_regrets[0], &u_enemy, v, false),
                );

                let mut visits_0 = visits[0].clone();
                let mut value_sum = vec![0.0f64; n_a * n_b];
                let mut q_0 = q[0].clone();
                matrix::apply_joint_backup(
                    &mut visits_0,
                    &mut value_sum,
                    &mut q_0,
                    n_b,
                    a_idx,
                    b_idx,
                    leaf_value,
                );
                push_f64(&mut out, &visits_0);
                push_f64(&mut out, &value_sum);
                push_f64(&mut out, &q_0);

                push_f64(&mut out, &matrix::normalize_average_strategy(&avg_strategy));
                out.push(matrix::select_root_action(
                    &avg_strategy,
                    &marginal_visits,
                    &prior,
                    None,
                ) as i64);
            }
            // the recorded root tensor + observation + memory + history ->
            // the action the bot commits with zero completed simulations
            //
            // rewrite-plan §5's tier 3, in the form a single frame can answer.
            // The full-search decision depends on cross-turn state (tree,
            // history digest, estimator windows) that no one frame carries, but
            // the *no-search* decision is a whole deployed path — M0 measured
            // the belief update alone overrunning the deadline at p99, so this
            // is what the bot plays whenever search does not complete — and §5
            // requires it to agree **exactly**.
            //
            // The tensor comes off the wire rather than being rebuilt, so a
            // divergence lands on the decision layer instead of on the tensor
            // builder that `tensor` already checks.
            "decide" => {
                let session = net_session.get_or_insert_with(|| {
                    crate::nn::inference::Session::load_default()
                        .unwrap_or_else(|e| panic!("loading the artifact: {e}"))
                });
                let obs = read_observation(&mut ints)?;
                let memory = read_memory(&mut ints)?;
                let belief = read_belief_lite(&mut ints)?;
                let tensor = ints.floats(network::IN_CHANNELS * network::CELLS)?;
                let (prev, recent) = read_action_history(&mut ints)?;
                let lam_pre = ints.f64s(1)?[0];
                let lam_post = ints.f64s(1)?[0];
                let log_clip = ints.f64s(1)?[0];
                let floor_frac = ints.f64s(1)?[0];

                let logits = session
                    .forward(&tensor, network::Heads::PolicyWdl)
                    .flat_logits();
                let mask = tactics::play_mask(&obs, &memory, None);
                let unshaped = network::legal_normalized_policy(&logits, &mask);
                let lam = if tactics::enemy_is_visible(&obs, &memory) {
                    lam_post
                } else {
                    lam_pre
                };
                let shaped = tactics::apply_pre_contact_prior(
                    &unshaped,
                    &obs,
                    &memory,
                    &mask,
                    Some(&belief),
                    lam,
                    log_clip,
                    floor_frac,
                    0.0,
                    prev,
                );
                let fallback = crate::runtime::highest_prior_legal(&shaped, &mask);
                let action = tactics::constrain_nn_action(
                    &obs,
                    &memory,
                    fallback,
                    prev,
                    &recent,
                    Some(&shaped),
                    None,
                );
                out.extend(fallback.iter().map(|&v| v as i64));
                out.extend(action.iter().map(|&v| v as i64));
                // The tie margin: how much shaped prior separates the committed
                // action from the runner-up. A divergence is only acceptable if
                // this is inside the prior's measured agreement, and the number
                // is what makes that judgeable instead of arguable.
                let mut ranked: Vec<f64> = (0..mask.len())
                    .filter(|&i| mask[i])
                    .map(|i| shaped[i])
                    .collect();
                ranked.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                let margin = if ranked.len() >= 2 {
                    ranked[0] - ranked[1]
                } else {
                    1.0
                };
                push_f64(&mut out, &[margin]);
            }
            // a root frame + a scripted evaluator + a recorded draw stream ->
            // the whole tree after N batches
            //
            // The surface that proves the *search*, which no other one reaches:
            // `decide` runs at zero simulations by design, and `matrix` checks
            // the arithmetic without the storage that feeds it. Here the tree
            // is built for real — selection, progressive widening, enemy-table
            // installation and eviction, leaf expansion, backup — and every
            // statistic it accumulates is compared.
            //
            // The evaluator is *scripted*, not the network. The two engines'
            // priors agree to 6.6e-7, which is enough to reorder a near-tie in
            // the candidate list, and a search comparison that could fail on
            // the last bit of a softmax would prove nothing about the search.
            // With identical priors on both sides, any disagreement here is the
            // tree's.
            "search" => {
                let obs = read_observation(&mut ints)?;
                let memory = read_memory(&mut ints)?;
                let belief = read_belief(&mut ints)?;
                let prior = ints.f64s(N_ACTIONS)?;
                let value = ints.f64s(1)?[0];
                let config = crate::search::SearchConfig {
                    depth: ints.n()?,
                    pending_batch: ints.n()?,
                    n_particles: ints.n()?,
                    max_nodes: ints.n()?,
                    max_enemy_tables: ints.n()?,
                    ..Default::default()
                };
                let batches = ints.n()?;
                let freeze = ints.next()? != 0;
                let vary = ints.next()? != 0;
                let rng = crate::support::rng::SharedRng::new(Box::new(Replay::new(read_draws(&mut ints)?)));
                let scripted = crate::search::ScriptedEvaluator { prior, value };
                let mut evaluator: Box<dyn crate::search::SearchEvaluator> = if vary {
                    Box::new(VaryingEvaluator { inner: scripted })
                } else {
                    Box::new(scripted)
                };
                let mut controller =
                    crate::search::SearchController::new(belief.seat, config, rng.handle());
                controller.ensure_root(evaluator.as_mut(), &obs, &memory, &belief);
                for _ in 0..batches {
                    controller.run_batch(evaluator.as_mut(), &belief, None, freeze);
                }

                let tree = &controller.tree;
                out.extend([
                    tree.completed_simulations as i64,
                    tree.nodes.len() as i64,
                    tree.table_hits as i64,
                    tree.table_misses as i64,
                ]);
                push_f64(&mut out, &[tree.eviction_loss, tree.total_joint_visits]);
                let root = tree.node(tree.root.expect("ensure_root always sets one"));
                out.push(root.n);
                out.push(root.actions.len() as i64);
                out.extend(root.actions.iter().map(|&a| a as i64));
                push_f64(&mut out, &root.prior);
                push_f64(&mut out, &root.regret);
                push_f64(&mut out, &root.avg_strategy);
                out.push(root.enemy_tables.len() as i64);
                for table in &root.enemy_tables {
                    out.push(table.actions.len() as i64);
                    out.extend(table.actions.iter().map(|&a| a as i64));
                    push_f64(&mut out, &table.prior);
                    push_f64(&mut out, &table.regret);
                    push_f64(&mut out, &table.avg_strategy);
                    out.extend([table.n_self as i64, table.last_used, table.touch_count]);
                    push_f64(&mut out, &table.visits);
                    push_f64(&mut out, &table.q);
                }
                push_f64(&mut out, &tree.root_marginal_visits());
                match tree.root_action_index() {
                    Ok(index) => out.push(index as i64),
                    Err(_) => out.push(-1),
                }
                match controller.best_action() {
                    Some(action) => {
                        out.push(1);
                        out.extend(action.iter().map(|&v| v as i64));
                    }
                    None => out.extend([0, -1, -1, -1, -1, -1]),
                }
                // The degradation path, on the tree that was just built. It
                // lives here rather than in its own surface because its
                // interesting branch — "the search says pass, the policy
                // fallback does not" — needs a real root to say pass.
                for (completed, has_root, fallback) in [
                    (0u64, false, [1, 0, 0, 0, 0]),
                    (0, true, [0, 1, 1, 0, 0]),
                    (tree.completed_simulations, true, [0, 1, 1, 0, 0]),
                    (tree.completed_simulations, true, [1, 0, 0, 0, 0]),
                ] {
                    let (action, level) = crate::runtime::select_degraded_action(
                        completed,
                        has_root,
                        Some(fallback),
                        &controller,
                    );
                    out.extend(action.iter().map(|&v| v as i64));
                    out.push(level as i64);
                }

                // Every node, not only the root — M6's correction to this
                // surface. A leaf value comes from the evaluator and is applied
                // unchanged to every edge on the path, so nothing a child
                // computes ever reaches the root's statistics; and `Replay`
                // hands back the oracle's sampled index whatever distribution
                // this side built, so a divergence inside a child does not even
                // change the tree's shape. Comparing the root alone therefore
                // could not see the enemy-hash cache, the memory fold, or any
                // other per-node arithmetic. The nodes are in creation order on
                // both sides.
                let tree = &controller.tree;
                out.push(tree.nodes.len() as i64);
                for node in &tree.nodes {
                    out.extend([node.n, node.turn as i64]);
                    out.extend(node.memory_digest.iter().map(|&b| b as i64));
                    out.push(node.reservoir.n() as i64);
                    out.push(node.actions.len() as i64);
                    out.extend(node.actions.iter().map(|&a| a as i64));
                    push_f64(&mut out, &node.prior);
                    push_f64(&mut out, &node.regret);
                    push_f64(&mut out, &node.avg_strategy);
                    out.push(node.enemy_tables.len() as i64);
                    for table in &node.enemy_tables {
                        out.extend([table.n_self as i64, table.last_used, table.touch_count]);
                        out.push(table.actions.len() as i64);
                        out.extend(table.actions.iter().map(|&a| a as i64));
                        push_f64(&mut out, &table.prior);
                        push_f64(&mut out, &table.regret);
                        push_f64(&mut out, &table.avg_strategy);
                        push_f64(&mut out, &table.visits);
                        push_f64(&mut out, &table.q);
                    }
                }
                out.push(crate::support::rng::Rng::consumed(&rng) as i64);
            }
            // The eviction decision, driven directly.
            //
            // M6 added this for the reason the `runtime` surface exists: two of
            // the retention rule's four behaviours cannot be reached by running
            // a search at all. The score is `last_used + 0.25 * ln1p(touches)`,
            // so the touch term can only decide a tie in `last_used` — and
            // `last_used` is the node's visit counter at the table's last
            // backup, which advances on every backup, so no two tables in a
            // real tree ever hold the same one. Stating the table set is the
            // only way to ask the question.
            "evict" => {
                let count = ints.n()?;
                let node_n = ints.next()?;
                let max_tables = ints.n()?;
                let mut tree = crate::search::tree::SearchTree::new(1024, max_tables, 0);
                let at = tree
                    .make_node([0u8; 32], 0, [0u8; 32], Vec::new(), [0u8; 32], 0.0, 1)
                    .map_err(|_| "the first node always fits".to_string())?;
                let mut pins: Vec<[u8; 32]> = Vec::new();
                for _ in 0..count {
                    let seed = ints.next()? as u8;
                    let last_used = ints.next()?;
                    let touch_count = ints.next()?;
                    let pinned = ints.next()? != 0;
                    let hash = [seed; 32];
                    tree.get_or_create_enemy_table(at, hash, &[1], &[1.0]);
                    let table = tree
                        .node_mut(at)
                        .table_mut(&hash)
                        .expect("just installed");
                    table.last_used = last_used;
                    table.touch_count = touch_count;
                    if pinned {
                        pins.push(hash);
                    }
                }
                for hash in pins {
                    tree.pin_enemy(at, hash);
                }
                tree.node_mut(at).n = node_n;
                let arriving = [ints.next()? as u8; 32];
                tree.get_or_create_enemy_table(at, arriving, &[1], &[1.0]);
                let node = tree.node(at);
                out.push(node.enemy_tables.len() as i64);
                for table in &node.enemy_tables {
                    out.extend([
                        table.info_hash[0] as i64,
                        table.last_used,
                        table.touch_count,
                    ]);
                }
            }
            // scalar controller arithmetic with no state behind it
            //
            // `nearest_rank_p99` and `highest_prior_legal` are two of the three
            // places the runtime decides something on its own, and neither is
            // reachable through `decide`: the prior it is handed is already
            // legal-normalized, so the mask it applies never binds, and the
            // percentile only shows up in an admission decision the parity
            // harness does not replay. Feeding both directly is the only way
            // the harness can see them at all.
            "runtime" => {
                let count = ints.n()?;
                let samples = ints.f64s(count)?;
                match crate::runtime::nearest_rank_p99(&samples) {
                    Ok(value) => {
                        out.push(1);
                        push_f64(&mut out, &[value]);
                    }
                    Err(_) => out.extend([0, 0]),
                }
                let prior = ints.f64s(N_ACTIONS)?;
                let mask_ints = ints.ints(N_ACTIONS)?;
                let mask: Vec<bool> = mask_ints.iter().map(|v| *v != 0).collect();
                let action = crate::runtime::highest_prior_legal(&prior, &mask);
                out.extend(action.iter().map(|&v| v as i64));
            }
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

/// `i64` to text without pulling in a formatting dependency.
fn itoa(value: i64) -> String {
    value.to_string()
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
