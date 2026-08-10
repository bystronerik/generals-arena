//! Every `read_*` / `write_*` / `push_*` the surfaces share.
//!
//! Each layout here is positional and mirrored in `tests/parity_cases.py`.
//! Lengths are self-describing: a case starts with the dimensions that
//! determine how many integers follow, so a truncated stream fails loudly at
//! the point of truncation instead of silently shifting every later case.

use std::rc::Rc;

use crate::belief::{
    Action5, BeliefConfig, BeliefState, HistoryFrame, Particle,
};
use crate::board::memory::VisibleMemory;
use crate::board::observe::emit_observation;
use crate::support::rng::{Method, RecordedDraw};
use crate::board::state::GameState;
use crate::board::transition::{transition, Actions};
use crate::io::wire::Observation;
use crate::parity::ints::Ints;

pub(in crate::parity) fn fill_bools(dst: &mut [bool], src: &[i32]) {
    for (i, v) in src.iter().enumerate() {
        dst[i] = *v != 0;
    }
}

/// `h w time winner gp[4] armies[n] own0[n] own1[n] neutral[n] generals[n]
/// castles[n] mountains[n] passable[n]`
pub(in crate::parity) fn read_state(ints: &mut Ints) -> Result<GameState, String> {
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

pub(in crate::parity) fn write_state(out: &mut Vec<i64>, state: &GameState) {
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
pub(in crate::parity) fn read_observation(ints: &mut Ints) -> Result<Observation, String> {
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

pub(in crate::parity) fn write_observation(out: &mut Vec<i64>, obs: &Observation) {
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
pub(in crate::parity) fn read_memory(ints: &mut Ints) -> Result<VisibleMemory, String> {
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

pub(in crate::parity) fn write_memory(out: &mut Vec<i64>, memory: &VisibleMemory) {
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
pub(in crate::parity) fn push_f32(out: &mut Vec<i64>, values: &[f32]) {
    out.extend(values.iter().map(|v| v.to_bits() as i64));
}

/// f64 the same way. `to_bits()` is a `u64`, so the top bit rides as a
/// negative `i64` and the Python side reinterprets — the stream is a bit
/// channel, not a number channel.
pub(in crate::parity) fn push_f64(out: &mut Vec<i64>, values: &[f64]) {
    out.extend(values.iter().map(|v| v.to_bits() as i64));
}

// ------------------------------------------------------------------ M4 I/O

/// `n_draws` then, per draw: `code a b size replace weighted count values…`
///
/// `size` rides as `-1` for NumPy's scalar form, which is a *different* draw
/// from `size=1` and has to stay distinguishable — `Replay` refuses the swap,
/// and that refusal is most of what makes draw-site order a checked contract
/// rather than a hope (rewrite-plan §5).
pub(in crate::parity) fn read_draws(ints: &mut Ints) -> Result<Vec<RecordedDraw>, String> {
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
pub(in crate::parity) fn read_particle(ints: &mut Ints, seat: usize) -> Result<Particle, String> {
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
pub(in crate::parity) fn read_belief(ints: &mut Ints) -> Result<BeliefState, String> {
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
pub(in crate::parity) fn write_belief(out: &mut Vec<i64>, belief: &BeliefState) {
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
pub(in crate::parity) fn read_belief_lite(ints: &mut Ints) -> Result<BeliefState, String> {
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
pub(in crate::parity) fn read_action_history(ints: &mut Ints) -> Result<(Option<Action5>, Vec<Action5>), String> {
    let has_prev = ints.next()? != 0;
    let prev = ints.action()?;
    let count = ints.n()?;
    let mut recent = Vec::with_capacity(count);
    for _ in 0..count {
        recent.push(ints.action()?);
    }
    Ok((if has_prev { Some(prev) } else { None }, recent))
}

pub(in crate::parity) fn push_cell(out: &mut Vec<i64>, cell: Option<(usize, usize)>) {
    match cell {
        Some((r, c)) => out.extend([1, r as i64, c as i64]),
        None => out.extend([0, -1, -1]),
    }
}

pub(in crate::parity) fn push_action(out: &mut Vec<i64>, action: Option<Action5>) {
    match action {
        Some(action) => {
            out.push(1);
            out.extend(action.iter().map(|&v| v as i64));
        }
        None => out.extend([0, -1, -1, -1, -1, -1]),
    }
}
