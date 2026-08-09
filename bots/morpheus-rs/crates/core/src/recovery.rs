//! What happens when every particle is refuted at once.
//!
//! Port of `bots/morpheus/recovery.py`. [`filter_step`] does not downweight a
//! particle that fails the observation test, it kills it — so a belief can go
//! from confident to empty in a single frame, and something has to put a legal
//! one back. Three attempts, in order of how much they preserve:
//!
//! 1. [`rejuvenate`] — replay the last `recovery_lag` turns with a bounded
//!    beam over alternative enemy actions, keeping only histories that
//!    reproduce **every** stored observation. This is the only path that
//!    recovers real information rather than inventing a plausible board.
//! 2. Rewind one turn and enumerate: top-ranked policy actions plus every
//!    action that would have changed what we can see. Cheaper, shallower.
//! 3. [`maximum_entropy_reconstruction`] — give up on history and build the
//!    least-committed board consistent with the visible cells and the public
//!    totals. The result is marked `collapsed`, which drops the reported ESS
//!    to `1/n` so the search knows the belief is a guess.
//!
//! If all three fail the previous belief is kept unchanged. Recovery never
//! replaces a valid belief with an invalid one, and never raises: an
//! inconsistent frame is a reason to stop trusting the reconstruction, not a
//! reason to stop playing.
//!
//! The whole file rides M1's transition kernel — replaying an eight-deep
//! history across a beam of eight is where rewrite-plan §7's "large, mostly
//! inherited from the transition kernel" gain actually lands.

use std::rc::Rc;

use crate::action::{decode_action, encode_action, legal_mask};
use crate::belief::{
    filter_step, normalize_weights, resample, Action5, BeliefConfig, BeliefState, HistoryFrame,
    Particle, PASS_ACTION,
};
use crate::memory::{update_memory, VisibleMemory, TYPE_FOG, TYPE_STRUCTURE_FOG};
use crate::observe::{emit_observation, observations_match, visibility_mask};
use crate::proposal::{policy_action_probs, top_legal_actions, ProposalPolicy};
use crate::rng::{npsum, Rng};
use crate::state::GameState;
use crate::transition::{transition, Actions};
use crate::wire::Observation;

/// True when the enemy's action changes what Morpheus can see, versus a pass.
///
/// The cheap enumeration in [`recover_belief`] would otherwise only try
/// high-prior actions, and the actions that *caused* the collapse are exactly
/// the ones that moved something into or out of our vision.
pub fn is_vision_changing(
    state: &GameState,
    seat: usize,
    my_action: Action5,
    enemy_action: Action5,
) -> bool {
    let enemy_seat = 1 - seat;
    let mut actions_b: Actions = [PASS_ACTION; 2];
    actions_b[seat] = my_action;
    actions_b[enemy_seat] = enemy_action;
    let mut actions_pass = actions_b;
    actions_pass[enemy_seat] = PASS_ACTION;

    let (next_b, _) = transition(state, &actions_b);
    let (next_p, _) = transition(state, &actions_pass);
    visibility_mask(&next_b, seat) != visibility_mask(&next_p, seat)
}

pub fn vision_changing_actions(
    state: &GameState,
    seat: usize,
    my_action: Action5,
    legal_actions: &[Action5],
) -> Vec<Action5> {
    legal_actions
        .iter()
        .copied()
        .filter(|&a| a != PASS_ACTION && is_vision_changing(state, seat, my_action, a))
        .collect()
}

fn legal_enemy_actions(particle: &Particle, belief: &BeliefState) -> Vec<Action5> {
    let enemy_obs = emit_observation(&particle.state, belief.enemy_seat());
    let enemy_mem = update_memory(&particle.enemy_memory, &enemy_obs);
    let mask = legal_mask(&enemy_obs, &enemy_mem, None);
    mask.iter()
        .enumerate()
        .filter(|(_, &legal)| legal)
        .filter_map(|(index, _)| decode_action(index))
        .collect()
}

fn apply_joint(
    state: &GameState,
    seat: usize,
    my_action: Action5,
    enemy_action: Action5,
) -> GameState {
    let mut actions: Actions = [PASS_ACTION; 2];
    actions[seat] = my_action;
    actions[1 - seat] = enemy_action;
    transition(state, &actions).0
}

/// One live branch of the replay beam.
struct BeamEntry {
    weight: f64,
    state: Rc<GameState>,
    enemy_memory: Rc<VisibleMemory>,
    enemy_prev: Option<Action5>,
}

/// Policy-guided bounded beam replay of the last `recovery_lag` turns.
///
/// Accepts only histories that reproduce every stored observation. Admission
/// control may stop the replay early — `max_replayed_transitions` is checked
/// between every single transition, not per turn — and the current valid
/// belief is preserved on failure rather than replaced by a partial one.
pub fn rejuvenate<R: Rng + ?Sized>(
    belief: &BeliefState,
    rng: &mut R,
    mut policy: Option<&mut (dyn ProposalPolicy + '_)>,
) -> BeliefState {
    if belief.n() == 0 {
        return belief.clone();
    }
    let seeds: Vec<&Particle> = belief
        .particles
        .iter()
        .filter(|p| !p.history.is_empty())
        .collect();
    if seeds.is_empty() {
        return belief.clone();
    }

    let config = belief.config;
    let seat = belief.seat;
    let enemy_seat = belief.enemy_seat();
    let mut completed: Vec<(f64, Particle)> = Vec::new();
    let mut transitions_used = 0usize;

    for seed in seeds {
        if completed.len() >= config.max_completed_histories {
            break;
        }
        if transitions_used >= config.max_replayed_transitions {
            break;
        }

        let frames: &[Rc<HistoryFrame>] = &seed.history;
        // Start from the ancestor before the first stored frame, with an
        // empty enemy memory: what the opponent remembers is itself a guess,
        // and replaying it from scratch is the only version we can justify.
        let start_state = Rc::clone(&frames[0].state);
        let start_mem = Rc::new(VisibleMemory::empty(start_state.h, start_state.w));
        let mut beam = vec![BeamEntry {
            weight: 1.0,
            state: start_state,
            enemy_memory: start_mem,
            enemy_prev: None,
        }];

        for (t, frame) in frames.iter().enumerate() {
            if transitions_used >= config.max_replayed_transitions {
                break;
            }
            let mut next_beam: Vec<BeamEntry> = Vec::new();
            for entry in &beam {
                let temp = Particle {
                    state: Rc::clone(&entry.state),
                    weight: 1.0,
                    enemy_memory: Rc::clone(&entry.enemy_memory),
                    enemy_prev_action: entry.enemy_prev,
                    history: Vec::new(),
                };
                let probs = policy_action_probs(belief, &temp, policy.as_deref_mut());
                let enemy_obs = emit_observation(&entry.state, enemy_seat);
                let enemy_mem = update_memory(&entry.enemy_memory, &enemy_obs);
                let mask = legal_mask(&enemy_obs, &enemy_mem, None);

                let candidates: Vec<Action5> = if t == 0 {
                    // The first step is enumerated, not sampled: it is the one
                    // whose alternatives we can afford to be exhaustive about.
                    let mut candidates = top_legal_actions(&probs, &mask, 4);
                    for action in vision_changing_actions(
                        &entry.state,
                        seat,
                        frame.my_action,
                        &legal_enemy_actions(&temp, belief),
                    ) {
                        if !candidates.contains(&action) {
                            candidates.push(action);
                        }
                    }
                    candidates
                } else {
                    // Later steps sample. `beam_width * 2` draws for
                    // `beam_width` distinct actions: a bounded attempt, not a
                    // loop that could spin on a near-degenerate distribution.
                    let mut candidates: Vec<Action5> = Vec::new();
                    for _ in 0..config.beam_width * 2 {
                        let index = rng.choice_one(probs.len(), Some(&probs));
                        let action = decode_action(index).unwrap_or(PASS_ACTION);
                        if !candidates.contains(&action) {
                            candidates.push(action);
                        }
                        if candidates.len() >= config.beam_width {
                            break;
                        }
                    }
                    if candidates.is_empty() {
                        candidates.push(PASS_ACTION);
                    }
                    candidates
                };

                let enemy_mem = Rc::new(enemy_mem);
                for enemy_action in candidates {
                    if transitions_used >= config.max_replayed_transitions {
                        break;
                    }
                    transitions_used += 1;
                    let next_state = apply_joint(&entry.state, seat, frame.my_action, enemy_action);
                    if !observations_match(
                        &emit_observation(&next_state, seat),
                        &frame.observation_after,
                    ) {
                        continue;
                    }
                    // Acceptance weight: the product of policy probabilities
                    // along the path, floored so one improbable-but-real step
                    // cannot zero a history that otherwise explains everything.
                    let p_b = probs[encode_action(enemy_action)];
                    let new_weight = entry.weight * p_b.max(1e-12);
                    let next_state = Rc::new(next_state);
                    let e_obs = emit_observation(&next_state, enemy_seat);
                    next_beam.push(BeamEntry {
                        weight: new_weight,
                        state: next_state,
                        enemy_memory: Rc::new(update_memory(&enemy_mem, &e_obs)),
                        enemy_prev: Some(enemy_action),
                    });
                }
            }

            if next_beam.is_empty() {
                beam = Vec::new();
                break;
            }
            // Stable, matching Python's `list.sort`: two branches of equal
            // weight keep the order they were generated in, which is the order
            // the candidate lists imposed.
            next_beam.sort_by(|a, b| (-a.weight).partial_cmp(&(-b.weight)).unwrap());
            next_beam.truncate(config.beam_width);
            beam = next_beam;
        }

        for entry in beam {
            if completed.len() >= config.max_completed_histories {
                break;
            }
            completed.push((
                entry.weight,
                Particle {
                    state: entry.state,
                    weight: entry.weight,
                    enemy_memory: entry.enemy_memory,
                    enemy_prev_action: entry.enemy_prev,
                    // The observations in this window have already been
                    // matched, so the seed's frames describe this branch too.
                    history: seed.history.clone(),
                },
            ));
        }
    }

    if completed.is_empty() {
        return belief.clone(); // keep the current valid belief
    }

    let raw: Vec<f64> = completed.iter().map(|(w, _)| *w).collect();
    let total = npsum(&raw);
    let particles: Vec<Particle> = completed
        .iter()
        .zip(&raw)
        .map(|((_, p), &w)| p.with_weight(w / total))
        .collect();
    BeliefState {
        seat,
        particles: resample(&particles, config.n_particles, rng),
        config,
        collapsed: false,
    }
}

/// Cells that could hold something the seat cannot see.
fn hidden_cells(obs: &Observation, memory: &VisibleMemory) -> Vec<usize> {
    let mut cells = Vec::new();
    for i in 0..obs.h * obs.w {
        let t = obs.type_grid[i] as i32;
        if t != TYPE_FOG && t != TYPE_STRUCTURE_FOG {
            continue;
        }
        if obs.owner_grid[i] as i32 == 1 {
            continue;
        }
        if memory.known_mountain[i] {
            continue;
        }
        // Structure fog never seen up close might be a mountain, and a
        // mountain cannot hold land. Only count it when something proved the
        // ground passable.
        if t == TYPE_STRUCTURE_FOG && !memory.known_castle[i] && !memory.known_passable_base[i] {
            continue;
        }
        cells.push(i);
    }
    cells
}

/// Uniform-general, even-army reconstruction matching visible cells and totals.
///
/// The least-committed board consistent with what is known: the enemy general
/// is uniform over hidden cells, its army is spread evenly over the land the
/// public totals say it must hold. Every candidate still has to reproduce the
/// observation exactly before it is admitted, so "least committed" never means
/// "illegal".
pub fn maximum_entropy_reconstruction<R: Rng + ?Sized>(
    obs: &Observation,
    seat: usize,
    memory: &VisibleMemory,
    rng: &mut R,
    config: BeliefConfig,
) -> Result<BeliefState, String> {
    let (h, w) = (obs.h, obs.w);
    let n_cells = h * w;

    let mut mountains = vec![false; n_cells];
    let mut castles = vec![false; n_cells];
    let mut visible = vec![false; n_cells];
    for i in 0..n_cells {
        let t = obs.type_grid[i] as i32;
        visible[i] = t != TYPE_FOG && t != TYPE_STRUCTURE_FOG;
        mountains[i] = memory.known_mountain[i] || t == 2;
        castles[i] = memory.known_castle[i] || t == 3;
        if t == TYPE_STRUCTURE_FOG {
            // First sight of a structure: ground never proven passable is a
            // mountain, ground that was is a castle.
            if !memory.known_passable_base[i] && !memory.known_castle[i] {
                mountains[i] = true;
            }
            if memory.known_castle[i] || memory.known_passable_base[i] {
                castles[i] = true;
            }
        }
    }
    let passable: Vec<bool> = mountains.iter().map(|&m| !m).collect();

    let mut own_land_vis = 0i64;
    let mut opp_land_vis = 0i64;
    let mut own_army_vis = 0i64;
    let mut opp_army_vis = 0i64;
    for i in 0..n_cells {
        if !visible[i] {
            continue;
        }
        match obs.owner_grid[i] as i32 {
            1 => {
                own_land_vis += 1;
                own_army_vis += obs.army_grid[i] as i64;
            }
            2 => {
                opp_land_vis += 1;
                opp_army_vis += obs.army_grid[i] as i64;
            }
            _ => {}
        }
    }

    let need_opp_land = obs.opp_land as i64 - opp_land_vis;
    let need_opp_army = obs.opp_army as i64 - opp_army_vis;
    let need_own_land = obs.my_land as i64 - own_land_vis;
    let need_own_army = obs.my_army as i64 - own_army_vis;
    if need_opp_land < 0 || need_opp_army < 0 || need_own_land < 0 || need_own_army < 0 {
        return Err("observation totals inconsistent with visible cells".to_string());
    }

    let hidden = hidden_cells(obs, memory);
    if need_opp_land > hidden.len() as i64 {
        return Err("not enough hidden cells for enemy land total".to_string());
    }

    let n = config.n_particles;
    let gen_candidates: Vec<usize> = if need_opp_land >= 1 {
        hidden.clone()
    } else {
        Vec::new()
    };
    if need_opp_land >= 1 && gen_candidates.is_empty() {
        return Err("no hidden cell for enemy general".to_string());
    }

    let mut particles: Vec<Particle> = Vec::with_capacity(n);
    let weight = 1.0 / n as f64;
    let empty = Rc::new(VisibleMemory::empty(h, w));

    for _ in 0..n {
        let mut state = GameState::empty(h, w);
        state.time = obs.turn;
        state.winner = -1;

        for i in 0..n_cells {
            if !visible[i] {
                continue;
            }
            state.armies[i] = obs.army_grid[i];
            match obs.owner_grid[i] as i32 {
                1 => state.ownership[seat][i] = true,
                2 => state.ownership[1 - seat][i] = true,
                _ => {}
            }
            if obs.type_grid[i] as i32 == 4 {
                state.generals[i] = true;
            }
        }

        let mut enemy_cells: Vec<usize> = Vec::new();
        if need_opp_land >= 1 {
            let pick = rng.integer(0, gen_candidates.len() as i64) as usize;
            let general_at = gen_candidates[pick];
            let remaining: Vec<usize> = hidden.iter().copied().filter(|&c| c != general_at).collect();
            let extra_n = need_opp_land - 1;
            let mut extras: Vec<usize> = Vec::new();
            if extra_n > 0 {
                for index in rng.choice(remaining.len(), Some(extra_n as usize), false, None) {
                    extras.push(remaining[index as usize]);
                }
            }
            enemy_cells.push(general_at);
            enemy_cells.extend(extras);
            // Even split: the maximum-entropy allocation of a known army
            // total over a chosen set of cells.
            let base = need_opp_army / need_opp_land;
            let rem = need_opp_army % need_opp_land;
            for (i, &at) in enemy_cells.iter().enumerate() {
                state.ownership[1 - seat][at] = true;
                state.armies[at] = (base + if (i as i64) < rem { 1 } else { 0 }) as i32;
            }
            state.generals[general_at] = true;
        }

        // Our own hidden land: rare under fog for Morpheus, but the totals
        // are public and have to add up.
        if need_own_land > 0 {
            let free: Vec<usize> = hidden
                .iter()
                .copied()
                .filter(|c| !enemy_cells.contains(c))
                .collect();
            if need_own_land > free.len() as i64 {
                continue;
            }
            let picks = rng.choice(free.len(), Some(need_own_land as usize), false, None);
            let base = need_own_army / need_own_land;
            let rem = need_own_army % need_own_land;
            for (i, index) in picks.iter().enumerate() {
                let at = free[*index as usize];
                state.ownership[seat][at] = true;
                state.armies[at] = (base + if (i as i64) < rem { 1 } else { 0 }) as i32;
            }
        }

        for i in 0..n_cells {
            state.mountains[i] = mountains[i];
            state.castles[i] = castles[i];
            state.passable[i] = passable[i];
            state.ownership_neutral[i] =
                passable[i] && !state.ownership[0][i] && !state.ownership[1][i];
        }
        for s in 0..2 {
            state.general_positions[s] = [-1, -1];
            for i in 0..n_cells {
                if state.generals[i] && state.ownership[s][i] {
                    state.general_positions[s] = [(i / w) as i32, (i % w) as i32];
                    break;
                }
            }
        }
        // Our own general is never in doubt — memory latched it on turn one.
        if let Some(at) = (0..n_cells).find(|&i| memory.own_general[i]) {
            state.general_positions[seat] = [(at / w) as i32, (at % w) as i32];
            state.generals[at] = true;
            state.ownership[seat][at] = true;
        }

        if !observations_match(&emit_observation(&state, seat), obs) {
            continue;
        }
        particles.push(Particle::new(Rc::new(state), weight, Rc::clone(&empty)));
    }

    if particles.is_empty() {
        return Err("maximum-entropy reconstruction produced no valid particles".to_string());
    }

    Ok(BeliefState {
        seat,
        particles: resample(&normalize_weights(&particles), n, rng),
        config,
        collapsed: true,
    })
}

/// Collapse recovery: rejuvenate, rewind and enumerate, then max-entropy.
///
/// The Python takes a `my_action` here and never reads it — every replay uses
/// the action stored on the history frame, which is the one that actually
/// produced the observation being explained. It is dropped rather than
/// carried as `_my_action`, because a parameter that exists only to be
/// ignored invites a caller to believe it matters.
pub fn recover_belief<R: Rng + ?Sized>(
    belief: &BeliefState,
    real_obs: &Observation,
    memory: &VisibleMemory,
    rng: &mut R,
    mut policy: Option<&mut (dyn ProposalPolicy + '_)>,
) -> BeliefState {
    let seat = belief.seat;
    let enemy_seat = belief.enemy_seat();

    // 1) Rejuvenation from stored ancestors.
    if belief.particles.iter().any(|p| !p.history.is_empty()) {
        let rejuvenated = rejuvenate(belief, rng, policy.as_deref_mut());
        let positive: Vec<&Particle> = rejuvenated
            .particles
            .iter()
            .filter(|p| p.weight > 0.0)
            .collect();
        if !positive.is_empty() {
            // Matching the stored window is not the same as matching *this*
            // frame; the replay was validated against `observation_after`,
            // which stops one turn short when the collapse is fresh.
            let ok = positive
                .iter()
                .all(|p| observations_match(&emit_observation(&p.state, seat), real_obs));
            if ok {
                return rejuvenated;
            }
        }
    }

    // 2) Rewind one turn and enumerate.
    let mut rewind_ok: Vec<Particle> = Vec::new();
    for particle in &belief.particles {
        let Some(frame) = particle.history.last() else {
            continue;
        };
        let temp = Particle {
            state: Rc::clone(&frame.state),
            weight: 1.0,
            enemy_memory: Rc::clone(&particle.enemy_memory),
            enemy_prev_action: None,
            history: Vec::new(),
        };
        let probs = policy_action_probs(belief, &temp, policy.as_deref_mut());
        let enemy_obs = emit_observation(&frame.state, enemy_seat);
        let enemy_mem = update_memory(&VisibleMemory::empty(real_obs.h, real_obs.w), &enemy_obs);
        let mask = legal_mask(&enemy_obs, &enemy_mem, None);
        let mut candidates = top_legal_actions(&probs, &mask, 8);
        for action in vision_changing_actions(
            &frame.state,
            seat,
            frame.my_action,
            &legal_enemy_actions(&temp, belief),
        ) {
            if !candidates.contains(&action) {
                candidates.push(action);
            }
        }
        for enemy_action in candidates {
            let next_state = apply_joint(&frame.state, seat, frame.my_action, enemy_action);
            if !observations_match(&emit_observation(&next_state, seat), real_obs) {
                continue;
            }
            let next_state = Rc::new(next_state);
            let e_obs = emit_observation(&next_state, enemy_seat);
            rewind_ok.push(Particle {
                state: next_state,
                weight: 1.0,
                enemy_memory: Rc::new(update_memory(&enemy_mem, &e_obs)),
                enemy_prev_action: Some(enemy_action),
                history: particle.history.clone(),
            });
            break; // one explanation per particle is enough
        }
    }

    if !rewind_ok.is_empty() {
        return BeliefState {
            seat,
            particles: resample(
                &normalize_weights(&rewind_ok),
                belief.config.n_particles,
                rng,
            ),
            config: belief.config,
            collapsed: false,
        };
    }

    // 3) Maximum-entropy reconstruction, or the last valid belief.
    maximum_entropy_reconstruction(real_obs, seat, memory, rng, belief.config)
        .unwrap_or_else(|_| belief.clone())
}

/// One real-turn update: propose (unless injected), filter, recover if empty.
///
/// `enemy_actions` is injected by deterministic fixtures; the runtime lets
/// the proposal draw them. Recovery runs only when *every* particle failed the
/// exact observation check — a partial survival is not a collapse, and
/// reconstructing on one would throw away the particles that were right.
#[allow(clippy::too_many_arguments)]
pub fn update_belief<R: Rng + ?Sized>(
    belief: &BeliefState,
    my_action: Action5,
    real_obs: &Observation,
    memory: &VisibleMemory,
    rng: &mut R,
    mut policy: Option<&mut (dyn ProposalPolicy + '_)>,
    enemy_actions: Option<&[Action5]>,
    max_proposal_batch: usize,
) -> Result<BeliefState, String> {
    let drawn;
    let enemy_actions = match enemy_actions {
        Some(actions) => actions,
        None => {
            drawn = crate::proposal::propose_enemy_actions(
                belief,
                rng,
                policy.as_deref_mut(),
                max_proposal_batch,
                None,
            );
            &drawn
        }
    };
    let next = filter_step(belief, my_action, real_obs, enemy_actions, rng)?;
    if next.particles.iter().any(|p| p.weight > 0.0) {
        return Ok(next);
    }
    Ok(recover_belief(belief, real_obs, memory, rng, policy))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SmallRng;

    fn corridor(h: usize, w: usize) -> GameState {
        let mut state = GameState::empty(h, w);
        for i in 0..h * w {
            state.passable[i] = true;
            state.ownership_neutral[i] = true;
        }
        state
    }

    fn two_generals() -> GameState {
        let mut state = corridor(7, 7);
        for (seat, at, army) in [(0usize, 0usize, 9i32), (1, 48, 9)] {
            state.ownership[seat][at] = true;
            state.ownership_neutral[at] = false;
            state.generals[at] = true;
            state.armies[at] = army;
        }
        state.general_positions = [[0, 0], [6, 6]];
        state
    }

    fn belief_with_history(n_particles: usize) -> (BeliefState, Observation) {
        let config = BeliefConfig {
            n_particles,
            ..Default::default()
        };
        let state = two_generals();
        let particles: Vec<Particle> = (0..n_particles)
            .map(|_| {
                Particle::new(
                    Rc::new(state.clone()),
                    1.0 / n_particles as f64,
                    Rc::new(VisibleMemory::empty(7, 7)),
                )
            })
            .collect();
        let belief = BeliefState::new(0, particles, config);
        let (next, _) = transition(&state, &[PASS_ACTION, PASS_ACTION]);
        let real = emit_observation(&next, 0);
        let mut rng = SmallRng::seed_from_u64(2);
        let actions = vec![PASS_ACTION; n_particles];
        let advanced = filter_step(&belief, PASS_ACTION, &real, &actions, &mut rng).unwrap();
        (advanced, real)
    }

    #[test]
    fn a_pass_never_changes_what_we_can_see() {
        let state = two_generals();
        assert!(!is_vision_changing(&state, 0, PASS_ACTION, PASS_ACTION));
    }

    #[test]
    fn vision_changes_when_the_enemy_takes_something_of_ours() {
        // "Vision changing" is measured on *our* ownership footprint, not the
        // enemy's: `visibility_mask` is evaluated at `seat`. So an enemy step
        // into empty ground is invisible to this predicate and an enemy step
        // that captures one of our cells is not. That asymmetry is the whole
        // point — these are the actions that could explain why the belief
        // just died.
        let mut state = two_generals();
        state.ownership[0][24] = true; // (3,3)
        state.ownership_neutral[24] = false;
        state.armies[24] = 1;
        state.ownership[1][25] = true; // (3,4), adjacent, with enough to take it
        state.ownership_neutral[25] = false;
        state.armies[25] = 9;

        let capture = [0, 3, 4, 2, 0]; // left, onto (3,3)
        let elsewhere = [0, 6, 6, 0, 0]; // the enemy general steps up
        assert!(is_vision_changing(&state, 0, PASS_ACTION, capture));
        assert!(!is_vision_changing(&state, 0, PASS_ACTION, elsewhere));

        let changing = vision_changing_actions(&state, 0, PASS_ACTION, &[
            PASS_ACTION,
            capture,
            elsewhere,
        ]);
        assert_eq!(changing, vec![capture], "pass is excluded by construction");
    }

    #[test]
    fn rejuvenation_of_a_belief_without_history_returns_it_unchanged() {
        let config = BeliefConfig {
            n_particles: 2,
            ..Default::default()
        };
        let belief = BeliefState::new(
            0,
            vec![Particle::new(
                Rc::new(two_generals()),
                1.0,
                Rc::new(VisibleMemory::empty(7, 7)),
            )],
            config,
        );
        let mut rng = SmallRng::seed_from_u64(5);
        let out = rejuvenate(&belief, &mut rng, None);
        assert_eq!(out.n(), 1);
        assert_eq!(out.particles[0].weight, 1.0);
    }

    #[test]
    fn rejuvenation_reproduces_every_stored_observation() {
        let (belief, _) = belief_with_history(4);
        assert!(belief.particles.iter().all(|p| p.history.len() == 1));
        let mut rng = SmallRng::seed_from_u64(6);
        let out = rejuvenate(&belief, &mut rng, None);
        assert_eq!(out.n(), 4);
        for particle in &out.particles {
            let frame = &belief.particles[0].history[0];
            let replayed = apply_joint(
                &frame.state,
                0,
                frame.my_action,
                particle.enemy_prev_action.unwrap(),
            );
            assert!(observations_match(
                &emit_observation(&replayed, 0),
                &frame.observation_after
            ));
        }
    }

    #[test]
    fn recovery_falls_through_to_a_reconstruction_that_matches_the_frame() {
        // No history anywhere, so rejuvenation and rewind both decline and
        // the max-entropy path has to carry it.
        let state = two_generals();
        let obs = emit_observation(&state, 0);
        let memory = update_memory(&VisibleMemory::empty(7, 7), &obs);
        let config = BeliefConfig {
            n_particles: 4,
            ..Default::default()
        };
        let belief = BeliefState::new(
            0,
            vec![Particle::new(
                Rc::new(state),
                1.0,
                Rc::new(VisibleMemory::empty(7, 7)),
            )],
            config,
        );
        let mut rng = SmallRng::seed_from_u64(8);
        let out = recover_belief(&belief, &obs, &memory, &mut rng, None);
        assert!(out.collapsed, "a reconstruction is not a filtered belief");
        assert_eq!(out.n(), 4);
        for particle in &out.particles {
            assert!(observations_match(&emit_observation(&particle.state, 0), &obs));
        }
    }

    #[test]
    fn a_collapsed_belief_reports_the_lowest_confidence_it_can() {
        use crate::belief::ess_fraction;
        let state = two_generals();
        let obs = emit_observation(&state, 0);
        let memory = update_memory(&VisibleMemory::empty(7, 7), &obs);
        let config = BeliefConfig {
            n_particles: 8,
            ..Default::default()
        };
        let mut rng = SmallRng::seed_from_u64(3);
        let out = maximum_entropy_reconstruction(&obs, 0, &memory, &mut rng, config).unwrap();
        assert_eq!(ess_fraction(&out), 1.0 / 8.0);
    }

    #[test]
    fn an_impossible_frame_leaves_the_previous_belief_alone() {
        let state = two_generals();
        let mut obs = emit_observation(&state, 0);
        // Claim more enemy land than any board could hide.
        obs.opp_land = 10_000;
        let memory = update_memory(&VisibleMemory::empty(7, 7), &emit_observation(&state, 0));
        let config = BeliefConfig {
            n_particles: 2,
            ..Default::default()
        };
        let belief = BeliefState::new(
            0,
            vec![Particle::new(
                Rc::new(state),
                1.0,
                Rc::new(VisibleMemory::empty(7, 7)),
            )],
            config,
        );
        let mut rng = SmallRng::seed_from_u64(1);
        let out = recover_belief(&belief, &obs, &memory, &mut rng, None);
        assert_eq!(out.n(), 1);
        assert!(!out.collapsed, "nothing was reconstructed, so nothing collapsed");
    }
}
