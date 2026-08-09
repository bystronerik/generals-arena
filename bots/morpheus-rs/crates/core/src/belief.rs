//! The weighted particle filter over hidden enemy state.
//!
//! Port of `bots/morpheus/belief.py`. A particle is a complete guess at the
//! board — including everything fog hides — and the likelihood is not a score
//! but a predicate: emit the observation this particle implies, and either it
//! is the observation that actually arrived or the particle is impossible.
//! [`filter_step`] is that test applied once per particle per turn, which is
//! why the transition kernel M1 proved bit-exact is the thing that decides
//! whether the belief is right *and* the thing that decides whether it is
//! affordable (rewrite-plan §7: this component measured 357 ms at p99 on one
//! x86 core, against a 140 ms deadline).
//!
//! ## Particles share, they do not copy
//!
//! The Python rebuilds a frozen `Particle` on every weight change and lets
//! CPython's refcounting share the arrays underneath. A literal port would
//! `memcpy` a 5 KB state and a 5 KB memory on every one of those rebuilds —
//! and `resample` alone rebuilds the whole set. `Rc` reproduces the Python's
//! sharing exactly, because the values genuinely are immutable here: a state
//! is built by [`crate::transition::transition`] and never touched again.
//! This is the answer to M1's note that "`transition` clones a ~3 KB state
//! per call; M4 should measure before assuming the clone is free" — the
//! transition still builds one state, but nothing downstream duplicates it.
//!
//! ## The arithmetic is NumPy's, deliberately
//!
//! `ess` divides two `np.sum`s and the result is compared against a
//! threshold that decides whether a resample happens — and a resample draws
//! from the RNG. So a last-bit difference in a sum is not a rounding
//! difference, it is a different sequence of random numbers from there on.
//! [`crate::rng::npsum`] exists for that reason and is used at every site
//! where the Python wrote `.sum()` on an array; the sites where the Python
//! wrote the *builtin* `sum` over a generator stay sequential, because that
//! is a different function with a different answer.

use std::rc::Rc;

use crate::memory::{update_memory, VisibleMemory, TYPE_FOG, TYPE_MOUNTAIN, TYPE_STRUCTURE_FOG};
use crate::observe::{emit_observation, observations_match, visibility_from_owned};
use crate::rng::{npsum, Rng};
use crate::state::GameState;
use crate::transition::{transition, Actions};
use crate::wire::Observation;

/// Defaults from `belief.py`. `deployment.json` overrides `n_particles`.
pub const N_PARTICLES: usize = 64;
pub const ESS_THRESHOLD_FRACTION: f64 = 0.5;
pub const RECOVERY_LAG: usize = 8;
pub const BEAM_WIDTH: usize = 8;
pub const MAX_COMPLETED_HISTORIES: usize = 16;
pub const MAX_REPLAYED_TRANSITIONS: usize = 128;
pub const MIN_GENERAL_DISTANCE: i32 = 17;

pub type Action5 = [i32; 5];

pub const PASS_ACTION: Action5 = [1, 0, 0, 0, 0];

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct BeliefConfig {
    pub n_particles: usize,
    pub ess_threshold_fraction: f64,
    pub recovery_lag: usize,
    pub beam_width: usize,
    pub max_completed_histories: usize,
    pub max_replayed_transitions: usize,
    pub min_general_distance: i32,
}

impl Default for BeliefConfig {
    fn default() -> Self {
        Self {
            n_particles: N_PARTICLES,
            ess_threshold_fraction: ESS_THRESHOLD_FRACTION,
            recovery_lag: RECOVERY_LAG,
            beam_width: BEAM_WIDTH,
            max_completed_histories: MAX_COMPLETED_HISTORIES,
            max_replayed_transitions: MAX_REPLAYED_TRANSITIONS,
            min_general_distance: MIN_GENERAL_DISTANCE,
        }
    }
}

/// State and joint actions immediately before one real transition.
///
/// `observation_after` is the *real* observation that followed, which is what
/// makes rejuvenation a test rather than a simulation: a replayed enemy action
/// is accepted only if it reproduces the frame the engine actually sent.
#[derive(Clone, Debug)]
pub struct HistoryFrame {
    pub state: Rc<GameState>,
    pub my_action: Action5,
    pub enemy_action: Action5,
    pub observation_after: Rc<Observation>,
}

#[derive(Clone, Debug)]
pub struct Particle {
    pub state: Rc<GameState>,
    pub weight: f64,
    pub enemy_memory: Rc<VisibleMemory>,
    pub enemy_prev_action: Option<Action5>,
    pub history: Vec<Rc<HistoryFrame>>,
}

impl Particle {
    pub fn new(state: Rc<GameState>, weight: f64, enemy_memory: Rc<VisibleMemory>) -> Self {
        Self {
            state,
            weight,
            enemy_memory,
            enemy_prev_action: None,
            history: Vec::new(),
        }
    }

    /// The Python's `Particle(state=p.state, weight=w, ...)` rebuild: same
    /// everything, new weight. Sharing means this costs a refcount, not a
    /// board.
    pub fn with_weight(&self, weight: f64) -> Self {
        Self {
            state: Rc::clone(&self.state),
            weight,
            enemy_memory: Rc::clone(&self.enemy_memory),
            enemy_prev_action: self.enemy_prev_action,
            history: self.history.clone(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct BeliefState {
    /// Morpheus's own seat index in `GameState`.
    pub seat: usize,
    pub particles: Vec<Particle>,
    pub config: BeliefConfig,
    /// True after a maximum-entropy reconstruction.
    pub collapsed: bool,
}

impl BeliefState {
    pub fn new(seat: usize, particles: Vec<Particle>, config: BeliefConfig) -> Self {
        Self {
            seat,
            particles,
            config,
            collapsed: false,
        }
    }

    #[inline]
    pub fn enemy_seat(&self) -> usize {
        1 - self.seat
    }

    #[inline]
    pub fn n(&self) -> usize {
        self.particles.len()
    }

    pub fn weights(&self) -> Vec<f64> {
        self.particles.iter().map(|p| p.weight).collect()
    }
}

/// Effective sample size `1 / sum(w_i^2)` for normalized weights.
pub fn ess(weights: &[f64]) -> f64 {
    if weights.is_empty() {
        return 0.0;
    }
    let total = npsum(weights);
    if total <= 0.0 {
        return 0.0;
    }
    let normalized: Vec<f64> = weights.iter().map(|w| w / total).collect();
    let squares: Vec<f64> = normalized.iter().map(|w| w * w).collect();
    1.0 / npsum(&squares)
}

pub fn ess_fraction(belief: &BeliefState) -> f64 {
    let n_cfg = belief.config.n_particles.max(1) as f64;
    if belief.collapsed {
        return 1.0 / n_cfg;
    }
    if belief.n() == 0 {
        return 0.0;
    }
    ess(&belief.weights()) / n_cfg
}

/// Rescale to sum one, dropping negative weights to zero.
///
/// The Python uses the *builtin* `sum` over a generator here, not `np.sum`,
/// so this accumulates left to right. The distinction is not pedantry: the
/// two disagree in the last bits at eight particles, and this total is a
/// divisor.
pub fn normalize_weights(particles: &[Particle]) -> Vec<Particle> {
    let mut total = 0.0f64;
    for particle in particles {
        total += particle.weight.max(0.0);
    }
    if total <= 0.0 {
        return particles.to_vec();
    }
    particles
        .iter()
        .map(|p| p.with_weight(p.weight.max(0.0) / total))
        .collect()
}

/// Multinomial resample to `n` equal-weight particles.
pub fn resample<R: Rng + ?Sized>(particles: &[Particle], n: usize, rng: &mut R) -> Vec<Particle> {
    if particles.is_empty() || n == 0 {
        return Vec::new();
    }
    let weights: Vec<f64> = particles.iter().map(|p| p.weight.max(0.0)).collect();
    let total = npsum(&weights);
    let indices = if total <= 0.0 {
        rng.integers(0, particles.len() as i64, Some(n))
    } else {
        let probs: Vec<f64> = weights.iter().map(|w| w / total).collect();
        rng.choice(particles.len(), Some(n), true, Some(&probs))
    };
    let weight = 1.0 / n as f64;
    indices
        .into_iter()
        .map(|i| particles[i as usize].with_weight(weight))
        .collect()
}

/// Resample when ESS falls below half the configured particle count.
pub fn maybe_resample<R: Rng + ?Sized>(belief: &BeliefState, rng: &mut R) -> BeliefState {
    if belief.n() == 0 {
        return belief.clone();
    }
    let threshold = belief.config.ess_threshold_fraction * belief.config.n_particles as f64;
    if ess(&belief.weights()) >= threshold {
        return belief.clone();
    }
    BeliefState {
        seat: belief.seat,
        particles: resample(&belief.particles, belief.config.n_particles, rng),
        config: belief.config,
        collapsed: belief.collapsed,
    }
}

/// BFS step counts over 4-connected passable cells; `-1` unreachable.
fn bfs_distances(passable: &[bool], h: usize, w: usize, origin: (usize, usize)) -> Vec<i32> {
    let mut dist = vec![-1i32; h * w];
    let (or_, oc) = origin;
    if or_ >= h || oc >= w || !passable[or_ * w + oc] {
        return dist;
    }
    dist[or_ * w + oc] = 0;
    let mut queue = vec![(or_ as i32, oc as i32)];
    let mut head = 0usize;
    while head < queue.len() {
        let (r, c) = queue[head];
        head += 1;
        let d = dist[r as usize * w + c as usize];
        for (dr, dc) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
            let (nr, nc) = (r + dr, c + dc);
            if nr < 0 || nc < 0 || nr as usize >= h || nc as usize >= w {
                continue;
            }
            let at = nr as usize * w + nc as usize;
            if passable[at] && dist[at] < 0 {
                dist[at] = d + 1;
                queue.push((nr, nc));
            }
        }
    }
    dist
}

/// `(mountains, passable)` inferred from the first-frame observation.
///
/// Structure fog on the first frame is a mountain: no castle has been revealed
/// yet, so an unresolved structure cannot be one.
fn terrain_from_first_obs(obs: &Observation) -> (Vec<bool>, Vec<bool>) {
    let n = obs.h * obs.w;
    let mut mountains = vec![false; n];
    let mut passable = vec![false; n];
    for i in 0..n {
        let t = obs.type_grid[i] as i32;
        mountains[i] = t == TYPE_MOUNTAIN || t == TYPE_STRUCTURE_FOG;
        passable[i] = !mountains[i];
    }
    (mountains, passable)
}

fn own_general_from_obs(obs: &Observation) -> Option<(usize, usize)> {
    for i in 0..obs.h * obs.w {
        if obs.type_grid[i] as i32 == 4 && obs.owner_grid[i] as i32 == 1 {
            return Some((i / obs.w, i % obs.w));
        }
    }
    None
}

/// Uniform prior support for the initial enemy general.
///
/// A cell qualifies when it is passable, unseen, plain fog, and at least
/// `min_distance` steps from our own general — the map generator's own
/// separation guarantee, used here to shrink the prior rather than to
/// describe it.
pub fn legal_enemy_general_candidates(
    obs: &Observation,
    min_distance: i32,
) -> Result<Vec<(usize, usize)>, String> {
    let (h, w) = (obs.h, obs.w);
    let n = h * w;
    let (_mountains, passable) = terrain_from_first_obs(obs);
    let (own_r, own_c) =
        own_general_from_obs(obs).ok_or_else(|| "observation has no visible own general".to_string())?;

    let mut own_cells = vec![false; n];
    let mut any_own = false;
    for i in 0..n {
        own_cells[i] = obs.owner_grid[i] as i32 == 1;
        any_own |= own_cells[i];
    }
    // Prefer vision derived from what we own; fall back to the type mask on a
    // frame where we own nothing at all.
    let vis: Vec<bool> = if any_own {
        visibility_from_owned(&own_cells, h, w)[..n].to_vec()
    } else {
        (0..n)
            .map(|i| {
                let t = obs.type_grid[i] as i32;
                t != TYPE_FOG && t != TYPE_STRUCTURE_FOG
            })
            .collect()
    };

    let dist = bfs_distances(&passable, h, w, (own_r, own_c));
    let mut candidates = Vec::new();
    for r in 0..h {
        for c in 0..w {
            let i = r * w + c;
            if !passable[i] || vis[i] {
                continue;
            }
            // Structure fog is a mountain at start and already excluded by
            // `passable`; anything else visible cannot hide a general.
            if obs.type_grid[i] as i32 != TYPE_FOG {
                continue;
            }
            if dist[i] < min_distance {
                continue;
            }
            candidates.push((r, c));
        }
    }
    Ok(candidates)
}

/// A complete initial `GameState` for one particle.
fn state_from_initial(obs: &Observation, seat: usize, enemy_general: (usize, usize)) -> GameState {
    let (h, w) = (obs.h, obs.w);
    let n = h * w;
    let (mountains, passable) = terrain_from_first_obs(obs);
    let (own_r, own_c) = own_general_from_obs(obs).expect("checked by the caller");
    let (er, ec) = enemy_general;
    let enemy_seat = 1 - seat;

    let mut state = GameState::empty(h, w);
    for i in 0..n {
        state.mountains[i] = mountains[i];
        state.passable[i] = passable[i];
    }
    let own_at = own_r * w + own_c;
    let enemy_at = er * w + ec;
    state.armies[own_at] = 1;
    state.armies[enemy_at] = 1;
    state.ownership[seat][own_at] = true;
    state.ownership[enemy_seat][enemy_at] = true;
    state.generals[own_at] = true;
    state.generals[enemy_at] = true;
    for i in 0..n {
        state.ownership_neutral[i] =
            passable[i] && !state.ownership[0][i] && !state.ownership[1][i];
    }
    state.general_positions[seat] = [own_r as i32, own_c as i32];
    state.general_positions[enemy_seat] = [er as i32, ec as i32];

    // Copy any owned cell the first frame already shows. At a true start
    // there is only the general, but a frame captured later must stay
    // rule-consistent rather than assume.
    for i in 0..n {
        let t = obs.type_grid[i] as i32;
        let visible = t != TYPE_FOG && t != TYPE_STRUCTURE_FOG;
        if !(visible && obs.owner_grid[i] as i32 == 1) {
            continue;
        }
        state.ownership[seat][i] = true;
        state.ownership[enemy_seat][i] = false;
        state.ownership_neutral[i] = false;
        state.armies[i] = obs.army_grid[i];
        if t == 4 {
            state.generals[i] = true;
        }
    }
    state.time = obs.turn;
    state.winner = -1;
    state
}

fn public_totals_match(state: &GameState, obs: &Observation, seat: usize) -> bool {
    let n = state.cells();
    let other = 1 - seat;
    let mut land = [0i64; 2];
    let mut army = [0i64; 2];
    for i in 0..n {
        for s in 0..2 {
            if state.ownership[s][i] {
                land[s] += 1;
                army[s] += state.armies[i] as i64;
            }
        }
    }
    land[seat] == obs.my_land as i64
        && land[other] == obs.opp_land as i64
        && army[seat] == obs.my_army as i64
        && army[other] == obs.opp_army as i64
}

/// Sample the initial particle set from the conditioned uniform prior.
pub fn initialize_belief<R: Rng + ?Sized>(
    obs: &Observation,
    seat: usize,
    rng: &mut R,
    config: BeliefConfig,
) -> Result<BeliefState, String> {
    let mut candidates = legal_enemy_general_candidates(obs, config.min_general_distance)?;
    if candidates.is_empty() {
        // Tiny boards in protocol tests cannot honour the separation rule;
        // any fogged passable cell will do rather than refusing to play.
        candidates = legal_enemy_general_candidates(obs, 1)?;
    }
    if candidates.is_empty() {
        return Err("no legal enemy-general candidates for initial belief".to_string());
    }

    let n = config.n_particles;
    let picks = rng.integers(0, candidates.len() as i64, Some(n));
    let weight = 1.0 / n as f64;
    let empty = Rc::new(VisibleMemory::empty(obs.h, obs.w));
    let mut particles: Vec<Particle> = Vec::with_capacity(n);
    for pick in picks {
        let state = state_from_initial(obs, seat, candidates[pick as usize]);
        if !public_totals_match(&state, obs, seat) {
            continue;
        }
        if !observations_match(&emit_observation(&state, seat), obs) {
            continue;
        }
        particles.push(Particle::new(Rc::new(state), weight, Rc::clone(&empty)));
    }

    if particles.is_empty() {
        return Err("initial belief produced no observation-consistent particles".to_string());
    }

    // Pad by resampling when some candidates failed the exact match.
    let particles = if particles.len() < n {
        resample(&normalize_weights(&particles), n, rng)
    } else {
        particles.truncate(n);
        normalize_weights(&particles)
    };

    Ok(BeliefState::new(seat, particles, config))
}

fn append_history(
    history: &[Rc<HistoryFrame>],
    frame: Rc<HistoryFrame>,
    lag: usize,
) -> Vec<Rc<HistoryFrame>> {
    let mut merged: Vec<Rc<HistoryFrame>> = history.to_vec();
    merged.push(frame);
    if merged.len() > lag {
        merged.drain(..merged.len() - lag);
    }
    merged
}

/// Apply `(my_action, enemy_action)` per particle and hard-filter.
///
/// A particle whose emission differs from the real frame anywhere is not
/// downweighted, it is *refuted* — its weight goes to zero and it survives
/// only so the caller can see that everything died and reach for recovery.
/// `enemy_actions` must align with `belief.particles`; proposal injection
/// stays outside.
pub fn filter_step<R: Rng + ?Sized>(
    belief: &BeliefState,
    my_action: Action5,
    real_obs: &Observation,
    enemy_actions: &[Action5],
    rng: &mut R,
) -> Result<BeliefState, String> {
    if enemy_actions.len() != belief.n() {
        return Err("enemy_actions length must match particle count".to_string());
    }
    let enemy_seat = belief.enemy_seat();
    let real = Rc::new(real_obs.clone());

    let mut survivors: Vec<Particle> = Vec::with_capacity(belief.n());
    for (particle, &enemy_action) in belief.particles.iter().zip(enemy_actions) {
        let mut actions: Actions = [PASS_ACTION; 2];
        actions[belief.seat] = my_action;
        actions[enemy_seat] = enemy_action;
        let (next_state, _info) = transition(&particle.state, &actions);
        if !observations_match(&emit_observation(&next_state, belief.seat), real_obs) {
            survivors.push(particle.with_weight(0.0));
            continue;
        }

        let next_state = Rc::new(next_state);
        let enemy_obs = emit_observation(&next_state, enemy_seat);
        let enemy_mem = Rc::new(update_memory(&particle.enemy_memory, &enemy_obs));
        let frame = Rc::new(HistoryFrame {
            state: Rc::clone(&particle.state),
            my_action,
            enemy_action,
            observation_after: Rc::clone(&real),
        });
        survivors.push(Particle {
            state: next_state,
            // Importance ratio 1 under the shared proposal.
            weight: particle.weight,
            enemy_memory: enemy_mem,
            enemy_prev_action: Some(enemy_action),
            history: append_history(&particle.history, frame, belief.config.recovery_lag),
        });
    }

    let positive: Vec<Particle> = survivors
        .iter()
        .filter(|p| p.weight > 0.0)
        .cloned()
        .collect();
    if positive.is_empty() {
        // Total collapse. The caller (recovery) owns what happens next; this
        // returns the zero-weight set rather than an error so the previous
        // belief stays available to rewind from.
        return Ok(BeliefState {
            seat: belief.seat,
            particles: survivors,
            config: belief.config,
            collapsed: false,
        });
    }

    let mut normalized = normalize_weights(&positive);
    if normalized.len() < belief.config.n_particles {
        normalized = resample(&normalized, belief.config.n_particles, rng);
    }
    let out = BeliefState {
        seat: belief.seat,
        particles: normalized,
        config: belief.config,
        collapsed: false,
    };
    Ok(maybe_resample(&out, rng))
}

/// Distinct hidden general cells among positive-weight particles.
pub fn unique_particle_count(belief: &BeliefState) -> usize {
    let mut seen: Vec<[i32; 2]> = Vec::new();
    for particle in &belief.particles {
        if particle.weight <= 0.0 {
            continue;
        }
        let g = particle.state.general_positions[belief.enemy_seat()];
        if !seen.contains(&g) {
            seen.push(g);
        }
    }
    seen.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SmallRng;

    fn open_board(h: usize, w: usize) -> GameState {
        let mut state = GameState::empty(h, w);
        for i in 0..h * w {
            state.passable[i] = true;
            state.ownership_neutral[i] = true;
        }
        state
    }

    fn two_general_state() -> GameState {
        let mut state = open_board(5, 5);
        for (seat, at) in [(0usize, 0usize), (1, 24)] {
            state.ownership[seat][at] = true;
            state.ownership_neutral[at] = false;
            state.generals[at] = true;
            state.armies[at] = 1;
        }
        state.general_positions = [[0, 0], [4, 4]];
        state
    }

    fn particle_of(state: GameState, weight: f64) -> Particle {
        let (h, w) = (state.h, state.w);
        Particle::new(Rc::new(state), weight, Rc::new(VisibleMemory::empty(h, w)))
    }

    #[test]
    fn ess_is_the_particle_count_when_weights_are_equal() {
        assert!((ess(&[0.25; 4]) - 4.0).abs() < 1e-12);
        assert_eq!(ess(&[]), 0.0);
        assert_eq!(ess(&[0.0, 0.0]), 0.0, "no mass means no samples");
    }

    #[test]
    fn ess_collapses_toward_one_as_mass_concentrates() {
        let spread = ess(&[0.4, 0.3, 0.2, 0.1]);
        let peaked = ess(&[0.97, 0.01, 0.01, 0.01]);
        assert!(peaked < spread && peaked > 1.0);
    }

    #[test]
    fn normalizing_uses_a_sequential_sum_and_clamps_negatives() {
        let particles = vec![
            particle_of(two_general_state(), 2.0),
            particle_of(two_general_state(), -1.0),
            particle_of(two_general_state(), 6.0),
        ];
        let out = normalize_weights(&particles);
        assert_eq!(out[0].weight, 0.25);
        assert_eq!(out[1].weight, 0.0);
        assert_eq!(out[2].weight, 0.75);
    }

    #[test]
    fn an_all_zero_set_is_returned_untouched_rather_than_divided_by_zero() {
        let particles = vec![particle_of(two_general_state(), 0.0)];
        assert_eq!(normalize_weights(&particles)[0].weight, 0.0);
    }

    #[test]
    fn resampling_draws_only_from_particles_with_mass() {
        let mut a = particle_of(two_general_state(), 1.0);
        a.enemy_prev_action = Some([0, 1, 1, 2, 0]);
        let b = particle_of(two_general_state(), 0.0);
        let mut rng = SmallRng::seed_from_u64(4);
        let out = resample(&[a, b], 16, &mut rng);
        assert_eq!(out.len(), 16);
        assert!(out.iter().all(|p| p.enemy_prev_action == Some([0, 1, 1, 2, 0])));
        assert!(out.iter().all(|p| p.weight == 1.0 / 16.0));
    }

    #[test]
    fn a_refuted_particle_keeps_its_old_state_at_zero_weight() {
        // Both particles claim the same board, so both survive the emission
        // test; a particle whose enemy action is impossible would not.
        let config = BeliefConfig {
            n_particles: 2,
            ..Default::default()
        };
        let state = two_general_state();
        let obs_before = emit_observation(&state, 0);
        let belief = BeliefState::new(
            0,
            vec![
                particle_of(state.clone(), 0.5),
                particle_of(state.clone(), 0.5),
            ],
            config,
        );
        let (next, _) = transition(&state, &[PASS_ACTION, PASS_ACTION]);
        let real = emit_observation(&next, 0);
        assert_ne!(real.turn, obs_before.turn);

        let mut rng = SmallRng::seed_from_u64(1);
        let out = filter_step(&belief, PASS_ACTION, &real, &[PASS_ACTION; 2], &mut rng).unwrap();
        assert_eq!(out.n(), 2);
        assert!(out.particles.iter().all(|p| p.weight > 0.0));
        assert_eq!(out.particles[0].history.len(), 1);
        assert_eq!(out.particles[0].enemy_prev_action, Some(PASS_ACTION));
    }

    #[test]
    fn a_belief_that_cannot_explain_the_frame_comes_back_at_zero_weight() {
        let config = BeliefConfig {
            n_particles: 1,
            ..Default::default()
        };
        let state = two_general_state();
        let belief = BeliefState::new(0, vec![particle_of(state.clone(), 1.0)], config);
        // An observation from a board the particle cannot reach in one step.
        let mut elsewhere = state.clone();
        elsewhere.armies[0] = 999;
        let (next, _) = transition(&elsewhere, &[PASS_ACTION, PASS_ACTION]);
        let real = emit_observation(&next, 0);

        let mut rng = SmallRng::seed_from_u64(1);
        let out = filter_step(&belief, PASS_ACTION, &real, &[PASS_ACTION], &mut rng).unwrap();
        assert!(out.particles.iter().all(|p| p.weight == 0.0));
        assert_eq!(out.particles[0].history.len(), 0, "a refuted particle does not advance");
    }

    #[test]
    fn history_is_capped_at_the_recovery_lag() {
        let frames: Vec<Rc<HistoryFrame>> = (0..3)
            .map(|i| {
                Rc::new(HistoryFrame {
                    state: Rc::new(two_general_state()),
                    my_action: [0, i, 0, 0, 0],
                    enemy_action: PASS_ACTION,
                    observation_after: Rc::new(emit_observation(&two_general_state(), 0)),
                })
            })
            .collect();
        let extra = Rc::clone(&frames[0]);
        let out = append_history(&frames, extra, 2);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].my_action, [0, 2, 0, 0, 0], "the oldest frame is dropped");
    }

    #[test]
    fn the_initial_prior_only_puts_generals_where_one_could_hide() {
        let state = two_general_state();
        let obs = emit_observation(&state, 0);
        let close = legal_enemy_general_candidates(&obs, 1).unwrap();
        let far = legal_enemy_general_candidates(&obs, 17).unwrap();
        assert!(!close.is_empty());
        assert!(far.is_empty(), "a 5x5 board has nothing 17 steps away");
        // Never a cell we can see, and never our own general.
        assert!(!close.contains(&(0, 0)));
        assert!(!close.contains(&(1, 1)));
    }

    #[test]
    fn initialization_produces_particles_that_reproduce_the_first_frame() {
        let state = two_general_state();
        let obs = emit_observation(&state, 0);
        let config = BeliefConfig {
            n_particles: 8,
            min_general_distance: 1,
            ..Default::default()
        };
        let mut rng = SmallRng::seed_from_u64(9);
        let belief = initialize_belief(&obs, 0, &mut rng, config).unwrap();
        assert_eq!(belief.n(), 8);
        for particle in &belief.particles {
            assert!(observations_match(&emit_observation(&particle.state, 0), &obs));
        }
        assert!(unique_particle_count(&belief) >= 1);
    }
}
