//! Enemy-action proposals: what each particle assumes the opponent just did.
//!
//! Port of `bots/morpheus/proposal.py`. The filter needs one enemy action per
//! particle before it can advance any of them, and the quality of that guess
//! decides how many particles survive the observation test.
//!
//! **The proposal is rules-only, and there is no longer a policy path.** The
//! deployed path was already the uniform one: `deployment.json` shipped
//! `use_policy_proposal: false`, on a measurement — against macaria over 100
//! games per arm the learned proposal was worth `-0.07 ± 0.13` paired, and
//! skipping its forward returned ~13 ms per turn to the search. N1 removed the
//! other branch outright (joe-net-plan §3.3), for two reasons that are not
//! about that measurement:
//!
//! * joe's net needs the **enemy's** `AugState` — the enemy's seven-turn delta
//!   history, its accumulated latches, and its 512-step window over our own
//!   totals. The belief tracks a hypothesized `enemy_memory` per particle and
//!   nothing to reconstruct 44.8 KB of unobservable history from, so the
//!   input the policy branch would need does not exist (§3.4).
//! * The budget has room for 3–5 forwards on a whole turn (N0), so the
//!   proposal could not have one in any case.
//!
//! A knob that cannot be turned back on is a knob nobody can re-qualify, so
//! `use_policy_proposal` is gone from `deployment.json` rather than left in it
//! as a lie. [`policy_action_probs`] keeps its name and its callers and always
//! takes the uniform branch, which is what it already did in every rated game.
//!
//! ## Two float details that are not decoration
//!
//! [`softmax_masked`] sums 3,970 terms with [`npsum`], because `np.sum` is
//! pairwise and this total is a divisor. And `top_legal_actions` ranks with
//! [`argsort_desc_numpy`] rather than a stable sort, because under the uniform
//! proposal *every legal action carries the same probability* — the ordering
//! recovery walks is decided entirely by how NumPy's introsort breaks ties.

use crate::board::action::{decode_action, encode_action, legal_mask, N_ACTIONS, PASS_INDEX};
use crate::belief::{BeliefState, Particle, PASS_ACTION};
use crate::board::hashing::{memory_digest, observation_payload};
use crate::board::memory::{update_memory, VisibleMemory};
use crate::board::observe::emit_observation;
use crate::support::rng::{argsort_desc_numpy, npsum, Rng};
use crate::support::sha256::sha256;
use crate::io::wire::Observation;

pub type Action5 = [i32; 5];

/// Masks with this many legal actions or fewer are already decided.
/// Pass is always legal, so early turns are typically a single legal action.
pub const SINGLETON_LEGAL_MAX: usize = 1;

/// Passive counters for one [`propose_enemy_actions`] call.
///
/// Telemetry never changes a sampled action; it exists so the runtime can
/// report how much of the proposal budget the dedupe actually saved.
///
/// The last two fields are **pinned at zero** now that the proposal is
/// rules-only. They are kept rather than removed because the `propose` parity
/// surface's integer layout is shared with the Python oracle, which computes
/// the same two counters and reports the same zeros on the uniform branch it
/// has always taken.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct ProposalTelemetry {
    pub n_particles: usize,
    pub n_singleton_particles: usize,
    pub n_unique_info_keys: usize,
    pub n_unique_policy_inputs: usize,
    pub n_policy_batches: usize,
}

/// Softmax over the legal actions only, in `f64`.
pub fn softmax_masked(logits: &[f64], mask: &[bool]) -> Vec<f64> {
    let mut out = vec![0.0f64; logits.len()];
    if !mask.iter().any(|&m| m) {
        out[PASS_INDEX] = 1.0;
        return out;
    }
    // Illegal logits are pushed to -1e9 *before* the max, so an all-illegal
    // maximum cannot dominate the shift.
    let clipped: Vec<f64> = logits
        .iter()
        .zip(mask)
        .map(|(&l, &m)| if m { l } else { -1e9 })
        .collect();
    let peak = clipped.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = clipped
        .iter()
        .zip(mask)
        .map(|(&c, &m)| if m { (c - peak).exp() } else { 0.0 })
        .collect();
    let total = npsum(&exp);
    if total <= 0.0 {
        out[PASS_INDEX] = 1.0;
        return out;
    }
    for (slot, value) in out.iter_mut().zip(&exp) {
        *slot = value / total;
    }
    out
}

/// One-hot over the sole legal action, or pass when the mask is empty.
pub fn singleton_probs(mask: &[bool]) -> Vec<f64> {
    let mut probs = vec![0.0f64; mask.len()];
    match mask.iter().position(|&m| m) {
        Some(index) => probs[index] = 1.0,
        None => probs[PASS_INDEX] = 1.0,
    }
    probs
}

/// Uniform over currently legal actions; pass is always among them.
pub fn uniform_legal_probs(obs: &Observation, memory: &VisibleMemory) -> Vec<f64> {
    let mask = legal_mask(obs, memory, None);
    let n = mask.iter().filter(|&&m| m).count();
    let mut probs = vec![0.0f64; N_ACTIONS];
    if n == 0 {
        probs[PASS_INDEX] = 1.0;
        return probs;
    }
    let share = 1.0 / n as f64;
    for (slot, &legal) in probs.iter_mut().zip(mask.iter()) {
        if legal {
            *slot = share;
        }
    }
    probs
}

pub fn sample_from_probs<R: Rng + ?Sized>(probs: &[f64], rng: &mut R) -> Action5 {
    let index = rng.choice_one(probs.len(), Some(probs));
    decode_action(index).unwrap_or(PASS_ACTION)
}

/// Pre-tensor dedupe key: enemy observation, enemy memory, previous action.
pub fn proposal_info_key(
    enemy_obs: &Observation,
    enemy_mem: &VisibleMemory,
    previous_action: Option<Action5>,
) -> [u8; 32] {
    let prev_idx: i32 = match previous_action {
        Some(action) => encode_action(action) as i32,
        None => -1,
    };
    let payload = observation_payload(enemy_obs);
    let digest = memory_digest(enemy_mem);
    sha256(&[&payload, &digest, &prev_idx.to_le_bytes()])
}

/// `(unique particle indices, per-particle index into that list)`.
pub fn dedupe_proposal_keys(keys: &[[u8; 32]]) -> (Vec<usize>, Vec<usize>) {
    let mut unique: Vec<usize> = Vec::new();
    let mut mapping: Vec<usize> = Vec::with_capacity(keys.len());
    for (index, key) in keys.iter().enumerate() {
        match unique.iter().position(|&u| &keys[u] == key) {
            Some(at) => mapping.push(at),
            None => {
                mapping.push(unique.len());
                unique.push(index);
            }
        }
    }
    (unique, mapping)
}

/// The enemy observation and folded memory each particle implies.
fn enemy_views(belief: &BeliefState) -> (Vec<Observation>, Vec<VisibleMemory>) {
    let enemy_seat = belief.enemy_seat();
    let mut observations = Vec::with_capacity(belief.n());
    let mut memories = Vec::with_capacity(belief.n());
    for particle in &belief.particles {
        let enemy_obs = emit_observation(&particle.state, enemy_seat);
        memories.push(update_memory(&particle.enemy_memory, &enemy_obs));
        observations.push(enemy_obs);
    }
    (observations, memories)
}

/// Sample one enemy action per particle, uniformly over each particle's own
/// legal mask.
///
/// The information keys are still built and still deduplicated, because
/// `n_unique_info_keys` is what the runtime reports as the proposal's
/// redundancy and the `propose` parity surface checks it. Nothing is skipped
/// on the strength of that dedupe any more: with no network at the far end
/// there is nothing to save.
pub fn propose_enemy_actions<R: Rng + ?Sized>(
    belief: &BeliefState,
    rng: &mut R,
    telemetry: Option<&mut ProposalTelemetry>,
) -> Vec<Action5> {
    let mut counters = ProposalTelemetry::default();
    if belief.n() == 0 {
        if let Some(slot) = telemetry {
            *slot = counters;
        }
        return Vec::new();
    }
    let (enemy_obs_list, enemy_mem_list) = enemy_views(belief);

    let keys: Vec<[u8; 32]> = (0..belief.n())
        .map(|i| {
            proposal_info_key(
                &enemy_obs_list[i],
                &enemy_mem_list[i],
                belief.particles[i].enemy_prev_action,
            )
        })
        .collect();
    counters.n_particles = belief.n();
    counters.n_unique_info_keys = dedupe_proposal_keys(&keys).0.len();

    let mut actions = Vec::with_capacity(belief.n());
    for (obs, memory) in enemy_obs_list.iter().zip(&enemy_mem_list) {
        let probs = uniform_legal_probs(obs, memory);
        if probs.iter().filter(|&&p| p > 0.0).count() <= SINGLETON_LEGAL_MAX {
            counters.n_singleton_particles += 1;
        }
        actions.push(sample_from_probs(&probs, rng));
    }
    if let Some(slot) = telemetry {
        *slot = counters;
    }
    actions
}

/// A length-3970 probability vector for one particle's enemy seat.
///
/// Uniform over the particle's legal mask. It kept its name through the port
/// because that is what every caller in `recovery.rs` asks it for and because
/// the distribution it returns is the one the deployed bot has always used —
/// `use_policy_proposal` was off in every rated game.
pub fn policy_action_probs(belief: &BeliefState, particle: &Particle) -> Vec<f64> {
    let enemy_obs = emit_observation(&particle.state, belief.enemy_seat());
    let enemy_mem = update_memory(&particle.enemy_memory, &enemy_obs);
    uniform_legal_probs(&enemy_obs, &enemy_mem)
}

/// Highest-prior legal actions, always including pass when it is legal.
///
/// Pass goes first unconditionally, then the ranked remainder, deduplicated —
/// the mask and the decode are not injective on the pass slot, so the same
/// action can appear twice without the `seen` set.
pub fn top_legal_actions(probs: &[f64], mask: &[bool], k: usize) -> Vec<Action5> {
    let scores: Vec<f64> = probs
        .iter()
        .zip(mask)
        .map(|(&p, &m)| if m { p } else { -1.0 })
        .collect();
    let order = argsort_desc_numpy(&scores);

    let mut out: Vec<Action5> = Vec::new();
    if mask[PASS_INDEX] {
        out.push(PASS_ACTION);
    }
    for index in order {
        if !mask[index] {
            continue;
        }
        let Some(action) = decode_action(index) else {
            continue;
        };
        if out.contains(&action) {
            continue;
        }
        out.push(action);
        if out.len() >= k + 1 {
            break;
        }
    }
    out
}

/// Hash of the enemy observation, for table and batch deduplication.
pub fn enemy_info_key(belief: &BeliefState, particle: &Particle) -> Vec<u8> {
    observation_payload(&emit_observation(&particle.state, belief.enemy_seat()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belief::BeliefConfig;
    use crate::support::rng::SmallRng;
    use crate::board::state::GameState;
    use std::rc::Rc;

    fn small_belief() -> BeliefState {
        let mut state = GameState::empty(5, 5);
        for i in 0..25 {
            state.passable[i] = true;
            state.ownership_neutral[i] = true;
        }
        for (seat, at, army) in [(0usize, 0usize, 9i32), (1, 24, 9)] {
            state.ownership[seat][at] = true;
            state.ownership_neutral[at] = false;
            state.generals[at] = true;
            state.armies[at] = army;
        }
        state.general_positions = [[0, 0], [4, 4]];
        let memory = Rc::new(VisibleMemory::empty(5, 5));
        BeliefState::new(
            0,
            vec![Particle::new(Rc::new(state), 1.0, memory)],
            BeliefConfig {
                n_particles: 1,
                ..Default::default()
            },
        )
    }

    #[test]
    fn a_uniform_proposal_spreads_evenly_over_the_legal_mask() {
        let belief = small_belief();
        let enemy_obs = emit_observation(&belief.particles[0].state, 1);
        let enemy_mem = update_memory(&belief.particles[0].enemy_memory, &enemy_obs);
        let probs = uniform_legal_probs(&enemy_obs, &enemy_mem);
        let legal: Vec<f64> = probs.iter().copied().filter(|&p| p > 0.0).collect();
        assert!(legal.len() > 1, "a nine-army general has moves");
        assert!(legal.iter().all(|&p| p == legal[0]));
        assert!((npsum(&probs) - 1.0).abs() < 1e-12);
        assert!(probs[PASS_INDEX] > 0.0, "pass is always legal");
    }

    #[test]
    fn the_uniform_proposal_samples_one_action_per_particle() {
        let belief = small_belief();
        let mut rng = SmallRng::seed_from_u64(3);
        let mut telemetry = ProposalTelemetry::default();
        let actions = propose_enemy_actions(&belief, &mut rng, Some(&mut telemetry));
        assert_eq!(actions.len(), 1);
        assert_eq!(telemetry.n_particles, 1);
        assert_eq!(telemetry.n_unique_info_keys, 1);
        assert_eq!(telemetry.n_policy_batches, 0, "no policy, no forward");
    }

    #[test]
    fn a_masked_softmax_puts_no_mass_on_an_illegal_action() {
        let mut mask = vec![false; N_ACTIONS];
        mask[10] = true;
        mask[20] = true;
        mask[PASS_INDEX] = true;
        let logits: Vec<f64> = (0..N_ACTIONS).map(|i| (i % 7) as f64).collect();
        let probs = softmax_masked(&logits, &mask);
        for (index, &p) in probs.iter().enumerate() {
            if !mask[index] {
                assert_eq!(p, 0.0, "index {index}");
            }
        }
        assert!((npsum(&probs) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn an_empty_mask_falls_back_to_pass() {
        let mask = vec![false; N_ACTIONS];
        assert_eq!(softmax_masked(&vec![0.0; N_ACTIONS], &mask)[PASS_INDEX], 1.0);
        assert_eq!(singleton_probs(&mask)[PASS_INDEX], 1.0);
    }

    #[test]
    fn a_singleton_mask_is_deterministic() {
        let mut mask = vec![false; N_ACTIONS];
        mask[77] = true;
        let probs = singleton_probs(&mask);
        assert_eq!(probs[77], 1.0);
        assert_eq!(probs.iter().filter(|&&p| p > 0.0).count(), 1);
    }

    #[test]
    fn identical_particles_share_one_proposal_key() {
        let belief = small_belief();
        let enemy_obs = emit_observation(&belief.particles[0].state, 1);
        let enemy_mem = update_memory(&belief.particles[0].enemy_memory, &enemy_obs);
        let a = proposal_info_key(&enemy_obs, &enemy_mem, None);
        let b = proposal_info_key(&enemy_obs, &enemy_mem, None);
        let c = proposal_info_key(&enemy_obs, &enemy_mem, Some([0, 1, 1, 2, 0]));
        assert_eq!(a, b);
        assert_ne!(a, c, "the previous action is part of the key");

        let (unique, mapping) = dedupe_proposal_keys(&[a, b, c, a]);
        assert_eq!(unique, vec![0, 2]);
        assert_eq!(mapping, vec![0, 0, 1, 0]);
    }

    #[test]
    fn top_legal_actions_leads_with_pass_and_returns_k_plus_one() {
        let mut mask = vec![false; N_ACTIONS];
        for index in [5usize, 9, 40, 300, 900] {
            mask[index] = true;
        }
        mask[PASS_INDEX] = true;
        let mut probs = vec![0.0f64; N_ACTIONS];
        probs[300] = 0.5;
        probs[9] = 0.3;
        probs[40] = 0.1;
        let top = top_legal_actions(&probs, &mask, 2);
        assert_eq!(top.len(), 3);
        assert_eq!(top[0], PASS_ACTION);
        assert_eq!(top[1], decode_action(300).unwrap());
        assert_eq!(top[2], decode_action(9).unwrap());
    }

    #[test]
    fn top_legal_actions_never_offers_an_illegal_move() {
        let mut mask = vec![false; N_ACTIONS];
        mask[11] = true;
        mask[PASS_INDEX] = true;
        let probs = vec![1.0f64 / 2.0; N_ACTIONS];
        for action in top_legal_actions(&probs, &mask, 8) {
            assert!(mask[encode_action(action)], "{action:?} is not legal");
        }
    }
}
