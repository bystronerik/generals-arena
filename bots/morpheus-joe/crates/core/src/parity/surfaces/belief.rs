//! The particle filter, end to end and one step at a time.

use crate::belief::{filter_step, BeliefConfig};
use crate::board::memory::update_memory;
use crate::board::observe::emit_observation;
use crate::belief::proposal::{
    propose_enemy_actions, uniform_legal_probs,
    ProposalTelemetry,
};
use crate::belief::recovery::maximum_entropy_reconstruction;
use crate::belief::reservoir::ParticleReservoir;
use crate::support::rng::Replay;
use crate::parity::codec::*;
use crate::parity::ints::Ints;
use crate::parity::Ctx;

/// first-frame observation + seat + config + draws -> the initial
/// particle set, or a refusal
///
/// The corpus cannot reach this by replay: it records beliefs that
/// already exist, never the frame that created one. Mutation
/// testing found the consequence — deleting the minimum-separation
/// rule and deleting the "never a cell we can see" rule both
/// survived every other surface, because nothing called
/// `legal_enemy_general_candidates` at all.
pub(in crate::parity) fn initbelief(
    ints: &mut Ints,
    out: &mut Vec<i64>,
    _ctx: &mut Ctx,
) -> Result<(), String> {
    let obs = read_observation(ints)?;
    let seat = ints.n()?;
    let n_particles = ints.n()?;
    let min_general_distance = ints.next()? as i32;
    let mut rng = Replay::new(read_draws(ints)?);
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
            write_belief(out, &belief);
        }
        Err(_) => out.push(0),
    }
    out.push(rng.consumed() as i64);
    Ok(())
}

/// a belief + a recorded draw stream -> each particle's proposal
/// distribution and the action sampled from it
///
/// Both halves, deliberately. `Replay` hands back the oracle's
/// index whatever distribution this side computed, so checking only
/// the sampled actions would pass even if every probability were
/// wrong. The distribution rides out **sparsely** — a uniform mask
/// has a few hundred non-zeros out of 3,970 — which keeps the
/// channel honest without making it enormous.
pub(in crate::parity) fn propose(
    ints: &mut Ints,
    out: &mut Vec<i64>,
    _ctx: &mut Ctx,
) -> Result<(), String> {
    let belief = read_belief(ints)?;
    let mut rng = Replay::new(read_draws(ints)?);
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
        propose_enemy_actions(&belief, &mut rng, Some(&mut telemetry));
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
    Ok(())
}

/// belief + my action + the real frame + one enemy action per
/// particle + draws -> the filtered belief
pub(in crate::parity) fn filter(
    ints: &mut Ints,
    out: &mut Vec<i64>,
    _ctx: &mut Ctx,
) -> Result<(), String> {
    let belief = read_belief(ints)?;
    let my_action = ints.action()?;
    let real_obs = read_observation(ints)?;
    let mut enemy_actions = Vec::with_capacity(belief.n());
    for _ in 0..belief.n() {
        enemy_actions.push(ints.action()?);
    }
    let mut rng = Replay::new(read_draws(ints)?);
    let next = filter_step(&belief, my_action, &real_obs, &enemy_actions, &mut rng)?;
    write_belief(out, &next);
    out.push(rng.consumed() as i64);
    Ok(())
}

/// a belief carrying histories + draws -> the rejuvenated belief
pub(in crate::parity) fn rejuvenate(
    ints: &mut Ints,
    out: &mut Vec<i64>,
    _ctx: &mut Ctx,
) -> Result<(), String> {
    let belief = read_belief(ints)?;
    let mut rng = Replay::new(read_draws(ints)?);
    write_belief(out, &crate::belief::recovery::rejuvenate(&belief, &mut rng));
    out.push(rng.consumed() as i64);
    Ok(())
}

/// observation + seat + memory + particle count + draws -> the
/// maximum-entropy reconstruction, or a refusal
///
/// The refusal is part of the contract: the Python raises
/// `ValueError` on an inconsistent frame and `recover_belief`
/// catches it to keep the last valid belief. A port that
/// reconstructed something anyway would look correct here and be
/// wrong in play, so the `ok` flag is compared before the belief.
pub(in crate::parity) fn maxent(
    ints: &mut Ints,
    out: &mut Vec<i64>,
    _ctx: &mut Ctx,
) -> Result<(), String> {
    let obs = read_observation(ints)?;
    let seat = ints.n()?;
    let memory = read_memory(ints)?;
    let n_particles = ints.n()?;
    let mut rng = Replay::new(read_draws(ints)?);
    let config = BeliefConfig {
        n_particles,
        ..Default::default()
    };
    match maximum_entropy_reconstruction(&obs, seat, &memory, &mut rng, config) {
        Ok(belief) => {
            out.push(1);
            write_belief(out, &belief);
        }
        Err(_) => out.push(0),
    }
    out.push(rng.consumed() as i64);
    Ok(())
}

/// capacity + arrivals + a belief + draws -> the reservoir after a
/// fixed script: admit everything twice, sample once, then replace
/// from the belief
///
/// The script is fixed rather than corpus-driven because nothing
/// the reservoir does depends on which board is in a particle. What
/// it depends on is arrival *count* against capacity, which the
/// double pass crosses, and the fixed-seed resample in
/// `replace_from_belief` — the one place in the belief layer where
/// the Python's generator is a local, and where play-time answers
/// legitimately differ (see `reservoir.rs`).
pub(in crate::parity) fn reservoir(
    ints: &mut Ints,
    out: &mut Vec<i64>,
    _ctx: &mut Ctx,
) -> Result<(), String> {
    let capacity = ints.n()?;
    let seat = ints.n()?;
    let arrivals = ints.n()?;
    let mut incoming = Vec::with_capacity(arrivals);
    for _ in 0..arrivals {
        incoming.push(read_particle(ints, seat)?);
    }
    let belief = read_belief(ints)?;
    let mut rng = Replay::new(read_draws(ints)?);

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
        write_state(out, &particle.state);
    }

    match reservoir.sample(&mut rng) {
        Ok(particle) => {
            out.push(1);
            write_state(out, &particle.state);
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
        write_state(out, &particle.state);
    }
    out.push(rng.consumed() as i64);
    Ok(())
}
