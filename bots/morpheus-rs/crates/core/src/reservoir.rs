//! A bounded particle bag attached to one search node.
//!
//! Port of `bots/morpheus/reservoir.py`. The search keys nodes by observable
//! history, and every node collects the particles that reached it. Unbounded
//! that is a memory leak with a plausible name, so admission is Algorithm R:
//! the `t`-th arrival replaces a uniformly chosen resident with probability
//! `capacity/t`, which keeps a uniform sample of everything that ever arrived
//! without storing it.
//!
//! Nothing in M4 drives this — the search does, at M5. It is ported here
//! because rewrite-plan §4 puts `reservoir.py` in this milestone and because
//! it is the last piece of the belief layer that M5 would otherwise have to
//! stop and write.
//!
//! ## The fixed seed inside `replace_from_belief`
//!
//! The Python builds `np.random.default_rng(0)` *inside* the method, so the
//! resample there does not consume the bot's shared stream. That is a
//! deliberate property — the node's contents must not depend on how much
//! searching happened before — and it is preserved here by taking the
//! generator as an argument, so a caller can hand over a dedicated one.
//!
//! What is **not** preserved is which particles that generator picks.
//! Reproducing `default_rng(0)` would mean reimplementing PCG64 and NumPy's
//! `choice`, which rewrite-plan §5 explicitly declined to do. The inputs are
//! equal-weight after `normalize_weights`, so both implementations draw from
//! the same distribution and differ only in the sample. The parity harness
//! injects the oracle's recorded draws and therefore checks this exactly; in
//! play the two bots make different, equally valid picks. That is the one
//! place in the belief layer where "same distribution" replaces "same
//! answer", and it is recorded here rather than discovered later.

use crate::belief::{normalize_weights, resample, BeliefConfig, BeliefState, Particle};
use crate::support::rng::{npsum, Rng};

pub struct ParticleReservoir {
    pub capacity: usize,
    pub particles: Vec<Particle>,
    /// Every arrival ever offered, including the ones Algorithm R rejected.
    pub admitted_count: usize,
    /// Bumped whenever the particle list changes; backup caches enemy hashes
    /// against it, so a stale cache is detectable rather than silent.
    pub version: u64,
}

impl ParticleReservoir {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            particles: Vec::new(),
            admitted_count: 0,
            version: 0,
        }
    }

    #[inline]
    pub fn n(&self) -> usize {
        self.particles.len()
    }

    pub fn clear(&mut self) {
        self.particles.clear();
        self.admitted_count = 0;
        self.version += 1;
    }

    /// Copy the filtered belief into this reservoir.
    ///
    /// See the module note on the generator: pass one whose stream nothing
    /// else consumes.
    pub fn replace_from_belief<R: Rng + ?Sized>(&mut self, belief: &BeliefState, rng: &mut R) {
        let positive: Vec<Particle> = belief
            .particles
            .iter()
            .filter(|p| p.weight > 0.0)
            .cloned()
            .collect();
        if positive.is_empty() {
            self.particles = Vec::new();
            self.admitted_count = 0;
            self.version += 1;
            return;
        }
        let n = self.capacity.min(belief.config.n_particles);
        self.particles = resample(&normalize_weights(&positive), n, rng);
        self.admitted_count = self.particles.len();
        self.version += 1;
    }

    /// Draw one resident, by weight.
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<&Particle, String> {
        if self.particles.is_empty() {
            return Err("empty particle reservoir".to_string());
        }
        let weights: Vec<f64> = self.particles.iter().map(|p| p.weight.max(0.0)).collect();
        let total = npsum(&weights);
        let index = if total <= 0.0 {
            rng.integer(0, self.particles.len() as i64) as usize
        } else {
            let probs: Vec<f64> = weights.iter().map(|w| w / total).collect();
            rng.choice_one(self.particles.len(), Some(&probs))
        };
        Ok(&self.particles[index])
    }

    /// Admit one arriving particle; Algorithm R once over capacity.
    ///
    /// The arrival is stored at weight 1, not at the weight it carried: a
    /// reservoir is a uniform sample of arrivals, and importing the filter's
    /// weights would count the same evidence twice.
    pub fn admit<R: Rng + ?Sized>(&mut self, particle: &Particle, rng: &mut R) {
        let arriving = particle.with_weight(1.0);
        self.admitted_count += 1;
        if self.n() < self.capacity {
            self.particles.push(arriving);
            self.version += 1;
            return;
        }
        let t = self.admitted_count as f64;
        if rng.random_one() < (self.capacity as f64 / t) {
            let index = rng.integer(0, self.capacity as i64) as usize;
            self.particles[index] = arriving;
            self.version += 1;
        }
    }

    /// The reservoir as a belief, for code that only knows that shape.
    pub fn as_belief(&self, seat: usize, config: Option<BeliefConfig>) -> BeliefState {
        BeliefState::new(
            seat,
            self.particles.clone(),
            config.unwrap_or(BeliefConfig {
                n_particles: self.capacity,
                ..Default::default()
            }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::VisibleMemory;
    use crate::support::rng::SmallRng;
    use crate::state::GameState;
    use std::rc::Rc;

    fn particle(tag: i32, weight: f64) -> Particle {
        let mut state = GameState::empty(2, 2);
        state.armies[0] = tag;
        Particle::new(Rc::new(state), weight, Rc::new(VisibleMemory::empty(2, 2)))
    }

    #[test]
    fn arrivals_fill_the_reservoir_before_anything_is_evicted() {
        let mut reservoir = ParticleReservoir::new(3);
        let mut rng = SmallRng::seed_from_u64(1);
        for tag in 0..3 {
            reservoir.admit(&particle(tag, 0.5), &mut rng);
        }
        assert_eq!(reservoir.n(), 3);
        assert_eq!(reservoir.admitted_count, 3);
        assert!(reservoir.particles.iter().all(|p| p.weight == 1.0));
        assert_eq!(reservoir.version, 3);
    }

    #[test]
    fn a_full_reservoir_stays_at_capacity_however_many_arrive() {
        let mut reservoir = ParticleReservoir::new(4);
        let mut rng = SmallRng::seed_from_u64(2);
        for tag in 0..200 {
            reservoir.admit(&particle(tag, 1.0), &mut rng);
        }
        assert_eq!(reservoir.n(), 4);
        assert_eq!(reservoir.admitted_count, 200);
    }

    #[test]
    fn algorithm_r_keeps_admitting_late_arrivals_but_rarely() {
        // With capacity 4 and 400 arrivals the acceptance rate falls as 4/t,
        // so the tail is sampled but does not dominate. The assertion is on
        // the shape, not on a particular draw: some evictions happen, and far
        // fewer than the number of arrivals.
        let mut reservoir = ParticleReservoir::new(4);
        let mut rng = SmallRng::seed_from_u64(9);
        for tag in 0..400 {
            reservoir.admit(&particle(tag, 1.0), &mut rng);
        }
        let evictions = reservoir.version - 4;
        assert!(evictions > 0, "nothing was ever replaced");
        assert!(evictions < 100, "{evictions} replacements is not 4/t");
    }

    #[test]
    fn sampling_an_empty_reservoir_is_an_error_not_a_panic() {
        let reservoir = ParticleReservoir::new(2);
        let mut rng = SmallRng::seed_from_u64(1);
        assert!(reservoir.sample(&mut rng).is_err());
    }

    #[test]
    fn a_belief_with_no_mass_empties_the_reservoir_rather_than_filling_it() {
        let mut reservoir = ParticleReservoir::new(4);
        reservoir.admit(&particle(1, 1.0), &mut SmallRng::seed_from_u64(1));
        let belief = BeliefState::new(
            0,
            vec![particle(2, 0.0), particle(3, 0.0)],
            BeliefConfig {
                n_particles: 4,
                ..Default::default()
            },
        );
        reservoir.replace_from_belief(&belief, &mut SmallRng::seed_from_u64(0));
        assert_eq!(reservoir.n(), 0);
        assert_eq!(reservoir.admitted_count, 0);
    }

    #[test]
    fn replacing_from_a_belief_takes_the_smaller_of_the_two_counts() {
        let mut reservoir = ParticleReservoir::new(3);
        let belief = BeliefState::new(
            0,
            vec![particle(1, 0.5), particle(2, 0.5)],
            BeliefConfig {
                n_particles: 8,
                ..Default::default()
            },
        );
        reservoir.replace_from_belief(&belief, &mut SmallRng::seed_from_u64(0));
        assert_eq!(reservoir.n(), 3, "capacity binds below the belief's count");
        assert_eq!(reservoir.admitted_count, 3);
    }
}
