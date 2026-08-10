//! `bench_belief`: the belief layer's cost, measured where it is bimodal.
//!
//! A benchmark that only ever feeds explainable observations reports the
//! happy-path number against a baseline whose p99 is dominated by recovery, so
//! each case carries both kinds and both are timed.

use std::io::{BufRead, Write};

use crate::belief::filter_step;
use crate::board::hashing::observation_payload;
use crate::board::memory::VisibleMemory;
use crate::belief::proposal::propose_enemy_actions;
use crate::belief::recovery::recover_belief;
use crate::io::wire::Observation;
use crate::parity::codec::*;
use crate::parity::ints::Ints;

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
pub(in crate::parity) struct VaryingEvaluator {
    pub(in crate::parity) inner: crate::search::ScriptedEvaluator,
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
