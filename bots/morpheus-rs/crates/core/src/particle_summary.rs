//! Weighted particle aggregates, as the six belief planes the tensor reads.
//!
//! Port of `bots/morpheus/particle_summary.py`. This is the join M2 left open:
//! `tensor.rs` has taken a [`BeliefSummary`] as an input since then, fed by the
//! Python from the recorded corpus, precisely so the M2/M4 boundary stayed a
//! clean one. Nothing else was missing.
//!
//! The planes are in **raw army units**, not compressed — `build_tensor` owns
//! the compression, and doing it twice would be silently wrong rather than
//! loudly wrong.
//!
//! Accumulation is `f64` throughout and narrows to `f32` once, at the end,
//! exactly where the Python's `.astype(np.float32)` sits. M2's finding applies
//! directly: a version that accumulated in `f32` would stay well inside §5's
//! 1e-6 tensor budget and still be wrong, so the harness compares these planes
//! bit-for-bit.

use crate::belief::{ess_fraction, BeliefState};
use crate::observe::visibility_from_owned;
use crate::rng::npsum;
use crate::tensor::BeliefSummary;

/// Weighted particle aggregates for the belief planes.
///
/// An empty belief yields a 1×1 zero summary, matching the Python. That shape
/// is deliberately not the board's: a caller that forwards it into a tensor
/// build gets a length mismatch instead of a plausible all-zero belief.
pub fn summarize_belief(belief: &BeliefState) -> BeliefSummary {
    if belief.n() == 0 {
        return BeliefSummary::zeros(1, 1);
    }

    let first = &belief.particles[0].state;
    let (h, w) = (first.h, first.w);
    let n = h * w;
    let enemy = belief.enemy_seat();

    // Normalize, or fall back to uniform when nothing has mass — the planes
    // still have to describe *some* distribution for the network.
    let raw: Vec<f64> = belief.particles.iter().map(|p| p.weight.max(0.0)).collect();
    let total = npsum(&raw);
    let weights: Vec<f64> = if total <= 0.0 {
        vec![1.0 / belief.n() as f64; belief.n()]
    } else {
        raw.iter().map(|value| value / total).collect()
    };

    let mut owner = vec![0.0f64; n];
    let mut army_mean = vec![0.0f64; n];
    let mut army_sq = vec![0.0f64; n];
    let mut general = vec![0.0f64; n];
    let mut castle_owner = vec![0.0f64; n];
    let mut visibility = vec![0.0f64; n];

    for (&weight, particle) in weights.iter().zip(&belief.particles) {
        let state = &particle.state;
        let owned = &state.ownership[enemy];
        let vision = visibility_from_owned(owned, h, w);
        for i in 0..n {
            if owned[i] {
                owner[i] += weight;
                let army = state.armies[i] as f64;
                army_mean[i] += weight * army;
                army_sq[i] += weight * (army * army);
                if state.castles[i] {
                    castle_owner[i] += weight;
                }
            }
            if vision[i] {
                visibility[i] += weight;
            }
        }
        let g = state.general_positions[enemy];
        if g[0] >= 0 && (g[0] as usize) < h && g[1] >= 0 && (g[1] as usize) < w {
            general[g[0] as usize * w + g[1] as usize] += weight;
        }
    }

    // Variance clamped at zero: the two accumulators are computed
    // independently, so cancellation can leave `E[x²] - E[x]²` slightly
    // negative on a plane where every particle agrees.
    let army_std: Vec<f32> = (0..n)
        .map(|i| {
            let var = (army_sq[i] - army_mean[i] * army_mean[i]).max(0.0);
            var.sqrt() as f32
        })
        .collect();

    let narrow = |values: &[f64]| values.iter().map(|&v| v as f32).collect::<Vec<f32>>();

    BeliefSummary {
        enemy_owner: narrow(&owner),
        enemy_army_mean: narrow(&army_mean),
        enemy_army_std: army_std,
        enemy_general: narrow(&general),
        enemy_castle_owner: narrow(&castle_owner),
        enemy_visibility: narrow(&visibility),
        ess_fraction: ess_fraction(belief) as f32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belief::{BeliefConfig, Particle};
    use crate::memory::VisibleMemory;
    use crate::state::GameState;
    use std::rc::Rc;

    fn enemy_at(cell: usize, army: i32) -> GameState {
        let mut state = GameState::empty(3, 3);
        for i in 0..9 {
            state.passable[i] = true;
            state.ownership_neutral[i] = true;
        }
        state.ownership[1][cell] = true;
        state.ownership_neutral[cell] = false;
        state.generals[cell] = true;
        state.armies[cell] = army;
        state.general_positions = [[-1, -1], [(cell / 3) as i32, (cell % 3) as i32]];
        state
    }

    fn belief_of(entries: &[(usize, i32, f64)]) -> BeliefState {
        let memory = Rc::new(VisibleMemory::empty(3, 3));
        BeliefState::new(
            0,
            entries
                .iter()
                .map(|&(cell, army, weight)| {
                    Particle::new(Rc::new(enemy_at(cell, army)), weight, Rc::clone(&memory))
                })
                .collect(),
            BeliefConfig {
                n_particles: entries.len().max(1),
                ..Default::default()
            },
        )
    }

    #[test]
    fn an_empty_belief_summarizes_to_a_one_by_one_zero() {
        let belief = belief_of(&[]);
        let summary = summarize_belief(&belief);
        assert_eq!(summary.enemy_owner.len(), 1);
        assert_eq!(summary.ess_fraction, 0.0);
    }

    #[test]
    fn a_certain_belief_puts_all_the_mass_on_one_cell() {
        let summary = summarize_belief(&belief_of(&[(4, 12, 1.0)]));
        assert_eq!(summary.enemy_owner[4], 1.0);
        assert_eq!(summary.enemy_general[4], 1.0);
        assert_eq!(summary.enemy_army_mean[4], 12.0);
        assert_eq!(summary.enemy_army_std[4], 0.0, "no spread, no deviation");
        assert!(summary.enemy_visibility.iter().all(|&v| v == 1.0), "3x3 board, one owner");
    }

    #[test]
    fn disagreement_shows_up_as_spread_not_as_a_wrong_mean() {
        // Two equally likely armies on the same cell: mean 6, sd 4.
        let summary = summarize_belief(&belief_of(&[(0, 2, 0.5), (0, 10, 0.5)]));
        assert_eq!(summary.enemy_army_mean[0], 6.0);
        assert_eq!(summary.enemy_army_std[0], 4.0);
        assert_eq!(summary.enemy_owner[0], 1.0);
    }

    #[test]
    fn a_weightless_belief_falls_back_to_uniform_rather_than_dividing_by_zero() {
        let summary = summarize_belief(&belief_of(&[(0, 4, 0.0), (8, 4, 0.0)]));
        assert_eq!(summary.enemy_owner[0], 0.5);
        assert_eq!(summary.enemy_owner[8], 0.5);
    }

    #[test]
    fn the_planes_carry_raw_armies_not_compressed_ones() {
        // 4096 is `ARMY_SCALE`; a compressed plane would read 1.0 here.
        let summary = summarize_belief(&belief_of(&[(4, 4096, 1.0)]));
        assert_eq!(summary.enemy_army_mean[4], 4096.0);
    }
}
