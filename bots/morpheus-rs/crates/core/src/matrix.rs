//! Simultaneous-matrix math for the search: regret matching plus.
//!
//! Port of `bots/morpheus/matrix.py`. No tree storage lives here — nodes,
//! enemy tables and eviction are [`crate::tree`]. Every value is in the root
//! player's perspective, so backup never flips a sign.
//!
//! **One reduction here cannot be bit-exact, and the oracle is the reason.**
//! The Python writes `q_eff @ sigma_enemy` and `sigma_self @ u_self`, and
//! NumPy sends `@` on f64 to BLAS — Accelerate on the laptop, OpenBLAS on the
//! x86 container. Measured against 3,000 random simplex vectors of the widths
//! the search actually uses, BLAS agrees with neither a sequential sum (mean
//! 0.5 ulp, max 3) nor NumPy's own pairwise reduction (mean 0.4 ulp, max 3).
//! There is no order to copy: `np.dot` is a different answer on a different
//! host, so the Python bot disagrees with *itself* across machines here, the
//! same way `np.argsort` does under AVX-512 (see [`crate::support::rng`]).
//!
//! So the dot products reduce with [`npsum`] — NumPy's pairwise order, the
//! closest thing to a convention this crate already reproduces — the `matrix`
//! parity surface carries a measured tolerance instead of bit-exactness, and
//! the decision gate expects the resulting flips on near-ties.

use crate::support::rng::{npsum, Rng};

/// Progressive-widening caps and the exploration floor. Initial guesses in the
/// Python, and still unmeasured; M7 owns them.
pub const SELF_WIDENING_CAP: usize = 16;
pub const SELF_WIDENING_COEFF: f64 = 2.0;
pub const SELF_WIDENING_FLOOR: usize = 8;
pub const ENEMY_WIDENING_CAP: usize = 12;
pub const ENEMY_WIDENING_COEFF: f64 = 1.5;
pub const EXPLORATION_FLOOR: f64 = 0.05;
pub const EXPLORATION_NUMERATOR: f64 = 0.5;

/// `K(N) = min(cap, 1 + floor(coeff * sqrt(N)))`.
pub fn widening_limit(n: i64, coeff: f64, cap: usize) -> usize {
    let n = n.max(0) as f64;
    let raw = 1.0 + (coeff * n.sqrt()).floor();
    (raw as usize).min(cap)
}

/// The self limit carries a floor: `K(0) = 1` would lock the average strategy
/// onto pass before any policy expansion widens the root.
pub fn self_widening_limit(n: i64) -> usize {
    widening_limit(n, SELF_WIDENING_COEFF, SELF_WIDENING_CAP).max(SELF_WIDENING_FLOOR)
}

pub fn enemy_widening_limit(n: i64) -> usize {
    widening_limit(n, ENEMY_WIDENING_COEFF, ENEMY_WIDENING_CAP)
}

/// `epsilon(N) = max(floor, numerator / sqrt(1 + N))`.
pub fn exploration_epsilon(n: i64) -> f64 {
    let n = n.max(0) as f64;
    (EXPLORATION_NUMERATOR / (1.0 + n).sqrt()).max(EXPLORATION_FLOOR)
}

fn clamp_positive(values: &[f64]) -> Vec<f64> {
    values.iter().map(|&v| v.max(0.0)).collect()
}

/// Normalize a vector of non-negative mass, or fall back to uniform.
fn normalized_or_uniform(values: &[f64]) -> Vec<f64> {
    let positive = clamp_positive(values);
    let total = npsum(&positive);
    if total <= 0.0 {
        let n = values.len().max(1) as f64;
        return vec![1.0 / n; values.len()];
    }
    positive.iter().map(|&v| v / total).collect()
}

/// Normalize positive regrets; with none, fall back to the prior.
pub fn regret_matching_strategy(regrets: &[f64], prior: &[f64]) -> Vec<f64> {
    assert_eq!(
        regrets.len(),
        prior.len(),
        "regrets and prior shape mismatch"
    );
    let positive = clamp_positive(regrets);
    let total = npsum(&positive);
    if total <= 0.0 {
        return normalized_or_uniform(prior);
    }
    positive.iter().map(|&v| v / total).collect()
}

/// `sigma = (1 - epsilon) * regret_strategy + epsilon * prior`.
pub fn mixed_strategy(regrets: &[f64], prior: &[f64], n: i64) -> Vec<f64> {
    let p = normalized_or_uniform(prior);
    let regret_sigma = regret_matching_strategy(regrets, &p);
    let eps = exploration_epsilon(n);
    regret_sigma
        .iter()
        .zip(&p)
        .map(|(&r, &q)| (1.0 - eps) * r + eps * q)
        .collect()
}

/// Visited entries use Q; unvisited ones use first-play urgency.
pub fn effective_q(visits: &[f64], q: &[f64], first_play: f64) -> Vec<f64> {
    visits
        .iter()
        .zip(q)
        .map(|(&n, &value)| if n > 0.0 { value } else { first_play })
        .collect()
}

/// `(u_self, u_enemy, v)` for one enemy-hash submatrix, `q_eff` row-major
/// `n_self × n_enemy`.
pub fn matrix_utilities(
    sigma_self: &[f64],
    sigma_enemy: &[f64],
    q_eff: &[f64],
) -> (Vec<f64>, Vec<f64>, f64) {
    let (n_a, n_b) = (sigma_self.len(), sigma_enemy.len());
    assert_eq!(q_eff.len(), n_a * n_b, "Q shape mismatch");

    let mut scratch = vec![0.0f64; n_a.max(n_b)];

    let mut u_self = Vec::with_capacity(n_a);
    for i in 0..n_a {
        for j in 0..n_b {
            scratch[j] = q_eff[i * n_b + j] * sigma_enemy[j];
        }
        u_self.push(npsum(&scratch[..n_b]));
    }

    let mut u_enemy = Vec::with_capacity(n_b);
    for j in 0..n_b {
        for i in 0..n_a {
            scratch[i] = sigma_self[i] * q_eff[i * n_b + j];
        }
        u_enemy.push(npsum(&scratch[..n_a]));
    }

    let v = dot(sigma_self, &u_self);
    (u_self, u_enemy, v)
}

/// The `sigma_A · u_self` of the Python, in NumPy's pairwise order.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    let products: Vec<f64> = a.iter().zip(b).map(|(&x, &y)| x * y).collect();
    npsum(&products)
}

/// Particle-weighted `u_self` across enemy hashes; `v = sigma_A · u_self`.
pub fn aggregate_self_utilities(
    sigma_self: &[f64],
    enemy_weights: &[f64],
    enemy_sigmas: &[Vec<f64>],
    q_eff_list: &[Vec<f64>],
) -> (Vec<f64>, f64) {
    assert!(
        enemy_weights.len() == enemy_sigmas.len() && enemy_weights.len() == q_eff_list.len(),
        "enemy weight / sigma / Q list length mismatch"
    );
    let positive = clamp_positive(enemy_weights);
    let total = npsum(&positive);
    let w: Vec<f64> = if total > 0.0 {
        positive.iter().map(|&v| v / total).collect()
    } else if !positive.is_empty() {
        vec![1.0 / positive.len() as f64; positive.len()]
    } else {
        Vec::new()
    };

    let mut u_self = vec![0.0f64; sigma_self.len()];
    for ((&weight, sigma_b), q_eff) in w.iter().zip(enemy_sigmas).zip(q_eff_list) {
        if weight <= 0.0 {
            continue;
        }
        let (u_h, _, _) = matrix_utilities(sigma_self, sigma_b, q_eff);
        for (slot, value) in u_self.iter_mut().zip(&u_h) {
            *slot += weight * value;
        }
    }
    let v = dot(sigma_self, &u_self);
    (u_self, v)
}

/// Regret matching plus: cumulative regrets clipped at zero, never negative.
pub fn regret_plus_update(
    regrets: &[f64],
    utilities: &[f64],
    value: f64,
    maximizing: bool,
) -> Vec<f64> {
    regrets
        .iter()
        .zip(utilities)
        .map(|(&r, &u)| {
            let updated = if maximizing {
                r + u - value
            } else {
                r + value - u
            };
            updated.max(0.0)
        })
        .collect()
}

pub fn accumulate_average_strategy(avg: &mut [f64], sigma: &[f64]) {
    for (slot, &value) in avg.iter_mut().zip(sigma) {
        *slot += value;
    }
}

pub fn normalize_average_strategy(avg: &[f64]) -> Vec<f64> {
    let positive = clamp_positive(avg);
    let total = npsum(&positive);
    if total <= 0.0 {
        let n = avg.len().max(1);
        return vec![1.0 / n as f64; n];
    }
    positive.iter().map(|&v| v / total).collect()
}

/// `rng.choice(len(p), p=p/total)`, or a bare integer draw when nothing has
/// mass. Both forms are on the recorded stream and stay distinguishable.
pub fn sample_index<R: Rng + ?Sized>(probs: &[f64], rng: &mut R) -> usize {
    let positive = clamp_positive(probs);
    let total = npsum(&positive);
    if total <= 0.0 {
        return rng.integer(0, positive.len() as i64) as usize;
    }
    let normalized: Vec<f64> = positive.iter().map(|&v| v / total).collect();
    rng.choice_one(normalized.len(), Some(&normalized))
}

/// Update `N`, `W`, `Q` for one joint entry, in place.
pub fn apply_joint_backup(
    visits: &mut [f64],
    value_sum: &mut [f64],
    q: &mut [f64],
    n_enemy: usize,
    a_idx: usize,
    b_idx: usize,
    leaf_value: f64,
) {
    let at = a_idx * n_enemy + b_idx;
    visits[at] += 1.0;
    value_sum[at] += leaf_value;
    q[at] = value_sum[at] / visits[at];
}

/// Largest normalized `S_A`; ties by marginal visits, then prior, then index.
///
/// Near-zero prior entries drop out first, so early progressive-widening mass
/// on pass cannot win after the live network prior has moved elsewhere. The
/// Python spells the ordering `np.lexsort((-prior, -visits, -avg))`, whose last
/// key is the primary one and whose stability makes the lowest index win a
/// three-way tie.
pub fn select_root_action(
    avg_strategy: &[f64],
    marginal_visits: &[f64],
    prior: &[f64],
    legal_mask: Option<&[bool]>,
) -> usize {
    let mut avg = normalize_average_strategy(avg_strategy);
    let mut visits = marginal_visits.to_vec();
    let mut prior = prior.to_vec();

    if let Some(mask) = legal_mask {
        for i in 0..avg.len() {
            if !mask[i] {
                avg[i] = -1.0;
                visits[i] = -1.0;
                prior[i] = -1.0;
            }
        }
    }

    let max_prior = prior.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let max_prior = if prior.is_empty() { 0.0 } else { max_prior };
    if max_prior > 0.0 {
        let cutoff = (1e-6 * max_prior).max(1e-12);
        let active: Vec<bool> = prior.iter().map(|&p| p > cutoff).collect();
        let any = active.iter().any(|&a| a);
        let all = active.iter().all(|&a| a);
        if any && !all {
            for (i, &keep) in active.iter().enumerate() {
                if !keep {
                    avg[i] = -1.0;
                    visits[i] = -1.0;
                }
            }
        }
    }

    let mut best = 0usize;
    for i in 1..avg.len() {
        let better = (avg[i], visits[i], prior[i]) > (avg[best], visits[best], prior[best]);
        if better {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_self_limit_never_drops_below_its_floor() {
        assert_eq!(self_widening_limit(0), SELF_WIDENING_FLOOR);
        assert_eq!(self_widening_limit(-5), SELF_WIDENING_FLOOR);
        // 1 + floor(2 * sqrt(100)) = 21, capped at 16.
        assert_eq!(self_widening_limit(100), SELF_WIDENING_CAP);
    }

    #[test]
    fn the_enemy_limit_has_no_floor_and_caps() {
        assert_eq!(enemy_widening_limit(0), 1);
        assert_eq!(enemy_widening_limit(1000), ENEMY_WIDENING_CAP);
    }

    #[test]
    fn regret_matching_falls_back_to_the_prior_when_nothing_is_positive() {
        let sigma = regret_matching_strategy(&[-1.0, -2.0], &[3.0, 1.0]);
        assert!((sigma[0] - 0.75).abs() < 1e-15);
        assert!((sigma[1] - 0.25).abs() < 1e-15);
    }

    #[test]
    fn a_dead_prior_and_dead_regrets_give_a_uniform_strategy() {
        let sigma = regret_matching_strategy(&[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0]);
        assert_eq!(sigma, vec![1.0 / 3.0; 3]);
    }

    #[test]
    fn regret_plus_never_goes_negative() {
        let out = regret_plus_update(&[0.5], &[0.0], 10.0, true);
        assert_eq!(out, vec![0.0]);
    }

    #[test]
    fn the_root_selector_breaks_ties_by_visits_then_prior_then_index() {
        // Equal average strategy; the second entry has more visits.
        let idx = select_root_action(&[1.0, 1.0], &[1.0, 5.0], &[0.5, 0.5], None);
        assert_eq!(idx, 1);
        // Equal everything: lowest index, matching lexsort's stability.
        let idx = select_root_action(&[1.0, 1.0], &[3.0, 3.0], &[0.5, 0.5], None);
        assert_eq!(idx, 0);
    }

    #[test]
    fn a_prior_the_network_zeroed_cannot_win_on_stale_average_mass() {
        // Index 0 owns the average strategy but the live prior has dropped it.
        let idx = select_root_action(&[9.0, 1.0], &[0.0, 0.0], &[0.0, 1.0], None);
        assert_eq!(idx, 1);
    }
}
