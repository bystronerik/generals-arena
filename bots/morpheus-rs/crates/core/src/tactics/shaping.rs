//! The hand-written score, and how much of it reaches the prior.
//!
//! [`heuristic_action_scores`] is a per-action tactical ranking computed
//! independently of the network — expand before contact, seek after it — and
//! the scores are *relative* only: [`blend_prior`] decides how far one of them
//! may move an action, bounded by a log clip, so the absolute magnitudes carry
//! no meaning beyond their ratios.
//!
//! General captures are deliberately unscored here. They are already
//! guaranteed by [`super::candidates::mandatory_action_indices`] and forced by
//! [`super::constrain::constrain_nn_action`], so a score term would duplicate
//! a rule that cannot be outvoted anyway.
//!
//! The Python has a scalar reference implementation *and* a vectorized one
//! for this path, held together by `test_heuristic_scores_parity.py`. This
//! port has one: a per-action loop mirroring the vectorized arithmetic site
//! by site, because the vectorized branch is the one that plays. Where the
//! two Pythons differ in float width or in operation order, the *vectorized*
//! one is authoritative here.

use crate::board::action::{decode_action, live_build_cost, PASS_INDEX};
use crate::belief::{Action5, BeliefState};
use crate::board::memory::{
    VisibleMemory, OWNER_ENEMY, OWNER_NEUTRAL,
};
use crate::support::rng::npsum;
use crate::board::transition::{BASE_COST, DEATHTOUCH_TURN};
use crate::io::wire::Observation;

use super::*;

/// Per-action tactical score over the play mask, independent of the network.
///
/// Expand before contact, seek after it. The scores are a *relative* ranking
/// signal only — [`blend_prior`] decides how much of it reaches the root prior,
/// so the absolute magnitudes carry no meaning beyond their ratios.
///
/// General captures are deliberately unscored: they are already guaranteed by
/// [`mandatory_action_indices`] and forced by [`constrain_nn_action`], so a
/// score term would duplicate a rule that cannot be outvoted anyway.
pub fn heuristic_action_scores(
    obs: &Observation,
    memory: &VisibleMemory,
    mask: &[bool],
    belief: Option<&BeliefState>,
    prev_action: Option<Action5>,
) -> Vec<f64> {
    let g = Grids::new(obs);
    let tables = decode_tables();
    let mut out = vec![0.0f64; N_ACTIONS];
    let seen_enemy = enemy_is_visible(obs, memory);
    let turn = turn_of(obs);
    let castle_w = castle_timing_weight(turn);

    // Builds: base-price only, and prefer cells that keep a useful remnant.
    // The cost grid is priced only when a build is actually legal, as in the
    // Python — it is the most expensive thing on this path and legal builds are
    // rare (measured: 8 turns out of 256).
    let build_idx: Vec<usize> = (0..PASS_INDEX)
        .filter(|&index| mask[index] && tables.kind[index] == 2)
        .collect();
    if !build_idx.is_empty() {
        let cost = live_build_cost(obs, memory);
        for index in build_idx {
            let (sr, sc) = (tables.sr[index] as i32, tables.sc[index] as i32);
            let at = g.at(sr, sc);
            let b_army = (obs.army_grid[at] as i64).max(1) as f64;
            let at_base = if cost[at] == BASE_COST { 1.0 } else { 0.0 };
            out[index] = castle_w * (10.0 + 0.2 * b_army.min(100.0)) * at_base;
        }
    }

    // Moves whose destination is on the board.
    let move_idx: Vec<usize> = (0..PASS_INDEX)
        .filter(|&index| {
            mask[index]
                && tables.kind[index] == 0
                && g.inside(tables.tr[index] as i32, tables.tc[index] as i32)
        })
        .collect();

    if seen_enemy {
        let target = enemy_seek_target(obs, memory, belief);
        let goals = seek_goals(obs, memory, belief);
        let dist_field = if goals.is_empty() {
            None
        } else {
            Some(path_distance_field(obs, &goals))
        };
        let exclude = movable_exclude_cell(obs, memory);
        let king = wave_assembly_cell(obs, memory, belief, exclude);
        let king_dist = king.map(|cell| path_distance_field(obs, &[cell]));
        let (share, max_own, tot) = army_concentration(obs, exclude);
        let gen_known = enemy_general_visible(obs, memory);
        let gen_cell = known_enemy_general_cell(obs, memory);
        let hunt_cell = gen_cell.or_else(|| believed_enemy_general(belief));
        let hunt_dist = hunt_cell.map(|cell| path_distance_field(obs, &[cell]));
        let savings_site = castle_build_site(obs, memory);
        let savings_dist = savings_site.map(|cell| path_distance_field(obs, &[cell]));
        let reveal_grid = reveal_count_grid(obs);

        for &index in &move_idx {
            let (sr, sc) = (tables.sr[index] as i32, tables.sc[index] as i32);
            let (tr, tc) = (tables.tr[index] as i32, tables.tc[index] as i32);
            let army_i = g.army(sr, sc);
            let dest_owner = g.owner(tr, tc);
            let dest_army = g.army(tr, tc);
            let reveal = reveal_grid[g.at(tr, tc)] as f64;
            let efficiency = reveal / (1.0 + dest_army.max(0) as f64);
            let progress = path_progress(sr, sc, tr, tc, dist_field.as_ref(), target);
            let bias = direction_bias(progress, dest_owner);
            let surplus = (army_i - dest_army - 1).max(0) as f64;
            let thrash = tip_thrash_factor(army_i, max_own, tot);
            let army_w = wave_weight(army_i);
            let atk_w = attack_weight(army_i);
            let gather = if dest_owner == 1 {
                stack_gather_factor(army_i, dest_army, progress)
            } else {
                1.0
            };

            // Strong attack reward: the raw network rarely proposes takes
            // (probe ~0.8%).
            let mut enemy_score = atk_w
                * thrash
                * bias
                * (ENEMY_TAKE_BASE
                    + 40.0 * progress.max(0.0)
                    + 8.0 * reveal
                    + 3.5 * surplus.min(200.0));
            let is_gen_touch = match gen_cell {
                Some(cell) => tr as usize == cell.0 && tc as usize == cell.1,
                None => false,
            };
            let hunt_prog = match (&hunt_dist, hunt_cell) {
                (Some(field), Some(cell)) => {
                    path_progress(sr, sc, tr, tc, Some(field), Some(cell))
                }
                _ => 0.0,
            };
            let hunt_factor = 1.0 + HUNT_PROGRESS_BONUS * hunt_prog.max(0.0);
            if gen_known {
                // Kill, don't farm: takes that shorten the path to the known
                // general keep the boost; sideways border chew is damped.
                enemy_score *= if is_gen_touch || progress > 0.0 {
                    1.35
                } else {
                    GENERAL_CHEW_DAMP
                };
            }
            enemy_score *= hunt_factor;

            let neutral_score = (atk_w
                * thrash
                * bias
                * (NEUTRAL_CARVE_BASE
                    + 30.0 * progress.max(0.0)
                    + 12.0 * efficiency
                    + 8.0 * reveal))
                * hunt_factor;

            let committed = is_committed_army(army_i, max_own, tot, share);
            let mut k_prog = match &king_dist {
                Some(field) if !committed => path_progress(sr, sc, tr, tc, Some(field), king),
                Some(_) => 0.0,
                None => 0.0,
            };
            // Castle savings: tips inside the site's catchment gather to the
            // build site instead of the front assembly, until the pile reaches
            // BASE_COST and the hard rule builds.
            if let (Some(field), Some(site)) = (&savings_dist, savings_site) {
                let src_b = field.get(sr, sc);
                if src_b >= 0 && src_b <= CASTLE_CATCHMENT {
                    k_prog = if committed {
                        0.0
                    } else {
                        path_progress(sr, sc, tr, tc, Some(field), Some(site))
                    };
                }
            }
            // Army-weighted: a 100-army hinterland stack's gather move must
            // outrank a 3-army tip shuffle, so weight by attack_weight.
            let king_gather =
                atk_w * (14.0 + 22.0 * k_prog) * stack_gather_factor(army_i, dest_army, k_prog);
            let own_forward =
                (atk_w * thrash * bias * (18.0 + 32.0 * progress) * gather) * hunt_factor;
            // The raw network often picks retreat; keep mass near zero here.
            let own_idle = 0.008 * army_w * bias * gather;
            let own_score = if k_prog > 0.0 && share < GATHER_SHARE_MIN {
                king_gather
            } else if progress > 0.0 {
                own_forward
            } else {
                own_idle
            };

            let mut score = if dest_owner == OWNER_ENEMY && surplus > 0.0 {
                enemy_score
            } else if dest_owner == OWNER_NEUTRAL {
                neutral_score
            } else if dest_owner == 1 {
                own_score
            } else {
                0.01 * army_w * gather
            };
            // Deathtouch: any executed touch wins outright, so the surplus gate
            // must not suppress it. Pre-800 an underpowered touch stays in the
            // near-zero branch — feeding a defended general is a loss.
            if turn >= DEATHTOUCH_TURN && is_gen_touch {
                score = DEATHTOUCH_SCORE;
            }
            out[index] = out[index].max(score);
        }
    } else if !move_idx.is_empty() {
        // No enemy yet: hunt fog with a formed wave; urgency rises with turn.
        let urgency = fog_urgency(turn, false);
        let own_struct = own_structure_mask(obs, memory);
        let fog_target = enemy_seek_target(obs, memory, belief);
        let fog_goals = seek_goals(obs, memory, belief);
        let fog_dist = if fog_goals.is_empty() {
            None
        } else {
            Some(path_distance_field(obs, &fog_goals))
        };
        let reveal_grid = reveal_count_grid(obs);

        // Sources of the frontier moves in this candidate set — the Python
        // builds this grid from the frontier subset and then reads it for
        // *every* move, so the two passes are ordered, not independent.
        let mut frontier_src = vec![false; obs.h * obs.w];
        for &index in &move_idx {
            let (sr, sc) = (tables.sr[index] as i32, tables.sc[index] as i32);
            let (tr, tc) = (tables.tr[index] as i32, tables.tc[index] as i32);
            if g.owner(tr, tc) == OWNER_NEUTRAL {
                frontier_src[g.at(sr, sc)] = true;
            }
        }

        for &index in &move_idx {
            let (sr, sc) = (tables.sr[index] as i32, tables.sc[index] as i32);
            let (tr, tc) = (tables.tr[index] as i32, tables.tc[index] as i32);
            let src_army = g.army(sr, sc);
            let dest_owner = g.owner(tr, tc);
            let dest_army = g.army(tr, tc);
            let progress = path_progress(sr, sc, tr, tc, fog_dist.as_ref(), fog_target);
            let struct_src = own_struct[g.at(sr, sc)] && src_army >= STRUCTURE_IDLE_ARMY;

            if dest_owner == OWNER_NEUTRAL {
                let reveal = reveal_grid[g.at(tr, tc)] as f64;
                let eff = reveal / (1.0 + dest_army.max(0) as f64);
                let ew = explore_wave_weight(src_army);
                let bias_f = direction_bias(progress, OWNER_NEUTRAL);
                let mut f_score = ew
                    * urgency
                    * bias_f
                    * (4.0
                        + PRE_REVEAL_WEIGHT * reveal
                        + PRE_EFFICIENCY_WEIGHT * eff
                        + PRE_PROGRESS_WEIGHT * progress.max(0.0));
                // Leaving a fat structure into fog is the right pre-contact
                // move.
                if struct_src {
                    f_score *= 2.5 + 0.04 * src_army.min(80) as f64;
                }
                out[index] = out[index].max(f_score);
            } else if dest_owner == 1 {
                let bias_c = direction_bias(progress, 1);
                let gather = stack_gather_factor(src_army, dest_army, progress);
                let tip_bonus = if frontier_src[g.at(tr, tc)] { 2.0 } else { 1.0 };
                let c_score = if struct_src {
                    if progress < 0.0 {
                        0.01
                    } else {
                        urgency
                            * tip_bonus
                            * bias_c
                            * (5.0 + 0.1 * src_army.min(100) as f64 + 8.0 * progress.max(0.0))
                            * gather
                    }
                } else if progress > 0.0 {
                    explore_wave_weight(src_army)
                        * urgency
                        * tip_bonus
                        * bias_c
                        * (0.8 + 4.0 * progress)
                        * gather
                } else {
                    0.008 * bias_c * gather
                };
                out[index] = out[index].max(c_score);
            }
        }
    }

    // Commitment hysteresis, applied last and only where the score is already
    // positive, so a banned or retreating continuation stays dead.
    if let Some(prev) = prev_action {
        if let Some((_, _, p_tr, p_tc)) = move_dest(prev) {
            if g.inside(p_tr, p_tc) {
                for index in 0..PASS_INDEX {
                    if tables.kind[index] != 0 {
                        continue;
                    }
                    if tables.sr[index] as i32 != p_tr || tables.sc[index] as i32 != p_tc {
                        continue;
                    }
                    if out[index] > 0.0 {
                        out[index] *= CONTINUATION_BONUS;
                    }
                }
            }
        }
    }

    for index in 0..N_ACTIONS {
        out[index] = if mask[index] { out[index].max(0.0) } else { 0.0 };
    }
    out
}

/// Blend a network prior with heuristic scores under a bounded nudge.
///
/// `shaped = softmax(log p + lam * clip(log(h / gmean(h)), ±log_clip))`.
///
/// Centering by the geometric mean is load-bearing: raw scores sit around ~120,
/// so without it every action saturates the clip and the heuristic term
/// collapses to a constant. `floor_frac` lifts the prior to a fraction of its
/// own maximum first, so an action the network zeroed cannot be resurrected
/// past the clip bound.
pub fn blend_prior(
    nn_prior: &[f64],
    scores: &[f64],
    mask: &[bool],
    lam: f64,
    log_clip: f64,
    floor_frac: f64,
    floor_abs: f64,
) -> Vec<f64> {
    let n = mask.len();
    let prior_a: Vec<f64> = (0..n)
        .map(|i| if mask[i] { nn_prior[i].max(0.0) } else { 0.0 })
        .collect();
    let h: Vec<f64> = (0..n)
        .map(|i| if mask[i] { scores[i].max(0.0) } else { 0.0 })
        .collect();

    let renormalized = || -> Vec<f64> {
        let total = npsum(&prior_a);
        if total > 0.0 {
            return prior_a.iter().map(|&v| v / total).collect();
        }
        let live = mask.iter().filter(|&&m| m).count();
        if live == 0 {
            return prior_a.clone();
        }
        mask.iter()
            .map(|&m| if m { 1.0 / live as f64 } else { 0.0 })
            .collect()
    };

    if lam == 0.0 || log_clip == 0.0 {
        return renormalized();
    }

    let scored: Vec<bool> = (0..n).map(|i| mask[i] && h[i] > 0.0).collect();
    if !scored.iter().any(|&v| v) {
        return renormalized();
    }

    let prior_max = prior_a.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let floor = floor_abs.max(floor_frac * prior_max);
    let p: Vec<f64> = (0..n)
        .map(|i| if mask[i] { prior_a[i].max(floor) } else { 0.0 })
        .collect();
    if !p.iter().any(|&v| v > 0.0) {
        return renormalized();
    }

    // Center before clipping so the clip measures a ratio, not a magnitude.
    let logs: Vec<f64> = (0..n).filter(|&i| scored[i]).map(|i| h[i].ln()).collect();
    let log_gmean = npsum(&logs) / logs.len() as f64;

    let mut logits = vec![f64::NEG_INFINITY; n];
    for i in 0..n {
        if !(mask[i] && p[i] > 0.0) {
            continue;
        }
        let log_h = if scored[i] {
            (h[i].ln() - log_gmean).clamp(-log_clip, log_clip)
        } else {
            // Actions with no heuristic score sit at the bottom of the clip
            // range rather than at zero — unless the clip is infinite, which
            // is the retired legacy regime and keeps the hard zero.
            (f64::NEG_INFINITY).clamp(-log_clip, log_clip)
        };
        logits[i] = p[i].ln() + lam * log_h;
    }

    let finite: Vec<bool> = logits.iter().map(|v| v.is_finite()).collect();
    if !finite.iter().any(|&v| v) {
        return renormalized();
    }
    let top = (0..n)
        .filter(|&i| finite[i])
        .map(|i| logits[i])
        .fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = (0..n)
        .map(|i| if finite[i] { (logits[i] - top).exp() } else { 0.0 })
        .collect();
    let total = npsum(&weights);
    if total <= 0.0 {
        return renormalized();
    }
    weights.iter().map(|&v| v / total).collect()
}

/// Reshape the root prior: expand pre-contact, seek after contact.
#[allow(clippy::too_many_arguments)]
pub fn apply_pre_contact_prior(
    prior: &[f64],
    obs: &Observation,
    memory: &VisibleMemory,
    mask: &[bool],
    belief: Option<&BeliefState>,
    lam: f64,
    log_clip: f64,
    floor_frac: f64,
    floor_abs: f64,
    prev_action: Option<Action5>,
) -> Vec<f64> {
    let scores = heuristic_action_scores(obs, memory, mask, belief, prev_action);
    blend_prior(prior, &scores, mask, lam, log_clip, floor_frac, floor_abs)
}

// -------------------------------------------------------------- hard rules

/// Argmax of a shaped prior among legal non-pass moves.
///
/// With `require_progress`, own-land retreats are skipped so a passive network
/// top is redirected without running a full heuristic policy.
pub fn best_prior_legal_action(
    prior: &[f64],
    obs: &Observation,
    memory: &VisibleMemory,
    mask: &[bool],
    prev_action: Option<Action5>,
    recent_actions: &[Action5],
    require_progress: bool,
) -> Option<Action5> {
    if prior.len() != mask.len() {
        return None;
    }
    let g = Grids::new(obs);
    let goals = seek_goals(obs, memory, None);
    let dist_field = if goals.is_empty() {
        None
    } else {
        Some(path_distance_field(obs, &goals))
    };
    let target = enemy_seek_target(obs, memory, None);
    let reveal = reveal_count_grid(obs);

    let mut best_idx: Option<usize> = None;
    let mut best_score = -1.0f64;
    for index in 0..PASS_INDEX {
        if !mask[index] {
            continue;
        }
        let action = match decode_action(index) {
            Some(action) => action,
            None => continue,
        };
        if blocks_oscillation(action, prev_action, obs, recent_actions, false, Some(&reveal)) {
            continue;
        }
        if require_progress {
            let (sr, sc, tr, tc) = match move_dest(action) {
                Some(ends) => ends,
                None => continue,
            };
            if !g.inside(tr, tc) {
                continue;
            }
            let dest_o = g.owner(tr, tc);
            let prog = path_progress(sr, sc, tr, tc, dist_field.as_ref(), target);
            if dest_o == 1 && prog <= 0.0 {
                continue;
            }
        }
        if prior[index] > best_score {
            best_score = prior[index];
            best_idx = Some(index);
        }
    }
    best_idx.and_then(decode_action)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_blend_returns_the_renormalized_prior_when_lambda_is_zero() {
        let mut mask = vec![false; N_ACTIONS];
        mask[0] = true;
        mask[1] = true;
        let mut prior = vec![0.0; N_ACTIONS];
        prior[0] = 0.25;
        prior[1] = 0.75;
        let scores = vec![1.0; N_ACTIONS];
        let out = blend_prior(&prior, &scores, &mask, 0.0, 1.0, 0.0, 0.0);
        assert!((out[0] - 0.25).abs() < 1e-12);
        assert!((out[1] - 0.75).abs() < 1e-12);
    }

    #[test]
    fn the_blend_clip_bounds_how_far_one_heuristic_can_move_an_action() {
        let mut mask = vec![false; N_ACTIONS];
        mask[0] = true;
        mask[1] = true;
        let mut prior = vec![0.0; N_ACTIONS];
        prior[0] = 0.5;
        prior[1] = 0.5;
        let mut scores = vec![0.0; N_ACTIONS];
        scores[0] = 1.0;
        scores[1] = 1.0e9; // a billion times larger
        let out = blend_prior(&prior, &scores, &mask, 1.0, default_shaping_log_clip(), 0.0, 0.0);
        // Bounded at 10x either way: the ratio cannot exceed 100.
        assert!(out[1] / out[0] <= 100.0 + 1e-9);
        assert!(out[1] / out[0] > 99.0);
    }
}
