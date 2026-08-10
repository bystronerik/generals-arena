//! Play mask, prior shaping, hard rules, and the three planners.
//!
//! Port of `bots/morpheus/tactics.py` — 2,283 lines and the largest single
//! surface in the rewrite, with no speed story to motivate it (rewrite-plan §7
//! calls it "pure translation, little gain, highest port effort per line").
//! It is here because the decision depends on it and the shipped binary cannot
//! call Python.
//!
//! The rules themselves are the Python's; every constant below carries the
//! measurement or the field-observed loss that set it, copied from the source
//! it was written in. Nothing here is retuned — M5 ports behaviour, M7 tunes.
//!
//! Two structural notes for anyone extending this file:
//!
//! * The Python has a scalar reference implementation *and* a vectorized one
//!   for the scoring path, held together by `test_heuristic_scores_parity.py`.
//!   This port has one implementation: a per-action loop that mirrors the
//!   vectorized arithmetic site by site, because the vectorized branch is the
//!   one that plays. Where the two Pythons differ in float width or in
//!   operation order, the *vectorized* one is authoritative here.
//! * Action tables index the padded 21×21 layout while the board grids index
//!   the live `H×W`. Every read of a grid at a decoded `(row, col)` therefore
//!   bounds-checks first; the Python gets away with implicit masking because
//!   only legal indices survive its `flatnonzero`.
//!
//! ## The directory
//!
//! The five surfaces the parity harness already separates — `playmask`,
//! `candidates`, `planners`, `shaping`, `constrain` — plus the leaves they
//! share. The shared leaves are out: [`params`] is the tuning surface,
//! [`geometry`] the action decoding, [`pathing`] the distance fields,
//! [`query`] the read-only board questions, [`weights`] the scalar terms, and
//! [`oscillation`] the two-cell shuffle rule. The mask and the four
//! planners are out too: [`play_mask`], [`seek`], [`castle`], [`defense`]
//! and [`kill`]. Each planner is a named subsystem in the rewrite plan, so
//! they are four files rather than a `planners/` directory — a third level
//! of nesting buys nothing at ~120 lines apiece.

pub mod castle;
pub mod defense;
pub mod geometry;
pub mod kill;
pub mod oscillation;
pub mod params;
pub mod pathing;
pub mod play_mask;
pub mod query;
pub mod seek;
pub mod weights;

pub use castle::*;
pub use defense::*;
pub use geometry::*;
pub use kill::*;
pub use oscillation::*;
pub use params::*;
pub use pathing::*;
pub use play_mask::*;
pub use query::*;
pub use seek::*;
pub use weights::*;

use crate::board::action::{decode_action, encode_action, live_build_cost, PASS_INDEX};
use crate::belief::{Action5, BeliefState};
use crate::board::memory::{
    VisibleMemory, OWNER_ENEMY, OWNER_NEUTRAL, TYPE_GENERAL,
};
use crate::support::rng::npsum;
use crate::board::transition::{BASE_COST, DEATHTOUCH_TURN, DIRECTIONS};
use crate::io::wire::Observation;

// ---------------------------------------------------------- candidate listing

/// Legal move indices whose destination is a visible enemy general.
pub fn general_capture_indices(
    obs: &Observation,
    memory: &VisibleMemory,
    mask: &[bool],
) -> Vec<usize> {
    let g = Grids::new(obs);
    let n = obs.h * obs.w;
    let mut out = Vec::new();
    for i in 0..n {
        let owner_enemy = obs.owner_grid[i] as i32 == 2;
        let is_gen = obs.type_grid[i] as i32 == TYPE_GENERAL || memory.known_enemy_general[i];
        if !(owner_enemy && is_gen) {
            continue;
        }
        let (r, c) = ((i / obs.w) as i32, (i % obs.w) as i32);
        for (d, (dr, dc)) in DIRECTIONS.iter().enumerate() {
            let (sr, sc) = (r - dr, c - dc);
            if !g.inside(sr, sc) || g.owner(sr, sc) != 1 {
                continue;
            }
            for split in 0..2 {
                let index = encode_action([0, sr, sc, d as i32, split]);
                if mask[index] {
                    out.push(index);
                }
            }
        }
    }
    out
}

/// Legal moves onto or from a cell adjacent to a visible enemy army source.
pub fn visible_enemy_source_interaction_indices(
    obs: &Observation,
    mask: &[bool],
) -> Vec<usize> {
    let g = Grids::new(obs);
    let n = obs.h * obs.w;
    let sources: Vec<bool> = (0..n)
        .map(|i| obs.owner_grid[i] as i32 == 2 && obs.army_grid[i] >= 1)
        .collect();
    if !sources.iter().any(|&v| v) {
        return Vec::new();
    }
    let mut interact = sources.clone();
    for i in 0..n {
        if !sources[i] {
            continue;
        }
        let (r, c) = ((i / obs.w) as i32, (i % obs.w) as i32);
        for (dr, dc) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
            let (nr, nc) = (r + dr, c + dc);
            if g.inside(nr, nc) {
                interact[g.at(nr, nc)] = true;
            }
        }
    }
    let tables = decode_tables();
    let mut out = Vec::new();
    for index in 0..PASS_INDEX {
        if !mask[index] || tables.kind[index] != 0 {
            continue;
        }
        let (sr, sc) = (tables.sr[index] as i32, tables.sc[index] as i32);
        let (tr, tc) = (tables.tr[index] as i32, tables.tc[index] as i32);
        if !g.inside(tr, tc) {
            continue;
        }
        if interact[g.at(tr, tc)] || (g.inside(sr, sc) && interact[g.at(sr, sc)]) {
            out.push(index);
        }
    }
    out
}

/// Legal moves whose destination has the given owner, biggest source first.
fn dest_owner_indices(obs: &Observation, mask: &[bool], owner: i32) -> Vec<usize> {
    let g = Grids::new(obs);
    let tables = decode_tables();
    let mut scored: Vec<(i64, usize)> = Vec::new();
    for index in 0..PASS_INDEX {
        if !mask[index] || tables.kind[index] != 0 {
            continue;
        }
        let (sr, sc) = (tables.sr[index] as i32, tables.sc[index] as i32);
        let (tr, tc) = (tables.tr[index] as i32, tables.tc[index] as i32);
        if !g.inside(tr, tc) {
            continue;
        }
        if g.owner(tr, tc) != owner {
            continue;
        }
        scored.push((g.army(sr, sc), index));
    }
    scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    scored.into_iter().map(|(_, index)| index).collect()
}

/// Legal moves whose destination is currently unowned.
pub fn frontier_expand_indices(obs: &Observation, mask: &[bool]) -> Vec<usize> {
    dest_owner_indices(obs, mask, OWNER_NEUTRAL)
}

/// Legal moves whose destination is enemy-owned land.
pub fn enemy_attack_indices(obs: &Observation, mask: &[bool]) -> Vec<usize> {
    dest_owner_indices(obs, mask, OWNER_ENEMY)
}

/// Captures, then enemy interactions, then attacks, then frontier expands.
///
/// Pass rides at the front only when it is the sole legal action.
pub fn mandatory_action_indices(
    obs: &Observation,
    memory: &VisibleMemory,
    mask: &[bool],
) -> Vec<usize> {
    let mut extras = general_capture_indices(obs, memory, mask);
    extras.extend(visible_enemy_source_interaction_indices(obs, mask));
    extras.extend(enemy_attack_indices(obs, mask));
    extras.extend(frontier_expand_indices(obs, mask));

    let has_nonpass = mask[..PASS_INDEX].iter().any(|&v| v);
    let mut ordered: Vec<usize> = Vec::new();
    if mask[PASS_INDEX] && !has_nonpass {
        ordered.push(PASS_INDEX);
    }
    ordered.extend(extras);

    let mut seen = vec![false; N_ACTIONS];
    let mut out = Vec::new();
    for index in ordered {
        if seen[index] || !mask[index] {
            continue;
        }
        seen[index] = true;
        out.push(index);
    }
    out
}

/// Mandatory actions first (stable), then the rest by descending prior.
///
/// The Python's `np.argsort(-prior[remaining], kind="stable")` is a stable sort
/// on the negated prior, so equal masses keep ascending index order.
pub fn policy_ordered_candidates(
    prior: &[f64],
    mask: &[bool],
    mandatory: &[usize],
    limit: usize,
) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::new();
    let mut seen = vec![false; mask.len()];
    for &index in mandatory {
        if seen[index] || !mask[index] {
            continue;
        }
        seen[index] = true;
        out.push(index);
        if out.len() >= limit {
            return out;
        }
    }
    let mut remaining: Vec<usize> = (0..mask.len())
        .filter(|&i| mask[i] && !seen[i])
        .collect();
    if remaining.is_empty() {
        return out;
    }
    remaining.sort_by(|&a, &b| {
        let (pa, pb) = (prior.get(a).copied().unwrap_or(0.0), prior.get(b).copied().unwrap_or(0.0));
        (-pa).partial_cmp(&(-pb)).unwrap_or(std::cmp::Ordering::Equal)
    });
    for index in remaining {
        out.push(index);
        if out.len() >= limit {
            break;
        }
    }
    out
}

// ------------------------------------------------------------- prior shaping

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

/// Army that actually moves for a capture action (full or half split).
fn capture_moved_army(index: usize, obs: &Observation) -> i64 {
    let action = match decode_action(index) {
        Some(action) => action,
        None => return 0,
    };
    let g = Grids::new(obs);
    let src = g.army(action[1], action[2]);
    if action[4] == 1 {
        src / 2
    } else {
        src - 1
    }
}

/// Keep the network/search choice except for hard rules and a prior re-rank.
///
/// Hard rules, in the order they fire: a winning general capture (any touch
/// from deathtouch); the kill-window march; emergency defense; the castle
/// build and its tithe; the garrison release; never pass when another move
/// exists; never own-land oscillation. Soft redirects (pass, retreat, lateral
/// home) pick the best legal action from the shaped root prior rather than
/// running a full heuristic policy; with no prior, only hard rules apply.
#[allow(clippy::too_many_arguments)]
pub fn constrain_nn_action(
    obs: &Observation,
    memory: &VisibleMemory,
    action: Action5,
    prev_action: Option<Action5>,
    recent_actions: &[Action5],
    prior: Option<&[f64]>,
    belief: Option<&BeliefState>,
) -> Action5 {
    // `belief` is on the signature and never read — and that is the Python's
    // behaviour, not an omission here. rewrite-plan §5 records a "final-state
    // wrinkle" saying the hard-rule layer *consumes* the belief, so decision
    // parity needs the captured snapshot as an input to this call. It does
    // not: `runtime.py` passes the argument, `constrain_nn_action` accepts it,
    // and nothing in the body touches it. The belief does reach the decision —
    // through `heuristic_action_scores` on the shaping path — but not through
    // the hard rules, so a `constrain` parity case needs no belief at all.
    // The parameter stays because the oracle has it and dropping it would hide
    // the discrepancy the next reader of §5 should find.
    let _ = belief;

    let g = Grids::new(obs);
    let mask = play_mask(obs, memory, None);
    let turn = turn_of(obs);

    let mut caps = general_capture_indices(obs, memory, &mask);
    if !caps.is_empty() {
        if turn < DEATHTOUCH_TURN {
            // Kill calculus: forcing an underpowered capture feeds the defense.
            // Only a touch that actually wins the clash is forced; otherwise
            // fall through and let scoring gather next door.
            caps.retain(|&index| {
                let a = match decode_action(index) {
                    Some(a) => a,
                    None => return false,
                };
                let d_r = a[1] + DIRECTIONS[a[3] as usize].0;
                let d_c = a[2] + DIRECTIONS[a[3] as usize].1;
                capture_moved_army(index, obs) > g.army(d_r, d_c)
            });
        }
        if !caps.is_empty() {
            // Python's `max` keeps the *first* maximum; Rust's `max_by_key`
            // keeps the last. On two adjacent stacks of equal size that is a
            // different move, so the tie rule is spelled out rather than
            // inherited from the standard library.
            let mut best = caps[0];
            let mut best_army = capture_moved_army(best, obs);
            for &index in &caps[1..] {
                let army = capture_moved_army(index, obs);
                if army > best_army {
                    best_army = army;
                    best = index;
                }
            }
            return decode_action(best).unwrap();
        }
    }

    // Kill window vs. emergency defense: whoever connects first wins, so the
    // race is explicit. A tie goes to the KILL — opponents park waves near our
    // general in endgames, and yielding on ties froze finishing marches for
    // entire endgames. Forced defense additionally requires the threat to be
    // imminent, not radius loitering.
    let threat = general_threat(obs, memory);
    let plan = kill_plan(obs, memory);
    if let Some((steps, _, first)) = plan {
        if threat.map_or(true, |(_, d)| steps <= d) && mask[encode_action(first)] {
            return first;
        }
    }
    if let Some(threat) = threat {
        if threat.1 <= DEFENSE_FORCE_WITHIN {
            if let Some(defend) = defend_general_move(obs, memory, threat) {
                if mask[encode_action(defend)] {
                    return defend;
                }
            }
        }
    }

    // Castle build (hard rule): the network gives builds ~zero prior mass,
    // which the blend clip cannot resurrect, so an affordable base-price build
    // on the savings site is forced. Kills always outrank it.
    let site = castle_build_site(obs, memory);
    if let Some(site) = site {
        let build_idx = encode_action([2, site.0 as i32, site.1 as i32, 0, 0]);
        if mask[build_idx] {
            return [2, site.0 as i32, site.1 as i32, 0, 0];
        }
    }

    if let Some(site) = site {
        // Tithe: every Nth turn one gather step funds the site. It yields to an
        // enemy take — never trade a capture for a shuffle.
        let chosen_is_take = match move_dest(action) {
            Some((_, _, ctr, ctc)) => g.inside(ctr, ctc) && g.owner(ctr, ctc) == OWNER_ENEMY,
            None => false,
        };
        if !chosen_is_take && turn % CASTLE_TITHE_PERIOD == 0 {
            if let Some(tithe) = castle_tithe_move(obs, memory, site) {
                if mask[encode_action(tithe)] {
                    return tithe;
                }
            }
        }
    }

    let chosen = action;
    let has_nonpass = mask[..PASS_INDEX].iter().any(|&v| v);
    let is_pass = chosen[0] == 1 || chosen == [1, 0, 0, 0, 0];

    // Garrison release. Fires only while the floor is active, the garrison
    // holds at least `GARRISON_RELEASE_FACTOR` × floor, the chosen action is
    // not already shipping from the general, and no live threat is bearing
    // down on it — never ship the garrison mid-emergency.
    if threat.is_none() && (GARRISON_FLOOR_FROM..DEATHTOUCH_TURN).contains(&turn) {
        if let Some(gcell) = own_general_cell(obs, memory) {
            let n = obs.h * obs.w;
            let ga = obs.army_grid[gcell.0 * obs.w + gcell.1] as i64;
            let own_total: i64 = (0..n)
                .filter(|&i| obs.owner_grid[i] as i32 == 1)
                .map(|i| obs.army_grid[i] as i64)
                .sum();
            let floor = garrison_floor(own_total);
            let chosen_from_gen = chosen[0] == 0
                && chosen[1] as usize == gcell.0
                && chosen[2] as usize == gcell.1;
            if ga as f64 >= GARRISON_RELEASE_FACTOR * floor as f64 && !chosen_from_gen {
                let releases: Vec<usize> = (0..4)
                    .map(|d| encode_action([0, gcell.0 as i32, gcell.1 as i32, d, 1]))
                    .filter(|&index| mask[index])
                    .collect();
                if !releases.is_empty() {
                    let mut best = releases[0];
                    if let Some(prior) = prior {
                        // First maximum again, matching Python's `max`.
                        for &index in &releases[1..] {
                            if prior[index] > prior[best] {
                                best = index;
                            }
                        }
                    }
                    return decode_action(best).unwrap();
                }
            }
        }
    }

    let redirect = |require_progress: bool| -> Option<Action5> {
        let prior = prior?;
        best_prior_legal_action(
            prior,
            obs,
            memory,
            &mask,
            prev_action,
            recent_actions,
            require_progress,
        )
    };

    if is_pass && has_nonpass {
        if let Some(alt) = redirect(false) {
            return alt;
        }
        // Last resort: any legal non-pass, keeping the no-pass hard rule.
        let index = (0..PASS_INDEX).find(|&i| mask[i]).unwrap();
        return decode_action(index).unwrap();
    }

    if blocks_oscillation(chosen, prev_action, obs, recent_actions, false, None) {
        if let Some(alt) = redirect(false) {
            return alt;
        }
    }

    let seek_target = if enemy_is_visible(obs, memory) {
        enemy_seek_target(obs, memory, None)
    } else {
        None
    };
    let goals = seek_goals(obs, memory, None);
    let dist_field = if goals.is_empty() {
        None
    } else {
        Some(path_distance_field(obs, &goals))
    };
    let ends = move_dest(chosen);
    if let (Some((sr, sc, tr, tc)), Some(_)) = (ends, prior) {
        if g.inside(tr, tc) {
            let dest_o = g.owner(tr, tc);
            let progress = path_progress(sr, sc, tr, tc, dist_field.as_ref(), seek_target);
            // A passive own-land shuffle or retreat: re-rank the shaped prior.
            // Short-circuited on `dest_o` first, as the Python is, so the
            // whole-board dilation only runs on an own-land destination.
            let passive = dest_o == 1
                && (progress < 0.0
                    || (progress <= 0.0 && newly_revealed_cells(obs, tr, tc) == 0));
            if passive {
                if let Some(alt) = redirect(true) {
                    if alt != chosen {
                        return alt;
                    }
                }
            }
        }
    }

    // Pre-contact: a fat pile on the general while the network retreats —
    // evacuate via the prior if an expand or leave-structure move ranks highest
    // among progressive options.
    if prior.is_some() && !enemy_is_visible(obs, memory) {
        if structure_idle_army(obs, memory) >= STRUCTURE_IDLE_ARMY {
            let own_struct = own_structure_mask(obs, memory);
            let leaves = match ends {
                Some((sr, sc, _, _)) => {
                    g.inside(sr, sc)
                        && own_struct[g.at(sr, sc)]
                        && g.army(sr, sc) >= STRUCTURE_IDLE_ARMY
                }
                None => false,
            };
            if !leaves {
                if let Some(alt) = redirect(true) {
                    if alt != chosen {
                        return alt;
                    }
                }
            }
        }
    }

    chosen
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
