//! The last word: hard rules applied to the action the network chose.
//!
//! Everything upstream is a preference — a mask, an ordering, a blend.
//! [`constrain_nn_action`] is the only place a decision is overruled outright,
//! and it is where the kill window, the general defence and the capture rules
//! outrank whatever the search came back with.

use crate::board::action::{decode_action, encode_action, PASS_INDEX};
use crate::belief::{Action5, BeliefState};
use crate::board::memory::{
    VisibleMemory, OWNER_ENEMY,
};
use crate::board::transition::{DEATHTOUCH_TURN, DIRECTIONS};
use crate::io::wire::Observation;

use super::*;

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
