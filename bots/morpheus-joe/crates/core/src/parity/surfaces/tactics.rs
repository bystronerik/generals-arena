//! The five hand-written decision surfaces.

use crate::board::action::{
    legal_mask, N_ACTIONS,
};
use crate::tactics;
use crate::parity::codec::*;
use crate::parity::ints::Ints;
use crate::parity::Ctx;

/// observation + memory -> the play mask, plus the two sub-rules
/// that shape it
///
/// Emitted alongside the mask because both are *subtractive*: the
/// garrison floor and the castle anchor each clear bits and then
/// withdraw entirely if that would leave no non-pass action, so a
/// port that never applied either would produce a mask identical to
/// `legal_mask` on most frames and differ only where it matters.
/// The counts make "how many bits did each rule remove" a number.
pub(in crate::parity) fn playmask(
    ints: &mut Ints,
    out: &mut Vec<i64>,
    _ctx: &mut Ctx,
) -> Result<(), String> {
    let obs = read_observation(ints)?;
    let memory = read_memory(ints)?;
    let mask = tactics::play_mask(&obs, &memory, None);
    let base = legal_mask(&obs, &memory, None);
    out.extend(mask.iter().map(|&v| v as i64));
    out.push(
        base.iter()
            .zip(mask.iter())
            .filter(|(&b, &m)| b && !m)
            .count() as i64,
    );
    out.push(tactics::enemy_is_visible(&obs, &memory) as i64);
    let own_total: i64 = (0..obs.h * obs.w)
        .filter(|&i| obs.owner_grid[i] as i32 == 1)
        .map(|i| obs.army_grid[i] as i64)
        .sum();
    out.push(tactics::garrison_floor(own_total));
    out.push(tactics::max_threat_arrival(&obs, &memory));
    Ok(())
}

/// observation + memory + lite belief + previous action + a network
/// prior -> the heuristic scores and the blended prior
///
/// rewrite-plan §5 budgets 1e-9 for these in f64. Both sides are in
/// f64 throughout and they agree to the **last bit**, so that is
/// what the harness enforces, following M2's precedent: a tolerance
/// nothing approaches is a check that cannot fail.
pub(in crate::parity) fn shaping(
    ints: &mut Ints,
    out: &mut Vec<i64>,
    _ctx: &mut Ctx,
) -> Result<(), String> {
    let obs = read_observation(ints)?;
    let memory = read_memory(ints)?;
    let belief = read_belief_lite(ints)?;
    let (prev, _recent) = read_action_history(ints)?;
    let prior = ints.f64s(N_ACTIONS)?;
    let lam = ints.f64s(1)?[0];
    let log_clip = ints.f64s(1)?[0];
    let floor_frac = ints.f64s(1)?[0];
    // The mask is recomputed on both sides rather than sent: it is
    // 3,970 integers a case, and `playmask` already proves it.
    let mask = tactics::play_mask(&obs, &memory, None);
    let scores = tactics::heuristic_action_scores(
        &obs,
        &memory,
        &mask,
        Some(&belief),
        prev,
    );
    push_f64(out, &scores);
    push_f64(
        out,
        &tactics::blend_prior(&prior, &scores, &mask, lam, log_clip, floor_frac, 0.0),
    );
    Ok(())
}

/// observation + memory + prior + widening limit -> the mandatory
/// list and the ordered candidate list
pub(in crate::parity) fn candidates(
    ints: &mut Ints,
    out: &mut Vec<i64>,
    _ctx: &mut Ctx,
) -> Result<(), String> {
    let obs = read_observation(ints)?;
    let memory = read_memory(ints)?;
    let prior = ints.f64s(N_ACTIONS)?;
    let limit = ints.n()?;
    let use_play_mask = ints.next()? != 0;
    let mask = if use_play_mask {
        tactics::play_mask(&obs, &memory, None)
    } else {
        legal_mask(&obs, &memory, None)
    };
    let mandatory = tactics::mandatory_action_indices(&obs, &memory, &mask);
    out.push(mandatory.len() as i64);
    out.extend(mandatory.iter().map(|&i| i as i64));
    let candidates =
        tactics::policy_ordered_candidates(&prior, &mask, &mandatory, limit);
    out.push(candidates.len() as i64);
    out.extend(candidates.iter().map(|&i| i as i64));
    Ok(())
}

/// observation + memory + lite belief -> every planner's answer
///
/// One surface for the whole battery because they share their
/// expensive input (a BFS field) and because a planner that returns
/// `None` everywhere is the failure mode worth catching: each answer
/// rides with an explicit present/absent flag rather than a sentinel
/// that could be confused with a real cell.
pub(in crate::parity) fn planners(
    ints: &mut Ints,
    out: &mut Vec<i64>,
    _ctx: &mut Ctx,
) -> Result<(), String> {
    let obs = read_observation(ints)?;
    let memory = read_memory(ints)?;
    let belief = read_belief_lite(ints)?;

    push_cell(out, tactics::own_general_cell(&obs, &memory));
    push_cell(out, tactics::known_enemy_general_cell(&obs, &memory));
    push_cell(out, tactics::king_cell(&obs, None));
    push_cell(out, tactics::believed_enemy_general(Some(&belief)));
    push_cell(out, tactics::enemy_seek_target(&obs, &memory, Some(&belief)));
    let goals = tactics::seek_goals(&obs, &memory, Some(&belief));
    out.push(goals.len() as i64);
    for (r, c) in &goals {
        out.extend([*r as i64, *c as i64]);
    }
    let exclude = tactics::own_general_cell(&obs, &memory);
    push_cell(
        out,
        tactics::wave_assembly_cell(&obs, &memory, Some(&belief), exclude),
    );
    let (share, max_own, total) = tactics::army_concentration(&obs, exclude);
    push_f64(out, &[share]);
    out.extend([max_own, total]);
    out.push(tactics::structure_idle_army(&obs, &memory));
    out.push(tactics::max_threat_arrival(&obs, &memory));

    let site = tactics::castle_build_site(&obs, &memory);
    push_cell(out, site);
    match site {
        Some(site) => {
            push_action(out, tactics::castle_tithe_move(&obs, &memory, site))
        }
        None => push_action(out, None),
    }

    let threat = tactics::general_threat(&obs, &memory);
    match threat {
        Some((cell, d)) => {
            out.extend([1, cell.0 as i64, cell.1 as i64, d as i64]);
            push_action(
                out,
                tactics::defend_general_move(&obs, &memory, (cell, d)),
            );
        }
        None => {
            out.extend([0, -1, -1, -1]);
            push_action(out, None);
        }
    }

    match tactics::kill_plan(&obs, &memory) {
        Some((steps, margin, first)) => {
            out.extend([1, steps as i64, margin]);
            push_action(out, Some(first));
        }
        None => {
            out.extend([0, -1, -1]);
            push_action(out, None);
        }
    }

    // The two whole-board fields the scoring path rides on.
    let reveal = tactics::reveal_count_grid(&obs);
    out.extend(reveal.iter().copied());
    let goal_cells: Vec<(usize, usize)> = goals.clone();
    let field = tactics::path_distance_field(&obs, &goal_cells);
    out.extend(field.dist.iter().map(|&v| v as i64));
    Ok(())
}

/// observation + memory + a chosen action + history + an optional
/// prior -> the action the hard rules commit
///
/// The belief is deliberately absent: `constrain_nn_action` takes one
/// and never reads it (see `tactics.rs`), which contradicts
/// rewrite-plan §5's "final-state wrinkle". Passing one here would
/// paper over that.
pub(in crate::parity) fn constrain(
    ints: &mut Ints,
    out: &mut Vec<i64>,
    _ctx: &mut Ctx,
) -> Result<(), String> {
    let obs = read_observation(ints)?;
    let memory = read_memory(ints)?;
    let chosen = ints.action()?;
    let (prev, recent) = read_action_history(ints)?;
    let has_prior = ints.next()? != 0;
    let prior = ints.f64s(N_ACTIONS)?;
    let action = tactics::constrain_nn_action(
        &obs,
        &memory,
        chosen,
        prev,
        &recent,
        if has_prior { Some(&prior) } else { None },
        None,
    );
    out.extend(action.iter().map(|&v| v as i64));
    // The oscillation predicate on the chosen move, both ways: with
    // the precomputed reveal grid the redirect loop uses, and with
    // the whole-board dilation the Python calls. They must agree.
    let reveal = tactics::reveal_count_grid(&obs);
    out.push(
        tactics::blocks_oscillation(chosen, prev, &obs, &recent, false, None) as i64,
    );
    out.push(tactics::blocks_oscillation(
        chosen,
        prev,
        &obs,
        &recent,
        false,
        Some(&reveal),
    ) as i64);
    Ok(())
}
