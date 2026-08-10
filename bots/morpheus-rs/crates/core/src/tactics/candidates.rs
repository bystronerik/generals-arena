//! Which actions the search is allowed to consider, and in what order.
//!
//! [`mandatory_action_indices`] is the set that cannot be dropped whatever
//! the prior says — a general capture first among them — and
//! [`policy_ordered_candidates`] is the widening order everything else takes.
//! The index lists below it are the cheap board questions that feed both.

use crate::board::action::{encode_action, PASS_INDEX};
use crate::board::memory::{
    VisibleMemory, OWNER_ENEMY, OWNER_NEUTRAL, TYPE_GENERAL,
};
use crate::board::transition::DIRECTIONS;
use crate::io::wire::Observation;

use super::*;

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
