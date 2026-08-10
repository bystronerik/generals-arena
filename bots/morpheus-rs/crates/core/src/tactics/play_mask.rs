//! The play mask: which legal actions Morpheus will consider at all.
//!
//! `legal_mask` says what the rules allow; this says what the bot plays. It
//! is the narrowest of the five decision surfaces and the one furthest
//! upstream — the network's prior is renormalized over it, the search's
//! candidate sets are drawn from it, and an action it clears cannot be
//! reinstated later. The garrison floor and the castle anchor are the two
//! rules that take actions away; both are reversed under threat, because a
//! rule that starves the defence is worse than the loss it prevents.

use crate::board::action::{legal_mask, PASS_INDEX};
use crate::board::memory::{
    VisibleMemory, OWNER_ENEMY,
};
use crate::board::state::MAX_CELLS;
use crate::board::transition::{BASE_COST, DEATHTOUCH_TURN};
use crate::io::wire::Observation;

use super::*;

pub fn garrison_floor(own_total_army: i64) -> i64 {
    let scaled = (GARRISON_FLOOR_FRAC * own_total_army.max(0) as f64) as i64;
    GARRISON_FLOOR_CAP.min(GARRISON_FLOOR_MIN.max(scaled))
}

/// Largest arrival army any visible stack can land on our general.
///
/// A stack at path distance `d` sheds one per hop, so its arrival is
/// `army - d`. This is what the garrison must strictly exceed to survive; the
/// threat-aware floor below keeps the mask from ever letting search split the
/// garrison beneath it.
pub fn max_threat_arrival(obs: &Observation, memory: &VisibleMemory) -> i64 {
    let gcell = match own_general_cell(obs, memory) {
        Some(cell) => cell,
        None => return 0,
    };
    let field = path_distance_field(obs, &[gcell]);
    let n = obs.h * obs.w;
    let mut worst = 0i64;
    for i in 0..n {
        if obs.owner_grid[i] as i32 != OWNER_ENEMY {
            continue;
        }
        let d = field.dist[i];
        if !(1..=DEFENSE_RADIUS).contains(&d) {
            continue;
        }
        worst = worst.max(obs.army_grid[i] as i64 - d as i64);
    }
    worst
}

/// Ban general-sourced moves that would drop the garrison below the floor.
///
/// Active from `GARRISON_FLOOR_FROM` until deathtouch (the kill phase is
/// all-in). A winning capture of the visible enemy general stays legal, and if
/// banning would leave no non-pass action the ban is skipped: protocol safety
/// over garrison policy.
fn apply_garrison_floor(
    obs: &Observation,
    memory: &VisibleMemory,
    mask: &mut [bool; N_ACTIONS],
) {
    let turn = turn_of(obs);
    if !(GARRISON_FLOOR_FROM..DEATHTOUCH_TURN).contains(&turn) {
        return;
    }
    let gcell = match own_general_cell(obs, memory) {
        Some(cell) => cell,
        None => return,
    };
    let g = Grids::new(obs);
    let n = obs.h * obs.w;
    let own_total: i64 = (0..n)
        .filter(|&i| obs.owner_grid[i] as i32 == 1)
        .map(|i| obs.army_grid[i] as i64)
        .sum();
    let floor = garrison_floor(own_total).max(max_threat_arrival(obs, memory) + 1);
    let ga = obs.army_grid[gcell.0 * obs.w + gcell.1] as i64;

    let tables = decode_tables();
    let enemy_gen = known_enemy_general_cell(obs, memory);

    let mut ban = [false; PASS_INDEX];
    let mut any_gen_src = false;
    let mut any_ban = false;
    for index in 0..PASS_INDEX {
        if !mask[index] || tables.kind[index] != 0 {
            continue;
        }
        if tables.sr[index] as usize != gcell.0 || tables.sc[index] as usize != gcell.1 {
            continue;
        }
        any_gen_src = true;
        // Channels 0–3 are full moves (leave 1 behind), 4–7 half splits
        // (leave `ceil(a/2)`).
        let channel = index / (PASS_INDEX / 9);
        let is_half = (4..=7).contains(&channel);
        let remaining = if is_half { ga - ga / 2 } else { 1 };
        if remaining >= floor {
            continue;
        }
        if let Some(target) = enemy_gen {
            let (tr, tc) = (tables.tr[index] as i32, tables.tc[index] as i32);
            let onto_gen = tr as usize == target.0 && tc as usize == target.1;
            let defender = if g.inside(tr, tc) { g.army(tr, tc) } else { 0 };
            let moved = if is_half { ga / 2 } else { ga - 1 };
            if onto_gen && moved > defender {
                continue;
            }
        }
        ban[index] = true;
        any_ban = true;
    }
    if !any_gen_src || !any_ban {
        return;
    }
    let mut candidate = *mask;
    for index in 0..PASS_INDEX {
        if ban[index] {
            candidate[index] = false;
        }
    }
    if !candidate[..PASS_INDEX].iter().any(|&v| v) {
        return;
    }
    *mask = candidate;
}

/// Pin the savings pile: no non-combat move leaves an underfunded site.
///
/// Without this the pile leaked — tips gathered to the site and the wave scores
/// marched the stack away before it reached the price (live probe: zero builds
/// in a full game). Combat moves stay legal, and the ban is keyed to the
/// *current* site, so an approaching enemy frees the old pile the same turn.
fn apply_castle_anchor(obs: &Observation, memory: &VisibleMemory, mask: &mut [bool; N_ACTIONS]) {
    let site = match castle_build_site(obs, memory) {
        Some(site) => site,
        None => return,
    };
    if general_threat(obs, memory).is_some() {
        // Defense of the general outranks castle savings: the anchored pile may
        // be the reinforcement that saves the game.
        return;
    }
    let g = Grids::new(obs);
    if obs.army_grid[site.0 * obs.w + site.1] as i64 >= BASE_COST as i64 {
        return;
    }
    let tables = decode_tables();
    let mut ban = [false; PASS_INDEX];
    let mut any_src = false;
    let mut any_ban = false;
    for index in 0..PASS_INDEX {
        if !mask[index] || tables.kind[index] != 0 {
            continue;
        }
        if tables.sr[index] as usize != site.0 || tables.sc[index] as usize != site.1 {
            continue;
        }
        any_src = true;
        let (tr, tc) = (tables.tr[index] as i32, tables.tc[index] as i32);
        let dest_enemy = g.inside(tr, tc) && g.owner(tr, tc) == OWNER_ENEMY;
        if !dest_enemy {
            ban[index] = true;
            any_ban = true;
        }
    }
    if !any_src || !any_ban {
        return;
    }
    let mut candidate = *mask;
    for index in 0..PASS_INDEX {
        if ban[index] {
            candidate[index] = false;
        }
    }
    if !candidate[..PASS_INDEX].iter().any(|&v| v) {
        return;
    }
    *mask = candidate;
}

/// The legal mask with Morpheus play rules applied.
///
/// Pass is illegal when any non-pass action exists. Until an enemy cell is
/// visible, moves *onto* the own general or an own castle are illegal — no idle
/// piles on structures — while leaving one stays legal so a large stack can
/// evacuate toward fog.
pub fn play_mask(
    obs: &Observation,
    memory: &VisibleMemory,
    cost_grid: Option<&[i32; MAX_CELLS]>,
) -> [bool; N_ACTIONS] {
    let base = legal_mask(obs, memory, cost_grid);
    let mut mask = base;
    if mask[..PASS_INDEX].iter().any(|&v| v) {
        mask[PASS_INDEX] = false;
    }

    apply_garrison_floor(obs, memory, &mut mask);
    apply_castle_anchor(obs, memory, &mut mask);

    if enemy_is_visible(obs, memory) {
        if !mask.iter().any(|&v| v) {
            return base;
        }
        return mask;
    }

    let g = Grids::new(obs);
    let own_struct = own_structure_mask(obs, memory);
    let tables = decode_tables();
    for index in 0..PASS_INDEX {
        if !mask[index] || tables.kind[index] != 0 {
            continue;
        }
        let (tr, tc) = (tables.tr[index] as i32, tables.tc[index] as i32);
        if !g.inside(tr, tc) {
            mask[index] = false;
            continue;
        }
        if own_struct[g.at(tr, tc)] {
            mask[index] = false;
        }
    }

    if !mask.iter().any(|&v| v) {
        let mut restored = base;
        if restored[..PASS_INDEX].iter().any(|&v| v) {
            restored[PASS_INDEX] = false;
        }
        return restored;
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::action::encode_action;
    use crate::board::memory::TYPE_GENERAL;

    fn board(h: usize, w: usize) -> Observation {
        let mut obs = Observation::with_dims(h, w);
        for i in 0..h * w {
            obs.type_grid[i] = 1;
        }
        obs
    }

    #[test]
    fn pass_is_illegal_whenever_anything_else_is_playable() {
        let mut obs = board(3, 3);
        obs.owner_grid[0] = 1;
        obs.army_grid[0] = 5;
        let memory = VisibleMemory::empty(3, 3);
        let mask = play_mask(&obs, &memory, None);
        assert!(!mask[PASS_INDEX]);
        assert!(mask[..PASS_INDEX].iter().any(|&v| v));
    }

    #[test]
    fn a_board_with_no_move_keeps_pass_legal() {
        let obs = board(3, 3);
        let memory = VisibleMemory::empty(3, 3);
        let mask = play_mask(&obs, &memory, None);
        assert!(mask[PASS_INDEX]);
    }

    #[test]
    fn before_contact_moves_onto_the_own_general_are_banned() {
        let mut obs = board(1, 3);
        // Own general at (0,1), an owned stack at (0,0) that could step onto it.
        obs.type_grid[1] = TYPE_GENERAL as u8;
        obs.owner_grid[1] = 1;
        obs.army_grid[1] = 3;
        obs.owner_grid[0] = 1;
        obs.army_grid[0] = 9;
        let mut memory = VisibleMemory::empty(1, 3);
        memory.own_general[1] = true;
        let mask = play_mask(&obs, &memory, None);
        // (0,0) moving right (direction 3) lands on the general.
        assert!(!mask[encode_action([0, 0, 0, 3, 0])]);
    }

    #[test]
    fn the_garrison_floor_scales_and_clamps() {
        assert_eq!(garrison_floor(0), GARRISON_FLOOR_MIN);
        assert_eq!(garrison_floor(10_000), GARRISON_FLOOR_CAP);
        // 4% of 350 is 14, between the floor and the cap.
        assert_eq!(garrison_floor(350), 14);
    }
}
