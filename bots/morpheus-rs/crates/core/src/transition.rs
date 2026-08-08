//! The exact competition transition: builds first, then a deathtouch-wrapped step.
//!
//! Port of `bots/morpheus/transition.py`, which itself mirrors the engine's
//! composition without importing it. This is the kernel the whole rewrite is
//! betting on — measured at 142.6 ms p99 on the M3 Pro and 357.5 ms on one x86
//! core (M0 baseline), applied once per particle per turn *and* once per
//! simulation, which is why it is ported first.
//!
//! Truncation at turn 1200 is a driver check (`at_truncation`), not part of
//! `transition` — same boundary the engine's `make_transition` draws.
//!
//! Several behaviours here look like bugs and are the specification. Each is
//! commented where it lives; the pattern is that the Python mirrors JAX
//! semantics (index wrapping, bitwise ops on int32 action fields) that a
//! straight reading of the intent would get subtly wrong.

use crate::state::{GameInfo, GameState, MAX_CELLS};

/// `pass` field: 0 move, 1 skip, 2 build.
pub const BUILD: i32 = 2;
pub const PASS_ACTION: [i32; 5] = [1, 0, 0, 0, 0];

/// `(dr, dc)` for direction codes 0..3 — up, down, left, right.
pub const DIRECTIONS: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

// Competition modifiers (`GeneralsEnv(mode="competition")`).
pub const BASE_COST: i32 = 35;
pub const PROXIMITY_PENALTY: i32 = 14;
pub const PROXIMITY_DECAY: i32 = 2;
pub const RADIUS: i32 = (PROXIMITY_PENALTY - 1) / PROXIMITY_DECAY;
pub const DEATHTOUCH_TURN: i32 = 800;
pub const TRUNCATION_TURN: i32 = 1200;

/// One turn's joint action, seat-indexed.
pub type Actions = [[i32; 5]; 2];

/// Driver-boundary hard draw: true when time has reached the cap.
pub fn at_truncation(state: &GameState) -> bool {
    state.winner < 0 && state.time >= TRUNCATION_TURN
}

/// Live castle price for `seat`, per cell.
///
/// Each own structure surcharges cells within a Manhattan radius, decaying by
/// `PROXIMITY_DECAY` per step. Structures stack: two nearby castles both add.
pub fn build_cost_grid(state: &GameState, seat: usize) -> [i32; MAX_CELLS] {
    let mut cost = [BASE_COST; MAX_CELLS];
    let (h, w) = (state.h as i32, state.w as i32);

    for sr in 0..h {
        for sc in 0..w {
            let src = (sr * w + sc) as usize;
            let structure = (state.castles[src] || state.generals[src])
                && state.ownership[seat][src];
            if !structure {
                continue;
            }
            for di in -RADIUS..=RADIUS {
                for dj in -RADIUS..=RADIUS {
                    let surcharge = PROXIMITY_PENALTY - PROXIMITY_DECAY * (di.abs() + dj.abs());
                    if surcharge <= 0 {
                        continue;
                    }
                    // The Python pads the structure grid and shifts it, so a
                    // structure near an edge surcharges only cells that exist.
                    let (r, c) = (sr + di, sc + dj);
                    if r < 0 || c < 0 || r >= h || c >= w {
                        continue;
                    }
                    cost[(r * w + c) as usize] += surcharge;
                }
            }
        }
    }
    cost
}

fn apply_one_build(state: &mut GameState, seat: usize, action: [i32; 5], cost: &[i32; MAX_CELLS]) {
    if action[0] != BUILD {
        return;
    }
    let (r, c) = (action[1], action[2]);
    let in_bounds = state.in_bounds(r, c);
    // The Python clamps before indexing, then gates on `in_bounds`, so an
    // out-of-bounds build reads a clamped cell and is rejected anyway.
    let rs = r.clamp(0, state.h as i32 - 1) as usize;
    let cs = c.clamp(0, state.w as i32 - 1) as usize;
    let at = rs * state.w + cs;

    let owns = state.ownership[seat][at];
    let plain = !state.generals[at] && !state.castles[at];
    let affords = state.armies[at] >= cost[at];
    let alive = state.winner < 0;
    if !(in_bounds && owns && plain && affords && alive) {
        return;
    }
    state.armies[at] -= cost[at];
    state.castles[at] = true;
}

/// Resolve both builds, then rewrite every BUILD action to pass.
///
/// Seat 0's build resolves first and changes the board seat 1 prices against —
/// a castle seat 0 just placed raises seat 1's cost nearby in the same turn.
pub fn apply_build_actions(state: &mut GameState, actions: &mut Actions) {
    for seat in 0..2 {
        if actions[seat][0] == BUILD {
            let cost = build_cost_grid(state, seat);
            apply_one_build(state, seat, actions[seat], &cost);
        }
    }
    for seat in 0..2 {
        if actions[seat][0] == BUILD {
            actions[seat] = PASS_ACTION;
        }
    }
}

/// Which seat moves first: chase > reinforce > smaller source; seat 0 on a
/// full tie; seat 1 when only seat 0 passes.
pub fn determine_move_order(state: &GameState, actions: &Actions) -> usize {
    let [pass_0, row_0, col_0, dir_0, _] = actions[0];
    let [pass_1, row_1, col_1, dir_1, _] = actions[1];

    // Bitwise, matching JAX's `pass_0 & ~pass_1` over int32 action fields.
    // Not `pass_0 == 1 && pass_1 != 1`: with a build (2) still in the field
    // those disagree, and the engine's answer is this one.
    let only_p0_passes = (pass_0 & !pass_1) != 0;

    let (d0r, d0c) = DIRECTIONS[dir_0.clamp(0, 3) as usize];
    let (d1r, d1c) = DIRECTIONS[dir_1.clamp(0, 3) as usize];
    let (di_0, dj_0) = (row_0 + d0r, col_0 + d0c);
    let (di_1, dj_1) = (row_1 + d1r, col_1 + d1c);

    let p0_chasing = di_0 == row_1 && dj_0 == col_1;
    let p1_chasing = di_1 == row_0 && dj_1 == col_0;

    // NumPy's negative-index wrap, load-bearing: a pass is `[1,0,0,0,0]`, so
    // `di = -1` and "reinforcing" is read off the *last row*. See
    // `GameState::wrapped_idx`.
    let p0_reinforcing = state
        .wrapped_idx(di_0, dj_0)
        .map(|i| state.ownership[0][i])
        .unwrap_or(false);
    let p1_reinforcing = state
        .wrapped_idx(di_1, dj_1)
        .map(|i| state.ownership[1][i])
        .unwrap_or(false);

    let army_0 = state
        .wrapped_idx(row_0, col_0)
        .map(|i| state.armies[i])
        .unwrap_or(0);
    let army_1 = state
        .wrapped_idx(row_1, col_1)
        .map(|i| state.armies[i])
        .unwrap_or(0);

    let p1_wins_by_chase = p1_chasing && !p0_chasing;
    let tie_on_chase = p0_chasing == p1_chasing;
    let p1_wins_by_reinforce = tie_on_chase && p1_reinforcing && !p0_reinforcing;
    let tie_on_reinforce = p0_reinforcing == p1_reinforcing;
    let p1_wins_by_army = tie_on_chase && tie_on_reinforce && army_1 < army_0;

    if p1_wins_by_chase || p1_wins_by_reinforce || p1_wins_by_army || only_p0_passes {
        1
    } else {
        0
    }
}

/// How many units leave the source: half when splitting, else all but one.
///
/// The clamp is the Python's `max(0, min(raw, source_army - 1))` and does the
/// real work: a cell holding one unit (or none) moves nothing, and no move
/// ever empties its source. Written as that exact composition rather than a
/// three-way `clamp`, which was equivalent but so defensive that breaking the
/// `- 1` above it changed nothing at all.
#[inline]
pub fn army_to_move(source_army: i32, split: i32) -> i32 {
    let raw = if split == 1 {
        source_army.div_euclid(2)
    } else {
        source_army - 1
    };
    raw.min(source_army - 1).max(0)
}

fn apply_move(
    state: &mut GameState,
    seat: usize,
    src: usize,
    dst: usize,
    army: i32,
) {
    let target_owner_0 = state.ownership[0][dst];
    let target_owner_1 = state.ownership[1][dst];
    let target_neutral = state.ownership_neutral[dst];
    let moving_to_own = (seat == 0 && target_owner_0) || (seat == 1 && target_owner_1);

    if moving_to_own {
        state.armies[dst] += army;
        state.armies[src] -= army;
        return;
    }

    let target_army = state.armies[dst];
    let attacker_wins = army > target_army;
    state.armies[dst] = (target_army - army).abs();
    state.armies[src] -= army;

    if !attacker_wins {
        return;
    }
    state.ownership[seat][dst] = true;
    if target_owner_0 && seat == 1 {
        state.ownership[0][dst] = false;
    }
    if target_owner_1 && seat == 0 {
        state.ownership[1][dst] = false;
    }
    if target_neutral {
        state.ownership_neutral[dst] = false;
    }
    if state.generals[dst] {
        state.winner = seat as i32;
    }
}

/// Is this a move that would actually execute, and where to?
///
/// Shared by `execute_action` and the deathtouch check so the two can never
/// disagree about what "a valid move" means.
fn resolve_move(
    state: &GameState,
    seat: usize,
    si: i32,
    sj: i32,
    direction: i32,
    split: i32,
) -> Option<(usize, usize)> {
    if !state.in_bounds(si, sj) {
        return None;
    }
    let (dr, dc) = DIRECTIONS[direction.clamp(0, 3) as usize];
    let (di, dj) = (si + dr, sj + dc);
    if !state.in_bounds(di, dj) {
        return None;
    }
    let src = state.idx(si as usize, sj as usize);
    let dst = state.idx(di as usize, dj as usize);
    let army = army_to_move(state.armies[src], split);
    if state.ownership[seat][src] && army > 0 && state.passable[dst] {
        Some((src, dst))
    } else {
        None
    }
}

pub fn execute_action(state: &mut GameState, seat: usize, action: [i32; 5]) {
    let [pass_turn, si, sj, direction, split] = action;
    // Only `pass == 1` skips. A BUILD that reached here (it should have been
    // rewritten) is executed as a move, exactly as the Python does.
    if pass_turn == 1 {
        return;
    }
    if let Some((src, dst)) = resolve_move(state, seat, si, sj, direction, split) {
        let army = army_to_move(state.armies[src], split);
        apply_move(state, seat, src, dst, army);
    }
}

fn transfer_loser_cells_to_winner(state: &mut GameState) {
    let winner = state.winner as usize;
    let loser = 1 - winner;
    // Captured *before* the loser's mask is cleared. The Python reads
    // `state.ownership[loser]` off the original tuple while writing into a
    // copy, so the neutral update sees the pre-clear mask; folding the two
    // steps together here would quietly change the result.
    let loser_mask = state.ownership[loser];
    for i in 0..MAX_CELLS {
        state.ownership[winner][i] |= loser_mask[i];
        state.ownership[loser][i] = false;
        state.ownership_neutral[i] &= !loser_mask[i];
    }
}

/// Growth at the *post-increment* time: every cell on the 50-tick, structures
/// on even ticks.
pub fn global_update(state: &mut GameState) {
    let cells = state.cells();
    if state.time % 50 == 0 {
        for i in 0..cells {
            state.armies[i] += state.ownership[0][i] as i32 + state.ownership[1][i] as i32;
        }
    }
    if state.time % 2 == 0 {
        for i in 0..cells {
            if state.generals[i] || state.castles[i] {
                state.armies[i] += state.ownership[0][i] as i32 + state.ownership[1][i] as i32;
            }
        }
    }
}

pub fn get_info(state: &GameState) -> GameInfo {
    let mut army = [0i64; 2];
    let mut land = [0i64; 2];
    for seat in 0..2 {
        for i in 0..state.cells() {
            if state.ownership[seat][i] {
                army[seat] += state.armies[i] as i64;
                land[seat] += 1;
            }
        }
    }
    GameInfo {
        army,
        land,
        is_done: state.winner >= 0,
        winner: state.winner,
        time: state.time,
    }
}

/// Base `game.step` without the build or deathtouch modifiers.
pub fn step_base(state: &GameState, actions: &Actions) -> (GameState, GameInfo) {
    let mut next = state.clone();
    let done_before = state.winner >= 0;

    let first = determine_move_order(state, actions);
    execute_action(&mut next, first, actions[first]);
    execute_action(&mut next, 1 - first, actions[1 - first]);

    if !done_before {
        next.time += 1;
    }
    if next.winner >= 0 {
        transfer_loser_cells_to_winner(&mut next);
    } else {
        global_update(&mut next);
    }
    let info = get_info(&next);
    (next, info)
}

/// True iff a valid move from this state lands on the enemy general.
fn executes_onto_general(state: &GameState, seat: usize, action: [i32; 5]) -> bool {
    let [pass_turn, si, sj, direction, split] = action;
    if pass_turn != 0 {
        return false;
    }
    let Some((_, dst)) = resolve_move(state, seat, si, sj, direction, split) else {
        return false;
    };
    let g = state.general_positions[1 - seat];
    if g[0] < 0 || g[1] < 0 {
        return false;
    }
    dst == state.idx(g[0] as usize, g[1] as usize)
}

/// Deathtouch wrapper around `step_base`, with builds already rewritten.
///
/// From turn 800 any move that *executes* onto the enemy general wins outright,
/// however large the defence. Two simultaneous touches are a draw, and a
/// defender who captures the attacker's source cell first stops the touch —
/// which is why `t_second` is evaluated against the mid-state, after the first
/// seat has already moved.
pub fn step_deathtouch(
    state: &GameState,
    actions: &Actions,
    turn: i32,
) -> (GameState, GameInfo) {
    let active = state.winner < 0 && state.time >= turn;

    let first = determine_move_order(state, actions);
    let t_first = executes_onto_general(state, first, actions[first]);
    let mut mid = state.clone();
    execute_action(&mut mid, first, actions[first]);
    let t_second = executes_onto_general(&mid, 1 - first, actions[1 - first]);

    let touch = if first == 0 {
        [t_first && active, t_second && active]
    } else {
        [t_second && active, t_first && active]
    };

    let (mut next, _) = step_base(state, actions);

    // Both generals fell inside one step: the first seat's capture set a
    // winner in `mid`, and the full step ended with a different one.
    let both_captured = state.winner < 0 && mid.winner >= 0 && next.winner != mid.winner;
    let both = (touch[0] && touch[1]) || both_captured;
    let one = (touch[0] ^ touch[1]) && !both_captured;
    let toucher = if touch[0] { 0 } else { 1 };

    let winner = if both {
        -1
    } else if one {
        toucher
    } else {
        next.winner
    };
    let need_transfer = one && winner >= 0 && state.winner < 0;
    next.winner = winner;
    if need_transfer {
        transfer_loser_cells_to_winner(&mut next);
    }

    let mut info = get_info(&next);
    // A mutual touch ends the game with no winner: done, but drawn.
    info.is_done = info.is_done || both;
    (next, info)
}

/// The exact competition transition: builds, then the deathtouch-wrapped step.
pub fn transition(state: &GameState, actions: &Actions) -> (GameState, GameInfo) {
    let has_build = actions[0][0] == BUILD || actions[1][0] == BUILD;
    if !has_build {
        if state.winner < 0 && state.time >= DEATHTOUCH_TURN {
            return step_deathtouch(state, actions, DEATHTOUCH_TURN);
        }
        return step_base(state, actions);
    }

    let mut built = state.clone();
    let mut rewritten = *actions;
    apply_build_actions(&mut built, &mut rewritten);
    if built.winner < 0 && built.time >= DEATHTOUCH_TURN {
        step_deathtouch(&built, &rewritten, DEATHTOUCH_TURN)
    } else {
        step_base(&built, &rewritten)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two seats on a 3×3 board, generals in opposite corners.
    fn board() -> GameState {
        let mut s = GameState::empty(3, 3);
        for i in 0..9 {
            s.passable[i] = true;
            s.ownership_neutral[i] = true;
        }
        s.ownership[0][0] = true;
        s.ownership_neutral[0] = false;
        s.generals[0] = true;
        s.armies[0] = 10;
        s.ownership[1][8] = true;
        s.ownership_neutral[8] = false;
        s.generals[8] = true;
        s.armies[8] = 10;
        s.general_positions = [[0, 0], [2, 2]];
        s
    }

    #[test]
    fn all_but_one_leaves_a_unit_behind() {
        assert_eq!(army_to_move(10, 0), 9);
        assert_eq!(army_to_move(10, 1), 5);
        assert_eq!(army_to_move(1, 0), 0);
        assert_eq!(army_to_move(1, 1), 0);
        assert_eq!(army_to_move(0, 0), 0);
        assert_eq!(army_to_move(3, 1), 1);
    }

    #[test]
    fn a_move_onto_a_neutral_cell_captures_it() {
        let s = board();
        let actions = [[0, 0, 0, 3, 0], PASS_ACTION]; // seat 0 right
        let (next, info) = transition(&s, &actions);
        assert!(next.ownership[0][1]);
        assert!(!next.ownership_neutral[1]);
        assert_eq!(next.armies[1], 9);
        assert_eq!(info.land[0], 2);
    }

    #[test]
    fn growth_lands_on_the_post_increment_clock() {
        // time 0 -> 1 after the step: neither 50 nor even, so no growth.
        let mut s = board();
        let (a, _) = transition(&s, &[PASS_ACTION, PASS_ACTION]);
        assert_eq!(a.armies[0], 10);
        // time 1 -> 2: even, so the general grows.
        s.time = 1;
        let (b, _) = transition(&s, &[PASS_ACTION, PASS_ACTION]);
        assert_eq!(b.armies[0], 11);
        // time 49 -> 50: every owned cell grows, and 50 is even, so a
        // general grows twice.
        s.time = 49;
        let (c, _) = transition(&s, &[PASS_ACTION, PASS_ACTION]);
        assert_eq!(c.armies[0], 12);
    }

    #[test]
    fn capturing_the_general_ends_it_and_transfers_the_land() {
        let mut s = board();
        s.ownership[0][7] = true;
        s.ownership_neutral[7] = false;
        s.armies[7] = 50;
        s.ownership[1][5] = true;
        s.ownership_neutral[5] = false;
        let actions = [[0, 2, 1, 3, 0], PASS_ACTION]; // seat 0 (2,1) -> (2,2)
        let (next, info) = transition(&s, &actions);
        assert_eq!(next.winner, 0);
        assert!(info.is_done);
        assert!(next.ownership[0][5], "the loser's cells go to the winner");
        assert!(!next.ownership[1].iter().any(|&x| x));
    }

    #[test]
    fn before_turn_800_one_unit_does_not_kill_a_general() {
        let mut s = board();
        s.time = 799;
        s.ownership[0][7] = true;
        s.ownership_neutral[7] = false;
        s.armies[7] = 2; // moves 1
        let (next, _) = transition(&s, &[[0, 2, 1, 3, 0], PASS_ACTION]);
        assert_eq!(next.winner, -1);
        assert!(next.ownership[1][8], "the general holds");
    }

    #[test]
    fn from_turn_800_one_unit_is_lethal() {
        let mut s = board();
        s.time = 800;
        s.ownership[0][7] = true;
        s.ownership_neutral[7] = false;
        s.armies[7] = 2;
        let (next, info) = transition(&s, &[[0, 2, 1, 3, 0], PASS_ACTION]);
        assert_eq!(next.winner, 0);
        assert!(info.is_done);
    }

    #[test]
    fn simultaneous_deathtouch_is_a_draw() {
        let mut s = GameState::empty(1, 4);
        for i in 0..4 {
            s.passable[i] = true;
        }
        s.time = 900;
        s.generals[0] = true;
        s.ownership[0][0] = true;
        s.armies[0] = 5;
        s.ownership[0][1] = true;
        s.armies[1] = 2;
        s.generals[3] = true;
        s.ownership[1][3] = true;
        s.armies[3] = 5;
        // The seats are interleaved: each one's spare cell sits next to the
        // *other's* general, so both touches execute in the same step.
        s.ownership[0][2] = true;
        s.armies[2] = 2;
        s.ownership[1][1] = true;
        s.armies[1] = 2;
        s.general_positions = [[0, 0], [0, 3]];

        let (next, info) = transition(&s, &[[0, 0, 2, 3, 0], [0, 0, 1, 2, 0]]);
        assert_eq!(next.winner, -1, "a mutual touch has no winner");
        assert!(info.is_done, "but the game is over");
    }

    #[test]
    fn truncation_is_a_driver_check_not_part_of_the_step() {
        let mut s = board();
        s.time = TRUNCATION_TURN;
        assert!(at_truncation(&s));
        let (next, info) = transition(&s, &[PASS_ACTION, PASS_ACTION]);
        assert_eq!(next.winner, -1);
        assert!(!info.is_done, "the step does not know about the cap");
    }

    #[test]
    fn a_build_the_cell_cannot_afford_is_refused() {
        // (1,1) is two steps from the general at (0,0), so it prices at
        // 35 + (14 - 2*2) = 45. Forty units is not enough, and the refusal is
        // silent: no castle, no spend. Verified against the Python oracle.
        let mut s = board();
        s.ownership[0][4] = true;
        s.ownership_neutral[4] = false;
        s.armies[4] = 40;
        let (next, _) = transition(&s, &[[BUILD, 1, 1, 0, 0], PASS_ACTION]);
        assert!(!next.castles[4]);
        assert_eq!(next.armies[4], 40);
    }

    #[test]
    fn an_affordable_build_pays_its_price_and_then_grows() {
        // Far corner of a 1x4 board: distance 3 from the general, so
        // 35 + (14 - 2*3) = 43. Sixty pays it, and t=1 -> 2 is even, so the
        // new castle grows once in the same step: 60 - 43 + 1 = 18.
        let mut s = GameState::empty(1, 4);
        for i in 0..4 {
            s.passable[i] = true;
        }
        s.time = 1;
        s.generals[0] = true;
        s.ownership[0][0] = true;
        s.armies[0] = 1;
        s.ownership[0][3] = true;
        s.armies[3] = 60;
        s.general_positions = [[0, 0], [0, 3]];
        let (next, _) = transition(&s, &[[BUILD, 0, 3, 0, 0], PASS_ACTION]);
        assert!(next.castles[3]);
        assert_eq!(next.armies[3], 18);
    }

    #[test]
    fn build_prices_rise_near_an_own_structure() {
        let s = board();
        let cost = build_cost_grid(&s, 0);
        // The general at (0,0) surcharges its own cell and its neighbours,
        // decaying by 2 per Manhattan step out to a radius of 6 — which on a
        // 3x3 board reaches every cell, the far corner included.
        assert_eq!(cost[0], BASE_COST + PROXIMITY_PENALTY);
        assert_eq!(cost[1], BASE_COST + PROXIMITY_PENALTY - PROXIMITY_DECAY);
        assert_eq!(cost[8], BASE_COST + PROXIMITY_PENALTY - 4 * PROXIMITY_DECAY);
    }

    #[test]
    fn only_seat_zero_passing_hands_the_first_move_to_seat_one() {
        let s = board();
        let order = determine_move_order(&s, &[PASS_ACTION, [0, 2, 2, 0, 0]]);
        assert_eq!(order, 1);
    }

    #[test]
    fn a_chase_moves_first() {
        let mut s = GameState::empty(1, 3);
        for i in 0..3 {
            s.passable[i] = true;
        }
        s.ownership[0][0] = true;
        s.armies[0] = 5;
        s.ownership[1][1] = true;
        s.armies[1] = 5;
        // Seat 0 steps onto seat 1's source; seat 1 steps away. Seat 0 chases,
        // so seat 0 resolves first even though seat 1 has the smaller source.
        let order = determine_move_order(&s, &[[0, 0, 0, 3, 0], [0, 0, 1, 3, 0]]);
        assert_eq!(order, 0);
    }
}
