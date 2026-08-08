//! Fogged observation emission from a full board state.
//!
//! Port of `bots/morpheus/observe.py`, which mirrors the engine's
//! `get_visibility` + `get_observation` + wire encoding without importing it.
//!
//! This is what makes the belief filter checkable: a particle is a guess at
//! the true state, and the test of that guess is whether emitting from it
//! reproduces the observation that actually arrived. So this has to agree with
//! the engine cell for cell — a fog rule that is off by one cell turns a
//! correct particle into a rejected one.

use crate::memory::{
    TYPE_CASTLE, TYPE_FOG, TYPE_GENERAL, TYPE_MOUNTAIN, TYPE_PLAIN, TYPE_STRUCTURE_FOG,
};
use crate::state::{GameState, MAX_CELLS};
use crate::transition::get_info;
use crate::wire::Observation;

/// Chebyshev-1 (3×3) visibility around every owned cell.
pub fn visibility_mask(state: &GameState, seat: usize) -> [bool; MAX_CELLS] {
    let mut visible = [false; MAX_CELLS];
    let (h, w) = (state.h as i32, state.w as i32);
    for r in 0..h {
        for c in 0..w {
            if !state.ownership[seat][(r * w + c) as usize] {
                continue;
            }
            for dr in -1..=1 {
                for dc in -1..=1 {
                    let (nr, nc) = (r + dr, c + dc);
                    if nr < 0 || nc < 0 || nr >= h || nc >= w {
                        continue;
                    }
                    visible[(nr * w + nc) as usize] = true;
                }
            }
        }
    }
    visible
}

/// Perspective-relative observation with competition fog.
///
/// Owner codes are relative — 1 is always "mine" — so a seat never needs to
/// know whether it is player 0 or 1.
pub fn emit_observation(state: &GameState, seat: usize) -> Observation {
    let opponent = 1 - seat;
    let visible = visibility_mask(state, seat);
    let info = get_info(state);
    let cells = state.cells();

    let mut obs = Observation::with_dims(state.h, state.w);
    obs.turn = state.time;
    obs.my_land = info.land[seat] as i32;
    obs.my_army = info.army[seat] as i32;
    obs.opp_land = info.land[opponent] as i32;
    obs.opp_army = info.army[opponent] as i32;

    for i in 0..cells {
        let structure = state.mountains[i] || state.castles[i];
        // Order matters and mirrors the Python's overwrites: plain, then the
        // two fog kinds, then visible terrain on top. A structure under fog
        // reads as STRUCTURE_FOG — the shape is known, the identity is not.
        let t = if visible[i] {
            if state.generals[i] {
                TYPE_GENERAL
            } else if state.castles[i] {
                TYPE_CASTLE
            } else if state.mountains[i] {
                TYPE_MOUNTAIN
            } else {
                TYPE_PLAIN
            }
        } else if structure {
            TYPE_STRUCTURE_FOG
        } else {
            TYPE_FOG
        };
        obs.type_grid[i] = t as u8;

        obs.owner_grid[i] = if visible[i] && state.ownership[seat][i] {
            1
        } else if visible[i] && state.ownership[opponent][i] {
            2
        } else {
            0
        };

        obs.army_grid[i] = if visible[i] { state.armies[i] } else { 0 };
    }
    obs
}

/// Exact match on visible cells, types, owners, armies, turn, public totals.
///
/// The belief filter's likelihood is this predicate: a particle whose emission
/// differs anywhere is impossible, not merely unlikely, and gets weight zero.
pub fn observations_match(simulated: &Observation, real: &Observation) -> bool {
    simulated.h == real.h
        && simulated.w == real.w
        && simulated.turn == real.turn
        && simulated.my_land == real.my_land
        && simulated.my_army == real.my_army
        && simulated.opp_land == real.opp_land
        && simulated.opp_army == real.opp_army
        && simulated.type_grid == real.type_grid
        && simulated.owner_grid == real.owner_grid
        && simulated.army_grid == real.army_grid
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_seat_board() -> GameState {
        let mut s = GameState::empty(5, 5);
        for i in 0..25 {
            s.passable[i] = true;
            s.ownership_neutral[i] = true;
        }
        s.ownership[0][0] = true;
        s.ownership_neutral[0] = false;
        s.generals[0] = true;
        s.armies[0] = 7;
        s.ownership[1][24] = true;
        s.ownership_neutral[24] = false;
        s.generals[24] = true;
        s.armies[24] = 9;
        s.general_positions = [[0, 0], [4, 4]];
        s
    }

    #[test]
    fn a_seat_sees_a_three_by_three_around_what_it_owns() {
        let s = two_seat_board();
        let visible = visibility_mask(&s, 0);
        assert!(visible[0] && visible[1] && visible[5] && visible[6]);
        assert!(!visible[2], "two cells away stays dark");
        assert!(!visible[24]);
    }

    #[test]
    fn fog_hides_armies_and_owners_but_not_the_shape_of_structures() {
        let mut s = two_seat_board();
        s.castles[12] = true; // centre, out of both seats' sight
        s.mountains[7] = true;
        let obs = emit_observation(&s, 0);

        assert_eq!(obs.type_grid[12] as i32, TYPE_STRUCTURE_FOG);
        assert_eq!(obs.type_grid[7] as i32, TYPE_STRUCTURE_FOG);
        assert_eq!(obs.type_grid[24] as i32, TYPE_FOG, "the enemy general is dark");
        assert_eq!(obs.owner_grid[24], 0);
        assert_eq!(obs.army_grid[24], 0);
    }

    #[test]
    fn owner_codes_are_relative_to_the_asking_seat() {
        let mut s = two_seat_board();
        // Adjacent, so each seat is inside the other's 3x3 and both cells
        // appear in both emissions — with the codes swapped.
        s.ownership[1][24] = false;
        s.generals[24] = false;
        s.ownership[1][1] = true;
        s.ownership_neutral[1] = false;
        s.generals[1] = true;
        s.armies[1] = 9;
        s.general_positions[1] = [0, 1];

        let a = emit_observation(&s, 0);
        let b = emit_observation(&s, 1);
        assert_eq!(a.owner_grid[0], 1);
        assert_eq!(b.owner_grid[0], 2);
        assert_eq!(a.owner_grid[1], 2);
        assert_eq!(b.owner_grid[1], 1);
    }

    #[test]
    fn public_totals_are_engine_truth_not_what_the_seat_can_see() {
        // Land and army counts include cells the asking seat cannot see;
        // that is how the wire protocol works, and a belief filter that
        // assumed otherwise would reject every correct particle.
        let s = two_seat_board();
        let obs = emit_observation(&s, 0);
        assert_eq!(obs.opp_land, 1);
        assert_eq!(obs.opp_army, 9);
        assert_eq!(obs.army_grid[24], 0, "still invisible, though it is counted");
    }

    #[test]
    fn an_emission_matches_itself_and_not_a_changed_board() {
        let s = two_seat_board();
        let obs = emit_observation(&s, 0);
        assert!(observations_match(&obs, &emit_observation(&s, 0)));

        let mut moved = s.clone();
        moved.armies[0] += 1;
        assert!(!observations_match(&obs, &emit_observation(&moved, 0)));
    }
}
