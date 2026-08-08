//! The 49×21×21 observation tensor.
//!
//! Port of `bots/morpheus/tensor.py`. Perspective-relative: both seats use the
//! same builder, and owner code 1 always means "mine".
//!
//! **Float width is part of the contract.** The Python mixes `float64` and
//! `float32` deliberately — `army_value` computes a log in double and casts
//! down, while the coordinate planes are `float32` arithmetic throughout
//! (`np.arange(H, dtype=np.float32) / 20.0` stays single). Each site below
//! mirrors its counterpart's width. Computing everything in `f64` and casting
//! at the end would be *more* accurate and would fail tier-2 parity, because
//! the network was trained on the rounding the Python actually does.
//!
//! Belief planes arrive through an injected `BeliefSummary`; aggregating
//! particles into one is M4's job (`particle_summary.py`).

use crate::memory::{
    VisibleMemory, OWNER_ENEMY, OWNER_ME, OWNER_NEUTRAL, TYPE_FOG, TYPE_STRUCTURE_FOG,
};
use crate::wire::Observation;

pub const PAD: usize = 21;
pub const N_PLANES: usize = 49;
pub const PLANE_CELLS: usize = PAD * PAD;
pub const TENSOR_LEN: usize = N_PLANES * PLANE_CELLS;
/// Initial guess; the model manifest records replacements.
pub const ARMY_SCALE: f64 = 4096.0;
pub const TRUNCATION_TURN: i32 = 1200;
pub const DEATHTOUCH_TURN: i32 = 800;

// Plane indices (0-based; the spec table is 1-based).
pub const P_BOARD_MASK: usize = 0;
pub const P_VISIBLE_NOW: usize = 1;
pub const P_FOG_NONSTRUCTURE: usize = 2;
pub const P_FOG_STRUCTURE: usize = 3;
pub const P_KNOWN_MOUNTAIN: usize = 4;
pub const P_KNOWN_PASSABLE: usize = 5;
pub const P_KNOWN_CASTLE: usize = 6;
pub const P_OWN_GENERAL: usize = 7;
pub const P_KNOWN_ENEMY_GENERAL: usize = 8;
pub const P_OWNED_NOW: usize = 9;
pub const P_ENEMY_VISIBLE: usize = 10;
pub const P_NEUTRAL_VISIBLE: usize = 11;
pub const P_OWNED_ARMY: usize = 12;
pub const P_ENEMY_ARMY_VISIBLE: usize = 13;
pub const P_EVER_VISIBLE: usize = 14;
pub const P_SIGHT_AGE: usize = 15;
pub const P_REMEMBERED_OWNED: usize = 16;
pub const P_REMEMBERED_ENEMY: usize = 17;
pub const P_REMEMBERED_NEUTRAL: usize = 18;
pub const P_REMEMBERED_ENEMY_ARMY: usize = 19;
pub const P_REMEMBERED_OWN_CASTLE: usize = 20;
pub const P_REMEMBERED_ENEMY_CASTLE: usize = 21;
pub const P_BELIEF_ENEMY_OWNER: usize = 22;
pub const P_BELIEF_ENEMY_ARMY_MEAN: usize = 23;
pub const P_BELIEF_ENEMY_ARMY_STD: usize = 24;
pub const P_BELIEF_ENEMY_GENERAL: usize = 25;
pub const P_BELIEF_ENEMY_CASTLE_OWNER: usize = 26;
pub const P_BELIEF_ENEMY_VISIBILITY: usize = 27;
pub const P_BELIEF_OWNER_ENTROPY: usize = 28;
pub const P_PREV_MOVE_SOURCE: usize = 29;
pub const P_PREV_MOVE_DEST: usize = 30;
pub const P_PREV_MOVE_KIND: usize = 31;
pub const P_PREV_BUILD_CELL: usize = 32;
pub const P_ROW_COORD: usize = 33;
pub const P_COL_COORD: usize = 34;
pub const P_ROW_FROM_GENERAL: usize = 35;
pub const P_COL_FROM_GENERAL: usize = 36;
pub const P_TURN_FRACTION: usize = 37;
pub const P_PRE_DEATHTOUCH: usize = 38;
pub const P_DEATHTOUCH_ACTIVE: usize = 39;
pub const P_STRUCTURE_GROWTH_NEXT: usize = 40;
pub const P_BULK_GROWTH_COUNTDOWN: usize = 41;
pub const P_OWN_LAND_FRACTION: usize = 42;
pub const P_ENEMY_LAND_FRACTION: usize = 43;
pub const P_OWN_ARMY_TOTAL: usize = 44;
pub const P_ENEMY_ARMY_TOTAL: usize = 45;
pub const P_LAND_MARGIN: usize = 46;
pub const P_ARMY_MARGIN: usize = 47;
pub const P_BELIEF_ESS: usize = 48;

/// Particle aggregates injected into the belief planes, in **raw army units**.
///
/// Raw, not normalized: `build_tensor` applies `army_value` itself, so the
/// summary and the visible-army planes go through the same compression.
#[derive(Clone)]
pub struct BeliefSummary {
    pub enemy_owner: Vec<f32>,
    pub enemy_army_mean: Vec<f32>,
    pub enemy_army_std: Vec<f32>,
    pub enemy_general: Vec<f32>,
    pub enemy_castle_owner: Vec<f32>,
    pub enemy_visibility: Vec<f32>,
    pub ess_fraction: f32,
}

impl BeliefSummary {
    pub fn zeros(h: usize, w: usize) -> Self {
        let z = vec![0.0f32; h * w];
        Self {
            enemy_owner: z.clone(),
            enemy_army_mean: z.clone(),
            enemy_army_std: z.clone(),
            enemy_general: z.clone(),
            enemy_castle_owner: z.clone(),
            enemy_visibility: z,
            ess_fraction: 0.0,
        }
    }
}

/// `clip(log1p(max(x, 0)) / log1p(scale), 0, 1)`, computed in double.
///
/// Armies span four orders of magnitude across a game, so the network sees
/// them log-compressed rather than raw.
#[inline]
pub fn army_value(x: f64, scale: f64) -> f32 {
    let denom = scale.ln_1p();
    let out = x.max(0.0).ln_1p() / denom;
    out.clamp(0.0, 1.0) as f32
}

/// Binary entropy in bits. Zero at both ends, where the log is undefined.
#[inline]
pub fn binary_entropy_bits(p: f32) -> f32 {
    let p = p as f64;
    if p <= 0.0 || p >= 1.0 {
        return 0.0;
    }
    let h = -p * p.ln() - (1.0 - p) * (1.0 - p).ln();
    (h / std::f64::consts::LN_2) as f32
}

fn own_general_rc(memory: &VisibleMemory) -> (usize, usize) {
    for r in 0..memory.h {
        for c in 0..memory.w {
            if memory.own_general[r * memory.w + c] {
                return (r, c);
            }
        }
    }
    (0, 0)
}

/// `(49, 21, 21)` as a flat `f32` buffer, plane-major.
///
/// `memory` must already include `obs` — call `update_memory` first, or use
/// `observation_tensor`.
pub fn build_tensor(
    obs: &Observation,
    memory: &VisibleMemory,
    belief: Option<&BeliefSummary>,
    previous_action: Option<[i32; 5]>,
    army_scale: f64,
) -> Vec<f32> {
    debug_assert_eq!(memory.h, obs.h);
    debug_assert_eq!(memory.w, obs.w);
    let (h, w) = (obs.h, obs.w);
    let owned_belief;
    let belief = match belief {
        Some(b) => b,
        None => {
            owned_belief = BeliefSummary::zeros(h, w);
            &owned_belief
        }
    };

    let mut tensor = vec![0.0f32; TENSOR_LEN];
    let turn = obs.turn;
    let (gr, gc) = own_general_rc(memory);

    // `set` writes into the padded plane; cells outside the live board stay
    // zero, which is what makes `board_mask` meaningful to the network.
    let mut set = |plane: usize, r: usize, c: usize, value: f32| {
        tensor[plane * PLANE_CELLS + r * PAD + c] = value;
    };

    for r in 0..h {
        for c in 0..w {
            let i = r * w + c;
            let t = obs.type_grid[i] as i32;
            let owner = obs.owner_grid[i] as i32;
            let army = obs.army_grid[i] as f64;
            let visible = t != TYPE_FOG && t != TYPE_STRUCTURE_FOG;

            set(P_BOARD_MASK, r, c, 1.0);
            set(P_VISIBLE_NOW, r, c, visible as i32 as f32);
            set(P_FOG_NONSTRUCTURE, r, c, (t == TYPE_FOG) as i32 as f32);
            set(P_FOG_STRUCTURE, r, c, (t == TYPE_STRUCTURE_FOG) as i32 as f32);
            set(P_KNOWN_MOUNTAIN, r, c, memory.known_mountain[i] as i32 as f32);
            set(P_KNOWN_PASSABLE, r, c, memory.known_passable_base[i] as i32 as f32);
            set(P_KNOWN_CASTLE, r, c, memory.known_castle[i] as i32 as f32);
            set(P_OWN_GENERAL, r, c, memory.own_general[i] as i32 as f32);
            set(
                P_KNOWN_ENEMY_GENERAL,
                r,
                c,
                memory.known_enemy_general[i] as i32 as f32,
            );

            let owned = owner == OWNER_ME;
            let enemy = owner == OWNER_ENEMY;
            set(P_OWNED_NOW, r, c, owned as i32 as f32);
            set(P_ENEMY_VISIBLE, r, c, enemy as i32 as f32);
            // Neutral only counts on a visible, non-mountain cell: an unseen
            // cell is unknown, not neutral.
            let passable_vis = visible && !memory.known_mountain[i];
            let neutral = owner == OWNER_NEUTRAL && passable_vis && t != TYPE_FOG;
            set(P_NEUTRAL_VISIBLE, r, c, neutral as i32 as f32);

            set(
                P_OWNED_ARMY,
                r,
                c,
                army_value(if owned { army } else { 0.0 }, army_scale),
            );
            set(
                P_ENEMY_ARMY_VISIBLE,
                r,
                c,
                army_value(if enemy { army } else { 0.0 }, army_scale),
            );

            set(P_EVER_VISIBLE, r, c, memory.ever_visible[i] as i32 as f32);
            // Sight age is 0 for a cell never seen — not "maximally stale",
            // which the `ever_visible` plane is there to distinguish.
            if memory.ever_visible[i] {
                let age = (turn - memory.last_seen_turn[i]) as f64 / TRUNCATION_TURN as f64;
                set(P_SIGHT_AGE, r, c, age.clamp(0.0, 1.0) as f32);
            }

            let rem_owner = memory.remembered_owner[i];
            let seen = memory.ever_visible[i];
            set(P_REMEMBERED_OWNED, r, c, (seen && rem_owner == OWNER_ME) as i32 as f32);
            set(P_REMEMBERED_ENEMY, r, c, (seen && rem_owner == OWNER_ENEMY) as i32 as f32);
            set(
                P_REMEMBERED_NEUTRAL,
                r,
                c,
                (seen && rem_owner == OWNER_NEUTRAL) as i32 as f32,
            );
            if seen && rem_owner == OWNER_ENEMY {
                set(
                    P_REMEMBERED_ENEMY_ARMY,
                    r,
                    c,
                    army_value(memory.remembered_army[i] as f64, army_scale),
                );
            }
            let remembered_castle = seen && memory.remembered_was_castle[i];
            set(
                P_REMEMBERED_OWN_CASTLE,
                r,
                c,
                (remembered_castle && memory.remembered_castle_owner[i] == OWNER_ME) as i32 as f32,
            );
            set(
                P_REMEMBERED_ENEMY_CASTLE,
                r,
                c,
                (remembered_castle && memory.remembered_castle_owner[i] == OWNER_ENEMY) as i32
                    as f32,
            );

            let p_enemy = belief.enemy_owner[i];
            set(P_BELIEF_ENEMY_OWNER, r, c, p_enemy);
            set(
                P_BELIEF_ENEMY_ARMY_MEAN,
                r,
                c,
                army_value(belief.enemy_army_mean[i] as f64, army_scale),
            );
            set(
                P_BELIEF_ENEMY_ARMY_STD,
                r,
                c,
                army_value(belief.enemy_army_std[i] as f64, army_scale),
            );
            set(P_BELIEF_ENEMY_GENERAL, r, c, belief.enemy_general[i]);
            set(P_BELIEF_ENEMY_CASTLE_OWNER, r, c, belief.enemy_castle_owner[i]);
            set(P_BELIEF_ENEMY_VISIBILITY, r, c, belief.enemy_visibility[i]);
            set(P_BELIEF_OWNER_ENTROPY, r, c, binary_entropy_bits(p_enemy));

            // Coordinate planes are float32 arithmetic in the Python
            // (`np.arange(..., dtype=np.float32) / 20.0`), so they are f32
            // here. Doing them in f64 would round differently in the last bit.
            set(P_ROW_COORD, r, c, r as f32 / 20.0);
            set(P_COL_COORD, r, c, c as f32 / 20.0);
            set(P_ROW_FROM_GENERAL, r, c, (r as f32 - gr as f32) / 20.0);
            set(P_COL_FROM_GENERAL, r, c, (c as f32 - gc as f32) / 20.0);
        }
    }

    // Scalars, painted across the live board only; padding stays zero.
    let my_land = obs.my_land as f64;
    let opp_land = obs.opp_land as f64;
    let my_army = obs.my_army as f64;
    let opp_army = obs.opp_army as f64;
    let constants: [(usize, f64); 12] = [
        (P_TURN_FRACTION, turn as f64 / TRUNCATION_TURN as f64),
        (
            P_PRE_DEATHTOUCH,
            (((DEATHTOUCH_TURN - turn) as f64) / DEATHTOUCH_TURN as f64).clamp(0.0, 1.0),
        ),
        (P_DEATHTOUCH_ACTIVE, if turn >= DEATHTOUCH_TURN { 1.0 } else { 0.0 }),
        (
            P_STRUCTURE_GROWTH_NEXT,
            if (turn + 1) % 2 == 0 { 1.0 } else { 0.0 },
        ),
        (
            P_BULK_GROWTH_COUNTDOWN,
            (((50 - ((turn + 1) % 50)) % 50) as f64) / 49.0,
        ),
        (P_OWN_LAND_FRACTION, my_land / 441.0),
        (P_ENEMY_LAND_FRACTION, opp_land / 441.0),
        (P_OWN_ARMY_TOTAL, army_value(my_army, army_scale) as f64),
        (P_ENEMY_ARMY_TOTAL, army_value(opp_army, army_scale) as f64),
        (P_LAND_MARGIN, (my_land - opp_land) / (my_land + opp_land + 1.0)),
        (P_ARMY_MARGIN, (my_army - opp_army) / (my_army + opp_army + 1.0)),
        (P_BELIEF_ESS, belief.ess_fraction as f64),
    ];
    for (plane, value) in constants {
        let v = value as f32;
        for r in 0..h {
            for c in 0..w {
                tensor[plane * PLANE_CELLS + r * PAD + c] = v;
            }
        }
    }

    paint_previous_action(&mut tensor, previous_action, h, w);
    tensor
}

/// Paint the perspective player's own last action into planes 29–32.
///
/// Only the seat's own action: the opponent's is not observable, and a plane
/// that pretended otherwise would train the network on information it will not
/// have in play.
fn paint_previous_action(tensor: &mut [f32], action: Option<[i32; 5]>, h: usize, w: usize) {
    let Some([pass_f, row, col, direction, split]) = action else {
        return;
    };
    if pass_f == 1 {
        return;
    }
    let in_board = |r: i32, c: i32| r >= 0 && c >= 0 && (r as usize) < h && (c as usize) < w;
    if !in_board(row, col) {
        return;
    }
    let (r, c) = (row as usize, col as usize);
    if pass_f == 2 {
        tensor[P_PREV_BUILD_CELL * PLANE_CELLS + r * PAD + c] = 1.0;
        return;
    }
    tensor[P_PREV_MOVE_SOURCE * PLANE_CELLS + r * PAD + c] = 1.0;
    let (dr, dc) = [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)][direction.clamp(0, 3) as usize];
    let (dest_r, dest_c) = (row + dr, col + dc);
    if in_board(dest_r, dest_c) {
        let (dr_u, dc_u) = (dest_r as usize, dest_c as usize);
        tensor[P_PREV_MOVE_DEST * PLANE_CELLS + dr_u * PAD + dc_u] = 1.0;
        // Half-splits are marked at 0.5 so one plane carries both the
        // destination and which kind of move reached it.
        tensor[P_PREV_MOVE_KIND * PLANE_CELLS + dr_u * PAD + dc_u] =
            if split == 1 { 0.5 } else { 1.0 };
    }
}

/// Update memory from `obs`, then build the tensor.
pub fn observation_tensor(
    obs: &Observation,
    memory: &VisibleMemory,
    belief: Option<&BeliefSummary>,
    previous_action: Option<[i32; 5]>,
    army_scale: f64,
) -> (Vec<f32>, VisibleMemory) {
    let next = crate::memory::update_memory(memory, obs);
    let tensor = build_tensor(obs, &next, belief, previous_action, army_scale);
    (tensor, next)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(h: usize, w: usize) -> Observation {
        let mut o = Observation::with_dims(h, w);
        o.turn = 10;
        for i in 0..h * w {
            o.type_grid[i] = 1; // plain
            o.owner_grid[i] = 0;
        }
        o
    }

    fn at(tensor: &[f32], plane: usize, r: usize, c: usize) -> f32 {
        tensor[plane * PLANE_CELLS + r * PAD + c]
    }

    #[test]
    fn the_tensor_is_always_padded_to_the_full_square() {
        let o = obs(3, 4);
        let m = VisibleMemory::empty(3, 4);
        let t = build_tensor(&o, &m, None, None, ARMY_SCALE);
        assert_eq!(t.len(), TENSOR_LEN);
        assert_eq!(at(&t, P_BOARD_MASK, 2, 3), 1.0);
        assert_eq!(at(&t, P_BOARD_MASK, 3, 0), 0.0, "outside the live board");
        assert_eq!(at(&t, P_BOARD_MASK, 0, 4), 0.0);
    }

    #[test]
    fn army_compression_is_monotone_and_bounded() {
        assert_eq!(army_value(0.0, ARMY_SCALE), 0.0);
        assert_eq!(army_value(-5.0, ARMY_SCALE), 0.0, "negatives clamp to zero");
        assert!((army_value(ARMY_SCALE, ARMY_SCALE) - 1.0).abs() < 1e-6);
        assert_eq!(army_value(1e9, ARMY_SCALE), 1.0, "and saturate at one");
        assert!(army_value(10.0, ARMY_SCALE) < army_value(100.0, ARMY_SCALE));
    }

    #[test]
    fn entropy_is_zero_at_the_ends_and_one_bit_in_the_middle() {
        assert_eq!(binary_entropy_bits(0.0), 0.0);
        assert_eq!(binary_entropy_bits(1.0), 0.0);
        assert!((binary_entropy_bits(0.5) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn constant_planes_stop_at_the_board_edge() {
        let mut o = obs(2, 2);
        o.my_land = 100;
        o.opp_land = 50;
        let m = VisibleMemory::empty(2, 2);
        let t = build_tensor(&o, &m, None, None, ARMY_SCALE);
        let want = (100.0f64 / 441.0) as f32;
        assert_eq!(at(&t, P_OWN_LAND_FRACTION, 1, 1), want);
        assert_eq!(at(&t, P_OWN_LAND_FRACTION, 2, 0), 0.0, "leaked into padding");
    }

    #[test]
    fn a_previous_move_paints_source_destination_and_kind() {
        let o = obs(4, 4);
        let m = VisibleMemory::empty(4, 4);
        let full = build_tensor(&o, &m, None, Some([0, 1, 1, 3, 0]), ARMY_SCALE);
        assert_eq!(at(&full, P_PREV_MOVE_SOURCE, 1, 1), 1.0);
        assert_eq!(at(&full, P_PREV_MOVE_DEST, 1, 2), 1.0);
        assert_eq!(at(&full, P_PREV_MOVE_KIND, 1, 2), 1.0);

        let half = build_tensor(&o, &m, None, Some([0, 1, 1, 3, 1]), ARMY_SCALE);
        assert_eq!(at(&half, P_PREV_MOVE_KIND, 1, 2), 0.5, "a split reads as a half");
    }

    #[test]
    fn a_previous_pass_paints_nothing() {
        let o = obs(4, 4);
        let m = VisibleMemory::empty(4, 4);
        let t = build_tensor(&o, &m, None, Some([1, 0, 0, 0, 0]), ARMY_SCALE);
        for plane in [P_PREV_MOVE_SOURCE, P_PREV_MOVE_DEST, P_PREV_MOVE_KIND, P_PREV_BUILD_CELL] {
            let sum: f32 = t[plane * PLANE_CELLS..(plane + 1) * PLANE_CELLS].iter().sum();
            assert_eq!(sum, 0.0, "plane {plane}");
        }
    }

    #[test]
    fn a_move_off_the_board_edge_paints_a_source_but_no_destination() {
        let o = obs(3, 3);
        let m = VisibleMemory::empty(3, 3);
        let t = build_tensor(&o, &m, None, Some([0, 0, 0, 0, 0]), ARMY_SCALE);
        assert_eq!(at(&t, P_PREV_MOVE_SOURCE, 0, 0), 1.0);
        let dest: f32 = t[P_PREV_MOVE_DEST * PLANE_CELLS..(P_PREV_MOVE_DEST + 1) * PLANE_CELLS]
            .iter()
            .sum();
        assert_eq!(dest, 0.0);
    }

    #[test]
    fn sight_age_is_zero_for_a_cell_never_seen() {
        let mut o = obs(1, 2);
        o.turn = 500;
        let mut m = VisibleMemory::empty(1, 2);
        m.ever_visible[0] = true;
        m.last_seen_turn[0] = 200;
        let t = build_tensor(&o, &m, None, None, ARMY_SCALE);
        assert!((at(&t, P_SIGHT_AGE, 0, 0) - (300.0 / 1200.0)).abs() < 1e-7);
        assert_eq!(at(&t, P_SIGHT_AGE, 0, 1), 0.0);
        assert_eq!(at(&t, P_EVER_VISIBLE, 0, 1), 0.0, "and it says so separately");
    }

    #[test]
    fn coordinates_are_measured_from_the_own_general() {
        let o = obs(5, 5);
        let mut m = VisibleMemory::empty(5, 5);
        m.own_general[2 * 5 + 3] = true;
        let t = build_tensor(&o, &m, None, None, ARMY_SCALE);
        assert_eq!(at(&t, P_ROW_FROM_GENERAL, 2, 3), 0.0);
        assert_eq!(at(&t, P_COL_FROM_GENERAL, 2, 3), 0.0);
        assert_eq!(at(&t, P_ROW_FROM_GENERAL, 4, 3), 2.0 / 20.0);
        assert_eq!(at(&t, P_COL_FROM_GENERAL, 2, 0), -3.0 / 20.0);
    }
}
