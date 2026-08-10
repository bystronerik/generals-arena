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

use std::sync::OnceLock;

use crate::board::action::{decode_action, encode_action, legal_mask, live_build_cost, PASS_INDEX};
use crate::belief::{Action5, BeliefState};
use crate::board::memory::{
    VisibleMemory, OWNER_ENEMY, OWNER_NEUTRAL, TYPE_CASTLE, TYPE_FOG, TYPE_GENERAL, TYPE_MOUNTAIN,
    TYPE_STRUCTURE_FOG,
};
use crate::board::observe::visibility_from_owned;
use crate::support::rng::npsum;
use crate::board::state::MAX_CELLS;
use crate::board::transition::{BASE_COST, DEATHTOUCH_TURN, DIRECTIONS};
use crate::io::wire::Observation;

pub const N_ACTIONS: usize = PASS_INDEX + 1;

/// Fog-hunt urgency keeps rising until first enemy sight.
pub const FOG_URGENCY_TURN_SCALE: f64 = 40.0;
/// Tip armies below this are slow explorers — prefer a formed wave.
pub const EXPLORE_WAVE_MIN: i64 = 3;
/// Reverse along any of the last this-many army moves is banned.
pub const OSCILLATION_HISTORY: usize = 8;
/// Commitment hysteresis. Measured without it, the chosen source tile jumped
/// ≥3 Manhattan on 27% of consecutive move turns — plans died to tie-break
/// jitter. It only amplifies already-positive scores.
pub const CONTINUATION_BONUS: f64 = 1.5;
/// Once the enemy general is latched, enemy takes that do not shorten the path
/// to it are farming. Measured: a 1200-turn draw in which the border was chewed
/// for 850 turns while the general sat at 1–9 army, unseen.
pub const GENERAL_CHEW_DAMP: f64 = 0.3;
/// A bonus per step of progress toward the known/believed general, never a goal
/// swap: replacing the enemy-land goal set with the believed cell lost the own
/// general, because incursions near home stopped counting as progress.
pub const HUNT_PROGRESS_BONUS: f64 = 0.75;

/// Garrison floor. The floor is fog-proof: from `GARRISON_FLOOR_FROM` until
/// deathtouch a move may leave the general only if what stays behind is at or
/// above the floor. Kept small deliberately — the first cut (min 12 / 6% / cap
/// 40) hoarded, and all king/gather logic excludes the floored general so army
/// assembles forward.
pub const GARRISON_FLOOR_FROM: i32 = 100;
pub const GARRISON_FLOOR_MIN: i64 = 10;
pub const GARRISON_FLOOR_FRAC: f64 = 0.04;
pub const GARRISON_FLOOR_CAP: i64 = 18;
/// Release is a hard rule, not a score: the trained policy assigns almost no
/// mass to splits and the bounded blend can lift an action at most 10×, so the
/// legal half-release topped the heuristic ranking yet never got played.
pub const GARRISON_RELEASE_FACTOR: f64 = 2.0;

/// Castle economics. Payback is 2× price in turns. Full-price builds are never
/// rewarded: the site must cost exactly `BASE_COST`.
pub const CASTLE_TARGET: i64 = 2;
pub const CASTLE_WINDOW_UNTIL: i32 = 500;
pub const CASTLE_SAFE_ENEMY_DIST: i32 = 4;
pub const CASTLE_CATCHMENT: i32 = 6;
/// One action per turn means rear logistics can never win the global argmax, so
/// every Nth turn `constrain` dedicates the turn to one gather step. Raised
/// from 3 after the tax showed up as a mid-game land collapse.
pub const CASTLE_TITHE_PERIOD: i32 = 5;

/// Aggression bases, tuned against opponents that out-tempo a defensive bot.
pub const ENEMY_TAKE_BASE: f64 = 160.0;
pub const NEUTRAL_CARVE_BASE: f64 = 45.0;
pub const PRE_REVEAL_WEIGHT: f64 = 20.0;
pub const PRE_EFFICIENCY_WEIGHT: f64 = 10.0;
pub const PRE_PROGRESS_WEIGHT: f64 = 4.0;
/// Post-deathtouch any executed touch wins outright, so it outranks everything.
pub const DEATHTOUCH_SCORE: f64 = 1.0e6;

/// Prior-shaping blend: `lam` is the trust knob, `log_clip` bounds how far one
/// heuristic may move an action, `floor_frac` keeps a network zero from being
/// resurrected.
pub const DEFAULT_SHAPING_LAMBDA: f64 = 1.0;
pub const DEFAULT_SHAPING_FLOOR_FRAC: f64 = 1e-3;
/// `ln 10` — at most a 10× nudge either way.
pub fn default_shaping_log_clip() -> f64 {
    10.0f64.ln()
}

pub const WAVE_ARMY_SOFT_CAP: i64 = 60;
/// Own-land pile merges at or above this are stacking waste.
pub const STACK_GATHER_BAN: i64 = 16;
/// Large stacks may march onto thin own cells toward the enemy.
pub const COMMIT_DEST_ARMY_MAX: f64 = 8.0;
pub const ATTACK_ARMY_CAP: i64 = 200;
pub const CASTLE_EARLY_UNTIL: i32 = 200;
/// Pre-contact, armies at or above this on a structure should leave, not idle.
pub const STRUCTURE_IDLE_ARMY: i64 = 18;
/// Below this share of total army in the king stack, sweep land into it.
pub const GATHER_SHARE_MIN: f64 = 0.5;
pub const COMMIT_ARMY_FRAC: f64 = 0.35;
pub const COMMIT_TOTAL_FRAC: f64 = 0.15;

/// Emergency defense. Detection reaches `DEFENSE_RADIUS`; the forced
/// reinforcement fires only when arrival is imminent, so a wave loitering at
/// the detection edge does not divert the army every turn.
pub const DEFENSE_RADIUS: i32 = 4;
pub const DEFENSE_FORCE_WITHIN: i32 = 3;

/// Max length of a forced finishing march on a visible enemy general.
pub const KILL_HORIZON: i32 = 6;

pub type Cell = (usize, usize);

// --------------------------------------------------------------- decode tables

/// Static `(kind, sr, sc, tr, tc)` per non-pass action index.
///
/// `tr`/`tc` are move destinations, unclipped: they may fall outside a board
/// smaller than the padded layout, so callers bounds-check against the live
/// `H`/`W`. For builds the destination equals the source.
pub struct DecodeTables {
    pub kind: [i8; PASS_INDEX],
    pub sr: [i16; PASS_INDEX],
    pub sc: [i16; PASS_INDEX],
    pub tr: [i16; PASS_INDEX],
    pub tc: [i16; PASS_INDEX],
}

static DECODE_TABLES: OnceLock<DecodeTables> = OnceLock::new();

pub fn decode_tables() -> &'static DecodeTables {
    DECODE_TABLES.get_or_init(|| {
        let mut tables = DecodeTables {
            kind: [0; PASS_INDEX],
            sr: [0; PASS_INDEX],
            sc: [0; PASS_INDEX],
            tr: [0; PASS_INDEX],
            tc: [0; PASS_INDEX],
        };
        for index in 0..PASS_INDEX {
            let action = decode_action(index).expect("every non-pass index decodes");
            let (kind, r, c, d) = (action[0], action[1], action[2], action[3]);
            tables.kind[index] = kind as i8;
            tables.sr[index] = r as i16;
            tables.sc[index] = c as i16;
            if kind == 0 {
                tables.tr[index] = (r + DIRECTIONS[d as usize].0) as i16;
                tables.tc[index] = (c + DIRECTIONS[d as usize].1) as i16;
            } else {
                tables.tr[index] = r as i16;
                tables.tc[index] = c as i16;
            }
        }
        tables
    })
}

// ------------------------------------------------------------- grid accessors

/// Read-only view of the three observation grids at the live stride.
#[derive(Clone, Copy)]
pub struct Grids<'a> {
    pub obs: &'a Observation,
    pub h: i32,
    pub w: i32,
}

impl<'a> Grids<'a> {
    pub fn new(obs: &'a Observation) -> Self {
        Self {
            obs,
            h: obs.h as i32,
            w: obs.w as i32,
        }
    }

    #[inline]
    pub fn inside(&self, r: i32, c: i32) -> bool {
        r >= 0 && c >= 0 && r < self.h && c < self.w
    }

    #[inline]
    pub fn at(&self, r: i32, c: i32) -> usize {
        (r * self.w + c) as usize
    }

    #[inline]
    pub fn kind(&self, r: i32, c: i32) -> i32 {
        self.obs.type_grid[self.at(r, c)] as i32
    }

    #[inline]
    pub fn owner(&self, r: i32, c: i32) -> i32 {
        self.obs.owner_grid[self.at(r, c)] as i32
    }

    #[inline]
    pub fn army(&self, r: i32, c: i32) -> i64 {
        self.obs.army_grid[self.at(r, c)] as i64
    }
}

fn turn_of(obs: &Observation) -> i32 {
    obs.turn
}

// ---------------------------------------------------------------- move helpers

pub type MoveSegment = ((i32, i32), (i32, i32));

/// `(sr, sc, tr, tc)` for a move action, else `None`.
pub fn move_dest(action: Action5) -> Option<(i32, i32, i32, i32)> {
    if action[0] != 0 {
        return None;
    }
    let (sr, sc, d) = (action[1], action[2], action[3]);
    if !(0..4).contains(&d) {
        return None;
    }
    Some((
        sr,
        sc,
        sr + DIRECTIONS[d as usize].0,
        sc + DIRECTIONS[d as usize].1,
    ))
}

pub fn move_segment(action: Action5) -> Option<MoveSegment> {
    move_dest(action).map(|(sr, sc, tr, tc)| ((sr, sc), (tr, tc)))
}

pub fn is_reverse_segment(a: MoveSegment, b: MoveSegment) -> bool {
    a.0 == b.1 && a.1 == b.0
}

pub fn is_reverse_move(action: Action5, prev: Option<Action5>) -> bool {
    let prev = match prev {
        Some(prev) => prev,
        None => return false,
    };
    match (move_segment(action), move_segment(prev)) {
        (Some(cur), Some(old)) => is_reverse_segment(cur, old),
        _ => false,
    }
}

fn oscillation_history(
    prev_action: Option<Action5>,
    recent_actions: &[Action5],
) -> Vec<MoveSegment> {
    let source: Vec<Action5> = if !recent_actions.is_empty() {
        recent_actions.to_vec()
    } else if let Some(prev) = prev_action {
        vec![prev]
    } else {
        Vec::new()
    };
    let out: Vec<MoveSegment> = source.iter().filter_map(|&a| move_segment(a)).collect();
    if out.len() > OSCILLATION_HISTORY {
        out[out.len() - OSCILLATION_HISTORY..].to_vec()
    } else {
        out
    }
}

/// True when the move reverses a recent own-corridor edge.
///
/// The whole window, not only the previous step, so `A→B→C` then `C→B→A`
/// ping-pong is banned. A reverse onto enemy land, or onto a cell that unlocks
/// new vision, stays legal.
///
/// `reveal` is an optional precomputed [`reveal_count_grid`]. The Python calls
/// `newly_revealed_cells` here, which runs two whole-board dilations *per
/// candidate action*; the grid is the same quantity computed once, which is
/// what makes the redirect loop affordable at all. The two agree by
/// construction — see [`reveal_count_grid`] — and the `oscillation` parity
/// surface runs both paths on every case so the equivalence is checked rather
/// than asserted.
pub fn blocks_oscillation(
    action: Action5,
    prev_action: Option<Action5>,
    obs: &Observation,
    recent_actions: &[Action5],
    force_allow: bool,
    reveal: Option<&[i64]>,
) -> bool {
    if force_allow {
        return false;
    }
    let seg = match move_segment(action) {
        Some(seg) => seg,
        None => return false,
    };
    let g = Grids::new(obs);
    let (tr, tc) = seg.1;
    if !g.inside(tr, tc) {
        return true;
    }
    if g.owner(tr, tc) == OWNER_ENEMY {
        return false;
    }
    let revealed = match reveal {
        Some(grid) => grid[g.at(tr, tc)],
        None => newly_revealed_cells(obs, tr, tc),
    };
    if revealed > 0 {
        return false;
    }
    oscillation_history(prev_action, recent_actions)
        .into_iter()
        .any(|old| is_reverse_segment(seg, old))
}

/// How many currently invisible cells become visible if we own `dest`.
pub fn newly_revealed_cells(obs: &Observation, dest_r: i32, dest_c: i32) -> i64 {
    let g = Grids::new(obs);
    let n = obs.h * obs.w;
    let mut owned = vec![false; n];
    for i in 0..n {
        owned[i] = obs.owner_grid[i] as i32 == 1;
    }
    if owned[g.at(dest_r, dest_c)] {
        return 0;
    }
    let before = visibility_from_owned(&owned, obs.h, obs.w);
    owned[g.at(dest_r, dest_c)] = true;
    let after = visibility_from_owned(&owned, obs.h, obs.w);
    (0..n).filter(|&i| after[i] && !before[i]).count() as i64
}

/// Per-cell `newly_revealed_cells` for the whole board at once.
///
/// Visibility is a 3×3 dilation of ownership, so owning one new cell reveals
/// exactly the currently-invisible cells inside that cell's 3×3 box. Cells we
/// already own reveal nothing.
pub fn reveal_count_grid(obs: &Observation) -> Vec<i64> {
    let (h, w) = (obs.h, obs.w);
    let n = h * w;
    let mut owned = vec![false; n];
    for i in 0..n {
        owned[i] = obs.owner_grid[i] as i32 == 1;
    }
    let before = visibility_from_owned(&owned, h, w);
    let mut out = vec![0i64; n];
    for r in 0..h as i32 {
        for c in 0..w as i32 {
            let at = (r * w as i32 + c) as usize;
            if owned[at] {
                continue;
            }
            let mut box_count = 0i64;
            for dr in -1..=1i32 {
                for dc in -1..=1i32 {
                    let (nr, nc) = (r + dr, c + dc);
                    if nr < 0 || nc < 0 || nr >= h as i32 || nc >= w as i32 {
                        continue;
                    }
                    if !before[(nr * w as i32 + nc) as usize] {
                        box_count += 1;
                    }
                }
            }
            out[at] = box_count;
        }
    }
    out
}

// ----------------------------------------------------------------- weightings

pub fn castle_timing_weight(turn: i32) -> f64 {
    let t = turn.max(0);
    if t <= CASTLE_EARLY_UNTIL {
        return 2.3 - 0.9 * (t as f64 / CASTLE_EARLY_UNTIL as f64);
    }
    let over = (t - CASTLE_EARLY_UNTIL) as f64;
    (1.2 - over / 280.0).max(0.2)
}

pub fn explore_wave_weight(army: i64) -> f64 {
    let a = army.max(1);
    if a < EXPLORE_WAVE_MIN {
        return 0.12;
    }
    1.0 + 0.9 * (a.min(80) as f64).ln_1p()
}

pub fn fog_urgency(turn: i32, enemy_seen: bool) -> f64 {
    if enemy_seen {
        return 1.0;
    }
    let t = turn.max(0) as f64;
    1.0 + (t / FOG_URGENCY_TURN_SCALE).powf(1.15)
}

pub fn wave_weight(army: i64) -> f64 {
    let a = army.min(WAVE_ARMY_SOFT_CAP).max(1);
    1.0 + 0.25 * (a as f64).ln_1p()
}

pub fn attack_weight(army: i64) -> f64 {
    let a = army.min(ATTACK_ARMY_CAP).max(1);
    1.0 + 0.85 * (a as f64).ln_1p()
}

/// True when this stack may freestyle; else it must gather.
///
/// When the board is dispersed, `max_army` alone is a bad signal: a tip of 11
/// against a total of 200 looks committed at `0.35 * max`.
pub fn is_committed_army(army: i64, max_army: i64, total: i64, share: f64) -> bool {
    let tot = total.max(1);
    let mx = max_army.max(1);
    if share < GATHER_SHARE_MIN {
        let floor = STACK_GATHER_BAN.max((COMMIT_TOTAL_FRAC * tot as f64) as i64);
        return army >= mx && army >= floor;
    }
    army as f64 >= COMMIT_ARMY_FRAC * mx as f64 || army as f64 >= COMMIT_TOTAL_FRAC * tot as f64
}

/// Downweight tiny tip attacks while a much larger stack sits idle.
pub fn tip_thrash_factor(src_army: i64, max_army: i64, total: i64) -> f64 {
    let src = src_army.max(1);
    let big = max_army.max(1);
    let tot = total.max(1);
    let share = big as f64 / tot as f64;
    if big < STACK_GATHER_BAN && share >= GATHER_SHARE_MIN {
        return 1.0;
    }
    if is_committed_army(src, big, tot, share) {
        return 1.0;
    }
    0.08 + 0.6 * (src as f64 / big as f64)
}

/// Own-land friction: near-free forward merges, brutal on retreat dumps.
pub fn stack_gather_factor(src_army: i64, dest_army: i64, progress: f64) -> f64 {
    let src = src_army.max(1);
    let dest = dest_army.max(0) as f64;
    if progress > 0.0 {
        let x = dest / 80.0;
        return 1.0 / (1.0 + x * x);
    }
    if progress < 0.0 {
        return 0.03 / (1.0 + dest / 4.0);
    }
    let x = dest / 18.0;
    let feed = 1.0 / (1.0 + x * x);
    if src >= STACK_GATHER_BAN && dest > COMMIT_DEST_ARMY_MAX {
        feed * 0.25
    } else {
        feed
    }
}

/// Reward steps toward the seek target; punish retreats; mild on lateral.
pub fn direction_bias(progress: f64, dest_owner: i32) -> f64 {
    if dest_owner == OWNER_ENEMY {
        return 4.0 + 2.0 * progress.max(0.0);
    }
    if progress > 0.0 {
        return if dest_owner == 1 {
            1.5 + 1.2 * progress
        } else {
            1.8 + 1.4 * progress
        };
    }
    if progress < 0.0 {
        return 0.06;
    }
    if dest_owner == 1 {
        0.25
    } else {
        0.55
    }
}

/// Manhattan progress toward `target` (+1 closer, −1 farther, 0 lateral).
pub fn move_progress(sr: i32, sc: i32, tr: i32, tc: i32, target: Option<Cell>) -> f64 {
    let target = match target {
        Some(target) => target,
        None => return 0.0,
    };
    let before = (sr - target.0 as i32).abs() + (sc - target.1 as i32).abs();
    let after = (tr - target.0 as i32).abs() + (tc - target.1 as i32).abs();
    (before - after) as f64
}

/// Path-aware progress: the drop in BFS distance to the seek goals.
///
/// Falls back to Manhattan when the field is missing or both ends are
/// unreachable, so fog carving that opens a route still gets a signal.
pub fn path_progress(
    sr: i32,
    sc: i32,
    tr: i32,
    tc: i32,
    dist_field: Option<&DistanceField>,
    fallback_target: Option<Cell>,
) -> f64 {
    let field = match dist_field {
        Some(field) => field,
        None => return move_progress(sr, sc, tr, tc, fallback_target),
    };
    if !field.inside(sr, sc) || !field.inside(tr, tc) {
        return move_progress(sr, sc, tr, tc, fallback_target);
    }
    let before = field.get(sr, sc);
    let after = field.get(tr, tc);
    if before < 0 && after < 0 {
        return move_progress(sr, sc, tr, tc, fallback_target);
    }
    if before < 0 {
        return 2.0;
    }
    if after < 0 {
        return -2.0;
    }
    (before - after) as f64
}

// ------------------------------------------------------------- path distances

/// BFS distance to the nearest goal through passable cells; `-1` unreachable.
pub struct DistanceField {
    pub h: i32,
    pub w: i32,
    pub dist: Vec<i32>,
}

impl DistanceField {
    #[inline]
    pub fn inside(&self, r: i32, c: i32) -> bool {
        r >= 0 && c >= 0 && r < self.h && c < self.w
    }

    #[inline]
    pub fn get(&self, r: i32, c: i32) -> i32 {
        self.dist[(r * self.w + c) as usize]
    }
}

pub fn path_distance_field(obs: &Observation, goals: &[Cell]) -> DistanceField {
    let (h, w) = (obs.h as i32, obs.w as i32);
    let n = obs.h * obs.w;
    let mut passable = vec![false; n];
    for i in 0..n {
        let t = obs.type_grid[i] as i32;
        passable[i] = t != TYPE_MOUNTAIN && t != TYPE_STRUCTURE_FOG;
    }
    let mut dist = vec![-1i32; n];
    let mut queue: Vec<(i32, i32)> = Vec::with_capacity(n);
    for &(gr, gc) in goals {
        let (gr, gc) = (gr as i32, gc as i32);
        if gr < 0 || gc < 0 || gr >= h || gc >= w {
            continue;
        }
        let at = (gr * w + gc) as usize;
        if !passable[at] || dist[at] == 0 {
            continue;
        }
        dist[at] = 0;
        queue.push((gr, gc));
    }
    let mut head = 0usize;
    while head < queue.len() {
        let (r, c) = queue[head];
        head += 1;
        let base = dist[(r * w + c) as usize];
        for (dr, dc) in DIRECTIONS {
            let (nr, nc) = (r + dr, c + dc);
            if nr < 0 || nc < 0 || nr >= h || nc >= w {
                continue;
            }
            let at = (nr * w + nc) as usize;
            if !passable[at] || dist[at] >= 0 {
                continue;
            }
            dist[at] = base + 1;
            queue.push((nr, nc));
        }
    }
    DistanceField { h, w, dist }
}

// ------------------------------------------------------------- board queries

pub fn enemy_is_visible(obs: &Observation, memory: &VisibleMemory) -> bool {
    let n = obs.h * obs.w;
    if (0..n).any(|i| obs.owner_grid[i] as i32 == OWNER_ENEMY) {
        return true;
    }
    (0..n).any(|i| memory.known_enemy_general[i])
}

pub fn enemy_general_visible(obs: &Observation, memory: &VisibleMemory) -> bool {
    let n = obs.h * obs.w;
    if (0..n)
        .any(|i| obs.type_grid[i] as i32 == TYPE_GENERAL && obs.owner_grid[i] as i32 == OWNER_ENEMY)
    {
        return true;
    }
    (0..n).any(|i| memory.known_enemy_general[i])
}

/// Own general and own castles, latched or currently visible.
pub fn own_structure_mask(obs: &Observation, memory: &VisibleMemory) -> Vec<bool> {
    let n = obs.h * obs.w;
    let mut out = vec![false; n];
    for i in 0..n {
        let own = obs.owner_grid[i] as i32 == 1;
        let t = obs.type_grid[i] as i32;
        let gen = memory.own_general[i] || (t == TYPE_GENERAL && own);
        let castle = memory.known_castle[i] || t == TYPE_CASTLE;
        out[i] = gen || (castle && own);
    }
    out
}

/// Largest army sitting on an own general or castle.
pub fn structure_idle_army(obs: &Observation, memory: &VisibleMemory) -> i64 {
    let structures = own_structure_mask(obs, memory);
    let n = obs.h * obs.w;
    let mut best: Option<i64> = None;
    for i in 0..n {
        if structures[i] {
            let a = obs.army_grid[i] as i64;
            best = Some(match best {
                Some(current) => current.max(a),
                None => a,
            });
        }
    }
    best.unwrap_or(0)
}

/// Latched, else visible, own general cell.
pub fn own_general_cell(obs: &Observation, memory: &VisibleMemory) -> Option<Cell> {
    let n = obs.h * obs.w;
    for i in 0..n {
        if memory.own_general[i] {
            return Some((i / obs.w, i % obs.w));
        }
    }
    for i in 0..n {
        if obs.type_grid[i] as i32 == TYPE_GENERAL && obs.owner_grid[i] as i32 == 1 {
            return Some((i / obs.w, i % obs.w));
        }
    }
    None
}

/// Latched, else currently visible, enemy general cell.
pub fn known_enemy_general_cell(obs: &Observation, memory: &VisibleMemory) -> Option<Cell> {
    let n = obs.h * obs.w;
    for i in 0..n {
        if memory.known_enemy_general[i] {
            return Some((i / obs.w, i % obs.w));
        }
    }
    for i in 0..n {
        if obs.type_grid[i] as i32 == TYPE_GENERAL && obs.owner_grid[i] as i32 == OWNER_ENEMY {
            return Some((i / obs.w, i % obs.w));
        }
    }
    None
}

/// Cell holding the largest own army; ties by row-major order.
///
/// `exclude` drops one cell (the floored general) so gather targets a stack
/// that can actually march, and falls back to the whole board when there is no
/// other own cell.
pub fn king_cell(obs: &Observation, exclude: Option<Cell>) -> Option<Cell> {
    let n = obs.h * obs.w;
    let mut own: Vec<bool> = (0..n).map(|i| obs.owner_grid[i] as i32 == 1).collect();
    if let Some(cell) = exclude {
        let at = cell.0 * obs.w + cell.1;
        if own[at] && own.iter().filter(|&&v| v).count() > 1 {
            own[at] = false;
        }
    }
    let max_a = (0..n)
        .filter(|&i| own[i])
        .map(|i| obs.army_grid[i] as i64)
        .max()?;
    (0..n)
        .find(|&i| own[i] && obs.army_grid[i] as i64 == max_a)
        .map(|i| (i / obs.w, i % obs.w))
}

/// `(max_share, max_army, total_army)` on own land.
///
/// `max_army` ignores `exclude` so the commitment and thrash logic measures
/// against the largest *movable* stack; `total_army` stays the full total.
pub fn army_concentration(obs: &Observation, exclude: Option<Cell>) -> (f64, i64, i64) {
    let n = obs.h * obs.w;
    let own: Vec<bool> = (0..n).map(|i| obs.owner_grid[i] as i32 == 1).collect();
    if !own.iter().any(|&v| v) {
        return (0.0, 0, 0);
    }
    let total: i64 = (0..n)
        .filter(|&i| own[i])
        .map(|i| obs.army_grid[i] as i64)
        .sum();
    let mut movable = own.clone();
    if let Some(cell) = exclude {
        let at = cell.0 * obs.w + cell.1;
        if movable[at] && movable.iter().filter(|&&v| v).count() > 1 {
            movable[at] = false;
        }
    }
    let max_a = (0..n)
        .filter(|&i| movable[i])
        .map(|i| obs.army_grid[i] as i64)
        .max()
        .unwrap_or(0);
    if total <= 0 {
        return (0.0, max_a, 0);
    }
    (max_a as f64 / total as f64, max_a, total)
}

/// The general cell while the garrison floor pins it, else `None`.
fn movable_exclude_cell(obs: &Observation, memory: &VisibleMemory) -> Option<Cell> {
    let turn = turn_of(obs);
    if !(GARRISON_FLOOR_FROM..DEATHTOUCH_TURN).contains(&turn) {
        return None;
    }
    own_general_cell(obs, memory)
}

// ------------------------------------------------------------- seek targeting

/// Mode of the particle posterior over the enemy general's cell.
///
/// Weighted, with the particle count breaking ties when every weight is zero,
/// and first-seen order breaking the rest — Python's `max` keeps the first
/// maximum and its `dict` preserves insertion order, so the tie rule is the
/// order the particles introduced each cell.
pub fn believed_enemy_general(belief: Option<&BeliefState>) -> Option<Cell> {
    let belief = belief?;
    if belief.n() == 0 {
        return None;
    }
    let enemy = belief.enemy_seat();
    let mut mass: Vec<(Cell, f64, usize)> = Vec::new();
    for particle in &belief.particles {
        let g = particle.state.general_positions[enemy];
        if g[0] < 0 || g[1] < 0 {
            continue;
        }
        let cell = (g[0] as usize, g[1] as usize);
        match mass.iter_mut().find(|entry| entry.0 == cell) {
            Some(entry) => {
                entry.1 += particle.weight.max(0.0);
                entry.2 += 1;
            }
            None => mass.push((cell, particle.weight.max(0.0), 1)),
        }
    }
    let mut best: Option<&(Cell, f64, usize)> = None;
    for entry in &mass {
        let better = match best {
            None => true,
            Some(current) => (entry.1, entry.2) > (current.1, current.2),
        };
        if better {
            best = Some(entry);
        }
    }
    best.map(|entry| entry.0)
}

/// Cells to drive toward: enemy general, else enemy land, else the posterior.
///
/// Enemy land stays ahead of the belief on purpose — it is what keeps
/// incursions near home scored as progress, and the border under broad
/// pressure. The directed hunt is a separate bonus field in
/// [`heuristic_action_scores`], not a goal replacement.
pub fn seek_goals(
    obs: &Observation,
    memory: &VisibleMemory,
    belief: Option<&BeliefState>,
) -> Vec<Cell> {
    let n = obs.h * obs.w;
    let rc = |i: usize| (i / obs.w, i % obs.w);

    if let Some(i) = (0..n).find(|&i| memory.known_enemy_general[i]) {
        return vec![rc(i)];
    }
    if let Some(i) = (0..n)
        .find(|&i| obs.type_grid[i] as i32 == TYPE_GENERAL && obs.owner_grid[i] as i32 == OWNER_ENEMY)
    {
        return vec![rc(i)];
    }
    let enemy: Vec<Cell> = (0..n)
        .filter(|&i| obs.owner_grid[i] as i32 == OWNER_ENEMY)
        .map(rc)
        .collect();
    if !enemy.is_empty() {
        return enemy;
    }
    if let Some(cell) = believed_enemy_general(belief) {
        return vec![cell];
    }
    let own = (0..n).find(|&i| memory.own_general[i]).or_else(|| {
        (0..n).find(|&i| obs.type_grid[i] as i32 == TYPE_GENERAL && obs.owner_grid[i] as i32 == 1)
    });
    match own {
        Some(i) => {
            let (gr, gc) = rc(i);
            vec![(obs.h - 1 - gr, obs.w - 1 - gc)]
        }
        None => Vec::new(),
    }
}

/// Drive toward the known/believed enemy general, else the nearest enemy land.
///
/// "Nearest" is measured from the largest own stack so the king does not aim
/// at a centroid behind mountains.
pub fn enemy_seek_target(
    obs: &Observation,
    memory: &VisibleMemory,
    belief: Option<&BeliefState>,
) -> Option<Cell> {
    let goals = seek_goals(obs, memory, belief);
    if goals.is_empty() {
        return None;
    }
    if goals.len() == 1 {
        return Some(goals[0]);
    }
    let king = king_cell(obs, None)?;
    let mut best = goals[0];
    let mut best_d = i64::MAX;
    for goal in &goals {
        let d = (goal.0 as i64 - king.0 as i64).abs() + (goal.1 as i64 - king.1 as i64).abs();
        if d < best_d {
            best_d = d;
            best = *goal;
        }
    }
    Some(best)
}

/// Where the attack wave forms: the frontmost own cell toward the hunt target.
///
/// Gathering used to target the largest movable stack, a target that moves
/// whenever any cell's army changes, so flows chased it and the army stayed
/// dribbled. This point is deterministic and rolls forward with each take.
pub fn wave_assembly_cell(
    obs: &Observation,
    memory: &VisibleMemory,
    belief: Option<&BeliefState>,
    exclude: Option<Cell>,
) -> Option<Cell> {
    let hunt = known_enemy_general_cell(obs, memory).or_else(|| believed_enemy_general(belief));
    let hunt = match hunt {
        Some(cell) => cell,
        None => return king_cell(obs, exclude),
    };
    let n = obs.h * obs.w;
    let mut own: Vec<bool> = (0..n).map(|i| obs.owner_grid[i] as i32 == 1).collect();
    if let Some(cell) = exclude {
        let at = cell.0 * obs.w + cell.1;
        if own[at] && own.iter().filter(|&&v| v).count() > 1 {
            own[at] = false;
        }
    }
    if !own.iter().any(|&v| v) {
        return None;
    }
    let field = path_distance_field(obs, &[hunt]);
    let mut dmin = i32::MAX;
    for i in 0..n {
        if own[i] && field.dist[i] >= 0 {
            dmin = dmin.min(field.dist[i]);
        }
    }
    if dmin == i32::MAX {
        return king_cell(obs, exclude);
    }
    let front: Vec<usize> = (0..n)
        .filter(|&i| own[i] && field.dist[i] == dmin)
        .collect();
    let max_a = front
        .iter()
        .map(|&i| obs.army_grid[i] as i64)
        .max()
        .unwrap_or(0);
    front
        .into_iter()
        .find(|&i| obs.army_grid[i] as i64 == max_a)
        .map(|i| (i / obs.w, i % obs.w))
}

// ------------------------------------------------------------------ play mask

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

// ------------------------------------------------------------------- planners

/// Savings/build site: the base-price own plain cell nearest the general.
///
/// Active only inside the build window with fewer than `CASTLE_TARGET` own
/// castles. The site must cost exactly `BASE_COST` — never pay a surcharge —
/// and sit at least `CASTLE_SAFE_ENEMY_DIST` Manhattan from any visible enemy
/// cell, because a castle on the front is a gift.
pub fn castle_build_site(obs: &Observation, memory: &VisibleMemory) -> Option<Cell> {
    let turn = turn_of(obs);
    if !(GARRISON_FLOOR_FROM..=CASTLE_WINDOW_UNTIL).contains(&turn) {
        return None;
    }
    let n = obs.h * obs.w;
    let visible_castles = (0..n)
        .filter(|&i| obs.type_grid[i] as i32 == TYPE_CASTLE && obs.owner_grid[i] as i32 == 1)
        .count() as i64;
    let latched = (0..n)
        .filter(|&i| memory.known_castle[i] && obs.owner_grid[i] as i32 == 1)
        .count() as i64;
    if visible_castles.max(latched) >= CASTLE_TARGET {
        return None;
    }

    let cost = live_build_cost(obs, memory);
    let candidate: Vec<bool> = (0..n)
        .map(|i| {
            let t = obs.type_grid[i] as i32;
            obs.owner_grid[i] as i32 == 1
                && cost[i] == BASE_COST
                && t != TYPE_GENERAL
                && t != TYPE_CASTLE
                && t != TYPE_MOUNTAIN
        })
        .collect();
    if !candidate.iter().any(|&v| v) {
        return None;
    }

    let enemy_cells: Vec<Cell> = (0..n)
        .filter(|&i| obs.owner_grid[i] as i32 == OWNER_ENEMY)
        .map(|i| (i / obs.w, i % obs.w))
        .collect();
    let gcell = own_general_cell(obs, memory);

    let mut best: Option<Cell> = None;
    let mut best_key: Option<(i64, i64, usize, usize)> = None;
    for i in 0..n {
        if !candidate[i] {
            continue;
        }
        let (r, c) = (i / obs.w, i % obs.w);
        if !enemy_cells.is_empty() {
            let d_enemy = enemy_cells
                .iter()
                .map(|&(er, ec)| {
                    (er as i64 - r as i64).abs() + (ec as i64 - c as i64).abs()
                })
                .min()
                .unwrap();
            if d_enemy < CASTLE_SAFE_ENEMY_DIST as i64 {
                continue;
            }
        }
        let d_gen = match gcell {
            Some((gr, gc)) => (gr as i64 - r as i64).abs() + (gc as i64 - c as i64).abs(),
            None => 0,
        };
        // Sticky: a cell already holding a pile outranks a marginally closer
        // empty one, so the site does not churn and strand its savings.
        let pile = (obs.army_grid[i] as i64).min(BASE_COST as i64);
        let key = (-pile, d_gen, r, c);
        if best_key.is_none() || key < best_key.unwrap() {
            best_key = Some(key);
            best = Some((r, c));
        }
    }
    best
}

/// Opponent army free to move: scoreboard total minus one pinned per cell.
pub fn opponent_mobile(obs: &Observation) -> i64 {
    (obs.opp_army as i64 - obs.opp_land as i64).max(0)
}

/// One gather-step toward the savings site: the biggest catchment tip moves.
///
/// A full move along the BFS gradient through own land only — the tithe is
/// logistics, never combat.
pub fn castle_tithe_move(
    obs: &Observation,
    memory: &VisibleMemory,
    site: Cell,
) -> Option<Action5> {
    let g = Grids::new(obs);
    let field = path_distance_field(obs, &[site]);
    let gcell = own_general_cell(obs, memory);
    let n = obs.h * obs.w;

    let mut best: Option<Action5> = None;
    let mut best_army = 1i64;
    for i in 0..n {
        if obs.owner_grid[i] as i32 != 1 {
            continue;
        }
        let cell = (i / obs.w, i % obs.w);
        if cell == site || Some(cell) == gcell {
            continue;
        }
        let d0 = field.dist[i];
        if !(1..=CASTLE_CATCHMENT).contains(&d0) {
            continue;
        }
        let a = obs.army_grid[i] as i64;
        if a <= best_army {
            continue;
        }
        let (r, c) = (cell.0 as i32, cell.1 as i32);
        for (d, (dr, dc)) in DIRECTIONS.iter().enumerate() {
            let (nr, nc) = (r + dr, c + dc);
            if !g.inside(nr, nc) {
                continue;
            }
            if field.get(nr, nc) != d0 - 1 || g.owner(nr, nc) != 1 {
                continue;
            }
            best = Some([0, r, c, d as i32, 0]);
            best_army = a;
            break;
        }
    }
    best
}

/// A visible enemy stack that can take our general: `(cell, arrival_steps)`.
///
/// A stack at path distance `d` arrives with `army - d` while the garrison
/// grows `d / 2`. Ties count as threats — the attacker may collect en route, so
/// the estimate errs toward defense. From deathtouch any stack that can reach
/// with one army is lethal.
pub fn general_threat(obs: &Observation, memory: &VisibleMemory) -> Option<(Cell, i32)> {
    let gcell = own_general_cell(obs, memory)?;
    let garrison = obs.army_grid[gcell.0 * obs.w + gcell.1] as i64;
    let turn = turn_of(obs);
    let field = path_distance_field(obs, &[gcell]);
    let n = obs.h * obs.w;

    let mut best: Option<(Cell, i32)> = None;
    let mut best_army = 0i64;
    for i in 0..n {
        if obs.owner_grid[i] as i32 != OWNER_ENEMY {
            continue;
        }
        let d = field.dist[i];
        if !(1..=DEFENSE_RADIUS).contains(&d) {
            continue;
        }
        let army = obs.army_grid[i] as i64;
        let arrival = army - d as i64;
        let lethal = if turn >= DEATHTOUCH_TURN {
            arrival >= 1
        } else {
            arrival >= garrison + (d / 2) as i64
        };
        if !lethal {
            continue;
        }
        let cell = (i / obs.w, i % obs.w);
        let better = match best {
            None => true,
            Some((_, best_d)) => (d, -army) < (best_d, -best_army),
        };
        if better {
            best = Some((cell, d));
            best_army = army;
        }
    }
    best
}

/// Best emergency response: capture the threat stack, else reinforce.
///
/// Reinforcement only counts if it lands on the general before the threat does
/// (`r <= d - 1`); the biggest such stack moves one gradient step home.
pub fn defend_general_move(
    obs: &Observation,
    memory: &VisibleMemory,
    threat: (Cell, i32),
) -> Option<Action5> {
    let (tcell, d) = threat;
    let gcell = own_general_cell(obs, memory)?;
    let g = Grids::new(obs);
    let t_army = obs.army_grid[tcell.0 * obs.w + tcell.1] as i64;

    // (a) Kill the threat outright from an adjacent own cell.
    let mut best_cap: Option<(i64, Action5)> = None;
    for (dd, (dr, dc)) in DIRECTIONS.iter().enumerate() {
        let sr = tcell.0 as i32 - dr;
        let sc = tcell.1 as i32 - dc;
        if !g.inside(sr, sc) {
            continue;
        }
        if g.owner(sr, sc) != 1 || (sr as usize, sc as usize) == gcell {
            continue;
        }
        let moved = g.army(sr, sc) - 1;
        if moved > t_army && best_cap.map_or(true, |(best, _)| moved > best) {
            best_cap = Some((moved, [0, sr, sc, dd as i32, 0]));
        }
    }
    if let Some((_, action)) = best_cap {
        return Some(action);
    }

    // (b) Reinforce the general in time.
    let field = path_distance_field(obs, &[gcell]);
    let n = obs.h * obs.w;
    let mut best: Option<(i64, Action5)> = None;
    for i in 0..n {
        if obs.owner_grid[i] as i32 != 1 || (obs.army_grid[i] as i64) < 2 {
            continue;
        }
        let cell = (i / obs.w, i % obs.w);
        if cell == gcell {
            continue;
        }
        let rr = field.dist[i];
        if !(1..=d - 1).contains(&rr) {
            continue;
        }
        let a = obs.army_grid[i] as i64;
        if best.map_or(false, |(best_a, _)| a <= best_a) {
            continue;
        }
        let (r, c) = (cell.0 as i32, cell.1 as i32);
        for (dd, (dr, dc)) in DIRECTIONS.iter().enumerate() {
            let (nr, nc) = (r + dr, c + dc);
            if !g.inside(nr, nc) {
                continue;
            }
            if field.get(nr, nc) != rr - 1 || g.owner(nr, nc) != 1 {
                continue;
            }
            best = Some((a, [0, r, c, dd as i32, 0]));
            break;
        }
    }
    best.map(|(_, action)| action)
}

/// Best winning march on the visible enemy general: `(steps, margin, first)`.
///
/// A greedy descent of the BFS gradient from each nearby own stack: full moves
/// that collect own armies en route, pay for neutral and enemy cells, and must
/// arrive with strictly more than the garrison plus its growth over the march.
/// From deathtouch any arrival wins. Replans every turn; if the window closes,
/// the plan silently disappears.
pub fn kill_plan(obs: &Observation, memory: &VisibleMemory) -> Option<(i32, i64, Action5)> {
    let g = Grids::new(obs);
    let gcell = known_enemy_general_cell(obs, memory)?;
    if g.owner(gcell.0 as i32, gcell.1 as i32) != OWNER_ENEMY {
        return None;
    }
    let garrison = g.army(gcell.0 as i32, gcell.1 as i32);
    let turn = turn_of(obs);
    let field = path_distance_field(obs, &[gcell]);

    let walk = |sr: i32, sc: i32| -> Option<(i32, i64, Action5)> {
        let mut army = g.army(sr, sc);
        let mut cur = (sr, sc);
        let mut first: Option<Action5> = None;
        let mut steps = 0i32;
        while field.get(cur.0, cur.1) > 1 {
            // Prefer collecting own armies; else the cheapest cell to cross.
            let mut best_n: Option<(i32, i32, usize)> = None;
            let mut best_key: Option<(i32, i64)> = None;
            for (d, (dr, dc)) in DIRECTIONS.iter().enumerate() {
                let (nr, nc) = (cur.0 + dr, cur.1 + dc);
                if !g.inside(nr, nc) {
                    continue;
                }
                if field.get(nr, nc) != field.get(cur.0, cur.1) - 1 {
                    continue;
                }
                // Fogged cells hide their army — a march priced on unknown
                // costs is a doomed march. Visible cells only.
                let t = g.kind(nr, nc);
                if t == TYPE_FOG || t == TYPE_STRUCTURE_FOG {
                    continue;
                }
                let o = g.owner(nr, nc);
                let a_n = g.army(nr, nc);
                let key = if o == 1 { (0, -a_n) } else { (1, a_n) };
                if best_key.is_none() || key < best_key.unwrap() {
                    best_key = Some(key);
                    best_n = Some((nr, nc, d));
                }
            }
            let (nr, nc, d) = best_n?;
            let moved = army - 1;
            if moved <= 0 {
                return None;
            }
            let o = g.owner(nr, nc);
            let a_n = g.army(nr, nc);
            if o == 1 {
                army = moved + a_n;
            } else {
                if moved <= a_n {
                    return None;
                }
                army = moved - a_n;
            }
            if first.is_none() {
                first = Some([0, cur.0, cur.1, d as i32, 0]);
            }
            cur = (nr, nc);
            steps += 1;
        }
        // The touch itself.
        let moved = army - 1;
        steps += 1;
        let need = if turn >= DEATHTOUCH_TURN {
            0
        } else {
            garrison + ((steps + 1) / 2) as i64
        };
        if moved <= need {
            return None;
        }
        if first.is_none() {
            // Already adjacent: the touch is the first move.
            for (d, (dr, dc)) in DIRECTIONS.iter().enumerate() {
                let (nr, nc) = (cur.0 + dr, cur.1 + dc);
                if nr >= 0 && nc >= 0 && nr as usize == gcell.0 && nc as usize == gcell.1 {
                    first = Some([0, cur.0, cur.1, d as i32, 0]);
                }
            }
        }
        Some((steps, moved - need, first?))
    };

    let n = obs.h * obs.w;
    let mut best: Option<(i32, i64, Action5)> = None;
    for i in 0..n {
        if obs.owner_grid[i] as i32 != 1 || (obs.army_grid[i] as i64) < 2 {
            continue;
        }
        let d0 = field.dist[i];
        if !(1..=KILL_HORIZON).contains(&d0) {
            continue;
        }
        let (r, c) = ((i / obs.w) as i32, (i % obs.w) as i32);
        let plan = match walk(r, c) {
            Some(plan) => plan,
            None => continue,
        };
        let better = match best {
            None => true,
            Some((bs, bm, _)) => (plan.0, -plan.1) < (bs, -bm),
        };
        if better {
            best = Some(plan);
        }
    }
    best
}

/// First step of the best winning kill march.
pub fn winning_kill_move(obs: &Observation, memory: &VisibleMemory) -> Option<Action5> {
    kill_plan(obs, memory).map(|plan| plan.2)
}

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

    fn board(h: usize, w: usize) -> Observation {
        let mut obs = Observation::with_dims(h, w);
        for i in 0..h * w {
            obs.type_grid[i] = 1;
        }
        obs
    }

    #[test]
    fn the_decode_tables_agree_with_the_codec() {
        let tables = decode_tables();
        for index in 0..PASS_INDEX {
            let action = decode_action(index).unwrap();
            assert_eq!(tables.kind[index] as i32, action[0]);
            assert_eq!(tables.sr[index] as i32, action[1]);
            assert_eq!(tables.sc[index] as i32, action[2]);
            if action[0] == 0 {
                let (dr, dc) = DIRECTIONS[action[3] as usize];
                assert_eq!(tables.tr[index] as i32, action[1] + dr);
                assert_eq!(tables.tc[index] as i32, action[2] + dc);
            } else {
                assert_eq!(tables.tr[index] as i32, action[1]);
            }
        }
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
    fn the_distance_field_walks_around_a_mountain() {
        let mut obs = board(3, 3);
        let centre = obs.idx(1, 1);
        obs.type_grid[centre] = TYPE_MOUNTAIN as u8;
        let field = path_distance_field(&obs, &[(0, 0)]);
        assert_eq!(field.get(0, 0), 0);
        assert_eq!(field.get(1, 1), -1);
        assert_eq!(field.get(2, 2), 4);
    }

    #[test]
    fn the_garrison_floor_scales_and_clamps() {
        assert_eq!(garrison_floor(0), GARRISON_FLOOR_MIN);
        assert_eq!(garrison_floor(10_000), GARRISON_FLOOR_CAP);
        // 4% of 350 is 14, between the floor and the cap.
        assert_eq!(garrison_floor(350), 14);
    }

    #[test]
    fn an_oscillating_reverse_is_blocked_but_an_enemy_take_is_not() {
        let mut obs = board(1, 3);
        obs.owner_grid[0] = 1;
        obs.army_grid[0] = 5;
        obs.owner_grid[1] = 1;
        obs.army_grid[1] = 5;
        // Previous move went (0,1) -> (0,0); the reverse is (0,0) -> (0,1).
        let prev = [0, 0, 1, 2, 0];
        assert!(blocks_oscillation(
            [0, 0, 0, 3, 0],
            Some(prev),
            &obs,
            &[],
            false,
            None
        ));
        obs.owner_grid[1] = 2;
        assert!(!blocks_oscillation(
            [0, 0, 0, 3, 0],
            Some(prev),
            &obs,
            &[],
            false,
            None
        ));
    }

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
