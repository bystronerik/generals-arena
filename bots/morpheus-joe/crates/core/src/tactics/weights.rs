//! The scalar shaping terms: how much a thing is worth, as one number.
//!
//! Fog urgency, the wave and attack weights, the thrash and gather factors,
//! and the direction bias. Each is a small function of army size, turn, or
//! progress that the scorer multiplies into a per-action score; none of them
//! reads the board.

use crate::board::memory::OWNER_ENEMY;

use super::*;

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
