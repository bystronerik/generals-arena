//! Every tuning constant the hand-written rules read.
//!
//! One contiguous block at the top of `tactics.py` and one file here, because
//! this is the bot's tuning surface: a retune changes numbers in this file and
//! nothing else. Each constant keeps the measurement or the field-observed
//! loss that set it, copied from the source it was written in.

use crate::board::action::PASS_INDEX;

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
