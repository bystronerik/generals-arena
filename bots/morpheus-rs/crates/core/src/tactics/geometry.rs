//! Action decoding and the board's coordinate arithmetic.
//!
//! [`decode_tables`] is the static `(kind, sr, sc, tr, tc)` table the rules
//! index instead of re-decoding, [`Grids`] the read-only view of the three
//! observation grids at the live stride, and the rest turns an action into the
//! segment it moves along. Destinations here are *unclipped*: the tables index
//! the padded 21x21 layout while the grids index the live `HxW`, so every
//! caller bounds-checks — the Python gets away with implicit masking
//! because only legal indices survive its `flatnonzero`.

use std::sync::OnceLock;

use crate::board::action::{decode_action, PASS_INDEX};
use crate::belief::Action5;
use crate::board::transition::DIRECTIONS;
use crate::io::wire::Observation;

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

pub(super) fn turn_of(obs: &Observation) -> i32 {
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

#[cfg(test)]
mod tests {
    use super::*;
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
}
