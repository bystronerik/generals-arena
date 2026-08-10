//! The competition board state, laid out for copying rather than for reading.
//!
//! Port of `bots/morpheus/state.py`. Field names and meanings match the
//! engine's own `GameState` so a differential test can compare cell-wise.
//!
//! **Fixed-size inline arrays, not `Vec`.** The competition preset pads every
//! board to 21×21, so the largest board is known at compile time and a state
//! is ~3 KB of plain bytes. That is what rewrite-plan §6 asks for: particles
//! and search scratch live in one preallocated slab, and "clone a particle" is
//! a `memcpy` rather than nine heap allocations. `h`/`w` describe the live
//! region; the row stride is `w`, so cells outside it are never touched.
//!
//! `Clone` but deliberately **not** `Copy`: at this size an implicit copy at a
//! call site is a real cost, and it should be visible in the source.

pub const MAX_DIM: usize = 21;
pub const MAX_CELLS: usize = MAX_DIM * MAX_DIM;

/// Complete board state. Mirrors `competition-module/generals/core/game.py`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct GameState {
    pub h: usize,
    pub w: usize,
    pub armies: [i32; MAX_CELLS],
    /// Per-seat ownership. `ownership[seat][idx]`.
    pub ownership: [[bool; MAX_CELLS]; 2],
    pub ownership_neutral: [bool; MAX_CELLS],
    pub generals: [bool; MAX_CELLS],
    pub castles: [bool; MAX_CELLS],
    pub mountains: [bool; MAX_CELLS],
    pub passable: [bool; MAX_CELLS],
    /// `[seat][0] = row`, `[seat][1] = col`. `-1` when unknown.
    pub general_positions: [[i32; 2]; 2],
    /// Pre-step timestep.
    pub time: i32,
    /// `-1` ongoing, otherwise the winning seat.
    pub winner: i32,
}

/// Per-turn totals and the terminal flag after one transition.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GameInfo {
    pub army: [i64; 2],
    pub land: [i64; 2],
    pub is_done: bool,
    pub winner: i32,
    pub time: i32,
}

impl GameState {
    pub fn empty(h: usize, w: usize) -> Self {
        Self {
            h,
            w,
            armies: [0; MAX_CELLS],
            ownership: [[false; MAX_CELLS]; 2],
            ownership_neutral: [false; MAX_CELLS],
            generals: [false; MAX_CELLS],
            castles: [false; MAX_CELLS],
            mountains: [false; MAX_CELLS],
            passable: [false; MAX_CELLS],
            general_positions: [[-1, -1], [-1, -1]],
            time: 0,
            winner: -1,
        }
    }

    #[inline]
    pub fn idx(&self, row: usize, col: usize) -> usize {
        row * self.w + col
    }

    #[inline]
    pub fn cells(&self) -> usize {
        self.h * self.w
    }

    #[inline]
    pub fn in_bounds(&self, row: i32, col: i32) -> bool {
        row >= 0 && col >= 0 && (row as usize) < self.h && (col as usize) < self.w
    }

    /// Index for a possibly-negative coordinate, wrapping as NumPy does.
    ///
    /// `_determine_move_order` in the Python reads `ownership[seat, di, dj]`
    /// with raw arithmetic that can go negative — a pass action is
    /// `[1, 0, 0, 0, 0]`, so `di = 0 + DIRECTIONS[0][0] = -1` — and NumPy
    /// resolves `-1` as the last row. That wrap is load-bearing: it decides
    /// move order on every turn where either seat passes. Positive overflow
    /// would raise in NumPy, so it cannot occur for in-contract actions;
    /// `None` here marks that domain rather than inventing a value for it.
    #[inline]
    pub fn wrapped_idx(&self, row: i32, col: i32) -> Option<usize> {
        let r = if row < 0 { row + self.h as i32 } else { row };
        let c = if col < 0 { col + self.w as i32 } else { col };
        if r < 0 || c < 0 || r as usize >= self.h || c as usize >= self.w {
            return None;
        }
        Some(r as usize * self.w + c as usize)
    }
}
