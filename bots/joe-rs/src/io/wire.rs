//! The stdio wire protocol, agent side.
//!
//! Mirrors `competition-module/competition/protocol.py` (the engine-side
//! half and the authority). All integers, line-delimited, ASCII:
//!
//! ```text
//! handshake, once:   <player_id> <H> <W>
//! per turn, in:      <turn> <my_land> <my_army> <opp_land> <opp_army>
//!                    H lines of W ints   # type
//!                    H lines of W ints   # owner
//!                    H lines of W ints   # army
//! per turn, out:     <pass> <row> <col> <dir> <split>
//! ```
//!
//! The engine ends a game by closing the agent's stdin, so EOF anywhere a
//! frame could start — or mid-frame — means "game over, exit cleanly".

use std::io::{BufRead, Write};

/// Cell types on the wire (`protocol.py`).
pub const TYPE_FOG: i32 = 0;
pub const TYPE_PLAIN: i32 = 1;
pub const TYPE_MOUNTAIN: i32 = 2;
pub const TYPE_CASTLE: i32 = 3;
pub const TYPE_GENERAL: i32 = 4;
pub const TYPE_STRUCTURE_IN_FOG: i32 = 5;

pub const OWNER_ME: i32 = 1;
pub const OWNER_OPP: i32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Handshake {
    pub player_id: u8,
    pub h: usize,
    pub w: usize,
}

/// One turn's view, exactly as it came off the wire. Grids are flat,
/// row-major, allocated once at handshake and reused every turn.
#[derive(Debug, Clone, Default)]
pub struct Observation {
    pub h: usize,
    pub w: usize,
    pub turn: i32,
    pub my_land: i32,
    pub my_army: i32,
    pub opp_land: i32,
    pub opp_army: i32,
    pub type_grid: Vec<i32>,
    pub owner_grid: Vec<i32>,
    pub army_grid: Vec<i32>,
}

impl Observation {
    pub fn with_dims(h: usize, w: usize) -> Self {
        Self {
            h,
            w,
            type_grid: vec![0; h * w],
            owner_grid: vec![0; h * w],
            army_grid: vec![0; h * w],
            ..Default::default()
        }
    }
}

/// `pass row col dir split`. `pass`: 0 move, 1 skip, 2 build a castle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Action {
    pub pass: u8,
    pub row: u16,
    pub col: u16,
    pub dir: u8,
    pub split: u8,
}

pub const PASS: Action = Action { pass: 1, row: 0, col: 0, dir: 0, split: 0 };

#[derive(Debug)]
pub enum WireError {
    Io(std::io::Error),
    Malformed(String),
}

impl std::fmt::Display for WireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WireError::Io(e) => write!(f, "io: {e}"),
            WireError::Malformed(m) => write!(f, "malformed frame: {m}"),
        }
    }
}

impl std::error::Error for WireError {}

impl From<std::io::Error> for WireError {
    fn from(e: std::io::Error) -> Self {
        WireError::Io(e)
    }
}

type Result<T> = std::result::Result<T, WireError>;

/// Read one line, `Ok(false)` at EOF. The buffer is cleared first, so the
/// caller keeps one allocation for the whole game.
fn read_line<R: BufRead>(reader: &mut R, buf: &mut String) -> Result<bool> {
    buf.clear();
    Ok(reader.read_line(buf)? != 0)
}

/// Parse exactly `want` integers into `out` (cleared first).
fn parse_ints(line: &str, want: usize, out: &mut Vec<i32>, what: &str) -> Result<()> {
    out.clear();
    for token in line.split_ascii_whitespace() {
        match token.parse::<i32>() {
            Ok(value) => out.push(value),
            Err(_) => {
                return Err(WireError::Malformed(format!("{what}: bad integer {token:?}")))
            }
        }
    }
    if out.len() != want {
        return Err(WireError::Malformed(format!(
            "{what}: expected {want} integers, got {}",
            out.len()
        )));
    }
    Ok(())
}

pub fn read_handshake<R: BufRead>(reader: &mut R) -> Result<Option<Handshake>> {
    let mut line = String::new();
    if !read_line(reader, &mut line)? {
        return Ok(None);
    }
    let mut parts = Vec::new();
    parse_ints(&line, 3, &mut parts, "handshake")?;
    let (player_id, h, w) = (parts[0], parts[1], parts[2]);
    if !(0..=1).contains(&player_id) || h <= 0 || w <= 0 {
        return Err(WireError::Malformed(format!(
            "handshake: player_id {player_id}, dims {h}x{w}"
        )));
    }
    Ok(Some(Handshake { player_id: player_id as u8, h: h as usize, w: w as usize }))
}

/// Fill `obs` with the next turn's frame. `Ok(false)` at EOF (including
/// mid-frame: the engine went away, which is a normal game end).
pub fn read_observation<R: BufRead>(
    reader: &mut R,
    obs: &mut Observation,
    line: &mut String,
    scratch: &mut Vec<i32>,
) -> Result<bool> {
    if !read_line(reader, line)? {
        return Ok(false);
    }
    parse_ints(line, 5, scratch, "scalars")?;
    obs.turn = scratch[0];
    obs.my_land = scratch[1];
    obs.my_army = scratch[2];
    obs.opp_land = scratch[3];
    obs.opp_army = scratch[4];

    let (h, w) = (obs.h, obs.w);
    for grid_idx in 0..3 {
        for row in 0..h {
            if !read_line(reader, line)? {
                return Ok(false);
            }
            parse_ints(line, w, scratch, "grid row")?;
            let grid = match grid_idx {
                0 => &mut obs.type_grid,
                1 => &mut obs.owner_grid,
                _ => &mut obs.army_grid,
            };
            grid[row * w..(row + 1) * w].copy_from_slice(scratch);
        }
    }
    Ok(true)
}

/// Write and flush one reply. Flushing is not optional: the engine blocks on
/// this line, so a buffered reply is a timeout.
pub fn write_action<W: Write>(writer: &mut W, action: Action) -> Result<()> {
    writeln!(
        writer,
        "{} {} {} {} {}",
        action.pass, action.row, action.col, action.dir, action.split
    )?;
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn handshake_parses() {
        let mut r = Cursor::new("1 21 18\n");
        let hs = read_handshake(&mut r).unwrap().unwrap();
        assert_eq!(hs, Handshake { player_id: 1, h: 21, w: 18 });
    }

    #[test]
    fn handshake_at_eof_is_none() {
        let mut r = Cursor::new("");
        assert!(read_handshake(&mut r).unwrap().is_none());
    }

    #[test]
    fn observation_round_trip() {
        let (h, w) = (2usize, 3usize);
        let mut text = String::from("7 1 2 3 4\n");
        for g in 0..3 {
            for row in 0..h {
                let cells: Vec<String> =
                    (0..w).map(|col| (g * 100 + row * 10 + col).to_string()).collect();
                text.push_str(&cells.join(" "));
                text.push('\n');
            }
        }
        let mut r = Cursor::new(text);
        let mut obs = Observation::with_dims(h, w);
        let mut line = String::new();
        let mut scratch = Vec::new();
        assert!(read_observation(&mut r, &mut obs, &mut line, &mut scratch).unwrap());
        assert_eq!(obs.turn, 7);
        assert_eq!(obs.army_grid, vec![200, 201, 202, 210, 211, 212]);
        // EOF next.
        assert!(!read_observation(&mut r, &mut obs, &mut line, &mut scratch).unwrap());
    }
}
