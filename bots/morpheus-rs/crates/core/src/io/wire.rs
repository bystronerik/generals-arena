//! The stdio wire protocol, agent side.
//!
//! Mirrors `competition-module/competition/protocol.py`, which is the
//! engine-side half and the authority. All integers, line-delimited, ASCII:
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
//! The engine ends a game by closing the agent's stdin. There is no explicit
//! signal, so EOF at any point where a frame could start means "game over,
//! exit cleanly" — and EOF *mid-frame* means the engine went away while we
//! were reading, which is the same thing and not an error to report.
//!
//! Grids are stored flat and row-major. The board is at most 21×21 (`pad_to`
//! in the competition preset), so a frame is small; what matters is that
//! reading one allocates nothing per turn beyond the line buffer, because this
//! runs inside a 150 ms budget on one core.

use std::io::{BufRead, Write};

/// Cell types, perspective-relative. `StructureInFog` is impassable like a
/// mountain — it is a castle or general known to be there but not visible.
pub const TYPE_FOG: u8 = 0;
pub const TYPE_PLAIN: u8 = 1;
pub const TYPE_MOUNTAIN: u8 = 2;
pub const TYPE_CASTLE: u8 = 3;
pub const TYPE_GENERAL: u8 = 4;
pub const TYPE_STRUCTURE_IN_FOG: u8 = 5;

pub const OWNER_NEUTRAL: u8 = 0;
pub const OWNER_ME: u8 = 1;
pub const OWNER_OPP: u8 = 2;

/// `(dr, dc)` for direction codes 0..3 — up, down, left, right.
pub const DIRECTIONS: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Handshake {
    pub player_id: u8,
    pub h: usize,
    pub w: usize,
}

/// One turn's view, exactly as it came off the wire.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Observation {
    pub h: usize,
    pub w: usize,
    pub turn: i32,
    pub my_land: i32,
    pub my_army: i32,
    pub opp_land: i32,
    pub opp_army: i32,
    pub type_grid: Vec<u8>,
    pub owner_grid: Vec<u8>,
    pub army_grid: Vec<i32>,
}

impl Observation {
    #[inline]
    pub fn idx(&self, row: usize, col: usize) -> usize {
        row * self.w + col
    }

    /// Allocate the grids once, for a board whose size the handshake fixed.
    /// Reused across turns so the per-turn path never grows the heap.
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

impl Action {
    pub fn mv(row: u16, col: u16, dir: u8, split: u8) -> Self {
        Self { pass: 0, row, col, dir, split }
    }

    pub fn build(row: u16, col: u16) -> Self {
        Self { pass: 2, row, col, dir: 0, split: 0 }
    }
}

#[derive(Debug)]
pub enum WireError {
    Io(std::io::Error),
    /// A frame arrived that this parser cannot make sense of. Distinct from
    /// EOF: the engine is still there and said something unexpected.
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

/// Read one line, `Ok(false)` at EOF. The buffer is cleared first, so callers
/// keep one allocation for the whole game.
fn read_line<R: BufRead>(reader: &mut R, buf: &mut String) -> Result<bool> {
    buf.clear();
    Ok(reader.read_line(buf)? != 0)
}

fn parse_ints<T: std::str::FromStr>(line: &str, want: usize, what: &str) -> Result<Vec<T>> {
    let mut out = Vec::with_capacity(want);
    for token in line.split_ascii_whitespace() {
        match token.parse::<T>() {
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
    Ok(out)
}

pub fn read_handshake<R: BufRead>(reader: &mut R) -> Result<Option<Handshake>> {
    let mut line = String::new();
    if !read_line(reader, &mut line)? {
        return Ok(None);
    }
    let parts: Vec<i64> = parse_ints(&line, 3, "handshake")?;
    let (player_id, h, w) = (parts[0], parts[1], parts[2]);
    if !(0..=1).contains(&player_id) || h <= 0 || w <= 0 {
        return Err(WireError::Malformed(format!(
            "handshake: player_id {player_id}, dims {h}x{w}"
        )));
    }
    Ok(Some(Handshake { player_id: player_id as u8, h: h as usize, w: w as usize }))
}

/// Fill `obs` with the next turn's frame. `Ok(false)` at EOF.
///
/// Takes `&mut Observation` rather than returning one so the grids allocated
/// at handshake are reused for every turn of the game.
pub fn read_observation<R: BufRead>(
    reader: &mut R,
    obs: &mut Observation,
    line: &mut String,
) -> Result<bool> {
    if !read_line(reader, line)? {
        return Ok(false);
    }
    let scalars: Vec<i32> = parse_ints(line, 5, "scalars")?;
    obs.turn = scalars[0];
    obs.my_land = scalars[1];
    obs.my_army = scalars[2];
    obs.opp_land = scalars[3];
    obs.opp_army = scalars[4];

    let (h, w) = (obs.h, obs.w);
    for row in 0..h {
        if !read_line(reader, line)? {
            return Ok(false); // engine closed mid-frame; same as game over
        }
        let values: Vec<u8> = parse_ints(line, w, "type grid")?;
        obs.type_grid[row * w..(row + 1) * w].copy_from_slice(&values);
    }
    for row in 0..h {
        if !read_line(reader, line)? {
            return Ok(false);
        }
        let values: Vec<u8> = parse_ints(line, w, "owner grid")?;
        obs.owner_grid[row * w..(row + 1) * w].copy_from_slice(&values);
    }
    for row in 0..h {
        if !read_line(reader, line)? {
            return Ok(false);
        }
        let values: Vec<i32> = parse_ints(line, w, "army grid")?;
        obs.army_grid[row * w..(row + 1) * w].copy_from_slice(&values);
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

    /// A frame in exactly the shape `protocol.py::encode_observation` emits.
    fn frame(h: usize, w: usize, turn: i32) -> String {
        let mut out = format!("{turn} 1 2 3 4\n");
        for grid in 0..3 {
            for row in 0..h {
                let cells: Vec<String> =
                    (0..w).map(|col| ((grid * 100 + row * 10 + col) % 6).to_string()).collect();
                out.push_str(&cells.join(" "));
                out.push('\n');
            }
        }
        out
    }

    #[test]
    fn handshake_parses() {
        let mut r = Cursor::new("1 21 18\n");
        let hs = read_handshake(&mut r).unwrap().unwrap();
        assert_eq!(hs, Handshake { player_id: 1, h: 21, w: 18 });
    }

    #[test]
    fn handshake_at_eof_is_none_not_an_error() {
        let mut r = Cursor::new("");
        assert!(read_handshake(&mut r).unwrap().is_none());
    }

    #[test]
    fn handshake_rejects_a_third_player() {
        let mut r = Cursor::new("2 21 21\n");
        assert!(read_handshake(&mut r).is_err());
    }

    #[test]
    fn observation_round_trips_a_real_shaped_frame() {
        let (h, w) = (4usize, 3usize);
        let mut r = Cursor::new(frame(h, w, 7));
        let mut obs = Observation::with_dims(h, w);
        let mut line = String::new();
        assert!(read_observation(&mut r, &mut obs, &mut line).unwrap());

        assert_eq!((obs.turn, obs.my_land, obs.my_army), (7, 1, 2));
        assert_eq!((obs.opp_land, obs.opp_army), (3, 4));
        assert_eq!(obs.type_grid.len(), h * w);
        // Row-major, and the three grids are read in protocol order.
        assert_eq!(obs.type_grid[obs.idx(1, 2)], ((0 * 100 + 12) % 6) as u8);
        assert_eq!(obs.owner_grid[obs.idx(1, 2)], ((100 + 12) % 6) as u8);
        assert_eq!(obs.army_grid[obs.idx(1, 2)], ((200 + 12) % 6) as i32);
    }

    #[test]
    fn successive_turns_reuse_one_observation() {
        let (h, w) = (2usize, 2usize);
        let mut input = frame(h, w, 1);
        input.push_str(&frame(h, w, 2));
        let mut r = Cursor::new(input);
        let mut obs = Observation::with_dims(h, w);
        let mut line = String::new();

        assert!(read_observation(&mut r, &mut obs, &mut line).unwrap());
        assert_eq!(obs.turn, 1);
        let capacity = obs.type_grid.capacity();
        assert!(read_observation(&mut r, &mut obs, &mut line).unwrap());
        assert_eq!(obs.turn, 2);
        assert_eq!(obs.type_grid.capacity(), capacity, "per-turn reallocation");

        // Third read hits EOF: game over, not an error.
        assert!(!read_observation(&mut r, &mut obs, &mut line).unwrap());
    }

    #[test]
    fn a_frame_cut_short_reads_as_game_over() {
        // The engine closing stdin mid-frame is how a finished game can look
        // from here. It must not be reported as a protocol violation.
        let full = frame(4, 3, 1);
        let truncated: String = full.lines().take(3).collect::<Vec<_>>().join("\n") + "\n";
        let mut r = Cursor::new(truncated);
        let mut obs = Observation::with_dims(4, 3);
        let mut line = String::new();
        assert!(!read_observation(&mut r, &mut obs, &mut line).unwrap());
    }

    #[test]
    fn a_short_row_is_malformed_not_eof() {
        let mut r = Cursor::new("1 1 1 1 1\n0 0\n");
        let mut obs = Observation::with_dims(2, 3);
        let mut line = String::new();
        assert!(read_observation(&mut r, &mut obs, &mut line).is_err());
    }

    #[test]
    fn actions_serialize_in_protocol_order() {
        let mut out = Vec::new();
        write_action(&mut out, PASS).unwrap();
        write_action(&mut out, Action::mv(3, 4, 2, 1)).unwrap();
        write_action(&mut out, Action::build(5, 6)).unwrap();
        assert_eq!(String::from_utf8(out).unwrap(), "1 0 0 0 0\n0 3 4 2 1\n2 5 6 0 0\n");
    }
}
