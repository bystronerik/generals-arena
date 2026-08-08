//! Parity subcommands: run one ported surface over recorded cases.
//!
//! The Rust half of the tier-1 harness in rewrite-plan §5. `pytest` under
//! `bots/morpheus-rs/tests/` builds cases from the M0 corpus, runs the Python
//! oracle on them, invokes the binary on the same cases, and compares.
//!
//! **Format: a flat stream of integers, not JSON.** The plan says "canonical
//! JSON", and this is a deliberate, narrow deviation. Rust's standard library
//! has no JSON, so the choice was a dependency in the *shipped* binary or two
//! hundred lines of hand-rolled parser — for a machine-to-machine channel
//! whose entire payload is integers. The 10,000-file unpacked limit is the
//! binding sandbox constraint (§1), the crate is zero-dependency to protect
//! it, and integers compare bit-exactly with no float formatting to argue
//! about. Reading is the same `parse_ints` the wire protocol already uses.
//!
//! Every layout below is positional and shared with `tests/parity_cases.py`.
//! Lengths are self-describing: each case starts with the dimensions that
//! determine how many integers follow, so a truncated stream fails loudly at
//! the point of truncation instead of silently shifting every later case.

use std::io::{BufRead, Write};

use crate::action::{legal_mask, live_build_cost, N_ACTIONS};
use crate::memory::VisibleMemory;
use crate::observe::emit_observation;
use crate::state::GameState;
use crate::transition::{determine_move_order, transition, Actions};
use crate::wire::Observation;

/// Positional reader over the whitespace-separated integer stream.
pub struct Ints {
    values: Vec<i64>,
    at: usize,
}

impl Ints {
    pub fn read_all<R: BufRead>(reader: &mut R) -> Result<Self, String> {
        let mut text = String::new();
        reader
            .read_to_string(&mut text)
            .map_err(|e| format!("reading cases: {e}"))?;
        let mut values = Vec::new();
        for token in text.split_ascii_whitespace() {
            values.push(
                token
                    .parse::<i64>()
                    .map_err(|_| format!("not an integer: {token:?}"))?,
            );
        }
        Ok(Self { values, at: 0 })
    }

    pub fn next(&mut self) -> Result<i64, String> {
        let value = *self
            .values
            .get(self.at)
            .ok_or_else(|| format!("case stream ended after {} integers", self.at))?;
        self.at += 1;
        Ok(value)
    }

    fn n(&mut self) -> Result<usize, String> {
        Ok(self.next()? as usize)
    }

    fn ints(&mut self, count: usize) -> Result<Vec<i32>, String> {
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            out.push(self.next()? as i32);
        }
        Ok(out)
    }

    fn action(&mut self) -> Result<[i32; 5], String> {
        Ok([
            self.next()? as i32,
            self.next()? as i32,
            self.next()? as i32,
            self.next()? as i32,
            self.next()? as i32,
        ])
    }
}

fn fill_bools(dst: &mut [bool], src: &[i32]) {
    for (i, v) in src.iter().enumerate() {
        dst[i] = *v != 0;
    }
}

/// `h w time winner gp[4] armies[n] own0[n] own1[n] neutral[n] generals[n]
/// castles[n] mountains[n] passable[n]`
fn read_state(ints: &mut Ints) -> Result<GameState, String> {
    let h = ints.n()?;
    let w = ints.n()?;
    let mut state = GameState::empty(h, w);
    state.time = ints.next()? as i32;
    state.winner = ints.next()? as i32;
    for seat in 0..2 {
        state.general_positions[seat] = [ints.next()? as i32, ints.next()? as i32];
    }
    let n = h * w;
    let armies = ints.ints(n)?;
    state.armies[..n].copy_from_slice(&armies);
    let planes = ints.ints(n * 7)?;
    fill_bools(&mut state.ownership[0], &planes[0..n]);
    fill_bools(&mut state.ownership[1], &planes[n..2 * n]);
    fill_bools(&mut state.ownership_neutral, &planes[2 * n..3 * n]);
    fill_bools(&mut state.generals, &planes[3 * n..4 * n]);
    fill_bools(&mut state.castles, &planes[4 * n..5 * n]);
    fill_bools(&mut state.mountains, &planes[5 * n..6 * n]);
    fill_bools(&mut state.passable, &planes[6 * n..7 * n]);
    Ok(state)
}

fn write_state(out: &mut Vec<i64>, state: &GameState) {
    let n = state.cells();
    out.push(state.h as i64);
    out.push(state.w as i64);
    out.push(state.time as i64);
    out.push(state.winner as i64);
    for seat in 0..2 {
        out.push(state.general_positions[seat][0] as i64);
        out.push(state.general_positions[seat][1] as i64);
    }
    out.extend(state.armies[..n].iter().map(|&v| v as i64));
    for plane in [
        &state.ownership[0][..n],
        &state.ownership[1][..n],
        &state.ownership_neutral[..n],
        &state.generals[..n],
        &state.castles[..n],
        &state.mountains[..n],
        &state.passable[..n],
    ] {
        out.extend(plane.iter().map(|&v| v as i64));
    }
}

/// `h w turn my_land my_army opp_land opp_army type[n] owner[n] army[n]`
fn read_observation(ints: &mut Ints) -> Result<Observation, String> {
    let h = ints.n()?;
    let w = ints.n()?;
    let mut obs = Observation::with_dims(h, w);
    obs.turn = ints.next()? as i32;
    obs.my_land = ints.next()? as i32;
    obs.my_army = ints.next()? as i32;
    obs.opp_land = ints.next()? as i32;
    obs.opp_army = ints.next()? as i32;
    let n = h * w;
    for i in 0..n {
        obs.type_grid[i] = ints.next()? as u8;
    }
    for i in 0..n {
        obs.owner_grid[i] = ints.next()? as u8;
    }
    for i in 0..n {
        obs.army_grid[i] = ints.next()? as i32;
    }
    Ok(obs)
}

fn write_observation(out: &mut Vec<i64>, obs: &Observation) {
    let n = obs.h * obs.w;
    out.extend([
        obs.h as i64,
        obs.w as i64,
        obs.turn as i64,
        obs.my_land as i64,
        obs.my_army as i64,
        obs.opp_land as i64,
        obs.opp_army as i64,
    ]);
    out.extend(obs.type_grid[..n].iter().map(|&v| v as i64));
    out.extend(obs.owner_grid[..n].iter().map(|&v| v as i64));
    out.extend(obs.army_grid[..n].iter().map(|&v| v as i64));
}

/// `h w` then the eleven planes, in `VisibleMemory` field order.
fn read_memory(ints: &mut Ints) -> Result<VisibleMemory, String> {
    let h = ints.n()?;
    let w = ints.n()?;
    let n = h * w;
    let mut memory = VisibleMemory::empty(h, w);
    let planes = ints.ints(n * 11)?;
    fill_bools(&mut memory.known_mountain, &planes[0..n]);
    fill_bools(&mut memory.known_passable_base, &planes[n..2 * n]);
    fill_bools(&mut memory.known_castle, &planes[2 * n..3 * n]);
    fill_bools(&mut memory.own_general, &planes[3 * n..4 * n]);
    fill_bools(&mut memory.known_enemy_general, &planes[4 * n..5 * n]);
    fill_bools(&mut memory.ever_visible, &planes[5 * n..6 * n]);
    memory.last_seen_turn[..n].copy_from_slice(&planes[6 * n..7 * n]);
    memory.remembered_owner[..n].copy_from_slice(&planes[7 * n..8 * n]);
    memory.remembered_army[..n].copy_from_slice(&planes[8 * n..9 * n]);
    fill_bools(&mut memory.remembered_was_castle, &planes[9 * n..10 * n]);
    memory.remembered_castle_owner[..n].copy_from_slice(&planes[10 * n..11 * n]);
    Ok(memory)
}

/// Run one parity kind end to end over stdin, writing one line per case.
pub fn run<R: BufRead, W: Write>(kind: &str, reader: &mut R, writer: &mut W) -> Result<(), String> {
    let mut ints = Ints::read_all(reader)?;
    let cases = ints.n()?;
    let mut out: Vec<i64> = Vec::new();

    for case in 0..cases {
        out.clear();
        match kind {
            // state + both actions -> next state + info
            "transition" => {
                let state = read_state(&mut ints)?;
                let actions: Actions = [ints.action()?, ints.action()?];
                let (next, info) = transition(&state, &actions);
                write_state(&mut out, &next);
                out.extend([
                    info.army[0],
                    info.army[1],
                    info.land[0],
                    info.land[1],
                    info.is_done as i64,
                    info.winner as i64,
                    info.time as i64,
                ]);
            }
            // state + both actions -> which seat resolves first
            //
            // Its own surface rather than a detail of `transition`, because
            // move order is mostly *unobservable* downstream: when the two
            // moves do not interact, either order gives the same board. A
            // mutation that broke the NumPy index wrap here survived the
            // end-to-end transition check over hundreds of recorded positions
            // and is caught immediately by this one.
            "order" => {
                let state = read_state(&mut ints)?;
                let actions: Actions = [ints.action()?, ints.action()?];
                out.push(determine_move_order(&state, &actions) as i64);
            }
            // state + seat -> the fogged observation that seat receives
            "observe" => {
                let state = read_state(&mut ints)?;
                let seat = ints.n()?;
                write_observation(&mut out, &emit_observation(&state, seat));
            }
            // observation + memory -> the 3970-long legal mask
            "mask" => {
                let obs = read_observation(&mut ints)?;
                let memory = read_memory(&mut ints)?;
                let mask = legal_mask(&obs, &memory, None);
                out.reserve(N_ACTIONS);
                out.extend(mask.iter().map(|&v| v as i64));
            }
            // observation + memory -> the live build-cost grid
            "cost" => {
                let obs = read_observation(&mut ints)?;
                let memory = read_memory(&mut ints)?;
                let cost = live_build_cost(&obs, &memory);
                out.extend(cost[..obs.h * obs.w].iter().map(|&v| v as i64));
            }
            other => return Err(format!("unknown parity kind {other:?}")),
        }

        let mut line = String::with_capacity(out.len() * 4);
        for (i, value) in out.iter().enumerate() {
            if i > 0 {
                line.push(' ');
            }
            line.push_str(itoa(*value).as_str());
        }
        writeln!(writer, "{line}").map_err(|e| format!("case {case}: {e}"))?;
    }
    writer.flush().map_err(|e| format!("flush: {e}"))?;
    Ok(())
}

/// `i64` to text without pulling in a formatting dependency.
fn itoa(value: i64) -> String {
    value.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn tiny_state_ints() -> String {
        // 1x2 board, time 0, winner -1, generals at (0,0) and (0,1).
        let mut s = String::from("1 2 0 -1 0 0 0 1 ");
        s.push_str("5 3 "); // armies
        s.push_str("1 0 "); // own0
        s.push_str("0 1 "); // own1
        s.push_str("0 0 "); // neutral
        s.push_str("1 1 "); // generals
        s.push_str("0 0 "); // castles
        s.push_str("0 0 "); // mountains
        s.push_str("1 1 "); // passable
        s
    }

    #[test]
    fn a_transition_case_round_trips_through_the_stream() {
        let input = format!("1 {} 1 0 0 0 0 1 0 0 0 0", tiny_state_ints());
        let mut out = Vec::new();
        run("transition", &mut Cursor::new(input), &mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        let values: Vec<i64> = text
            .split_ascii_whitespace()
            .map(|t| t.parse().unwrap())
            .collect();
        // h w time winner + 4 general coords + 8 planes of 2 cells + 7 info
        assert_eq!(values.len(), 4 + 4 + 8 * 2 + 7);
        assert_eq!(values[0], 1);
        assert_eq!(values[1], 2);
        assert_eq!(values[2], 1, "a double pass still advances the clock");
    }

    #[test]
    fn a_truncated_stream_fails_loudly() {
        let mut out = Vec::new();
        let err = run("transition", &mut Cursor::new("1 1 2 0"), &mut out).unwrap_err();
        assert!(err.contains("ended after"), "{err}");
    }

    #[test]
    fn an_unknown_kind_is_refused() {
        let mut out = Vec::new();
        assert!(run("wat", &mut Cursor::new("1"), &mut out).is_err());
    }
}
