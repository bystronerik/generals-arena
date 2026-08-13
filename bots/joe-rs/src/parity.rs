//! Parity subcommands: `joe-rs parity <surface>` reads cases from stdin as a
//! flat whitespace-separated integer stream and writes results to stdout in
//! the same shape (the morpheus-rs format — floats travel as their f32 bit
//! patterns, so bit-exact comparison has no float formatting to argue about).
//!
//! Surfaces (port-plan §6): `raw`, `cost`, `mask`, `obs`, `forward`,
//! `decide`, `sequence`. Every layout is positional and shared with
//! `tests/parity_cases.py`; each case starts with its dimensions, so a
//! truncated stream fails loudly at the truncation point.

use std::io::{Read, Write};
use std::path::PathBuf;

use crate::action::{argmax, decode_action};
use crate::net::{Net, N_LOGITS, NUM_BINS};
use crate::obs::{
    augment_obs, build_cost_from_raw, compute_build_mask_from_raw, compute_valid_move_mask,
    frame_to_raw, normalize_observations, prepare_action_mask, AugScratch, AugState, CELLS,
    HISTORY, N_ACTION_CHANNELS, N_CHANNELS, N_RAW_CHANNELS, TEMPORAL_WINDOW,
};
use crate::wire::Observation;

/// Sequential integer reader over the whole input stream.
struct Ints {
    values: Vec<i64>,
    pos: usize,
}

impl Ints {
    fn read_all<R: Read>(reader: &mut R) -> Result<Self, String> {
        let mut text = String::new();
        reader.read_to_string(&mut text).map_err(|e| format!("read stdin: {e}"))?;
        let mut values = Vec::new();
        for token in text.split_ascii_whitespace() {
            values.push(token.parse::<i64>().map_err(|_| format!("bad integer {token:?}"))?);
        }
        Ok(Self { values, pos: 0 })
    }

    fn n(&mut self) -> Result<i64, String> {
        if self.pos >= self.values.len() {
            return Err("truncated input stream".into());
        }
        self.pos += 1;
        Ok(self.values[self.pos - 1])
    }

    fn usize(&mut self) -> Result<usize, String> {
        let v = self.n()?;
        usize::try_from(v).map_err(|_| format!("expected non-negative integer, got {v}"))
    }

    fn f32_bits(&mut self) -> Result<f32, String> {
        let v = self.n()?;
        let bits = u32::try_from(v).map_err(|_| format!("bad f32 bit pattern {v}"))?;
        Ok(f32::from_bits(bits))
    }

    fn f32_vec(&mut self, len: usize) -> Result<Vec<f32>, String> {
        (0..len).map(|_| self.f32_bits()).collect()
    }

    fn bool_vec(&mut self, len: usize) -> Result<Vec<bool>, String> {
        (0..len).map(|_| Ok(self.n()? != 0)).collect()
    }
}

fn push_f32s(out: &mut Vec<i64>, values: &[f32]) {
    out.extend(values.iter().map(|v| v.to_bits() as i64));
}

fn push_bools(out: &mut Vec<i64>, values: &[bool]) {
    out.extend(values.iter().map(|&v| v as i64));
}

/// CRC-32 (the zlib polynomial, 0xEDB88320) over the little-endian bytes of
/// the f32 values — the sequence surface's per-turn tensor digest, mirrored
/// by `zlib.crc32` in `tools/capture_fixtures.py`.
fn crc32(values: &[f32]) -> i64 {
    let mut table = [0u32; 256];
    for (i, entry) in table.iter_mut().enumerate() {
        let mut c = i as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB88320 ^ (c >> 1) } else { c >> 1 };
        }
        *entry = c;
    }
    let mut crc = !0u32;
    for v in values {
        for b in v.to_bits().to_le_bytes() {
            crc = table[((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8);
        }
    }
    (!crc) as i64
}

fn read_state(ints: &mut Ints) -> Result<AugState, String> {
    let mut s = AugState::zeros();
    s.army_stack = ints.f32_vec(HISTORY * CELLS)?;
    s.enemy_stack = ints.f32_vec(HISTORY * CELLS)?;
    s.last_army = ints.f32_vec(CELLS)?;
    s.last_enemy_army = ints.f32_vec(CELLS)?;
    s.castles = ints.bool_vec(CELLS)?;
    s.generals = ints.bool_vec(CELLS)?;
    s.mountains = ints.bool_vec(CELLS)?;
    s.seen = ints.bool_vec(CELLS)?;
    s.enemy_seen = ints.bool_vec(CELLS)?;
    s.last_enemy_army_seen_value = ints.f32_vec(CELLS)?;
    s.last_enemy_army_seen_timestep = ints.f32_vec(CELLS)?;
    s.opponent_army_history = ints.f32_vec(TEMPORAL_WINDOW)?;
    s.opponent_land_history = ints.f32_vec(TEMPORAL_WINDOW)?;
    s.temporal_step = ints.n()? as i32;
    Ok(s)
}

fn push_state(out: &mut Vec<i64>, s: &AugState) {
    push_f32s(out, &s.army_stack);
    push_f32s(out, &s.enemy_stack);
    push_f32s(out, &s.last_army);
    push_f32s(out, &s.last_enemy_army);
    push_bools(out, &s.castles);
    push_bools(out, &s.generals);
    push_bools(out, &s.mountains);
    push_bools(out, &s.seen);
    push_bools(out, &s.enemy_seen);
    push_f32s(out, &s.last_enemy_army_seen_value);
    push_f32s(out, &s.last_enemy_army_seen_timestep);
    push_f32s(out, &s.opponent_army_history);
    push_f32s(out, &s.opponent_land_history);
    out.push(s.temporal_step as i64);
}

/// Read one wire-shaped frame: `turn my_land my_army opp_land opp_army` then
/// the three grids, into an `Observation` with dims already set.
fn read_frame(ints: &mut Ints, obs: &mut Observation) -> Result<(), String> {
    obs.turn = ints.n()? as i32;
    obs.my_land = ints.n()? as i32;
    obs.my_army = ints.n()? as i32;
    obs.opp_land = ints.n()? as i32;
    obs.opp_army = ints.n()? as i32;
    let n = obs.h * obs.w;
    for grid in [&mut obs.type_grid, &mut obs.owner_grid, &mut obs.army_grid] {
        for v in grid.iter_mut().take(n) {
            *v = ints.n()? as i32;
        }
    }
    Ok(())
}

/// Compute raw + cost + masks for one frame — the shared front half of the
/// per-turn path.
struct FrameDerived {
    raw: Vec<f32>,
    cost: Vec<i32>,
    move_mask: Vec<bool>,
    build_mask: Vec<bool>,
}

fn derive_frame(obs: &Observation) -> FrameDerived {
    let mut d = FrameDerived {
        raw: Vec::new(),
        cost: Vec::new(),
        move_mask: Vec::new(),
        build_mask: Vec::new(),
    };
    frame_to_raw(obs, &mut d.raw);
    build_cost_from_raw(&d.raw, obs.h, obs.w, &mut d.cost);
    compute_valid_move_mask(&d.raw, obs.h, obs.w, &mut d.move_mask);
    compute_build_mask_from_raw(&d.raw, obs.h, obs.w, &d.cost, &mut d.build_mask);
    d
}

fn artifact_dir() -> PathBuf {
    crate::artifact_dir()
}

pub fn run(surface: &str) -> Result<(), String> {
    let stdin = std::io::stdin();
    let mut ints = Ints::read_all(&mut stdin.lock())?;
    let cases = ints.usize()?;
    let mut out: Vec<i64> = Vec::new();

    let mut net: Option<Net> = None;
    let load_net = || -> Result<Net, String> { Net::load(&artifact_dir()) };

    for _ in 0..cases {
        match surface {
            "raw" => {
                let (h, w) = (ints.usize()?, ints.usize()?);
                let mut obs = Observation::with_dims(h, w);
                read_frame(&mut ints, &mut obs)?;
                let mut raw = Vec::new();
                frame_to_raw(&obs, &mut raw);
                push_f32s(&mut out, &raw);
            }
            "cost" => {
                let (h, w) = (ints.usize()?, ints.usize()?);
                let raw = ints.f32_vec(N_RAW_CHANNELS * h * w)?;
                let mut cost = Vec::new();
                build_cost_from_raw(&raw, h, w, &mut cost);
                out.extend(cost.iter().map(|&v| v as i64));
            }
            "mask" => {
                let (h, w) = (ints.usize()?, ints.usize()?);
                let raw = ints.f32_vec(N_RAW_CHANNELS * h * w)?;
                let mut cost = Vec::new();
                build_cost_from_raw(&raw, h, w, &mut cost);
                let mut move_mask = Vec::new();
                compute_valid_move_mask(&raw, h, w, &mut move_mask);
                let mut build_mask = Vec::new();
                compute_build_mask_from_raw(&raw, h, w, &cost, &mut build_mask);
                push_bools(&mut out, &move_mask);
                push_bools(&mut out, &build_mask);
            }
            "obs" => {
                let (h, w) = (ints.usize()?, ints.usize()?);
                let raw = ints.f32_vec(N_RAW_CHANNELS * h * w)?;
                let state = read_state(&mut ints)?;
                let mut cost = Vec::new();
                build_cost_from_raw(&raw, h, w, &mut cost);
                let mut next = AugState::zeros();
                let mut scratch = AugScratch::new();
                let mut aug = vec![0f32; N_CHANNELS * CELLS];
                augment_obs(&raw, h, w, &cost, &state, &mut next, &mut scratch, &mut aug);
                push_f32s(&mut out, &aug);
                push_state(&mut out, &next);
            }
            "forward" | "decide" => {
                let (h, w) = (ints.usize()?, ints.usize()?);
                let mut aug = ints.f32_vec(N_CHANNELS * CELLS)?;
                let move_mask = ints.bool_vec(h * w * 4)?;
                let build_mask = ints.bool_vec(h * w)?;
                let temporal = ints.f32_vec(2 * TEMPORAL_WINDOW)?;
                if net.is_none() {
                    net = Some(load_net()?);
                }
                let net = net.as_ref().unwrap();
                normalize_observations(&mut aug);
                let mut penalties = vec![0f32; N_ACTION_CHANNELS * CELLS];
                prepare_action_mask(&move_mask, &build_mask, h, w, &mut penalties);
                let fwd = net.forward(&aug, &penalties, &temporal)?;
                if surface == "forward" {
                    push_f32s(&mut out, &fwd.logits);
                    out.push(fwd.value.to_bits() as i64);
                    push_f32s(&mut out, &fwd.value_bins);
                } else {
                    let idx = argmax(&fwd.logits);
                    let a = decode_action(idx);
                    out.push(idx as i64);
                    out.extend([
                        a.pass_field as i64,
                        a.row as i64,
                        a.col as i64,
                        a.dir as i64,
                        a.is_half as i64,
                    ]);
                }
            }
            "log1p" => {
                // f32 bits in -> `log1p(x) * (1/5)` bits out (channel 21's
                // exact computation), for the exhaustive domain sweep.
                let n = ints.usize()?;
                for _ in 0..n {
                    let x = ints.f32_bits()?;
                    out.push((crate::xla_math::log1p(x) * crate::xla_math::RECIP_5).to_bits()
                        as i64);
                }
            }
            "sequence" => {
                // One whole game through the Rust state machine: per turn the
                // FNV hash of the augmented tensor, then the final state.
                let (h, w, turns) = (ints.usize()?, ints.usize()?, ints.usize()?);
                let mut obs = Observation::with_dims(h, w);
                let mut state = AugState::zeros();
                let mut next = AugState::zeros();
                let mut scratch = AugScratch::new();
                let mut aug = vec![0f32; N_CHANNELS * CELLS];
                for _ in 0..turns {
                    read_frame(&mut ints, &mut obs)?;
                    let d = derive_frame(&obs);
                    augment_obs(&d.raw, h, w, &d.cost, &state, &mut next, &mut scratch, &mut aug);
                    std::mem::swap(&mut state, &mut next);
                    out.push(crc32(&aug));
                }
                push_state(&mut out, &state);
            }
            other => return Err(format!("unknown parity surface {other:?}")),
        }
    }

    if ints.pos != ints.values.len() {
        return Err(format!(
            "unconsumed input: {} of {} integers left",
            ints.values.len() - ints.pos,
            ints.values.len()
        ));
    }

    let stdout = std::io::stdout();
    let mut writer = std::io::BufWriter::new(stdout.lock());
    let mut buf = String::with_capacity(1 << 20);
    for (i, v) in out.iter().enumerate() {
        buf.push_str(&v.to_string());
        buf.push(if (i + 1) % 16 == 0 { '\n' } else { ' ' });
        if buf.len() > (1 << 20) - 32 {
            writer.write_all(buf.as_bytes()).map_err(|e| e.to_string())?;
            buf.clear();
        }
    }
    buf.push('\n');
    writer.write_all(buf.as_bytes()).map_err(|e| e.to_string())?;
    writer.flush().map_err(|e| e.to_string())?;
    Ok(())
}

// NUM_BINS and N_LOGITS are part of the forward surface's contract; assert
// they stay in sync with the layouts above.
const _: () = assert!(N_LOGITS == N_ACTION_CHANNELS * CELLS);
const _: () = assert!(NUM_BINS == 128);
