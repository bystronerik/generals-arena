//! The Morpheus network, written out as fixed-shape kernels.
//!
//! Port of `bots/morpheus/network.py`: a 3×3 stem into 12 inverted residual
//! blocks (`64 → 128 → 64`, pointwise expand, depthwise 3×3 with dilation
//! cycle 1/2/4, pointwise project), GroupNorm(8) and ReLU6 throughout, then
//! eleven heads over a shared trunk. 249,316 parameters, ~0.2 GFLOPs per
//! forward at batch 1.
//!
//! **Why this is hand-written rather than a framework graph.** Plan §3 ranked
//! candle first and a bespoke kernel last, on the reasoning that batch-1
//! latency was dispatch-overhead-bound and `gemm` would recover most of it.
//! Measured at M3, both premises were wrong: candle costs 4.4× TorchScript at
//! batch 1 and 6.0× at batch 4, tract 1.4× and 2.3×, and candle's reason is
//! specific — it implements grouped convolution by looping over groups, so the
//! depthwise layer becomes 128 separate convolutions at 1.14 ms per call
//! against the 0.28 ms the same layer costs here. §3's decision rule is
//! measurement-driven, and the measurement put this file at the top of the
//! ladder. Figures:
//! `docs/research/measurements/morpheus-rs-inference-bench.md`.
//!
//! **Layout.** Activations are channel-major planes of `STRIDE` floats, where
//! `STRIDE` is 441 real cells rounded up to 448 so every GEMM's `n` is a whole
//! number of register tiles. The seven pad columns start at zero and stay
//! there: the stem's im2col writes zeros into them, every later op is
//! pointwise or reads by (row, col), and the two reductions that could see
//! them — GroupNorm's statistics and the masked global pool — both loop over
//! the 441 real cells. Nothing anywhere divides by the padded width.
//!
//! **One sample at a time.** A batch of 4 runs the graph four times rather
//! than widening `n` to 1792. It costs a little arithmetic intensity and buys
//! constant memory, a single set of shapes, and no allocation on the per-turn
//! path (§6) — and it still beats the batched TorchScript it replaces, because
//! four sequential forwards here are cheaper than one batch-4 forward there.

use crate::gemm::{gemm_bias, matvec, NR};
use crate::safetensors::SafeTensors;

pub const BOARD: usize = 21;
pub const CELLS: usize = BOARD * BOARD;
/// Cells padded up to a whole number of GEMM column tiles.
pub const STRIDE: usize = CELLS.next_multiple_of(NR);
pub const IN_CHANNELS: usize = 49;
pub const TRUNK: usize = 64;
pub const EXPANSION: usize = 128;
pub const POLICY_CHANNELS: usize = 9;
pub const N_ARMY_BINS: usize = 16;
pub const N_BLOCKS: usize = 12;
pub const GROUP_NORM_GROUPS: usize = 8;
pub const FEAT: usize = TRUNK * 2;
pub const DILATION_CYCLE: [usize; 3] = [1, 2, 4];
/// The widest dilation in the cycle. Sizes the depthwise halo; a cycle that
/// grew past it would index out of the padded row and panic rather than read
/// the wrong cell.
const MAX_DILATION: usize = 4;
/// `nn.GroupNorm`'s default. The export does not record it, so it is pinned
/// here and a mutation that moves it is caught by the `net` parity surface.
const GROUP_NORM_EPS: f64 = 1e-5;
/// `masked_fill(-1e9)` before the max, matching `masked_global_features`.
const MASK_FILL: f32 = -1e9;

/// Flat policy layout: `channel · 441 + row · 21 + col`, pass last.
pub const N_ACTIONS: usize = POLICY_CHANNELS * CELLS + 1;
pub const PASS_INDEX: usize = N_ACTIONS - 1;

struct Block {
    pw_expand: Vec<f32>,  // [EXPANSION][TRUNK]
    gn_expand: (Vec<f32>, Vec<f32>),
    dw: Vec<f32>, // [EXPANSION][9]
    gn_dw: (Vec<f32>, Vec<f32>),
    pw_project: Vec<f32>, // [TRUNK][EXPANSION]
    gn_project: (Vec<f32>, Vec<f32>),
    dilation: usize,
}

struct Head1x1 {
    weight: Vec<f32>, // [out][TRUNK]
    bias: Vec<f32>,
    out: usize,
}

struct HeadLinear {
    weight: Vec<f32>, // [out][FEAT]
    bias: Vec<f32>,
    out: usize,
}

/// Every head the full artifact exposes, in `MorpheusOutput` order.
///
/// Spatial heads are `[channels][441]` in the flat policy layout; scalar heads
/// are as many floats as they have outputs. Which of them a forward pass
/// actually writes depends on its [`Heads`] argument — the rest keep whatever
/// the previous call left there, so a caller must not read what its entry
/// point did not compute.
#[derive(Clone)]
pub struct Output {
    pub policy: Vec<f32>,
    pub pass_logit: f32,
    pub wdl_logits: [f32; 3],
    pub hidden_owner: Vec<f32>,
    pub enemy_army_bins: Vec<f32>,
    pub enemy_general: Vec<f32>,
    pub hidden_castle: Vec<f32>,
    pub land_margin: f32,
    pub army_margin: f32,
    pub castle_margin: f32,
    pub turns_to_termination: f32,
}

impl Output {
    pub fn new() -> Self {
        Self {
            policy: vec![0.0; POLICY_CHANNELS * CELLS],
            pass_logit: 0.0,
            wdl_logits: [0.0; 3],
            hidden_owner: vec![0.0; CELLS],
            enemy_army_bins: vec![0.0; N_ARMY_BINS * CELLS],
            enemy_general: vec![0.0; CELLS],
            hidden_castle: vec![0.0; CELLS],
            land_margin: 0.0,
            army_margin: 0.0,
            castle_margin: 0.0,
            turns_to_termination: 0.0,
        }
    }

    /// `flatten_policy_logits`: the 3969 spatial logits with pass appended.
    pub fn flat_logits(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(N_ACTIONS);
        out.extend_from_slice(&self.policy);
        out.push(self.pass_logit);
        out
    }
}

/// Scratch the forward pass needs, allocated once and reused every call.
///
/// §6 asks for zero heap allocation on the per-turn path; this is how the
/// network keeps that promise. `im2col` is the big one at 790 KB, and it is
/// the reason the stem is one GEMM instead of nine.
struct Scratch {
    input: Vec<f32>,   // [IN_CHANNELS][STRIDE]
    im2col: Vec<f32>,  // [IN_CHANNELS*9][STRIDE]
    trunk: Vec<f32>,   // [TRUNK][STRIDE]
    residual: Vec<f32>, // [TRUNK][STRIDE]
    wide_a: Vec<f32>,  // [EXPANSION][STRIDE]
    wide_b: Vec<f32>,  // [EXPANSION][STRIDE]
    feat: Vec<f32>,    // [FEAT]
    spatial: Vec<f32>, // [N_ARMY_BINS][STRIDE] — widest spatial head
}

impl Scratch {
    fn new() -> Self {
        Self {
            input: vec![0.0; IN_CHANNELS * STRIDE],
            im2col: vec![0.0; IN_CHANNELS * 9 * STRIDE],
            trunk: vec![0.0; TRUNK * STRIDE],
            residual: vec![0.0; TRUNK * STRIDE],
            wide_a: vec![0.0; EXPANSION * STRIDE],
            wide_b: vec![0.0; EXPANSION * STRIDE],
            feat: vec![0.0; FEAT],
            spatial: vec![0.0; N_ARMY_BINS * STRIDE],
        }
    }
}

/// Which heads a forward pass computes. The Python ships three TorchScript
/// files for this; one graph with a switch is the same saving without three
/// artifacts to keep in step.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Heads {
    /// `forward_policy`: belief proposal and enemy priors.
    Policy,
    /// `forward_policy_wdl`: root and leaf evaluation.
    PolicyWdl,
    /// `forward`: everything, for recovery and diagnostics.
    All,
}

pub struct Network {
    stem: Vec<f32>, // [TRUNK][IN_CHANNELS*9], im2col-ordered
    gn_stem: (Vec<f32>, Vec<f32>),
    blocks: Vec<Block>,
    policy: Head1x1,
    hidden_owner: Head1x1,
    enemy_army_bins: Head1x1,
    enemy_general: Head1x1,
    hidden_castle: Head1x1,
    pass_fc: HeadLinear,
    wdl: HeadLinear,
    land_margin: HeadLinear,
    army_margin: HeadLinear,
    castle_margin: HeadLinear,
    turns_to_termination: HeadLinear,
    scratch: Scratch,
}

fn head_1x1(st: &SafeTensors, name: &str, out: usize) -> Result<Head1x1, String> {
    Ok(Head1x1 {
        weight: st.get_shaped(&format!("{name}.weight"), &[out, TRUNK, 1, 1])?.to_vec(),
        bias: st.get_shaped(&format!("{name}.bias"), &[out])?.to_vec(),
        out,
    })
}

fn head_linear(st: &SafeTensors, name: &str, out: usize) -> Result<HeadLinear, String> {
    Ok(HeadLinear {
        weight: st.get_shaped(&format!("{name}.weight"), &[out, FEAT])?.to_vec(),
        bias: st.get_shaped(&format!("{name}.bias"), &[out])?.to_vec(),
        out,
    })
}

fn group_norm_params(
    st: &SafeTensors,
    name: &str,
    channels: usize,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    Ok((
        st.get_shaped(&format!("{name}.weight"), &[channels])?.to_vec(),
        st.get_shaped(&format!("{name}.bias"), &[channels])?.to_vec(),
    ))
}

impl Network {
    pub fn from_safetensors(st: &SafeTensors) -> Result<Self, String> {
        // Conv2d stores `[out][in][ky][kx]`; the stem GEMM wants the reduction
        // axis laid out exactly as im2col builds it, `in·9 + ky·3 + kx`. Those
        // are the same order, so this is a copy rather than a transpose — but
        // it is written out because the two orders agreeing is a fact worth
        // stating rather than one worth rediscovering during a parity failure.
        let stem_raw = st.get_shaped("stem.weight", &[TRUNK, IN_CHANNELS, 3, 3])?;
        let mut stem = vec![0f32; TRUNK * IN_CHANNELS * 9];
        for oc in 0..TRUNK {
            for ic in 0..IN_CHANNELS {
                for k in 0..9 {
                    stem[oc * IN_CHANNELS * 9 + ic * 9 + k] =
                        stem_raw[(oc * IN_CHANNELS + ic) * 9 + k];
                }
            }
        }

        let mut blocks = Vec::with_capacity(N_BLOCKS);
        for i in 0..N_BLOCKS {
            let p = format!("blocks.{i}");
            blocks.push(Block {
                pw_expand: st
                    .get_shaped(&format!("{p}.pw_expand.weight"), &[EXPANSION, TRUNK, 1, 1])?
                    .to_vec(),
                gn_expand: group_norm_params(st, &format!("{p}.gn_expand"), EXPANSION)?,
                dw: st
                    .get_shaped(&format!("{p}.dw.weight"), &[EXPANSION, 1, 3, 3])?
                    .to_vec(),
                gn_dw: group_norm_params(st, &format!("{p}.gn_dw"), EXPANSION)?,
                pw_project: st
                    .get_shaped(&format!("{p}.pw_project.weight"), &[TRUNK, EXPANSION, 1, 1])?
                    .to_vec(),
                gn_project: group_norm_params(st, &format!("{p}.gn_project"), TRUNK)?,
                dilation: DILATION_CYCLE[i % DILATION_CYCLE.len()],
            });
        }

        Ok(Self {
            stem,
            gn_stem: group_norm_params(st, "gn_stem", TRUNK)?,
            blocks,
            policy: head_1x1(st, "policy", POLICY_CHANNELS)?,
            hidden_owner: head_1x1(st, "hidden_owner", 1)?,
            enemy_army_bins: head_1x1(st, "enemy_army_bins", N_ARMY_BINS)?,
            enemy_general: head_1x1(st, "enemy_general", 1)?,
            hidden_castle: head_1x1(st, "hidden_castle", 1)?,
            pass_fc: head_linear(st, "pass_fc", 1)?,
            wdl: head_linear(st, "wdl", 3)?,
            land_margin: head_linear(st, "land_margin", 1)?,
            army_margin: head_linear(st, "army_margin", 1)?,
            castle_margin: head_linear(st, "castle_margin", 1)?,
            turns_to_termination: head_linear(st, "turns_to_termination", 1)?,
            scratch: Scratch::new(),
        })
    }

    /// Run one sample. `x` is the 49×441 tensor in plane-major order.
    pub fn forward_into(&mut self, x: &[f32], heads: Heads, out: &mut Output) {
        assert_eq!(x.len(), IN_CHANNELS * CELLS, "tensor is 49 planes of 441");
        // Widen 441-cell planes into 448-cell strided planes. The pad columns
        // were zeroed at construction and nothing below writes them.
        for c in 0..IN_CHANNELS {
            self.scratch.input[c * STRIDE..c * STRIDE + CELLS]
                .copy_from_slice(&x[c * CELLS..(c + 1) * CELLS]);
        }

        im2col_3x3(&self.scratch.input, IN_CHANNELS, 1, &mut self.scratch.im2col);
        gemm_bias(
            TRUNK,
            IN_CHANNELS * 9,
            STRIDE,
            &self.stem,
            &self.scratch.im2col,
            None,
            STRIDE,
            &mut self.scratch.trunk,
        );
        group_norm(&mut self.scratch.trunk, TRUNK, &self.gn_stem.0, &self.gn_stem.1);
        relu6(&mut self.scratch.trunk);

        for bi in 0..self.blocks.len() {
            // Split the borrow: the block's weights are read while scratch is
            // written. Index rather than iterate, so both borrows are fields.
            let block = &self.blocks[bi];
            let s = &mut self.scratch;
            s.residual.copy_from_slice(&s.trunk);

            gemm_bias(EXPANSION, TRUNK, STRIDE, &block.pw_expand, &s.trunk, None, STRIDE, &mut s.wide_a);
            group_norm(&mut s.wide_a, EXPANSION, &block.gn_expand.0, &block.gn_expand.1);
            relu6(&mut s.wide_a);

            depthwise_3x3(&s.wide_a, EXPANSION, block.dilation, &block.dw, &mut s.wide_b);
            group_norm(&mut s.wide_b, EXPANSION, &block.gn_dw.0, &block.gn_dw.1);
            relu6(&mut s.wide_b);

            gemm_bias(TRUNK, EXPANSION, STRIDE, &block.pw_project, &s.wide_b, None, STRIDE, &mut s.trunk);
            group_norm(&mut s.trunk, TRUNK, &block.gn_project.0, &block.gn_project.1);
            for i in 0..s.trunk.len() {
                s.trunk[i] += s.residual[i];
            }
            relu6(&mut s.trunk);
        }

        self.masked_global_features();

        self.spatial_head_into(Head::Policy, &mut out.policy);
        out.pass_logit = self.linear_head(&self.pass_fc)[0];
        if heads == Heads::Policy {
            return;
        }
        let wdl = self.linear_head(&self.wdl);
        out.wdl_logits = [wdl[0], wdl[1], wdl[2]];
        if heads == Heads::PolicyWdl {
            return;
        }
        self.spatial_head_into(Head::HiddenOwner, &mut out.hidden_owner);
        self.spatial_head_into(Head::EnemyArmyBins, &mut out.enemy_army_bins);
        self.spatial_head_into(Head::EnemyGeneral, &mut out.enemy_general);
        self.spatial_head_into(Head::HiddenCastle, &mut out.hidden_castle);
        out.land_margin = self.linear_head(&self.land_margin)[0];
        out.army_margin = self.linear_head(&self.army_margin)[0];
        out.castle_margin = self.linear_head(&self.castle_margin)[0];
        out.turns_to_termination = self.linear_head(&self.turns_to_termination)[0];
    }

    pub fn forward(&mut self, x: &[f32], heads: Heads) -> Output {
        let mut out = Output::new();
        self.forward_into(x, heads, &mut out);
        out
    }

    /// Per-stage wall time for one forward, in the order the stages run.
    ///
    /// Not a debug aid to be deleted: M7 re-derives every deadline knob from a
    /// per-component cost table, and the whole point of the rewrite is that
    /// those costs change. Keeping the breakdown in the shipped binary means
    /// the table can be re-measured on the deployment host rather than
    /// inferred from a development laptop. It also earned its place
    /// immediately — the first run showed GroupNorm outweighing every
    /// convolution in the graph.
    pub fn profile_forward(&mut self, x: &[f32], iters: usize) -> Vec<(&'static str, f64)> {
        use std::time::Instant;
        let mut totals = [0f64; 6];
        for _ in 0..iters {
            for c in 0..IN_CHANNELS {
                self.scratch.input[c * STRIDE..c * STRIDE + CELLS]
                    .copy_from_slice(&x[c * CELLS..(c + 1) * CELLS]);
            }
            let t = Instant::now();
            im2col_3x3(&self.scratch.input, IN_CHANNELS, 1, &mut self.scratch.im2col);
            totals[0] += t.elapsed().as_secs_f64();

            let t = Instant::now();
            gemm_bias(
                TRUNK,
                IN_CHANNELS * 9,
                STRIDE,
                &self.stem,
                &self.scratch.im2col,
                None,
                STRIDE,
                &mut self.scratch.trunk,
            );
            totals[1] += t.elapsed().as_secs_f64();
            group_norm(&mut self.scratch.trunk, TRUNK, &self.gn_stem.0, &self.gn_stem.1);
            relu6(&mut self.scratch.trunk);

            // The real order, timed in place. An earlier version grouped the
            // three GroupNorms together and ran the depthwise layer on
            // un-normalised activations; it happened to report the same
            // totals, but a profiler that reorders the program is profiling a
            // different program and there is no reason to keep one.
            for bi in 0..self.blocks.len() {
                let block = &self.blocks[bi];
                let s = &mut self.scratch;
                s.residual.copy_from_slice(&s.trunk);

                let t = Instant::now();
                gemm_bias(EXPANSION, TRUNK, STRIDE, &block.pw_expand, &s.trunk, None, STRIDE, &mut s.wide_a);
                totals[2] += t.elapsed().as_secs_f64();
                let t = Instant::now();
                group_norm(&mut s.wide_a, EXPANSION, &block.gn_expand.0, &block.gn_expand.1);
                totals[4] += t.elapsed().as_secs_f64();
                let t = Instant::now();
                relu6(&mut s.wide_a);
                totals[5] += t.elapsed().as_secs_f64();

                let t = Instant::now();
                depthwise_3x3(&s.wide_a, EXPANSION, block.dilation, &block.dw, &mut s.wide_b);
                totals[3] += t.elapsed().as_secs_f64();
                let t = Instant::now();
                group_norm(&mut s.wide_b, EXPANSION, &block.gn_dw.0, &block.gn_dw.1);
                totals[4] += t.elapsed().as_secs_f64();
                let t = Instant::now();
                relu6(&mut s.wide_b);
                totals[5] += t.elapsed().as_secs_f64();

                let t = Instant::now();
                gemm_bias(TRUNK, EXPANSION, STRIDE, &block.pw_project, &s.wide_b, None, STRIDE, &mut s.trunk);
                totals[2] += t.elapsed().as_secs_f64();
                let t = Instant::now();
                group_norm(&mut s.trunk, TRUNK, &block.gn_project.0, &block.gn_project.1);
                totals[4] += t.elapsed().as_secs_f64();
                let t = Instant::now();
                for i in 0..s.trunk.len() {
                    s.trunk[i] += s.residual[i];
                }
                relu6(&mut s.trunk);
                totals[5] += t.elapsed().as_secs_f64();
            }
        }
        let scale = 1e3 / iters as f64;
        ["im2col", "stem_gemm", "pointwise", "depthwise", "group_norm", "elementwise"]
            .iter()
            .zip(totals)
            .map(|(name, total)| (*name, total * scale))
            .collect()
    }

    /// `masked_global_features`: masked mean and masked max, concatenated.
    ///
    /// The mask is input plane 0 (the board mask), compared at 0.5 exactly as
    /// the Python does, and the `denom == 0` guards are kept: a board with no
    /// live cells cannot occur in a match, but the Python defines an answer for
    /// it and a port that defines a different one is a port that diverges on
    /// the first malformed frame.
    fn masked_global_features(&mut self) {
        let mask = &self.scratch.input[0..CELLS];
        let mut denom = 0f64;
        for &m in mask.iter() {
            denom += m as f64;
        }
        let safe = if denom > 0.0 { denom } else { 1.0 };
        for c in 0..TRUNK {
            let plane = &self.scratch.trunk[c * STRIDE..c * STRIDE + CELLS];
            let mut sum = 0f64;
            let mut mx = f32::NEG_INFINITY;
            for (p, &v) in plane.iter().enumerate() {
                sum += (v * mask[p]) as f64;
                let candidate = if mask[p] < 0.5 { MASK_FILL } else { v };
                if candidate > mx {
                    mx = candidate;
                }
            }
            self.scratch.feat[c] = (sum / safe) as f32;
            self.scratch.feat[TRUNK + c] = if denom > 0.0 { mx } else { 0.0 };
        }
    }

    fn spatial_head_into(&mut self, which: Head, out: &mut [f32]) {
        let head = match which {
            Head::Policy => &self.policy,
            Head::HiddenOwner => &self.hidden_owner,
            Head::EnemyArmyBins => &self.enemy_army_bins,
            Head::EnemyGeneral => &self.enemy_general,
            Head::HiddenCastle => &self.hidden_castle,
        };
        gemm_bias(
            head.out,
            TRUNK,
            STRIDE,
            &head.weight,
            &self.scratch.trunk,
            Some(&head.bias),
            STRIDE,
            &mut self.scratch.spatial,
        );
        for c in 0..head.out {
            out[c * CELLS..(c + 1) * CELLS]
                .copy_from_slice(&self.scratch.spatial[c * STRIDE..c * STRIDE + CELLS]);
        }
    }

    fn linear_head(&self, head: &HeadLinear) -> [f32; 3] {
        let mut y = [0f32; 3];
        matvec(
            head.out,
            FEAT,
            &head.weight,
            &self.scratch.feat,
            &head.bias,
            &mut y[..head.out],
        );
        y
    }
}

enum Head {
    Policy,
    HiddenOwner,
    EnemyArmyBins,
    EnemyGeneral,
    HiddenCastle,
}

/// Lower a `channels × 21 × 21` plane stack into `channels·9 × 448` im2col
/// rows, zero-padded, so the stem is one GEMM instead of nine accumulating
/// ones.
fn im2col_3x3(src: &[f32], channels: usize, dilation: usize, dst: &mut [f32]) {
    let d = dilation as isize;
    for c in 0..channels {
        for ky in 0..3usize {
            for kx in 0..3usize {
                let row = (c * 9 + ky * 3 + kx) * STRIDE;
                dst[row..row + STRIDE].fill(0.0);
                let dy = (ky as isize - 1) * d;
                let dx = (kx as isize - 1) * d;
                for r in 0..BOARD as isize {
                    let sr = r + dy;
                    if sr < 0 || sr >= BOARD as isize {
                        continue;
                    }
                    // Column window where both source and destination are on
                    // the board; outside it the tap contributes a pad zero.
                    let lo = (-dx).max(0) as usize;
                    let hi = (BOARD as isize - dx).min(BOARD as isize) as usize;
                    if lo >= hi {
                        continue;
                    }
                    let dst_base = row + r as usize * BOARD + lo;
                    let src_base = c * STRIDE + sr as usize * BOARD + (lo as isize + dx) as usize;
                    dst[dst_base..dst_base + (hi - lo)]
                        .copy_from_slice(&src[src_base..src_base + (hi - lo)]);
                }
            }
        }
    }
}

/// Depthwise 3×3 with dilation: one 9-tap filter per channel, zero padded.
///
/// **The plane is copied into a haloed frame so each tap is one long axpy.**
/// The board is 21×21; the frame is 29×29, the board with `MAX_DILATION` cells
/// of zero on every side. In that frame a tap is a *uniform shift*: output
/// cell `i` reads `i + dy·29 + dx`, and off-board reads land in the halo,
/// which is zero, which is exactly what the padding means. So all nine taps
/// run over one contiguous 601-element span instead of 21 rows of 21.
///
/// The 601 covers a little more than the 441 real cells — the halo columns
/// between rows come along for the ride and are dropped on the way out. That
/// is 36% wasted arithmetic traded for vector length: 21 floats is five
/// vectors and a remainder, and the remainder was the problem.
///
/// **It pays on x86 and does nothing on arm64.** The layer goes 1.52 ms →
/// 0.66 ms on one x86 core — a third of the forward down to a seventh, and the
/// whole batch-1 forward from 5.07 ms to 4.09 ms — while the M3 Pro moves
/// 0.284 ms → 0.287 ms. NEON was already handling five-and-a-bit vectors per
/// row fine. x86 is the platform the competition runs on, so this ships; the
/// laptop would have voted to skip it, and did, until someone measured the
/// other machine.
///
/// The frame is sized by `MAX_DILATION`, and the arithmetic below depends on
/// that: the widest tap reads `4·29 + 4 = 120` cells either side of the span,
/// and the span plus that reach is exactly 841. A wider dilation cycle needs a
/// wider halo, which is why `MAX_DILATION` is asserted rather than assumed.
/// The assert is documentation; the real guard is that an over-wide dilation
/// drives `base` negative, and the slice index panics rather than quietly
/// reading the wrong cells.
///
/// A note for whoever profiles this next: **per-stage numbers here are not
/// stable across builds.** Earlier in M3 the previous implementation measured
/// 1.10 ms and then 0.28 ms on the same host from edits *elsewhere in the
/// crate*; three candidate causes were tested and refuted, leaving whole-crate
/// LTO (`lto = "fat"`, `codegen-units = 1`) as the unproven remainder. Compare
/// implementations in the same build, and trust the whole-forward figure.
fn depthwise_3x3(src: &[f32], channels: usize, dilation: usize, weight: &[f32], dst: &mut [f32]) {
    /// Zero cells on each side of the board in the working frame.
    const HALO: usize = MAX_DILATION;
    /// Row stride of the haloed frame.
    const HW: usize = BOARD + 2 * HALO;
    /// Cells in the haloed frame.
    const FRAME: usize = HW * HW;
    /// Where the board's cell (0, 0) sits in the frame.
    const RUN_START: usize = HALO * HW + HALO;
    /// From the board's first cell through its last, halo columns included.
    const RUN_LEN: usize = (BOARD - 1) * HW + BOARD;

    debug_assert!(dilation <= MAX_DILATION, "halo is sized for {MAX_DILATION}");

    // Zeroed once for the whole call: every channel overwrites the interior
    // and nothing ever writes the halo, so it stays zero by construction.
    let mut frame = [0f32; FRAME];
    let mut acc = [0f32; FRAME];

    for c in 0..channels {
        for r in 0..BOARD {
            let from = c * STRIDE + r * BOARD;
            let to = RUN_START + r * HW;
            frame[to..to + BOARD].copy_from_slice(&src[from..from + BOARD]);
        }

        let w = &weight[c * 9..c * 9 + 9];
        for tap in 0..9usize {
            let dy = (tap / 3) as isize - 1;
            let dx = (tap % 3) as isize - 1;
            // Non-negative by construction: the most negative shift is
            // -(4·29 + 4) = -120, and the span starts at 120.
            let base = (RUN_START as isize
                + dy * dilation as isize * HW as isize
                + dx * dilation as isize) as usize;
            let read: &[f32; RUN_LEN] = frame[base..].first_chunk().unwrap();
            let write: &mut [f32; RUN_LEN] = acc[RUN_START..].first_chunk_mut().unwrap();
            let wv = w[tap];
            if tap == 0 {
                // Store rather than accumulate, which zeroes the span for free
                // — and, less obviously, stops the halo cells inside it from
                // carrying a previous channel's values forward.
                for i in 0..RUN_LEN {
                    write[i] = wv * read[i];
                }
            } else {
                for i in 0..RUN_LEN {
                    write[i] = wv.mul_add(read[i], write[i]);
                }
            }
        }

        for r in 0..BOARD {
            let from = RUN_START + r * HW;
            let to = c * STRIDE + r * BOARD;
            dst[to..to + BOARD].copy_from_slice(&acc[from..from + BOARD]);
        }
        // Pad columns are never read downstream, but leaving a previous call's
        // values there would make the buffer depend on history.
        dst[c * STRIDE + CELLS..(c + 1) * STRIDE].fill(0.0);
    }
}

/// `nn.GroupNorm(8, C)`: statistics over each group's channels *and* cells.
///
/// Accumulated in f64. That is not the width PyTorch uses, and it is the right
/// choice here for the reason M2's tensor port made the opposite one: the
/// tensor feeds a network trained on the Python's exact rounding, while this
/// is an internal reduction whose only contract is agreeing with TorchScript
/// to 1e-5. f64 sums are closer to the true value than a naive f32 loop, hence
/// closer to PyTorch's own cascaded summation than a naive f32 loop would be.
///
/// Only the 441 real cells enter the statistics; the seven pad columns per
/// plane are not board.
fn group_norm(buf: &mut [f32], channels: usize, weight: &[f32], bias: &[f32]) {
    /// Independent accumulator lanes. One `sum += x` chain over 7,056 values
    /// is latency-bound on the adder, not throughput-bound; eight chains let
    /// the pipeline stay full and let LLVM vectorize. Fixing this alone took
    /// GroupNorm from the most expensive stage in the graph to the third.
    const LANES: usize = 8;

    let per_group = channels / GROUP_NORM_GROUPS;
    for g in 0..GROUP_NORM_GROUPS {
        let lo = g * per_group;
        let mut sums = [0f64; LANES];
        let mut sqs = [0f64; LANES];
        for c in lo..lo + per_group {
            let plane = &buf[c * STRIDE..c * STRIDE + CELLS];
            let mut chunks = plane.chunks_exact(LANES);
            for chunk in &mut chunks {
                for i in 0..LANES {
                    let x = chunk[i] as f64;
                    sums[i] += x;
                    sqs[i] = x.mul_add(x, sqs[i]);
                }
            }
            // 441 is not a multiple of eight; the tail lands in lane 0.
            for &v in chunks.remainder() {
                let x = v as f64;
                sums[0] += x;
                sqs[0] = x.mul_add(x, sqs[0]);
            }
        }
        let sum: f64 = sums.iter().sum();
        let sum_sq: f64 = sqs.iter().sum();
        let n = (per_group * CELLS) as f64;
        let mean = sum / n;
        let var = (sum_sq / n - mean * mean).max(0.0);
        let inv_std = 1.0 / (var + GROUP_NORM_EPS).sqrt();
        for c in lo..lo + per_group {
            let scale = (inv_std * weight[c] as f64) as f32;
            let shift = (bias[c] as f64 - mean * inv_std * weight[c] as f64) as f32;
            for v in &mut buf[c * STRIDE..c * STRIDE + CELLS] {
                *v = v.mul_add(scale, shift);
            }
        }
    }
}

fn relu6(buf: &mut [f32]) {
    for v in buf.iter_mut() {
        *v = v.clamp(0.0, 6.0);
    }
}

/// `V = p_win − p_loss` from perspective-player WDL logits.
///
/// **In f32, then widened** — the width is part of the contract, exactly as it
/// is for the tensor builder. `torch.softmax` runs at the tensor's dtype, so
/// the Python's value is an f32 quantity that `.item()` widens on the way out,
/// and every decision the oracle has ever made was made on that rounding.
/// Computing it in f64 here is more accurate and disagrees in the eighth
/// decimal, which is a difference the `prior` surface can see and tier-3
/// decision parity would eventually trip over on a near-tie.
pub fn wdl_value(logits: [f32; 3]) -> f64 {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: [f32; 3] = [
        (logits[0] - max).exp(),
        (logits[1] - max).exp(),
        (logits[2] - max).exp(),
    ];
    let total = exps[0] + exps[1] + exps[2];
    ((exps[0] / total) - (exps[2] / total)) as f64
}

/// Search backup: the root keeps `V`, an enemy node negates it.
pub fn backup_value(logits: [f32; 3], from_root: bool) -> f64 {
    let v = wdl_value(logits);
    if from_root {
        v
    } else {
        -v
    }
}

/// Softmax over legal actions only; illegal entries stay exactly zero.
///
/// `masked_fill(finfo(f32).min)` then softmax then `* mask`, as in
/// `network.py`. The arithmetic is f32 and the *result* is f64, mirroring a
/// caller that softmaxes an f32 tensor and casts the answer — see
/// [`wdl_value`] for why the width is copied rather than improved on.
pub fn legal_normalized_policy(logits: &[f32], mask: &[bool]) -> Vec<f64> {
    assert_eq!(logits.len(), N_ACTIONS);
    assert_eq!(mask.len(), N_ACTIONS);
    let mut max = f32::NEG_INFINITY;
    for i in 0..N_ACTIONS {
        let v = if mask[i] { logits[i] } else { f32::MIN };
        if v > max {
            max = v;
        }
    }
    let mut exps = vec![0f32; N_ACTIONS];
    let mut total = 0f32;
    for i in 0..N_ACTIONS {
        let v = if mask[i] { logits[i] } else { f32::MIN };
        let e = (v - max).exp();
        exps[i] = e;
        total += e;
    }
    // The Python multiplies the softmax by the mask rather than reasoning
    // about `exp(finfo.min - max)`; an illegal action is exactly zero, not
    // merely negligible, and the harness checks that distinction.
    (0..N_ACTIONS)
        .map(|i| if mask[i] { (exps[i] / total) as f64 } else { 0.0 })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stride_covers_the_board_in_whole_tiles() {
        assert_eq!(STRIDE, 448);
        assert_eq!(STRIDE % NR, 0);
        assert!(STRIDE >= CELLS);
    }

    #[test]
    fn im2col_places_taps_and_pads_with_zero() {
        let mut src = vec![0f32; STRIDE];
        for r in 0..BOARD {
            for c in 0..BOARD {
                src[r * BOARD + c] = (r * BOARD + c) as f32;
            }
        }
        let mut dst = vec![0f32; 9 * STRIDE];
        im2col_3x3(&src, 1, 1, &mut dst);
        // Centre tap (ky=1, kx=1) is the identity.
        for p in 0..CELLS {
            assert_eq!(dst[4 * STRIDE + p], src[p]);
        }
        // Top-left tap at cell (0, 0) reads off-board and must be zero.
        assert_eq!(dst[0], 0.0);
        // Top-left tap at (5, 5) reads (4, 4).
        assert_eq!(dst[5 * BOARD + 5], src[4 * BOARD + 4]);
        // Pad columns are never written.
        for k in 0..9 {
            for p in CELLS..STRIDE {
                assert_eq!(dst[k * STRIDE + p], 0.0);
            }
        }
    }

    #[test]
    fn depthwise_matches_a_direct_per_cell_convolution() {
        for &dilation in &DILATION_CYCLE {
            let channels = 3;
            let mut src = vec![0f32; channels * STRIDE];
            let mut seed = 5u32;
            for c in 0..channels {
                for p in 0..CELLS {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    src[c * STRIDE + p] = ((seed >> 8) as f32 / 16_777_216.0) - 0.5;
                }
            }
            let weight: Vec<f32> = (0..channels * 9).map(|i| (i as f32 * 0.37).sin()).collect();
            let mut got = vec![0f32; channels * STRIDE];
            depthwise_3x3(&src, channels, dilation, &weight, &mut got);

            for c in 0..channels {
                for r in 0..BOARD as isize {
                    for col in 0..BOARD as isize {
                        let mut want = 0f32;
                        for ky in 0..3isize {
                            for kx in 0..3isize {
                                let sr = r + (ky - 1) * dilation as isize;
                                let sc = col + (kx - 1) * dilation as isize;
                                if sr < 0 || sr >= BOARD as isize || sc < 0 || sc >= BOARD as isize
                                {
                                    continue;
                                }
                                want += weight[c * 9 + (ky * 3 + kx) as usize]
                                    * src[c * STRIDE + sr as usize * BOARD + sc as usize];
                            }
                        }
                        let g = got[c * STRIDE + r as usize * BOARD + col as usize];
                        assert!((g - want).abs() < 1e-5, "c={c} r={r} col={col}: {g} vs {want}");
                    }
                }
            }
        }
    }

    #[test]
    fn group_norm_standardises_each_group_independently() {
        let channels = 16;
        let mut buf = vec![0f32; channels * STRIDE];
        for c in 0..channels {
            for p in 0..CELLS {
                // Each group gets a different scale and offset; after
                // normalising with unit weight and zero bias every group must
                // land on mean 0, variance 1.
                buf[c * STRIDE + p] = (c as f32 + 1.0) * ((p % 7) as f32) + c as f32;
            }
        }
        let weight = vec![1f32; channels];
        let bias = vec![0f32; channels];
        group_norm(&mut buf, channels, &weight, &bias);
        let per_group = channels / GROUP_NORM_GROUPS;
        for g in 0..GROUP_NORM_GROUPS {
            let mut sum = 0f64;
            let mut sq = 0f64;
            for c in g * per_group..(g + 1) * per_group {
                for &v in &buf[c * STRIDE..c * STRIDE + CELLS] {
                    sum += v as f64;
                    sq += (v as f64) * (v as f64);
                }
            }
            let n = (per_group * CELLS) as f64;
            assert!((sum / n).abs() < 1e-4, "group {g} mean {}", sum / n);
            assert!((sq / n - 1.0).abs() < 1e-3, "group {g} var {}", sq / n);
        }
    }

    #[test]
    fn wdl_value_is_win_minus_loss() {
        assert!((wdl_value([0.0, 0.0, 0.0]) - 0.0).abs() < 1e-12);
        let v = wdl_value([2.0, 0.0, -2.0]);
        assert!(v > 0.8, "{v}");
        assert!((backup_value([2.0, 0.0, -2.0], false) + v).abs() < 1e-12);
    }

    #[test]
    fn legal_policy_zeroes_illegal_actions_and_sums_to_one() {
        let mut logits = vec![0f32; N_ACTIONS];
        for (i, v) in logits.iter_mut().enumerate() {
            *v = (i % 13) as f32 * 0.1;
        }
        let mut mask = vec![false; N_ACTIONS];
        mask[7] = true;
        mask[9] = true;
        mask[PASS_INDEX] = true;
        let prior = legal_normalized_policy(&logits, &mask);
        let total: f64 = prior.iter().sum();
        // 1e-6, not 1e-12: the softmax divides in f32 on purpose (the width is
        // copied from `torch.softmax`, see the function's docs), so the masses
        // sum to one to about a float's worth of precision and no further.
        assert!((total - 1.0).abs() < 1e-6, "{total}");
        for i in 0..N_ACTIONS {
            if !mask[i] {
                assert_eq!(prior[i], 0.0);
            } else {
                assert!(prior[i] > 0.0);
            }
        }
    }
}
