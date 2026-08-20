//! The HistoryTransformer forward pass, dependency-free (port-plan §9, R1).
//!
//! Single sample, fixed shapes: 39×21×21 obs → 49 patch tokens of 351 →
//! 52 tokens × 384 through seven pre-norm blocks (MHSA 8 heads / head_dim 48,
//! then 384→1536 SiLU→384) → policy head (49×90, unpatchified to 10×21×21
//! masked logits) and value head (128 bins dotted with the exported
//! `bin_centers`). Deployment is float32 everywhere — the checkpoint's
//! `use_bf16` was already off in the Python sibling.
//!
//! This used to be candle (the port plan's first choice, and R1's tripwire
//! never fired on latency). What retired it was intake, not speed: the
//! 93-crate vendored build is what tournament qualification rejected, so the
//! whole graph — ~7 distinct GEMM shapes plus LayerNorm, softmax, and SiLU —
//! now runs on the in-house kernel in `gemm.rs`, the morpheus-rs precedent.
//! Op order and float width mirror the candle path site by site (reciprocal
//! multiplies where candle used them, plain multiply-then-add in LayerNorm);
//! the GEMM accumulation order necessarily differs, which is exactly what the
//! tier-2 parity tolerance in `tests/test_parity.py` exists to bound.
//!
//! Weights come from `artifact/model.safetensors` (written by
//! `tools/convert_artifact.py`); the loader refuses to start on a schema
//! tag, tensor name, shape, or dtype mismatch — the same guardrail the
//! Python loader gets from `tree_deserialise_leaves`. Every activation buffer
//! is preallocated at load into a `Scratch` behind a `RefCell`, so a turn
//! allocates nothing but its own `ForwardOut`.

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::path::Path;

use crate::board::obs::{CELLS, N_ACTION_CHANNELS, N_CHANNELS, PAD, TEMPORAL_WINDOW};
use crate::io::json;
use crate::nn::gemm::{gemm_bias, PackedB};
use crate::nn::safetensors::SafeTensors;

pub const TENSOR_SCHEMA: &str = "joe-net-v1";
pub const EMBED: usize = 384;
pub const DEPTH: usize = 7;
pub const N_HEAD: usize = 8;
pub const HEAD_DIM: usize = EMBED / N_HEAD; // 48
pub const PATCH: usize = 3;
pub const GRID_PATCHES: usize = PAD / PATCH; // 7
pub const N_PATCHES: usize = GRID_PATCHES * GRID_PATCHES; // 49
pub const PATCH_DIM: usize = N_CHANNELS * PATCH * PATCH; // 351
pub const N_TOKENS: usize = N_PATCHES + 3; // value + 2 temporal + patches
pub const NUM_BINS: usize = 128;
pub const FF_DIM: usize = 1536;
pub const POLICY_OUT: usize = N_ACTION_CHANNELS * PATCH * PATCH; // 90
pub const N_LOGITS: usize = N_ACTION_CHANNELS * CELLS; // 4410
const TEMPORAL_HIDDEN: usize = 512;

struct Linear {
    weights: PackedB, // (in, out), strip-major — packed once at load
    bias: Vec<f32>,   // (out,)
}

impl Linear {
    /// `y = x @ W^T + b` on a (rows, in) matrix into a (rows, out) buffer.
    fn forward_into(&self, rows: usize, x: &[f32], y: &mut [f32]) {
        gemm_bias(rows, x, &self.weights, &self.bias, y);
    }
}

struct LayerNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl LayerNorm {
    /// Equinox LayerNorm: biased variance, eps 1e-5, applied per token.
    /// Mirrors the candle op order: mean, centered, mean of squares,
    /// `sqrt(var + eps)`, divide, then multiply-add the affine — the divide
    /// stays a true divide and the affine stays an unfused `mul` + `add`.
    fn forward_into(&self, rows: usize, x: &[f32], y: &mut [f32]) {
        for r in 0..rows {
            let row = &x[r * EMBED..(r + 1) * EMBED];
            let out = &mut y[r * EMBED..(r + 1) * EMBED];
            // EMBED = 384 = 48 x 8: eight accumulator lanes so the
            // reductions vectorize instead of serializing on add latency.
            let mut lanes = [0f32; 8];
            for chunk in row.chunks_exact(8) {
                for (l, v) in chunk.iter().enumerate() {
                    lanes[l] += v;
                }
            }
            let sum: f32 = lanes.iter().sum();
            let mean = sum / EMBED as f32;
            let mut vlanes = [0f32; 8];
            for chunk in row.chunks_exact(8) {
                for (l, v) in chunk.iter().enumerate() {
                    let d = v - mean;
                    vlanes[l] = d.mul_add(d, vlanes[l]);
                }
            }
            let var_sum: f32 = vlanes.iter().sum();
            let denom = (var_sum / EMBED as f32 + 1e-5).sqrt();
            for j in 0..EMBED {
                out[j] = (row[j] - mean) / denom * self.weight[j] + self.bias[j];
            }
        }
    }
}

/// Vectorizable exp: Cephes-style range reduction and a 6th-order
/// polynomial, max relative error ~7.6e-8 on [-4, 4], clamped to the finite
/// range. Replaces libm `expf`, whose call blocks vectorization of every
/// loop it sits in.
#[inline]
fn exp_poly(x: f32) -> f32 {
    const LOG2E: f32 = 1.442_695_04;
    const LN2_HI: f32 = 0.693_359_375;
    const LN2_LO: f32 = -2.121_944_4e-4;
    let x = x.clamp(-87.3, 88.7);
    let kf = (x * LOG2E).round();
    let r = kf.mul_add(-LN2_HI, x);
    let r = kf.mul_add(-LN2_LO, r);
    let mut p = 1.987_569_1e-4f32;
    p = p.mul_add(r, 1.398_199_9e-3);
    p = p.mul_add(r, 8.333_452e-3);
    p = p.mul_add(r, 4.166_579_5e-2);
    p = p.mul_add(r, 1.666_666_6e-1);
    p = p.mul_add(r, 0.5);
    p = p.mul_add(r * r, r) + 1.0;
    f32::from_bits(((p.to_bits() as i32) + ((kf as i32) << 23)) as u32)
}

/// `x * sigmoid(x)`, in candle's f32 form `v / (1 + exp(-v))`.
fn silu_in_place(values: &mut [f32]) {
    for v in values.iter_mut() {
        *v /= 1.0 + exp_poly(-*v);
    }
}

/// Row softmax, the candle op order: subtract the max, exp, sum, divide.
fn softmax_in_place(row: &mut [f32]) {
    let mut max = f32::NEG_INFINITY;
    for &v in row.iter() {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0f32;
    for v in row.iter_mut() {
        *v = exp_poly(*v - max);
        sum += *v;
    }
    for v in row.iter_mut() {
        *v /= sum;
    }
}

struct Block {
    norm1: LayerNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    norm2: LayerNorm,
    ff1: Linear,
    ff2: Linear,
}

/// Preallocated activation buffers for the fixed shapes — one set, reused
/// every turn (port-plan §5's zero-per-turn-allocation target).
struct Scratch {
    patched: Vec<f32>,      // (49, 351)
    x: Vec<f32>,            // (52, 384) — the residual stream
    normed: Vec<f32>,       // (52, 384)
    q: Vec<f32>,            // (52, 384)
    k: Vec<f32>,            // (52, 384)
    v: Vec<f32>,            // (52, 384)
    ctx: Vec<f32>,          // (52, 384) — attention context, head-major columns
    proj: Vec<f32>,         // (52, 384) — out-proj / ff2 output before residual
    ff: Vec<f32>,           // (52, 1536)
    scores: Vec<f32>,       // (52, 52) — one head at a time
    qh: Vec<f32>,           // (52, 48) — one head's Q, packed contiguous
    kt: PackedB,            // (48, 52) — one head's K, strip-major for gemm
    vh: PackedB,            // (52, 48) — one head's V, strip-major for gemm
    ch: Vec<f32>,           // (52, 48) — one head's context before scatter
    zero_tokens: Vec<f32>,  // (52,) zero bias for the score gemm
    zero_head: Vec<f32>,    // (48,) zero bias for the context gemm
    hist: Vec<f32>,         // (2, 512) — scaled temporal windows
    hidden: Vec<f32>,       // (512,) — temporal MLP hidden
    tokens: Vec<f32>,       // (2, 384) — temporal tokens
    patch_logits: Vec<f32>, // (49, 90)
}

impl Scratch {
    fn new() -> Self {
        Self {
            patched: vec![0.0; N_PATCHES * PATCH_DIM],
            x: vec![0.0; N_TOKENS * EMBED],
            normed: vec![0.0; N_TOKENS * EMBED],
            q: vec![0.0; N_TOKENS * EMBED],
            k: vec![0.0; N_TOKENS * EMBED],
            v: vec![0.0; N_TOKENS * EMBED],
            ctx: vec![0.0; N_TOKENS * EMBED],
            proj: vec![0.0; N_TOKENS * EMBED],
            ff: vec![0.0; N_TOKENS * FF_DIM],
            scores: vec![0.0; N_TOKENS * N_TOKENS],
            qh: vec![0.0; N_TOKENS * HEAD_DIM],
            kt: PackedB::zeroed(HEAD_DIM, N_TOKENS),
            vh: PackedB::zeroed(N_TOKENS, HEAD_DIM),
            ch: vec![0.0; N_TOKENS * HEAD_DIM],
            zero_tokens: vec![0.0; N_TOKENS],
            zero_head: vec![0.0; HEAD_DIM],
            hist: vec![0.0; 2 * TEMPORAL_WINDOW],
            hidden: vec![0.0; TEMPORAL_HIDDEN],
            tokens: vec![0.0; 2 * EMBED],
            patch_logits: vec![0.0; N_PATCHES * POLICY_OUT],
        }
    }
}

pub struct Net {
    embedder: Linear,
    value_token: Vec<f32>,  // (1, 384)
    pos_encoding: Vec<f32>, // (52, 384)
    blocks: Vec<Block>,
    norm_out: LayerNorm,
    policy_head: Linear,
    value_head: Linear,
    army_l1: Linear,
    army_l2: Linear,
    land_l1: Linear,
    land_l2: Linear,
    temporal_type_embed: Vec<f32>, // (2, 384)
    bin_centers: Vec<f32>,         // (128,)
    scratch: RefCell<Scratch>,
}

/// The forward pass's outputs: masked flat logits, the value scalar, and the
/// raw value-bin logits (the parity surface covers the whole network).
pub struct ForwardOut {
    pub logits: Vec<f32>, // (4410,)
    pub value: f32,
    pub value_bins: Vec<f32>, // (128,)
}

/// The artifact's tensors plus the set of names already claimed, so the
/// loader can refuse an artifact with leftovers — the same guardrail the
/// candle path got from draining its `HashMap`.
struct Tensors {
    st: SafeTensors,
    taken: BTreeSet<String>,
}

fn take(tensors: &mut Tensors, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
    // Dtype is already enforced file-wide: the safetensors reader rejects
    // anything that is not F32 at parse time.
    let data = tensors.st.get_shaped(name, shape)?.to_vec();
    tensors.taken.insert(name.to_string());
    Ok(data)
}

fn take_linear(
    tensors: &mut Tensors,
    prefix: &str,
    out_dim: usize,
    in_dim: usize,
) -> Result<Linear, String> {
    let weight = take(tensors, &format!("{prefix}.weight"), &[out_dim, in_dim])?;
    let bias = take(tensors, &format!("{prefix}.bias"), &[out_dim])?;
    let mut weights = PackedB::zeroed(in_dim, out_dim);
    for o in 0..out_dim {
        for i in 0..in_dim {
            weights.set(i, o, weight[o * in_dim + i]);
        }
    }
    Ok(Linear { weights, bias })
}

fn take_norm(tensors: &mut Tensors, prefix: &str) -> Result<LayerNorm, String> {
    Ok(LayerNorm {
        weight: take(tensors, &format!("{prefix}.weight"), &[EMBED])?,
        bias: take(tensors, &format!("{prefix}.bias"), &[EMBED])?,
    })
}

impl Net {
    /// Load and schema-check the artifact. `dir` is `artifact/` next to the
    /// crate; the manifest's `tensor_schema` tag and the network block are
    /// verified before any tensor is touched.
    pub fn load(dir: &Path) -> Result<Self, String> {
        let manifest_text = std::fs::read_to_string(dir.join("manifest.json"))
            .map_err(|e| format!("read manifest.json: {e}"))?;
        let manifest = json::parse(&manifest_text).map_err(|e| format!("manifest.json: {e}"))?;
        let schema = manifest.get("tensor_schema").and_then(|v| v.as_str());
        if schema != Some(TENSOR_SCHEMA) {
            return Err(format!(
                "tensor_schema {schema:?}, this binary needs {TENSOR_SCHEMA:?}"
            ));
        }
        let net_cfg = manifest.field("network").map_err(|e| format!("manifest: {e}"))?;
        for (key, want) in [
            ("pad_to", PAD as i64),
            ("history_size", 7),
            ("depth", DEPTH as i64),
            ("embed_dim", EMBED as i64),
            ("n_head", N_HEAD as i64),
            ("ff_factor", 4),
            ("patch_size", PATCH as i64),
            ("num_bins", NUM_BINS as i64),
        ] {
            let got = net_cfg.int_field(key).map_err(|e| format!("manifest network: {e}"))?;
            if got != want {
                return Err(format!("manifest network.{key} = {got}, expected {want}"));
            }
        }

        let st = SafeTensors::load(&dir.join("model.safetensors"))
            .map_err(|e| format!("load model.safetensors: {e}"))?;
        let mut tensors = Tensors { st, taken: BTreeSet::new() };

        let embedder = take_linear(&mut tensors, "embedder", EMBED, PATCH_DIM)?;
        let value_token = take(&mut tensors, "value_token", &[1, EMBED])?;
        let pos_encoding = take(&mut tensors, "pos_encoding", &[N_TOKENS, EMBED])?;
        let mut blocks = Vec::with_capacity(DEPTH);
        for i in 0..DEPTH {
            let p = format!("transformer_layers.{i}");
            blocks.push(Block {
                norm1: take_norm(&mut tensors, &format!("{p}.norm1"))?,
                q: take_linear(&mut tensors, &format!("{p}.attn.q_proj"), EMBED, EMBED)?,
                k: take_linear(&mut tensors, &format!("{p}.attn.k_proj"), EMBED, EMBED)?,
                v: take_linear(&mut tensors, &format!("{p}.attn.v_proj"), EMBED, EMBED)?,
                out: take_linear(&mut tensors, &format!("{p}.attn.out_proj"), EMBED, EMBED)?,
                norm2: take_norm(&mut tensors, &format!("{p}.norm2"))?,
                ff1: take_linear(&mut tensors, &format!("{p}.ff_linear1"), FF_DIM, EMBED)?,
                ff2: take_linear(&mut tensors, &format!("{p}.ff_linear2"), EMBED, FF_DIM)?,
            });
        }
        let norm_out = take_norm(&mut tensors, "norm_out")?;
        let policy_head = take_linear(&mut tensors, "policy_head", POLICY_OUT, EMBED)?;
        let value_head = take_linear(&mut tensors, "value_head", NUM_BINS, EMBED)?;
        let army_l1 =
            take_linear(&mut tensors, "temporal_encoder.army_l1", TEMPORAL_HIDDEN, TEMPORAL_WINDOW)?;
        let army_l2 = take_linear(&mut tensors, "temporal_encoder.army_l2", EMBED, TEMPORAL_HIDDEN)?;
        let land_l1 =
            take_linear(&mut tensors, "temporal_encoder.land_l1", TEMPORAL_HIDDEN, TEMPORAL_WINDOW)?;
        let land_l2 = take_linear(&mut tensors, "temporal_encoder.land_l2", EMBED, TEMPORAL_HIDDEN)?;
        let temporal_type_embed = take(&mut tensors, "temporal_type_embed", &[2, EMBED])?;
        let bin_centers = take(&mut tensors, "bin_centers", &[NUM_BINS])?;

        let extra: Vec<&str> =
            tensors.st.names().filter(|name| !tensors.taken.contains(*name)).collect();
        if !extra.is_empty() {
            return Err(format!("artifact has unexpected tensors: {extra:?}"));
        }

        Ok(Self {
            embedder,
            value_token,
            pos_encoding,
            blocks,
            norm_out,
            policy_head,
            value_head,
            army_l1,
            army_l2,
            land_l1,
            land_l2,
            temporal_type_embed,
            bin_centers,
            scratch: RefCell::new(Scratch::new()),
        })
    }

    /// One forward pass. `aug_norm` is the normalized (39, 21, 21) tensor,
    /// `penalties` the (10, 21, 21) mask plane from `prepare_action_mask`,
    /// `temporal` the (2, 512) opponent stat windows (raw counts — the /50
    /// lives inside the temporal encoder, as in Python).
    pub fn forward(
        &self,
        aug_norm: &[f32],
        penalties: &[f32],
        temporal: &[f32],
    ) -> Result<ForwardOut, String> {
        assert_eq!(aug_norm.len(), N_CHANNELS * CELLS);
        assert_eq!(penalties.len(), N_LOGITS);
        assert_eq!(temporal.len(), 2 * TEMPORAL_WINDOW);

        let scratch = &mut *self.scratch.borrow_mut();
        let Scratch {
            patched,
            x,
            normed,
            q,
            k,
            v,
            ctx,
            proj,
            ff,
            scores,
            qh,
            kt,
            vh,
            ch,
            zero_tokens,
            zero_head,
            hist,
            hidden,
            tokens,
            patch_logits,
        } = scratch;

        // Patchify: (39, 21, 21) -> (49, 351), feature order (c, mi, mj) —
        // the reshape/transpose in `HistoryTransformer._forward`.
        for gi in 0..GRID_PATCHES {
            for gj in 0..GRID_PATCHES {
                let token = gi * GRID_PATCHES + gj;
                for c in 0..N_CHANNELS {
                    for mi in 0..PATCH {
                        for mj in 0..PATCH {
                            let src = c * CELLS + (gi * PATCH + mi) * PAD + (gj * PATCH + mj);
                            patched[token * PATCH_DIM + c * PATCH * PATCH + mi * PATCH + mj] =
                                aug_norm[src];
                        }
                    }
                }
            }
        }
        // Embed straight into the patch rows (3..52) of the residual stream.
        self.embedder.forward_into(N_PATCHES, patched, &mut x[3 * EMBED..]);

        // Temporal tokens: two independent 512 -> 512 -> 384 MLPs with SiLU,
        // on history / 50, plus the type embedding. The division happens as a
        // reciprocal multiply because that is what XLA compiles it to
        // (xla_math::RECIP_50 — the candle path did the same, and for the
        // same reason: candle's scalar div is also a reciprocal multiply).
        for (dst, src) in hist.iter_mut().zip(temporal) {
            *dst = src * crate::xla_math::RECIP_50;
        }
        self.army_l1.forward_into(1, &hist[..TEMPORAL_WINDOW], hidden);
        silu_in_place(hidden);
        self.army_l2.forward_into(1, hidden, &mut tokens[..EMBED]);
        self.land_l1.forward_into(1, &hist[TEMPORAL_WINDOW..], hidden);
        silu_in_place(hidden);
        self.land_l2.forward_into(1, hidden, &mut tokens[EMBED..]);
        for (t, e) in tokens.iter_mut().zip(&self.temporal_type_embed) {
            *t += e;
        }

        // Sequence: [VALUE, TEMPORAL_ARMY, TEMPORAL_LAND, PATCH_0..48] + pos.
        x[..EMBED].copy_from_slice(&self.value_token);
        x[EMBED..3 * EMBED].copy_from_slice(tokens);
        for (xv, pv) in x.iter_mut().zip(&self.pos_encoding) {
            *xv += pv;
        }

        // The candle path divided the score matrix by this f64 scale, which
        // is an `affine(1/scale, 0)` — a reciprocal multiply folded in f64,
        // cast to f32. Mirrored exactly.
        let scale = (HEAD_DIM as f64).sqrt();
        let inv_scale = (1.0 / scale) as f32;
        for block in &self.blocks {
            block.norm1.forward_into(N_TOKENS, x, normed);
            block.q.forward_into(N_TOKENS, normed, q);
            block.k.forward_into(N_TOKENS, normed, k);
            block.v.forward_into(N_TOKENS, normed, v);
            for h in 0..N_HEAD {
                let off = h * HEAD_DIM;
                // Pack the head — Q rows contiguous, K and V into the
                // strip-major panels gemm_bias streams — then run QK^T and
                // the context product through the GEMM kernel. ~30 KB of
                // copies against two register-tiled GEMM calls.
                for i in 0..N_TOKENS {
                    qh[i * HEAD_DIM..(i + 1) * HEAD_DIM]
                        .copy_from_slice(&q[i * EMBED + off..i * EMBED + off + HEAD_DIM]);
                    vh.fill_row(i, &v[i * EMBED + off..i * EMBED + off + HEAD_DIM]);
                    kt.fill_col(i, &k[i * EMBED + off..i * EMBED + off + HEAD_DIM]);
                }
                gemm_bias(N_TOKENS, qh, kt, zero_tokens, scores);
                for s in scores.iter_mut() {
                    *s *= inv_scale;
                }
                for i in 0..N_TOKENS {
                    softmax_in_place(&mut scores[i * N_TOKENS..(i + 1) * N_TOKENS]);
                }
                gemm_bias(N_TOKENS, scores, vh, zero_head, ch);
                for i in 0..N_TOKENS {
                    ctx[i * EMBED + off..i * EMBED + off + HEAD_DIM]
                        .copy_from_slice(&ch[i * HEAD_DIM..(i + 1) * HEAD_DIM]);
                }
            }
            block.out.forward_into(N_TOKENS, ctx, proj);
            for (xv, pv) in x.iter_mut().zip(proj.iter()) {
                *xv += pv;
            }
            block.norm2.forward_into(N_TOKENS, x, normed);
            block.ff1.forward_into(N_TOKENS, normed, ff);
            silu_in_place(ff);
            block.ff2.forward_into(N_TOKENS, ff, proj);
            for (xv, pv) in x.iter_mut().zip(proj.iter()) {
                *xv += pv;
            }
        }
        self.norm_out.forward_into(N_TOKENS, x, normed);

        // Value head: 128 bin logits -> softmax -> dot with bin centers.
        let mut value_bins = vec![0f32; NUM_BINS];
        self.value_head.forward_into(1, &normed[..EMBED], &mut value_bins);
        let mut value_probs = value_bins.clone();
        softmax_in_place(&mut value_probs);
        let value: f32 = value_probs
            .iter()
            .zip(&self.bin_centers)
            .map(|(p, c)| p * c)
            .sum();

        // Policy head: per-patch logits, unpatchified to (10, 21, 21), plus
        // the -1e9 mask.
        self.policy_head.forward_into(N_PATCHES, &normed[3 * EMBED..], patch_logits);
        let mut logits = vec![0f32; N_LOGITS];
        for gi in 0..GRID_PATCHES {
            for gj in 0..GRID_PATCHES {
                let token = gi * GRID_PATCHES + gj;
                for a in 0..N_ACTION_CHANNELS {
                    for mi in 0..PATCH {
                        for mj in 0..PATCH {
                            let dst = a * CELLS + (gi * PATCH + mi) * PAD + (gj * PATCH + mj);
                            logits[dst] = patch_logits
                                [token * POLICY_OUT + a * PATCH * PATCH + mi * PATCH + mj]
                                + penalties[dst];
                        }
                    }
                }
            }
        }

        Ok(ForwardOut { logits, value, value_bins })
    }
}
