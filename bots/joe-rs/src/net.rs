//! The HistoryTransformer forward pass in candle (port-plan §5).
//!
//! Single sample, fixed shapes: 39×21×21 obs → 49 patch tokens of 351 →
//! 52 tokens × 384 through five pre-norm blocks (MHSA 8 heads / head_dim 48,
//! then 384→1152 SiLU→384) → policy head (49×90, unpatchified to 10×21×21
//! masked logits) and value head (128 bins dotted with the exported
//! `bin_centers`). Deployment is float32 everywhere — the checkpoint's
//! `use_bf16` was already off in the Python sibling.
//!
//! Weights come from `artifact/model.safetensors` (written by
//! `tools/convert_artifact.py`); the loader refuses to start on a schema
//! tag, tensor name, shape, or dtype mismatch — the same guardrail the
//! Python loader gets from `tree_deserialise_leaves`.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Tensor, D};
use candle_nn::ops::softmax;

use crate::obs::{CELLS, N_ACTION_CHANNELS, N_CHANNELS, PAD, TEMPORAL_WINDOW};

pub const TENSOR_SCHEMA: &str = "joe-net-v1";
pub const EMBED: usize = 384;
pub const DEPTH: usize = 5;
pub const N_HEAD: usize = 8;
pub const HEAD_DIM: usize = EMBED / N_HEAD; // 48
pub const PATCH: usize = 3;
pub const GRID_PATCHES: usize = PAD / PATCH; // 7
pub const N_PATCHES: usize = GRID_PATCHES * GRID_PATCHES; // 49
pub const PATCH_DIM: usize = N_CHANNELS * PATCH * PATCH; // 351
pub const N_TOKENS: usize = N_PATCHES + 3; // value + 2 temporal + patches
pub const NUM_BINS: usize = 128;
pub const FF_DIM: usize = 1152;
pub const POLICY_OUT: usize = N_ACTION_CHANNELS * PATCH * PATCH; // 90
pub const N_LOGITS: usize = N_ACTION_CHANNELS * CELLS; // 4410

struct Linear {
    weight_t: Tensor, // (in, out) — transposed once at load
    bias: Tensor,     // (out,)
}

impl Linear {
    /// `y = x @ W^T + b` on a (tokens, in) matrix.
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        x.matmul(&self.weight_t)?.broadcast_add(&self.bias)
    }
}

struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
}

impl LayerNorm {
    /// Equinox LayerNorm: biased variance, eps 1e-5, applied per token.
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let mean = x.mean_keepdim(D::Minus1)?;
        let centered = x.broadcast_sub(&mean)?;
        let var = centered.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = centered.broadcast_div(&(var + 1e-5)?.sqrt()?)?;
        normed.broadcast_mul(&self.weight)?.broadcast_add(&self.bias)
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

pub struct Net {
    embedder: Linear,
    value_token: Tensor,  // (1, 384)
    pos_encoding: Tensor, // (52, 384)
    blocks: Vec<Block>,
    norm_out: LayerNorm,
    policy_head: Linear,
    value_head: Linear,
    army_l1: Linear,
    army_l2: Linear,
    land_l1: Linear,
    land_l2: Linear,
    temporal_type_embed: Tensor, // (2, 384)
    bin_centers: Vec<f32>,       // (128,)
    device: Device,
}

/// The forward pass's outputs: masked flat logits, the value scalar, and the
/// raw value-bin logits (the parity surface covers the whole network).
pub struct ForwardOut {
    pub logits: Vec<f32>, // (4410,)
    pub value: f32,
    pub value_bins: Vec<f32>, // (128,)
}

fn take(
    tensors: &mut HashMap<String, Tensor>,
    name: &str,
    shape: &[usize],
) -> Result<Tensor, String> {
    let t = tensors
        .remove(name)
        .ok_or_else(|| format!("artifact missing tensor {name}"))?;
    if t.dtype() != DType::F32 {
        return Err(format!("{name}: dtype {:?}, expected f32", t.dtype()));
    }
    if t.dims() != shape {
        return Err(format!("{name}: shape {:?}, expected {shape:?}", t.dims()));
    }
    Ok(t)
}

fn take_linear(
    tensors: &mut HashMap<String, Tensor>,
    prefix: &str,
    out_dim: usize,
    in_dim: usize,
) -> Result<Linear, String> {
    let weight = take(tensors, &format!("{prefix}.weight"), &[out_dim, in_dim])?;
    let bias = take(tensors, &format!("{prefix}.bias"), &[out_dim])?;
    let weight_t = weight.t().and_then(|t| t.contiguous()).map_err(|e| e.to_string())?;
    Ok(Linear { weight_t, bias })
}

fn take_norm(tensors: &mut HashMap<String, Tensor>, prefix: &str) -> Result<LayerNorm, String> {
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
        let manifest: serde_json::Value =
            serde_json::from_str(&manifest_text).map_err(|e| format!("manifest.json: {e}"))?;
        if manifest["tensor_schema"] != TENSOR_SCHEMA {
            return Err(format!(
                "tensor_schema {:?}, this binary needs {TENSOR_SCHEMA:?}",
                manifest["tensor_schema"]
            ));
        }
        let net_cfg = &manifest["network"];
        for (key, want) in [
            ("pad_to", PAD as i64),
            ("history_size", 7),
            ("depth", DEPTH as i64),
            ("embed_dim", EMBED as i64),
            ("n_head", N_HEAD as i64),
            ("ff_factor", 3),
            ("patch_size", PATCH as i64),
            ("num_bins", NUM_BINS as i64),
        ] {
            if net_cfg[key] != want {
                return Err(format!("manifest network.{key} = {}, expected {want}", net_cfg[key]));
            }
        }

        let device = Device::Cpu;
        let mut tensors = candle_core::safetensors::load(dir.join("model.safetensors"), &device)
            .map_err(|e| format!("load model.safetensors: {e}"))?;

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
        let army_l1 = take_linear(&mut tensors, "temporal_encoder.army_l1", 512, TEMPORAL_WINDOW)?;
        let army_l2 = take_linear(&mut tensors, "temporal_encoder.army_l2", EMBED, 512)?;
        let land_l1 = take_linear(&mut tensors, "temporal_encoder.land_l1", 512, TEMPORAL_WINDOW)?;
        let land_l2 = take_linear(&mut tensors, "temporal_encoder.land_l2", EMBED, 512)?;
        let temporal_type_embed = take(&mut tensors, "temporal_type_embed", &[2, EMBED])?;
        let bin_centers = take(&mut tensors, "bin_centers", &[NUM_BINS])?
            .to_vec1::<f32>()
            .map_err(|e| e.to_string())?;

        if !tensors.is_empty() {
            let mut extra: Vec<&String> = tensors.keys().collect();
            extra.sort();
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
            device,
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
        self.forward_inner(aug_norm, penalties, temporal).map_err(|e| e.to_string())
    }

    fn forward_inner(
        &self,
        aug_norm: &[f32],
        penalties: &[f32],
        temporal: &[f32],
    ) -> candle_core::Result<ForwardOut> {
        assert_eq!(aug_norm.len(), N_CHANNELS * CELLS);
        assert_eq!(penalties.len(), N_LOGITS);
        assert_eq!(temporal.len(), 2 * TEMPORAL_WINDOW);

        // Patchify: (39, 21, 21) -> (49, 351), feature order (c, mi, mj) —
        // the reshape/transpose in `HistoryTransformer._forward`.
        let mut patched = vec![0f32; N_PATCHES * PATCH_DIM];
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
        let x = Tensor::from_slice(&patched, (N_PATCHES, PATCH_DIM), &self.device)?;
        let x = self.embedder.forward(&x)?; // (49, 384)

        // Temporal tokens: two independent 512 -> 512 -> 384 MLPs with SiLU,
        // on history / 50, plus the type embedding. The division happens
        // here as a true f32 divide (candle's scalar div is a reciprocal
        // multiply, which is not the same bits).
        let hist_scaled: Vec<f32> =
            temporal.iter().map(|v| v * crate::xla_math::RECIP_50).collect();
        let hist = Tensor::from_slice(&hist_scaled, (2, TEMPORAL_WINDOW), &self.device)?;
        let army = hist.narrow(0, 0, 1)?;
        let land = hist.narrow(0, 1, 1)?;
        let army_tok = self.army_l2.forward(&candle_nn::ops::silu(&self.army_l1.forward(&army)?)?)?;
        let land_tok = self.land_l2.forward(&candle_nn::ops::silu(&self.land_l1.forward(&land)?)?)?;
        let temporal_tokens =
            Tensor::cat(&[&army_tok, &land_tok], 0)?.add(&self.temporal_type_embed)?;

        // Sequence: [VALUE, TEMPORAL_ARMY, TEMPORAL_LAND, PATCH_0..48] + pos.
        let mut x = Tensor::cat(&[&self.value_token, &temporal_tokens, &x], 0)?
            .add(&self.pos_encoding)?; // (52, 384)

        let scale = (HEAD_DIM as f64).sqrt();
        for block in &self.blocks {
            let normed = block.norm1.forward(&x)?;
            let q = block.q.forward(&normed)?.reshape((N_TOKENS, N_HEAD, HEAD_DIM))?;
            let k = block.k.forward(&normed)?.reshape((N_TOKENS, N_HEAD, HEAD_DIM))?;
            let v = block.v.forward(&normed)?.reshape((N_TOKENS, N_HEAD, HEAD_DIM))?;
            let q = q.transpose(0, 1)?.contiguous()?; // (8, 52, 48)
            let k = k.transpose(0, 1)?.contiguous()?;
            let v = v.transpose(0, 1)?.contiguous()?;
            let attn = (q.matmul(&k.transpose(1, 2)?)? / scale)?; // (8, 52, 52)
            let attn = softmax(&attn, D::Minus1)?;
            let out = attn.matmul(&v)?; // (8, 52, 48)
            let out = out.transpose(0, 1)?.reshape((N_TOKENS, EMBED))?;
            let out = block.out.forward(&out)?;
            x = x.add(&out)?;
            let h = block.norm2.forward(&x)?;
            let h = block.ff2.forward(&candle_nn::ops::silu(&block.ff1.forward(&h)?)?)?;
            x = x.add(&h)?;
        }
        let x = self.norm_out.forward(&x)?; // (52, 384)

        // Value head: 128 bin logits -> softmax -> dot with bin centers.
        let value_emb = x.narrow(0, 0, 1)?;
        let value_bins_t = self.value_head.forward(&value_emb)?; // (1, 128)
        let value_probs = softmax(&value_bins_t, D::Minus1)?
            .squeeze(0)?
            .to_vec1::<f32>()?;
        let value_bins = value_bins_t.squeeze(0)?.to_vec1::<f32>()?;
        let value: f32 = value_probs
            .iter()
            .zip(&self.bin_centers)
            .map(|(p, c)| p * c)
            .sum();

        // Policy head: per-patch logits, unpatchified to (10, 21, 21), plus
        // the -1e9 mask.
        let patch_emb = x.narrow(0, 3, N_PATCHES)?;
        let patch_logits = self.policy_head.forward(&patch_emb)?.flatten_all()?.to_vec1::<f32>()?;
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
