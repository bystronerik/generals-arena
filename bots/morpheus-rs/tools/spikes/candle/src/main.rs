// M3 engine spike: the Morpheus graph in candle, timed at batch 1/4/8.
//
// Usage: candle-spike <model.safetensors> [iters]
//
// Writes p50/p99 per batch to stdout as `batch p50_ms p99_ms`, and dumps the
// policy/pass/wdl of a fixed pseudo-random input so the numbers can be checked
// against TorchScript.

use candle_core::{DType, Device, Tensor};
use candle_nn::{conv2d_no_bias, group_norm, linear, Conv2d, Conv2dConfig, GroupNorm, Linear, Module, VarBuilder};
use std::time::Instant;

const IN_CHANNELS: usize = 49;
const BOARD: usize = 21;
const TRUNK: usize = 64;
const EXPANSION: usize = 128;
const POLICY_CH: usize = 9;
const N_BLOCKS: usize = 12;
const GROUPS: usize = 8;
const DILATIONS: [usize; 3] = [1, 2, 4];

struct Block {
    pw_expand: Conv2d,
    gn_expand: GroupNorm,
    dw: Conv2d,
    gn_dw: GroupNorm,
    pw_project: Conv2d,
    gn_project: GroupNorm,
}

fn relu6(x: &Tensor) -> candle_core::Result<Tensor> {
    x.clamp(0.0f32, 6.0f32)
}

impl Block {
    fn new(vb: VarBuilder, dilation: usize) -> candle_core::Result<Self> {
        let pw = Conv2dConfig { padding: 0, stride: 1, dilation: 1, groups: 1, cudnn_fwd_algo: None };
        let dwc = Conv2dConfig { padding: dilation, stride: 1, dilation, groups: EXPANSION, cudnn_fwd_algo: None };
        Ok(Self {
            pw_expand: conv2d_no_bias(TRUNK, EXPANSION, 1, pw, vb.pp("pw_expand"))?,
            gn_expand: group_norm(GROUPS, EXPANSION, 1e-5, vb.pp("gn_expand"))?,
            dw: conv2d_no_bias(EXPANSION, EXPANSION, 3, dwc, vb.pp("dw"))?,
            gn_dw: group_norm(GROUPS, EXPANSION, 1e-5, vb.pp("gn_dw"))?,
            pw_project: conv2d_no_bias(EXPANSION, TRUNK, 1, pw, vb.pp("pw_project"))?,
            gn_project: group_norm(GROUPS, TRUNK, 1e-5, vb.pp("gn_project"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let y = relu6(&self.gn_expand.forward(&self.pw_expand.forward(x)?)?)?;
        let y = relu6(&self.gn_dw.forward(&self.dw.forward(&y)?)?)?;
        let y = self.gn_project.forward(&self.pw_project.forward(&y)?)?;
        relu6(&(x + y)?)
    }
}

struct Net {
    stem: Conv2d,
    gn_stem: GroupNorm,
    blocks: Vec<Block>,
    policy: Conv2d,
    pass_fc: Linear,
    wdl: Linear,
}

impl Net {
    fn new(vb: VarBuilder) -> candle_core::Result<Self> {
        let stem_cfg = Conv2dConfig { padding: 1, stride: 1, dilation: 1, groups: 1, cudnn_fwd_algo: None };
        let pw = Conv2dConfig { padding: 0, stride: 1, dilation: 1, groups: 1, cudnn_fwd_algo: None };
        let mut blocks = Vec::with_capacity(N_BLOCKS);
        for i in 0..N_BLOCKS {
            blocks.push(Block::new(vb.pp(format!("blocks.{i}")), DILATIONS[i % 3])?);
        }
        Ok(Self {
            stem: conv2d_no_bias(IN_CHANNELS, TRUNK, 3, stem_cfg, vb.pp("stem"))?,
            gn_stem: group_norm(GROUPS, TRUNK, 1e-5, vb.pp("gn_stem"))?,
            blocks,
            policy: candle_nn::conv2d(TRUNK, POLICY_CH, 1, pw, vb.pp("policy"))?,
            pass_fc: linear(TRUNK * 2, 1, vb.pp("pass_fc"))?,
            wdl: linear(TRUNK * 2, 3, vb.pp("wdl"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
        let mask = x.narrow(1, 0, 1)?;
        let mut h = relu6(&self.gn_stem.forward(&self.stem.forward(x)?)?)?;
        for b in &self.blocks {
            h = b.forward(&h)?;
        }
        // masked global mean/max, mirroring network.masked_global_features
        let denom = mask.sum((2, 3))?; // N x 1
        let mean = (h.broadcast_mul(&mask)?.sum((2, 3))? / denom.broadcast_as((h.dim(0)?, TRUNK))?)?;
        let neg = mask.broadcast_as(h.shape())?.affine(-1e9, 1e9)?; // 0 where mask=1, 1e9 where 0
        let masked = (&h - neg)?;
        let mx = masked.max(3)?.max(2)?;
        let feat = Tensor::cat(&[mean, mx], 1)?;
        Ok((self.policy.forward(&h)?, self.pass_fc.forward(&feat)?, self.wdl.forward(&feat)?))
    }
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    // nearest-rank, same convention as the Python telemetry
    let n = sorted.len();
    let rank = ((q * n as f64).ceil() as usize).max(1).min(n);
    sorted[rank - 1]
}

fn main() -> candle_core::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("model.safetensors path");
    let iters: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(300);
    let dev = Device::Cpu;
    let load = Instant::now();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &dev)? };
    let net = Net::new(vb)?;
    let load_ms = load.elapsed().as_secs_f64() * 1e3;
    println!("load_ms {load_ms:.3}");

    for &batch in &[1usize, 4, 8] {
        // deterministic pseudo-random input, same LCG as the bespoke bench
        let mut seed: u32 = 12345;
        let n = batch * IN_CHANNELS * BOARD * BOARD;
        let mut data = Vec::with_capacity(n);
        for _ in 0..n {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            data.push(((seed >> 8) as f32 / 16777216.0) * 2.0 - 1.0);
        }
        let x = Tensor::from_vec(data, (batch, IN_CHANNELS, BOARD, BOARD), &dev)?;
        for _ in 0..10 {
            net.forward(&x)?;
        }
        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = Instant::now();
            let (p, pa, w) = net.forward(&x)?;
            std::hint::black_box((&p, &pa, &w));
            times.push(t.elapsed().as_secs_f64() * 1e3);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "batch {batch} p50 {:.3} p99 {:.3} min {:.3}",
            percentile(&times, 0.50),
            percentile(&times, 0.99),
            times[0]
        );
    }
    Ok(())
}
