//! The one hot arithmetic kernel: `C = A·B + bias` in f32, register-blocked.
//!
//! Every linear layer in the joe graph is this product: `A` is an activation
//! matrix of at most 52 tokens, `B` is a weight matrix stored transposed at
//! load — `(in, out)` row-major, so the inner loop streams contiguous rows —
//! and `bias` is one value per output column. Adapted from the morpheus-rs
//! kernel (`crates/core/src/nn/gemm.rs`), which won that bot's M3 inference
//! shoot-out against candle and TorchScript; joe's shapes are token-major
//! where morpheus's were cell-major, so bias lives on columns here, and the
//! policy head's 90 output columns need a real column tail (90 = 5·16 + 10).
//!
//! **No intrinsics, no `unsafe`.** The shapes are fixed and small, so what the
//! kernel has to get right is register blocking and loop order; the vector
//! instructions themselves are LLVM's job, and it does that job well when the
//! inner loop is a fixed-length array with no aliasing. Keeping it in safe
//! Rust also means it compiles to NEON on the development Mac and to whatever
//! the x86 target level allows without a second implementation to keep in step.
//!
//! **`mul_add`, not `a * b + c`.** Rust compiles floating-point with
//! contraction off, so `acc += a * b` emits a separate multiply and add and
//! never a fused multiply-add. Writing the fusion explicitly is what took the
//! morpheus kernel from 11 to 38 GFLOP/s; the rounding changes (one rounding
//! per term instead of two, which is *more* accurate), and the tier-2 parity
//! gate measures what it costs against the JAX oracle rather than assuming it
//! is free. The `HAS_HARDWARE_FMA` caveat from morpheus applies unchanged: on
//! a target with no FMA instruction `mul_add` falls back to libm's exact
//! `fmaf()` at a measured 49× slowdown, which is why `.cargo/config.toml`
//! pins `target-cpu=x86-64-v3` and `selfcheck` reports the build's features.

/// Columns per register tile. Four NEON/AVX f32 vectors' worth.
pub const NR: usize = 16;
/// Rows per register tile. `MR × NR` accumulators must stay in registers.
const MR: usize = 4;

/// `c[r][o] = bias[o] + Σ_k a[r][k] · b[k][o]` — all row-major, `a` is
/// (rows × k_dim), `b` is (k_dim × n_dim), `c` is (rows × n_dim).
pub fn gemm_bias(
    rows: usize,
    k_dim: usize,
    n_dim: usize,
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    c: &mut [f32],
) {
    debug_assert_eq!(a.len(), rows * k_dim);
    debug_assert_eq!(b.len(), k_dim * n_dim);
    debug_assert_eq!(bias.len(), n_dim);
    debug_assert_eq!(c.len(), rows * n_dim);

    let n_main = n_dim - n_dim % NR;
    let mut n0 = 0;
    while n0 < n_main {
        let mut m0 = 0;
        while m0 + MR <= rows {
            let mut acc = [[0f32; NR]; MR];
            // `&[f32; NR]` below, not `&[f32]`: the length in the type keeps
            // the bounds check out of the innermost loop.
            let arows: [&[f32]; MR] = [
                &a[m0 * k_dim..(m0 + 1) * k_dim],
                &a[(m0 + 1) * k_dim..(m0 + 2) * k_dim],
                &a[(m0 + 2) * k_dim..(m0 + 3) * k_dim],
                &a[(m0 + 3) * k_dim..(m0 + 4) * k_dim],
            ];
            for k in 0..k_dim {
                let brow: &[f32; NR] = b[k * n_dim + n0..].first_chunk().unwrap();
                for (i, row) in acc.iter_mut().enumerate() {
                    let av = arows[i][k];
                    for j in 0..NR {
                        row[j] = av.mul_add(brow[j], row[j]);
                    }
                }
            }
            for (i, row) in acc.iter().enumerate() {
                let base = (m0 + i) * n_dim + n0;
                for j in 0..NR {
                    c[base + j] = row[j] + bias[n0 + j];
                }
            }
            m0 += MR;
        }
        // Row tail: the temporal MLPs and the value head run one token at a
        // time, and 52 tokens leave nothing over 4·13 — so this path carries
        // whole layers, not decoration.
        while m0 < rows {
            let arow = &a[m0 * k_dim..(m0 + 1) * k_dim];
            let mut acc = [0f32; NR];
            for k in 0..k_dim {
                let brow: &[f32; NR] = b[k * n_dim + n0..].first_chunk().unwrap();
                let av = arow[k];
                for j in 0..NR {
                    acc[j] = av.mul_add(brow[j], acc[j]);
                }
            }
            let base = m0 * n_dim + n0;
            for j in 0..NR {
                c[base + j] = acc[j] + bias[n0 + j];
            }
            m0 += 1;
        }
        n0 += NR;
    }

    // Column tail: the policy head is 90 wide, leaving 10 columns here. The
    // inner loop has a runtime trip count, so it vectorizes worse than the
    // main tile — acceptable for 10 of 90 columns of one small GEMM.
    let n_tail = n_dim - n_main;
    if n_tail > 0 {
        for m0 in 0..rows {
            let arow = &a[m0 * k_dim..(m0 + 1) * k_dim];
            let mut acc = [0f32; NR];
            for k in 0..k_dim {
                let brow = &b[k * n_dim + n_main..(k * n_dim) + n_dim];
                let av = arow[k];
                for (j, bv) in brow.iter().enumerate() {
                    acc[j] = av.mul_add(*bv, acc[j]);
                }
            }
            let base = m0 * n_dim + n_main;
            for j in 0..n_tail {
                c[base + j] = acc[j] + bias[n_main + j];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The obvious triple loop, which is the point: it is a different program.
    fn naive(rows: usize, k: usize, n: usize, a: &[f32], b: &[f32], bias: &[f32]) -> Vec<f32> {
        let mut c = vec![0f32; rows * n];
        for i in 0..rows {
            for j in 0..n {
                let mut sum = 0f32;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = sum + bias[j];
            }
        }
        c
    }

    fn pseudo(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn matches_the_naive_product_on_the_graph_shapes() {
        // Every (rows, k, n) the network actually uses: embedder, the
        // attention projections, both FF layers, the policy head (its 90
        // exercises the column tail), the value head and temporal MLPs
        // (row tails at rows=1).
        for &(rows, k, n) in &[
            (49usize, 351usize, 384usize),
            (52, 384, 384),
            (52, 384, 1536),
            (52, 1536, 384),
            (49, 384, 90),
            (1, 384, 128),
            (1, 512, 512),
            (1, 512, 384),
        ] {
            let a = pseudo(rows * k, 7 + rows as u32);
            let b = pseudo(k * n, 11 + k as u32);
            let bias = pseudo(n, 13 + n as u32);
            let want = naive(rows, k, n, &a, &b, &bias);
            let mut got = vec![0f32; rows * n];
            gemm_bias(rows, k, n, &a, &b, &bias, &mut got);
            for i in 0..rows * n {
                let diff = (got[i] - want[i]).abs();
                assert!(
                    diff <= 2e-4 * want[i].abs().max(1.0),
                    "rows={rows} k={k} n={n} at {i}: {} vs {}",
                    got[i],
                    want[i]
                );
            }
        }
    }

    #[test]
    fn bias_is_added_once_per_column() {
        let (rows, k, n) = (5usize, 3usize, 20usize); // exercises the tail too
        let a = vec![0f32; rows * k];
        let b = vec![0f32; k * n];
        let bias: Vec<f32> = (0..n).map(|j| j as f32).collect();
        let mut c = vec![9f32; rows * n];
        gemm_bias(rows, k, n, &a, &b, &bias, &mut c);
        for i in 0..rows {
            for j in 0..n {
                assert_eq!(c[i * n + j], j as f32);
            }
        }
    }
}
