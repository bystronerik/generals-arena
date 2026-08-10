//! The one hot arithmetic kernel: `C = A·B` in f32, register-blocked.
//!
//! Every convolution in the Morpheus graph except the depthwise one is a
//! matrix multiply against the 448-wide cell plane, so this file is where the
//! network's FLOPs live. It exists because the two off-the-shelf engines the
//! plan ranked ahead of a bespoke kernel both lost the M3 shoot-out — see
//! `docs/research/measurements/morpheus-rs-inference-bench.md`.
//!
//! **No intrinsics, no `unsafe`.** The shapes are fixed and small, so what the
//! kernel has to get right is register blocking and loop order; the vector
//! instructions themselves are LLVM's job, and it does that job well when the
//! inner loop is a fixed-length array with no aliasing. Keeping it in safe
//! Rust also means it compiles to NEON on the development Mac and to whatever
//! the x86 target level allows without a second implementation to keep in
//! step — which matters because the compile target is `x86-64-v3` and the
//! sandbox host cannot be probed directly (§9, R4).
//!
//! **`mul_add`, not `a * b + c`.** This is the single largest fact in the
//! file. Rust compiles floating-point with contraction off, so `acc += a * b`
//! emits a separate multiply and add and never a fused multiply-add — LLVM is
//! not permitted to fuse them, because fusing changes the result. Writing the
//! fusion explicitly took the pointwise layers from 11 to 38 GFLOP/s and the
//! whole forward from 11.3 ms to 5.5 ms, which is the difference between
//! losing to TorchScript and beating it. The rounding does change: one
//! rounding per term instead of two, which is *more* accurate, and the `net`
//! parity surface measures what it costs against TorchScript rather than
//! assuming it is free.
//!
//! Loop order is `n` outer, `m` inner. That keeps the 16-column strip of `B`
//! resident in L1 across every row block of `A`, which is the right way round
//! for these shapes: `B` is the activation plane (up to 441×448 for the stem,
//! 790 KB) while `A` is the weight matrix (at most 128×128).

/// Is there a hardware fused multiply-add in this build?
///
/// **`mul_add` is a catastrophic pessimisation without one.** `f32::mul_add`
/// promises a single rounding, so a target with no FMA instruction cannot
/// approximate it with a multiply and an add — it calls libm's `fmaf()`, which
/// emulates the exact result in software. Measured on a one-core x86 container
/// built at baseline `x86-64`: **277 ms per forward against 5.6 ms**, a 49×
/// slowdown that compiles cleanly, produces correct output, and would blow the
/// 150 ms deadline on every move of every game.
///
/// The build is supposed to prevent this — `.cargo/config.toml` sets
/// `target-cpu=x86-64-v3`, which implies FMA — but cargo discovers that file
/// from the *working directory*, not from `--manifest-path`, so a build
/// launched from the wrong cwd silently ignores it. That is exactly how the
/// 277 ms measurement happened. The launchers now `cd` first; this constant
/// exists so the binary can also say what it actually got.
pub const HAS_HARDWARE_FMA: bool = cfg!(any(
    target_feature = "fma",  // x86-64-v3 and up
    target_feature = "neon", // every aarch64
    target_arch = "aarch64",
));

/// Columns per register tile. Four NEON/AVX f32 vectors' worth.
pub const NR: usize = 16;
/// Rows per register tile. `MR × NR` accumulators must stay in registers.
const MR: usize = 4;

/// `c[m][n] = bias[m] + Σ_k a[m][k] · b[k][n]`, row-major, `b` row stride `bs`.
///
/// `c` has row stride `bs` as well — output and activation planes share the
/// padded 448-cell stride so nothing has to be repacked between layers.
/// `bias` of `None` means accumulate from zero.
pub fn gemm_bias(
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    bs: usize,
    c: &mut [f32],
) {
    debug_assert_eq!(a.len(), m_dim * k_dim);
    debug_assert!(b.len() >= (k_dim - 1) * bs + n_dim);
    debug_assert!(c.len() >= (m_dim - 1) * bs + n_dim);
    debug_assert_eq!(n_dim % NR, 0, "callers pad the cell plane to a multiple of NR");

    let mut n0 = 0;
    while n0 < n_dim {
        let mut m0 = 0;
        while m0 + MR <= m_dim {
            let mut acc = [[0f32; NR]; MR];
            // `&[f32; NR]` below, not `&[f32]`: the length in the type keeps
            // the bounds check out of the innermost loop. Worth a little on
            // its own; the fusion in the FMA itself is worth far more.
            let arows: [&[f32]; MR] = [
                &a[m0 * k_dim..(m0 + 1) * k_dim],
                &a[(m0 + 1) * k_dim..(m0 + 2) * k_dim],
                &a[(m0 + 2) * k_dim..(m0 + 3) * k_dim],
                &a[(m0 + 3) * k_dim..(m0 + 4) * k_dim],
            ];
            for k in 0..k_dim {
                let brow: &[f32; NR] = b[k * bs + n0..].first_chunk().unwrap();
                for (i, row) in acc.iter_mut().enumerate() {
                    let av = arows[i][k];
                    for j in 0..NR {
                        row[j] = av.mul_add(brow[j], row[j]);
                    }
                }
            }
            for (i, row) in acc.iter().enumerate() {
                let base = (m0 + i) * bs + n0;
                let add = bias.map_or(0.0, |v| v[m0 + i]);
                for j in 0..NR {
                    c[base + j] = row[j] + add;
                }
            }
            m0 += MR;
        }
        // Row tail: the policy head is 9 channels wide and the auxiliary heads
        // are 1, so this path is not decoration.
        while m0 < m_dim {
            let mut acc = [0f32; NR];
            for k in 0..k_dim {
                let av = a[m0 * k_dim + k];
                let brow = &b[k * bs + n0..k * bs + n0 + NR];
                for j in 0..NR {
                    acc[j] = av.mul_add(brow[j], acc[j]);
                }
            }
            let base = m0 * bs + n0;
            let add = bias.map_or(0.0, |v| v[m0]);
            for j in 0..NR {
                c[base + j] = acc[j] + add;
            }
            m0 += 1;
        }
        n0 += NR;
    }
}

/// `y[m] = bias[m] + Σ_k a[m][k] · x[k]` — the scalar heads on the 128-wide
/// global feature vector. Small enough that shape beats blocking.
pub fn matvec(m_dim: usize, k_dim: usize, a: &[f32], x: &[f32], bias: &[f32], y: &mut [f32]) {
    debug_assert_eq!(a.len(), m_dim * k_dim);
    debug_assert_eq!(x.len(), k_dim);
    for m in 0..m_dim {
        let row = &a[m * k_dim..(m + 1) * k_dim];
        let mut sum = 0f32;
        for k in 0..k_dim {
            sum += row[k] * x[k];
        }
        y[m] = sum + bias[m];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The obvious triple loop, which is the point: it is a different program.
    fn naive(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], bs: usize) -> Vec<f32> {
        let mut c = vec![0f32; m * bs];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0f32;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * bs + j];
                }
                c[i * bs + j] = sum;
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
        // Every (m, k) the network actually uses, including the row tails.
        for &(m, k) in &[(64, 441), (128, 64), (64, 128), (9, 64), (16, 64), (1, 64)] {
            let n = 448;
            let a = pseudo(m * k, 7 + m as u32);
            let b = pseudo(k * n, 11 + k as u32);
            let want = naive(m, k, n, &a, &b, n);
            let mut got = vec![0f32; m * n];
            gemm_bias(m, k, n, &a, &b, None, n, &mut got);
            for i in 0..m * n {
                let diff = (got[i] - want[i]).abs();
                assert!(
                    diff <= 2e-4 * want[i].abs().max(1.0),
                    "m={m} k={k} at {i}: {} vs {}",
                    got[i],
                    want[i]
                );
            }
        }
    }

    #[test]
    fn bias_is_added_once_per_row() {
        let (m, k, n) = (5usize, 3usize, 16usize);
        let a = vec![0f32; m * k];
        let b = vec![0f32; k * n];
        let bias: Vec<f32> = (0..m).map(|i| i as f32).collect();
        let mut c = vec![9f32; m * n];
        gemm_bias(m, k, n, &a, &b, Some(&bias), n, &mut c);
        for i in 0..m {
            for j in 0..n {
                assert_eq!(c[i * n + j], i as f32);
            }
        }
    }

    #[test]
    fn matvec_matches_a_hand_sum() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = [1.0f32, 0.5, -2.0];
        let bias = [10.0f32, -1.0];
        let mut y = [0f32; 2];
        matvec(2, 3, &a, &x, &bias, &mut y);
        assert_eq!(y[0], 1.0 + 1.0 - 6.0 + 10.0);
        assert_eq!(y[1], 4.0 + 2.5 - 12.0 - 1.0);
    }
}
