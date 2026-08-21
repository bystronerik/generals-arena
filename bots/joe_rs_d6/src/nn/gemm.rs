//! The one hot arithmetic kernel: `C = A·B + bias` in f32, register-blocked,
//! with `B` pre-packed strip-major.
//!
//! Every linear layer in the joe graph is this product: `A` is an activation
//! matrix of at most 52 tokens, `B` a weight matrix packed once at load, and
//! `bias` one value per output column. The attention products run through
//! the same kernel — their `B` panels are packed per head at the same cost
//! the old transpose copies paid. Adapted from the morpheus-rs kernel
//! lineage; joe's shapes are token-major where morpheus's were cell-major,
//! so bias lives on columns here, and the policy head's 90 output columns
//! need a real column tail (90 = 5·16 + 10).
//!
//! **Strip-major `B`** ([`PackedB`]): for each 16-column strip, the k rows'
//! 16 values sit contiguous; the `n % 16` tail columns follow as one
//! `k × n_tail` block. A row-major `(k, n)` matrix walks 6 KB apart between
//! consecutive `k` steps on the 1536-wide FF layer — past what the hardware
//! prefetcher tracks — and the 2026-08-20 sweep priced that walk at 1.66×
//! on Modal x86 (GEMM 29.9 → 18.0 ms per forward, every shape faster, qkvo
//! included) and 1.12× on the arm64 dev box. The same sweep refuted f16
//! in-RAM weights: with packing in place, the two F16C converts per k step
//! cost more than the halved bandwidth saves (18.65 vs 18.00 ms). Numbers:
//! `docs/research/measurements/joe-rs-packed-b-sweep.md`.
//!
//! **Two paths, one answer.** On x86-64 with AVX2+FMA detected at runtime
//! the tiles run through explicit intrinsics; everywhere else — the aarch64
//! development Macs included — the portable safe kernel runs and LLVM
//! autovectorizes it. Both paths keep exactly one accumulator per output
//! element and walk `k` in order, so their outputs are bit-identical — to
//! each other and to the pre-packing kernels they replaced;
//! `tests::avx2_matches_the_portable_kernel_bitwise` holds the paths
//! together.
//!
//! **`mul_add`, not `a * b + c`.** Rust compiles floating-point with
//! contraction off, so `acc += a * b` emits a separate multiply and add and
//! never a fused multiply-add. Writing the fusion explicitly is what took
//! the morpheus kernel from 11 to 38 GFLOP/s; the rounding changes (one
//! rounding per term instead of two, which is *more* accurate), and the
//! tier-2 parity gate measures what it costs against the JAX oracle rather
//! than assuming it is free. The `HAS_HARDWARE_FMA` caveat from morpheus
//! applies unchanged: on a target with no FMA instruction `mul_add` falls
//! back to libm's exact `fmaf()` at a measured 49× slowdown, which is why
//! `.cargo/config.toml` pins `target-cpu=x86-64-v3` and `selfcheck` reports
//! the build's features.

/// Columns per register tile. Four NEON/AVX f32 vectors' worth.
pub const NR: usize = 16;
/// Rows per register tile. `MR × NR` accumulators must stay in registers.
/// The 2026-08-20 sweep measured MR 5/6/8 slower on both paths.
const MR: usize = 4;

/// A `(k_dim, n_dim)` matrix stored strip-major: for each 16-column strip,
/// `k_dim` rows of 16 contiguous values; then the tail columns as one
/// `k_dim × (n_dim % 16)` block. The kernel reads it strictly forward.
pub struct PackedB {
    data: Vec<f32>,
    pub k_dim: usize,
    pub n_dim: usize,
}

impl PackedB {
    pub fn zeroed(k_dim: usize, n_dim: usize) -> Self {
        Self { data: vec![0f32; k_dim * n_dim], k_dim, n_dim }
    }

    /// Pack a row-major `(k_dim, n_dim)` matrix.
    pub fn from_row_major(b: &[f32], k_dim: usize, n_dim: usize) -> Self {
        assert_eq!(b.len(), k_dim * n_dim);
        let mut packed = Self::zeroed(k_dim, n_dim);
        for kk in 0..k_dim {
            for nn in 0..n_dim {
                packed.set(kk, nn, b[kk * n_dim + nn]);
            }
        }
        packed
    }

    fn n_main(&self) -> usize {
        self.n_dim - self.n_dim % NR
    }

    /// Write element `(kk, nn)` of the logical `(k_dim, n_dim)` matrix.
    /// Per-element layout math — fine for load-time packing; the per-turn
    /// attention fills use `fill_row` / `fill_col` instead.
    #[inline]
    pub fn set(&mut self, kk: usize, nn: usize, value: f32) {
        let n_main = self.n_main();
        let idx = if nn < n_main {
            (nn / NR) * self.k_dim * NR + kk * NR + nn % NR
        } else {
            n_main * self.k_dim + kk * (self.n_dim - n_main) + (nn - n_main)
        };
        self.data[idx] = value;
    }

    /// Write logical row `kk` from a contiguous slice — 16-value copies per
    /// strip, so the fill vectorizes.
    pub fn fill_row(&mut self, kk: usize, row: &[f32]) {
        debug_assert_eq!(row.len(), self.n_dim);
        let n_main = self.n_main();
        let mut n0 = 0;
        let mut strip_off = 0;
        while n0 < n_main {
            self.data[strip_off + kk * NR..strip_off + kk * NR + NR]
                .copy_from_slice(&row[n0..n0 + NR]);
            strip_off += self.k_dim * NR;
            n0 += NR;
        }
        let n_tail = self.n_dim - n_main;
        if n_tail > 0 {
            let base = n_main * self.k_dim + kk * n_tail;
            self.data[base..base + n_tail].copy_from_slice(&row[n_main..]);
        }
    }

    /// Write logical column `nn` from a contiguous slice — the layout base
    /// is computed once, so the loop is plain strided stores.
    pub fn fill_col(&mut self, nn: usize, col: &[f32]) {
        debug_assert_eq!(col.len(), self.k_dim);
        let n_main = self.n_main();
        if nn < n_main {
            let base = (nn / NR) * self.k_dim * NR + nn % NR;
            for (kk, &v) in col.iter().enumerate() {
                self.data[base + kk * NR] = v;
            }
        } else {
            let n_tail = self.n_dim - n_main;
            let base = n_main * self.k_dim + (nn - n_main);
            for (kk, &v) in col.iter().enumerate() {
                self.data[base + kk * n_tail] = v;
            }
        }
    }
}

/// `c[r][o] = bias[o] + Σ_k a[r][k] · b[k][o]` — `a` is (rows × k_dim)
/// row-major, `b` packed, `c` (rows × n_dim) row-major.
pub fn gemm_bias(rows: usize, a: &[f32], b: &PackedB, bias: &[f32], c: &mut [f32]) {
    debug_assert_eq!(a.len(), rows * b.k_dim);
    debug_assert_eq!(bias.len(), b.n_dim);
    debug_assert_eq!(c.len(), rows * b.n_dim);

    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: the required features were just detected on this CPU.
        unsafe { avx::gemm_bias_avx2(rows, b.k_dim, b.n_dim, a, &b.data, bias, c) };
        return;
    }

    gemm_bias_portable(rows, b.k_dim, b.n_dim, a, &b.data, bias, c);
}

/// The safe autovectorized kernel — every target without AVX2+FMA, and the
/// bit-identical reference the intrinsics path is tested against.
///
/// `bp` is the packed data as a plain slice, not `&PackedB`: a slice
/// parameter carries `noalias`, while a Vec data pointer loaded through a
/// struct reference does not — and without it LLVM refuses to vectorize
/// this kernel. Measured 6.5× on NEON (packbisect, 2026-08-20).
fn gemm_bias_portable(
    rows: usize,
    k_dim: usize,
    n_dim: usize,
    a: &[f32],
    bp: &[f32],
    bias: &[f32],
    c: &mut [f32],
) {
    let n_main = n_dim - n_dim % NR;
    let mut n0 = 0;
    let mut strip_off = 0;
    while n0 < n_main {
        let strip = &bp[strip_off..strip_off + k_dim * NR];
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
                let brow: &[f32; NR] = strip[k * NR..].first_chunk().unwrap();
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
                let brow: &[f32; NR] = strip[k * NR..].first_chunk().unwrap();
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
        strip_off += k_dim * NR;
        n0 += NR;
    }

    column_tail(rows, k_dim, n_dim, n_main, a, &bp[strip_off..], bias, c);
}

/// Column tail over the packed `k × n_tail` block: the policy head is 90
/// wide, leaving 10 columns here, and the attention score panel is 52 wide,
/// leaving 4. The inner loop has a runtime trip count, so it vectorizes
/// worse than the main tile — acceptable for these widths. Shared by both
/// kernels, so the tail cannot drift between them.
fn column_tail(
    rows: usize,
    k_dim: usize,
    n_dim: usize,
    n_main: usize,
    a: &[f32],
    tail: &[f32],
    bias: &[f32],
    c: &mut [f32],
) {
    let n_tail = n_dim - n_main;
    if n_tail == 0 {
        return;
    }
    for m0 in 0..rows {
        let arow = &a[m0 * k_dim..(m0 + 1) * k_dim];
        let mut acc = [0f32; NR];
        for k in 0..k_dim {
            let brow = &tail[k * n_tail..(k + 1) * n_tail];
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

#[cfg(target_arch = "x86_64")]
mod avx {
    use super::{column_tail, MR, NR};
    use std::arch::x86_64::*;

    /// The MR=4 × NR=16 tile in AVX2+FMA intrinsics: 8 ymm accumulators,
    /// two B vectors and one A broadcast live per `k` step, and the packed
    /// strip makes both B loads sequential. Loop order, accumulation order,
    /// and the bias add match `gemm_bias_portable` operation for operation,
    /// which is what makes the result bit-identical.
    ///
    /// # Safety
    /// The caller must have detected `avx2` and `fma` on the running CPU.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn gemm_bias_avx2(
        rows: usize,
        k_dim: usize,
        n_dim: usize,
        a: &[f32],
        bp: &[f32],
        bias: &[f32],
        c: &mut [f32],
    ) {
        let n_main = n_dim - n_dim % NR;
        let mut n0 = 0;
        let mut strip_off = 0;
        while n0 < n_main {
            let strip = bp.as_ptr().add(strip_off);
            let bias0 = _mm256_loadu_ps(bias.as_ptr().add(n0));
            let bias1 = _mm256_loadu_ps(bias.as_ptr().add(n0 + 8));
            let mut m0 = 0;
            while m0 + MR <= rows {
                let mut acc = [[_mm256_setzero_ps(); 2]; MR];
                for k in 0..k_dim {
                    let bpk = strip.add(k * NR);
                    let b0 = _mm256_loadu_ps(bpk);
                    let b1 = _mm256_loadu_ps(bpk.add(8));
                    for i in 0..MR {
                        let av = _mm256_set1_ps(*a.get_unchecked((m0 + i) * k_dim + k));
                        acc[i][0] = _mm256_fmadd_ps(av, b0, acc[i][0]);
                        acc[i][1] = _mm256_fmadd_ps(av, b1, acc[i][1]);
                    }
                }
                for (i, row) in acc.iter().enumerate() {
                    let cp = c.as_mut_ptr().add((m0 + i) * n_dim + n0);
                    _mm256_storeu_ps(cp, _mm256_add_ps(row[0], bias0));
                    _mm256_storeu_ps(cp.add(8), _mm256_add_ps(row[1], bias1));
                }
                m0 += MR;
            }
            // Row tail, one token at a time — same layers as the portable
            // kernel's tail.
            while m0 < rows {
                let mut acc0 = _mm256_setzero_ps();
                let mut acc1 = _mm256_setzero_ps();
                for k in 0..k_dim {
                    let bpk = strip.add(k * NR);
                    let av = _mm256_set1_ps(*a.get_unchecked(m0 * k_dim + k));
                    acc0 = _mm256_fmadd_ps(av, _mm256_loadu_ps(bpk), acc0);
                    acc1 = _mm256_fmadd_ps(av, _mm256_loadu_ps(bpk.add(8)), acc1);
                }
                let cp = c.as_mut_ptr().add(m0 * n_dim + n0);
                _mm256_storeu_ps(cp, _mm256_add_ps(acc0, bias0));
                _mm256_storeu_ps(cp.add(8), _mm256_add_ps(acc1, bias1));
                m0 += 1;
            }
            strip_off += k_dim * NR;
            n0 += NR;
        }

        column_tail(rows, k_dim, n_dim, n_main, a, &bp[strip_off..], bias, c);
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

    /// Every (rows, k, n) the network actually uses: embedder, the
    /// attention projections, both FF layers, the per-head attention
    /// products (the 52-wide score panel exercises the 4-column tail), the
    /// policy head (its 90 exercises the 10-column tail), the value head
    /// and temporal MLPs (row tails at rows=1).
    const GRAPH_SHAPES: &[(usize, usize, usize)] = &[
        (49, 351, 384),
        (52, 384, 384),
        (52, 384, 1536),
        (52, 1536, 384),
        (52, 48, 52),
        (52, 52, 48),
        (49, 384, 90),
        (1, 384, 128),
        (1, 512, 512),
        (1, 512, 384),
    ];

    #[test]
    fn matches_the_naive_product_on_the_graph_shapes() {
        for &(rows, k, n) in GRAPH_SHAPES {
            let a = pseudo(rows * k, 7 + rows as u32);
            let b = pseudo(k * n, 11 + k as u32);
            let bias = pseudo(n, 13 + n as u32);
            let want = naive(rows, k, n, &a, &b, &bias);
            let packed = PackedB::from_row_major(&b, k, n);
            let mut got = vec![0f32; rows * n];
            gemm_bias(rows, &a, &packed, &bias, &mut got);
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
        let b = PackedB::zeroed(k, n);
        let bias: Vec<f32> = (0..n).map(|j| j as f32).collect();
        let mut c = vec![9f32; rows * n];
        gemm_bias(rows, &a, &b, &bias, &mut c);
        for i in 0..rows {
            for j in 0..n {
                assert_eq!(c[i * n + j], j as f32);
            }
        }
    }

    /// `set` must place every element where the kernel reads it back from:
    /// packing via `set` equals packing via `from_row_major` by construction,
    /// so this pins the layout with a strip count, a tail, and both fills.
    #[test]
    fn set_agrees_with_from_row_major() {
        let (k, n) = (5usize, 52usize); // three strips and a 4-column tail
        let b = pseudo(k * n, 3);
        let packed = PackedB::from_row_major(&b, k, n);
        let mut filled = PackedB::zeroed(k, n);
        for nn in 0..n {
            for kk in 0..k {
                filled.set(kk, nn, b[kk * n + nn]);
            }
        }
        assert_eq!(packed.data, filled.data);

        let mut by_row = PackedB::zeroed(k, n);
        for kk in 0..k {
            by_row.fill_row(kk, &b[kk * n..(kk + 1) * n]);
        }
        assert_eq!(packed.data, by_row.data);

        let mut by_col = PackedB::zeroed(k, n);
        let mut col = vec![0f32; k];
        for nn in 0..n {
            for kk in 0..k {
                col[kk] = b[kk * n + nn];
            }
            by_col.fill_col(nn, &col);
        }
        assert_eq!(packed.data, by_col.data);
    }

    /// The dispatch must never change the answer: on an AVX2+FMA host the
    /// intrinsics tiles produce the same bits as the portable kernel. On
    /// other hosts (the arm64 dev box) this reduces to portable == portable.
    #[test]
    fn avx2_matches_the_portable_kernel_bitwise() {
        for &(rows, k, n) in GRAPH_SHAPES {
            let a = pseudo(rows * k, 17 + rows as u32);
            let b = pseudo(k * n, 19 + k as u32);
            let bias = pseudo(n, 23 + n as u32);
            let packed = PackedB::from_row_major(&b, k, n);
            let mut want = vec![0f32; rows * n];
            gemm_bias_portable(rows, k, n, &a, &packed.data, &bias, &mut want);
            let mut got = vec![0f32; rows * n];
            gemm_bias(rows, &a, &packed, &bias, &mut got);
            for i in 0..rows * n {
                assert_eq!(
                    got[i].to_bits(),
                    want[i].to_bits(),
                    "rows={rows} k={k} n={n} at {i}: {} vs {}",
                    got[i],
                    want[i]
                );
            }
        }
    }
}
