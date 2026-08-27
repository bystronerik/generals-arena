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
//! **Three paths, one answer.** On x86-64 the dispatcher prefers AVX-512F
//! (one zmm register covers a whole 16-column strip), then AVX2+FMA;
//! everywhere else — the aarch64 development Macs included — the portable
//! safe kernel runs and LLVM autovectorizes it. The compile target stays
//! `x86-64-v3` because the M0 CPU probe
//! (`docs/research/measurements/morpheus-rs-cpu-probe.md`) saw v3 hosts in
//! the fleet next to the v4 majority, so AVX-512 exists only behind runtime
//! detection. Every path keeps exactly one accumulator per output element
//! and walks `k` in order, so their outputs are bit-identical — to each
//! other and to the pre-packing kernels they replaced; the
//! `*_matches_the_portable_kernel_bitwise` tests hold the paths together.
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
    /// attention fills use `fill_row` / `fill_from_cols` instead.
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

    /// Fill the whole panel from a column-major source: logical column
    /// `nn` is `src[nn * src_stride..][..k_dim]`. This is the transpose the
    /// attention K pack needs — one head's K is 48 of the 384 values in
    /// each of 52 rows, and the score panel wants those 52 rows as columns.
    ///
    /// It writes strip by strip, so every store is `NR` contiguous values.
    /// Filling the same panel one column at a time stores one value every
    /// `NR` instead, and 52 columns × 48 scattered stores is what the head
    /// loop used to pay: the blocked form measured 1.6× on the arm64 dev
    /// box, and both forms write the identical bytes.
    pub fn fill_from_cols(&mut self, src: &[f32], src_stride: usize) {
        debug_assert!(self.n_dim == 0 || src.len() >= (self.n_dim - 1) * src_stride + self.k_dim);
        let n_main = self.n_main();
        let mut strip_off = 0;
        let mut n0 = 0;
        while n0 < n_main {
            for kk in 0..self.k_dim {
                let dst = &mut self.data[strip_off + kk * NR..strip_off + kk * NR + NR];
                for (j, d) in dst.iter_mut().enumerate() {
                    *d = src[(n0 + j) * src_stride + kk];
                }
            }
            strip_off += self.k_dim * NR;
            n0 += NR;
        }
        let n_tail = self.n_dim - n_main;
        for kk in 0..self.k_dim {
            let base = n_main * self.k_dim + kk * n_tail;
            for (j, d) in self.data[base..base + n_tail].iter_mut().enumerate() {
                *d = src[(n_main + j) * src_stride + kk];
            }
        }
    }
}

/// The kernel `gemm_bias` dispatches to on this host, by name — `selfcheck`
/// reports it, because the fleet mixes v3 and v4 hosts and nothing else
/// makes the runtime choice visible. Must stay the mirror of the dispatch
/// order in `gemm_bias` below.
pub fn kernel_name() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            return "avx512";
        }
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return "avx2";
        }
    }
    "portable"
}

/// `c[r][o] = bias[o] + Σ_k a[r][k] · b[k][o]` — `a` is (rows × k_dim)
/// row-major, `b` packed, `c` (rows × n_dim) row-major.
pub fn gemm_bias(rows: usize, a: &[f32], b: &PackedB, bias: &[f32], c: &mut [f32]) {
    debug_assert_eq!(a.len(), rows * b.k_dim);
    debug_assert_eq!(bias.len(), b.n_dim);
    debug_assert_eq!(c.len(), rows * b.n_dim);

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            // SAFETY: avx512f was just detected on this CPU.
            unsafe { avx512::gemm_bias_avx512(rows, b.k_dim, b.n_dim, a, &b.data, bias, c) };
            return;
        }
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: the required features were just detected on this CPU.
            unsafe { avx::gemm_bias_avx2(rows, b.k_dim, b.n_dim, a, &b.data, bias, c) };
            return;
        }
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
///
/// `#[inline(never)]` is the second half of that same trap. `Net::forward`
/// reaches this through eleven `Linear::forward_into` sites inside the block
/// loop, and under `lto = "fat"` + `codegen-units = 1` LLVM inlines every one
/// of them into that single enormous function. The tile's `MR * NR`
/// accumulators then compete for vector registers with the rest of the
/// forward pass and get spilled. Keeping the kernel out of line restores the
/// standalone codegen: on the arm64 dev box unclejoe's forward drops 64.4 -> 39.1 ms
/// p50 (1.65x, A/B/A/B interleaved), and every GEMM step in
/// `bench --forward-stages` improves — the four 384x384 projections 311 ->
/// 188 us a call, `ff1` 1319 -> 777.
///
/// It is deliberately **not** on the two intrinsics kernels, which cannot hit
/// this: a `#[target_feature]` function is never inlined into a caller
/// without those features, so `gemm_bias` already calls them out of line. The
/// x86 asm for the whole crate is instruction-identical with and without this
/// attribute — only panic-location line numbers move — so this is an arm64
/// development fix with no effect on the competition host.
#[inline(never)]
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
/// leaving 4. Shared by all three kernels, so the tail cannot drift between
/// them.
///
/// The width decides which form runs. [`column_tail_dyn`] takes it as a
/// runtime value, and a runtime trip count on the innermost loop leaves one
/// row's `k` chain — 48 or 384 dependent FMAs — with nothing else in flight.
/// That is why the 52 × 48 × 52 score panel spent a third of its time on
/// 7.7 % of its FLOPs. [`column_tail_n`] takes the width as a constant and
/// blocks `MR` rows, so the width unrolls and `MR` chains run at once: the
/// 4-wide tail measured 2.9× and the 10-wide tail 1.7× on the arm64 dev box.
/// The graph has exactly these two widths; any other still runs the dynamic
/// form, and `tail_widths_agree_bitwise` holds the two forms together.
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
    match n_dim - n_main {
        0 => {}
        4 => column_tail_n::<4>(rows, k_dim, n_dim, n_main, a, tail, bias, c),
        10 => column_tail_n::<10>(rows, k_dim, n_dim, n_main, a, tail, bias, c),
        _ => column_tail_dyn(rows, k_dim, n_dim, n_main, a, tail, bias, c),
    }
}

/// The tail at a compile-time width, `MR` rows at a time. Every output
/// element still keeps one accumulator and walks `k` in order, exactly as
/// [`column_tail_dyn`] does, so the two forms agree bit for bit.
fn column_tail_n<const NT: usize>(
    rows: usize,
    k_dim: usize,
    n_dim: usize,
    n_main: usize,
    a: &[f32],
    tail: &[f32],
    bias: &[f32],
    c: &mut [f32],
) {
    let mut m0 = 0;
    while m0 + MR <= rows {
        let mut acc = [[0f32; NT]; MR];
        let block = &a[m0 * k_dim..(m0 + MR) * k_dim];
        for k in 0..k_dim {
            // `&[f32; NT]`, not `&[f32]`: the length in the type is what
            // keeps the bounds check out of the innermost loop.
            let brow: &[f32; NT] = tail[k * NT..].first_chunk().unwrap();
            for (i, row) in acc.iter_mut().enumerate() {
                let av = block[i * k_dim + k];
                for j in 0..NT {
                    row[j] = av.mul_add(brow[j], row[j]);
                }
            }
        }
        for (i, row) in acc.iter().enumerate() {
            let base = (m0 + i) * n_dim + n_main;
            for j in 0..NT {
                c[base + j] = row[j] + bias[n_main + j];
            }
        }
        m0 += MR;
    }
    // Row tail: 52 rows leave nothing over 4·13, 49 rows leave one.
    while m0 < rows {
        let arow = &a[m0 * k_dim..(m0 + 1) * k_dim];
        let mut acc = [0f32; NT];
        for k in 0..k_dim {
            let brow: &[f32; NT] = tail[k * NT..].first_chunk().unwrap();
            let av = arow[k];
            for j in 0..NT {
                acc[j] = av.mul_add(brow[j], acc[j]);
            }
        }
        let base = m0 * n_dim + n_main;
        for j in 0..NT {
            c[base + j] = acc[j] + bias[n_main + j];
        }
        m0 += 1;
    }
}

/// The tail at a runtime width — the fallback for any `n % 16` the graph
/// does not use, and the reference the specializations are tested against.
fn column_tail_dyn(
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

#[cfg(target_arch = "x86_64")]
mod avx512 {
    use super::{column_tail, NR};
    use std::arch::x86_64::*;

    /// Rows per AVX-512 tile. One zmm register covers a whole 16-column
    /// strip, so a row costs one accumulator instead of the AVX2 tile's
    /// two — the sweep result that MR above 4 loses does not carry over,
    /// because that was a 16-register ymm budget. 13 accumulators + one B
    /// vector + one broadcast use 15 of 32 zmm registers, 13 chains cover
    /// the FMA latency × throughput product on the two-pipe hosts, and 13
    /// divides 52: every token-stream GEMM runs with no row tail.
    const MR: usize = 13;

    /// The MR=13 × NR=16 tile in AVX-512F intrinsics. Loop order,
    /// accumulation order, and the bias add match `gemm_bias_portable`
    /// operation for operation — one fused chain per output element, `k` in
    /// order — which is what makes the result bit-identical.
    ///
    /// # Safety
    /// The caller must have detected `avx512f` on the running CPU.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn gemm_bias_avx512(
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
            let biasv = _mm512_loadu_ps(bias.as_ptr().add(n0));
            let mut m0 = 0;
            while m0 + MR <= rows {
                let mut acc = [_mm512_setzero_ps(); MR];
                for k in 0..k_dim {
                    let bv = _mm512_loadu_ps(strip.add(k * NR));
                    for i in 0..MR {
                        let av = _mm512_set1_ps(*a.get_unchecked((m0 + i) * k_dim + k));
                        acc[i] = _mm512_fmadd_ps(av, bv, acc[i]);
                    }
                }
                for (i, &accv) in acc.iter().enumerate() {
                    let cp = c.as_mut_ptr().add((m0 + i) * n_dim + n0);
                    _mm512_storeu_ps(cp, _mm512_add_ps(accv, biasv));
                }
                m0 += MR;
            }
            // Row tail, one token at a time. The rows=1 heads and the
            // 49-row embedder/policy GEMMs land here; a single chain is
            // latency-bound, but those GEMMs are ~1% of the FLOPs.
            while m0 < rows {
                let mut acc = _mm512_setzero_ps();
                for k in 0..k_dim {
                    let av = _mm512_set1_ps(*a.get_unchecked(m0 * k_dim + k));
                    acc = _mm512_fmadd_ps(av, _mm512_loadu_ps(strip.add(k * NR)), acc);
                }
                let cp = c.as_mut_ptr().add(m0 * n_dim + n0);
                _mm512_storeu_ps(cp, _mm512_add_ps(acc, biasv));
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

        // `fill_from_cols` reads the same matrix transposed, so hand it a
        // column-major copy: same panel, written strip by strip.
        let mut colmajor = vec![0f32; k * n];
        for nn in 0..n {
            for kk in 0..k {
                colmajor[nn * k + kk] = b[kk * n + nn];
            }
        }
        let mut by_col = PackedB::zeroed(k, n);
        by_col.fill_from_cols(&colmajor, k);
        assert_eq!(packed.data, by_col.data);
    }

    /// `column_tail` picks a specialization by width. Every width must give
    /// the dynamic form's bits back, or a shape change silently moves the
    /// answer — the specializations are a speed choice, never a numeric one.
    #[test]
    fn tail_widths_agree_bitwise() {
        let (rows, k) = (7usize, 9usize); // a row tail under MR=4 as well
        let a = pseudo(rows * k, 31);
        for n_tail in 1..NR {
            let n_dim = NR + n_tail; // one full strip, then this tail
            let n_main = NR;
            let tail = pseudo(k * n_tail, 37 + n_tail as u32);
            let bias = pseudo(n_dim, 41);
            let mut want = vec![0f32; rows * n_dim];
            column_tail_dyn(rows, k, n_dim, n_main, &a, &tail, &bias, &mut want);
            let mut got = vec![0f32; rows * n_dim];
            column_tail(rows, k, n_dim, n_main, &a, &tail, &bias, &mut got);
            for i in 0..rows * n_dim {
                assert_eq!(got[i].to_bits(), want[i].to_bits(), "n_tail={n_tail} at {i}");
            }
        }
    }

    /// One graph-shape sweep of `kernel` against the portable reference,
    /// bit for bit. `kernel` runs inside the caller's `unsafe` obligation:
    /// the caller detects the features first.
    #[cfg(target_arch = "x86_64")]
    fn assert_bitwise_matches_portable(
        name: &str,
        kernel: unsafe fn(usize, usize, usize, &[f32], &[f32], &[f32], &mut [f32]),
    ) {
        for &(rows, k, n) in GRAPH_SHAPES {
            let a = pseudo(rows * k, 17 + rows as u32);
            let b = pseudo(k * n, 19 + k as u32);
            let bias = pseudo(n, 23 + n as u32);
            let packed = PackedB::from_row_major(&b, k, n);
            let mut want = vec![0f32; rows * n];
            gemm_bias_portable(rows, k, n, &a, &packed.data, &bias, &mut want);
            let mut got = vec![0f32; rows * n];
            // SAFETY: forwarded from the caller, who detected the features.
            unsafe { kernel(rows, k, n, &a, &packed.data, &bias, &mut got) };
            for i in 0..rows * n {
                assert_eq!(
                    got[i].to_bits(),
                    want[i].to_bits(),
                    "{name}: rows={rows} k={k} n={n} at {i}: {} vs {}",
                    got[i],
                    want[i]
                );
            }
        }
    }

    /// Each intrinsics path the host can run, against the portable kernel,
    /// bitwise — direct calls, not the dispatcher, so an AVX-512 host still
    /// covers the AVX2 tiles it would never dispatch to. On the arm64 dev
    /// box this test is empty; the x86 CI moment is the Modal container.
    #[test]
    fn simd_paths_match_the_portable_kernel_bitwise() {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                assert_bitwise_matches_portable("avx2", avx::gemm_bias_avx2);
            }
            if std::arch::is_x86_feature_detected!("avx512f") {
                assert_bitwise_matches_portable("avx512", avx512::gemm_bias_avx512);
            }
        }
    }

    /// The dispatch must never change the answer: whatever path
    /// `gemm_bias` picks on this host produces the same bits as the
    /// portable kernel. On non-x86 hosts this reduces to portable ==
    /// portable.
    #[test]
    fn dispatch_matches_the_portable_kernel_bitwise() {
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
