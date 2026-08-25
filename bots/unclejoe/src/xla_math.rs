//! Bit-faithful mirrors of XLA CPU's f32 math on the sites the obs pipeline
//! uses (port-plan §4, M2 rule: mirror op semantics site by site).
//!
//! Two deviations from naive Rust, both read off the oracle rather than
//! assumed (see docs/bots/joe-rs/parity.md):
//!
//! * **Division by a constant is a reciprocal multiply.** XLA's algebraic
//!   simplifier rewrites `x / c` to `x * (1/c)`; Rust's `/` is correctly
//!   rounded division and differs by 1 ULP on some inputs. The `RECIP_*`
//!   constants below fold `1/c` exactly as XLA does.
//! * **`log1p` is XLA's own vectorized polynomial** (`xla.log1p.v8f32`),
//!   not libm's. The implementation below is transcribed op for op from the
//!   optimized LLVM IR that `jax.jit(jnp.log1p)` emits on this machine
//!   (plain mul/add — XLA emits no FMA here; constants taken bit-exactly
//!   from the IR). `tests/test_unclejoe_parity.py::test_channel21_log1p_exhaustive`
//!   sweeps the whole channel-21 input domain (integer turn counters)
//!   against the live JAX oracle, so an XLA upgrade that changes the
//!   lowering is caught, not silently absorbed.

pub const RECIP_50: f32 = 1.0 / 50.0;
pub const RECIP_5: f32 = 1.0 / 5.0;

/// XLA's f32 log on `u = 1 + x`, from the `xla.log1p.v8f32` expansion:
/// Cephes mantissa polynomial in three interleaved strands, then the tail
/// `((x ⊖ 0.5x²) + fma(poly, x³, e·q1)) ⊕ e·q2`.
///
/// The dumped LLVM IR shows plain fmul/fadd, but the object code XLA
/// actually runs carries `fmla`/`fmls` throughout: XLA compiles with FP
/// contraction on, so the backend fuses every *single-use* fmul into its
/// consuming fadd/fsub (first operand preferred). The `mul_add` calls below
/// reproduce exactly that fusion pattern — no more, no less. `x²` and `x³`
/// are multi-use and therefore stay unfused, as in the object code.
///
/// Valid for finite positive `u`; special-case masks (u ≤ 0, ±inf, nan)
/// are omitted because `1 + counter` cannot reach them.
fn plog_xla(u: f32) -> f32 {
    const MIN_NORMAL: u32 = 0x0080_0000;
    const SQRTHF: u32 = 0x3F3504F3;
    const P0: u32 = 0x3D9021BB;
    const P1: u32 = 0xBDEBD1B8;
    const P2: u32 = 0x3DEF251A;
    const P3: u32 = 0xBDFE5D4F;
    const P4: u32 = 0x3E11E9BF;
    const P5: u32 = 0xBE2AAE50;
    const P6: u32 = 0x3E4CCEAC;
    const P7: u32 = 0xBE7FFFFC;
    const P8: u32 = 0x3EAAAAAA;
    const Q1: u32 = 0xB95E8083;
    const Q2: u32 = 0x3F318000;
    let c = f32::from_bits;

    let clamped = if u > c(MIN_NORMAL) { u } else { c(MIN_NORMAL) };
    let bits = clamped.to_bits();
    let m = c((bits & 0x807F_FFFF) | 0x3F00_0000); // mantissa in [0.5, 1)
    let e1 = ((bits >> 23) as i32 - 127) as f32 + 1.0;
    let below = m < c(SQRTHF);
    let tm = if below { m } else { 0.0 };
    let e = e1 - if below { 1.0 } else { 0.0 };
    let x = (m - 1.0) + tm;

    let x2 = x * x; // multi-use: stays a plain multiply
    let x3 = x2 * x;
    let mut y = x.mul_add(c(P0), c(P1));
    let mut y1 = x.mul_add(c(P3), c(P4));
    let mut y2 = x.mul_add(c(P6), c(P7));
    y = y.mul_add(x, c(P2));
    y1 = y1.mul_add(x, c(P5));
    y2 = y2.mul_add(x, c(P8));
    y = y.mul_add(x3, y1);
    y = y.mul_add(x3, y2);

    // fadd(fmul(y, x3), fmul(e, q1)): the first-operand fmul wins the
    // contraction, the second stays a plain multiply.
    let s = y.mul_add(x3, e * c(Q1));
    // fsub(x, fmul(x2, 0.5)) -> fmls.
    let t = x2.mul_add(-0.5, x);
    e.mul_add(c(Q2), t + s)
}

/// XLA's f32 `log1p`: below `|x| < sqrt(2) - 1` a rational approximation,
/// above it `plog_xla(1 + x)` — a threshold select, not Kahan's u-trick.
/// Constants, evaluation order, and the FMA contraction pattern transcribed
/// from the same dump.
pub fn log1p(x: f32) -> f32 {
    const THRESHOLD: u32 = 0x3ED413CD; // ~0.41421357 (sqrt(2) - 1)
    let c = f32::from_bits;

    if x.abs() < c(THRESHOLD) {
        let x2 = x * x; // multi-use
        // Rational num/den, Horner with FMA; the `x * 0` lead terms are in
        // the IR (folded lead coefficients).
        let mut den = x.mul_add(0.0, 1.0);
        den = den.mul_add(x, c(0x417101AD));
        den = den.mul_add(x, c(0x42A6185B));
        den = den.mul_add(x, c(0x435DC32D));
        den = den.mul_add(x, c(0x439A8CA3));
        den = den.mul_add(x, c(0x43586D8A));
        den = den.mul_add(x, c(0x42707982));
        let mut num = x.mul_add(0.0, 4.527e-5);
        num = num.mul_add(x, c(0x3EFF40C5));
        num = num.mul_add(x, c(0x40D284FA));
        num = num.mul_add(x, c(0x41EF4B9C));
        num = num.mul_add(x, c(0x4273CC76));
        num = num.mul_add(x, c(0x426473AD));
        num = num.mul_add(x, c(0x41A05101));
        let q = num / den;
        let r = (x * x2) * q;
        x + x2.mul_add(-0.5, r)
    } else {
        plog_xla(1.0 + x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log1p_agrees_with_libm_to_one_ulp_on_integers() {
        // The bit-exact oracle is JAX (pytest sweep); libm within 1 ULP is
        // the standalone sanity check that the polynomial is right.
        for i in 0..100_000u32 {
            let x = i as f32;
            let got = log1p(x);
            let want = x.ln_1p();
            let d = (got.to_bits() as i64 - want.to_bits() as i64).abs();
            assert!(d <= 1, "log1p({x}): {got} vs libm {want} ({d} ULP)");
        }
    }

    #[test]
    fn log1p_zero_is_zero() {
        assert_eq!(log1p(0.0).to_bits(), 0.0f32.to_bits());
    }
}
