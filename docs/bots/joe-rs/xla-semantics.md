# XLA float semantics the port mirrors

Two places where "the same arithmetic in Rust" is not what the deployed
Python joe computes, found when tier-1 parity failed by 1 ULP and fixed by
reading the oracle instead of guessing (2026-08-14). Both live in
`bots/joe-rs/src/xla_math.rs`.

## Division by a constant is a reciprocal multiply

XLA's algebraic simplifier rewrites `x / c` to `x * (1/c)` with `1/c` folded
in f32. Measured directly: `jnp.arange(21)/(21-1)` matches
`j * (1f32/20f32)` bit-for-bit and *not* correctly-rounded division; same
for every `/50` site and `%50` (which matches trunc-mod). Every constant
division in `obs.rs`/`net.rs` is therefore a `RECIP_*` multiply.

## log1p is XLA's own polynomial, with backend FMA contraction

Neither libm's `log1pf` (575/65537 integer inputs differ) nor Eigen's
`plog` variants matched. The truth came from two dumps:

- `XLA_FLAGS=--xla_dump_to=...` on `jax.jit(jnp.log1p)` shows the
  `xla.log1p.v8f32` expansion: `|x| < sqrt(2)-1` takes a rational
  small-path, else Cephes' mantissa polynomial in three interleaved strands
  with the tail `((x - 0.5x²) + (poly·x³ + e·q1)) + e·q2` — plain
  fmul/fadd in the IR, constants pinned by bit pattern.
- The dumped **object file** disassembles to 106 `fmla`/`fmls`: XLA
  compiles with FP contraction on, so the backend fuses every *single-use*
  fmul into its consuming fadd/fsub (first operand preferred when both are
  fmuls). The Rust port reproduces exactly that fusion pattern with
  `f32::mul_add` — `x²`/`x³` are multi-use and stay unfused.

With both mirrored, `log1p(counter) * (1/5)` (channel 21) is bit-exact over
the whole integer domain 0..16384 —
`test_parity.py::test_channel21_log1p_exhaustive` re-derives the oracle
from live JAX on every run, so an XLA upgrade that changes the lowering is
caught rather than silently absorbed.

## Scope, and what it means for the rest of the pipeline

FP contraction is global in XLA-emitted code, but the rest of the obs
pipeline is immune: its multiplies are by exact 0/1 indicator channels or
feed no add, so fusion cannot change any rounding. That is why every other
augmented channel is bit-exact with plain Rust ops — confirmed empirically
by the sequence gate over every turn of every corpus game. The candle
forward pass never needed this treatment; it lives in tier 2's tolerance.

Both dev (arm64) and the compile target (x86-64-v3) have hardware FMA, so
`mul_add` is a real fused instruction on both — and these mirrors hold on
the platform where the fixtures and the arena referee run.
