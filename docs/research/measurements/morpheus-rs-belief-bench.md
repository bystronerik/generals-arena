# Morpheus-rs belief update: Python vs Rust

The M4 exit-gate measurement of [the rewrite plan](../../bots/morpheus-rs/rewrite-plan.md),
which asks for "measured particle-transitions time at n=8 (expect ≥10× vs
Python)". Both implementations run over the **same recorded beliefs** from the
M0 corpus. What the belief layer is and why it is shaped this way:
[`belief.md`](../../bots/morpheus-rs/belief.md).

Reproduce with:

```bash
python bots/morpheus-rs/tools/bench_belief.py --corpus data/morpheus/morpheus-rs/morpheus-rs-m0 --iters 5
```

Raw quantiles: [`morpheus-rs-belief-bench.json`](morpheus-rs-belief-bench.json).

## Method, and three choices in it

**Same inputs, not same games.** The [M0 baseline](morpheus-rs-baseline.md)
measured `particle_transitions` under live play, where the deadline controller
decides how often the update runs at all and the host decides how fast. That
number answers *what does a turn cost*. This one answers *what does the kernel
cost*, which is the question a port can be held to. Both belong in M7's
re-derivation; neither substitutes for the other.

**The timed unit is what the runtime charges.** `runtime.py` puts `filter_step`
and the `recover_belief` it may trigger inside a single `particle_transitions`
charge, so the baseline's 142.6 ms p99 is a mixture of a cheap path and an
expensive one. Each belief here is therefore timed against **two** target
frames:

- one built by advancing its leading particle, which at least that particle can
  explain — the healthy path, all filter;
- the turn's own recorded observation, which refutes every particle — the
  collapse path, where recovery runs.

The split is the useful artifact. Anyone re-deriving the budget needs to know
which of the two the 142.6 ms was.

**Minimum per belief, distribution across beliefs.** Repeating one belief and
taking a p99 measures the machine's scheduling noise. Taking the minimum per
belief and quantiles *across* beliefs measures the code against the range of
inputs it actually sees.

The two implementations are also checked for agreement on how many particles
survived each target, so a ratio can never be reported between two different
computations. (Parity proper is
[the harness](../../bots/morpheus-rs/parity-harness.md); this is a guard against
benchmarking a divergence.)

## Results

M3 Pro, single-threaded, 1,328 recorded beliefs — every one of them **n = 8**,
the shipped `n_particles`. Five iterations per belief; minimum per belief,
quantiles across beliefs. Zero targets where the two implementations kept a
different number of particles.

| block | Python p50 | p99 | Rust p50 | p99 | ×p50 | ×p99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `belief_proposal` | 3.208 | 4.069 | 0.357 | 0.397 | **9.0×** | **10.2×** |
| `particle_transitions`, filter only | 0.974 | 1.134 | 0.032 | 0.048 | **30.6×** | **23.8×** |
| `particle_transitions`, with recovery | 2.380 | 127.122 | 0.051 | 3.210 | **46.6×** | **39.6×** |

All figures in milliseconds. The plan's M4 exit gate asks for ≥10× on
`particle_transitions`; the measured range is 24–47× depending on which path,
and §7's estimate for the transition kernel was 10–50×.

### The 142 ms in the M0 baseline was recovery

This is the result worth carrying into M7. The
[M0 baseline](morpheus-rs-baseline.md) measured `particle_transitions` at
**142.6 ms p99** in live play and could not say what that time was. Split here,
the answer is unambiguous: the filter alone is ~1 ms at p99, and the same block
with recovery is **127 ms**. The baseline's p99 is a recovery p99 wearing the
component's name.

That reframes the budget. `deployment.json` reserves for a component whose cost
distribution is bimodal by a factor of a hundred, and a knob fitted to the
mixture is fitted to how often recovery happened to fire on the capture host.
M7 should budget the two paths separately.

### `belief_proposal` is the outlier, and it is not the sampling

9× against 24–47× elsewhere, and the reason is that the Python was never the
bottleneck here. Under the deployed uniform proposal the work per particle is
one legal mask and one SHA-256 information key, and the hash already runs in C.
At 0.4 ms against a 140 ms deadline none of this is worth optimizing; it is
recorded because it is the one place in the belief layer where the rewrite's
premise does not hold.

### The tail is where the rewrite pays

Look at `max` rather than `p50`: the Python's recovery path reaches 128.5 ms on
a single belief — most of an internal deadline for one component — against the
Rust's 4.2 ms. The mean tells the same story less dramatically (22.0 ms against
0.57 ms, 38.7×). Rewrite-plan §7 predicted that the filter's gain would be
"large, mostly inherited from the transition kernel", and the inheritance is
visible: rejuvenation replays an eight-deep history across a beam of eight, and
every step of that is the M1 kernel.

## What this does not settle

- **One host.** These are M3 Pro (arm64) numbers. M3's addendum is the standing
  warning: a depthwise layer that looked structurally bounded on arm64 gave up a
  third of its time on x86 to a memory-layout change, and three formulations
  compared on arm64 alone had been written up as evidence. An arm64-only profile
  is half a measurement.
- **Not a deployment budget.** `deployment.json` is fitted to live per-turn
  costs on a qualification host, and both of those are still M7's problem. What
  is settled here is the ratio between two implementations of the same
  function on the same inputs.
- **The proposal path measured is the deployed one.** `use_policy_proposal` is
  `false`, so `belief_proposal` here is a legal mask and an information key per
  particle, with no network forward. Turning the policy proposal on would move
  that row entirely.
