# Morpheus-rs belief filter

The particle filter over hidden enemy state, ported at milestone M4 of the
[rewrite plan](rewrite-plan.md). Five Python modules — `belief.py`,
`proposal.py`, `recovery.py`, `reservoir.py`, `particle_summary.py` — plus the
injected RNG the plan's §5 specified but nothing had needed until now.

The measurement that justified porting this first:
[`morpheus-rs-belief-bench.md`](../../research/measurements/morpheus-rs-belief-bench.md).
How it is proved equal: [`parity-harness.md`](parity-harness.md).

## What the filter is

A particle is a **complete** board — everything fog hides included. The
likelihood is not a score but a predicate: emit the observation this particle
implies, and either it is the frame the engine actually sent or the particle is
impossible. `filter_step` applies that test once per particle per turn.

Three consequences follow, and all three shape the port:

- **Nothing is approximately right.** A refuted particle gets weight zero, not
  a small weight. So the transition kernel and the fog rules M1 proved
  bit-exact are not merely inputs to the belief — a fog rule off by one cell
  turns a correct particle into a rejected one, and eight of those in a row is
  a collapse.
- **The whole belief can die in one frame.** That is what `recovery.py` is
  for, and why the expensive path is the one that matters (below).
- **Every branch is a transition.** Advancing eight particles, replaying an
  eight-deep history across a beam of eight, checking whether an enemy action
  would have changed our vision — all of it is the M1 kernel, called again.

## Where the time went

The runtime charges two blocks, and the port is measured against the same two,
over 1,328 recorded beliefs at the shipped `n_particles = 8`:

| block | what it runs | ×p50 | ×p99 |
| --- | --- | ---: | ---: |
| `belief_proposal` | `propose_enemy_actions` | 9.0× | 10.2× |
| `particle_transitions`, healthy | `filter_step` | **30.6×** | **23.8×** |
| `particle_transitions`, collapsed | `filter_step` + `recover_belief` | **46.6×** | **39.6×** |

The plan's §7 expected 10–50× for the transition kernel and "large, mostly
inherited" for the filter around it. Both hold.

**The two `particle_transitions` rows are the same component**, and separating
them answered a question M0 could not. `runtime.py` puts the filter and the
recovery it may trigger inside one charge, so the M0 baseline's 142.6 ms p99
was a mixture — of a **1.1 ms** path and a **127 ms** one. The baseline's p99 is
a recovery p99 wearing the component's name, and M7 should budget the two
separately rather than reserve for the blend.

`belief_proposal`'s 9× is the outlier, and it is not the sampling. Under the
deployed uniform proposal the work is one legal mask and one SHA-256
information key per particle, and the hash is the floor: it is already C in
Python. Nothing here is worth optimizing — 0.4 ms against a 140 ms deadline —
but it is the one place in the belief layer where the rewrite's premise does
not hold.

Figures and method:
[`morpheus-rs-belief-bench.md`](../../research/measurements/morpheus-rs-belief-bench.md).

## Particles share; they do not copy

The Python rebuilds a frozen `Particle` on every weight change and lets
refcounting share the arrays underneath. A literal port would `memcpy` a 5 KB
state and a 5 KB memory on each rebuild, and `resample` rebuilds the entire
set every time it fires.

So `Particle` holds `Rc<GameState>`, `Rc<VisibleMemory>`, and
`Vec<Rc<HistoryFrame>>`. This is not a liberty: the values are genuinely
immutable — a state is built by `transition` and never touched again — so `Rc`
reproduces the Python's sharing semantics exactly rather than approximating
them.

It also answers M1's open note, which asked M4 to *measure* before assuming the
per-call state clone was free. The transition still builds one state per
particle per turn. Nothing downstream duplicates it.

## The RNG is injected, and the injection is the test

Reproducing NumPy's bit generator in Rust would be fragile and worthless at
play time, so §5 chose record-and-replay instead. `rng::Rng` has one method per
NumPy call site — `integers`, `choice`, `random`, with NumPy's own
`size: None` versus `size: Some(1)` distinction preserved — and two
implementations:

- `Replay` walks a recorded stream and **verifies every argument** before
  handing back the recorded result. A port that samples from a population of a
  different size, or samples one extra time, fails on that call with the draw
  index and both signatures in the message.
- `SmallRng` (xoshiro256++) is what plays. It is never compared against NumPy.

The consequence is deliberate: **draw-site order is part of the ported
contract**. It is also why the `propose` parity surface compares the
*distributions* as well as the sampled actions — `Replay` returns the oracle's
index whatever this side computed, so checking only the actions would pass a
port whose every probability was wrong.

## Two NumPy behaviours are reproduced on purpose

Both have their own parity surface, because both are host-conditional and a
silent divergence in either would surface a long way from its cause.

### `np.sum` is pairwise

NumPy reduces `f64` arrays in eight interleaved lanes below 128 elements and
recurses in blocks above, so `iter().sum()` disagrees in the last bits from
eight elements onward. `ess()` divides two of those sums and compares the
result against a threshold that decides whether a resample runs — and a
resample **consumes a draw**. A last-bit difference is therefore not a rounding
difference; it desynchronizes the replay stream and every decision after it.

`rng::npsum` transcribes the reduction. `rng::normalize_weights` deliberately
does *not* use it: the Python writes the builtin `sum` over a generator there,
which is a different function with a different answer.

### `np.argsort` is an unstable introsort

`top_legal_actions` ranks the enemy proposal by descending probability. Under
the deployed uniform proposal **every legal action carries the same float**, so
the order recovery walks is decided entirely by how the sort breaks ties — and
NumPy's default is median-of-three quicksort with an insertion-sort floor and a
heapsort depth limit, whose tie order matches neither a stable sort nor
anything one would write from scratch. `rng::argsort_desc_numpy` transcribes
it.

**This one is host-conditional, and that is the oracle's property, not the
port's.** NumPy ≥ 2.0 dispatches `argsort` on 64-bit dtypes to `x86-simd-sort`
when the CPU has AVX-512-SKX — a different, also unstable algorithm. The M0 CPU
probe found Modal hosts both with and without AVX-512, so on some x86 hosts the
*Python bot itself* would order these candidates differently. The `argsort`
parity surface exists so that a host which dispatches elsewhere fails with a
named cause instead of surfacing as a recovery mismatch three layers up.

## Recovery, in the order it tries things

`recover_belief` runs only when *every* particle failed the observation test. A
partial survival is not a collapse, and reconstructing on one would discard the
particles that were right.

1. **Rejuvenate.** Replay the last `recovery_lag` turns with a bounded beam
   over alternative enemy actions, accepting only histories that reproduce
   every stored observation. The only path that recovers information rather
   than inventing a plausible board.
2. **Rewind one turn and enumerate.** Top-ranked policy actions plus every
   action that would have changed what *we* can see. That asymmetry is the
   point — `is_vision_changing` measures our own footprint, so an enemy step
   into empty ground is invisible to it and an enemy step that takes one of our
   cells is not. The second kind is what explains a collapse.
3. **Maximum-entropy reconstruction.** Give up on history: uniform general over
   hidden cells, the enemy's army spread evenly over the land the public totals
   require. Every candidate still has to reproduce the observation exactly, so
   "least committed" never means "illegal". The result is marked `collapsed`,
   which drops the reported ESS to `1/n` so the search knows it is holding a
   guess.

If all three fail the previous belief is kept, unchanged. Recovery never raises
and never replaces a valid belief with an invalid one.

`recover_belief` takes a `my_action` in the Python and never reads it — every
replay uses the action stored on the history frame, which is the one that
actually produced the observation being explained. The port drops the
parameter rather than carrying it as `_my_action`.

## The one place play-time answers legitimately differ

`ParticleReservoir::replace_from_belief` resamples with a generator the Python
builds *inside the method*: `np.random.default_rng(0)`. That the draws do not
come from the shared stream is a real property — a search node's contents must
not depend on how much searching happened before it — and the port preserves it
by taking the generator as an argument.

What the port does **not** reproduce is which particles that generator picks.
Doing so would mean reimplementing PCG64 and NumPy's `choice`, which §5
explicitly declined. The inputs are equal-weight after `normalize_weights`, so
both implementations draw from the same distribution and differ only in the
sample. The parity harness injects the oracle's recorded draws — patching the
factory for the duration, so the draws land in the same ordered stream — and
therefore checks this exactly. In play the two bots make different, equally
valid picks.

Nothing in M4 drives the reservoir; the search does, at M5. It is ported here
because §4 assigns it to this milestone and because it is the last piece of the
belief layer M5 would otherwise have to stop and write.

## What M4 does not do

The bot still plays M1's first-legal-move. The belief exists, is correct, and
is fast; nothing calls it in the *playing* path yet, because the decision needs
`search.py` and `tactics.py` (M5). `particle_summary` closes the join M2 left
open — `build_tensor` has taken a `BeliefSummary` as an input since then, fed
from the corpus — so the tensor path is now complete on the Rust side alone.
