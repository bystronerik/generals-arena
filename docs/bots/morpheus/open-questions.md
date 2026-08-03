# Open questions

These choices are deliberately deferred. Each question has a current safe
default and the evidence required for a final value. None changes Morpheus's
strategic design.

## Inference runtime and exact capacity

**Unknown:** Which CPU static-graph runtime and exact channel count meet the
competition environment.

**Current default:** Quantized 8-bit, 64 trunk channels, 12 inverted residual
blocks, about 0.35 million parameters.

**Evidence:** Cold-load time, batch latency, p99 move latency, resident memory,
and playing strength from at least two capacity points on one dedicated CPU
core.

## Army normalization

**Unknown:** Whether the `4096` log scale clips important late-game stacks.

**Current default:** `log1p(x) / log1p(4096)`.

**Evidence:** Army and stack quantiles from generated competition trajectories,
split by turn and outcome. Select a scale above the measured high quantile and
confirm that quantization error stays small for ordinary stacks.

## Particle count

**Unknown:** The smallest particle count that preserves calibrated general,
ownership, and army beliefs.

**Current default:** 64 particles; resample below half effective sample size.

**Evidence:** Hidden-state log loss, general-cell coverage, collapse frequency,
recovery frequency, update latency, and arena strength for 32, 64, and 128
particles.

## Belief recovery window

**Unknown:** How many enemy turns fixed-lag rejuvenation must replay.

**Current default:** 8 turns.

**Evidence:** Recovery success and CPU cost after forced proposal mismatch on
recorded trajectories. The chosen window must restore exact observation
consistency without consuming the normal move budget.

## Search width and depth

**Unknown:** Final self and enemy widening caps, coefficients, exploration
floor, and depth.

**Current default:** self cap 16, enemy cap 12, depth 16, and the formulas in
[search.md](search.md).

**Evidence:** Tactical suites for chase, reinforcement, castle, mutual capture,
and deathtouch, plus completed simulations and arena contrast. An omitted
low-prior enemy response is a failure even when average throughput improves.

## Simulation target and deadline reserve

**Unknown:** Sustainable completed simulations and the smallest safe reserve.

**Current default:** target 32, minimum 8, 125 ms internal deadline, and 10 ms
batch-admission guard.

**Evidence:** p50 and p99 reply time under cold and warm process conditions,
belief recovery turns, maximum matrix width, and CPU contention. Zero faults
is required.

## Tree size

**Unknown:** The reuse benefit beyond 4,096 nodes.

**Current default:** 4,096 nodes with off-root least-recently-used eviction.

**Evidence:** Reuse hit rate, value change after reroot, memory, and simulations
saved per turn across full-length games.

## Curriculum promotion

**Unknown:** The confidence rule that moves sampling from tactical states to
earlier and full-start states.

**Current default:** Advance only when each active state class produces a
non-degenerate WDL target.

**Evidence:** WDL counts and confidence intervals by state class, followed by
full-start decisive rate and held-out arena strength. Training loss alone is
not evidence.

## Opponent mixture

**Unknown:** Final league-to-roster ratio and sampling weights.

**Current default:** 70% checkpoint league and 30% fixed bot panel.

**Evidence:** Exploitability against held-out checkpoints, opponent coverage,
cycling, decisive rate, and pairwise arena contrast. Do not optimize the mix
against leaderboard rank.

## Auxiliary loss weights

**Unknown:** Relative weights for hidden state, final margins, and termination.

**Current default:** No fixed values.

**Evidence:** Ablations that hold self-play games and search settings constant.
Keep a head only when it improves belief calibration or arena strength without
reducing policy learning.

## Training exploration

**Unknown:** Root-noise concentration, action temperature, and deterministic
turn.

**Current default:** No fixed values; rated play always disables them.

**Evidence:** Policy entropy, action coverage, self-play cycling, and held-out
strength. Exploration that causes protocol faults or persistent random play is
rejected.

## Opponent belief approximation

**Unknown:** Whether level-zero last-seen belief is sufficient for enemy-action
proposals.

**Current default:** No recursive opponent particles.

**Evidence:** Enemy-action negative log likelihood on held-out self-play and
particle survival after real observations. If the proposal misses critical
actions, compare one bounded additional belief level within the same deadline.

## ResBot castle evidence

**Unknown:** Whether production candidates at distance 7 represent a stable
build preference or a replay-detector effect.

**Current default:** Use the finding as a diagnostic only.

**Evidence:** Action reconstruction that proves the spend cell and turn on a
held-out ResBot sample. The result still cannot specify ResBot internals.
