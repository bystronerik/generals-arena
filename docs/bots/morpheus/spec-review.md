# Spec review — 2026-08-03

A design review of the 12 Morpheus spec files against [RULES.md](../../../RULES.md),
[the competition protocol](../../competition/protocol.md), and
[the unified bot API](../../engine/unified-bot-api.md). Factual claims were
verified directly against `competition-module` source. No spec file was edited.

**Overall verdict:** a strong, unusually careful spec. The wire-protocol
details, transition ordering, deathtouch/draw semantics, and the two claims
most likely to be wrong — the growth-parity formulas and the "fogged general
renders as type 0" assumption — all check out against the engine source. The
search math is internally correct, and the ResBot/arena separation complies
with `AGENTS.md`. The one blocking finding is arithmetic, not conceptual.

## Blocking

### 1. The per-turn network-evaluation budget is never totaled

[belief-state.md](belief-state.md) (real-turn update), [runtime.md](runtime.md)
(simulation target, admission control), [network.md](network.md) (batch size),
[search.md](search.md) (enemy tables).

The real-turn belief update samples an enemy action **from the policy network
for every particle** — 64 enemy-perspective forwards per turn at the default
count. Search adds 1 root eval + 32 leaf evals, plus one enemy-perspective
eval for each new enemy information hash that needs a prior `P_B,h`. That is
roughly 100–130 forwards per turn. A ~0.35M-param net at 21×21 spatial
resolution is on the order of 100M MACs per forward, so ~10–13 GMACs must
complete in 125 ms — sustained ~100 GMACs/s on one core, at or beyond
realistic int8 single-core throughput, leaving nothing for ~100 tensor
constructions, particle transitions, hashing, and tree operations.

Each number is marked "initial guess" separately, but the open questions for
particle count and simulation target never note they compete for the same
deadline, and admission control tracks "tensor, inference, simulation, and
reply" cost with no line item for the belief update — the single largest
consumer.

**Fix:** add a per-turn evaluation-budget table (belief-update forwards +
root + leaves + enemy priors ≤ N) to `runtime.md`; add belief update to the
admission-control estimates; couple the particle-count and simulation-target
open questions; and name the obvious amortization — one large-batch forward
for all 64 particle proposals rather than batch-4, which changes the
arithmetic materially.

## Should-fix

### 2. Objective statement contradicts the training reward

[index.md](index.md) says Morpheus "maximizes the probability of capturing the
enemy general"; the reward in [training.md](training.md) (+1/0/−1,
undiscounted) and the value bootstrap `p_win − p_loss` optimize win *minus
loss* probability. These diverge exactly where it matters: a position offering
40% win / 60% loss beats a certain draw under "maximize capture probability"
but loses to it under the actual reward. **Fix:** reword the index objective
to "maximizes expected game outcome (win − loss)" — or, if must-win
risk-seeking is intended, change the reward, which is a bigger decision.

### 3. Curriculum bootstrap source is a silent gap

[training.md](training.md): curriculum items are "a seed plus an action
prefix" from legal `GeneralsEnv(mode="competition")` trajectories, and class 1
is "one tactical sequence from a general capture" — but the spec never says
who plays the games that produce capture-adjacent states before Morpheus can
produce them itself (the predecessor's record is 31/31 draws). Presumably
heuristic-bot games from the fixed panel, but that is an inference, and
[open-questions.md](open-questions.md) does not claim it as deferred.
**Fix:** state the trajectory sources per curriculum class.

### 4. Rejuvenation mechanism underspecified beyond its window length

[belief-state.md](belief-state.md): "replays the last 8 enemy turns … with
different legal enemy actions" describes a search over an astronomically large
space. Open-questions defers only the *window size*; how alternative histories
are proposed (policy-guided sampling? how many candidates? what CPU bound?) is
left to the implementer. **Fix:** specify policy-guided sampling with a
bounded candidate count, or add the proposal mechanism to open-questions.

### 5. Enemy-prior acquisition and enemy-table growth unstated

[search.md](search.md): each node keeps "one enemy table per enemy information
hash" with prior `P_B,h`. Nowhere does the spec say these priors come from an
enemy-perspective network forward (they must), nor bound the number of
distinct enemy hashes per node — up to the particle count per node, each
carrying candidates, regrets, priors, and joint `N/W/Q[a,h,b]` arrays.
**Fix:** state the prior's source and add a per-node cap on enemy tables
(reservoir-deduped, LRU-evicted); both compute (finding 1) and the 4,096-node
memory bound depend on it.

### 6. Training compute entirely unquantified

Self-play requires **both seats** to run the full particle-belief +
matrix-search stack; at the spec's own defaults that is order-10⁵ network
forwards per full 1200-turn game, and AlphaZero-style iteration needs many
thousands of games per checkpoint. No open question covers training scale,
hardware, or whether training-time search may use reduced particles/sims.
`index.md`'s "no implementation plan or schedule" scope does not cover this —
whether training is feasible at all on available hardware is a strategic
constraint. **Fix:** add an open question with the evidence needed (measured
self-play throughput, games-per-checkpoint target) and state whether
training-time settings may differ from match-time settings.

### 7. Spec location bypasses the skills workflow

The spec lives in `docs/bots/morpheus/`, but the strategist role in
`AGENTS.md` is done when the "Spec [is] in `docs/research/strategies/` with
diversity verdict," and `build-bot-from-spec` scaffolds from a spec under
`docs/research/strategies/` — every other bot has an entry there. An
implementer following the workflow will not find Morpheus. The broader
`docs/**` placement table is satisfied, so this is process friction, not a
violation. **Fix:** add a short pointer file
`docs/research/strategies/morpheus.md` linking here and restating the
diversity verdict.

### 8. The required submission-shaped check has no home

[evaluation.md](evaluation.md) correctly observes that the local gate cannot
enforce the 150 ms timer, 2 GB cap, or EOF exit, and requires a second
resource-enforcing check — but no such harness exists in the repo and the
spec does not say where it should live or who builds it. **Fix:** name it as
a new deliverable (e.g. under `arena/` or `scripts/`) or add it to
open-questions so it is not silently assumed to exist.

## Nits

- **`sight_age` undefined for never-seen cells.**
  [observation-tensor.md](observation-tensor.md) plane 16 is "turns since
  direct sight / 1200," but a cell with `ever_visible = 0` has no last sight.
  Define it (e.g. 1.0, or 0 with the `ever_visible` gate making it ignorable).
- **"Virtual simulations" batching named but never defined.**
  [runtime.md](runtime.md) runs "batches of up to 4 virtual simulations,"
  implying in-flight simulations coexist before backup, but
  [search.md](search.md) never says how an in-flight path is represented
  (virtual loss does not translate directly to regret-matching statistics;
  stochastic joint sampling may make it unnecessary — say which).
- **Degradation-table wording.** [runtime.md](runtime.md)'s "8–15 → root
  average strategy with no further widening" mixes an in-search policy (stop
  widening) into a table of *final decision* rules; rephrase as a search-time
  state rather than a decision row.

## Verified sound

- **Wire protocol and action encoding** — direction order, split codes,
  pass/build tuples, 4 public totals, EOF handling all match
  `competition-module/competition/protocol.py` exactly; 9×441+1 = 3970 checks
  out.
- **Growth-parity formulas** — verified against
  `competition-module/generals/core/game.py`: time increments *before*
  `global_update` (game.py:339), structures grow at `time % 2 == 0`
  (game.py:279), bulk at `time % 50 == 0` (game.py:266), so
  `(turn+1) mod 2 == 0` and the countdown formula in
  [observation-tensor.md](observation-tensor.md) are exactly right.
- **Fogged-general assumption** — `Observation.structures_in_fog` covers
  castles/mountains only (observation.py:29), so a fogged general renders as
  type 0 and initial type-5 cells are all mountains, as
  [belief-state.md](belief-state.md) and tensor plane 3 assume.
- **Transition ordering, deathtouch, draws, build cost/mask, move-legality
  mask** — all consistent with [RULES.md](../../../RULES.md), including the
  older simultaneous-capture-is-a-draw rule.
- **Search math** — the RM+ regret updates and root-perspective sign
  conventions are correct for the zero-sum matrix formulation;
  [network.md](network.md)'s perspective-negation rule agrees with
  [search.md](search.md)'s no-sign-alternation backup.
- **Network arithmetic** — the described architecture sums to ~0.3M params as
  claimed, and the dilation cycle's receptive field (~29-cell radius) covers
  21.
- **Process compliance** — ResBot replays kept strictly observational per
  `AGENTS.md`; evaluation flows through the content-hash registry,
  `data/games/`, and the pairwise decision rule; the 90%-identical-turns
  distance test matches
  [diversity-constraints.md](../../research/strategies/diversity-constraints.md);
  all cross-file links resolve.
