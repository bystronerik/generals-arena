# Belief state under fog

## Decision

Morpheus maintains an explicit weighted particle belief. Search uses
root-sampled information-set MCTS: each simulation samples one particle, but
all particles update the same tree nodes for the same observable history.

`N_PARTICLES = 64` is an **initial guess**. The runtime benchmark can change
the count without changing the belief design.

## Particle contents

Each particle contains:

- the complete competition game state;
- one legal enemy-general location;
- hidden enemy ownership and army values;
- castle positions and hidden castle ownership;
- the simulated enemy action history;
- the enemy's last-seen memory of Morpheus.

Terrain is not sampled after the first frame. With no starting castles,
initial type `5` cells are mountains. Type `0` cells are passable base terrain,
but one of them can contain the hidden enemy general.

## Initial belief

Enemy-general candidates must be passable, outside current vision, consistent
with every observed type, and at least 17 BFS steps from Morpheus's general.
Particles sample candidates from the competition generator's conditioned prior.

The initial enemy state has one general cell with its legal starting army and
no other enemy land. Every particle must match the four public land and army
totals.

## Real-turn update

After Morpheus sends action `a` and receives the next observation:

1. Build the enemy-perspective observation for every particle.
2. Sample enemy action `b` from the shared policy network.
3. Apply `(a,b)` with build-first resolution, move priority, combat, and growth.
4. Emit Morpheus's simulated observation.
5. Give zero weight to any particle that differs on a visible cell, type,
   owner, army, turn, or public total.
6. Normalize the surviving weights.
7. Resample when effective sample size falls below half the particle count.

The half-count threshold is an **initial guess**.

The proposal uses the same policy from the enemy perspective. Its belief
planes come from the enemy's simulated last-seen memory and a level-zero
state estimate. Morpheus does not build recursive beliefs about beliefs.

Morpheus deduplicates identical enemy information tensors by hash. It evaluates
all remaining particle proposals in one batch of up to 64, then samples one
action per particle from the matching policy. It does not run 64 serial
forwards.

## Rejuvenation and recovery

Resampling alone copies particles and loses alternatives. A rejuvenation move
replays the last 8 enemy turns from a stored ancestor with different legal
enemy actions. It accepts only histories that reproduce every observation and
public total. The 8-turn window is an **initial guess**.

Rejuvenation uses policy-guided bounded beam sampling. At the first replayed
turn it branches over pass, the top 4 legal policy actions, and any
vision-changing action. Later turns sample from the policy. It prunes a branch
as soon as a recorded observation or public total differs.

Beam width 8, at most 16 completed proposal histories, and at most 128 replayed
transitions per real turn are **initial guesses**. Admission control can stop
rejuvenation before these limits without changing the current valid belief.

If all particles fail:

1. Rewind one turn.
2. Enumerate pass, the policy's highest-prior enemy actions, and all actions
   that can change Morpheus's vision.
3. Keep exact observation matches.
4. If none match, reconstruct a maximum-entropy hidden allocation that matches
   visible state and public totals.

The reconstructed set receives minimum confidence in `belief_ess`. Search then
uses wider opponent exploration until normal filtering restores diversity.

## Search representation

A tree node is keyed by Morpheus's action-observation history, persistent
memory, and turn. It is not keyed by a hidden particle.

Each node keeps a bounded particle reservoir. A simulated joint action branches
on Morpheus's resulting observation hash. Different enemy actions that produce
the same Morpheus information state merge into the same child.

## Alternatives rejected

### One determinization

A single sampled world is rejected. It leaks one guessed general and army map
into every branch, which causes strategy fusion and brittle attacks. One wrong
sample can consume the whole move budget.

### Learned latent state

An unconstrained recurrent latent is rejected as the search state. A latent can
assign army to mountains, violate public totals, move a general, or forget a
castle without a rule-consistency check.

### Full nested beliefs

Recursive particles for what the enemy believes about Morpheus are rejected.
They exceed the CPU budget and do not terminate at a clear belief level.

## Known failure mode

The explicit belief can collapse when the opponent-policy proposal gives too
little probability to the action that occurred. Recovery preserves rule
consistency, but a reconstructed maximum-entropy state loses long-range enemy
history. `belief_ess` exposes this loss to the network and search.

## Part 05 executable definitions

These definitions close the specification gaps that Part 05 requires before
implementation. They are deliberate choices, not competition-module APIs.

### Conditioned initial-general prior

There is no repository API for the competition generator's conditioned prior.
Part 05 therefore samples as follows:

1. Build the terrain from the first observation: type `5` cells are mountains;
   type `0` cells and every visible non-mountain cell are passable base.
2. Legal enemy-general candidates are passable, outside Morpheus's current
   vision, consistent with every observed type, and at least 17 BFS steps from
   Morpheus's general over passable cells.
3. Sample candidates **uniformly**. Each accepted candidate has weight
   `1 / |candidates|`.
4. The initial enemy state has one general cell with starting army `1` and no
   other enemy land. Public land and army totals must match the observation.

Replace this uniform prior only when a generator-conditioned API exists and a
measurement shows a material survival or calibration gain.

### Vision-changing action

An enemy action `b` is **vision-changing** for Morpheus seat `p` in state `s`
given Morpheus action `a` when the visibility mask of seat `p` after
`transition(s, (a, b))` differs from the mask after
`transition(s, (a, PASS))`.

Visibility is the competition rule: every cell in the Chebyshev-1 neighborhood
of a cell owned by `p`.

### Maximum-entropy hidden allocation

When recovery must reconstruct particles from an observation and public totals:

1. Copy every visible cell's type, owner, and army from the observation into the
   reconstructed state. Keep known mountains and castles from memory.
2. Legal hidden cells are fogged, passable, and not owned by Morpheus in the
   observation.
3. Place the enemy general uniformly among legal hidden cells that keep the
   public totals feasible.
4. Choose the remaining enemy land cells uniformly among the other legal hidden
   cells so enemy land equals `opp_land`.
5. Split the remaining enemy army as evenly as possible across those enemy cells
   (each cell gets `floor(A/L)` or `ceil(A/L)`). That integer split maximizes
   entropy among fixed-land allocations with a fixed army sum.
6. Own cells and armies already fixed by the observation must leave
   `my_land` and `my_army` exact.

The reconstructed set sets `belief_ess` to the minimum confidence
`1 / N_PARTICLES` (the configured particle count).

### Importance and acceptance weights

- **Normal filter.** The shared policy is both the target prior and the
  proposal. Sample one enemy action per particle from that policy. The
  observation likelihood is exact: weight becomes `0` on any visible-cell,
  type, owner, army, turn, or public-total mismatch; otherwise the weight is
  unchanged (importance ratio `1`). Normalize survivors.
- **Rejuvenation and recovery proposals.** A completed enemy-action history
  `h = (b_1, …, b_L)` that reproduces every stored observation and public total
  receives acceptance weight
  `W(h) = ∏_t π_θ(b_t | enemy_info_t)`, where `π_θ` is the shared policy under
  the injectable proposal (uniform over legal actions when no network is
  injected). Histories that fail a check have weight `0`. Normalize `W(h)` over
  accepted histories before drawing replacement particles.
