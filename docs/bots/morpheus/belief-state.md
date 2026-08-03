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

## Rejuvenation and recovery

Resampling alone copies particles and loses alternatives. A rejuvenation move
replays the last 8 enemy turns from a stored ancestor with different legal
enemy actions. It accepts only histories that reproduce every observation and
public total. The 8-turn window is an **initial guess**.

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
