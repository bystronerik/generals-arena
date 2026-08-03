# Search

## Decision

Morpheus uses root-sampled information-set MCTS with a simultaneous zero-sum
matrix at each node. Regret matching plus selects both actions.

The transition is the exact competition transition. It applies builds first,
then chasing, reinforcing, and smaller-source priority, then combat and growth.
From turn 800, an executing touch wins regardless of army unless a chase first
removes its source. Mutual general capture or mutual touch is a draw.

## Node state

Each information-set node stores:

- observable-history hash and turn;
- bounded particle reservoir;
- node visits `N`;
- legal candidates `A`, prior `P_A`, regret `R_A`, and average strategy `S_A`;
- one enemy table per enemy information hash `h`, with legal candidates
  `B_h`, prior `P_B,h`, regret `R_B,h`, and average strategy `S_B,h`;
- joint visits `N[a,h,b]`;
- joint value sum `W[a,h,b]` and mean `Q[a,h,b]`;
- children keyed by Morpheus action and resulting observation hash.

All values use the root player's perspective. Backup does not alternate signs.

## Candidate actions

Actions enter each matrix in policy-prior order. Pass, legal general captures,
and legal actions that directly interact with a visible enemy source are always
eligible for widening.

The enemy information hash contains its simulated observation and persistent
memory. After a simulation samples a particle, it uses only the enemy actions
legal for that hash. An unavailable action receives no prior, visit, value, or
regret update for that particle.

`P_B,h` comes from the shared network on the enemy-perspective tensor. Each node
keeps at most 8 enemy tables, an **initial guess**. Hash deduplication reuses a
table. Weighted least-recently-used eviction removes an inactive table and its
joint statistics; the table used by a pending simulation cannot be evicted.

Progressive widening uses:

```text
K_self(N) = min(16, 1 + floor(2.0 * sqrt(N)))
K_enemy(N) = min(12, 1 + floor(1.5 * sqrt(N)))
```

The caps and coefficients are **initial guesses**. A newly added action keeps
its normalized network prior.

## Matrix selection

Unvisited joint entries use the node's network value as first-play urgency.
Visited entries use `Q[a,h,b]` for the sampled enemy information hash.

For each side, regret matching normalizes positive regrets. If all regrets are
zero, it uses the network prior. Search mixes the result with the prior:

```text
epsilon(N) = max(0.05, 0.5 / sqrt(1 + N))
sigma = (1 - epsilon) * regret_strategy + epsilon * prior
```

`0.05` and `0.5` are **initial guesses**. Search samples one `a` from the root
information set and one `b` from the sampled particle's enemy-information
table, then evaluates the joint action.

## Simulation

One simulation:

1. Samples one particle from the node belief.
2. Builds the enemy information hash and legal masks for both perspectives.
3. Samples the joint action from the current matrix strategies.
4. Applies the exact competition transition.
5. Ends with `+1`, `0`, or `-1` on win, draw, or loss.
6. Otherwise follows the child for Morpheus's resulting information state.
7. Expands one new child or stops at depth 16.
8. Bootstraps a nonterminal leaf with `p_win - p_loss`.

Depth 16 is an **initial guess**. Search uses no random rollout. A rollout to
turn 1200 is too slow and gives high-variance values under fog.

## Pending leaf batch

Search selects up to 4 paths from one frozen statistics snapshot without
virtual loss or temporary regret. Duplicate leaves are allowed. One batch
evaluates them; backup follows selection order and reserves no prior statistics.

## Matrix backup

The completed simulation updates `N[a,h,b]`, `W[a,h,b]`, and `Q[a,h,b]` on
its path. Morpheus action utility averages over the particle-weighted enemy
information hashes. Enemy regret updates only the sampled hash. For readability,
the formulas below suppress `h`:

```text
u_self(a) = sum_b sigma_enemy(b) * Q[a,b]
u_enemy(b) = sum_a sigma_self(a) * Q[a,b]
v = sum_a sigma_self(a) * u_self(a)
```

Unvisited `Q` entries use first-play urgency in these sums.

Regret matching plus updates:

```text
R_A[a] = max(0, R_A[a] + u_self(a) - v)
R_B[b] = max(0, R_B[b] + v - u_enemy(b))
```

The node also adds each current strategy to `S_A` and the sampled `S_B,h`.
This matrix backup learns mixed responses to chase, reinforcement, split,
build, and pass interactions.

## Root action

During rated play, Morpheus selects the legal action with the largest
normalized `S_A`. Ties use marginal visit count and then policy prior. Training
can sample from `S_A` with a configured temperature.

Only fully backed-up simulations affect root choice. A partial simulation has
no statistics.

## Tree reuse

After the real next observation, Morpheus follows the child keyed by its sent
action and the new observation hash. Enemy actions that led to the same
information state are already merged there.

Morpheus replaces the child's particle reservoir with the filtered real belief.
It keeps statistics only when turn, memory, and observation hashes match.
Otherwise it starts a new root.

## Alternatives rejected

Alternating PUCT and fixed sequential approximations are rejected because they
give the second player false action information. Decoupled UCT is rejected
because marginal values miss chasing and contested-cell interactions.

## Failure mode

Low visit counts make regrets noisy and can omit a low-prior enemy tactic.
Benchmark the opponent cap and exploration floor on tactical counterexamples.
