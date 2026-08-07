# Prior shaping

How tactical heuristics reach the root prior, and how far they are allowed to
move it. Root evaluations only — leaf priors and enemy priors are never shaped.

## The blend

Per legal action, over the play mask:

```text
h_norm = h / gmean(h over legal, positive)     # center BEFORE clipping
p      = max(nn_prior, floor_frac * max(nn_prior))
shaped = softmax( log p + lambda * clip(log h_norm, +-log_clip) )
```

`h` comes from `heuristic_action_scores`; the blend is `blend_prior`. Both live
in `bots/morpheus/tactics.py`, and `apply_pre_contact_prior` is the thin wrapper
that runs the pair.

### Knobs

| Knob | Default | Meaning |
| --- | --- | --- |
| `shaping_lambda_pre_contact` | `1.0` | Trust in the heuristic before an enemy cell is visible. `0` is the pure network prior. |
| `shaping_lambda_post_contact` | `1.0` | Same, after contact. Defaults equal; the split exists so a sweep can move them apart. |
| `shaping_log_clip` | `ln 10` | Bound on one heuristic's reach: at most a 10x nudge either way. Must be finite and positive. |
| `shaping_floor_frac` | `1e-3` | Prior floor as a fraction of the network's own top, so a network zero cannot be resurrected past the clip. |

All four are `deployment.json` fields, so every setting is a distinct
content-hash rated entity. Phase selection uses `enemy_is_visible`.

Two scoring behaviors sit outside these knobs, as `tactics.py` constants:

- **Commitment hysteresis** (`CONTINUATION_BONUS`, 1.5x): moves whose source
  is the previous move's destination are multiplied when they already score
  positive, so the shaped argmax keeps marching the stack it moved last turn
  instead of re-tie-breaking from scratch. It never revives a zero score, so
  bans and crushed retreats still win.
- **Wave-forming gather** (`stack_gather_factor`, `GATHER_SHARE_MIN`,
  `WAVE_ARMY_SOFT_CAP`): forward merges into a large stack are near-free,
  lateral merges are dampened rather than crushed, gathering into the king
  stack stays rewarded until it holds half the total army, and the gather
  score is weighted by `attack_weight` so heavy stacks move before tips.
  Motivated by measurement: waves used to reach enemy land with a median 6
  army (0.9% of total; 8% at first contact).

### Why each piece

- **Centering is load-bearing.** Raw scores sit around ~120. Without dividing by
  the geometric mean every action would saturate the clip and the heuristic term
  would collapse to a constant — the clip would silently disable shaping rather
  than bound it.
- **The clip is what makes this bounded.** The heuristic spans roughly nine
  orders of magnitude (measured on the contact fixture: `4.7e-6` to `3313` times
  the geometric mean) while the network prior spans two or three. Unbounded, the
  heuristic picks the action class and the network only ranks inside it.
- **The floor is relative, not absolute.** An absolute `1e-6` floor let any
  action the network had zeroed be revived by a large enough heuristic. A
  fraction of the network's own top keeps the revival inside the clip bound.

### What the bound costs

Compression is real and intended. On the contact fixture, 10 of 24 legal actions
sit above `10x` the geometric mean, so at `log_clip = ln 10` they all tie at the
ceiling and the network breaks the tie. An enemy take no longer strictly
outranks a fog carve on a flat prior. The heuristic *ranking* is unchanged — only
its reach is capped — and the actions that must be considered are guaranteed by
candidate inclusion, not by score (below).

## Hard rules (shaping cannot touch these)

Shaping is a ranking nudge. These are rules, and no value of `lambda` — `0`
included — disables any of them:

- `play_mask` — pass is illegal whenever another action exists; before contact,
  moves that stack onto the own general or castle are illegal.
- `mandatory_action_indices` — general captures, visible-enemy-source
  interactions, enemy attacks, and frontier expands are always in the root
  candidate set. Steering is by *inclusion*, not by score. Pinned by a test that
  runs every `lambda`.
- `constrain_nn_action` — forces an available general capture, forbids passing
  when a move exists, and blocks own-land oscillation.

General captures are deliberately *not* scored by `heuristic_action_scores`.
They are already guaranteed by inclusion and forced by `constrain_nn_action`, so
a score term would only duplicate a rule that cannot be outvoted. Before Part 17
the heuristic carried a `10^6` capture term for this; it was removed as dead
weight.

## Probe: who is deciding?

Agreement between the network's own preference and what the bot played is
recorded per turn, measured against the **unshaped** prior
(`NetworkEvaluator.last_unshaped_prior`):

| Key | Meaning |
| --- | --- |
| `nn_top_action` | argmax of the unshaped network prior (`-1` when no root ran) |
| `nn_top_prior_milli` | that action's unshaped mass x 1000 |
| `chosen_matches_nn_top` | the emitted action is the network's top choice |
| `chosen_in_nn_top3` | the emitted action is in the network's top 3 |
| `enemy_visible` | which shaping phase the turn used |

`training/morpheus/measure_online.py` aggregates these into
`nn_agreement.{pre_contact,post_contact}`. A near-zero agreement rate means the
heuristic is still choosing the action class.

## Measurement arms

Every arm is its own bot directory with its own `deployment.json`, so the
setting is part of the content hash. Never an environment variable inside a
rated bot.

```bash
python scripts/morpheus_shaping_variant.py uniform --evaluator uniform
```

`scripts/morpheus_shaping_variant.py` copies the closure, symlinks the 3.2 MB
export artifact (so every arm hashes identical model bytes), and applies the
overrides. `bots/morpheus-*/` is gitignored: an arm exists for one comparison
and is deleted afterwards, per the
[experiment protocol](../../research/experiment-protocol.md).

`"evaluator": "uniform"` swaps the network for a flat prior and a zero value
while keeping the same mask, the same blend, and the same hard rules — so the
contrast isolates what the network contributes. The uniform arm runs no forward
pass and therefore completes more simulations per turn; read completed
simulations alongside the win contrast rather than treating it as free.

C1 (net-value ablation) gates C2 (the `lambda` sweep): if the network
contributes nothing measurable, tuning how much to trust it is wasted arena
time. Both need a fixed panel, `--seat-policy alternate`, and a verdict read
from the pairwise contrast per [decision-rule.md](../../arena/decision-rule.md).

## Related

- [Design index](index.md), [network](network.md), [search](search.md),
  [runtime](runtime.md)
- [Part 17 plan](../../morpheus-implementation/17-prior-shaping-blend.md)
- [Decision rule](../../arena/decision-rule.md) for reading any contrast above
