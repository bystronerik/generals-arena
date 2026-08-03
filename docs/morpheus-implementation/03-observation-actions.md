# Part 03: Observation and actions

## Deliverable

Build persistent visible memory, the `49 × 21 × 21` network tensor, the
3970-logit action codec, legal masks, and rotation and reflection transforms.

The same code must work from either seat and on every competition rectangle.

**Touches**

- `bots/`: add memory, tensor, action, and symmetry modules plus tests.
- `arena/`: none.
- `scripts/`: add the army-scale measurement command.
- `data/bot_versions/`: none.

## Prerequisites

- [Part 01: Protocol shell](01-protocol-shell.md)
- [Part 02: Transition kernel](02-transition-kernel.md)
- [Part 00c: Measurement corpus](00c-measurement-corpus.md)

## Source specifications

- [Observation tensor](../bots/morpheus/observation-tensor.md)
- [Action space](../bots/morpheus/action-space.md)
- [Competition protocol](../competition/protocol.md)

## Defaults and replacement measurement

This part assumes the current army transform:

```text
clip(log1p(max(x, 0)) / log1p(4096), 0, 1)
```

Replace `4096` only after measuring army quantiles by turn and outcome and
quantization error for ordinary and extreme stacks. Record the selected scale
in the model manifest so a scale change moves the artifact identity.

## Implementation boundary

Update visible memory before building the tensor. Keep padding zero and use
`board_mask` to distinguish padding from mountains.

Build one perspective-relative tensor function for both seats. The action
codec must map all spatial channels and global pass to the exact wire tuple.
The legal mask must use the current observation and exact live build cost.
Pass must remain legal.

Symmetry transforms must remap coordinates, direction channels, previous
actions, generals, memory, belief planes, and policy targets together.

## Isolated test

```bash
python -m pytest \
  bots/morpheus/tests/test_observation_tensor.py \
  bots/morpheus/tests/test_action_space.py \
  bots/morpheus/tests/test_symmetry.py \
  -m morpheus -q
```

```bash
python scripts/morpheus_measure.py army-normalization \
  --trajectories data/trajectories/morpheus-bootstrap \
  --output docs/research/measurements/morpheus-army-normalization.json
```

```bash
python competition-module/competition/matchup.py \
  bots/morpheus/run.sh bots/smoke/run.sh \
  --mode competition --seed 0
```

## Specification gaps

Resolved in this part and recorded in
[observation-tensor.md](../bots/morpheus/observation-tensor.md):

- `belief_enemy_army_mean` averages raw particle armies, then applies
  `army_value`.
- `belief_enemy_army_std` applies `army_value` to the raw particle std.
- Previous-action planes encode the perspective player's last action only.

The 16 auxiliary army-bin edges belong to Part 04 and are not defined by the
current specs.

## Exit criterion

Answer `yes` if golden tensors match all 49 plane contracts, every unmasked
action is legal in the deployment transition, every legal action is reachable,
all symmetries round-trip, and the competition gate finishes.

Answer `no` for any plane, padding, direction, build-cost, or mask mismatch.
