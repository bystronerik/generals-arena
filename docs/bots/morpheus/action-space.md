# Action space

## Decision

The policy head emits 9 logits per cell on the canonical `21 × 21` board and
one global pass logit. The fixed output has `9 × 441 + 1 = 3970` logits.

| Channel | Action |
| ---: | --- |
| 0-3 | all-but-one move: up, down, left, right |
| 4-7 | half move: up, down, left, right |
| 8 | build a castle on this cell |
| global | pass |

The direction order is the competition protocol order. A selected logit maps
directly to `(pass, row, col, direction, split)`.

## Move mask

A move logit is legal only when all conditions are true:

- source is inside the actual `H × W` board;
- source is currently owned by Morpheus;
- source army is at least 2;
- destination is inside the actual board;
- destination type is not mountain `2` or hidden structure `5`.

Every move destination is adjacent to an owned source and is therefore visible
in the current observation. A capturable castle appears as type `3`, not type
`5`, before action selection. Morpheus masks every type `5` destination; the
persistent mountain/castle classification remains useful for belief and
planning beyond the current frontier.

Fog type `0` is passable by rule, although a valid source cannot have a
currently fogged adjacent destination under the competition visibility rule.

When source army is 2, half and all-but-one both send 1. Morpheus masks the half
logit and keeps the all-but-one logit. It keeps both split modes for every
larger source because they send different amounts.

## Build mask

A build logit is legal only when:

- the cell is inside the actual board;
- Morpheus currently owns it;
- its persistent type is plain;
- it is not either general or an existing castle;
- its army is at least the exact current build cost.

The cost includes Morpheus's general and every castle Morpheus owns. Captured
castles count. Enemy structures do not count.

A build emits `(2, row, col, 0, 0)`.

## Pass and normalization

Pass is always legal and emits `(1, 0, 0, 0, 0)`.

Illegal logits become negative infinity before softmax. The policy normalizes
only across legal logits. If pass is the only legal action, its probability is
1.

The opponent policy uses the same layout and a legal mask from the sampled
particle.

## Size independence

All action logits come from shared `1 × 1` spatial filters. The model does not
have a cell-specific fully connected policy layer. Board and padding masks
therefore preserve the same action meaning on every 18-21 rectangle.

Rotation and reflection augmentation remap direction channels and coordinates
together. It never changes the pass channel.

## Alternatives rejected

An autoregressive source-direction-split head is rejected because it requires
several dependent network calls inside each simulation. A flat fully connected
3970-way head is rejected because it gives each absolute cell separate weights
and weakens board-size transfer.

## Failure mode

The legal mask cannot prove that a fog move is tactically safe. It only proves
protocol legality from known terrain. Belief search, not the mask, evaluates a
hidden enemy army or castle on the destination.
