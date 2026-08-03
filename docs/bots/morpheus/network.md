# Neural network

## Decision

Morpheus uses one fully convolutional residual network for both seats. The same
weights produce policy and value from a perspective-relative tensor.

The initial model budget is about **0.35 million parameters**, an
**initial guess**. Deployment uses quantized 8-bit weights in a static CPU
graph. No GPU is required at match time.

## Trunk

The trunk receives the
[`49 × 21 × 21` tensor](observation-tensor.md) and keeps spatial resolution:

1. A `3 × 3` stem produces 64 channels.
2. Twelve inverted residual blocks use `64 → 128 → 64` channels.
3. Each block has a pointwise expansion, a depthwise `3 × 3` convolution, and
   a pointwise projection.
4. Depthwise dilation repeats `1, 2, 4`.
5. Group normalization and `ReLU6` follow trainable convolutions.
6. Residual connections preserve the 64-channel board representation.

The dilation cycle covers the full 21-cell extent without pooling. Pooling is
rejected because one-cell source and destination precision must survive.

## Policy head

A `1 × 1` projection emits the 9 spatial channels defined in
[action-space.md](action-space.md). A masked global mean and maximum of the
trunk feed one pass logit.

The policy head is shared for the enemy-action proposal. Perspective-relative
inputs remove any seat-specific output.

## Value head

Masked global mean and maximum features feed three logits:

- perspective-player win;
- draw;
- perspective-player loss.

Softmax gives `p_win`, `p_draw`, and `p_loss`. Search bootstraps with:

```text
V = p_win - p_loss
```

The tensor's perspective player defines the sign. A root-player evaluation
keeps `V`; an enemy-perspective evaluation negates `V` before root backup.

## Auxiliary heads

Training uses spatial heads for:

- hidden enemy ownership probability;
- enemy army in 16 logarithmic bins;
- enemy-general cell probability;
- hidden castle ownership probability.

Training also uses scalar regression heads for final land margin, final army
margin, final castle margin, and turns to termination.

These heads use engine truth during training. They do not change game reward.
At match time, the hidden-state heads are used only to propose particles after
belief collapse. The margin and termination heads are not used by search.

## Inference budget

The parameter count gives less than 0.5 MB of raw 8-bit weights and leaves the
2 GB limit dominated by the runtime, tree, and particles. The initial target is
one batch of 4 leaf evaluations at a time. Batch size 4 is an **initial guess**.

The deployment gate measures full inference, tensor construction, and search
on one CPU core. A parameter count alone does not prove the 150 ms deadline.

## Training behavior

The policy target is the root average strategy from simultaneous search. The
value target is terminal WDL. Auxiliary targets use the sampled engine state.

Board rotations and reflections augment training. They remap action direction,
coordinates, generals, memory, and belief planes together.

## Alternatives rejected

A transformer is rejected for the first design because repeated all-cell
attention is too costly across many leaf evaluations. A large AlphaZero-style
standard-convolution tower is rejected for the same deadline reason.

A policy-only network is rejected because it cannot compare hidden-state
outcomes or correct a prior with exact competition transitions.

## Failure mode

The small trunk can underfit long tactical interactions. Dilation gives global
reach, but it does not guarantee useful global reasoning. Search and the
capacity benchmark must detect this limit rather than silently increase model
size past the deadline.
