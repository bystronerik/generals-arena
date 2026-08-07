# Neural network

## Decision

Morpheus uses one fully convolutional residual network for both seats. The same
weights produce policy and value from a perspective-relative tensor.

The initial model budget was about **0.35 million parameters**, an
**initial guess**; the deployed artifact's manifest records **249,316**
parameters. Deployment uses **float32 TorchScript** weights on CPU. No
GPU is required at match time. Static int8 was the original design and was
abandoned after measurement: at this size the quantized graph is both slower
(GroupNorm has no int8 kernel, so every block pays dequant → GroupNorm → quant)
and lossy unless calibrated on real observation tensors. Measured on the
judge-like host at 0.25M and 2M parameters, float32 wins every batch shape; see
[`morpheus-float32-deployment.md`](../../research/measurements/morpheus-float32-deployment.md).

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
At match time no auxiliary head is evaluated: belief recovery reconstructs
particles with rule-based maximum-entropy allocation (optionally guided by the
policy head), not the hidden-state heads. The margin and termination heads are
not used by search. Online search loads dedicated policy and policy+WDL entry
points so auxiliary heads are not evaluated on the belief or leaf path.

## Inference budget

The parameter count gives roughly 1 MB of raw float32 weights per exported
graph (~3.2 MB across the three TorchScript files) and leaves the 2 GB limit
dominated by the runtime, tree, and particles.

When the policy proposal is enabled, belief propagation uses one enemy-proposal
batch of up to `max_proposal_batch` unique information tensors (design 64;
deployed 8). The deployed configuration uses the uniform proposal and runs no
belief-path forwards
([`belief-ablation-macaria.md`](../../research/measurements/belief-ablation-macaria.md)).
Search uses batches of up to 4 pending leaf evaluations. Root evaluation and
new enemy-table priors run as their own forwards (`enemy_prior_batch` is a
separate admission component). The batch sizes are **initial guesses**.

Online match-time inference uses dedicated export entry points so auxiliary
heads are not evaluated during search:

- policy + pass for belief proposal and enemy priors;
- policy + pass + WDL for root and leaf evaluation.

The full-head graph remains available for recovery and diagnostics.

The deployment gate measures both batch shapes, total forward-equivalents,
tensor construction, belief transition, and search on one CPU core. A parameter
count alone does not prove the 150 ms deadline.

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
