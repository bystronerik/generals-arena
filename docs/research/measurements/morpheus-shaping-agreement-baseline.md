# Morpheus shaping agreement baseline (Part 17 B)

How often the action Morpheus actually emits is the **network's own** top
choice, measured against the unshaped prior. This is the baseline the C1/C2
arms are read against.

Raw data:
[`morpheus-shaping-agreement-baseline.json`](morpheus-shaping-agreement-baseline.json).

## Method

3 recorded competition games per arm vs `cm_expander`, seeds 0/1/2, morpheus in
seat A, per-turn traces via `arena.matches.run_match --record`. Turns without a
completed root evaluation are excluded (agreement is undefined there). Phase
split is the probe's `enemy_visible`, the same selector the blend uses.

The shipped bound landed one commit before this measurement, so the "before"
column is a variant with `shaping_log_clip = 20`, `shaping_floor_frac = 0` —
effectively unbounded, since the observed heuristic log-ratios span about
`-12.3` to `+8.1`. The loader rejects an infinite clip, so this is how the
retired regime is reproduced for comparison.

## Result

| Arm | Phase | Turns | Chosen = NN top | Chosen in NN top-3 |
| --- | --- | ---: | ---: | ---: |
| unbounded (`log_clip = 20`) | pre-contact | 290 | 81.7% | 98.3% |
| unbounded (`log_clip = 20`) | post-contact | 948 | 70.8% | 90.2% |
| **shipped** (`log_clip = ln 10`) | pre-contact | 329 | **88.2%** | 99.1% |
| **shipped** (`log_clip = ln 10`) | post-contact | 701 | **75.9%** | 92.6% |

The bound moves agreement up by 6.5 points pre-contact and 5.1 post-contact, in
the intended direction.

## What this changes about the Part 17 premise

The plan's motivation was that "the heuristic picks the action class and the
network only ranks within the winning class." At the level this probe measures,
that was **overstated even before the bound**: the emitted action was already
the network's top choice on 71–82% of turns, and inside its top 3 on 90–98%.

The likely reason is that the exported policy is very peaked — mean mass on its
own top action is ~0.64 — so a heuristic needs a large multiplier to displace
it, and on most turns the heuristic and the network already agree. The unbounded
reshape mattered on the minority of turns where they disagreed, which is
precisely where a 6,000x override is most suspect and least defensible.

This does **not** invalidate the change (the disagreement turns are the ones
worth getting right, and the clip is what makes them the network's call), but it
does lower the prior on C1 showing a large net-value contrast. Read the C1 gate
on its own numbers.

## Caveats

Three games per arm against one opponent is a probe, not a verdict. It is
sufficient for a rate this stable (n = 290–948 turns per cell) and it is *not*
a strength claim — no winrate or rating contrast is quoted here. C1 and C2 still
need the fixed panel, `--seat-policy alternate`, and the pairwise contrast per
[decision-rule.md](../../arena/decision-rule.md).

## Reproducing

```bash
python scripts/morpheus_shaping_variant.py unbounded --log-clip 20.0 --floor-frac 0.0
python -m arena.matches.run_match bots/morpheus/run.sh bots/cm_expander/run.sh \
  --seed 0 --round p17-agreement-baseline --record
```
