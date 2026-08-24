# Joe model-size ceiling — x86 CPU latency ladder (Modal, 2026-08-24)

How big can a joe `HistoryTransformer` get before p99 ms/move crosses the
150 ms move budget on one x86 core, in the float32 jax-CPU deployment stack?
Extends [Phase 2](joe-phase2-cpu-latency.md) upward with a six-tier ladder
whose size ratios come from the released AverageJoe configs
(`average-joe/configs/{S,M,L}.yaml`): M → L is depth ×2.2, embed ×7/6, and
XL24 applies that multiplier to L a second time.

## Verdict

**Every tier on the ladder fits — including XL24 (depth 24, embed 528,
68.5M params, 9.1× M compute) at p99 116.7 ms on the measured host.** The
latency crossing extrapolates to ~13–14× M compute (~100M params), so the
150 ms budget is not what limits the ladder itself. What limits the choice
is host variance: per-core speed on Modal spans ~2× across hosts, and this
run's host is already ~1.7× slower than the Phase 2 host. With a 1.5×
host-variance margin (p99 ≤ 100 ms on this host), the safe ceiling is the
**L16–XL20 band: depth 16–20, embed 480–512, 38–54M params — roughly 3×
the compute of the currently trained M7F4.** Pure AverageJoe L (23.4M
params) is deep inside budget at p99 52 ms.

## Provenance

- Script: `scripts/joe_modal_size_ceiling.py`; nets from
  `training/joe/networks/`, random weights (latency does not depend on the
  weights). Raw records: [joe-size-ceiling-cpu.json](joe-size-ceiling-cpu.json).
- Same timed step as Phase 2: build-cost grid, 39-channel augmentation,
  move + build masks, forward, greedy argmax, decode, host transfer.
  3 games × 1,200 moves per tier.
- Stack: Modal `debian_slim` py3.12 x86_64, `jax==0.11.0` CPU, float32,
  every thread pool pinned to 1.
- **All six tiers ran interleaved inside one container** (game-level round
  robin), because identical code spans ~2× per-core speed across Modal
  hosts — only same-host contrasts count. M anchors the run to Phase 2.
- Two containers: strict (cpu request = limit = 1.0, 2 GB) mirroring the
  competition limits, and a burst-4 control (4 GB) giving true compute.

## Ladder and results (burst = true compute)

Cost model: depth × embed² × (4 + 2·ff), normalized to M.

| Tier | depth | embed | ff | params | scale | p50 ms | p90 ms | p99 ms | max ms | >150 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M (anchor) | 5 | 384 | 3 | 8.56M | 1.00× | 18.4 | 24.2 | 33.3 | 51 | 0/3600 |
| M7F4 (current) | 7 | 384 | 4 | 13.58M | 1.68× | 24.4 | 30.2 | 34.8 | 73 | 0/3600 |
| L (released) | 11 | 448 | 3 | 23.40M | 2.99× | 36.5 | 46.0 | 52.0 | 61 | 0/3600 |
| L16 | 16 | 480 | 3 | 38.28M | 5.00× | 54.5 | 69.1 | 89.0 | 127 | 0/3600 |
| XL20 | 20 | 512 | 3 | 53.92M | 7.11× | 76.3 | 93.6 | 108.7 | 288 | 4/3600 |
| XL24 | 24 | 528 | 3 | 68.46M | 9.07× | 86.0 | 101.5 | 116.7 | 208 | 6/3600 |

Latency grows clearly sublinearly in the cost model (9.07× compute buys
only 4.7× p50 over M): the small nets pay a fixed per-move overhead
(dispatch, obs pipeline, host transfer) that the big nets amortize. The
p99-vs-scale slope from L16 to XL24 is ~7 ms per M-unit; extrapolating to
150 ms gives the ~13–14× crossing quoted in the verdict.

The 4–6 moves over 150 ms at XL20/XL24 (max 288 ms) are isolated
shared-host stalls, not compute: they do not appear at L16 and below and
they do not scale with model size. At the competition's fault pricing
(pass + one fault per late move, 50 faults to forfeit) even taking them at
face value costs ~1–2 faults per 1,200-move game.

## Host anchoring

This host runs M at burst p50 18.4 ms / p99 33.3 ms where the Phase 2 host
measured 11.0 / 18.5 — ~1.7× slower per core, inside the known ~2× Modal
fleet spread. Absolute numbers above are therefore on the conservative
side, but a competition host is not guaranteed to be faster; the verdict's
1.5× margin covers a host as much slower than this one as this one is
slower than Phase 2's.

## Strict container: the CFS artifact, restated with a size trend

The strict 1-core-quota container reproduces the Phase 2 throttling
artifact — p99 390–530 ms at every size, spike magnitude independent of
net size — but adds one new fact: **throttle frequency grows with model
size.** Moves over 150 ms: M 68, M7F4 109, L 174, L16 268, XL20 395,
XL24 481 (of 3,600 each); at XL20+ the strict p90 is already ~480 ms.
More wall-time per move means more CFS periods crossed per move, so on a
quota-shaped host (not the dedicated core RULES.md promises) a big net is
throttled on ~13% of moves, not ~2%. Strict p50s stay clean (M 12.2,
XL24 69.5) and read ~1.2–1.5× faster than burst p50s on their (different)
host — same-host contrasts only, as ever.

Peak RSS with all six nets resident was 1.67 GB (strict, under the hard
2 GB cap). A single-net jax XL24 process measured **695 MB peak RSS**
(local `/usr/bin/time -l`, 20 jitted steps; the in-container per-net RSS
deltas imply ~900 MB on the Modal image) — comfortably inside 2 GB either
way.

## Caveats

- This prices the **jax-CPU python path** only. joe-rs has its own cost
  structure — never quote these numbers for joe-rs; its own ladder is
  [joe-rs-size-ceiling.md](joe-rs-size-ceiling.md) (linear scaling, lower
  ceiling).
- Latency ceiling only. Nothing here says a bigger net trains to be
  stronger, or how much longer 3–9× compute per training step takes on the
  vast.ai side.
- Random weights, random plausibly-scaled observations; FLOPs are
  input-independent.
- `cpu_model` is masked on Modal (gVisor), so the host class cannot be
  named — only anchored via M.

## Reproducing

```bash
.venv/bin/modal run scripts/joe_modal_size_ceiling.py > /tmp/joe_size_ceiling.log 2>&1 &
```
