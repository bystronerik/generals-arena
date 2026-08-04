# Morpheus implementation

This directory is the implementation plan for the
[Morpheus design](../bots/morpheus/index.md). The design files remain the source
of truth. This directory owns order, dependencies, tests, and exit criteria.

The plan uses the competition rules in [RULES.md](../../RULES.md) and the
repository process in [AGENTS.md](../../AGENTS.md). It does not use
`docs/bots/morpheus/spec-review.md`.

## Phase order

### Phase 0: retire external risk

1. [Process substrate](-1-process-substrate.md) defines placement and test gates.
2. [Modal JAX preflight](00-modal-jax-preflight.md) tests the A100 JAX path.
3. [Sandbox export preflight](00b-sandbox-export-preflight.md) tests the pinned
   8-bit runtime path.
4. [Measurement corpus](00c-measurement-corpus.md) records the panel data used
   by later measurements.
5. [Protocol shell](01-protocol-shell.md) creates the first runnable stub.
6. [Submission harness](08-submission-harness.md) proves that judge failures
   are detected.

Part 01 is the earliest point at which a degraded Morpheus can pass the
competition gate. The stub is not a trained Morpheus, is not eligible for
promotion, and must not enter a rating decision.

### Phase 1: deployable decision stack

1. [Transition kernel](02-transition-kernel.md)
2. [Observation and actions](03-observation-actions.md)
3. [Network and export](04-network-export.md)
4. [Belief filter](05-belief-filter.md)
5. [Simultaneous search](06-simultaneous-search.md)
6. [Runtime controller](07-runtime-controller.md)
7. [Online qualification](09-online-qualification.md)
8. [Complete-turn cost reduction](09a-complete-turn-cost.md)

### Phase 2: training inputs and self-play

1. [Curriculum data](10-curriculum-data.md)
2. [Self-play league](11-self-play-league.md)
3. [Training objective](12-training-objective.md)

### Phase 3: compute decision

1. [Modal compute gate](13-modal-compute-gate.md)

Parts 00, 12, and 13 can use only bounded qualification time. Part 12 charges
its A100 use to Part 13. The main training run cannot start until Part 13
returns `yes`.

### Phase 4: train and promote

1. [Trainer and checkpoints](14-trainer-checkpoints.md)
2. [Artifact freeze](15-artifact-freeze.md)
3. [Arena promotion](16-arena-promotion.md)

## Dependency graph

```mermaid
flowchart TD
    PN1["-1 Process substrate"]
    P00["00 Modal JAX preflight"]
    P00B["00b Sandbox export preflight"]
    P00C["00c Measurement corpus"]
    P01["01 Protocol shell"]
    P02["02 Transition kernel"]
    P03["03 Observation and actions"]
    P04["04 Network and export"]
    P05["05 Belief filter"]
    P06["06 Simultaneous search"]
    P07["07 Runtime controller"]
    P08["08 Submission harness"]
    P09["09 Online qualification"]
    P10["10 Curriculum data"]
    P11["11 Self-play league"]
    P12["12 Training objective"]
    P13["13 Modal compute gate"]
    P14["14 Trainer and checkpoints"]
    P15["15 Artifact freeze"]
    P16["16 Arena promotion"]

    PN1 --> P00
    PN1 --> P00B
    PN1 --> P00C
    PN1 --> P01
    PN1 --> P02
    PN1 --> P08
    P00B --> P04
    P00C --> P03
    P00C --> P05
    P00C --> P10
    P00C --> P11
    P01 --> P03
    P01 --> P08
    P02 --> P03
    P03 --> P04
    P02 --> P05
    P03 --> P05
    P04 --> P05
    P02 --> P06
    P04 --> P06
    P05 --> P06
    P01 --> P07
    P06 --> P07
    P04 --> P09
    P05 --> P09
    P06 --> P09
    P07 --> P09
    P08 --> P09
    P02 --> P10
    P03 --> P10
    P05 --> P10
    P00 --> P11
    P06 --> P11
    P07 --> P11
    P10 --> P11
    P03 --> P12
    P04 --> P12
    P10 --> P12
    P11 --> P12
    P09 --> P13
    P00C --> P13
    P10 --> P13
    P11 --> P13
    P12 --> P13
    P00 --> P13
    P10 --> P14
    P11 --> P14
    P12 --> P14
    P13 --> P14
    P08 --> P15
    P09 --> P15
    P14 --> P15
    P15 --> P16
```

## Gates

Every part that leaves `bots/morpheus/run.sh` runnable must finish this match:

```bash
python competition-module/competition/matchup.py \
  bots/morpheus/run.sh bots/smoke/run.sh \
  --mode competition --seed 0
```

The match must end by a win, loss, draw, or turn-1200 truncation.

Promotion also requires all of these results:

- Part 08 rejects every controlled judge failure.
- Part 09 accepts one deployment configuration with zero faults.
- Part 13 proves that the training schedule fits the compute budget.
- Part 15 accepts the exact frozen bundle.
- Part 16 either accepts the first artifact as a safe bootstrap baseline or
  reports the pairwise contrast required by
  [the decision rule](../arena/decision-rule.md) for later artifacts.

## Test placement

Part -1 registers the `morpheus` marker and excludes it from the default suite.
Fast pure tests can enter the normal suite only after a warm measurement.
Morpheus model, JAX, real-match, Modal, and subprocess tests use this explicit
gate:

```bash
python -m pytest -m morpheus -q
```

The normal warm suite must remain below the 7-second limit in
[AGENTS.md](../../AGENTS.md).

## Code boundary

The planned runtime closure is `bots/morpheus/`. Thin operator entry points are
under `scripts/`. The submission harness is under `arena/`.

`AGENTS.md` does not define a location for training-only modules. The proposed
location is `training/morpheus/`. Part -1 owns that placement rule and the
derived-data retention rule before training code lands.
