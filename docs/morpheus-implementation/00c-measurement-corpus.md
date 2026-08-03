# Part 00c: Measurement corpus

## Deliverable

Select and record a named competition panel before tensor, belief, curriculum,
and self-play measurements need trajectories.

The output is a versioned panel config plus verified trajectories under
`data/trajectories/morpheus-bootstrap/`.

**Touches**

- `arena/`: use the existing tournament and trajectory paths.
- `scripts/`: add corpus selection, recording, and coverage reporting.
- `data/bot_versions/`: register the existing source bots through the
  tournament parent.
- `bots/`: no bot changes.

## Prerequisites

- [Part -1: Process substrate](-1-process-substrate.md)

## Source specifications

- [Curriculum bootstrap](../bots/morpheus/training.md#sparse-reward-solution)
- [Opponent mixture](../bots/morpheus/training.md#opponent-mixture)
- [Arena trajectories](../arena/trajectories.md)
- [Decision rule panel requirements](../arena/decision-rule.md#setup)

## Defaults and replacement measurement

Do not invent a hidden panel. Before recording, write
`scripts/configs/morpheus/bootstrap-panel.json` with exact bot IDs, content
hashes, selection date, rating era, and role.

The panel must include the anchor, heuristic bots, and research bots. Select
members from the current registered competition pool and record the rating and
decisive-game evidence used.

## Implementation boundary

Run only competition-mode games. Use both seat orientations and record engine
trajectories. Keep scraped ResBot, classic, and remote games out.

Report board-size, turn-band, outcome, contact, sight, castle, and deathtouch
coverage. Keep source labels on every trajectory.

## Isolated test

```bash
python scripts/morpheus_corpus.py record \
  --panel scripts/configs/morpheus/bootstrap-panel.json \
  --round morpheus-bootstrap \
  --round-seed 7 \
  --seat-policy alternate \
  --output data/trajectories/morpheus-bootstrap
```

```bash
python scripts/morpheus_corpus.py verify \
  --trajectories data/trajectories/morpheus-bootstrap \
  --report docs/research/measurements/morpheus-bootstrap-corpus.json
```

```bash
python competition-module/competition/matchup.py \
  bots/smoke/run.sh bots/cm_expander/run.sh \
  --mode competition --seed 0
```

## Specification gaps

The specs describe panel categories but do not name members. This part owns
that decision and must finish before a later part consumes the panel.

No minimum trajectory count is specified. The coverage report must name every
missing class or event instead of treating absent data as zero.

## Exit criterion

Answer `yes` if the panel names exact registered hashes, all games are
competition mode, every trajectory verifies in one engine era, and the report
contains enough decisive and forced-mismatch cases for Parts 03, 05, and 10 to
run their measurements.

Answer `no` if the panel remains categorical, trajectories fail replay, or a
required measurement class has no data.
