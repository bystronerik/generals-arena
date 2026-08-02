# Stage B probes — `sosipolis-kubic-clocked-probes`

Date: 2026-08-02  
Bot: clocked Kubic shell (`sosipolis`)  
Opponent: `smoke`  
Grid: seeds 0–4, `--seat-policy alternate`, `--record` (10 games)  
All 10 games: sosipolis win (no faults).

Analyzer: [`scripts/analyze_sosipolis_clocked_probes.py`](../../../scripts/analyze_sosipolis_clocked_probes.py)

## Results

| Probe | Target | Result | Gate |
| --- | --- | --- | --- |
| Land @ t=50 | median ∈ [20, 25] | **19.5** (50% in band; values 15–22) | FAIL |
| Wave / gather land-gain | ≫ 1 (Kubic ~2.6×) | **1.95** median | PASS |
| Chain continue | high (~0.78; floor 0.50) | **0.95** median | PASS |
| Castle earliest | none before 116 | **100%** ok; first median **168** | PASS |
| Pass rate after t=50 | < 0.05 | **0.0** | PASS |
| Post-sight toward_frac | ≥ 0.70 | **0.52** median (sight rate **100%** vs smoke) | FAIL |

## Verdict

Architecture core **holds**: chain, castle clock, pass rate, and gather/wave
direction are green.

Two soft fails drive the next code work:

1. **Opening land@50 is thin** (median 19.5 vs Kubic 24). Raise early expand /
   flood tempo under the opening mask before Elo work.
2. **Strike toward_frac is weak** (0.52 vs gate 0.70) even with full sight vs
   smoke. Tighten tip march roots / toward bias after sight; gather residues
   still consolidating mid-strike may dilute the metric.

Wave/gather ratio (~2.0) passes the direction gate but sits below Kubic ~2.6× —
treat as secondary after land@50.

## Reproduce

```bash
python -m arena.tournaments.competition \
  bots/sosipolis/run.sh bots/smoke/run.sh \
  --round sosipolis-kubic-clocked-probes \
  --seeds 0-4 --seat-policy alternate --no-ratings --record --jobs 4

PYTHONPATH=. python scripts/analyze_sosipolis_clocked_probes.py \
  sosipolis-kubic-clocked-probes --write
```

Machine-readable: [`sosipolis-kubic-clocked-probes.json`](sosipolis-kubic-clocked-probes.json)
