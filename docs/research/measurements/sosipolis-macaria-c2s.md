# sosipolis-macaria-c2s — shared contact→sight panel

Date: 2026-08-02  
Round: `sosipolis-macaria-c2s`  
Grid: seeds 0–4, `--seat-policy alternate`, `--record` (10 games)  
Probe field: sticky `enemy_land_visible` (schema + macaria/sosipolis probes)  
Analyzer: `scripts/analyze_sosipolis_sight.py --bot <name>`

## Targets (Kubic wins)

| Metric | Target |
| --- | ---: |
| Contact median | ≤ 82 |
| Contact→sight median | ≤ 95.5 |
| Sight→kill median | ~20–24 |
| Sight rate | ~100% in wins |

## Results (same 10 games)

| Bot | W–L | Contact rate | Contact median | Sight rate | C→S median (n) | S→K median (wins) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **Kubic** (ref) | — | — | **82** | ~100% wins | **93** | **20–24** |
| **sosipolis** | 2–8 | 10/10 | **83** | **3/10 (30%)** | **115** (n=3) | **77** (n=2) |
| **macaria** | 8–2 | 10/10 | **83** | **8/10 (80%)** | **192.5** (n=8) | **1** (n=8) |

Both bots use `enemy_land_visible` on every seat (no phase fallback).

### Per-game highlights

- Sosipolis C→S values when sighted: 438, 96, 115. One win (seed 3 seat B) hit C→S **96** (near the ≤95.5 gate) then S→K **51**.
- Macaria often sights on the kill turn (S→K 0 or 1 in 5 of 8 wins). Long C→S does not mean a slow finish.

## Verdict

Contact clocks match (median 83 ≈ Kubic 82). Sosipolis still fails mainly on **sight rate / contact→sight**. Macaria’s contact→sight is longer in this panel, but sight→kill is near-instant.

## Reproduce

```bash
python -m arena.tournaments.competition \
  bots/sosipolis/run.sh bots/macaria/run.sh \
  --round sosipolis-macaria-c2s \
  --seeds 0-4 --seat-policy alternate --no-ratings --record --jobs 4
PYTHONPATH=. python scripts/analyze_sosipolis_sight.py \
  --round sosipolis-macaria-c2s --bot sosipolis
PYTHONPATH=. python scripts/analyze_sosipolis_sight.py \
  --round sosipolis-macaria-c2s --bot macaria
```
