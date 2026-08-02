# 020 — sosipolis: ContactMCTS + early castles

Bot: [`bots/sosipolis/`](../../../bots/sosipolis/).
Spec: [`../strategies/sosipolis.md`](../strategies/sosipolis.md).
Prior: [`019-sosipolis-dual-mcts.md`](019-sosipolis-dual-mcts.md).

## Hypothesis

On seeds 0–19 vs `macaria` (both seats), Sosipolis with ContactMCTS + early
castles raises win rate versus the dual-MCTS baseline, and median first-sight
timing moves toward Kubic’s ~182 (or below the prior loss band), while
per-move wall time stays ≤ 100 ms.

## Kubic castle calibration (observational)

Sample of 98 Kubic wins (`scripts/replay.py events`):

| Metric | Value |
| --- | --- |
| median castles | 1 |
| mean castles | 1.12 |
| % ≥1 castle | 76.5% |
| median first castle tick | 10 |

## Changes

1. **ContactMCTS** — phase when enemy land is known and general is unknown.
2. **Economy** — local RULES §03 build cost; thin early castle programme.
3. Phase machine: `search` → `contact` → `strike`.

## Gate

Seed 0 vs `smoke`: finished (draw or decisive), castles built ≥1 observed,
no faults.

## Macaria grids

| Round | Hash | Games | sosipolis W-L-D | Mean turns | Notes |
| --- | --- | ---: | --- | ---: | --- |
| sosipolis-r2 | `a3373cd60424` | 40 | 0-38-2 | 307.9 | contact+castle defaults; castles ≥1 in 92.5% |
| sosipolis-r2a | `30b8d8a63dd1` | 20 | 0-20-0 | 349.6 | Group A: keep (longer survival) |
| sosipolis-r2b | (A+B) | 20 | 0-20-0 | 416.6 | Group B: keep (longer survival) |
| sosipolis-r2c | `9e5fbe70ac78` | 20 | 0-20-0 | 289.6 | Group C: **revert** (shorter survival) |

Final shipped params = Group A + Group B, Group C reverted.
Final content hash: `d7646e380ea6`.

### Parameter revisions

| Group | Change | Verdict |
| --- | --- | --- |
| A Contact | `CONTACT_REWEIGHT` 0.65→0.80, `CONTACT_SECTOR_FOCUS` 0.85→0.95 | **keep** (mean turns +42 vs r2) |
| B Castle | `CASTLE_MAX` 2→1, `CASTLE_MIN_LAND` 8→5, `CASTLE_ABORT_TURN` 200→100 | **keep** (mean turns +67 vs r2a) |
| C Strike | `STRIKE_TOWARD_BIAS` 0.84→0.92, `FINISH_MARGIN` 2→1, `GATHER_WAVE_HINT` 4→6 | **revert** (mean turns −127 vs r2b) |

Full decision-rule Elo contrast remains **unproven** (≪ 200 games/arm). Win
rate vs macaria did not move off 0% on these grids. Survival length improved
under A+B; conversion still fails.

## Decision

**Keep ContactMCTS + castle programme + A/B param revisions as research
scaffold.** Hypothesis of beating macaria is **falsified** on this panel.
Next work is structural (defense / search conversion), not another strike
knob pass.

Reports: [`../measurements/sosipolis-r2.md`](../measurements/sosipolis-r2.md),
[`../measurements/sosipolis-r2a.md`](../measurements/sosipolis-r2a.md),
[`../measurements/sosipolis-r2b.md`](../measurements/sosipolis-r2b.md),
[`../measurements/sosipolis-r2c.md`](../measurements/sosipolis-r2c.md).
