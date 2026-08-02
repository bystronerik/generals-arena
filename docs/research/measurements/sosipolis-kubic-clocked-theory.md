# Sosipolis clocked Kubic MCTS — verification note

Date: 2026-08-02  
Behavioural source: [`../strategies/kubic-behavior-spec.md`](../strategies/kubic-behavior-spec.md)  
Grid: [`sosipolis-kubic-clocked-macaria-10.md`](sosipolis-kubic-clocked-macaria-10.md)

## What shipped

Hard Kubic §2 shell + mod-50 gather/wave masks around Search / Contact / Strike.
Shared helpers in `bots/sosipolis/components/conveyor.py` (not a pure-rule policy).
Castles earliest 116; tip floor 10; rare recall; chain head.

## Stage A — smoke gate

`sosipolis` vs `smoke`, `--mode competition --seed 0`: sosipolis win turn 360;
first castle turn **168** (≥116); 4 castles.

Seeds 1–4 vs smoke: all sosipolis wins; first castles when built were ≥218
(or 0 castles on seeds 2 and 4 — skip allowed).

## Stage B — behavioural probes (soft)

| Probe | Observation |
| --- | --- |
| No real castle before 116 | Pass (smoke seed 0 first @168; others ≥218 or skip) |
| Match finishes / no fault | Pass |
| Land @ t=50 ∈ [20,25] | Not measured this grid (no trajectory land sample) |
| Wave ≫ gather capture | Not measured this grid |
| Chain continue rate | Not measured this grid |
| Pass rate after t=50 | No fault / stall observed on smoke panel |

## Stage C — macaria A/B

| Item | Value |
| --- | --- |
| Round | `sosipolis-kubic-clocked-macaria-10` |
| Games | 20 (seeds 0–9, alternate seats) |
| Outcome | **sosipolis 6–14 (30.0% WR)** |
| Contrast | tip5 unclocked **17.5%**; pure conveyor **0/10** |

## Verdict

Architecture gate: **keep** for further tune. Not Elo-final. Clocked shell beats
both prior failure modes on this macaria panel. Next: land@50 / chain / capture
ratio probes with trajectories, then tip5-scale panel under the decision rule.
