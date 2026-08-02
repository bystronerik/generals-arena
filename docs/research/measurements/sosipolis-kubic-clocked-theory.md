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

Recorded panel: [`sosipolis-kubic-clocked-probes.md`](sosipolis-kubic-clocked-probes.md)
(10 games vs smoke, seeds 0–4 alternate).

| Probe | Observation |
| --- | --- |
| No real castle before 116 | **PASS** (first median 168) |
| Match finishes / no fault | **PASS** (10/10 sosipolis wins) |
| Land @ t=50 ∈ [20,25] | **FAIL** (median 19.5; 50% in band) |
| Wave ≫ gather capture | **PASS** (median ratio 1.95) |
| Chain continue rate | **PASS** (median 0.95) |
| Pass rate after t=50 | **PASS** (0.0) |
| Post-sight toward_frac ≥ 0.70 | **FAIL** (median 0.52; sight 100% vs smoke) |

## Stage C — macaria A/B

| Item | Value |
| --- | --- |
| Round | `sosipolis-kubic-clocked-macaria-10` |
| Games | 20 (seeds 0–9, alternate seats) |
| Outcome | **sosipolis 6–14 (30.0% WR)** |
| Contrast | tip5 unclocked **17.5%**; pure conveyor **0/10** |

## Verdict

Architecture gate: **keep** for further tune. Stage B shows chain/castle/pass
clocks work; opening land@50 and post-sight toward_frac need the next revision
before a tip5-scale Elo panel.
