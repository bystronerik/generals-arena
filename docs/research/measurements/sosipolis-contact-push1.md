# sosipolis-contact-push1 — push-expand contact (reverted)

Date: 2026-08-02  
Experiment: [`../experiments/030-sosipolis-contact-push.md`](../experiments/030-sosipolis-contact-push.md)  
Grid: seeds 0–4, `--seat-policy alternate`, `--record` (10 games)  
Analyzer: `scripts/analyze_sosipolis_sight.py --bot sosipolis`

## Results (sosipolis)

| Metric | Value |
| --- | ---: |
| W–L–D | 0–9–1 |
| Contact rate | 10/10 |
| Contact median | 81.5 |
| Sight rate | **0/10 (0%)** |
| Contact→sight | n/a |

Baseline contact-p1: sight **4/10**, c→sight **373**, W–L **3–7**.

## Verdict

Worse than baseline. Contact clock still fine; sight conversion collapsed.
Code reverted; see experiment 030.
