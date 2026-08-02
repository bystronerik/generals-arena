# 025 — sosipolis: Kubic tip mass (min tip + exclusive waves)

Bot: [`bots/sosipolis/`](../../../bots/sosipolis/).
Prior: [`024-sosipolis-tip-delivery.md`](024-sosipolis-tip-delivery.md).

## Hypothesis

Mass-first tip selection, absolute tip bars (`STRIKE_MIN_TIP` / army frac),
contested path cost, and exclusive pre-sight / pre-march feed waves raise median
tip army toward Kubic’s ~50–55 and increase kill-ready adjacent turns.

## Design

- New [`components/tip.py`](../../../bots/sosipolis/components/tip.py): mass-first
  `select_mass_tip`, contested `path_finish_need`, `tip_mass_target` (path +
  `STRIKE_MIN_TIP`), `tip_feed_target` (also `STRIKE_TIP_ARMY_FRAC`), exclusive
  `feed_tip_action`.
- March gate = `tip_is_ready` (path + min tip only). Frac is aspirational feed,
  not a march lock (tip2 bug: tip~60 still starved behind 0.4× army).
- Brain hard-overrides tip feed while underfed (contact + strike).
- Contact unlocks hunt march at `CONTACT_ASSAULT_STACK` (not frac).

## Gate

Seed 0 vs `smoke`: sosipolis capture, no faults (tip2 and tip3 builds).

## Macaria

| Round | Games | W-L-D | Mean turns | Notes |
| --- | ---: | ---: | ---: | --- |
| sosipolis-tip1 | 10 | 1-9-0 | 323 | tip delivery baseline |
| sosipolis-tip2 | 10 | 1-9-0 | 333 | mass + frac-in-ready |
| sosipolis-tip3 | 10 | 0-10-0 | 357 | march bar = path+min only |

Recorded A-seat tip2 (frac locked march):

| Seed | Sight | Strike | Tip close | Max tip | Kill-ready |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0–3 | none | 0 | — | — | — |
| 4 | 297 | 23 | 8/9 | **60** | 0 (frac~0.28) |

Recorded A-seat tip3 (path+min march):

| Seed | Sight | Strike | Winner |
| ---: | ---: | ---: | --- |
| 0 | none | 0 | macaria |
| 2 | 191 | 2 | **sosipolis** |
| 4 | 333 | 2 | **sosipolis** |

Tip mass on the one tip2 sight game reached Kubic’s ~50–55 band (`tip_max=60`).
Sight rate on A-seat tip2 fell (1/5). Grid winrate did not improve vs tip1.

## Decision

**Keep the tip module and march/feed split; revise next.** Absolute tip mass is
reachable. Exclusive contact feed + high min tip still delay sight vs Macaria,
and kill-ready adjacent stays near zero on long strike windows. Next: cut contact
exclusive-feed time (or lower `STRIKE_MIN_TIP` when path cost is already small)
so hunt resumes once the tip is assault-sized.

Reports: [`../measurements/sosipolis-tip2.md`](../measurements/sosipolis-tip2.md),
[`../measurements/sosipolis-tip3.md`](../measurements/sosipolis-tip3.md).
