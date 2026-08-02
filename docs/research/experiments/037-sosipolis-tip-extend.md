# 037 — sosipolis: tip-arrival extend along hunt line

Bot: [`bots/sosipolis/`](../../../bots/sosipolis/) — **reverted**.
Baseline: [`../measurements/sosipolis-contact-p1-diag.md`](../measurements/sosipolis-contact-p1-diag.md) (sight 40%).
Prior fail: [`036-sosipolis-no-teleport.md`](036-sosipolis-no-teleport.md) (foot filter → 10%).

## Defect

When the tip reaches an empty committed waypoint, invalidate free-picks
`macros[0]` and teleports 8–25 cells into empty fog.

## Hypothesis

On tip-arrival only: extend past the tip along home→first_contact→tip, else
pick the nearest macro to the tip. Leave belief, foot filters, jump caps, and
normal switches unchanged. Sight rate rises above contact-p1’s 40%.

## Revision (tried, then reverted)

| Change | Intent |
| --- | --- |
| Tip-arrival → `extend` macro | continue hunt line, no free re-pick |
| Fallback: nearest distinct macro to tip | local only if extend blocked |
| Tip-arrival skipped in `_commitment_invalid` free path | stop tip-arrival teleports |
| `CONTACT_EXTEND_STEPS=4` | named step budget |

## Gate

Seed 0 vs `smoke`: **draw at 1200** (4 castles). Weak vs baseline smoke wins.

## Measurement

Round: `sosipolis-extend1`.

| Round | Sight rate | Median C→S | W–L |
| --- | ---: | ---: | --- |
| contact-p1-diag | **4/10 (40%)** | 373 | 3–7 |
| extend1 | **1/10 (10%)** | 626 | 1–9 |

## Verdict

**Revert.** Tip-arrival extend alone did not raise sight and hurt the smoke
gate. Nearest-macro fallback can still be far when every scored macro is far;
voluntary switches (age / opposite-side) still teleport. Contact-p1 remains
best among 034–037.
