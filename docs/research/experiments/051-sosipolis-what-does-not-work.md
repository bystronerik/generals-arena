# 051 — sosipolis vs macaria: ten things that do not work

Bot: [`bots/sosipolis/`](../../../bots/sosipolis/) — follows 050.
**Negative results.** Nothing here is in the bot. The point is to stop the next
attempt paying for these again.

Baseline for every row: the committed bot at 050, **7/20 = 35%** against
`macaria`, one game per seed, clean harness (no probe wrapper — the bot is
deadline-driven and instrumentation changes outcomes).

## The target is reachable

`Kubic` beats `erik.bystron` (= `bots/macaria`) **11W–1L, 92%** in the scraped
corpus. An earlier claim in this session that Kubic never met macaria, and the
Elo argument built on it, were both wrong. 70% is not a new-bot problem.

Against macaria specifically Kubic reaches a **0.36** land share and holds
macaria to **0.19** — the inverse of our games. Kubic's corpus-wide 0.24
plateau is the wrong target for this matchup.

## What failed

| # | change | proxy it moved | wins /20 |
| --- | --- | --- | --- |
| 1 | expand into neutrals during contact | land 0.19 → 0.22 | worse (0/10, 2/10) |
| 2 | home muster — haul army to a threatened general | — | 3 |
| 3 | peek: one step to reveal a high-belief candidate | sight 3/10 → 4/10 | 4 |
| 4 | local sweep once the tip reaches its waypoint | closest 4 → 5 cells | 7 (no change) |
| 5 | raid: prefer taking macaria's cells | enemy share 41% → 61% | 5 |
| 6 | allow on-route neutrals in the wave clock filter | neutral share ↑ | 4 |
| 7 | frontier muster — gather to the front, not the tip | wave-start adjacency 43% → 64% | 7 (wash) |
| 8 | push the tip outward instead of walking it back | — | **1** |
| 9 | latched defence + wire the dead home bank + hold the general | reversals 14.8% → 22.7% (worse) | not run |
| 10 | sticky tip identity | **reversals 14.8% → 4.4%** | **3** |

Row 10 is the important one. It fixes the largest measurable defect in the bot
— see below — by a factor of 3.4, and halves the win rate. Row 8 copied a
decision read directly off a Kubic replay frame and scored 1/20.

## The defect that fixing made things worse

Post-t50 move quality, ours against Kubic's in the six head-to-head games with
reconstructed actions:

| | sosipolis | Kubic |
| --- | --- | --- |
| immediate reversals (A→B then B→A) | **14.8%** | 0.7% |
| stepped onto a cell left within 6 turns | **27.3%** | 2.5% |
| tip identity changes | **70% of turns** | — |

97% of the reversals coincide with a tip change. `select_mass_tip` re-picks by
raw mass every tick and `STRIKE_TIP_HOLD` only keeps a tip already holding 12
army *and* within 80% of the best stack, so a working tip is essentially never
held. Every rung — recall, the probe march, the gather rally, the wave — takes
"the tip" as its subject, and that subject is a different stack most turns.

Three independent attempts to stabilise it (rows 9, 10, and deleting the
recall arm that walks at an enemy it cannot beat) all made either the defect
or the win rate worse. The thrashing appears to be load-bearing.

## What did work, and the pattern

Every change that helped removed a defect; every change that imitated a Kubic
behaviour failed.

| worked | effect |
| --- | --- |
| 046 chase tie-break — a coordinate sort was picking the southernmost cell | never sighted → win on seed 1 |
| 046 `CONTACT_PREP_BUDGET_MS` 3 ms against a 4.5 ms refresh | 67% of ticks skipped the commit; also made the bot deterministic |
| 049 opening transit — the frontier has to sit next to the army source | land@50 16 → 22 |
| 050 sight floor blocking a kill that was already on | 5/20 → 7/20 |

## Measurements worth keeping

- Gather window is **correct**: Kubic is 86% own-land at residues 10–19 and
  78% at 20–27; ours is 10–27. Do not touch it.
- The gap is the **wave**: Kubic captures on 61–72% of wave moves, we manage
  38%.
- Chain continuation already matches: ours 76%, Kubic 75%.
- Enemy-capture share already matches: ours 18%, Kubic 20%.
- Funnel: contact 10/10, **sight 5/10**, win-given-sight 3/5. Closest approach
  to their general is a median of 4 cells, twice with 50+ army on the tip.
- `home_bank_target` / `home_bank_deficit` in `components/threat.py` are **dead
  code** — the bank was designed and never called, so the general drains on
  schedule with a killer walking in.

## Method note

Two of this session's own measurements were wrong and were caught by acting on
them:

- "72% of Kubic's post-50 gains are taken off macaria" counted cells changing
  owner, which includes the **whole-territory transfer when a general falls**.
  The reconstructed action logs put Kubic's enemy-target moves at 20% against
  our 18%. Row 5 was built on the error.
- The Elo framing above.

Prefer `_derived_actions/` over owner-grid diffs for anything about what Kubic
*did*. [`scripts/shadow_kubic_decisions.py`](../../../scripts/shadow_kubic_decisions.py)
replays a game and asks what sosipolis would choose from the same fogged
observation: 21% agreement, and on 528 of 1029 disagreements we pick the same
kind of move to a different cell.

## Reproduction

```bash
for s in $(seq 0 19); do
  python competition-module/competition/matchup.py \
    bots/sosipolis/run.sh bots/macaria/run.sh --mode competition --seed $s
done
python scripts/shadow_kubic_decisions.py 24988 24989 24990 24991 24992 24993
```
