# Diversity matrix (stub)

Tracking file for the **check-bot-diversity** skill. Hard rules live in
[`diversity-constraints.md`](diversity-constraints.md) — read that file first.

## How to use

1. Compare the new spec against every bot on the five axes in
   [`diversity-constraints.md` §3](diversity-constraints.md).
2. Record the verdict here: `distinct` / `overlap` / `duplicate`.
3. One table row per competition-roster bot. `classic_duel` is remote-only;
   see human-95-plan §2.3 carve-out. Research bots may appear in a separate
   block when reviewed.

## Matrix

| Bot | Primary objective | Risk | Time profile | Information | Army handling | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| `sosipolis` | general kill | aggressive | early castle + find-and-strike | fog memory + contact priors | early castle then strike gather | **distinct** |

`sosipolis` owns the research axis **directed triple-MCTS general search**
([`diversity-constraints.md` §3.3](diversity-constraints.md)). It does not
replace `fog_scout`, `general_hunter`, or `castle_builder`.

Round evidence: [`../measurements/round1.md`](../measurements/round1.md).
