---
name: check-bot-diversity
description: >-
  Compares a new or changed bot strategy against every existing bot on
  objective, risk, timing, information, and army-handling axes, then records the
  verdict in the diversity matrix. Use when adding a bot, reviewing a strategy
  spec, or checking that bots do not converge on the same behavior.
---

# Check bot diversity

## Model split

- Think model: decides the verdict and updates the diversity matrix
- Composer: nothing

**Composer must not invent a threshold.** When a value is absent from the specification, Composer stops and asks the think model.

## Inputs

- A new or changed specification under `docs/research/strategies/`
- All existing strategy files in that directory
- Latest round report: [`docs/research/measurements/`](../../../docs/research/measurements/)

## Output

- Verdict: `distinct` / `overlap` / `duplicate`
- Updated row in [`docs/research/strategies/diversity-matrix.md`](../../../docs/research/strategies/diversity-matrix.md) (create the file when absent)

## Diversity axes

One value per axis for each bot:

| Axis | Example values |
| --- | --- |
| Primary objective | land, castles, general kill, denial |
| Risk posture | defensive, balanced, aggressive |
| Time profile | early, mid, late, phase-switch |
| Information use | visible only, fog memory, route memory |
| Army handling | full stack, split, convoy, garrison reserve |

## Rules

- Two bots that match on **four axes or more** are a **duplicate**. Change the specification or drop the bot.
- One table, one row per bot.
- **Behavior evidence beats intent.** When the round report shows the same win, loss, and draw pattern against every opponent, mark **overlap** even when the axes differ.

Authority: [`docs/research/strategies/skills-workflow.md`](../../../docs/research/strategies/skills-workflow.md).

## Changelog

- 2026-07-31 — Initial skill from skills-workflow taxonomy (cause: skills-workflow build)
