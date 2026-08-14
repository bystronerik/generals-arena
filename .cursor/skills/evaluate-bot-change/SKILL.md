---
name: evaluate-bot-change
description: >-
  Measures a bot change as a pairwise rating contrast with a confidence
  interval, from stored games. Use when comparing bot versions, running an A/B
  round, judging a parameter revision, or deciding keep versus revert.
---

# Evaluate bot change

Follow [`docs/research/experiment-protocol.md`](../../../docs/research/experiment-protocol.md).
One **parameter group** per experiment when possible.

**Every threshold lives in
[`docs/arena/decision-rule.md`](../../../docs/arena/decision-rule.md).** Read it
before reporting a verdict; do not restate its numbers here or invent new ones.

## Model split

- Think model: writes the hypothesis, chooses the opponent panel, interprets the contrast
- Composer: runs the arms, stores games, refits, reports the contrast

**Composer must not invent a threshold.** When a value is absent from
`decision-rule.md`, Composer stops and asks the think model.

## What a decision is

A **pairwise contrast between two content hashes, inside one round**, with an
interval:

```python
from arena.records.ratings.cli import refit

fits = refit()                       # one independent fit per round
fit = fits["<round>"]                # the round both arms played in
delta = fit.delta("expand_plus@<baseline_hash>", "expand_plus@<candidate_hash>")
delta.value, delta.se, delta.ci, delta.p_stronger
```

Never a rank, and never a bare winrate. The two revisions are separate rated
entities because identity is the content hash, so "did expand_plus improve?" has a
direct answer.

**Both arms must be in the same round, and there is no pooled fit to fall back
on.** A baseline measured in an earlier round is not a comparator: two
byte-identical programs measured in different rounds fitted 46 Elo apart, which is
more than the thresholds. Re-measure the baseline in the candidate's round — a
frozen copy of the pre-change bot under `bots/`, kept for the comparison.

## Workflow

1. **Hypothesis** — one claim; optional short note under `docs/research/experiments/`.
2. **Baseline** — record the current hash (`python -m arena.records.fingerprint <bot>`).
   It must not be provisional. Fix the opponent panel and the round seed.
3. **Treat** — apply the change. The hash moves on its own; the registry
   records the new version when the round runs.
4. **Run both arms** on matched seeds with `--seat-policy alternate`.
5. **Store** — every game lands under `data/games/<round>/` before any refit.
6. **Refit and report** — `Δ ± SE`, `CI₉₅`, `P(B > A)`, games per arm, decisive
   games per arm, and the verdict from `decision-rule.md`. If `unproven`,
   report `fit.games_to_resolve(a, b, target_se=12.75)`.
7. **Decide** — keep or revert from the contrast only.

## Commands

```bash
source .venv/bin/activate   # if present; prefer python3.12
```

```bash
python -m arena.tournaments.competition \
  bots/<candidate>/run.sh bots/<panel...>/run.sh \
  --round <name> --games-per-pair 50 --round-seed <int> \
  --seat-policy alternate --strict-versions
```

```bash
python scripts/measure_heuristics.py --round round<N> --games-per-pair 50 --round-seed <int>
```

```bash
python -m arena.records.ratings --print --round round<N>
```

Read the verdict from `fits["<round>"].delta(baseline, candidate)`, never from a
rank. Both arms must be entities in that one round — which is why the baseline is
a frozen copy under its own bot id, kept for the duration of the comparison.

Round reports: [`docs/research/measurements/`](../../../docs/research/measurements/).
Override the roster with `--bots`; default is `DEFAULT_ROSTER` in the script.

## Seat policy

Seat is drawn per game from the seeded stream — it is no longer implied by
roster position, so a round does **not** need to double its games to control
for it.

| Round type | Policy |
| --- | --- |
| Large exploratory / regeneration | `--seat-policy random` (default), zero extra games |
| Decision arms | `--seat-policy alternate`, exactly 50/50 on matched map seeds |

`--swap-sides` no longer exists. It keyed the pair RNG on the oriented pair, so
the mirrored copy drew a different seed list: 2× the games for an unmatched
sample.

## Draw-heavy arms

Draws are **not** uninformative under the Davidson draw model — they pin a
rating band. The rule is quantitative: see gate 3 in `decision-rule.md`
(≥ 60 decisive games per arm). Do not mark an arm unproven merely because the
draw rate is high.

## Checklist

- [ ] Both hashes registered (`python -m arena.records.registry --verify`)
- [ ] Same opponent panel, `--round-seed`, and `--games-per-pair` on both arms
- [ ] `--seat-policy alternate` on decision arms
- [ ] One `engine_version` across both arms (`excluded.engine_mismatch == 0`)
- [ ] Competition mode on every match
- [ ] Games under `data/games/<round>/` before the refit
- [ ] One changed parameter group per experiment
- [ ] Verdict quoted with `Δ`, `SE`, `CI₉₅`, `P(B > A)` — never a rank

Schema: [`docs/arena/game-record-schema.md`](../../../docs/arena/game-record-schema.md).
Ratings: [`docs/arena/ratings.md`](../../../docs/arena/ratings.md).

## Rules

- Do not merge strategy changes without a measured contrast in `data/`.
- No strategy content in this skill — link `docs/` for domain knowledge.

## Changelog

- 2026-07-31 — Contrast-with-interval replaces "Elo delta"; thresholds moved to
  `docs/arena/decision-rule.md`; seat policy replaces seat-order swap
- 2026-07-31 — Rule C parallel rounds + per-round folders
- 2026-07-31 — Seat-order swap, draw rule, measure_heuristics pointer (cause: skills-workflow build)
