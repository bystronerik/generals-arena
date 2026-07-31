# Skills workflow design

Design for the Cursor skill set of this repo. This file is a specification. It
is not a `SKILL.md` file. An implementer agent writes the skill files from this
file.

Scope: process only. Game knowledge stays in `docs/competition/`,
`docs/engine/`, and `docs/bots/`. Strategy specs stay in
`docs/research/strategies/`.

## 1. Current loop

The loop that runs today has six steps:

1. A think model writes a strategy specification under
   `docs/research/strategies/<bot>.md`.
2. Composer 2.5 implements `bots/<bot>/agent.py` from that specification.
3. `scripts/measure_heuristics.py` runs a fixed grid, stores games under
   `data/games/`, and writes `docs/research/measurements/round<N>.{json,md}`.
4. A think model reads the round report and writes a parameter revision.
5. Composer 2.5 applies the parameter revision.
6. Ratings update from stored games only.

Two constraints act on the loop:

- **Diversity constraint.** New bots must not converge on the behavior of an
  existing bot.
- **Learned-bot gate.** The learned bot waits for about 10 heuristic bots plus a
  remote evaluation result.

The skill set below maps one skill to each step, plus meta skills for git
hygiene and skill self-improvement.

## 2. Skill inventory

Verdicts for the four skills under `.cursor/skills/`.

| Skill | Verdict | Reason |
| --- | --- | --- |
| `run-competition-match` | Keep + improve | Correct content. It misses fault detection, seat-order swap, and the draw-heavy reality of this repo. |
| `new-competition-bot` | Improve + rename to `build-bot-from-spec` | The scaffold step and the specification-to-code step always run together. Two skills with near-identical triggers cause the wrong skill to load. |
| `evaluate-bot-change` | Keep + improve | The A/B contract is good. It does not point at `scripts/measure_heuristics.py` or at the round report format. |
| `update-leaderboard` | Keep | Small, single purpose, already correct. Add one line about git hygiene for `data/`. |

No skill is removed. One skill is renamed.

### 2.1 `run-competition-match` — improvements

Add these sections:

- **Fault detection.** A match can finish and still be a failure. Read stderr
  for bot faults, protocol errors, and move-limit overruns. Report a fault as a
  failure even when the engine reports a draw.
- **Seat-order swap.** Run both `A vs B` and `B vs A` for any claim about
  strength. Map generation is not symmetric.
- **Castle telemetry.** The line
  `[matchup] castles built: <a> (<label>) vs <b> (<label>)` is the only castle
  signal that reaches the report. Capture stdout when castles matter.
- **Draw baseline.** Most stored games end as draws at the 1200-turn cap. A
  draw is a valid gate pass. A draw is not evidence of strength.

### 2.2 `new-competition-bot` → `build-bot-from-spec` — improvements

Keep the scaffold instructions. Add these rules:

- The skill takes one input: a path to `docs/research/strategies/<bot>.md`.
- Every threshold in the specification becomes a named module constant in
  `bots/<bot>/agent.py`. A later parameter revision edits constants only.
- `main.py` and `run.sh` stay unchanged unless the wire protocol changes.
- The bot must return a legal move or a pass inside the 150 ms move limit.
- The bot must not read hidden engine state.
- The skill ends with one verification match under `--mode competition`.

Migration cost of the rename: update the skills table in `AGENTS.md`, the
`bot-author` role row, and any link in `docs/bots/adding-a-bot.md`.

### 2.3 `evaluate-bot-change` — improvements

- Point at `scripts/measure_heuristics.py` for a full round and at
  `arena/run_match.py` for a single pair.
- Require a seat-order swap in every grid.
- Add a draw-heavy decision rule: when both arms draw every game, mark the
  result **unproven**, not neutral. Ask for a decisive opponent instead of a
  threshold change.
- Require one changed parameter group per experiment.

### 2.4 `update-leaderboard` — improvements

- Add one line: commit `data/ratings/` snapshots; do not commit
  `data/games/*.json` unless the user asks for the raw games.

## 3. New skills

### 3.1 `write-strategy-spec`

| Field | Value |
| --- | --- |
| Role | Strategist (think model) |
| Inputs | Bot name, target niche, `RULES.md`, `docs/competition/**`, the diversity register, the latest round report |
| Outputs | `docs/research/strategies/<bot>.md`, one link line in `docs/index.md` |
| Freedom | Medium. The skill gives a section template. The strategist chooses the content. |

Section template that the skill must give:

```text
# <bot> strategy specification
## Goal
## Diversity claim          (how this bot differs from every existing bot)
## Strategy features
## State
## Phases and thresholds     (named constants, one value per line)
## Threat or scoring model
## Candidate move rules
## Pseudocode for act()
## Edge cases
## Expected behavior against existing bots
## Experiment hypothesis     (one falsifiable claim + seed grid + metrics)
```

Rules the skill must state:

- Give every threshold a name and a value. Composer must not invent a number.
- Write one falsifiable hypothesis.
- Name the seed grid and the opponent set.
- Do not write Python. Pseudocode only.

`docs/research/strategies/garrison.md` is the reference example.

### 3.2 `check-bot-diversity`

| Field | Value |
| --- | --- |
| Role | Strategist (think model), cheap pass |
| Inputs | A new or changed specification, all files in `docs/research/strategies/`, the latest round report |
| Outputs | A verdict (`distinct` / `overlap` / `duplicate`) and an updated row in `docs/research/strategies/diversity-matrix.md` |
| Freedom | Medium |

The skill must define the diversity axes and require one value per axis:

| Axis | Example values |
| --- | --- |
| Primary objective | land, castles, general kill, denial |
| Risk posture | defensive, balanced, aggressive |
| Time profile | early, mid, late, phase-switch |
| Information use | visible only, fog memory, route memory |
| Army handling | full stack, split, convoy, garrison reserve |

Rules:

- Two bots that match on four axes or more are a **duplicate**. Change the
  specification or drop the bot.
- The register file `diversity-matrix.md` is one table. One row per bot. The
  skill creates the file when the file is absent.
- Behavior evidence beats intent. When the round report shows the same win,
  loss, and draw pattern against every opponent, mark **overlap** even when the
  axes differ.

### 3.3 `run-measurement-round`

| Field | Value |
| --- | --- |
| Role | Evaluator (Composer) |
| Inputs | Round name, bot list, seed list |
| Outputs | Games under `data/games/`, ratings under `data/ratings/`, `docs/research/measurements/round<N>.{json,md}` |
| Freedom | Low. Run the script. Do not write a new runner. |

Content the skill must give:

- Command: `python scripts/measure_heuristics.py --round round<N>`.
- Flags: `--no-wait` to fail fast on a missing `run.sh`, `--no-ratings` to store
  games without a global Elo update.
- The `NEW_BOTS` and `BASELINE_BOTS` lists in the script are the grid. Edit the
  lists in the script; do not pass an ad-hoc grid.
- Create `docs/research/measurements/` when the directory is absent.
- After the run, read the round Markdown file and report: draw rate, mean turns,
  decisive games, and any bot with zero decisive games.
- Never update ratings from stdout. Store first, then rate.

### 3.4 `tune-bot-parameters`

| Field | Value |
| --- | --- |
| Role | Implementer (Composer) |
| Inputs | A parameter revision note, `bots/<bot>/agent.py`, the round report |
| Outputs | Edited constants in `agent.py`, one experiment note under `docs/research/experiments/`, one verification match |
| Freedom | Low |

Rules:

- Edit named constants only. Do not restructure the decision code.
- Change one parameter group per revision.
- Record the old value and the new value in the experiment note.
- Re-run the same seeds and the same opponents as the baseline arm.
- Revert when the change does not improve the reported metric.

### 3.5 `commit-research-increment`

| Field | Value |
| --- | --- |
| Role | Any role, at the end of a step |
| Inputs | The current working tree |
| Outputs | One or more small commits |
| Freedom | Low |

Rules the skill must give:

- Commit after each loop step: specification, bot code, round report, ratings.
- Do not mix a specification commit with a bot code commit.
- Do not commit `data/games/*.json` unless the user asks for the raw games.
  Commit `data/ratings/` snapshots and the round report.
- Never commit `__pycache__/` or `.DS_Store`. Propose a `.gitignore` line when
  such a file appears in `git status`.
- Never commit without an explicit user request when the repo policy requires
  that. State the proposed message and wait.

### 3.6 `improve-skill-from-failure`

| Field | Value |
| --- | --- |
| Role | Meta, think model |
| Inputs | A failure description, the skill that was active, the game id or round name |
| Outputs | An edited `SKILL.md` plus one line in its `## Changelog` section |
| Freedom | Medium |

This skill holds the auto-improve rules in section 4.

### 3.7 Deferred: `learned-bot-readiness`

Do not build this skill now. Build the skill when the heuristic count reaches
about 10 and a remote evaluation path exists. The skill will check the learned-bot
gate: heuristic count, decisive-game rate, telemetry fields, and one remote
evaluation result. See `docs/research/learned-bot-plan.md`.

## 4. Auto-improve rules

A skill file is a product. The skill file changes when the workflow proves the
file wrong.

### 4.1 When to update a skill

| Trigger | Action |
| --- | --- |
| The same failure happens twice under the same skill | Mandatory update. Add the missing step or the missing check. |
| A skill sent the agent to a wrong path, a wrong flag, or a stale command | Immediate update. Correct the command. |
| The agent needed a step that the skill did not name, and the step worked | Add the step after the third use. |
| A skill loaded for the wrong task | Update the `description` line. Do not change the body. |
| A skill did not load for the right task | Add the missing trigger terms to the `description` line. |
| A step was not used in two measurement rounds | Delete the step. |
| A repo path, script name, or schema changed | Update every skill that names the path in the same commit. |

### 4.2 How to update a skill

- Change the smallest amount of text that removes the failure.
- Keep the body under 150 lines. Move detail into `docs/` and link to it.
- Add one line to a `## Changelog` section at the end of the file. Format:
  `- <date> — <change> (cause: <game id, round name, or failure>)`.
- Do not add strategy content. Link to `docs/` instead.
- Do not add a second way to do the same thing. Replace the old way.

### 4.3 What never enters a skill

- Bot strategy, thresholds, or scoring rules.
- Results from one match.
- Time-sensitive text such as "for now" or a date-bound instruction.
- A second command that does the same work as the first command.

## 5. Model split reminders

Each skill body carries one short block. The block tells the reader which model
class owns the step. Use this exact block, with the correct row filled in:

```markdown
## Model split

- Think model: <what the think model decides here>
- Composer: <what Composer executes here>
```

Ownership table for the whole loop:

| Step | Think model | Composer |
| --- | --- | --- |
| Strategy specification | Writes the specification, the thresholds, and the hypothesis | Nothing |
| Diversity review | Decides the verdict | Nothing |
| Bot implementation | Nothing | Writes `agent.py`, runs the verification match |
| Measurement round | Nothing | Runs the script, reports the numbers |
| Round interpretation | Reads the report, writes the parameter revision | Nothing |
| Parameter revision | Chooses the new values | Edits the constants, re-runs the grid |
| Failure diagnosis | Finds the cause, edits the skill | Reproduces the failure |
| Commits | Nothing | Stages and commits |

One rule to repeat in every skill body: **Composer must not invent a
threshold.** When a value is absent from the specification, Composer stops and
asks the think model.

## 6. Exact frontmatter description lines

Use these lines without change. Each line names what the skill does and when the
agent uses the skill.

### `run-competition-match`

```yaml
description: >-
  Runs Generals Competition local matches with --mode competition through
  matchup.py or arena/run_match.py, and checks the result for bot faults. Use
  when starting a match, debugging a stdio bot, verifying the competition gate,
  swapping seat order, or writing a game record under data/games/.
```

### `build-bot-from-spec` (renamed from `new-competition-bot`)

```yaml
description: >-
  Scaffolds bots/<name>/ from bots/smoke/ and implements agent.py from a
  strategy specification under docs/research/strategies/, keeping every
  threshold as a named constant. Use when adding a competition bot, turning a
  strategy spec into code, copying the smoke template, or wiring agent.py,
  main.py, and run.sh for matchup.py.
```

### `evaluate-bot-change`

```yaml
description: >-
  Measures a bot change with a fixed seed grid, both seat orders, before and
  after winrate, and an Elo delta from stored games. Use when comparing bot
  versions, running an A/B tournament, judging a parameter revision, or deciding
  keep versus revert from data/games metrics.
```

### `update-leaderboard`

```yaml
description: >-
  Rebuilds Elo leaderboard snapshots from stored data/games/ through
  arena/ratings.py and publishes JSON and Markdown under data/ratings/. Use when
  refreshing ratings, publishing the leaderboard, or closing a tournament or
  measurement round.
```

### `write-strategy-spec`

```yaml
description: >-
  Writes a competition bot strategy specification under
  docs/research/strategies/ with named thresholds, pseudocode, a diversity
  claim, and one falsifiable hypothesis. Use when designing a new heuristic bot,
  planning a strategy before any code, or revising a spec after a measurement
  round.
```

### `check-bot-diversity`

```yaml
description: >-
  Compares a new or changed bot strategy against every existing bot on
  objective, risk, timing, information, and army-handling axes, then records the
  verdict in the diversity matrix. Use when adding a bot, reviewing a strategy
  spec, or checking that bots do not converge on the same behavior.
```

### `run-measurement-round`

```yaml
description: >-
  Runs the fixed heuristic measurement grid through scripts/measure_heuristics.py,
  stores games under data/games/, updates ratings, and writes
  docs/research/measurements/round<N>.json and .md. Use when measuring a batch of
  bots, starting a new round, or refreshing the round report after bots change.
```

### `tune-bot-parameters`

```yaml
description: >-
  Applies a parameter revision to bots/<name>/agent.py by editing named
  constants only, then re-runs the baseline seed grid and writes an experiment
  note. Use when retuning a bot after a measurement round, changing thresholds,
  or applying a parameter revision from a strategy spec.
```

### `commit-research-increment`

```yaml
description: >-
  Creates small ordered commits for a research step and keeps raw data/games
  JSON out of git while committing ratings and round reports. Use when
  committing a strategy spec, bot code, a measurement round, or a leaderboard
  update.
```

### `improve-skill-from-failure`

```yaml
description: >-
  Updates a Cursor skill under .cursor/skills/ after a workflow failure or a
  repeated manual step, and records the cause in the skill changelog. Use when a
  skill gave a wrong command, missed a step, failed to load, or loaded for the
  wrong task.
```

## 7. Invocation policy

| Skill | `disable-model-invocation` |
| --- | --- |
| `run-competition-match` | omit (auto) |
| `build-bot-from-spec` | omit (auto) |
| `evaluate-bot-change` | omit (auto) |
| `update-leaderboard` | omit (auto) |
| `write-strategy-spec` | omit (auto) |
| `check-bot-diversity` | omit (auto) |
| `run-measurement-round` | omit (auto) |
| `tune-bot-parameters` | omit (auto) |
| `commit-research-increment` | omit (auto) |
| `improve-skill-from-failure` | `true` (named calls only) |

The loop skills must load from context. The meta skill must not load during
normal work.

## 8. Build order

| Priority | Work |
| --- | --- |
| P0 | `write-strategy-spec`, `build-bot-from-spec` (rename plus improvements), `run-measurement-round` |
| P1 | `tune-bot-parameters`, `check-bot-diversity`, improvements to `run-competition-match` and `evaluate-bot-change` |
| P2 | `commit-research-increment`, `improve-skill-from-failure`, one line in `update-leaderboard` |
| Defer | `learned-bot-readiness` |

## 9. Repo changes that follow

- `AGENTS.md`: update the skills table with the ten skills and the new name.
- `AGENTS.md`: add `strategist` to the subagent role table with
  `write-strategy-spec` and `check-bot-diversity`.
- `docs/index.md`: link this file, the diversity matrix, and the measurements
  directory.
- `docs/research/strategies/diversity-matrix.md`: create the register.
- `docs/research/measurements/`: create the directory on the first round.
- `.gitignore`: decide the `data/games/*.json` rule before the next commit.
