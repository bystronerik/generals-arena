---
name: improve-skill-from-failure
description: >-
  Updates a Cursor skill under .cursor/skills/ after a workflow failure or a
  repeated manual step, and records the cause in the skill changelog. Use when a
  skill gave a wrong command, missed a step, failed to load, or loaded for the
  wrong task.
disable-model-invocation: true
---

# Improve skill from failure

## Model split

- Think model: finds the cause, edits the skill
- Composer: reproduces the failure when needed

**Composer must not invent a threshold.** When a value is absent from the specification, Composer stops and asks the think model.

## Inputs

- Failure description
- The skill that was active
- Game id or round name

## Output

- Edited `.cursor/skills/<name>/SKILL.md`
- One line in that file's `## Changelog` section

## When to update

| Trigger | Action |
| --- | --- |
| Same failure twice under the same skill | Mandatory update. Add the missing step or check. |
| Skill sent agent to wrong path, flag, or stale command | Immediate update. Correct the command. |
| Agent needed a step the skill did not name, and the step worked | Add the step after the third use. |
| Skill loaded for the wrong task | Update the `description` line only. |
| Skill did not load for the right task | Add missing trigger terms to `description`. |
| Step unused in two measurement rounds | Delete the step. |
| Repo path, script name, or schema changed | Update every skill that names the path in the same commit. |

## How to update

- Change the smallest amount of text that removes the failure.
- Keep the body under **150 lines**. Move detail into `docs/` and link to it.
- Changelog format: `- <date> — <change> (cause: <game id, round name, or failure>)`.
- Do not add strategy content. Link to `docs/` instead.
- Do not add a second way to do the same thing. Replace the old way.

## What never enters a skill

- Bot strategy, thresholds, or scoring rules
- Results from one match
- Time-sensitive text such as "for now" or a date-bound instruction
- A second command that does the same work as the first command

Authority: [`docs/research/strategies/skills-workflow.md`](../../../docs/research/strategies/skills-workflow.md) section 4.

## Changelog

- 2026-07-31 — Initial skill from skills-workflow taxonomy (cause: skills-workflow build)
