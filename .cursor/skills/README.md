# Cursor skills index

Project skills for the Generals Competition research loop. Authority: [`docs/research/strategies/skills-workflow.md`](../../docs/research/strategies/skills-workflow.md).

## Model split

| Model | Owns |
| --- | --- |
| **Think model** | Parameter revision choices, skill fixes after failure |
| **Composer** | Match runs, measurement rounds, constant edits, commits when asked |

**Composer must not invent a threshold.** Missing values go back to the think model.

## Loop skills (auto-invoke)

| Step | Skill | Model | Path |
| --- | --- | --- | --- |
| 1. Verify match | run-competition-match | Composer | [run-competition-match/](run-competition-match/) |
| 2. Measurement round | run-measurement-round | Composer | [run-measurement-round/](run-measurement-round/) |
| 3. Parameter revision | tune-bot-parameters | Composer | [tune-bot-parameters/](tune-bot-parameters/) |
| A/B evaluation | evaluate-bot-change | Think + Composer | [evaluate-bot-change/](evaluate-bot-change/) |
| Ratings | update-leaderboard | Composer | [update-leaderboard/](update-leaderboard/) |
| Commits | commit-research-increment | Composer | [commit-research-increment/](commit-research-increment/) |
| Classic harness grid | run-classic-grid | Composer | [run-classic-grid/](run-classic-grid/) |
| Remote human block | run-remote-block | Composer | [run-remote-block/](run-remote-block/) |

## Meta (named invocation only)

| Skill | Model | Path |
| --- | --- | --- |
| structure-audit | Think | [structure-audit/](structure-audit/) |
| improve-skill-from-failure | Think | [improve-skill-from-failure/](improve-skill-from-failure/) |

## Deferred

`learned-bot-readiness` — build when ~10 heuristic bots exist and a remote evaluation path is ready.
