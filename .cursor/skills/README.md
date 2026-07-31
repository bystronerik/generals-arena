# Cursor skills index

Project skills for the Generals Competition research loop. Authority: [`docs/research/strategies/skills-workflow.md`](../../docs/research/strategies/skills-workflow.md).

## Model split

| Model | Owns |
| --- | --- |
| **Think model** | Strategy specs, diversity review, parameter revision choices, skill fixes after failure |
| **Composer** | Bot implementation, match runs, measurement rounds, constant edits, commits when asked |

**Composer must not invent a threshold.** Missing values go back to the think model.

## Loop skills (auto-invoke)

| Step | Skill | Path |
| --- | --- | --- |
| 1. Strategy spec | write-strategy-spec | [write-strategy-spec/](write-strategy-spec/) |
| 1b. Diversity check | check-bot-diversity | [check-bot-diversity/](check-bot-diversity/) |
| 2. Bot from spec | build-bot-from-spec | [build-bot-from-spec/](build-bot-from-spec/) |
| 2b. Verify match | run-competition-match | [run-competition-match/](run-competition-match/) |
| 3. Measurement round | run-measurement-round | [run-measurement-round/](run-measurement-round/) |
| 4. Parameter revision | tune-bot-parameters | [tune-bot-parameters/](tune-bot-parameters/) |
| A/B evaluation | evaluate-bot-change | [evaluate-bot-change/](evaluate-bot-change/) |
| Ratings | update-leaderboard | [update-leaderboard/](update-leaderboard/) |
| Commits | commit-research-increment | [commit-research-increment/](commit-research-increment/) |
| Core unit tests | analyze-and-test-core | [analyze-and-test-core/](analyze-and-test-core/) |
| Classic harness grid | run-classic-grid | [run-classic-grid/](run-classic-grid/) |
| Remote human block | run-remote-block | [run-remote-block/](run-remote-block/) |

## Meta (named invocation only)

| Skill | Path |
| --- | --- |
| improve-skill-from-failure | [improve-skill-from-failure/](improve-skill-from-failure/) |

## Deferred

`learned-bot-readiness` — build when ~10 heuristic bots exist and a remote evaluation path is ready. See [`docs/research/learned-bot-plan.md`](../../docs/research/learned-bot-plan.md).
