# Core-test skill design — `analyze-and-test-core`

Design for one Cursor project skill. This file is a specification. It is not a
`SKILL.md` file. An implementer agent (Composer) writes or edits
`.cursor/skills/analyze-and-test-core/SKILL.md` from this file. A first version
of that skill file already exists; section 15 gives the delta to apply.

Scope: process only. No bot strategy, no thresholds, no game knowledge. Skill
taxonomy authority:
[`skills-workflow.md`](skills-workflow.md). Findings authority:
[`docs/research/measurements/repo-velocity-review.md`](../measurements/repo-velocity-review.md)
(findings **T1**, **T2**, **T3**).

---

## 1. Problem the skill solves

At review time the repo had 14 tests in two files. The tests covered the unified
bot API mapping and remote result fidelity. They did not cover the code that turns
process output into stored numbers, and they did not cover the code that decides
which remote game counts toward the 95/100 human claim.

| Finding | Statement from the velocity review | Cost when it breaks |
| --- | --- | --- |
| T1 | Parsers and the game store are untested | One corrupt measurement round; every later Elo number is wrong |
| T2 | The block counter accepts a null `opponent_is_bot` | The 95/100 human claim counts a game against an unknown opponent |
| T3 | No unified test command in CI | A broken parser reaches the next round |

Every bot iteration reads these numbers. A silent parser defect is the most
expensive class of defect in this repo, because the defect is invisible in the
match output and permanent in `data/games/`.

Shipped state at the time of this design: commit `4c5f0ff` added
`tests/test_telemetry_parsers.py`, `tests/test_store.py`, `tests/test_ratings.py`,
`tests/test_tournament.py`, `tests/test_remote_block.py`,
`tests/test_classic_match.py`, `arena/remote_block.py`, and a first
`.cursor/skills/analyze-and-test-core/SKILL.md`. That commit closes most of T1
and all of T2. The suite holds 54 tests. Section 15 lists the delta between the
shipped skill file and this design.

## 2. Skill name

| Candidate | Verdict |
| --- | --- |
| `analyze-and-test-core` | **Chosen.** The name states both phases: a think-model analysis pass and a Composer write pass. |
| `test-core-logic` | Rejected. The name hides the analysis phase, so the think model skips the risk-surface step. |
| `add-tests` | Rejected. The trigger is too wide. The skill would load for bot behavior tests, which are out of scope. |

Directory: `.cursor/skills/analyze-and-test-core/SKILL.md`.

## 3. Role and triggers

| Field | Value |
| --- | --- |
| Role | `tester` (new role, see section 11) |
| Invocation | Auto. Omit `disable-model-invocation`. |
| Inputs | The planned or completed change, `tests/`, the velocity review, the modules in section 5 |
| Outputs | New or edited files under `tests/`, one fixture file per captured snippet, one report of the risk surface |
| Freedom | Medium for the analysis. Low for the test code. |

The skill must load for these four situations:

1. The user adds a feature from the velocity review pick list (for example A1,
   A3, E1, E4).
2. The user refactors `arena/` or moves logic between `arena/` and `scripts/`.
3. The user asks to verify a planned change before the change lands.
4. The user asks "are we tested enough", "what is untested", or "what breaks
   silently".

The skill must not load for a bot behavior question. Bot strength comes from
`run-measurement-round` and `evaluate-bot-change`, not from pytest.

## 4. Model split

The skill body must carry this exact block:

```markdown
## Model split

- Think model: lists the risk surface of the change, names the missing core
  tests, and ranks them by cost when the code breaks silently
- Composer: writes the minimal table-driven tests, captures the fixtures, runs
  the suite, and reports the pass count and the runtime
```

Rules that follow from the split:

- The think model chooses **which** core behavior needs a test and **why**. The
  think model does not write Python.
- Composer writes the test bodies. Composer does not add a target that the think
  model did not name.
- **Composer must not invent an expected value.** When the correct output of a
  function is unclear, Composer stops and asks the think model. A test that
  copies current behavior without a stated reason is not a test; it is a lock.
- Composer must not change production code to make a test pass. See section 9.

## 5. Core surface — the only allowed targets

"Core" means code that converts, stores, or classifies data, and where a wrong
result is silent. The skill must give this table and must refuse a target that
is not in it.

| ID | Target | Module | Symbols | Why it is core |
| --- | --- | --- | --- | --- |
| C1 | Match result mapping | `arena/match_loop.py` | `winner_seat`, `MatchLoopResult` fields | A wrong winner or turn count enters `data/games/` and Elo forever |
| C2 | Castle metric counting | `arena/competition_match.py` | castle tally in `run_competition_match` (`castles_built_a/b`) | The only castle signal that reaches a round report |
| C3 | Telemetry parse and merge | `arena/run_match.py` | `parse_bot_telemetry`, `apply_telemetry_to_record` | Schema v2 fields; generic extras (e.g. `first_city_capture_turn`) |
| C4 | Game record store | `arena/store.py` | `GameRecord.from_dict`, `save_game`, `load_game` | A malformed record breaks a whole round load |
| C5 | Rating idempotence | `arena/ratings.py` | `RatingBook.apply_game`, `apply_games`, `to_state`, `from_state`, `rebuild_from_games` | A double-counted game moves Elo with no visible error |
| C6 | Unified bot API mapping | `arena/bot_api.py` | `from_game_state`, `from_competition_remote_obs`, `to_client_move`, `translate_action_for_remote`, `StrategySession.act` fault path | The single observation and action contract for every bot |
| C7 | Fidelity session classification | `arena/remote_client.py` | `result_from_reason`, `opponent_is_bot`, `DECIDED_REASONS`, `FidelityRemoteSession._finish_with_reason` | Decides `counts_toward_block`, which defines the 95/100 claim |
| C8 | Classic match result contract | `arena/classic_match.py` | `run_classic_match` return contract, and the record writer that arrives with A1 | Classic results use a different ruleset and must never reach `data/games/` |
| C9 | Remote human-count filter | `arena/remote_block.py` | `counts_as_human_block_game`, `count_human_block_games` | The filter defines a human block; T2 lived here |
| C10 | Grid construction helpers | `arena/tournament.py`, `scripts/measure_heuristics.py` | `parse_seeds`, `bot_pairs`, `build_grid` | A wrong seed set or pair set silently changes what a round measures |

`scripts/remote_lobby_watch.py:_counted_human_games_since` is a thin delegate to
C9 and needs no separate test. Test the module, not the delegate.

Existing coverage, so that the skill does not duplicate work:

| File | Covers | Gap the skill may close |
| --- | --- | --- |
| `tests/test_bot_api.py` | C6 mapping for remote obs, game state, client move, pass | `StrategySession.act` fault counting; build action on the remote path |
| `tests/test_remote_client.py` | C7 reason mapping, `opponent_is_bot`, win, loss, disconnect records | `stall` and `receive_error` records; a null `opponent_is_bot` in the written record |
| `tests/test_telemetry_parsers.py` | C1, C2, C3 | Two castle lines in one capture; telemetry with one player only |
| `tests/test_store.py` | C4 | — |
| `tests/test_ratings.py` | C5 | `to_state` and `from_state` round trip; `rebuild_from_games` |
| `tests/test_tournament.py` | C10 for `parse_seeds` and `bot_pairs` | `build_grid` seat coverage after E2 lands |
| `tests/test_remote_block.py` | C9 including the null case | — |
| `tests/test_classic_match.py` | C8 smoke, marked `slow` | The classic record writer that arrives with A1 |

Priority order when the change does not point at one target: **C1, C9, C4, C5,
C3, C2, C10, C7 gap, C6 gap, C8**.

## 6. Out of scope

The skill must state these refusals in the body:

- No combinatorial edge-case explosion. Do not enumerate grid sizes, seeds,
  turn counts, or tile-type products. Two to four cases per behavior.
- No property tests, fuzz tests, or generated inputs.
- No test that runs a match grid or a full-length match. One smoke match may stay
  in the suite behind `@pytest.mark.slow` with a tiny board and a low truncation,
  as `tests/test_classic_match.py` does. Every other match run belongs to
  `run-competition-match`.
- No network call, no live generals.io, no sleep, no retry loop.
- No bot strategy assertion. "Bot X beats bot Y" is a measurement, not a test.
- No snapshot test of a whole Markdown report or a whole leaderboard file.
- No test that reads or writes the real `data/games/`, `data/ratings/`, or
  `data/remote_games/` directories.
- No new test framework, plugin, or runner. `pytest.ini` already sets
  `testpaths = tests` and `pythonpath = .`.

## 7. Test style rules

### 7.1 Table-driven, 2 to 4 cases

Every behavior gets one `pytest.mark.parametrize` table with two to four rows.
Each row carries a short label. The template the skill must give:

```python
@pytest.mark.parametrize(
    "label, winner_player_id, truncated, expected",
    [
        ("player 0 capture", 0, False, "a"),
        ("player 1 capture", 1, False, "b"),
        ("truncation draw", -1, True, "draw"),
    ],
)
def test_winner_seat(label, winner_player_id, truncated, expected):
    assert winner_seat(winner_player_id, truncated=truncated) == expected
```

Case selection rule, in this order: one normal case, one boundary case, one
rejection case. Add a fourth row only when a distinct failure mode exists.
Never add a row that repeats the shape of an existing row.

### 7.2 Fixtures from real captured output

Prefer a real captured snippet over a mock when the capture is cheap.

| Data | Source | Cost |
| --- | --- | --- |
| Matchup stdout lines | One `--mode competition` match already logged, or one new gate run | Cheap. Copy 3 to 5 lines. |
| Telemetry lines | Same match stderr | Cheap. Copy the last line per player. |
| Game record JSON | One file from `data/games/` | Cheap. Copy and trim. |
| Remote game record | One file from `data/remote_games/` | Cheap. Redact the opponent username. |
| Remote socket traffic | Live server | Expensive. Use `MagicMock`, as `tests/test_remote_client.py` does today. |

Fixture placement: `tests/fixtures/<name>.txt` or `.json`, one file per snippet.
Keep each fixture under 20 lines. Trim the snippet to the lines the parser
reads. Write a one-line comment in the test that names the source match or
record. Redact every username and every credential.

Mocks stay for one purpose only: the remote client boundary
(`FidelityRemoteSession.client`) and a strategy that must raise.

### 7.3 Isolation

- Pass the directory argument where the function accepts one: `games_dir=tmp_path`,
  `ratings_dir=tmp_path`, `log_dir=tmp_path`.
- Use `monkeypatch.setattr` only when no such argument exists, for example a
  module-level directory constant in a script.
- Never write outside `tmp_path`.

### 7.4 Budget

| Limit | Value |
| --- | --- |
| New test functions per invocation | 8 |
| Cases per table | 4 |
| Lines per test function | 25 |
| Whole suite runtime after the change | under 3 seconds with a warm cache |
| New dependencies | 0 |

When the analysis names more than eight tests, Composer writes the top eight by
the section 5 priority order and reports the rest as a remainder list.

Runtime note: 54 tests run in about 2 seconds with a warm cache. `arena/ratings.py`
imports `elote`, `elote` imports Matplotlib, and Matplotlib builds a font cache on
a first run in a fresh environment. That first run costs about 11 seconds. The
skill must report the warm number and must not treat the cold run as a
regression. Any test that needs more than 1 second belongs behind
`@pytest.mark.slow`, and the quick loop is `python -m pytest -q -m "not slow"`.
Declare the marker in `pytest.ini` so that pytest raises no unknown-mark
warning.

## 8. Procedure the skill must give

1. Read the change. When the change is planned and not written, read the plan
   text and the target modules.
2. Map the change onto the section 5 table. Name every touched target ID.
3. Read `tests/` and mark each target as covered, partly covered, or untested.
4. List the risk surface: for each untested behavior, one line with the wrong
   output that would pass unnoticed today.
5. Rank the list by cost and cut it to the section 7.4 budget.
6. Write the tests. One table per behavior. Capture the fixtures.
7. Run `python -m pytest -q`.
8. Report: new test count, total test count, runtime, and every target that
   stays untested.

## 9. When a test finds a defect

A new test can prove that production code is wrong. T2 was that case: the old
`_counted_human_games_since` skipped a record only when `opponent_is_bot is True`,
so a record with a null `opponent_is_bot` counted toward a human block. Commit
`4c5f0ff` fixed the defect in `arena/remote_block.py`, which now requires
`opponent_is_bot is False`. No `xfail` test exists today. The rule below applies
to the next defect of this class.

Rule for the skill:

- Write the test that states the correct behavior, for example
  "a record with a null `opponent_is_bot` does not count".
- Mark the test `@pytest.mark.xfail(strict=True, reason="T2: ...")` with a link
  to the velocity review.
- Report the defect to the user in plain text. Name the module, the symbol, and
  the wrong output.
- Do not fix the production code inside this skill. The fix belongs to the
  change author, and the fix removes the `xfail` mark in the same commit.

A missing seam is a report, not a refactor. When a function cannot be tested
without a monkeypatch of a module constant, the skill uses the monkeypatch and
reports the seam. The skill does not restructure `arena/` to make testing
easier.

## 10. Exact frontmatter description line

Use this line without change.

```yaml
description: >-
  Finds the untested core risk surface of the arena and adds minimal
  table-driven pytest cases for output parsers, the game record store, rating
  idempotence, bot_api mapping, remote fidelity classification, classic match
  results, and the remote human-count filter. Use when adding a feature from the
  velocity review, refactoring arena/, verifying a planned change before it
  lands, or asking whether the core logic is tested enough.
```

Full frontmatter shape:

```yaml
---
name: analyze-and-test-core
description: >-
  <the line above>
---
```

## 11. AGENTS.md role addition — `tester`

Add one row to the subagent role table in `AGENTS.md`:

| Role | Owns | Skills | Done when |
| --- | --- | --- | --- |
| tester | Core coverage under `tests/`; fixtures under `tests/fixtures/` | `analyze-and-test-core` | New tests pass, the suite stays under 3 s, and every untested core target is named |

Add one role subsection after `docs-keeper`:

```markdown
### tester

1. Start from the core surface table in
   [`docs/research/strategies/test-core-skill.md`](docs/research/strategies/test-core-skill.md).
2. Test core logic only: parsers, store, ratings idempotence, bot_api mapping,
   fidelity classification, classic match results, remote human-count filter.
3. Report a proven defect; do not fix production code inside the test step.
```

Also add one row to the Cursor skills table in `AGENTS.md`:

| Skill | Model | Path |
| --- | --- | --- |
| analyze-and-test-core | Think + Composer | `.cursor/skills/analyze-and-test-core/` |

## 12. Repo changes that follow

- `.cursor/skills/analyze-and-test-core/SKILL.md`: edit the shipped file with the
  section 15 delta.
- `.cursor/skills/README.md`: one row for the skill; the think and Composer
  split table gains "core coverage analysis" for the think model.
- `AGENTS.md`: the `tester` role row, the role subsection, the skills table row.
- `docs/index.md`: one link line to this design file.
- `pytest.ini`: one `markers` line for `slow`.
- `tests/fixtures/`: create on the first captured snippet.
- T3 stays open. A CI workflow is a separate change and is not part of this
  skill.

## 13. Done-when checklist

The skill is complete when every line is true.

1. `.cursor/skills/analyze-and-test-core/SKILL.md` exists with the exact
   `name` and `description` from section 10.
2. The body carries the model split block from section 4 without change.
3. The body carries the core surface table from section 5 with all ten target
   IDs and the priority order.
4. The body carries the out-of-scope list from section 6, including the "no
   combinatorial edge-case explosion" rule.
5. The body carries the table-driven template, the 2-to-4-case rule, the fixture
   preference over mocks, and the budget table.
6. The body carries the eight procedure steps and the command
   `python -m pytest -q`.
7. The body carries the defect rule from section 9, including the `xfail`
   instruction and the "no production fix in this skill" rule.
8. The body links `tests/`, the velocity review, T1, T2, and this design file.
9. The body stays under 150 lines and holds no strategy content and no threshold.
10. The body ends with a `## Changelog` section that carries the seed line from
    section 14.
11. `AGENTS.md` and `.cursor/skills/README.md` name the skill and the `tester`
    role.
12. `python -m pytest -q` passes on the current tree and stays under 3 seconds
    with a warm cache.

## 14. Changelog seed

The new `SKILL.md` starts with this single line under `## Changelog`:

```markdown
- 2026-07-31 — Initial skill from test-core-skill design; core surface limited to parsers, store, ratings, bot_api, fidelity, classic results, and the remote human-count filter (cause: velocity review T1, T2)
```

## 15. Delta against the shipped skill file

`.cursor/skills/analyze-and-test-core/SKILL.md` exists at commit `4c5f0ff`. The
shipped file is correct on the coverage map, the environment, the command, and the
"no edge-case explosion" refusal. Apply these edits; do not rewrite the file.

| # | Edit | Reason |
| --- | --- | --- |
| D1 | Replace the `description` line with the exact line in section 10 | The shipped line names modules but misses the four trigger situations in section 3, so the skill does not load on "are we tested enough" or on an `arena/` refactor |
| D2 | Replace the model split block with the block in section 4 | The shipped block says "picks which T1/T2 unit changed", which expires when T1 and T2 close. The risk-surface wording does not expire. |
| D3 | Replace "Composer must not invent a threshold" with "Composer must not invent an expected value" | A test has no thresholds. The wrong noun sends Composer to the strategy docs. |
| D4 | Add the target IDs C1 to C10 and the priority order to the coverage map table | An unranked map gives no answer when a change touches four modules at once |
| D5 | Add the out-of-scope list from section 6 | The shipped file refuses only edge-case explosions and subprocess grids. It permits a network test, a real `data/` write, and a bot behavior assertion. |
| D6 | Add the fixture policy from section 7.2, including `tests/fixtures/`, the 20-line limit, the username redaction rule, and mocks only at the remote client boundary | The shipped file says "paste a real stdout snippet" with no placement, no size limit, and no redaction rule |
| D7 | Add the budget table from section 7.4 and the warm-cache runtime note | "under a few seconds" in the shipped description is not a limit, and a cold Matplotlib font cache looks like a regression |
| D8 | Add the eight procedure steps from section 8 | The shipped file has a coverage map and no procedure, so the analysis phase can be skipped |
| D9 | Add the defect rule from section 9 | Without the rule, Composer fixes production code inside the test step and hides the defect |
| D10 | Add a `## Changelog` section with the section 14 seed line | Every other skill in this repo carries one; `improve-skill-from-failure` needs the section |
| D11 | Add links to `tests/`, this design file, and the velocity review | The shipped file names T1 and T2 without a link |
| D12 | Keep the `Optional slow smoke` section; add the `pytest.ini` marker line | The marker is undeclared today |

Constraint: the edited body stays under 150 lines. Move any detail that does not
fit into this design file and link to it.
