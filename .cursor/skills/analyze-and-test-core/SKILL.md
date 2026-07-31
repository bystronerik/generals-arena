---
name: analyze-and-test-core
description: >-
  Finds the untested core risk surface of the arena and adds minimal
  table-driven pytest cases for output parsers, the game record store, rating
  idempotence, bot_api mapping, remote fidelity classification, classic match
  results, and the remote human-count filter. Use when adding a feature from the
  velocity review, refactoring arena/, verifying a planned change before it
  lands, or asking whether the core logic is tested enough.
---

# Analyze and test core

Design: [`docs/research/strategies/test-core-skill.md`](../../docs/research/strategies/test-core-skill.md).
Findings: [`docs/research/measurements/repo-velocity-review.md`](../../docs/research/measurements/repo-velocity-review.md) (**T1**, **T2**).
Tests: [`tests/`](../../tests/).

## Model split

- Think model: lists the risk surface of the change, names the missing core
  tests, and ranks them by cost when the code breaks silently
- Composer: writes the minimal table-driven tests, captures the fixtures, runs
  the suite, and reports the pass count and the runtime

Think model chooses **which** core behavior needs a test and **why**; it does not write Python.
Composer writes test bodies only for targets the think model named.
**Composer must not invent an expected value.** When the correct output is unclear, Composer stops and asks the think model.
Composer must not change production code to make a test pass.

## Core surface (only allowed targets)

| ID | Target | Module | Symbols |
| --- | --- | --- | --- |
| C1 | Match result mapping | `arena/match_loop.py` | `winner_seat`, `MatchLoopResult` fields |
| C2 | Castle metric counting | `arena/competition_match.py` | castle tally in `run_competition_match` (`castles_built_a/b`) |
| C3 | Telemetry parse and merge | `arena/run_match.py` | `parse_bot_telemetry`, `apply_telemetry_to_record` |
| C4 | Game record store | `arena/store.py` | `GameRecord.from_dict`, `save_game`, `load_game` |
| C5 | Rating idempotence | `arena/ratings.py` | `RatingBook.apply_game`, `apply_games`, `to_state`, `from_state`, `rebuild_from_games` |
| C6 | Unified bot API mapping | `arena/bot_api.py` | `from_game_state`, `from_competition_remote_obs`, `to_client_move`, `translate_action_for_remote`, `StrategySession.act` fault path |
| C7 | Fidelity session classification | `arena/remote_client.py` | `result_from_reason`, `opponent_is_bot`, `DECIDED_REASONS`, `FidelityRemoteSession._finish_with_reason` |
| C8 | Classic match result contract | `arena/classic_match.py` | `run_classic_match` return contract, classic record writer |
| C9 | Remote human-count filter | `arena/remote_block.py` | `counts_as_human_block_game`, `count_human_block_games` |
| C10 | Grid construction helpers | `arena/tournament.py`, `scripts/measure_heuristics.py` | `parse_seeds`, `bot_pairs`, `build_grid` |

Priority when the change does not point at one target: **C1, C9, C4, C5, C3, C2, C10, C7 gap, C6 gap, C8**.

Existing coverage — extend only gaps; see design file section 5 for the full map.

## Out of scope

- No combinatorial edge-case explosion. Two to four cases per behavior.
- No property tests, fuzz tests, or generated inputs.
- No test that runs a match grid or a full-length match. One smoke match may stay behind `@pytest.mark.slow` (see below). Every other match run belongs to `run-competition-match`.
- No network call, no live generals.io, no sleep, no retry loop.
- No bot strategy assertion. "Bot X beats bot Y" is a measurement, not a test.
- No snapshot test of a whole Markdown report or a whole leaderboard file.
- No test that reads or writes the real `data/games/`, `data/ratings/`, or `data/remote_games/` directories.
- No new test framework, plugin, or runner. `pytest.ini` already sets `testpaths = tests` and `pythonpath = .`.

## Test style

One `pytest.mark.parametrize` table per behavior, two to four rows with short labels. Order: one normal, one boundary, one rejection; add a fourth only for a distinct failure mode.

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

**Fixtures:** prefer real captured snippets over mocks. Place under `tests/fixtures/<name>.txt` or `.json`, one file per snippet, under 20 lines. Trim to lines the parser reads. One-line comment in the test names the source match or record. Redact every username and credential. Mocks only at the remote client boundary (`FidelityRemoteSession.client`) and for strategies that must raise.

**Isolation:** pass `games_dir=tmp_path`, `ratings_dir=tmp_path`, or `log_dir=tmp_path` where accepted; use `monkeypatch.setattr` only when no such argument exists. Never write outside `tmp_path`.

| Limit | Value |
| --- | --- |
| New test functions per invocation | 8 |
| Cases per table | 4 |
| Lines per test function | 25 |
| Whole suite runtime after the change | under 3 seconds with a warm cache |
| New dependencies | 0 |

When the analysis names more than eight tests, write the top eight by priority and report the rest as a remainder list. Report the warm runtime; a cold Matplotlib font cache (~11 s on first run) is not a regression. Tests over 1 s belong behind `@pytest.mark.slow`; quick loop: `python -m pytest -q -m "not slow"`.

## Procedure

1. Read the change. When planned and not written, read the plan text and target modules.
2. Map the change onto the core surface table. Name every touched target ID.
3. Read `tests/` and mark each target as covered, partly covered, or untested.
4. List the risk surface: for each untested behavior, one line with the wrong output that would pass unnoticed today.
5. Rank the list by cost and cut it to the budget.
6. Write the tests. One table per behavior. Capture the fixtures.
7. Run `python -m pytest -q`.
8. Report: new test count, total test count, runtime, and every target that stays untested.

## When a test finds a defect

- Write the test that states the correct behavior.
- Mark `@pytest.mark.xfail(strict=True, reason="T2: ...")` with a link to the velocity review.
- Report the defect in plain text. Name the module, symbol, and wrong output.
- Do not fix production code inside this skill. The fix belongs to the change author; the fix removes the `xfail` mark in the same commit.
- A missing seam is a report, not a refactor. Use `monkeypatch` and report the seam; do not restructure `arena/`.

## Environment

From repo root with `.venv` active:

```bash
pip install pytest   # if missing
python -m pytest -q
```

Config: root `pytest.ini` sets `testpaths = tests`, `pythonpath = .`, and declares `slow` under `markers`.

## Optional slow smoke

`tests/test_classic_match.py` runs smoke vs smoke with 8×8 board and truncation 5. Marked `@pytest.mark.slow`. Skip in quick loops:

```bash
python -m pytest -q -m "not slow"
```

## Done when

- `python -m pytest -q` passes in `.venv` and stays under 3 s with a warm cache
- New core behavior has at least one parametrized case (2–4 rows)
- Every untested core target is named in the report
- Remote block filter tests cover human, bot, and null opponent

## Changelog

- 2026-07-31 — Initial skill from test-core-skill design; core surface limited to parsers, store, ratings, bot_api, fidelity, classic results, and the remote human-count filter (cause: velocity review T1, T2)
