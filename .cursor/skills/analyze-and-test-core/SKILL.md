---
name: analyze-and-test-core
description: >-
  Runs focused pytest on arena parsers, store round-trip, ratings idempotence,
  tournament helpers, and remote block counter. Use when changing run_match.py,
  store.py, ratings.py, tournament.py, remote_block.py, or before velocity
  review T1/T2 fixes to verify core behavior in under a few seconds.
---

# Analyze and test core

## Model split

- Think model: picks which T1/T2 unit changed and what assertion must hold
- Composer: adds or updates table-driven tests, runs `pytest tests/ -q`

**Composer must not invent a threshold.** Test fixtures use captured stdout snippets and minimal dict payloads only.

## Environment

From repo root with `.venv` active:

```bash
pip install pytest   # if missing
pytest tests/ -q
```

Config: root `pytest.ini` sets `testpaths = tests` and `pythonpath = .`.

## Coverage map (velocity review T1/T2)

| Module | Functions | Test file |
| --- | --- | --- |
| `arena/run_match.py` | `parse_matchup_output`, `parse_castles_built`, `parse_bot_telemetry`, `apply_telemetry_to_record` | `tests/test_run_match_parsers.py` |
| `arena/store.py` | `GameRecord.from_dict` / `to_dict`, v1→v2 | `tests/test_store.py` |
| `arena/ratings.py` | `RatingBook.apply_game` idempotence | `tests/test_ratings.py` |
| `arena/tournament.py` | `parse_seeds`, `bot_pairs` | `tests/test_tournament.py` |
| `arena/remote_block.py` | `counts_as_human_block_game`, `count_human_block_games` | `tests/test_remote_block.py` |
| `arena/bot_api.py` | unified obs/action | `tests/test_bot_api.py` (extend only for contract gaps) |
| `arena/remote_client.py` | result fidelity, `opponent_is_bot` | `tests/test_remote_client.py` |

## When to add tests

Add a table row when you change:

1. A regex or parser in `run_match.py` — paste a real stdout snippet as the fixture string.
2. `GameRecord` optional fields or defaults — extend v1/v2 dict fixtures in `test_store.py`.
3. Rating apply rules — assert second `apply_game` returns `False` and ratings unchanged.
4. Remote block counting — `opponent_is_bot` must be **exactly** `False` to count; `None` and `True` are excluded.

Do **not** add edge-case explosions or subprocess match grids here; use `run-competition-match` for integration.

## Optional slow smoke

`tests/test_classic_match.py` runs smoke vs smoke with 8×8 board and truncation 5. Marked `@pytest.mark.slow`. Skip in quick loops:

```bash
pytest tests/ -q -m "not slow"
```

## Done when

- `pytest tests/ -q` passes in `.venv`
- New parser or store behavior has at least one parametrized case
- Remote block filter tests cover human, bot, and null opponent
