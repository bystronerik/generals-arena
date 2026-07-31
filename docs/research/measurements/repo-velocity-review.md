# Repo velocity review — Phase 3 / bot iteration

Source: generals-arena review @ `145db98+` · interactive pick-list in
[repo-velocity-review.canvas.tsx](/Users/erikbystron/.cursor/projects/Users-erikbystron-Work-learning-generals-arena/canvases/repo-velocity-review.canvas.tsx)

Scope: iteration speed toward 95/100 human wins on remote and general bot
iteration. No fixes implemented in this review.

Measured facts: 202 stored games (118 draws at turn 1200; seat a 75 wins, b 9);
0 records with schema v2 telemetry; 14 tests in 0.76 s.

**Assumption:** public-lobby support in `client/` is WIP. Remote queue items
(A2, C2, S1) assume that work lands.

## Suggested pick order

| Order | Items | Reason |
| --- | --- | --- |
| 1 | E1, A3 | Cheap edits; every later measurement carries data |
| 2 | A1 | Largest single Phase 3 iteration win |
| 3 | A2, E4 | Bounded remote blocks with self-reporting |
| 4 | E2 | Competition grid produces decisive answers |
| 5 | R3, R1, R2 | Remove contradictions that mislead agents |
| 6 | S1 | Encode Phase 3 loop as skills after tools exist |
| 7 | T1, T2, A4, A5, A6, A9 | Durability and debuggability |
| 8 | Remaining P2 | Low cost, low urgency |

## P0 findings

| ID | Area | Problem | Est. speedup | Suggested fix |
| --- | --- | --- | --- | --- |
| A1 | Arena | Classic harness has no batch runner or record store | 5–10x per revision | `classic_tournament.py` + `measure_classic.py`; `data/classic_games/` |
| A2 | Remote | No queue timeout or requeue loop | Removes 30–60 min idle/batch | `--queue-timeout-seconds`, requeue with backoff |
| A3 | Bots | Four divergent `main.py`; 5/13 emit telemetry | 13x fewer metric edits | Shared `bots/_common/wire.py` |
| E1 | Eval | `measure_heuristics` drops v2 telemetry | 5–10 min/re-run avoided | Use `run_and_store` |
| E2 | Eval | Draw-dominated, seat-biased grid | ~1 round saved | Both seat orders; land margin at cap |
| S1 | Skills | No Phase 3 classic/remote skills | Plan read → procedure | `run-classic-grid`, `run-remote-block`, remote-operator role |

## P1 findings

| ID | Area | Problem | Est. speedup | Suggested fix |
| --- | --- | --- | --- | --- |
| A4 | Arena | `ArenaGameClient` copies `_on_game_update` | Hours on divergence | Hook in generals-client |
| A5 | Arena | `StrategySession` swallows exceptions | Blind → diagnosable block | Log traceback; store in stats |
| A6 | Bots | `strategy_common.py` triplicated | 1 vs 3 edits | Shared module + named constants |
| A9 | Code | No formal ABC/Protocol for arena stdio bots; contract is duck-typed (`Agent` + `__init__(player_id,H,W)` + `act(obs)->5-tuple`); competition-module JAX Agent unused by `bots/` | Fewer broken scaffolds; clearer Composer contracts | `typing.Protocol` in `arena/bot_api.py` (optional runtime import); update `docs/engine/unified-bot-api.md` |
| R1 | Repo | `.gitignore` vs skills on `data/ratings/` | Fewer commit dead-ends | Pick one rule |
| R2 | Repo | `requirements.txt` mixes torch sandbox | Clean setup seconds | Split requirements files |
| R3 | Repo | AGENTS/human-95 stale remote path | Minutes/session | Point to `client/generals_client` |
| E3 | Eval | `classic_duel` city metric dropped | No stdout hand-read | Generic telemetry parse |
| E4 | Eval | No `remote_report.py` | Manual → one command | Wilson bound + star bands script |
| T1 | Tests | Parsers/store untested | Corrupt round in <1 s | Table-driven parser tests |
| T2 | Tests | Block counter null `opponent_is_bot` | Protect 95/100 claim | Require `opponent_is_bot is False` |
| C1 | Scripts | `remote_lobby_watch` path hack | Env divergence | `arena/remote_env.py` |
| S2 | Skills | Skills point at broken measure script | Fixed by E1 | Fix E1 first |

## P2 summary

| ID | Area | Problem | Fix |
| --- | --- | --- | --- |
| A7 | Arena | JAX import for remote | Lazy import |
| A8 | Arena | Stale `__all__` | Update export list |
| R4 | Repo | Docs index drift | Add rows; fix matrix pointer |
| R5 | Repo | Wrong symbol in remote docs | `from_game_state` |
| R6 | Repo | README/.DS_Store | Update tree; gitignore |
| E5 | Eval | Elo weakness | Fix E2 first |
| E6 | Eval | No `--jobs` | Parallel matches |
| C2 | Scripts | `--public-server` ignored | `--server-url` or remove |
| C3 | Scripts | Bare `python` in `run.sh` | Venv python |
| C4 | Scripts | Dry-run, round name, wait gaps | Small script fixes |
| T3 | Tests | No CI | `pytest.ini` + unified command |

## Already good — do not change

1. Unified bot API (`arena/bot_api.py`) — single observation/action shape
2. Result fidelity tested — disconnect not counted
3. Store-then-rate discipline — idempotent `RatingBook`
4. Secret handling — credentials gitignored
5. Classic harness wraps engine without submodule edits
6. Offline dry run works without credentials
7. Skill files right size with model-split blocks
8. `human-95-plan.md` rigorous (fix stale paths only)
9. Many small docs with index
10. Fast test suite (14 tests, 0.76 s)
