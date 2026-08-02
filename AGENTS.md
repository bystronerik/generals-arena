# Agent workflow

Process rules for agents that work in this repo. Put game and bot knowledge in `docs/`. Do not put strategy logic here.

## Hard constraints

- Competition gameplay differs from classic generals.io. Prefer `RULES.md` and `GeneralsEnv(mode="competition")` over classic lore.
- `README.md`: structure, install, how to run. No deep strategy.
- `AGENTS.md` (this file): phases, file placement, verification. No game/bot strategy.
- All game and bot knowledge lives under `docs/` as many small files.
- Do not edit `competition-module` internals unless a bug blocks work. Wrap and document instead.
- Remote play uses `client/generals_client` via `arena/remote/bridge.py` and
  `scripts/remote_play.py`. That path targets **live generals.io** (classic
  rules), not the competition sandbox. Keep that distinction explicit.
- Scraped `generals.bot` leaderboard replays are **observational data, not arena
  matches**. They never enter `data/games/` or a rating fit. See
  [Leaderboard replays](#leaderboard-replays-scraped) below.

## File placement

| Kind | Where |
| --- | --- |
| Competition rules (processed) | `RULES.md` |
| Agent process only | `AGENTS.md` |
| Repo intro / install / run | `README.md` |
| Game, protocol, engine, research notes | `docs/**` (small topic files) |
| Stdio bots | `bots/<name>/` with `agent.py`, `main.py`, `run.sh` |
| Bot unit tests (optional) | `bots/<name>/tests/` — outside content hash |
| Match runner, ratings, store | `arena/` |
| Match JSON / rating snapshots (derived, gitignored) | `data/games/`, `data/ratings/` |
| Per-turn trajectories (derived, gitignored, opt-in `--record`) | `data/trajectories/` |
| Bot version registry (committed) | `data/bot_versions/` |
| Scraped leaderboard replays (derived, gitignored) | `competition-replays/<player>/{win,lose,draw}/` |
| Cursor skills | `.cursor/skills/` |

Do not put strategy content in `AGENTS.md` or skill files beyond process pointers that link into `docs/`.

## Verification gate

Every bot or arena change must prove a match finishes under competition mode:

```bash
python competition-module/competition/matchup.py \
  <bot_a/run.sh> \
  <bot_b/run.sh> \
  --mode competition --seed 0
```

Requirements:

- Use `--mode competition`.
- The match must reach a normal end (win, loss, or draw / truncation).
- Do not treat a classic or non-competition preset run as sufficient.

Also store the game under `data/games/` before you refit ratings. There is no
incremental rating update: every write refits the whole pool.

## Remote success criterion (classic rules)

The remote goal is one heuristic bot that wins **at least 95 of 100 logged games
against human opponents** on live generals.io through
`client/generals_client` (`arena/remote/bridge.py`, `scripts/remote_play.py`).

Process rules for that criterion:

- The local competition gate above still applies first. Remote play never
  replaces it.
- A block is 100 human games played by **one immutable bot commit**. Changing
  the bot ends the block.
- Games are logged under `data/remote_games/` (gitignored). Publish the block
  report under `docs/research/measurements/remote-block<N>.md`.
- Remote games never enter `data/games/` or `data/ratings/` — different
  ruleset.
- Plan, gates, and task order:
  [`docs/research/strategies/human-95-plan.md`](docs/research/strategies/human-95-plan.md).

## Leaderboard replays (scraped)

Finished competition-rules games from the `generals.bot` leaderboard, fetched by
the `competition-scraper` submodule:

```bash
python scripts/scrape_replays.py                    # default player: erik.bystron
python scripts/scrape_replays.py erik.bystron prady --concurrency 8
```

Process rules:

- Output goes to `competition-replays/<player>/{win,lose,draw}/` — gitignored
  except `.gitkeep`. Roughly 0.6 MB per replay; never commit them.
- Runs are incremental (existing `<id>.json` is skipped) and the list endpoint
  has no pagination, so history only accumulates by re-running periodically.
- These games are **observational data**. They are not produced by our runner,
  carry no bot version, and must never be written into `data/games/`,
  `data/ratings/`, or `data/remote_games/`.
- Needs `httpx`; `git submodule update --init competition-scraper` first.
- Format, field meanings, and the sampling caveats that affect analysis:
  [`docs/engine/leaderboard-replays.md`](docs/engine/leaderboard-replays.md).

## Subagent roles (workflow only)

Roles are process pointers. Put game and bot knowledge in `docs/`. Do not put strategy in this file or in skill bodies beyond links into `docs/`.

| Role | Owns | Skills | Done when |
| --- | --- | --- | --- |
| explorer | Read `docs/` + `RULES.md`; map engine APIs | `run-competition-match` to verify observations | Notes cite a finished `--mode competition` match; no unmeasured strategy claims |
| bot-author | Add or change `bots/<name>/`; keep stdio protocol intact | `build-bot-from-spec`, then `run-competition-match` | New/changed `run.sh` finishes a competition match |
| strategist | Write strategy specs; enforce bot diversity | `write-strategy-spec`, `check-bot-diversity` | Spec in `docs/research/strategies/` with diversity verdict |
| evaluator | Fixed grid; before/after winrate; **pairwise rating contrast with an interval** | `run-measurement-round`, `evaluate-bot-change`, `update-leaderboard` | Games in `data/games/`; verdict quoted per [`docs/arena/decision-rule.md`](docs/arena/decision-rule.md), never as a rank |
| remote-operator | Live classic blocks; human gate ladder | `run-remote-block`, `run-classic-grid` | Block report under `docs/research/measurements/`; logs in `data/remote_games/` |
| auditor | Structure and duplication review | `structure-audit` | Report names concrete files and lines; no edits without a follow-up ask |
| docs-keeper | Keep `docs/` small and accurate; sync with code + `RULES.md` | none required | Topic files stay single-purpose; no strategy moved into `AGENTS.md` |
| tester | Core coverage under `tests/`; fixtures under `tests/fixtures/` | `analyze-and-test-core` | New tests pass, the suite stays under 7 s, and every untested core target is named |

### explorer

1. Start at [`docs/index.md`](docs/index.md) and [`RULES.md`](RULES.md).
2. Map APIs from `competition-module` docs and DeepWiki; do not edit submodule internals unless a bug blocks work.
3. Confirm claims with `run-competition-match` (or raw `matchup.py --mode competition`).

### bot-author

1. Scaffold with skill `build-bot-from-spec` from a spec under `docs/research/strategies/`.
2. Keep `main.py` / wire protocol stable; change decision code under `bots/<name>/`.
3. Verify with `run-competition-match` before calling the bot ready.

### strategist

1. Start from [`RULES.md`](RULES.md) and [`docs/research/strategies/skills-workflow.md`](docs/research/strategies/skills-workflow.md).
2. Write specs with `write-strategy-spec`; run `check-bot-diversity` before implementation.
3. After a measurement round, choose parameter revisions for `tune-bot-parameters`. Do not write bot code.

### evaluator

1. Follow [`docs/research/experiment-protocol.md`](docs/research/experiment-protocol.md).
2. Use `run-measurement-round` for batch grids or `evaluate-bot-change` for A/B pairs.
3. Store games via `arena/matches/run_match.py` / `scripts/measure_heuristics.py`, then `update-leaderboard`.
4. Decide from the pairwise contrast in [`docs/arena/decision-rule.md`](docs/arena/decision-rule.md) — never from a leaderboard rank.

### remote-operator

1. Read [`docs/research/strategies/human-95-plan.md`](docs/research/strategies/human-95-plan.md) and [`docs/engine/remote-play-setup.md`](docs/engine/remote-play-setup.md).
2. Tune on the classic harness with `run-classic-grid` before spending live queue time.
3. Run human blocks with `run-remote-block`; publish `docs/research/measurements/remote-block<N>.md`.
4. Never feed remote games into `data/games/` or `data/ratings/`.

### docs-keeper

1. Prefer many small files under `docs/`.
2. Sync schema and protocol pages when arena or bot layout changes.
3. Reject strategy content in `AGENTS.md` and skill files; move it to `docs/`.

### tester

1. Start from the core surface table in
   [`docs/research/strategies/test-core-skill.md`](docs/research/strategies/test-core-skill.md).
2. Test core logic only: parsers, store, **rating order-invariance and the
   count-table digest**, the version registry, bot_api mapping, fidelity
   classification, classic match results, remote human-count filter.
3. Report a proven defect; do not fix production code inside the test step.
4. The suite budget is **7 s** with a warm cache. It was 3 s until the ratings
   refactor added three tests that cannot be made cheap without giving up what
   they check: the registry round trip needs a real `git` sandbox, the solver
   oracle needs a second, independently written implementation to disagree
   with, and the determinism check needs a subprocess with a different thread
   count. Treat 7 s as a ceiling to defend, not a number to ratchet — a new
   test that costs a second needs the same kind of argument.

## Cursor skills

Taxonomy: [`docs/research/strategies/skills-workflow.md`](docs/research/strategies/skills-workflow.md). Index: [`.cursor/skills/README.md`](.cursor/skills/README.md).

| Skill | Model | Path |
| --- | --- | --- |
| write-strategy-spec | Think | [`.cursor/skills/write-strategy-spec/`](.cursor/skills/write-strategy-spec/) |
| check-bot-diversity | Think | [`.cursor/skills/check-bot-diversity/`](.cursor/skills/check-bot-diversity/) |
| build-bot-from-spec | Composer | [`.cursor/skills/build-bot-from-spec/`](.cursor/skills/build-bot-from-spec/) |
| run-competition-match | Composer | [`.cursor/skills/run-competition-match/`](.cursor/skills/run-competition-match/) |
| run-measurement-round | Composer | [`.cursor/skills/run-measurement-round/`](.cursor/skills/run-measurement-round/) |
| tune-bot-parameters | Composer | [`.cursor/skills/tune-bot-parameters/`](.cursor/skills/tune-bot-parameters/) |
| evaluate-bot-change | Think + Composer | [`.cursor/skills/evaluate-bot-change/`](.cursor/skills/evaluate-bot-change/) |
| update-leaderboard | Composer | [`.cursor/skills/update-leaderboard/`](.cursor/skills/update-leaderboard/) |
| commit-research-increment | Composer | [`.cursor/skills/commit-research-increment/`](.cursor/skills/commit-research-increment/) |
| analyze-and-test-core | Think + Composer | [`.cursor/skills/analyze-and-test-core/`](.cursor/skills/analyze-and-test-core/) |
| run-classic-grid | Composer | [`.cursor/skills/run-classic-grid/`](.cursor/skills/run-classic-grid/) |
| run-remote-block | Composer | [`.cursor/skills/run-remote-block/`](.cursor/skills/run-remote-block/) |
| structure-audit | Think (named only) | [`.cursor/skills/structure-audit/`](.cursor/skills/structure-audit/) |
| improve-skill-from-failure | Think (named only) | [`.cursor/skills/improve-skill-from-failure/`](.cursor/skills/improve-skill-from-failure/) |

CLIs and modules the skills drive: `scripts/`, `arena/matches/run_match.py`, `arena/tournaments/competition.py`, `arena/records/ratings/`, `arena/records/registry.py`; data under `data/games/`, `data/ratings/`, and `data/bot_versions/`.

## Sources of truth (priority)

1. `RULES.md` + `GeneralsEnv(mode="competition")` + modifiers
2. `competition-module/competition/protocol.py` + `matchup.py` for submission-shaped bots
3. DeepWiki / submodule README for JAX env, in-process agents, remote classic play
4. Classic generals.io lore only when it does not conflict with (1)

See [`docs/sources.md`](docs/sources.md).
