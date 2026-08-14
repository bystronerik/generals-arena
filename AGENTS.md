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
- Scraped `generals.bot` replays — per-player leaderboard games and whole sprint
  tournaments alike — are **observational data, not arena matches**. They never
  enter `data/games/` or a rating fit. See
  [Leaderboard replays](#leaderboard-replays-scraped) and
  [Sprint replays](#sprint-replays-scraped) below.

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
| Scraped sprint tournaments (derived, gitignored) | `competition-replays/_sprints/<tourney-id>/` |
| Morpheus training-only modules | `training/morpheus/` |
| Morpheus local derived training data (gitignored) | `data/morpheus/` |
| Joe local checkpoint dirs (derived, gitignored) | `data/joe/<run>/` |
| Joe remote durable state (vast.ai runs) | R2 bucket `joe-training`, prefix `joe/<run>/` |
| Submission-harness failure fixtures | `tests/fixtures/submission_bots/` |
| Cursor skills | `.cursor/skills/` |

Modal Volumes remain the primary storage for remote Morpheus training runs.
Local `data/morpheus/` shards, checkpoints, and materializations stay
gitignored and never enter `data/games/`, `data/ratings/`, or
`data/bot_versions/`.

For Joe, the Modal Volume holds prototyping runs and the Cloudflare R2
bucket `joe-training` holds the durable state of interruptible vast.ai runs
(`training/joe/store.py`). The R2 token is scoped to that one bucket and
lives only in the gitignored `.env` or the environment — never in a script,
template, or commit. Local `data/joe/` dirs are derived and gitignored.
Interruptible runs launch through `scripts/joe_vast_train.py` (onstart:
`scripts/joe_vast_onstart.sh`). The vast.ai CLI is the pip package `vastai`
(console script `.venv/bin/vastai`). See
[`docs/engine/joe-vast-train.md`](docs/engine/joe-vast-train.md).

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

Also store the game under `data/games/<round>/` before you refit ratings. There is
no incremental rating update: every write refits every round. Ratings are fitted
**one round at a time** and no number is fitted across rounds, so a decision
contrast needs both arms in the same round — see
[`docs/arena/ratings.md`](docs/arena/ratings.md).

## Commit messages

Write all commit messages in ASD-STE100 Simplified Technical English.

Structure:

1. The first line is a short summary of the change. Keep it at 72 characters
   or less. Start with a verb in the imperative form ("Add", "Fix", "Remove").
2. The second line is empty.
3. The lines that follow describe the change in more detail. Give the context
   and the reasons for the change.

STE rules that apply most:

- Use the active voice.
- Write short sentences: not more than 20 words for an instruction, not more
  than 25 words for a description.
- Write about one topic per paragraph, with not more than six sentences.
- Do not use idioms, metaphors, or other figurative language.
- Use one name for one thing. Use the names that the code and the docs use.

## Modal jobs

Anything under `scripts/*_modal_*.py` runs remotely and can fail in ways the
local process does not report. **Check the job's output a few minutes after
starting it, before doing anything else, and confirm it got past startup.** A
Modal job that dies at import looks exactly like a Modal job that is working:
no output, process still alive, no error locally.

```bash
modal app list                  # find the ephemeral app id
modal app logs <app-id>         # the container's own traceback
```

Two failure shapes to expect:

- **Module-level code runs twice.** Modal re-imports the script inside the
  container to find the function, so every top-level statement executes there
  too — with only that file mounted. A `from arena... import ...` at module
  scope resolves locally and raises `ModuleNotFoundError` in the container,
  before the job's first line. Guard repo imports with `modal.is_local()`.
- **Missing local inputs fail late.** `add_local_file` is evaluated at image
  build, so a stale or deleted path raises after the run has apparently
  started.

Do not pipe a backgrounded `modal run` through `tail` or `head`: they buffer
until the process exits, which turns a crash into an apparent hang. Redirect to
a file, or read the app logs.

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
python scripts/scrape_replays.py --refile-only      # offline: repair outcome folders
```

Process rules:

- `win/lose/draw` is the **queried player's** result. An early scraper filed by
  side A's result instead; `--refile-only` repairs that offline and is safe to
  re-run. Verified clean on 2026-08-05: 13,088 checked, 0 refiled.
- Still derive outcomes from `Replay.outcome`, never the folder name — the
  folder is provenance, so a scraper regression shows up as `folder_disagrees`
  rather than as mislabeled training data.

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

## Sprint replays (scraped)

Whole `generals.bot` sprint tournaments — bot-vs-bot games over archived
checkpoints, competition rules, published as one results asset:

```bash
python scripts/scrape_sprint.py                     # default: sprint-2026-08-08
python scripts/scrape_sprint.py <url-or-path> --dry-run
```

Process rules:

- Different source, different scraper. The `competition-scraper` submodule
  speaks only `/api/leaderboard`; sprint replays are gzipped blobs named by a
  results asset, so `scripts/scrape_sprint.py` fetches them. Do not extend the
  submodule for this.
- Output goes to `competition-replays/_sprints/<tourney-id>/<pair>/` —
  gitignored, ~0.9 MB per replay. The `_sprints/` prefix keeps tournaments out
  of the per-player namespace.
- Same standing as leaderboard replays: **observational data**. Not our runner,
  no bot version, never into `data/games/`, `data/ratings/`, or
  `data/remote_games/`.
- Results carry the organizers' engine pin and resource limits, and their own
  note says the pin post-dates the 2026-07-27 move-order tiebreak flip. Read
  `run.ruleset` in the asset before comparing a sprint result to anything.
- Needs `httpx`, and the submodule for its pacing primitives.
- Format, sidecar shape, and the sampling caveats that affect analysis:
  [`docs/engine/sprint-replays.md`](docs/engine/sprint-replays.md).

## Subagent roles (workflow only)

Roles are process pointers. Put game and bot knowledge in `docs/`. Do not put strategy in this file or in skill bodies beyond links into `docs/`.

| Role | Owns | Skills | Done when |
| --- | --- | --- | --- |
| explorer | Read `docs/` + `RULES.md`; map engine APIs | `run-competition-match` to verify observations | Notes cite a finished `--mode competition` match; no unmeasured strategy claims |
| bot-author | Add or change `bots/<name>/`; keep stdio protocol intact | `build-bot-from-spec`, then `run-competition-match` | New/changed `run.sh` finishes a competition match |
| strategist | Write strategy specs; enforce bot diversity | `write-strategy-spec`, `check-bot-diversity` | Spec in `docs/research/strategies/` with diversity verdict |
| evaluator | Fixed grid; before/after winrate; **pairwise rating contrast with an interval, inside one round** | `run-measurement-round`, `evaluate-bot-change`, `update-leaderboard` | Games in `data/games/<round>/`; verdict quoted per [`docs/arena/decision-rule.md`](docs/arena/decision-rule.md), never as a rank and never across rounds |
| remote-operator | Live classic blocks; human gate ladder | `run-remote-block`, `run-classic-grid` | Block report under `docs/research/measurements/`; logs in `data/remote_games/` |
| auditor | Structure and duplication review | `structure-audit` | Report names concrete files and lines; no edits without a follow-up ask |
| docs-keeper | Keep `docs/` small and accurate; sync with code + `RULES.md` | none required | Topic files stay single-purpose; no strategy moved into `AGENTS.md` |
| tester | Core coverage under `tests/`; fixtures under `tests/fixtures/` | `analyze-and-test-core` | New tests pass, the suite stays under 15 s, and every untested core target is named |

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
4. Decide from the pairwise contrast in [`docs/arena/decision-rule.md`](docs/arena/decision-rule.md), taken **inside the round both arms played** — never from a rank, and never across rounds.

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
4. The suite budget is **15 s** with a warm cache, and it means the default
   `pytest -q` over every `testpaths` entry — not `pytest -q tests`. Treat 15 s
   as a ceiling to defend, not a number to ratchet — a new test that costs a
   second needs an argument for why it cannot be made cheap without giving up
   what it checks.

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
