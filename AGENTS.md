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
  tournaments alike — are **observational data, not arena matches**. See
  [What is never stored or rated](#what-is-never-stored-or-rated).
- Secrets — the R2 token, the vast.ai API key — live only in the gitignored
  `.env` or in the environment. Never in a script, a template, or a commit.

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
| Live classic session logs (derived, gitignored) | `data/remote_games/` |
| Scraped leaderboard replays (derived, gitignored) | `competition-replays/<player>/{win,lose,draw}/` |
| Scraped sprint tournaments (derived, gitignored) | `competition-replays/_sprints/<tourney-id>/` |
| Morpheus training-only modules | `training/morpheus/` |
| Morpheus local derived training data (gitignored) | `data/morpheus/` |
| Morpheus remote training runs | Modal Volumes (primary storage) |
| Joe local checkpoint dirs (derived, gitignored) | `data/joe/<run>/` |
| Joe remote durable state (vast.ai runs) | R2 bucket `joe-training`, prefix `joe/<run>/` |
| Submission-harness failure fixtures | `tests/fixtures/submission_bots/` |
| Shared Rust dev tooling | `tools/` |
| Rust submission packager (shared) | `arena/rust_bundle.py` + per-bot `bots/<name>/tools/package_submission.py` |
| Cursor skills | `.cursor/skills/` |

Joe's Modal Volume holds prototyping runs; R2 holds the durable state of the
interruptible vast.ai runs (`training/joe/store.py`) — launcher, credentials, and
resume behavior in [`docs/engine/joe-vast-train.md`](docs/engine/joe-vast-train.md).

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

## What is never stored or rated

Only matches our own runner produced under competition rules are stored and
rated. Each of the following stays out of `data/games/`, out of `data/ratings/`,
and out of every rating fit:

- **Live classic games** (`data/remote_games/`) — a different ruleset.
- **Scraped `generals.bot` replays**, leaderboard and sprint alike
  (`competition-replays/`) — observational data: not our runner, no bot version.
  They stay out of `data/remote_games/` too, and an outcome is derived from
  `Replay.outcome`, never from the folder name, so a scraper regression shows up
  as `folder_disagrees` and not as mislabeled training data. Commands, format,
  and caveats: [leaderboard](docs/engine/leaderboard-replays.md),
  [sprint](docs/engine/sprint-replays.md).
- **Derived training data** — `data/morpheus/` and `data/joe/` shards,
  checkpoints, and materializations. Gitignored, and out of `data/bot_versions/`
  too.

## Test suite budget

The budget is **15 s** warm, and it means the default `pytest -q` over every
`testpaths` entry in [`pytest.ini`](pytest.ini) — not `pytest -q tests`. Treat
15 s as a ceiling to defend, not a number to ratchet: a new test that costs a
second needs an argument for why it cannot be made cheap without giving up what
it checks. Test core logic only — parsers, store, **rating order-invariance and
the count-table digest**, the version registry, bot_api mapping, fidelity
classification, classic match results, remote human-count filter. Report a
proven defect; do not fix production code inside the test step.

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
local process does not report. **Check the job's own output a few minutes after
you start it, before you do anything else, and confirm it got past startup.** A
Modal job that dies at import looks exactly like one that is working. Log
commands, the two failure shapes, and the buffering trap:
[`docs/engine/modal-jobs.md`](docs/engine/modal-jobs.md).

## Subagent roles (workflow only)

| Role | Owns | Skills | Done when |
| --- | --- | --- | --- |
| explorer | Read `docs/` + `RULES.md`; map engine APIs | `run-competition-match` to verify observations | Notes cite a finished `--mode competition` match; no unmeasured strategy claims |
| bot-author | Add or change `bots/<name>/` from a spec under `docs/research/strategies/`; keep stdio protocol intact | `build-bot-from-spec`, then `run-competition-match` | New/changed `run.sh` finishes a competition match |
| strategist | Write strategy specs; enforce bot diversity; choose parameter revisions after a round | `write-strategy-spec`, `check-bot-diversity`, `tune-bot-parameters` | Spec in `docs/research/strategies/` with diversity verdict; no bot code written |
| evaluator | Fixed grid; before/after winrate; **pairwise rating contrast with an interval, inside one round** | `run-measurement-round`, `evaluate-bot-change`, `update-leaderboard` | Games stored in `data/games/<round>/`; verdict quoted per [`docs/arena/decision-rule.md`](docs/arena/decision-rule.md), never as a rank and never across rounds |
| remote-operator | Live classic blocks; human gate ladder | `run-remote-block`, `run-classic-grid` | Block report under `docs/research/measurements/`; logs in `data/remote_games/` |
| auditor | Structure and duplication review | `structure-audit` | Report names concrete files and lines; no edits without a follow-up ask |
| docs-keeper | Keep `docs/` small and single-purpose; sync schema and protocol pages with the code + `RULES.md` | none required | No strategy moved into `AGENTS.md` or a skill file |
| tester | Core coverage under `tests/`; fixtures under `tests/fixtures/` | none required | New tests pass, the suite holds the [15 s budget](#test-suite-budget), and every untested core target is named |

Start points: [`docs/index.md`](docs/index.md) + [`RULES.md`](RULES.md) for the
engine, [experiment-protocol](docs/research/experiment-protocol.md) for a
measurement round. Store games through `arena/matches/run_match.py` or
`scripts/measure_heuristics.py`, then run `update-leaderboard`.

## Cursor skills

Index and model split: [`.cursor/skills/README.md`](.cursor/skills/README.md).
Taxonomy: [`docs/research/strategies/skills-workflow.md`](docs/research/strategies/skills-workflow.md).

CLIs and modules the skills drive: `scripts/`, `arena/matches/run_match.py`, `arena/tournaments/competition.py`, `arena/records/ratings/`, `arena/records/registry.py`; data under `data/games/`, `data/ratings/`, and `data/bot_versions/`.

## Sources of truth (priority)

1. `RULES.md` + `GeneralsEnv(mode="competition")` + modifiers
2. `competition-module/competition/protocol.py` + `matchup.py` for submission-shaped bots
3. DeepWiki / submodule README for JAX env, in-process agents, remote classic play
4. Classic generals.io lore only when it does not conflict with (1)

See [`docs/sources.md`](docs/sources.md).
