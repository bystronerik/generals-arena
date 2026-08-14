# Generals Arena

Research bot arena for the [Generals Competition](https://www.generals.bot/) on top of the vendored [`competition-module`](competition-module) submodule ([strakam/generals-bots](https://github.com/strakam/generals-bots)).

Competition gameplay differs from classic generals.io. Read [`RULES.md`](RULES.md) first.

## Layout

```text
generals-arena/
├── README.md                 # this file
├── AGENTS.md                 # agent workflow only
├── RULES.md                  # competition rules (processed)
├── requirements.txt          # arena runtime deps (numpy)
├── requirements-dev.txt      # test-only deps (pytest, elote oracle)
├── requirements-sandbox.txt  # optional torch/jax sandbox pins
├── pytest.ini                # unified test discovery
├── competition-module/       # git submodule (engine + matchup)
├── client/                   # generals_client submodule (live generals.io)
├── competition-scraper/      # git submodule (generals.bot replay downloader)
├── bots/                     # stdio competition bots
├── arena/                    # match runner, ratings, store
├── tests/                    # core pytest suite
├── data/games/               # stored match outcomes (JSON gitignored)
├── data/ratings/             # fit + leaderboard snapshots (local; gitignored)
├── data/bot_versions/        # bot version registry (committed)
├── data/remote_games/        # live classic session logs (gitignored)
├── competition-replays/      # scraped generals.bot replays (gitignored)
├── scripts/                  # thin CLIs
├── docs/                     # game and bot knowledge (small files)
└── .cursor/skills/           # Cursor skills
```

## Install

Prefer **CPython 3.12**. System Python 3.14 can fail on pygame / engine pins. Use the project venv:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e competition-module   # local competition matches
pip install -e client               # generals_client wire for live play (optional)
pip install -r requirements.txt
pip install -r requirements-dev.txt  # to run the test suite
```

Optional: match the competition evaluation image library set:

```bash
pip install -r requirements-sandbox.txt
```

Submitted-bot sandbox pins also live in [`competition-module/competition/requirements.txt`](competition-module/competition/requirements.txt).

## Tests

```bash
source .venv/bin/activate
pytest
```

Config: [`pytest.ini`](pytest.ini). Core targets: [`docs/research/strategies/test-core-skill.md`](docs/research/strategies/test-core-skill.md).

## Smoke match

Direct matchup (no store):

```bash
source .venv/bin/activate
python competition-module/competition/matchup.py \
  bots/smoke/run.sh \
  competition-module/competition/agents/expander_python/run.sh \
  --mode competition --seed 0
```

Arena store (writes `data/games/<game_id>.json`):

```bash
python -m arena.matches.run_match \
  bots/smoke/run.sh \
  competition-module/competition/agents/expander_python/run.sh \
  --mode competition --seed 0

# or
python scripts/smoke_match.py --seed 0
```

A valid competition match must finish under `--mode competition`. Full games can run to 1200 turns.

Run `arena/` modules as modules (`python -m arena.<subpackage>.<name>`) from the repo root,
not as file paths — the package no longer bootstraps `sys.path` for direct
execution. Scripts under `scripts/` still run as `python scripts/<name>.py`.

## Tournament and leaderboard

```bash
source .venv/bin/activate
python scripts/tournament.py \
  bots/smoke/run.sh \
  competition-module/competition/agents/expander_python/run.sh \
  --seeds 0-2

python scripts/leaderboard.py --print
```

Ratings are a batch Bradley-Terry + Davidson-draws + seat-term fit, keyed on
`bot_id@content_hash`, with a covariance matrix behind every interval. The fit is
**per round**: each `data/games/<round>/` is fitted independently and there is no
pooled table, because two byte-identical programs measured in different rounds
fitted 46 Elo apart — more than the decision thresholds. Every table is anchored
inside its own round and carries a `scale` token; **ratings from two different
rounds are not comparable**, and a contrast needs both arms in one round. There is
no incremental update: every write refits every round.

Output lands under `data/ratings/` (gitignored) — `fits/<round>.json` per rated
round, plus `leaderboard.json` and `leaderboard.md`. `data/bot_versions/` is
committed, and `git log -p data/bot_versions/<bot>.json` is a bot's improvement
history. Commit round reports under `docs/research/measurements/` when publishing
results. See [`docs/arena/ratings.md`](docs/arena/ratings.md).

Decide keep-or-revert from the pairwise contrast, never from rank:
[`docs/arena/decision-rule.md`](docs/arena/decision-rule.md).

See [`docs/arena/tournament.md`](docs/arena/tournament.md),
[`docs/arena/ratings.md`](docs/arena/ratings.md), and
[`docs/arena/bot-version-registry.md`](docs/arena/bot-version-registry.md).

## Leaderboard replays

Download finished competition games from the `generals.bot` leaderboard into
`competition-replays/` (gitignored, ~0.6 MB per replay):

```bash
python scripts/scrape_replays.py erik.bystron
```

They are observational data only — never fed into `data/games/` or the rating
fit. See [`docs/engine/leaderboard-replays.md`](docs/engine/leaderboard-replays.md).

## Docs

Start at [`docs/index.md`](docs/index.md). Sources: [`docs/sources.md`](docs/sources.md).
