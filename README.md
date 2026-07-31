# Generals Arena

Research bot arena for the [Generals Competition](https://www.generals.bot/) on top of the vendored [`competition-module`](competition-module) submodule ([strakam/generals-bots](https://github.com/strakam/generals-bots)).

Competition gameplay differs from classic generals.io. Read [`RULES.md`](RULES.md) first.

## Layout

```text
generals-arena/
├── README.md                 # this file
├── AGENTS.md                 # agent workflow only
├── RULES.md                  # competition rules (processed)
├── requirements.txt          # arena deps + sandbox pin notes
├── competition-module/       # git submodule (engine + matchup)
├── bots/                     # stdio competition bots (Phase 1+)
├── arena/                    # match runner, ratings, store (Phase 2)
├── data/games/               # stored match outcomes
├── data/ratings/             # leaderboard snapshots
├── scripts/                  # thin CLIs
├── docs/                     # game and bot knowledge (small files)
├── .cursor/skills/           # Cursor skills (Phase 2)
└── prompts/                  # optional phase kickoff prompts
```

## Install

Prefer **CPython 3.12**. System Python 3.14 can fail on pygame / engine pins. Use the project venv:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e competition-module   # local competition matches
pip install -e client               # generals_client wire for live play (optional)
pip install -r requirements.txt
```

Submitted-bot sandbox pins live in [`competition-module/competition/requirements.txt`](competition-module/competition/requirements.txt). Use those when you need the evaluation image library set.

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
python arena/run_match.py \
  bots/smoke/run.sh \
  competition-module/competition/agents/expander_python/run.sh \
  --mode competition --seed 0

# or
python scripts/smoke_match.py --seed 0 --update-ratings
```

A valid competition match must finish under `--mode competition`. Full games can run to 1200 turns.

## Tournament and leaderboard

```bash
source .venv/bin/activate
python scripts/tournament.py \
  bots/smoke/run.sh \
  competition-module/competition/agents/expander_python/run.sh \
  --seeds 0-2

python scripts/leaderboard.py
```

See [`docs/arena/tournament.md`](docs/arena/tournament.md) and [`docs/arena/ratings.md`](docs/arena/ratings.md).

## Docs

Start at [`docs/index.md`](docs/index.md). Sources: [`docs/sources.md`](docs/sources.md).
