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

```bash
pip install -e competition-module
```

For arena tools (Phase 2+), also install root deps:

```bash
pip install -r requirements.txt
```

Submitted-bot sandbox pins live in [`competition-module/competition/requirements.txt`](competition-module/competition/requirements.txt). Use those when you need the evaluation image library set.

## Smoke match

After Phase 1 adds `bots/smoke/`:

```bash
python competition-module/competition/matchup.py \
  bots/smoke/run.sh \
  competition-module/competition/agents/expander_python/run.sh \
  --mode competition --seed 0
```

Until then, run two reference expanders:

```bash
python competition-module/competition/matchup.py --mode competition --seed 0
```

A valid competition match must finish under `--mode competition`.

## Docs

Start at [`docs/index.md`](docs/index.md). Sources: [`docs/sources.md`](docs/sources.md).
