# Benchmark agents (competition-module wrappers)

Fixed baselines that delegate to upstream JAX agents in
`competition-module/generals/agents/`. Arena heuristics are measured against
these bots; do **not** retune thresholds or edit upstream logic here.

## Layout

| Bot directory | Upstream class | Notes |
| --- | --- | --- |
| `bots/cm_random/` | `generals.agents.RandomAgent` | Stochastic valid-move baseline |
| `bots/cm_expander/` | `generals.agents.ExpanderAgent` | Aggressive expansion baseline |
| `bots/cm_hunter/` | `generals.agents.HunterAgent` | Garrison + conveyor + general hunt |
| `bots/cm_harvester/` | `generals.agents.harvester_agent.HarvesterAgent` | Hunter + neutral castle banking |

Each bot is a thin stdio shell:

- `agent.py` — `make_cm_agent()` from `bots/_common/cm_adapter.py`
- `main.py` / `run.sh` — same wire loop as `bots/smoke/`

The adapter maps stdio wire grids to `generals.core.observation.Observation`,
calls the upstream `act(observation, key)`, and maps the JAX action back to
`(pass, row, col, dir, split)`.

## Skipped upstream types

| Type | Reason |
| --- | --- |
| `generals.agents.Agent` | Abstract base; no decision logic |

## Verification

```bash
source .venv/bin/activate
python competition-module/competition/matchup.py \
  bots/cm_expander/run.sh bots/smoke/run.sh \
  --mode competition --seed 0
```

Run one match per benchmark bot before trusting the wrapper.

## Measurement grid

Default `scripts/measure_heuristics.py` does **not** include `cm_*` bots (to
keep the heuristic round grid size stable). To add benchmark matchups manually:

```bash
python -m arena.run_match \
  bots/expand_plus/run.sh bots/cm_hunter/run.sh \
  --mode competition --seed 0 --tag benchmark_vs_hunter
```

Or extend `BASELINE_BOTS` / add a `BENCHMARK_BOTS` list in
`scripts/measure_heuristics.py` when you want a dedicated benchmark round.

## Rules

- Never edit `competition-module/generals/agents/` from this repo.
- Never copy upstream decision code into `bots/cm_*/agent.py`.
- Strategy notes for arena heuristics stay under `docs/research/strategies/`;
  this file documents process and mapping only.
