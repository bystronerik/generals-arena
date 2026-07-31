# Adding a bot

How to add a bot under `bots/<name>/`.

## Layout

```text
bots/<name>/
├── agent.py    # decide actions from observation
├── main.py     # stdio protocol IO
├── probe.py    # optional: per-turn introspection (NOT part of the bot)
└── run.sh      # entry the matchup runner executes
```

Optional: `build.sh` beside `run.sh` for compile / weight prep.

## Per-turn introspection

Add `bots/<name>/probe.py` to see inside the bot on recorded matches:

```python
def extras(agent) -> dict:
    return {"phase": agent._core.memory.phase}
```

It sits beside the agent but is **not part of it**:

- excluded from the source closure, so it is in no content hash and no
  submission bundle — editing it never forks the bot's rating identity;
- loaded only by `arena.instrument.runner`, and only when a match runs with
  `--record`;
- **must never be imported by the agent.** `arena/records/fingerprint.py`
  raises if it is: unhashed code must be unreachable from the hashed program,
  or a probe could change how the bot plays without moving its hash;
- passive — read attributes, return them, change nothing;
- every key needs an entry in `arena/records/telemetry_schema.py`. An
  undeclared key fails the recorded match rather than landing untyped.

Bots emit no telemetry themselves. `main.py` and `_common/wire.py` are the
protocol and nothing else, because everything in them ships to the judge.
See [`docs/arena/trajectories.md`](../arena/trajectories.md).

## Pattern

Copy from `competition-module/competition/agents/expander_python/`. Keep the same wire protocol as [`../competition/protocol.md`](../competition/protocol.md).

## Register for matches

Pass your `run.sh` to matchup:

```bash
python competition-module/competition/matchup.py \
  bots/<name>/run.sh \
  competition-module/competition/agents/expander_python/run.sh \
  --mode competition --seed 0
```

## Rules

- Competition mode only for verification.
- Do not edit submodule agents in place; wrap under `bots/`.
- Put strategy notes in `docs/bots/` or research notes, not in `AGENTS.md`.

## Smoke bot

`bots/smoke/` is the reference stdio bot. Observation notes go in `docs/bots/smoke.md` after the first verified match.

## Heuristic bots

Scaffolded with the `build-bot-from-spec` skill from a spec under
`docs/research/strategies/`. One doc
file each, one experiment note each under `docs/research/experiments/`:

| Bot | Idea | Doc | Experiment note |
| --- | --- | --- | --- |
| `bots/expand_plus/` | expand | [`expand-plus.md`](expand-plus.md) | [`001-expand-plus-frontier-march.md`](../research/experiments/001-expand-plus-frontier-march.md) |
| `bots/castle_builder/` | castle-aware | [`castle-builder.md`](castle-builder.md) | [`002-castle-builder-early-investment.md`](../research/experiments/002-castle-builder-early-investment.md) |
| `bots/general_hunter/` | late-game hunt | [`general-hunter.md`](general-hunter.md) | [`003-general-hunter-deathtouch-beeline.md`](../research/experiments/003-general-hunter-deathtouch-beeline.md) |

Every one of these strategy changes needs a note before merge — see
[`docs/research/experiment-protocol.md`](../research/experiment-protocol.md)
and skill `evaluate-bot-change`.
