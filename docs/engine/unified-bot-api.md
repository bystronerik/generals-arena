# Unified bot API

One observation and action shape for every arena bot. Strategy code lives in
`bots/<name>/agent.py` and never imports competition-module or generals_client.

## Layout

```
bots/<name>/agent.py     Agent.act(UnifiedObservation) -> UnifiedAction
bots/_common/wire.py     shared stdio loop + generic telemetry
bots/_common/strategy_common.py   shared strategy helpers (StrategyContext)
        │
        ▼
arena/bot_api.py         UnifiedObservation, UnifiedAction, ArenaAgent, StrategySession, mappers
        │
   ┌────┴────┐
   ▼         ▼
stdio          remote
bots/<name>/   arena/remote/bridge.py  →  generals_client GameClient
main.py        (UnifiedBot)
(competition
 matchup)
```

## Agent contract

Arena bots implement a duck-typed class named ``Agent`` in ``bots/<name>/agent.py``.
The formal contract is :class:`arena.bot_api.ArenaAgent` — a :class:`typing.Protocol`
for static checking and docs. Bots do **not** need to import or inherit from it at
runtime.

| Method | Signature | Notes |
| --- | --- | --- |
| ``__init__`` | ``(player_id: int, H: int, W: int)`` | Called once after the stdio handshake |
| ``act`` | ``(obs) -> UnifiedAction`` | Return a 5-tuple each turn |
| ``telemetry_extras`` | ``() -> dict`` | Optional; keys appended to ``[telemetry]`` stderr line |

Example (runtime — no Protocol import):

```python
class Agent:
    def __init__(self, player_id, H, W):
        ...

    def act(self, obs):
        return (1, 0, 0, 0, 0)  # pass

    def telemetry_extras(self):
        return {"enemy_general_sighted": 0}
```

## Types

| Type | Shape | Meaning |
| --- | --- | --- |
| `UnifiedObservation` | dataclass | Same fields as `bots/<name>/main.py` `Observation` |
| `UnifiedAction` | 5-tuple | `(pass, row, col, dir, split)` — competition protocol |

`pass`: `0` = move, `1` = pass, `2` = build (competition only; rewritten to pass on live generals.io).

## How bots plug in

1. Implement `class Agent` in `bots/<name>/agent.py` matching :class:`ArenaAgent`.
2. Keep `main.py` as a thin call to ``bots/_common/wire.run_stdio``; keep `run.sh` unchanged.
3. For live play, use `scripts/remote_play.py` — it loads the same `agent.py` through `StrategySession`.

No per-bot wire code is required.

## Bridges

| Bridge | Module | Wire |
| --- | --- | --- |
| Stdio | `bots/_common/wire.py` via `bots/<name>/main.py` | competition `matchup.py` line protocol |
| Remote | `arena/remote/bridge.UnifiedBot` | `generals_client` (`GeneralsTransport` / `GameClient`) |
| Legacy remote | `arena/remote/adapter.StdioStrategyAdapter` | competition-module `generals.agents.Agent` (local harness only) |

Remote logging and fidelity rules live in `arena/remote/client.FidelityRemoteSession`
(`counts_toward_block`, `result_reason`, `opponent_is_bot`).

## Username policy (generals_client)

Live registration uses `generals_client.GameClient.register_username`. Usernames
**must** start with `[Bot]`. `arena/remote/bridge.ensure_bot_username` adds the
prefix when env omits it. This differs from the old competition-module client,
which stripped `[Bot]` for `botws.generals.io`.

## See also

- [`remote-play-setup.md`](remote-play-setup.md) — credentials and CLI
- [`remote-eval-heuristics.md`](remote-eval-heuristics.md) — evaluation rules
