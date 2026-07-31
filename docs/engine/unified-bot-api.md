# Unified bot API

One observation and action shape for every arena bot. Strategy code lives in
`bots/<name>/agent.py` and never imports competition-module or generals_client.

## Layout

```
bots/<name>/agent.py     Agent.act(UnifiedObservation) -> UnifiedAction
        │
        ▼
arena/bot_api.py         UnifiedObservation, UnifiedAction, StrategySession, mappers
        │
   ┌────┴────┐
   ▼         ▼
stdio          remote
bots/<name>/   arena/remote_bridge.py  →  generals_client GameClient
main.py        (UnifiedBot)
(competition
 matchup)
```

## Types

| Type | Shape | Meaning |
| --- | --- | --- |
| `UnifiedObservation` | dataclass | Same fields as `bots/<name>/main.py` `Observation` |
| `UnifiedAction` | 5-tuple | `(pass, row, col, dir, split)` — competition protocol |

`pass`: `0` = move, `1` = pass, `2` = build (competition only; rewritten to pass on live generals.io).

## How bots plug in

1. Implement `class Agent` in `bots/<name>/agent.py` with `act(self, obs)` returning a 5-tuple.
2. Keep `main.py` / `run.sh` unchanged for local competition stdio.
3. For live play, use `scripts/remote_play.py` — it loads the same `agent.py` through `StrategySession`.

No per-bot wire code is required.

## Bridges

| Bridge | Module | Wire |
| --- | --- | --- |
| Stdio | `bots/<name>/main.py` | competition `matchup.py` line protocol |
| Remote | `arena/remote_bridge.UnifiedBot` | `generals_client` (`GeneralsTransport` / `GameClient`) |
| Legacy remote | `arena/remote_adapter.StdioStrategyAdapter` | competition-module `generals.agents.Agent` (local harness only) |

Remote logging and fidelity rules live in `arena/remote_client.FidelityRemoteSession`
(`counts_toward_block`, `result_reason`, `opponent_is_bot`).

## Username policy (generals_client)

Live registration uses `generals_client.GameClient.register_username`. Usernames
**must** start with `[Bot]`. `arena/remote_bridge.ensure_bot_username` adds the
prefix when env omits it. This differs from the old competition-module client,
which stripped `[Bot]` for `botws.generals.io`.

## See also

- [`remote-play-setup.md`](remote-play-setup.md) — credentials and CLI
- [`remote-eval-heuristics.md`](remote-eval-heuristics.md) — evaluation rules
