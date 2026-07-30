# Competition vs classic

The Generals Competition ruleset is not classic generals.io.

## Main deltas

| Topic | Classic (typical) | Competition (`mode="competition"`) |
| --- | --- | --- |
| Castles | Neutral castles on the map | No neutrals; players **build** (`pass=2`) |
| Endgame | Capture general by army | Capture, plus **deathtouch** from turn **800** |
| Turn cap | Varies | Hard **1200** (draw if no win) |
| Map size | Often fixed / server maps | Rectangular sides **18–21**, pad to 21 |
| Spawn distance | Varies | Walking BFS ≥ **17** |
| Fog | Usually on | On (`perfect_info=False`) |
| Mutual general capture | Some servers swap spoils | Always a **draw** |

## Where to look

- Full rules: [`RULES.md`](../../RULES.md)
- Preset: `competition-module/generals/core/env.py` → `_MODE_PRESETS["competition"]`
- Build: [`build-castles.md`](build-castles.md)
- Deathtouch: [`deathtouch.md`](deathtouch.md)

Do not use classic strategy guides when they conflict with competition mechanics.
