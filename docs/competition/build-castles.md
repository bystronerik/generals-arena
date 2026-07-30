# Build castles

Competition maps start with **no neutral castles**. Players build them.

Code: `competition-module/generals/modifiers/build_castles.py`.

## Action

```text
2 <row> <col> 0 0
```

`pass` field value `2` means build. Last two fields are ignored.

## Rules

- Cell must be owned, plain land (not general, not already a castle).
- Cost comes from army on that cell.
- Base cost **35**.
- Surcharge per own structure (general + your castles): **max(0, 14 − 2 × Manhattan distance)**.
- Distance ≥ 7 adds nothing.
- Surcharges stack across structures.
- Enemy structures never affect your price.
- Captured enemy castles count as yours for production and for your future prices.
- Remainder army stays on the new castle (can be 0).
- Invalid build → silent pass.
- Builds resolve **before** either player's move each tick, then rewritten to passes for the base step.

## Env wiring

`GeneralsEnv(mode="competition")` sets `build_castles=True` and strips neutrals via `strip_neutral_castles`.

See also [`RULES.md`](../../RULES.md) section 03.
