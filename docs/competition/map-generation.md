# Map generation (competition)

Pinned by `GeneralsEnv(mode="competition")` in `competition-module/generals/core/env.py`.

| Parameter | Value |
| --- | --- |
| Side length | each side independent in **[18, 21]** |
| Pad | `pad_to=21` (smaller boards mountain-padded) |
| Mountains | preset density **(0.24, 0.26)** |
| Castles | generated then **stripped** (`build_castles`) |
| Spawn distance | walking BFS ≥ **17** |
| Fog | on |
| Truncation | **1200** turns |

## Official page note

[generals.bot/rules](https://www.generals.bot/rules) describes mountains as about 19–23% (65–105 tiles). The env preset uses 24–26%. Prefer the preset for local competition matches. See the cross-check table in [`RULES.md`](../../RULES.md).

## Connectivity

A passable path between generals is guaranteed by the generator.

## Local seeds

`matchup.py --seed N` makes research runs reproducible. Competition infra may not reuse seeds between graded matches.
