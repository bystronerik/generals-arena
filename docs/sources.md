# Sources

Priority order for truth:

1. Root [`RULES.md`](../RULES.md) + `GeneralsEnv(mode="competition")` + modifiers
2. `competition-module/competition/protocol.py` + `matchup.py`
3. DeepWiki / submodule README for engine APIs
4. Classic generals.io lore only when it does not conflict with (1)

## Official competition

- Site: [https://www.generals.bot/](https://www.generals.bot/)
- Rules: [https://www.generals.bot/rules](https://www.generals.bot/rules)
- Docs / environment: [https://www.generals.bot/docs](https://www.generals.bot/docs)

## Submodule

- Path: `competition-module/`
- Upstream: [strakam/generals-bots](https://github.com/strakam/generals-bots)
- Competition preset: `generals/core/env.py`
- Modifiers: `generals/modifiers/build_castles.py`, `deathtouch.py`
- Stdio + matchup: `competition/`

## DeepWiki

- [DeepWiki strakam/generals-bots](https://deepwiki.com/strakam/generals-bots)

Use DeepWiki for JAX env, in-process agents, training, and remote classic play.

**The competition stdio protocol is not on DeepWiki.** Use `competition-module/competition/` and [`competition/protocol.md`](competition/protocol.md).
