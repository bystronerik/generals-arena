"""Live generals.io play and its result logs.

`bridge` and `client` run arena strategies against the real server; `block`
and `report` aggregate the logs that come back. May depend on `arena.bot_api`
and `arena.records` (git pin and timestamps only), never on `arena.matches` or
`arena.tournaments` — live play must not pull JAX into its import path.
Import submodules directly.
"""

__all__: list[str] = []
