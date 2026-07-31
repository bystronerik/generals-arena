"""Stored game records, bot identity, ratings, and report rendering.

The data layer: schema and JSON IO for `data/games/`, source-closure hashes,
the bot version registry, the batch rating fit, and markdown tables. Depends on
no other arena subpackage — no JAX, no competition-module, no network. Import
submodules directly.
"""

__all__: list[str] = []
