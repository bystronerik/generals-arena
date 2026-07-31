"""Batch orchestration: many matches over a bot roster.

`competition` and `classic` mirror the runners in `arena.matches`, expanding a
roster into seeded pairs and driving them through the `parallel` worker pool.
May depend on `arena.matches` and `arena.records`, never on `arena.remote`.
Import submodules directly.
"""

__all__: list[str] = []
