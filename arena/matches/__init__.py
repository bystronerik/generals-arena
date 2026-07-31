"""Running a single local stdio match.

`loop` drives the competition-module machinery; `competition` and `classic`
configure it with their respective envs; `run_match` stores the result. This is
the only arena subpackage that imports JAX and competition-module, and it may
depend on `arena.records` but nothing else. Import submodules directly.
"""

__all__: list[str] = []
