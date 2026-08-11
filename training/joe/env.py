"""Competition env construction for training (plan section 4).

``mode="competition"`` pins ``min_generals_distance=17`` and mode is
authoritative over kwargs, so curriculum stages construct the env from
explicit kwargs equal to the preset except the distance window. Deriving the
kwargs from ``_MODE_PRESETS`` itself makes the equality structural; the test
suite still asserts it against a real ``mode="competition"`` env.

A new env object is built per stage instead of mutating distance attributes
in place: ``GeneralsEnv._make_pool_batch`` is jitted with ``self`` static
(hashed by identity), so mutating an env would silently reuse pool kernels
compiled for the old distance window.
"""

from generals.core.env import _MODE_PRESETS, GeneralsEnv


def competition_env_kwargs(min_generals_distance: int,
                           max_generals_distance: int | None,
                           pool_size: int) -> dict:
    """The competition preset as explicit kwargs, distance window swapped."""
    kwargs = dict(_MODE_PRESETS["competition"])
    kwargs["min_generals_distance"] = min_generals_distance
    kwargs["max_generals_distance"] = max_generals_distance
    kwargs["pool_size"] = pool_size
    return kwargs


def preset_min_generals_distance() -> int:
    return _MODE_PRESETS["competition"]["min_generals_distance"]


def make_competition_env(min_generals_distance: int,
                         max_generals_distance: int | None,
                         pool_size: int) -> GeneralsEnv:
    return GeneralsEnv(**competition_env_kwargs(
        min_generals_distance, max_generals_distance, pool_size))
