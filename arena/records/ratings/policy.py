"""
Which stored games enter the fit, and under what prior.

Eligibility is decided from each record's **own fields** — `mode`, `round`,
`engine_version`, both content hashes — never from the directory a record
happens to sit in. The classic/remote exclusion `AGENTS.md` mandates used to
be enforced by directory convention and reviewer discipline alone; here it is
an explicit check, and every rejection is counted rather than dropped
silently.
"""

from __future__ import annotations

from dataclasses import dataclass

from arena.records.registry import Registry
from arena.records.store import UNKNOWN_HASH, MIN_SCHEMA_VERSION, GameRecord

# Rejection reasons, in the order they are tested. These strings are the keys
# of `fit.json`'s `excluded` object, so they are part of the on-disk format.
MODE_NOT_ELIGIBLE = "mode_not_eligible"
SCHEMA_TOO_OLD = "schema_too_old"
UNKNOWN_CONTENT_HASH = "unknown_content_hash"
UNREGISTERED_HASH = "unregistered_hash"
ENGINE_MISMATCH = "engine_mismatch"
SELF_PLAY_EXCLUDED = "self_play_excluded"

REJECTION_REASONS = (
    MODE_NOT_ELIGIBLE,
    SCHEMA_TOO_OLD,
    UNKNOWN_CONTENT_HASH,
    UNREGISTERED_HASH,
    ENGINE_MISMATCH,
    SELF_PLAY_EXCLUDED,
)


@dataclass(frozen=True)
class Policy:
    """What counts as a rateable game."""

    modes: tuple[str, ...] = ("competition",)
    # None means "whatever era the current checkout is in"; `cli` resolves it
    # from the submodule HEAD before the fit runs.
    engine_version: str | None = None
    require_registered: bool = True
    # Self-play contributes nothing to any strength (the terms cancel) but is a
    # clean, strength-free estimator of the seat and draw parameters.
    include_self_play: bool = True
    min_games_display: int = 30

    def to_dict(self) -> dict:
        return {
            "modes": list(self.modes),
            "engine_version": self.engine_version,
            "require_registered": self.require_registered,
            "include_self_play": self.include_self_play,
            "min_games_display": self.min_games_display,
        }


@dataclass(frozen=True)
class Prior:
    """
    MAP, not MLE.

    `sigma = 200` is worth about three pseudo-games against the anchor:
    negligible against a 200-game arm (1.5% weight), but enough to keep an
    undefeated entity finite instead of running off to infinity.
    """

    mean: float = 1500.0
    sigma: float = 200.0
    seat_sigma: float = 200.0
    draw_sigma: float = 2.0

    def to_dict(self) -> dict:
        return {
            "mean": self.mean,
            "sigma": self.sigma,
            "seat_sigma": self.seat_sigma,
            "draw_sigma": self.draw_sigma,
        }


def entity_key(bot_id: str, content_hash: str) -> str:
    """The rated identity. Nothing in the model knows `bot_id` except as a label."""
    return f"{bot_id}@{content_hash}"


def rejection_reason(
    record: GameRecord, policy: Policy, registry: Registry | None = None
) -> str | None:
    """Why this record is not rateable, or None when it is."""
    if record.mode not in policy.modes:
        return MODE_NOT_ELIGIBLE
    if record.schema_version < MIN_SCHEMA_VERSION:
        return SCHEMA_TOO_OLD
    hashes = (record.bot_a_content_hash, record.bot_b_content_hash)
    if any(not h or h == UNKNOWN_HASH for h in hashes):
        return UNKNOWN_CONTENT_HASH
    if policy.engine_version is not None and record.engine_version != policy.engine_version:
        return ENGINE_MISMATCH
    if policy.require_registered:
        if registry is None:
            raise ValueError("policy.require_registered needs a registry")
        sides = ((record.bot_a, record.bot_a_content_hash), (record.bot_b, record.bot_b_content_hash))
        if not all(registry.is_registered(bot, digest) for bot, digest in sides):
            return UNREGISTERED_HASH
    if not policy.include_self_play and _is_self_play(record):
        return SELF_PLAY_EXCLUDED
    return None


def eligible(
    record: GameRecord, policy: Policy, registry: Registry | None = None
) -> bool:
    return rejection_reason(record, policy, registry) is None


def _is_self_play(record: GameRecord) -> bool:
    return entity_key(record.bot_a, record.bot_a_content_hash) == entity_key(
        record.bot_b, record.bot_b_content_hash
    )
