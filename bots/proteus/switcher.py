"""Strategy switching with hysteresis.

Holds the current strategy and decides when the classifier's opinion is
strong and stable enough to act on. Pure logic — offline-testable.

The counter map is two-wide because the measurement says two: a 32-game
per-cell grid of every pure core against every roster bot found metro
(pooled 0.596) and aegis (0.603) dominated by blitz (0.788) and boom (0.816),
and *uniquely best against nothing*. Aegis is actively harmful in two cells —
1-0-31 against garrison, where two turtles simply run out the RULES.md §07
draw, and 0.14 against boom. Neither is reachable, and neither should be;
see docs/research/strategies/proteus.md §2.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from proteus.classifier import AGGRESSOR, ECONOMY, UNKNOWN, Classification

#: What we play against each classification.
#:
#: `aggressor` and `unknown` share blitz, and that sharing is the design, not
#: an oversight: blitz is the spine, so "this opponent will come at us" and
#: "we do not know yet" are the same instruction — stay put. Only positive
#: evidence of an opponent that spends its army on land moves us off it.
COUNTER = {
    AGGRESSOR: "blitz",   # they walk fists at generals; race them, do not bank
    ECONOMY: "boom",      # castles, slow expanders, turtles, timed committers
    UNKNOWN: "default",
}


@dataclass
class Switcher:
    """Debounced strategy selection with an asymmetric spine.

    Leaving the spine is slow and returning to it is fast, because the two
    errors do not cost the same. Being boom against a real aggressor is the
    worst cell on the whole grid (0.36); being blitz against an economy costs
    at most 0.16. So the cheap error is the one to make.
    """

    #: The turn-0 spine. Source calibration, and re-measured here: mid-game
    #: switches *into* blitz cannot replay its opening (`opening_end = 50`),
    #: and a blitz that never acted has no chain and no strike stack, so it
    #: restarts from rebuild. Always-boom scores better on the mean (0.815
    #: against 0.787) but strictly worse on the floor (0.36 against 0.50),
    #: and proteus is bought for its floor.
    default: str = "blitz"

    min_confidence: float = 0.55
    """Confidence the classifier must carry before any streak accumulates.
    The economy score crosses this at turn 137 (`EVIDENCE_TURN` 60 plus
    0.55 of the 140-turn ramp)."""

    #: Streak needed to leave the spine. With `min_confidence` this puts the
    #: earliest economy switch at turn ~162 — after boom's own
    #: `avoid_enemy_adjacent_until` (160) and before its `commit_turn` (200),
    #: so the core arrives with its expansion phase intact and its endgame
    #: latch still ahead of it.
    leave_spine_streak: int = 25

    #: Streak needed to come back. Cheap on purpose: a fist that lands while
    #: we are still banking is fatal, and a few needless turns of blitz cost
    #: almost nothing. Six turns is under half the 14-turn rally window blitz
    #: needs to size its first wave, so the return is not too late to matter.
    return_spine_streak: int = 6

    #: Minimum turns between switches, applied only to leaving the spine.
    #: Returning to it is never rate-limited.
    cooldown: int = 60

    current: str = field(default="", init=False)
    label: str = field(default=UNKNOWN, init=False)
    _streak_label: str = field(default="", init=False)
    _streak: int = field(default=0, init=False)
    _last_switch_turn: int = field(default=-(10 ** 9), init=False)
    history: list[tuple[int, str, str]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.current = self.default

    def update(self, cls: Classification, turn: int) -> str:
        """Feed one classification; returns the strategy to play this turn."""
        proposal = COUNTER.get(cls.label, "default")
        if proposal == "default":
            proposal = self.default

        if cls.confidence >= self.min_confidence and proposal != self.current:
            if proposal == self._streak_label:
                self._streak += 1
            else:
                self._streak_label = proposal
                self._streak = 1

            returning = proposal == self.default
            needed = self.return_spine_streak if returning else self.leave_spine_streak
            cooldown = 0 if returning else self.cooldown

            if (
                self._streak >= needed
                and turn - self._last_switch_turn >= cooldown
            ):
                self.current = proposal
                self.label = cls.label
                self._last_switch_turn = turn
                self._streak = 0
                self.history.append((turn, cls.label, proposal))
        else:
            # A proposal that agrees with what we already play resets the
            # clock. That is what makes a single aggressor turn cancel an
            # in-progress drift toward economy.
            self._streak_label = ""
            self._streak = 0
        return self.current
