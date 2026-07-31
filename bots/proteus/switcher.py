"""Strategy switching with hysteresis.

Holds the current strategy label and decides when the classifier's opinion
is strong and stable enough to switch. Pure logic — offline-testable.
Verbatim port of the generals-bot adaptive switcher.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from proteus.classifier import Classification

#: counter-map: what we play against each opponent archetype.
#:
#: Calibrated on the source's pooled live evidence: blitz is the spine
#: (played from turn 0 — mid-game switches INTO blitz cold-start its
#: opening and measurably lose games); the single allowed economy switch is
#: INTO boom against a genuinely passive turtler; aegis is the defensive
#: posture with its own cheap-in / expensive-out streaks.
COUNTER = {
    "rusher": "blitz",    # mirror the rush; nothing else survives one
    "boomer": "blitz",    # punish greed before it scales
    "citier": "blitz",    # kill the castle program before it compounds
    "turtler": "boom",    # out-economy true passivity; don't feed a keep
    "unknown": "default",
}


@dataclass
class Switcher:
    """Debounced strategy selection.

    A switch happens only when the classifier proposes the same non-current
    counter for ``streak_needed`` consecutive updates with confidence >=
    ``min_confidence``, and at most every ``cooldown`` turns.
    """

    default: str = "blitz"
    min_confidence: float = 0.45
    streak_needed: int = 12
    cooldown: int = 50
    #: Switching INTO the defensive posture is cheap insurance — a rush that
    #: lands while we dither is fatal, while a few turns of needless
    #: turtling cost almost nothing.
    defense_streak: int = 5
    #: Leaving the defensive posture is expensive: rushers strike in waves,
    #: and abandoning defense between waves is how they win.
    leave_defense_streak: int = 60
    defense_strategy: str = "aegis"

    current: str = field(default="", init=False)
    label: str = field(default="unknown", init=False)
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
            if proposal == self.defense_strategy:
                needed = self.defense_streak
            elif self.current == self.defense_strategy:
                needed = self.leave_defense_streak
            else:
                needed = self.streak_needed
            cooldown = 0 if proposal == self.defense_strategy else self.cooldown
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
            self._streak_label = ""
            self._streak = 0
        return self.current
