"""
Coarse segmentation of a game into expansion / contest / collapse.

Deliberately crude — the point is to give every event a chapter to sit in, not
to model the game:

- **expansion** — tick 0 until first contact. Both sides eat neutral land and
  cannot yet interfere with each other.
- **contest** — first contact until collapse, if a collapse happens. Territory
  changes hands.
- **collapse** — from the first tick after which the eventual loser's tile
  count never again reaches `COLLAPSE_SHARE` of its own peak. On a draw the
  side with fewer final tiles plays the loser; if it never drops that far, or
  drops only in the last `MIN_COLLAPSE_TICKS` frames — a general falling out of
  a game that was still contested — there is no collapse phase and contest runs
  to the end.

A game with no contact at all is one long expansion.
"""

from __future__ import annotations

from dataclasses import dataclass

from arena.instrument.replay.events import EventLog
from arena.instrument.replay.loader import Replay
from arena.instrument.replay.metrics import TickMetrics

COLLAPSE_SHARE = 0.75
MIN_COLLAPSE_TICKS = 5


@dataclass(frozen=True)
class Phase:
    name: str
    start: int
    end: int
    note: str

    def contains(self, tick: int) -> bool:
        return self.start <= tick <= self.end

    def as_json(self) -> dict:
        return {"name": self.name, "start": self.start, "end": self.end, "note": self.note}


def losing_seat(replay: Replay, metrics: list[TickMetrics]) -> int:
    if replay.winner >= 0:
        return 1 - replay.winner
    final = metrics[-1].seats
    return 0 if final[0].tiles <= final[1].tiles else 1


def collapse_start(metrics: list[TickMetrics], loser: int, after: int) -> int | None:
    """First tick from which the loser stays below `COLLAPSE_SHARE` of its peak."""
    peak = max(entry.seats[loser].tiles for entry in metrics)
    floor = COLLAPSE_SHARE * peak
    start: int | None = None
    for entry in metrics:
        if entry.tick < after:
            continue
        if entry.seats[loser].tiles < floor:
            if start is None:
                start = entry.tick
        else:
            start = None
    return start


def segment(replay: Replay, metrics: list[TickMetrics], events: EventLog) -> list[Phase]:
    """Chop the game into at most three phases, always covering every tick."""
    last = metrics[-1].tick
    contact = events.first_contact
    if contact is None:
        return [Phase("expansion", 0, last, "no contact between the two territories")]

    loser = losing_seat(replay, metrics)
    collapse = collapse_start(metrics, loser, contact)
    phases = [Phase("expansion", 0, max(contact - 1, 0), "neutral land only, no contact")]
    if collapse is None or collapse <= contact or collapse > last - MIN_COLLAPSE_TICKS:
        phases.append(Phase("contest", contact, last, "territory contested to the end"))
        return phases

    phases.append(Phase("contest", contact, collapse - 1, "territory contested"))
    phases.append(
        Phase(
            "collapse",
            collapse,
            last,
            f"seat {loser} never recovers to {int(COLLAPSE_SHARE * 100)}% of its peak land",
        )
    )
    return phases


def phase_at(phases: list[Phase], tick: int) -> str:
    for phase in phases:
        if phase.contains(tick):
            return phase.name
    return "?"
