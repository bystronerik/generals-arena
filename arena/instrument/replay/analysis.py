"""
Run the whole pipeline over one replay and hold the result.

`Analysis.us` is the seat of the player the replay folder belongs to — matched
by name against the replay's `players`, falling back to the metadata's `a_side`
on a self-match, where both names are identical and the name says nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

from arena.instrument.replay.events import EventLog, detect_events
from arena.instrument.replay.fog import Vision, compute_vision
from arena.instrument.replay.loader import Replay
from arena.instrument.replay.metrics import (
    SeatSummary,
    TickMetrics,
    compute_metrics,
    summarize_seat,
)
from arena.instrument.replay.path import PathSummary, StackStep, compute_path, summarize_path
from arena.instrument.replay.phases import Phase, segment


class ForfeitReplay(ValueError):
    """The game ended at tick <= 1 — a forfeit, with nothing to analyse."""


@dataclass(frozen=True)
class Analysis:
    replay: Replay
    us: int
    them: int
    metrics: list[TickMetrics]
    vision: Vision
    steps: tuple[list[StackStep], list[StackStep]]
    events: EventLog
    phases: list[Phase]
    summaries: tuple[SeatSummary, SeatSummary]

    def seat_name(self, player: int) -> str:
        return self.replay.name(player)

    def path_summary(self, player: int) -> PathSummary:
        return summarize_path(self.steps[player])

    def phase_summaries(self, player: int) -> list[PathSummary]:
        return [
            summarize_path(self.steps[player], phase.name, phase.start, phase.end)
            for phase in self.phases
        ]


def analyze(replay: Replay, allow_forfeit: bool = False) -> Analysis:
    """Metrics, fog, paths, events, phases — in dependency order."""
    if replay.is_forfeit and not allow_forfeit:
        raise ForfeitReplay(
            f"{replay.match_id} ended at tick {replay.total_ticks}: a forfeit, not a played game"
        )
    us = replay.seat_of(replay.queried_player)
    metrics = compute_metrics(replay)
    vision = compute_vision(replay)
    steps = (
        compute_path(replay, metrics, vision, 0),
        compute_path(replay, metrics, vision, 1),
    )
    events = detect_events(replay, metrics, vision, steps)
    phases = segment(replay, metrics, events)
    return Analysis(
        replay=replay,
        us=us,
        them=1 - us,
        metrics=metrics,
        vision=vision,
        steps=steps,
        events=events,
        phases=phases,
        summaries=(summarize_seat(metrics, 0), summarize_seat(metrics, 1)),
    )
