"""
Timestamped incidents, inferred from frame-to-frame differences.

Replays store state, not actions, so every event here is a reading of what
changed between two frames. Two inferences are worth stating outright:

- **Castles.** Competition maps start with none; players build them, and a
  standing structure produces one army every other turn. So a castle is found
  by production: an owned cell that gains exactly +1 with no neighbour losing
  army, on a tick that is not one of the every-50 all-cell growth ticks, at
  least `CASTLE_CONFIRMATIONS` times. The build itself is invisible in the
  grids beyond the army it spends, so `castle_built` is timestamped at first
  observed production, a couple of ticks after the action.
- **Gather waves.** The signal we hunt: the largest stack strictly growing
  while it moves, for several ticks running — a bot picking army up and
  carrying it somewhere. Their *absence* is reported explicitly, because a bot
  that never gathers cannot ever threaten a general, and that is a flaw no
  win/loss column shows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from arena.instrument.replay.fog import Vision
from arena.instrument.replay.loader import Cell, Replay
from arena.instrument.replay.metrics import TickMetrics
from arena.instrument.replay.path import StackStep

BIG_CAPTURE_ARMY = 10
GATHER_MIN_TICKS = 5
STALL_MIN_TICKS = 50
STALL_OPPONENT_GAIN = 10
LOSS_STREAK_TICKS = 20
LOSS_STREAK_TILES = 5
CASTLE_CONFIRMATIONS = 2
BULK_GROWTH_PERIOD = 50
BULK_GROWTH_CELLS = 8

ORTHOGONAL = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass(frozen=True)
class Event:
    tick: int
    kind: str
    detail: str
    player: int | None = None
    cell: Cell | None = None
    data: dict = field(default_factory=dict)

    def as_json(self) -> dict:
        return {
            "tick": self.tick,
            "kind": self.kind,
            "player": self.player,
            "cell": list(self.cell) if self.cell else None,
            "detail": self.detail,
            **({"data": self.data} if self.data else {}),
        }


@dataclass
class EventLog:
    """Everything detected, plus the derived facts the report leans on."""

    events: list[Event]
    first_contact: int | None
    first_capture: tuple[int | None, int | None]
    castles: dict[Cell, int]
    gather_waves: tuple[int, int]

    def of_kind(self, kind: str, player: int | None = None) -> list[Event]:
        return [
            e
            for e in self.events
            if e.kind == kind and (player is None or e.player == player)
        ]

    def as_json(self) -> dict:
        return {
            "first_contact": self.first_contact,
            "first_capture": list(self.first_capture),
            "castles": [{"cell": list(c), "built_tick": t} for c, t in self.castles.items()],
            "gather_waves": list(self.gather_waves),
            "events": [e.as_json() for e in self.events],
        }


def _adjacent_to_enemy(replay: Replay, tick_index: int) -> bool:
    """True when any two orthogonally adjacent cells are owned by opposing seats."""
    owners = replay.ticks[tick_index].owners
    for r in range(replay.rows):
        row = owners[r]
        below = owners[r + 1] if r + 1 < replay.rows else None
        for c in range(replay.cols):
            owner = row[c]
            if owner < 0:
                continue
            if c + 1 < replay.cols and row[c + 1] == 1 - owner:
                return True
            if below is not None and below[c] == 1 - owner:
                return True
    return False


def _scan_grid_events(
    replay: Replay, big_capture: int
) -> tuple[list[Event], dict[Cell, int], tuple[int | None, int | None]]:
    """
    One diff pass: ownership changes, big captures, castle production, contact.

    Returns the events it can emit immediately plus the confirmed castle map,
    which needs the whole game before it can be trusted.
    """
    rows, cols = replay.rows, replay.cols
    generals = set(replay.generals)
    events: list[Event] = []
    changes: dict[Cell, list[tuple[int, int, int]]] = {}
    candidates: dict[Cell, list[int]] = {}
    first_capture: list[int | None] = [None, None]

    for index in range(1, len(replay.ticks)):
        before, after = replay.ticks[index - 1], replay.ticks[index]
        bulk = index % BULK_GROWTH_PERIOD == 0
        tick_producers: list[Cell] = []

        for r in range(rows):
            before_owner_row, after_owner_row = before.owners[r], after.owners[r]
            before_army_row, after_army_row = before.armies[r], after.armies[r]
            for c in range(cols):
                was, now = before_owner_row[c], after_owner_row[c]
                if was != now:
                    changes.setdefault((r, c), []).append((index, was, now))
                    if was >= 0 and now >= 0:
                        if first_capture[now] is None:
                            first_capture[now] = index
                        held = before_army_row[c]
                        if held >= big_capture:
                            events.append(
                                Event(
                                    tick=index,
                                    kind="big_capture",
                                    player=now,
                                    cell=(r, c),
                                    detail=f"took a cell holding {held} army",
                                    data={"army": held},
                                )
                            )
                elif now >= 0 and not bulk and (r, c) not in generals:
                    if after_army_row[c] - before_army_row[c] == 1 and not _fed_by_neighbour(
                        before, after, r, c, rows, cols
                    ):
                        tick_producers.append((r, c))

        if len(tick_producers) <= BULK_GROWTH_CELLS:
            for cell in tick_producers:
                candidates.setdefault(cell, []).append(index)

    castles = {
        cell: ticks[0]
        for cell, ticks in candidates.items()
        if len(ticks) >= CASTLE_CONFIRMATIONS
    }
    for cell, built in castles.items():
        owner = replay.ticks[built].owners[cell[0]][cell[1]]
        events.append(
            Event(
                tick=built,
                kind="castle_built",
                player=owner,
                cell=cell,
                detail="castle first produces here (build action is a tick or two earlier)",
            )
        )
        for tick, was, now in changes.get(cell, []):
            if tick > built and now >= 0:
                events.append(
                    Event(
                        tick=tick,
                        kind="castle_captured",
                        player=now,
                        cell=cell,
                        detail=f"castle taken from seat {was}",
                        data={"from": was},
                    )
                )

    for player in (0, 1):
        if first_capture[player] is not None:
            events.append(
                Event(
                    tick=first_capture[player],
                    kind="first_capture",
                    player=player,
                    detail="first enemy tile taken",
                )
            )

    for cell in replay.generals:
        for tick, was, now in changes.get(cell, []):
            if now >= 0 and was != now:
                events.append(
                    Event(
                        tick=tick,
                        kind="general_captured",
                        player=now,
                        cell=cell,
                        detail=f"captured seat {was}'s general",
                        data={"loser": was},
                    )
                )

    return events, castles, (first_capture[0], first_capture[1])


def _fed_by_neighbour(before, after, r: int, c: int, rows: int, cols: int) -> bool:
    """True when some orthogonal neighbour lost army — i.e. army arrived, not grew."""
    for dr, dc in ORTHOGONAL:
        rr, cc = r + dr, c + dc
        if 0 <= rr < rows and 0 <= cc < cols and after.armies[rr][cc] < before.armies[rr][cc]:
            return True
    return False


def find_first_contact(replay: Replay) -> int | None:
    """
    First tick the two territories touch, or None if they never do.

    Adjacency alone is complete: capturing a cell means moving into it from an
    owned neighbour, so the tick before any capture already has the two seats
    adjacent. Taking one argument is also what keeps `batch` and
    `detect_events` from answering this question two different ways.
    """
    for index in range(len(replay.ticks)):
        if _adjacent_to_enemy(replay, index):
            return index
    return None


def find_gather_waves(steps: list[StackStep], player: int, minimum: int) -> list[Event]:
    """Runs of >= `minimum` ticks where the largest stack moved and grew every tick."""
    events: list[Event] = []
    run: list[StackStep] = []
    entry_army = 0

    def flush() -> None:
        if len(run) >= minimum:
            events.append(
                Event(
                    tick=run[0].tick,
                    kind="gather_wave",
                    player=player,
                    cell=run[-1].pos,
                    detail=(
                        f"largest stack grew {entry_army}->{run[-1].army} over "
                        f"{len(run)} moving ticks, ending {run[-1].tick}"
                    ),
                    data={
                        "start": run[0].tick,
                        "end": run[-1].tick,
                        "from_army": entry_army,
                        "to_army": run[-1].army,
                        "end_dist": run[-1].dist,
                        "target_kind": run[-1].target_kind,
                    },
                )
            )
        run.clear()

    previous: StackStep | None = None
    for step in steps:
        if step.moved and previous is not None and step.army > previous.army:
            if not run:
                entry_army = previous.army
            run.append(step)
        else:
            flush()
        previous = step
    flush()
    return events


def find_loss_streaks(
    metrics: list[TickMetrics], player: int, window: int, minimum_loss: int
) -> list[Event]:
    """Maximal runs where the tile count never rises, long and deep enough to matter."""
    events: list[Event] = []
    start = 0
    for index in range(1, len(metrics) + 1):
        rising = index < len(metrics) and (
            metrics[index].seats[player].tiles > metrics[index - 1].seats[player].tiles
        )
        if rising or index == len(metrics):
            end = index - 1
            loss = metrics[start].seats[player].tiles - metrics[end].seats[player].tiles
            if end - start >= window and loss >= minimum_loss:
                events.append(
                    Event(
                        tick=start,
                        kind="tile_loss_streak",
                        player=player,
                        detail=(
                            f"lost {loss} tiles without a single gain, ticks {start}-{end}"
                        ),
                        data={"start": start, "end": end, "tiles_lost": loss},
                    )
                )
            start = index
    return events


def find_stalls(
    metrics: list[TickMetrics],
    player: int,
    window: int,
    opponent_gain: int,
) -> list[Event]:
    """Long stretches where the seat never exceeded its starting size while the enemy grew."""
    events: list[Event] = []
    enemy = 1 - player
    index = 0
    last = len(metrics) - 1
    while index < last:
        base = metrics[index].seats[player].tiles
        end = index
        while end + 1 <= last and metrics[end + 1].seats[player].tiles <= base:
            end += 1
        span = end - index
        gain = metrics[end].seats[enemy].tiles - metrics[index].seats[enemy].tiles
        if span > window and gain >= opponent_gain:
            events.append(
                Event(
                    tick=index,
                    kind="expansion_stall",
                    player=player,
                    detail=(
                        f"stuck at <= {base} tiles for {span} ticks "
                        f"(ticks {index}-{end}) while the opponent gained {gain}"
                    ),
                    data={
                        "start": index,
                        "end": end,
                        "tiles": base,
                        "opponent_gain": gain,
                    },
                )
            )
        index = max(end, index + 1)
    return events


def detect_events(
    replay: Replay,
    metrics: list[TickMetrics],
    vision: Vision,
    steps: tuple[list[StackStep], list[StackStep]],
    big_capture: int = BIG_CAPTURE_ARMY,
    gather_ticks: int = GATHER_MIN_TICKS,
    stall_ticks: int = STALL_MIN_TICKS,
) -> EventLog:
    """Run every detector and return one chronological log."""
    events, castles, first_capture = _scan_grid_events(replay, big_capture)
    contact = find_first_contact(replay)
    if contact is not None:
        events.append(
            Event(tick=contact, kind="first_contact", detail="the two territories touch")
        )

    waves = [0, 0]
    for player in (0, 1):
        seen = vision.first_general_sight[player]
        if seen is not None:
            events.append(
                Event(
                    tick=seen,
                    kind="first_general_sight",
                    player=player,
                    cell=replay.enemy_general(player),
                    detail="enemy general enters vision — knowable from here on",
                )
            )
        found = find_gather_waves(steps[player], player, gather_ticks)
        waves[player] = len(found)
        events.extend(found)
        events.extend(
            find_loss_streaks(metrics, player, LOSS_STREAK_TICKS, LOSS_STREAK_TILES)
        )
        events.extend(find_stalls(metrics, player, stall_ticks, STALL_OPPONENT_GAIN))

    ending = (
        f"seat {replay.winner} ({replay.name(replay.winner)}) wins"
        if replay.winner >= 0
        else "draw"
    )
    events.append(
        Event(tick=replay.total_ticks, kind="game_end", detail=ending)
    )

    events.sort(key=lambda e: (e.tick, e.kind, e.player if e.player is not None else -1))
    return EventLog(
        events=events,
        first_contact=contact,
        first_capture=first_capture,
        castles=castles,
        gather_waves=(waves[0], waves[1]),
    )
