"""
Per-tick economy for both seats, from one pass over each frame.

Distances are Manhattan and measured against the *enemy general's true cell*,
which the replay states and a player may never have seen — that is the point:
`nearest_tile_dist` says how close a side ever got, whether or not it knew.
Vision is a separate question, answered in `fog`.

The largest stack is the owned cell holding the most army. Ties keep last
tick's cell when it is still tied, so `path` sees a stack that walks rather
than one that teleports between equal piles.
"""

from __future__ import annotations

from dataclasses import dataclass

from arena.instrument.replay.loader import Cell, Replay


@dataclass(frozen=True)
class PlayerTick:
    """One seat's state at one tick."""

    army: int
    tiles: int
    max_stack: int
    max_stack_pos: Cell | None
    max_stack_dist: int | None
    general_army: int
    nearest_tile_dist: int | None
    nearest_tile_pos: Cell | None
    tiles_gained: int
    tiles_lost: int

    def as_json(self) -> dict:
        return {
            "army": self.army,
            "tiles": self.tiles,
            "max_stack": self.max_stack,
            "max_stack_pos": list(self.max_stack_pos) if self.max_stack_pos else None,
            "max_stack_dist": self.max_stack_dist,
            "general_army": self.general_army,
            "nearest_tile_dist": self.nearest_tile_dist,
            "nearest_tile_pos": list(self.nearest_tile_pos) if self.nearest_tile_pos else None,
            "tiles_gained": self.tiles_gained,
            "tiles_lost": self.tiles_lost,
        }


@dataclass(frozen=True)
class TickMetrics:
    tick: int
    seats: tuple[PlayerTick, PlayerTick]

    def as_json(self) -> dict:
        return {"tick": self.tick, "seats": [seat.as_json() for seat in self.seats]}


def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def compute_metrics(replay: Replay) -> list[TickMetrics]:
    """Walk every frame once, accumulating both seats' counters together."""
    rows, cols = replay.rows, replay.cols
    enemy_general = (replay.generals[1], replay.generals[0])
    out: list[TickMetrics] = []
    last_stack_pos: list[Cell | None] = [None, None]

    for index, frame in enumerate(replay.ticks):
        owners, armies = frame.owners, frame.armies
        previous = replay.ticks[index - 1].owners if index else None

        army = [0, 0]
        tiles = [0, 0]
        best = [0, 0]
        best_pos: list[Cell | None] = [None, None]
        near: list[int | None] = [None, None]
        near_pos: list[Cell | None] = [None, None]
        gained = [0, 0]
        lost = [0, 0]

        for r in range(rows):
            owner_row = owners[r]
            army_row = armies[r]
            previous_row = previous[r] if previous is not None else None
            for c in range(cols):
                owner = owner_row[c]
                if owner >= 0:
                    value = army_row[c]
                    army[owner] += value
                    tiles[owner] += 1
                    if value > best[owner]:
                        best[owner] = value
                        best_pos[owner] = (r, c)
                    target = enemy_general[owner]
                    distance = abs(r - target[0]) + abs(c - target[1])
                    if near[owner] is None or distance < near[owner]:
                        near[owner] = distance
                        near_pos[owner] = (r, c)
                if previous_row is not None:
                    before = previous_row[c]
                    if before != owner:
                        if owner >= 0:
                            gained[owner] += 1
                        if before >= 0:
                            lost[before] += 1

        seats = []
        for player in (0, 1):
            held = last_stack_pos[player]
            if (
                held is not None
                and owners[held[0]][held[1]] == player
                and armies[held[0]][held[1]] == best[player]
            ):
                best_pos[player] = held
            last_stack_pos[player] = best_pos[player]
            general = replay.generals[player]
            position = best_pos[player]
            seats.append(
                PlayerTick(
                    army=army[player],
                    tiles=tiles[player],
                    max_stack=best[player],
                    max_stack_pos=position,
                    max_stack_dist=(
                        manhattan(position, enemy_general[player]) if position else None
                    ),
                    general_army=(
                        armies[general[0]][general[1]]
                        if owners[general[0]][general[1]] == player
                        else 0
                    ),
                    nearest_tile_dist=near[player],
                    nearest_tile_pos=near_pos[player],
                    tiles_gained=gained[player],
                    tiles_lost=lost[player],
                )
            )
        out.append(TickMetrics(tick=index, seats=(seats[0], seats[1])))
    return out


@dataclass(frozen=True)
class SeatSummary:
    """Whole-game extremes for one seat."""

    peak_army: int
    peak_army_tick: int
    peak_tiles: int
    peak_tiles_tick: int
    peak_stack: int
    peak_stack_tick: int
    closest_approach: int | None
    closest_approach_tick: int | None
    final_army: int
    final_tiles: int

    def as_json(self) -> dict:
        return {
            "peak_army": self.peak_army,
            "peak_army_tick": self.peak_army_tick,
            "peak_tiles": self.peak_tiles,
            "peak_tiles_tick": self.peak_tiles_tick,
            "peak_stack": self.peak_stack,
            "peak_stack_tick": self.peak_stack_tick,
            "closest_approach": self.closest_approach,
            "closest_approach_tick": self.closest_approach_tick,
            "final_army": self.final_army,
            "final_tiles": self.final_tiles,
        }


def summarize_seat(metrics: list[TickMetrics], player: int) -> SeatSummary:
    """Extremes over the whole game. The closest approach ignores vision."""
    peak_army = peak_tiles = peak_stack = -1
    peak_army_tick = peak_tiles_tick = peak_stack_tick = 0
    closest: int | None = None
    closest_tick: int | None = None
    for entry in metrics:
        seat = entry.seats[player]
        if seat.army > peak_army:
            peak_army, peak_army_tick = seat.army, entry.tick
        if seat.tiles > peak_tiles:
            peak_tiles, peak_tiles_tick = seat.tiles, entry.tick
        if seat.max_stack > peak_stack:
            peak_stack, peak_stack_tick = seat.max_stack, entry.tick
        if seat.nearest_tile_dist is not None and (
            closest is None or seat.nearest_tile_dist < closest
        ):
            closest, closest_tick = seat.nearest_tile_dist, entry.tick
    last = metrics[-1].seats[player]
    return SeatSummary(
        peak_army=peak_army,
        peak_army_tick=peak_army_tick,
        peak_tiles=peak_tiles,
        peak_tiles_tick=peak_tiles_tick,
        peak_stack=peak_stack,
        peak_stack_tick=peak_stack_tick,
        closest_approach=closest,
        closest_approach_tick=closest_tick,
        final_army=last.army,
        final_tiles=last.tiles,
    )
