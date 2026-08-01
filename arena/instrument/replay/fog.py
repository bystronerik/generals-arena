"""
Reconstruct what each seat could see, tick by tick.

Competition fog: a player sees every cell within Chebyshev distance 1 of a cell
it owns — the tile itself and its eight neighbours. Sight is not memory in the
engine (scouted tiles fade back into fog), but a *bot* can remember, so this
module reports both readings:

- `first_general_sight[p]` — the first tick p's vision covered the enemy
  general's cell, or None if it never did;
- knowability — from that tick to the end of the game the location is known,
  because nothing can move a general. Everything downstream ("did the bot act
  on what it knew?") is measured against the knowable window, not against
  frame-by-frame sight, so a bot is never blamed for looking away.
"""

from __future__ import annotations

from dataclasses import dataclass

from arena.instrument.replay.loader import Cell, Frame, Replay

NEIGHBOURHOOD = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]


def sees_cell(frame: Frame, player: int, cell: Cell, rows: int, cols: int) -> bool:
    """True when `player` owns any cell in the target's 3x3 neighbourhood."""
    r, c = cell
    owners = frame.owners
    for dr, dc in NEIGHBOURHOOD:
        rr, cc = r + dr, c + dc
        if 0 <= rr < rows and 0 <= cc < cols and owners[rr][cc] == player:
            return True
    return False


def visible_enemy_tiles(frame: Frame, player: int, rows: int, cols: int) -> list[Cell]:
    """Enemy-owned cells inside `player`'s vision this tick."""
    owners = frame.owners
    enemy = 1 - player
    out: list[Cell] = []
    for r in range(rows):
        row = owners[r]
        for c in range(cols):
            if row[c] == enemy and sees_cell(frame, player, (r, c), rows, cols):
                out.append((r, c))
    return out


@dataclass(frozen=True)
class Vision:
    """When each seat first laid eyes on the other's general."""

    first_general_sight: tuple[int | None, int | None]

    def knows_general(self, player: int, tick: int) -> bool:
        seen = self.first_general_sight[player]
        return seen is not None and tick >= seen

    def as_json(self) -> dict:
        return {
            "first_general_sight": list(self.first_general_sight),
            "never_seen": [p for p in (0, 1) if self.first_general_sight[p] is None],
        }


def compute_vision(replay: Replay) -> Vision:
    """One 3x3 probe per seat per tick around the opposing general."""
    rows, cols = replay.rows, replay.cols
    first: list[int | None] = [None, None]
    for index, frame in enumerate(replay.ticks):
        for player in (0, 1):
            if first[player] is None and sees_cell(
                frame, player, replay.enemy_general(player), rows, cols
            ):
                first[player] = index
        if first[0] is not None and first[1] is not None:
            break
    return Vision(first_general_sight=(first[0], first[1]))
