"""
Where the largest stack went, and whether it went at anything.

The replay stores frames, not actions, so movement is inferred: compare the
largest stack's cell between consecutive ticks and classify the transition as

- `hold` — same cell;
- `move` — an adjacent cell, i.e. the pile walked one step;
- `jump` — anywhere else, i.e. some *other* pile became the largest. A jump is
  not movement and is excluded from every fraction below.

**Target definition.** The stack is judged against the best target its owner
could know about at that tick:

1. the enemy general, from the first tick its cell entered the owner's vision
   onwards (a general cannot move, so once seen it stays known);
2. before that, the nearest enemy-owned tile currently inside the owner's
   vision, measured from the stack;
3. otherwise `no target` — the side is playing blind, and the tick counts
   toward neither `toward` nor `away`.

`toward` and `away` compare the Manhattan distance of the old and new cell to
the *same* target, so a target that changes between ticks cannot manufacture
progress. There is no third case: an orthogonal step changes exactly one
coordinate by one, so against a fixed target the distance always moves by ±1.
"""

from __future__ import annotations

from dataclasses import dataclass

from arena.instrument.replay.fog import Vision, sees_cell
from arena.instrument.replay.loader import Cell, Replay
from arena.instrument.replay.metrics import TickMetrics, manhattan

TARGET_GENERAL = "enemy_general"
TARGET_VISIBLE_TILE = "visible_enemy_tile"
TARGET_NONE = "none"


@dataclass(frozen=True)
class StackStep:
    """One tick of the largest stack's life."""

    tick: int
    pos: Cell | None
    army: int
    kind: str
    target: Cell | None
    target_kind: str
    dist: int | None
    delta: int | None

    @property
    def moved(self) -> bool:
        return self.kind == "move"

    @property
    def toward(self) -> bool:
        return self.moved and self.delta is not None and self.delta < 0

    @property
    def away(self) -> bool:
        return self.moved and self.delta is not None and self.delta > 0

    def as_json(self) -> dict:
        return {
            "tick": self.tick,
            "pos": list(self.pos) if self.pos else None,
            "army": self.army,
            "kind": self.kind,
            "target": list(self.target) if self.target else None,
            "target_kind": self.target_kind,
            "dist": self.dist,
            "delta": self.delta,
        }


@dataclass(frozen=True)
class PathSummary:
    """Toward/away accounting over a tick range."""

    label: str
    start: int
    end: int
    moves: int
    toward: int
    away: int
    blind_moves: int

    @property
    def directed(self) -> int:
        return self.toward + self.away

    @property
    def toward_fraction(self) -> float | None:
        return self.toward / self.directed if self.directed else None

    @property
    def away_fraction(self) -> float | None:
        return self.away / self.directed if self.directed else None

    def as_json(self) -> dict:
        return {
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "moves": self.moves,
            "toward": self.toward,
            "away": self.away,
            "blind_moves": self.blind_moves,
            "toward_fraction": self.toward_fraction,
            "away_fraction": self.away_fraction,
        }


def _nearest_visible_enemy_tile(
    replay: Replay, tick: int, player: int, origin: Cell
) -> Cell | None:
    """Closest enemy tile to `origin` that `player` can currently see, if any."""
    frame = replay.ticks[tick]
    rows, cols, enemy = replay.rows, replay.cols, 1 - player
    candidates: list[tuple[int, Cell]] = []
    for r in range(rows):
        row = frame.owners[r]
        for c in range(cols):
            if row[c] == enemy:
                candidates.append((abs(r - origin[0]) + abs(c - origin[1]), (r, c)))
    for _, cell in sorted(candidates):
        if sees_cell(frame, player, cell, rows, cols):
            return cell
    return None


def compute_path(
    replay: Replay, metrics: list[TickMetrics], vision: Vision, player: int
) -> list[StackStep]:
    """One `StackStep` per tick after the first, for `player`'s largest stack."""
    enemy_general = replay.enemy_general(player)
    steps: list[StackStep] = []

    for index in range(1, len(metrics)):
        seat = metrics[index].seats[player]
        before = metrics[index - 1].seats[player]
        position, previous = seat.max_stack_pos, before.max_stack_pos

        if position is None or previous is None:
            kind = "gone"
        elif position == previous:
            kind = "hold"
        elif manhattan(position, previous) == 1:
            kind = "move"
        else:
            kind = "jump"

        target: Cell | None = None
        target_kind = TARGET_NONE
        if position is not None:
            if vision.knows_general(player, index):
                target, target_kind = enemy_general, TARGET_GENERAL
            else:
                target = _nearest_visible_enemy_tile(replay, index, player, position)
                target_kind = TARGET_VISIBLE_TILE if target else TARGET_NONE

        dist = manhattan(position, target) if position and target else None
        delta = None
        if kind == "move" and target is not None and previous is not None:
            delta = dist - manhattan(previous, target)

        steps.append(
            StackStep(
                tick=index,
                pos=position,
                army=seat.max_stack,
                kind=kind,
                target=target,
                target_kind=target_kind,
                dist=dist,
                delta=delta,
            )
        )
    return steps


def summarize_path(
    steps: list[StackStep], label: str = "game", start: int = 0, end: int | None = None
) -> PathSummary:
    """
    Count moves in `[start, end]` by whether they closed on the target.

    A move with no target is `blind` and counts toward neither side; otherwise
    the step closed or it did not, there being no zero delta to allow for.
    """
    last = steps[-1].tick if steps else 0
    end = last if end is None else end
    moves = toward = away = blind = 0
    for step in steps:
        if step.tick < start or step.tick > end or not step.moved:
            continue
        moves += 1
        if step.delta is None:
            blind += 1
        elif step.delta < 0:
            toward += 1
        else:
            away += 1
    return PathSummary(
        label=label,
        start=start,
        end=end,
        moves=moves,
        toward=toward,
        away=away,
        blind_moves=blind,
    )
