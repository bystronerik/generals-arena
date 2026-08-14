"""
Hand-built stand-ins for scraped leaderboard replays.

A real replay is ~0.6 MB and the suite budget is 15 s (AGENTS.md, "Test suite
budget"), so
no test opens `competition-replays/`. These builders emit the same JSON shape
as the scraper (docs/engine/leaderboard-replays.md) on boards small enough that
every expected number can be counted by hand.

Only the fields the analysis reads are simulated. Army is *not* conserved
across a `set()`: fixtures place tiles where the test needs them, which is
fine because nothing under `arena/instrument/replay/` checks conservation.
`march()` does follow the real move rule, so the growing-stack patterns the
detectors hunt for look the way they do in a real game.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

NEUTRAL = -1
Cell = tuple[int, int]


class GameBuilder:
    """
    Mutate the board, `commit()` a tick, repeat.

    Nothing is committed by the constructor: place whatever tick 0 needs, then
    call `commit()` for the initial frame.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        generals: list[Cell],
        players: tuple[str, str] = ("us_bot", "them_bot"),
        general_army: int = 1,
        mountains: tuple[Cell, ...] = (),
        seed: int = 7,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.generals = [tuple(cell) for cell in generals]
        self.players = tuple(players)
        self.mountains = [tuple(cell) for cell in mountains]
        self.seed = seed
        self.owners = [[NEUTRAL] * cols for _ in range(rows)]
        self.armies = [[0] * cols for _ in range(rows)]
        self.frames: list[dict[str, Any]] = []
        for seat, cell in enumerate(self.generals):
            self.set(cell, seat, general_army)

    def set(self, cell: Cell, owner: int, army: int) -> GameBuilder:
        r, c = cell
        self.owners[r][c] = owner
        self.armies[r][c] = army
        return self

    def grow(self, cell: Cell, by: int = 1) -> GameBuilder:
        r, c = cell
        self.armies[r][c] += by
        return self

    def march(self, src: Cell, dst: Cell, leave: int = 1) -> GameBuilder:
        """Move all but `leave` army from `src` to `dst`, resolving the landing."""
        (sr, sc), (dr, dc) = src, dst
        mover = self.owners[sr][sc]
        moving = self.armies[sr][sc] - leave
        self.set(src, mover, leave)
        if self.owners[dr][dc] == mover:
            self.set(dst, mover, self.armies[dr][dc] + moving)
        elif moving > self.armies[dr][dc]:
            self.set(dst, mover, moving - self.armies[dr][dc])
        else:
            self.set(dst, self.owners[dr][dc], self.armies[dr][dc] - moving)
        return self

    def transfer_all(self, loser: int, winner: int) -> GameBuilder:
        """Every tile the loser held changes hands — what a general capture does."""
        for r in range(self.rows):
            for c in range(self.cols):
                if self.owners[r][c] == loser:
                    self.owners[r][c] = winner
        return self

    def commit(self, times: int = 1) -> GameBuilder:
        for _ in range(times):
            self.frames.append(
                {
                    "armies": [row[:] for row in self.armies],
                    "owners": [row[:] for row in self.owners],
                }
            )
        return self

    def raw(self, winner: int = -1, total_ticks: int | None = None) -> dict[str, Any]:
        return {
            "version": 1,
            "dims": {"rows": self.rows, "cols": self.cols},
            "players": list(self.players),
            "seed": self.seed,
            "mountains": [list(cell) for cell in self.mountains],
            "castles": [],
            "generals": [list(cell) for cell in self.generals],
            "ticks": self.frames,
            "winner": winner,
            "total_ticks": len(self.frames) - 1 if total_ticks is None else total_ticks,
        }


def meta_row(
    match_id: str,
    a_name: str = "us_bot",
    b_name: str = "them_bot",
    a_side: int = 0,
    winner: str = "A",
    turns: int = 18,
) -> dict[str, Any]:
    return {
        "id": match_id,
        "a_name": a_name,
        "b_name": b_name,
        "a_side": a_side,
        "created_at": "2026-08-01T00:00:00Z",
        "seed": 7,
        "turns": turns,
        "winner": winner,
    }


def write_game(
    root: Path,
    player: str,
    outcome: str,
    match_id: str,
    raw: dict[str, Any],
    meta: dict[str, Any] | None = None,
) -> Path:
    """Lay a replay out the way the scraper does: `<player>/<outcome>/<id>.json`."""
    directory = Path(root) / player / outcome
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{match_id}.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    if meta is not None:
        (directory / f"{match_id}.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return path


def attack_game(players: tuple[str, str] = ("us_bot", "them_bot")) -> dict[str, Any]:
    """
    19 ticks on 6x6. Seat 0's general sits at (5,0), seat 1's at (0,5).

    Hand-computable landmarks:

    | tick | what happens |
    | --- | --- |
    | 2-5 | seat 0 eats column 0 upward, seat 1 eats column 5 downward |
    | 6 | seat 0's scout lands on (1,4): first contact *and* first sight of (0,5) |
    | 9-13 | the gather wave: the stack grows 9 -> 14 over exactly 5 moving ticks |
    | 14-17 | it keeps walking row 0 but shrinks onto neutral land: no second wave |
    | 18 | it takes the general at (0,5); seat 1's four tiles transfer |
    """
    g = GameBuilder(6, 6, generals=[(5, 0), (0, 5)], players=players)
    g.commit()  # t0: two generals holding 1 each
    g.grow((5, 0)).grow((0, 5)).commit()  # t1
    g.grow((5, 0)).grow((0, 5)).set((4, 0), 0, 2).set((1, 5), 1, 1).commit()  # t2
    g.grow((5, 0)).grow((0, 5)).set((3, 0), 0, 2).set((2, 5), 1, 1).commit()  # t3
    g.grow((5, 0)).grow((0, 5)).set((2, 0), 0, 2).set((3, 5), 1, 1).commit()  # t4
    g.grow((5, 0)).set((1, 0), 0, 2).commit()  # t5: seat 1's general stops at 5
    g.grow((5, 0)).set((1, 4), 0, 1).commit()  # t6: contact + sight
    g.grow((5, 0)).set((0, 0), 0, 2).commit()  # t7
    g.grow((5, 0)).commit()  # t8: the general holds 9, the biggest pile on the board
    walk = (
        ((5, 0), (4, 0)),
        ((4, 0), (3, 0)),
        ((3, 0), (2, 0)),
        ((2, 0), (1, 0)),
        ((1, 0), (0, 0)),
        ((0, 0), (0, 1)),
        ((0, 1), (0, 2)),
        ((0, 2), (0, 3)),
        ((0, 3), (0, 4)),
    )
    for src, dst in walk:  # t9..t17
        g.march(src, dst).commit()
    g.march((0, 4), (0, 5)).transfer_all(loser=1, winner=0).commit()  # t18
    return g.raw(winner=0)


def blind_game(players: tuple[str, str] = ("us_bot", "them_bot")) -> dict[str, Any]:
    """
    8 ticks on 6x6. Seat 0 pokes at its own corner and nothing else: it never
    touches seat 1, never sees (0,5), and never moves its biggest stack. Seat 1
    wins on land. The flaw signature `batch` is built to count.
    """
    g = GameBuilder(6, 6, generals=[(5, 0), (0, 5)], players=players)
    g.commit()  # t0
    for _ in range(3):  # t1..t3
        g.grow((5, 0)).grow((0, 5)).commit()
    g.set((4, 0), 0, 1).set((1, 5), 1, 1).commit()  # t4
    g.set((4, 1), 0, 1).set((2, 5), 1, 1).commit()  # t5
    g.set((3, 5), 1, 1).commit()  # t6
    g.set((3, 4), 1, 1).commit()  # t7
    return g.raw(winner=1)


def forfeit_game(players: tuple[str, str] = ("us_bot", "them_bot")) -> dict[str, Any]:
    """`total_ticks == 1`: a scoring artefact, not a played game."""
    g = GameBuilder(6, 6, generals=[(5, 0), (0, 5)], players=players)
    g.commit(2)
    return g.raw(winner=1)
