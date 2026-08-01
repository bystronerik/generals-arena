"""
Find and load one scraped replay into typed frames. Read-only.

A replay lives at `competition-replays/<player>/{win,lose,draw}/<id>.json`, with
the list-endpoint row beside it as `<id>.meta.json`.

**The folder is not the queried player's result.** It is side A's, and the
queried player is side A only about half the time — 846 of the 1680 pairs on
disk at 2026-08-01 are side B, and 835 of those sit in a folder that states the
opposite of what actually happened to them. So `Replay.folder` is provenance
only and `Replay.outcome` is derived here from `winner` and `seat_of`, which is
the one source that cannot be filed wrong. Never filter or count on the folder.

Two fields of the raw format are not usable as written:

- `castles` was empty in every replay of the 2026-08-01 pull, and the rules say
  maps start with no castles at all — players build them mid-game. Starting
  structures are therefore derived from tick 0 (`initial_structures`, empty in
  practice) and built castles are inferred from production in `events`.
- `total_ticks <= 1` is a forfeit, not a played game. `Replay.is_forfeit` flags
  it; callers filter rather than analysing one frame of noise.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from arena.paths import REPO_ROOT

REPLAYS_DIR = REPO_ROOT / "competition-replays"
OUTCOMES = ("win", "lose", "draw")
FORFEIT_TICKS = 1

Cell = tuple[int, int]


class ReplayNotFound(FileNotFoundError):
    """No replay with that id under the player's win/lose/draw folders."""


@dataclass(frozen=True)
class Frame:
    """One tick: parallel `rows x cols` grids. `owners` is -1 for neutral."""

    armies: list[list[int]]
    owners: list[list[int]]


@dataclass(frozen=True)
class Meta:
    """
    The list-endpoint row.

    `winner` is `A`/`B`/`D` against `a_name`/`b_name`, neither of which is
    reliably the queried player — see the module docstring.
    """

    match_id: str
    a_name: str | None
    b_name: str | None
    a_side: int | None
    created_at: str | None
    seed: int | None
    turns: int | None
    winner: str | None

    @classmethod
    def parse(cls, match_id: str, raw: dict[str, Any]) -> Meta:
        return cls(
            match_id=str(raw.get("id", match_id)),
            a_name=raw.get("a_name"),
            b_name=raw.get("b_name"),
            a_side=raw.get("a_side"),
            created_at=raw.get("created_at"),
            seed=raw.get("seed"),
            turns=raw.get("turns"),
            winner=raw.get("winner"),
        )


@dataclass(frozen=True)
class Replay:
    """One finished leaderboard game, plus where it was found."""

    path: Path
    match_id: str
    queried_player: str
    folder: str
    version: int
    rows: int
    cols: int
    players: tuple[str, str]
    seed: int | None
    mountains: frozenset[Cell]
    generals: tuple[Cell, Cell]
    ticks: tuple[Frame, ...]
    winner: int
    total_ticks: int
    meta: Meta | None

    @property
    def outcome(self) -> str:
        """
        `win` / `lose` / `draw` for the queried player, from the replay itself.

        Not from `folder`, which records side A's result and disagrees with
        this for essentially every side-B game.
        """
        if self.winner < 0:
            return "draw"
        return "win" if self.winner == self.seat_of(self.queried_player) else "lose"

    @property
    def folder_disagrees(self) -> bool:
        """The scraper filed this under someone else's result."""
        return self.folder != self.outcome

    @property
    def is_forfeit(self) -> bool:
        return self.total_ticks <= FORFEIT_TICKS

    @property
    def is_self_match(self) -> bool:
        """Same account on both seats — the outcome label names a seat, not an agent."""
        return self.players[0] == self.players[1]

    def enemy_general(self, player: int) -> Cell:
        return self.generals[1 - player]

    def name(self, player: int) -> str:
        return self.players[player]

    def seat_of(self, player_name: str) -> int:
        """
        The queried player's seat index.

        By name against `players`; on a self-match both names match, so the
        metadata's `a_side` decides. Seat 0 is the fallback of last resort.
        """
        if not self.is_self_match:
            for index, name in enumerate(self.players):
                if name == player_name:
                    return index
        if self.meta is not None and self.meta.a_side in (0, 1):
            return int(self.meta.a_side)
        return 0


def player_dir(player: str, root: Path = REPLAYS_DIR) -> Path:
    return root / player.replace("/", "_")


def iter_replay_paths(
    player: str, folder: str = "all", root: Path = REPLAYS_DIR
) -> Iterator[tuple[str, Path]]:
    """
    Yield `(folder, path)` for a player's replays, id-ordered, without loading them.

    Selection here is by directory, which is side A's result — selecting on the
    queried player's own outcome means opening the file first. See `batch`.
    """
    wanted = OUTCOMES if folder == "all" else (folder,)
    base = player_dir(player, root)
    for name in wanted:
        directory = base / name
        if not directory.is_dir():
            continue
        paths = [p for p in directory.glob("*.json") if not p.name.endswith(".meta.json")]
        for path in sorted(paths, key=lambda p: _sort_key(p.stem)):
            yield name, path


def _sort_key(stem: str) -> tuple[int, str]:
    return (int(stem), "") if stem.isdigit() else (1 << 62, stem)


def find_replay(player: str, match_id: str, root: Path = REPLAYS_DIR) -> tuple[str, Path]:
    """Locate `<match_id>.json` across the player's outcome folders."""
    base = player_dir(player, root)
    for name in OUTCOMES:
        candidate = base / name / f"{match_id}.json"
        if candidate.is_file():
            return name, candidate
    if not base.is_dir():
        raise ReplayNotFound(f"no replays for {player!r} under {base}")
    raise ReplayNotFound(f"no replay {match_id!r} under {base}/{{{','.join(OUTCOMES)}}}")


def load_replay(
    path: Path, queried_player: str, folder: str, match_id: str | None = None
) -> Replay:
    """Parse a replay file and its metadata sidecar into a `Replay`."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    identifier = match_id or path.stem
    meta_path = path.with_name(f"{identifier}.meta.json")
    meta = None
    if meta_path.is_file():
        meta = Meta.parse(identifier, json.loads(meta_path.read_text(encoding="utf-8")))

    dims = raw["dims"]
    generals = [tuple(cell) for cell in raw["generals"]]
    players = list(raw["players"])
    return Replay(
        path=path,
        match_id=identifier,
        queried_player=queried_player,
        folder=folder,
        version=int(raw.get("version", 1)),
        rows=int(dims["rows"]),
        cols=int(dims["cols"]),
        players=(str(players[0]), str(players[1])),
        seed=raw.get("seed"),
        mountains=frozenset((int(r), int(c)) for r, c in raw.get("mountains", [])),
        generals=(generals[0], generals[1]),
        ticks=tuple(Frame(armies=f["armies"], owners=f["owners"]) for f in raw["ticks"]),
        winner=int(raw.get("winner", -1)),
        total_ticks=int(raw.get("total_ticks", len(raw["ticks"]) - 1)),
        meta=meta,
    )


def open_replay(player: str, match_id: str, root: Path = REPLAYS_DIR) -> Replay:
    """`find_replay` + `load_replay`, the usual entry point."""
    folder, path = find_replay(player, match_id, root)
    return load_replay(path, player, folder, match_id)


def initial_structures(replay: Replay) -> list[Cell]:
    """
    Neutral cells holding army at tick 0 and not a general.

    The rules say competition maps ship with no castles, and the raw `castles`
    field is empty in practice, so this is expected to be empty — it exists so
    a map generator that ever changes shows up as data rather than as a wrong
    castle count.
    """
    frame = replay.ticks[0]
    generals = set(replay.generals)
    return [
        (r, c)
        for r in range(replay.rows)
        for c in range(replay.cols)
        if frame.owners[r][c] == -1 and frame.armies[r][c] > 0 and (r, c) not in generals
    ]
