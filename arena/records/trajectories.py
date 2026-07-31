"""
Per-turn trajectories: what the engine saw, recorded beside the game record.

A trajectory is the seed plus the action sequence plus per-turn scalar series.
It is deliberately **not** a sequence of states: `(seed, engine_version,
actions)` determines every state the game ever had, so storing states would be
caching, not recording — and it is the difference between ~12 KB and ~3 MB a
game. States come back by replay (`--replay`), and either seat's fog view comes
back from a state through `get_observation`.

Three files per recorded game, all gzipped jsonl, all keyed by `game_id`:

    data/trajectories/<round>/<game_id>.traj.jsonl.gz      engine, canonical
    data/trajectories/<round>/<game_id>.trace.a.jsonl.gz   probe, seat A
    data/trajectories/<round>/<game_id>.trace.b.jsonl.gz   probe, seat B

Nothing here touches `GameRecord` or the rating fit: trajectories live in their
own root, are opt-in per run, and are gitignored. See
docs/arena/trajectories.md.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

from arena.paths import REPO_ROOT

TRAJECTORIES_DIR = REPO_ROOT / "data" / "trajectories"

# Bumped when a line's shape changes. The reader refuses anything else rather
# than guessing which fields a file has.
FORMAT_VERSION = 1

# A digest every N turns, plus one at the end. Costs microseconds and turns
# "the replay diverged" into "the replay diverged in this century".
DIGEST_EVERY = 100

_ROUND_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")

SEATS = ("a", "b")


class TrajectoryError(RuntimeError):
    """A trajectory file is unreadable, or of a format this code does not know."""


# --- paths ------------------------------------------------------------------


def round_trajectory_dir(round_name: str, *, root: Path | None = None) -> Path:
    """`data/trajectories/<round>/`, with the same name rule as game records."""
    name = round_name.strip()
    if not name or not _ROUND_NAME_RE.match(name):
        raise ValueError(f"invalid round name {round_name!r}; use letters, digits, . _ -")
    return (root or TRAJECTORIES_DIR) / name


def trajectory_path(game_id: str, directory: Path) -> Path:
    return directory / f"{game_id}.traj.jsonl.gz"


def trace_path(game_id: str, seat: str, directory: Path) -> Path:
    if seat not in SEATS:
        raise ValueError(f"seat must be one of {SEATS} (got {seat!r})")
    return directory / f"{game_id}.trace.{seat}.jsonl.gz"


# --- gzipped jsonl io -------------------------------------------------------


def write_jsonl_gz(lines: Sequence[dict[str, Any]], path: Path) -> Path:
    """
    Write compact gzipped jsonl, atomically.

    `.tmp` then `os.replace`, so a worker killed mid-write leaves either the
    whole trajectory or nothing — never a half-readable one that a later replay
    would silently truncate.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = "".join(json.dumps(line, separators=(",", ":")) + "\n" for line in lines)
    try:
        # mtime=0 so two identical trajectories are byte-identical.
        with gzip.GzipFile(filename="", mode="wb", fileobj=tmp.open("wb"), mtime=0) as gz:
            gz.write(payload.encode("utf-8"))
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def read_jsonl_gz(path: Path) -> Iterator[dict[str, Any]]:
    """Yield each line of a gzipped jsonl file as a dict."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for number, text in enumerate(handle, start=1):
            text = text.strip()
            if not text:
                continue
            try:
                line = json.loads(text)
            except json.JSONDecodeError as exc:
                raise TrajectoryError(f"{path}:{number} is not JSON: {exc}") from exc
            if not isinstance(line, dict):
                raise TrajectoryError(f"{path}:{number} is not an object")
            yield line


def gzip_into_place(source: Path, destination: Path) -> Path | None:
    """
    Compress a runner's plain-jsonl trace into the trajectory directory.

    Returns None when the source is missing or empty — a bot with no probe, or
    one that never got a turn, simply has no trace. The source is removed
    either way.
    """
    if not source.is_file():
        return None
    try:
        text = source.read_text(encoding="utf-8")
    finally:
        source.unlink(missing_ok=True)
    if not text.strip():
        return None

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with gzip.GzipFile(filename="", mode="wb", fileobj=tmp.open("wb"), mtime=0) as gz:
            gz.write(text.encode("utf-8"))
        os.replace(tmp, destination)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return destination


# --- state digest -----------------------------------------------------------


def state_digest(state) -> str:
    """
    sha256 over the fields that define a position: armies, ownership, castles.

    Enough to localize a replay divergence, cheap enough to take every hundred
    turns. Not a state serialization — it identifies a state, it does not
    restore one.
    """
    import numpy as np

    sha = hashlib.sha256()
    for array in (state.armies, state.ownership, state.castles):
        buffer = np.ascontiguousarray(np.asarray(array))
        sha.update(str(buffer.dtype).encode("ascii"))
        sha.update(str(buffer.shape).encode("ascii"))
        sha.update(buffer.tobytes())
    return f"sha256:{sha.hexdigest()}"


# --- recording --------------------------------------------------------------


@dataclass
class TrajectoryRecorder:
    """
    Accumulates one game's engine-side trajectory, then writes it once.

    Per turn this appends a small dict to a list; the single gzip stream is
    paid at match end. That is what keeps recording inside the 2% wall-clock
    budget — a per-turn flush would not.
    """

    game_id: str
    seed: int
    mode: str
    round_name: str
    engine_version: str
    bot_a: str
    bot_b: str
    directory: Path
    # Board dimensions are not known until the loop has built the board, so the
    # caller constructs the recorder and the loop fills these in.
    H: int = 0
    W: int = 0
    digest_every: int = DIGEST_EVERY
    _lines: list[dict[str, Any]] = field(default_factory=list, init=False)
    _finished: bool = field(default=False, init=False)

    def set_dims(self, H: int, W: int) -> None:
        self.H = int(H)
        self.W = int(W)

    def trace_destination(self, seat: str) -> Path:
        """Where this game's probe trace for `seat` belongs."""
        return trace_path(self.game_id, seat, self.directory)

    def header(self) -> dict[str, Any]:
        return {
            "v": FORMAT_VERSION,
            "game_id": self.game_id,
            "seed": self.seed,
            "mode": self.mode,
            "round": self.round_name,
            "engine_version": self.engine_version,
            "bot_a": self.bot_a,
            "bot_b": self.bot_b,
            "H": self.H,
            "W": self.W,
        }

    def record_turn(
        self,
        turn: int,
        action_a: Sequence[int],
        action_b: Sequence[int],
        land: Sequence[int],
        army: Sequence[int],
        *,
        state=None,
    ) -> None:
        """
        Log one stepped turn.

        `action_a`/`action_b` are the decoded actions the engine **applied**,
        invalid moves included: resolving those is the engine's job, and a
        replay that dropped them would not be the same game.
        """
        self._lines.append(
            {
                "t": turn,
                "a": [int(x) for x in action_a],
                "b": [int(x) for x in action_b],
                "land": [int(x) for x in land],
                "army": [int(x) for x in army],
            }
        )
        if state is not None and self.digest_every and turn % self.digest_every == 0:
            self._lines.append({"t": turn, "digest": state_digest(state)})

    def finish(
        self,
        *,
        winner: str,
        turns: int,
        terminated: bool,
        truncated: bool,
        state=None,
    ) -> None:
        """Close the trajectory with the outcome and a final state digest."""
        if state is not None:
            self._lines.append({"t": turns, "digest": state_digest(state)})
        self._lines.append(
            {
                "end": {
                    "winner": winner,
                    "turns": turns,
                    "terminated": terminated,
                    "truncated": truncated,
                }
            }
        )
        self._finished = True

    def write(self) -> Path:
        """Write `<game_id>.traj.jsonl.gz`. Only the worker that played it does."""
        if not self._finished:
            raise TrajectoryError(
                f"trajectory for {self.game_id} was never finished; a file without "
                f"an end line cannot be replayed"
            )
        return write_jsonl_gz(
            [self.header(), *self._lines],
            trajectory_path(self.game_id, self.directory),
        )


# --- reading ----------------------------------------------------------------


@dataclass(frozen=True)
class TurnFrame:
    turn: int
    action_a: tuple[int, ...]
    action_b: tuple[int, ...]
    land: tuple[int, int]
    army: tuple[int, int]


@dataclass(frozen=True)
class Trajectory:
    """One game's recorded engine truth, ready to replay or reduce."""

    header: dict[str, Any]
    frames: tuple[TurnFrame, ...]
    digests: dict[int, str]
    end: dict[str, Any]

    @property
    def game_id(self) -> str:
        return str(self.header["game_id"])

    @property
    def seed(self) -> int:
        return int(self.header["seed"])

    @property
    def engine_version(self) -> str:
        return str(self.header["engine_version"])

    @property
    def actions(self) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
        return tuple((f.action_a, f.action_b) for f in self.frames)

    def series(self, name: str, seat: str) -> tuple[int, ...]:
        """
        A per-turn engine series: `land`, `army`, or `land_margin`.

        `land_margin` is A-perspective on both seats — the margin is one
        quantity, and its sign already says whose it is.
        """
        index = SEATS.index(seat)
        if name == "land":
            return tuple(f.land[index] for f in self.frames)
        if name == "army":
            return tuple(f.army[index] for f in self.frames)
        if name == "land_margin":
            return tuple(f.land[0] - f.land[1] for f in self.frames)
        raise ValueError(f"unknown engine series {name!r}")


def read_trajectory(path: Path) -> Trajectory:
    """Parse a `.traj.jsonl.gz` file, rejecting anything malformed."""
    lines = iter(read_jsonl_gz(path))
    try:
        header = next(lines)
    except StopIteration:
        raise TrajectoryError(f"{path} is empty") from None

    version = header.get("v")
    if version != FORMAT_VERSION:
        raise TrajectoryError(
            f"{path} is trajectory format v{version}; this code reads v{FORMAT_VERSION}"
        )

    frames: list[TurnFrame] = []
    digests: dict[int, str] = {}
    end: dict[str, Any] | None = None
    for line in lines:
        if "end" in line:
            end = line["end"]
            continue
        if "digest" in line:
            digests[int(line["t"])] = str(line["digest"])
            continue
        frames.append(
            TurnFrame(
                turn=int(line["t"]),
                action_a=tuple(int(x) for x in line["a"]),
                action_b=tuple(int(x) for x in line["b"]),
                land=(int(line["land"][0]), int(line["land"][1])),
                army=(int(line["army"][0]), int(line["army"][1])),
            )
        )

    if end is None:
        raise TrajectoryError(f"{path} has no end line; the match did not finish writing")
    return Trajectory(header=header, frames=tuple(frames), digests=digests, end=end)


def read_trace(path: Path) -> dict[str, list[Any]]:
    """
    A seat's probe trace as `{key: [value per recorded turn]}`.

    Keys are collected across the whole trace, so a probe that starts emitting
    a key mid-game yields a series shorter than the game — which is honest:
    the earlier turns did not measure it.
    """
    series: dict[str, list[Any]] = {}
    for line in read_jsonl_gz(path):
        for name, value in line.items():
            if name == "t":
                continue
            series.setdefault(name, []).append(value)
    return series
