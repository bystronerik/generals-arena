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


def capture_path(game_id: str, seat: str, directory: Path) -> Path:
    """
    Where a seat's rich capture belongs (see `arena.instrument.capture`).

    Unlike a trace, a capture is written straight here rather than into the
    harness's scratch directory: the capture module gzips it itself, and the
    files are large enough that a copy afterwards is worth avoiding.
    """
    if seat not in SEATS:
        raise ValueError(f"seat must be one of {SEATS} (got {seat!r})")
    return directory / f"{game_id}.capture.{seat}.jsonl.gz"


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

    def capture_destination(self, seat: str) -> Path:
        """Where this game's rich capture for `seat` belongs, if one is armed."""
        return capture_path(self.game_id, seat, self.directory)

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

        Always from `seat`'s point of view, so `land_margin_b` means "B's lead"
        exactly as `land_margin_a` means A's — the same convention the
        engine-truth finals already use.
        """
        index = SEATS.index(seat)
        if name == "land":
            return tuple(f.land[index] for f in self.frames)
        if name == "army":
            return tuple(f.army[index] for f in self.frames)
        if name == "land_margin":
            other = 1 - index
            return tuple(f.land[index] - f.land[other] for f in self.frames)
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


# --- replay -----------------------------------------------------------------
#
# The engine is imported lazily throughout this section: `arena.matches.loop`
# imports this module on every match, recorded or not, and paying a jax import
# to write four dicts a turn would be absurd.


class EraMismatch(RuntimeError):
    """The trajectory was recorded under a different engine than this checkout."""


def require_same_era(traj: Trajectory, *, engine: str | None = None) -> None:
    """
    Refuse to replay across a `competition-module` bump.

    A rules or engine change moves what a state transitions to, so replaying an
    old trajectory under a new engine produces a different game while looking
    like a successful reconstruction. Same reasoning as the rating era
    boundary: records from two eras never pool.
    """
    from arena.records.store import engine_version

    current = engine or engine_version()
    if traj.engine_version != current:
        raise EraMismatch(
            f"{traj.game_id} was recorded under engine {traj.engine_version} but "
            f"this checkout is {current}; check out that submodule revision to "
            f"replay it"
        )


def replay_states(traj: Trajectory) -> Iterator[tuple[int, Any, Any]]:
    """
    Yield `(turn, state, info)` for the recorded game, turn by turn.

    Turn 0 is the starting board with no `info`. This is also the RL
    materializer: `get_observation(state, player)` turns any yielded state into
    either seat's fog view, which is why observations are not stored.
    """
    import jax.numpy as jnp

    from arena.matches.loop import make_board, make_transition  # engine imports
    from generals import GeneralsEnv

    env = GeneralsEnv(mode=str(traj.header["mode"]))
    state = make_board(env, traj.seed)
    transition = make_transition(env)

    yield 0, state, None
    for frame in traj.frames:
        actions = jnp.stack(
            [
                jnp.array(frame.action_a, dtype=jnp.int32),
                jnp.array(frame.action_b, dtype=jnp.int32),
            ]
        )
        state, info = transition(state, actions)
        yield frame.turn, state, info


@dataclass(frozen=True)
class VerifyReport:
    """The outcome of replaying one trajectory against what it recorded."""

    game_id: str
    turns: int
    digests_checked: int
    mismatches: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.mismatches

    def summary(self) -> str:
        if self.ok:
            return (
                f"{self.game_id}: replayed {self.turns} turn(s), scalars and "
                f"{self.digests_checked} digest(s) match"
            )
        return f"{self.game_id}: {len(self.mismatches)} mismatch(es)\n  " + "\n  ".join(
            self.mismatches
        )


def verify_trajectory(traj: Trajectory, *, engine: str | None = None) -> VerifyReport:
    """
    Replay and check every claim the trajectory makes.

    A game reconstructs **iff** all four hold: the scalars match on every turn,
    every recorded digest matches, the turn count matches, and the outcome
    matches. Anything less is a divergence, and the digests say which century
    it started in.
    """
    from arena.matches.loop import winner_seat

    require_same_era(traj, engine=engine)

    mismatches: list[str] = []
    checked = 0
    by_turn = {frame.turn: frame for frame in traj.frames}
    winner_player = -1
    last_turn = 0
    info = None

    for turn, state, info in replay_states(traj):
        if turn == 0:
            continue
        last_turn = turn
        frame = by_turn[turn]
        land = (int(info.land[0]), int(info.land[1]))
        army = (int(info.army[0]), int(info.army[1]))
        if land != frame.land:
            mismatches.append(f"turn {turn}: land {land} != recorded {frame.land}")
        if army != frame.army:
            mismatches.append(f"turn {turn}: army {army} != recorded {frame.army}")
        if turn in traj.digests:
            checked += 1
            got = state_digest(state)
            if got != traj.digests[turn]:
                mismatches.append(f"turn {turn}: state digest differs")
        if bool(info.is_done):
            winner_player = int(info.winner)
            break

    recorded = traj.end
    truncated = winner_player < 0
    replayed = {
        "winner": winner_seat(winner_player, truncated=truncated),
        "turns": last_turn,
        "terminated": winner_player >= 0,
        "truncated": truncated,
    }
    for field_name, value in replayed.items():
        if recorded.get(field_name) != value:
            mismatches.append(
                f"end.{field_name}: {value!r} != recorded {recorded.get(field_name)!r}"
            )

    return VerifyReport(
        game_id=traj.game_id,
        turns=last_turn,
        digests_checked=checked,
        mismatches=tuple(mismatches),
    )


def materialize(traj: Trajectory, path: Path, *, engine: str | None = None) -> Path:
    """
    Write the replayed game as dense per-turn arrays (`.npz`).

    A cache, not a primary: it is regenerable from the trajectory at any time,
    which is exactly why states are not stored in the first place. Layout is
    deliberately unopinionated — the RL work decides its own shard format when
    it starts.
    """
    import numpy as np

    require_same_era(traj, engine=engine)

    armies, ownership, castles = [], [], []
    for _turn, state, _info in replay_states(traj):
        armies.append(np.asarray(state.armies, dtype=np.int16))
        ownership.append(np.asarray(state.ownership, dtype=bool))
        castles.append(np.asarray(state.castles, dtype=bool))

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        armies=np.stack(armies),
        ownership=np.stack(ownership),
        castles=np.stack(castles),
        actions=np.array([[f.action_a, f.action_b] for f in traj.frames], dtype=np.int16),
        land=np.array([f.land for f in traj.frames], dtype=np.int32),
        army=np.array([f.army for f in traj.frames], dtype=np.int32),
        winner=str(traj.end.get("winner", "")),
        game_id=traj.game_id,
        seed=traj.seed,
    )
    return path


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


# --- cli --------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Inspect, verify, or materialize a recorded trajectory "
            "(see docs/arena/trajectories.md)."
        )
    )
    parser.add_argument("trajectory", type=Path, help="path to a .traj.jsonl.gz")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--verify",
        action="store_true",
        help="replay and assert the recorded scalars, digests, and outcome",
    )
    group.add_argument(
        "--replay",
        action="store_true",
        help="replay and print one line per turn",
    )
    group.add_argument(
        "--materialize",
        type=Path,
        metavar="OUT.npz",
        help="replay into dense per-turn arrays (a regenerable cache)",
    )
    args = parser.parse_args(argv)

    traj = read_trajectory(args.trajectory)

    if args.materialize:
        print(f"[trajectories] wrote {materialize(traj, args.materialize)}")
        return 0

    if args.replay:
        require_same_era(traj)
        for turn, _state, info in replay_states(traj):
            if turn == 0:
                continue
            print(f"turn {turn:>5}  land={_ints2(info.land)}  army={_ints2(info.army)}")
        return 0

    if args.verify:
        report = verify_trajectory(traj)
        print(report.summary())
        return 0 if report.ok else 1

    print(
        f"{traj.game_id}: seed={traj.seed} engine={traj.engine_version} "
        f"turns={len(traj.frames)} digests={len(traj.digests)} end={traj.end}"
    )
    return 0


def _ints2(array) -> tuple[int, int]:
    return int(array[0]), int(array[1])


if __name__ == "__main__":
    raise SystemExit(main())
