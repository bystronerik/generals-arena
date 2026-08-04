"""Self-play training shard schema (Part 11).

Shards are replay-verifiable training artifacts. They never enter data/games/
or the rating fit. Retention path: data/morpheus/self_play/ (gitignored) or an
explicit --output directory / Modal Volume.
"""

from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

SHARD_FORMAT_VERSION = 1
SHARD_SUFFIX = ".shard.jsonl.gz"


@dataclass(frozen=True)
class SparsePolicy:
    """Normalized root average strategy over candidate action indices."""

    indices: tuple[int, ...]
    probs: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.indices) != len(self.probs):
            raise ValueError("policy indices and probs length mismatch")
        if self.indices and abs(sum(self.probs) - 1.0) > 1e-5:
            raise ValueError("policy probs must sum to 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "indices": [int(x) for x in self.indices],
            "probs": [float(x) for x in self.probs],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> SparsePolicy | None:
        if data is None:
            return None
        return cls(
            indices=tuple(int(x) for x in data["indices"]),
            probs=tuple(float(x) for x in data["probs"]),
        )


@dataclass(frozen=True)
class TruthTargets:
    """Engine-truth labels for one turn (seat-independent board fields)."""

    ownership_digest: str
    armies_digest: str
    generals_digest: str
    castles_digest: str
    land: tuple[int, int]
    army: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ownership_digest": self.ownership_digest,
            "armies_digest": self.armies_digest,
            "generals_digest": self.generals_digest,
            "castles_digest": self.castles_digest,
            "land": list(self.land),
            "army": list(self.army),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TruthTargets:
        return cls(
            ownership_digest=str(data["ownership_digest"]),
            armies_digest=str(data["armies_digest"]),
            generals_digest=str(data["generals_digest"]),
            castles_digest=str(data["castles_digest"]),
            land=(int(data["land"][0]), int(data["land"][1])),
            army=(int(data["army"][0]), int(data["army"][1])),
        )


@dataclass(frozen=True)
class ShardTurn:
    turn: int
    action_a: tuple[int, ...]
    action_b: tuple[int, ...]
    policy_a: SparsePolicy | None
    policy_b: SparsePolicy | None
    truth: TruthTargets
    state_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "t": self.turn,
            "a": list(self.action_a),
            "b": list(self.action_b),
            "policy_a": None if self.policy_a is None else self.policy_a.to_dict(),
            "policy_b": None if self.policy_b is None else self.policy_b.to_dict(),
            "truth": self.truth.to_dict(),
            "digest": self.state_digest,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ShardTurn:
        return cls(
            turn=int(data["t"]),
            action_a=tuple(int(x) for x in data["a"]),
            action_b=tuple(int(x) for x in data["b"]),
            policy_a=SparsePolicy.from_dict(data.get("policy_a")),
            policy_b=SparsePolicy.from_dict(data.get("policy_b")),
            truth=TruthTargets.from_dict(data["truth"]),
            state_digest=str(data["digest"]),
        )


@dataclass
class SelfPlayShard:
    """One finished self-play or panel game with replayable actions and targets."""

    game_id: str
    seed: int
    engine_version: str
    source: str
    learner_seat: int
    seats: tuple[dict[str, Any], dict[str, Any]]
    H: int
    W: int
    turns: list[ShardTurn] = field(default_factory=list)
    winner: str = "draw"
    terminated: bool = False
    truncated: bool = True
    value_targets: tuple[float, float] = (0.0, 0.0)
    format_version: int = SHARD_FORMAT_VERSION
    recursive_opponent_particles: bool = False
    notes: list[str] = field(default_factory=list)

    def to_header(self) -> dict[str, Any]:
        return {
            "v": self.format_version,
            "game_id": self.game_id,
            "seed": self.seed,
            "mode": "competition",
            "engine_version": self.engine_version,
            "source": self.source,
            "learner_seat": self.learner_seat,
            "seats": list(self.seats),
            "H": self.H,
            "W": self.W,
            "recursive_opponent_particles": self.recursive_opponent_particles,
            "notes": list(self.notes),
        }

    def to_end(self) -> dict[str, Any]:
        return {
            "end": {
                "winner": self.winner,
                "turns": len(self.turns),
                "terminated": self.terminated,
                "truncated": self.truncated,
                "value_targets": list(self.value_targets),
            }
        }

    def lines(self) -> list[dict[str, Any]]:
        return [self.to_header(), *[t.to_dict() for t in self.turns], self.to_end()]

    @classmethod
    def from_lines(cls, lines: Sequence[Mapping[str, Any]]) -> SelfPlayShard:
        if not lines:
            raise ValueError("empty shard")
        header = lines[0]
        if int(header.get("v", -1)) != SHARD_FORMAT_VERSION:
            raise ValueError(
                f"unsupported shard format v{header.get('v')}; "
                f"expected v{SHARD_FORMAT_VERSION}"
            )
        end: dict[str, Any] | None = None
        turns: list[ShardTurn] = []
        for line in lines[1:]:
            if "end" in line:
                end = dict(line["end"])
                continue
            turns.append(ShardTurn.from_dict(line))
        if end is None:
            raise ValueError(f"shard {header.get('game_id')} has no end line")
        values = end.get("value_targets") or [0.0, 0.0]
        seats = tuple(header["seats"])  # type: ignore[assignment]
        if len(seats) != 2:
            raise ValueError("shard seats must have length 2")
        return cls(
            game_id=str(header["game_id"]),
            seed=int(header["seed"]),
            engine_version=str(header["engine_version"]),
            source=str(header["source"]),
            learner_seat=int(header["learner_seat"]),
            seats=(dict(seats[0]), dict(seats[1])),
            H=int(header["H"]),
            W=int(header["W"]),
            turns=turns,
            winner=str(end["winner"]),
            terminated=bool(end["terminated"]),
            truncated=bool(end["truncated"]),
            value_targets=(float(values[0]), float(values[1])),
            format_version=int(header["v"]),
            recursive_opponent_particles=bool(
                header.get("recursive_opponent_particles", False)
            ),
            notes=list(header.get("notes") or []),
        )


def shard_path(game_id: str, directory: Path) -> Path:
    return Path(directory) / f"{game_id}{SHARD_SUFFIX}"


def write_shard(shard: SelfPlayShard, directory: Path) -> Path:
    """Atomic gzipped jsonl write under directory (never data/games/)."""
    directory = Path(directory)
    if "data/games" in directory.as_posix() or directory.name == "games":
        # Hard guard: self-play shards must not land in rated storage.
        raise ValueError(
            f"refusing to write self-play shard under rated path {directory}"
        )
    directory.mkdir(parents=True, exist_ok=True)
    path = shard_path(shard.game_id, directory)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = "".join(
        json.dumps(line, separators=(",", ":")) + "\n" for line in shard.lines()
    )
    try:
        with gzip.GzipFile(filename="", mode="wb", fileobj=tmp.open("wb"), mtime=0) as gz:
            gz.write(payload.encode("utf-8"))
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def read_shard(path: Path) -> SelfPlayShard:
    path = Path(path)
    lines: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for number, text in enumerate(handle, start=1):
            text = text.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number} is not JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number} is not an object")
            lines.append(row)
    return SelfPlayShard.from_lines(lines)


def iter_shards(directory: Path) -> Iterator[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        return iter(())
    return iter(sorted(directory.glob(f"*{SHARD_SUFFIX}")))


def value_targets_from_winner(winner: str) -> tuple[float, float]:
    """Seat-0 / seat-1 WDL targets: win=+1, draw=0, loss=-1."""
    if winner == "a":
        return (1.0, -1.0)
    if winner == "b":
        return (-1.0, 1.0)
    if winner == "draw":
        return (0.0, 0.0)
    raise ValueError(f"unknown winner {winner!r}")
