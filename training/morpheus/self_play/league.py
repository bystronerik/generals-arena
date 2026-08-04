"""Immutable checkpoint league for Morpheus self-play (Part 11).

Snapshots are immutable for the duration of one training epoch. The league
holds the current learner, promoted best, recent snapshots, and measured
exploiters. Mutation of a registered snapshot during a frozen epoch is
rejected.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

ROLES = frozenset({"learner", "best", "recent", "exploiter"})
KIND_CHECKPOINT = "checkpoint"
KIND_STUB = "stub"
KINDS = frozenset({KIND_CHECKPOINT, KIND_STUB})


class LeagueError(RuntimeError):
    """League roster or epoch rules were violated."""


@dataclass(frozen=True)
class Snapshot:
    """One immutable opponent identity for the current epoch."""

    snapshot_id: str
    role: str
    kind: str
    content_digest: str
    path: str | None = None
    weight: float = 1.0
    meta: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise LeagueError(f"unknown role {self.role!r}; expected one of {sorted(ROLES)}")
        if self.kind not in KINDS:
            raise LeagueError(f"unknown kind {self.kind!r}; expected one of {sorted(KINDS)}")
        if float(self.weight) < 0.0:
            raise LeagueError("snapshot weight must be >= 0")
        if not self.snapshot_id:
            raise LeagueError("snapshot_id must be non-empty")
        if not self.content_digest:
            raise LeagueError("content_digest must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["meta"] = dict(self.meta)
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Snapshot:
        meta = data.get("meta") or {}
        if isinstance(meta, dict):
            meta_items = tuple(sorted((str(k), meta[k]) for k in meta))
        else:
            meta_items = tuple(meta)
        return cls(
            snapshot_id=str(data["snapshot_id"]),
            role=str(data["role"]),
            kind=str(data["kind"]),
            content_digest=str(data["content_digest"]),
            path=(str(data["path"]) if data.get("path") is not None else None),
            weight=float(data.get("weight", 1.0)),
            meta=meta_items,
        )


def digest_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def digest_path(path: Path) -> str:
    """Content digest over a checkpoint directory or file."""
    path = Path(path)
    if path.is_file():
        return digest_bytes(path.read_bytes())
    if not path.is_dir():
        raise LeagueError(f"snapshot path missing: {path}")
    parts: list[bytes] = []
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = child.relative_to(path).as_posix().encode("utf-8")
        parts.append(rel)
        parts.append(child.read_bytes())
    return digest_bytes(b"\0".join(parts))


def stub_snapshot(
    *,
    snapshot_id: str,
    role: str,
    weight: float = 1.0,
    stub_tag: str = "uniform-v0",
) -> Snapshot:
    """Deterministic UniformEvaluator stand-in for smoke and unit tests."""
    return Snapshot(
        snapshot_id=snapshot_id,
        role=role,
        kind=KIND_STUB,
        content_digest=digest_bytes(stub_tag.encode("utf-8")),
        path=None,
        weight=float(weight),
        meta=(("stub_tag", stub_tag),),
    )


def checkpoint_snapshot(
    *,
    snapshot_id: str,
    role: str,
    path: Path,
    weight: float = 1.0,
) -> Snapshot:
    path = Path(path)
    return Snapshot(
        snapshot_id=snapshot_id,
        role=role,
        kind=KIND_CHECKPOINT,
        content_digest=digest_path(path),
        path=str(path.resolve()),
        weight=float(weight),
    )


class League:
    """Epoch-scoped roster of immutable snapshots."""

    def __init__(self) -> None:
        self._epoch: int = 0
        self._frozen: bool = False
        self._snapshots: dict[str, Snapshot] = {}

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def frozen(self) -> bool:
        return self._frozen

    def snapshots(self) -> tuple[Snapshot, ...]:
        return tuple(self._snapshots[k] for k in sorted(self._snapshots))

    def get(self, snapshot_id: str) -> Snapshot:
        try:
            return self._snapshots[snapshot_id]
        except KeyError as exc:
            raise LeagueError(f"unknown snapshot {snapshot_id!r}") from exc

    def learner(self) -> Snapshot:
        learners = [s for s in self._snapshots.values() if s.role == "learner"]
        if len(learners) != 1:
            raise LeagueError(
                f"league requires exactly one learner (got {len(learners)})"
            )
        return learners[0]

    def begin_epoch(self) -> int:
        """Open a new epoch. Existing snapshots stay; mutation stays blocked until freeze."""
        self._epoch += 1
        self._frozen = False
        return self._epoch

    def freeze_epoch(self) -> None:
        if not self._snapshots:
            raise LeagueError("cannot freeze an empty league")
        self.learner()  # validate
        self._frozen = True

    def register(self, snapshot: Snapshot) -> Snapshot:
        """Add a snapshot. Rejected when the epoch is frozen or the id already exists."""
        if self._frozen:
            raise LeagueError(
                f"epoch {self._epoch} is frozen; cannot register {snapshot.snapshot_id!r}"
            )
        existing = self._snapshots.get(snapshot.snapshot_id)
        if existing is not None:
            raise LeagueError(
                f"snapshot {snapshot.snapshot_id!r} already registered; "
                "replace it only in a new epoch via replace()"
            )
        self._snapshots[snapshot.snapshot_id] = snapshot
        return snapshot

    def replace(self, snapshot: Snapshot) -> Snapshot:
        """Replace a snapshot id. Rejected while the epoch is frozen."""
        if self._frozen:
            raise LeagueError(
                f"epoch {self._epoch} is frozen; snapshot mutation rejected "
                f"for {snapshot.snapshot_id!r}"
            )
        if snapshot.snapshot_id not in self._snapshots:
            raise LeagueError(
                f"cannot replace unknown snapshot {snapshot.snapshot_id!r}"
            )
        self._snapshots[snapshot.snapshot_id] = snapshot
        return snapshot

    def require_immutable(self, snapshot_id: str, content_digest: str) -> None:
        """Reject when a caller presents a mutated digest for a registered id."""
        snap = self.get(snapshot_id)
        if snap.content_digest != content_digest:
            raise LeagueError(
                f"snapshot {snapshot_id!r} mutated: "
                f"registered {snap.content_digest} != presented {content_digest}"
            )

    def sample_opponent(
        self,
        rng,
        *,
        exclude_ids: Iterable[str] | None = None,
    ) -> Snapshot:
        """Weighted sample over non-excluded snapshots (usually excluding learner)."""
        exclude = set(exclude_ids or ())
        candidates = [
            s for s in self._snapshots.values() if s.snapshot_id not in exclude
        ]
        if not candidates:
            # Fall back to learner versus itself when the league has one entry.
            candidates = list(self._snapshots.values())
        if not candidates:
            raise LeagueError("league has no snapshots to sample")
        weights = [max(float(s.weight), 0.0) for s in candidates]
        total = sum(weights)
        if total <= 0.0:
            weights = [1.0] * len(candidates)
            total = float(len(candidates))
        probs = [w / total for w in weights]
        index = int(rng.choice(len(candidates), p=probs))
        return candidates[index]

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self._epoch,
            "frozen": self._frozen,
            "snapshots": [s.to_dict() for s in self.snapshots()],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> League:
        league = cls()
        league._epoch = int(data.get("epoch") or 0)
        for row in data.get("snapshots") or []:
            snap = Snapshot.from_dict(row)
            league._snapshots[snap.snapshot_id] = snap
        if bool(data.get("frozen")):
            league._frozen = True
        return league

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: Path) -> League:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


def smoke_league(*, stub_tag: str = "uniform-v0") -> League:
    """Minimal frozen league for smoke runs: learner + best + recent + exploiter."""
    league = League()
    league.begin_epoch()
    league.register(
        stub_snapshot(snapshot_id="learner", role="learner", weight=1.0, stub_tag=stub_tag)
    )
    league.register(
        stub_snapshot(snapshot_id="best", role="best", weight=1.0, stub_tag=stub_tag)
    )
    league.register(
        stub_snapshot(snapshot_id="recent-0", role="recent", weight=1.0, stub_tag=stub_tag)
    )
    league.register(
        stub_snapshot(
            snapshot_id="exploiter-0", role="exploiter", weight=1.0, stub_tag=stub_tag
        )
    )
    league.freeze_epoch()
    return league
