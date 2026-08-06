"""Deterministic replay-buffer reader for Part 14 trainer samples.

Samples are seat-local tensors with targets. They live under a training volume
path (never ``data/games/``). Window and class-balance policy come from the
run config — this module only stores and samples.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from training.morpheus.objective.targets import SeatTargets
from training.morpheus.trainer.sample import TrainSample

SAMPLE_SUFFIX = ".sample.npz"
INDEX_NAME = "buffer_index.json"


class BufferError(ValueError):
    """Replay buffer is missing, corrupt, or empty."""


@dataclass(frozen=True)
class BufferCursor:
    """Consumed-shard cursor persisted inside a checkpoint."""

    next_index: int
    consumed_ids: tuple[str, ...]
    epoch: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "next_index": int(self.next_index),
            "consumed_ids": list(self.consumed_ids),
            "epoch": int(self.epoch),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> BufferCursor:
        raw = dict(data or {})
        return cls(
            next_index=int(raw.get("next_index") or 0),
            consumed_ids=tuple(str(x) for x in (raw.get("consumed_ids") or ())),
            epoch=int(raw.get("epoch") or 0),
        )


@dataclass
class ReplayBuffer:
    """Ordered window of TrainSample rows with deterministic sampling."""

    samples: list[TrainSample]
    sample_ids: list[str]
    class_ids: list[str]
    window_size: int

    def __post_init__(self) -> None:
        if len(self.samples) != len(self.sample_ids) or len(self.samples) != len(
            self.class_ids
        ):
            raise BufferError("samples, sample_ids, and class_ids length mismatch")
        if self.window_size < 1:
            raise BufferError("window_size must be >= 1")
        if len(self.samples) > self.window_size:
            self.samples = self.samples[-self.window_size :]
            self.sample_ids = self.sample_ids[-self.window_size :]
            self.class_ids = self.class_ids[-self.window_size :]

    def __len__(self) -> int:
        return len(self.samples)

    def append(self, sample: TrainSample, *, sample_id: str, class_id: str) -> None:
        self.samples.append(sample)
        self.sample_ids.append(str(sample_id))
        self.class_ids.append(str(class_id))
        if len(self.samples) > self.window_size:
            overflow = len(self.samples) - self.window_size
            del self.samples[:overflow]
            del self.sample_ids[:overflow]
            del self.class_ids[:overflow]

    def sample_batch(
        self,
        rng: np.random.Generator,
        *,
        batch_size: int,
        class_balance: Mapping[str, float] | None = None,
    ) -> list[TrainSample]:
        if not self.samples:
            raise BufferError("replay buffer is empty")
        if batch_size < 1:
            raise BufferError("batch_size must be >= 1")
        n = len(self.samples)
        if class_balance:
            weights = np.zeros(n, dtype=np.float64)
            for i, cid in enumerate(self.class_ids):
                weights[i] = float(class_balance.get(str(cid), 0.0))
            if float(weights.sum()) <= 0.0:
                weights = np.ones(n, dtype=np.float64)
            probs = weights / weights.sum()
        else:
            probs = None
        replace = n < batch_size
        idxs = rng.choice(n, size=batch_size, replace=replace, p=probs)
        return [self.samples[int(i)] for i in idxs]


def _targets_to_arrays(targets: SeatTargets) -> dict[str, np.ndarray]:
    return {
        "policy": np.asarray(targets.policy, dtype=np.float32),
        "wdl": np.asarray(targets.wdl, dtype=np.float32),
        "value": np.asarray([targets.value], dtype=np.float32),
        "hidden_owner": np.asarray(targets.hidden_owner, dtype=np.float32),
        "enemy_army_bin": np.asarray(targets.enemy_army_bin, dtype=np.int16),
        "enemy_general": np.asarray(targets.enemy_general, dtype=np.float32),
        "hidden_castle": np.asarray(targets.hidden_castle, dtype=np.float32),
        "board_mask": np.asarray(targets.board_mask, dtype=np.float32),
        "land_margin": np.asarray([targets.land_margin], dtype=np.float32),
        "army_margin": np.asarray([targets.army_margin], dtype=np.float32),
        "castle_margin": np.asarray([targets.castle_margin], dtype=np.float32),
        "turns_to_termination": np.asarray(
            [targets.turns_to_termination], dtype=np.float32
        ),
    }


def _targets_from_arrays(arrays: Mapping[str, np.ndarray]) -> SeatTargets:
    return SeatTargets(
        policy=np.asarray(arrays["policy"], dtype=np.float64),
        wdl=np.asarray(arrays["wdl"], dtype=np.float64),
        value=float(np.asarray(arrays["value"]).reshape(-1)[0]),
        hidden_owner=np.asarray(arrays["hidden_owner"], dtype=np.float32),
        enemy_army_bin=np.asarray(arrays["enemy_army_bin"], dtype=np.int16),
        enemy_general=np.asarray(arrays["enemy_general"], dtype=np.float32),
        hidden_castle=np.asarray(arrays["hidden_castle"], dtype=np.float32),
        land_margin=float(np.asarray(arrays["land_margin"]).reshape(-1)[0]),
        army_margin=float(np.asarray(arrays["army_margin"]).reshape(-1)[0]),
        castle_margin=float(np.asarray(arrays["castle_margin"]).reshape(-1)[0]),
        turns_to_termination=float(
            np.asarray(arrays["turns_to_termination"]).reshape(-1)[0]
        ),
        board_mask=np.asarray(arrays["board_mask"], dtype=np.float32),
    )


def write_sample(
    directory: Path,
    sample: TrainSample,
    *,
    sample_id: str | None = None,
    class_id: str = "1",
) -> Path:
    """Atomic npz write for one training sample."""
    directory = Path(directory)
    if "data/games" in directory.as_posix():
        raise BufferError(f"refusing to write training samples under {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    sid = str(sample_id or sample.item_id)
    path = directory / f"{sid}{SAMPLE_SUFFIX}"
    tmp = path.with_name(path.name + ".tmp.npz")
    meta = {
        "item_id": sample.item_id,
        "sample_seat": int(sample.sample_seat),
        "source_label": sample.source_label,
        "outcome": sample.outcome,
        "class_id": str(class_id),
        "sample_id": sid,
    }
    payload = {
        "tensor": np.asarray(sample.tensor, dtype=np.float32),
        "legal_mask": np.asarray(sample.legal_mask, dtype=np.bool_),
        "meta_json": np.asarray([json.dumps(meta, sort_keys=True)]),
        **_targets_to_arrays(sample.targets),
    }
    try:
        np.savez_compressed(tmp, **payload)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def read_sample(path: Path) -> tuple[TrainSample, str, str]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta_json"][0]))
        targets = _targets_from_arrays(data)
        sample = TrainSample(
            item_id=str(meta["item_id"]),
            sample_seat=int(meta["sample_seat"]),
            tensor=np.asarray(data["tensor"], dtype=np.float32),
            legal_mask=np.asarray(data["legal_mask"], dtype=bool),
            targets=targets,
            source_label=str(meta.get("source_label") or ""),
            outcome=meta.get("outcome"),
        )
        return sample, str(meta.get("sample_id") or path.stem), str(meta.get("class_id") or "1")


def iter_sample_paths(directory: Path) -> Iterator[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        return iter(())
    return iter(sorted(directory.glob(f"*{SAMPLE_SUFFIX}")))


def load_replay_buffer(
    directory: Path,
    *,
    window_size: int,
    max_samples: int | None = None,
    progress_every: int = 500,
) -> ReplayBuffer:
    directory = Path(directory)
    paths = list(iter_sample_paths(directory))
    if max_samples is not None:
        paths = paths[: int(max_samples)]
    total = len(paths)
    print(
        f"[buffer] load start dir={directory} files={total} window_size={window_size}",
        flush=True,
    )
    samples: list[TrainSample] = []
    ids: list[str] = []
    classes: list[str] = []
    t0 = time.perf_counter()
    every = max(1, int(progress_every))
    for i, path in enumerate(paths, start=1):
        sample, sid, cid = read_sample(path)
        samples.append(sample)
        ids.append(sid)
        classes.append(cid)
        if i == total or i % every == 0:
            elapsed = time.perf_counter() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            print(
                f"[buffer] load {i}/{total} elapsed_s={elapsed:.1f} files_per_s={rate:.1f}",
                flush=True,
            )
    if not samples:
        raise BufferError(f"no samples under {directory}")
    print(
        f"[buffer] load done samples={len(samples)} wall_s={time.perf_counter() - t0:.1f}",
        flush=True,
    )
    return ReplayBuffer(
        samples=samples,
        sample_ids=ids,
        class_ids=classes,
        window_size=int(window_size),
    )


def write_buffer_index(directory: Path, sample_ids: Sequence[str]) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / INDEX_NAME
    payload = {"sample_ids": list(sample_ids)}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
