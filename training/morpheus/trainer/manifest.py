"""Immutable run manifest for a Morpheus training run (Part 14).

Curriculum class movement and arena checkpoint acceptance both appear as
``promotion`` in older specs. This manifest uses distinct event names so a
resume log cannot confuse them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

# Distinct from arena acceptance and from training snapshot writes.
EVENT_CURRICULUM_CLASS_ADVANCE = "curriculum_class_advance"
EVENT_ARENA_CHECKPOINT_ACCEPT = "arena_checkpoint_accept"
EVENT_SNAPSHOT_SAVED = "snapshot_saved"
EVENT_DEPLOYMENT_CALIBRATION = "deployment_calibration"

RUN_MANIFEST_VERSION = 1
MANIFEST_NAME = "run_manifest.json"

KNOWN_EVENTS = frozenset(
    {
        EVENT_CURRICULUM_CLASS_ADVANCE,
        EVENT_ARENA_CHECKPOINT_ACCEPT,
        EVENT_SNAPSHOT_SAVED,
        EVENT_DEPLOYMENT_CALIBRATION,
    }
)


class ManifestError(ValueError):
    """Run manifest is missing, mutable, or incompatible."""


@dataclass(frozen=True)
class RunManifest:
    """Self-describing identity for one training run. Written once; never mutated."""

    run_id: str
    schema_version: int
    engine_era: str
    tensor_schema: str
    action_schema: str
    architecture_version: str
    objective_name: str
    objective: dict[str, Any]
    scope: str
    promotable_main_run: bool
    layout: dict[str, Any]
    trainer: dict[str, Any]
    part13_verdict: str
    part13_fallback: str | None = None
    deployment: dict[str, Any] = field(default_factory=dict)
    event_vocabulary: dict[str, str] = field(
        default_factory=lambda: {
            "curriculum_movement": EVENT_CURRICULUM_CLASS_ADVANCE,
            "arena_acceptance": EVENT_ARENA_CHECKPOINT_ACCEPT,
            "training_snapshot": EVENT_SNAPSHOT_SAVED,
            "deployment_calibration": EVENT_DEPLOYMENT_CALIBRATION,
        }
    )
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ManifestError("run_id must be non-empty")
        if int(self.schema_version) != RUN_MANIFEST_VERSION:
            raise ManifestError(
                f"unsupported run manifest schema_version={self.schema_version}; "
                f"expected {RUN_MANIFEST_VERSION}"
            )
        if bool(self.promotable_main_run) and self.scope != "promotable_main_run":
            raise ManifestError(
                "promotable_main_run requires scope='promotable_main_run'"
            )
        if bool(self.promotable_main_run) and self.part13_verdict != "yes":
            raise ManifestError(
                "promotable_main_run requires Part 13 verdict=yes "
                f"(got {self.part13_verdict!r})"
            )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["notes"] = list(self.notes)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunManifest:
        if not isinstance(data, Mapping):
            raise ManifestError("run manifest must be an object")
        required = (
            "run_id",
            "schema_version",
            "engine_era",
            "tensor_schema",
            "action_schema",
            "architecture_version",
            "objective_name",
            "objective",
            "scope",
            "promotable_main_run",
            "layout",
            "trainer",
            "part13_verdict",
        )
        missing = [k for k in required if k not in data]
        if missing:
            raise ManifestError(f"run manifest missing keys: {missing}")
        notes = data.get("notes") or ()
        return cls(
            run_id=str(data["run_id"]),
            schema_version=int(data["schema_version"]),
            engine_era=str(data["engine_era"]),
            tensor_schema=str(data["tensor_schema"]),
            action_schema=str(data["action_schema"]),
            architecture_version=str(data["architecture_version"]),
            objective_name=str(data["objective_name"]),
            objective=dict(data["objective"]),
            scope=str(data["scope"]),
            promotable_main_run=bool(data["promotable_main_run"]),
            layout=dict(data["layout"]),
            trainer=dict(data["trainer"]),
            part13_verdict=str(data["part13_verdict"]),
            part13_fallback=(
                None
                if data.get("part13_fallback") is None
                else str(data["part13_fallback"])
            ),
            deployment=dict(data.get("deployment") or {}),
            event_vocabulary=dict(
                data.get("event_vocabulary")
                or {
                    "curriculum_movement": EVENT_CURRICULUM_CLASS_ADVANCE,
                    "arena_acceptance": EVENT_ARENA_CHECKPOINT_ACCEPT,
                    "training_snapshot": EVENT_SNAPSHOT_SAVED,
                    "deployment_calibration": EVENT_DEPLOYMENT_CALIBRATION,
                }
            ),
            notes=tuple(str(x) for x in notes),
        )

    def compatibility_key(self) -> dict[str, str | int | bool]:
        """Fields that must match before resume is allowed."""
        return {
            "schema_version": int(self.schema_version),
            "engine_era": self.engine_era,
            "tensor_schema": self.tensor_schema,
            "action_schema": self.action_schema,
            "architecture_version": self.architecture_version,
            "run_id": self.run_id,
            "objective_name": self.objective_name,
            "scope": self.scope,
            "promotable_main_run": bool(self.promotable_main_run),
        }


def write_run_manifest(manifest: RunManifest, directory: Path) -> Path:
    """Atomic write. Refuse to overwrite an existing manifest (immutable)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MANIFEST_NAME
    if path.is_file():
        existing = load_run_manifest(directory)
        if existing.to_dict() != manifest.to_dict():
            raise ManifestError(
                f"run manifest already exists at {path} and differs; "
                "run manifests are immutable"
            )
        return path
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    try:
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def load_run_manifest(directory: Path) -> RunManifest:
    path = Path(directory) / MANIFEST_NAME
    if not path.is_file():
        raise ManifestError(f"run manifest missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return RunManifest.from_dict(data)


def assert_resume_compatible(stored: RunManifest, presented: RunManifest) -> None:
    """Resume only when schema, engine era, tensor/action schema, and run match."""
    a = stored.compatibility_key()
    b = presented.compatibility_key()
    mismatches = {k: (a[k], b[k]) for k in a if a[k] != b[k]}
    if mismatches:
        raise ManifestError(f"resume rejected; incompatible fields: {mismatches}")
