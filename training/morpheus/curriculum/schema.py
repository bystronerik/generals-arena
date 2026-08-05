"""Curriculum item and manifest JSON schema."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from training.morpheus.curriculum.confidence import DEFAULT_CONFIDENCE_RULE
from training.morpheus.curriculum.definitions import (
    CLASS_NAMES,
    belief_rng_seed,
    is_banned_source,
    item_id_for,
)


@dataclass(frozen=True)
class CurriculumItem:
    """One reachable prefix: engine era, seed, source, joint-action length."""

    item_id: str
    class_id: int
    engine_version: str
    map_seed: int
    source_label: str
    prefix_len: int
    game_id: str | None = None
    trajectory_relpath: str | None = None
    pre_contact_distance: int | None = None
    first_contact_turn: int | None = None
    first_sight_turn: int | None = None
    outcome: str | None = None
    decisive: bool | None = None
    sample_seat: int | None = None

    def __post_init__(self) -> None:
        if self.class_id not in CLASS_NAMES:
            raise ValueError(f"unknown class_id {self.class_id}")
        if is_banned_source(self.source_label):
            raise ValueError(f"banned source_label {self.source_label!r}")
        if self.prefix_len < 0:
            raise ValueError("prefix_len must be >= 0")
        if self.class_id == 5 and self.prefix_len != 0:
            raise ValueError("class 5 requires prefix_len == 0")
        if self.sample_seat is not None and self.sample_seat not in (0, 1):
            raise ValueError(f"sample_seat must be 0 or 1, got {self.sample_seat}")

    @property
    def class_name(self) -> str:
        return CLASS_NAMES[self.class_id]

    def belief_seed(self, seat: int) -> int:
        return belief_rng_seed(
            engine_version=self.engine_version,
            map_seed=self.map_seed,
            source_label=self.source_label,
            prefix_len=self.prefix_len,
            game_id=self.game_id,
            seat=seat,
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["class_name"] = self.class_name
        d["belief_seed_seat0"] = self.belief_seed(0)
        d["belief_seed_seat1"] = self.belief_seed(1)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CurriculumItem:
        keys = {
            "item_id",
            "class_id",
            "engine_version",
            "map_seed",
            "source_label",
            "prefix_len",
            "game_id",
            "trajectory_relpath",
            "pre_contact_distance",
            "first_contact_turn",
            "first_sight_turn",
            "outcome",
            "decisive",
            "sample_seat",
        }
        payload = {k: data.get(k) for k in keys}
        payload["map_seed"] = int(payload["map_seed"])
        payload["prefix_len"] = int(payload["prefix_len"])
        payload["class_id"] = int(payload["class_id"])
        if payload["sample_seat"] is not None:
            payload["sample_seat"] = int(payload["sample_seat"])
        return cls(**payload)  # type: ignore[arg-type]

    @classmethod
    def build(
        cls,
        *,
        class_id: int,
        engine_version: str,
        map_seed: int,
        source_label: str,
        prefix_len: int,
        game_id: str | None = None,
        trajectory_relpath: str | None = None,
        pre_contact_distance: int | None = None,
        first_contact_turn: int | None = None,
        first_sight_turn: int | None = None,
        outcome: str | None = None,
        decisive: bool | None = None,
        sample_seat: int | None = None,
    ) -> CurriculumItem:
        return cls(
            item_id=item_id_for(
                engine_version=engine_version,
                map_seed=map_seed,
                source_label=source_label,
                prefix_len=prefix_len,
                game_id=game_id,
                class_id=class_id,
            ),
            class_id=class_id,
            engine_version=engine_version,
            map_seed=int(map_seed),
            source_label=source_label,
            prefix_len=int(prefix_len),
            game_id=game_id,
            trajectory_relpath=trajectory_relpath,
            pre_contact_distance=pre_contact_distance,
            first_contact_turn=first_contact_turn,
            first_sight_turn=first_sight_turn,
            outcome=outcome,
            decisive=decisive,
            sample_seat=sample_seat,
        )


@dataclass
class CurriculumManifest:
    """Bounded curriculum pilot: items plus the executable confidence rule."""

    panel_name: str
    panel_path: str
    engine_version: str
    source_label: str
    confidence_rule: dict[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_CONFIDENCE_RULE)
    )
    items: list[CurriculumItem] = field(default_factory=list)
    excluded_sources: list[str] = field(
        default_factory=lambda: ["resbot", "scraped_resbot", "leaderboard_resbot"]
    )
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for item in self.items:
            key = str(item.class_id)
            counts[key] = counts.get(key, 0) + 1
        return {
            "panel_name": self.panel_name,
            "panel_path": self.panel_path,
            "engine_version": self.engine_version,
            "source_label": self.source_label,
            "excluded_sources": list(self.excluded_sources),
            "confidence_rule": self.confidence_rule,
            "class_counts": counts,
            "item_count": len(self.items),
            "notes": list(self.notes),
            "items": [item.to_dict() for item in self.items],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CurriculumManifest:
        items = [CurriculumItem.from_dict(row) for row in data.get("items") or []]
        return cls(
            panel_name=str(data["panel_name"]),
            panel_path=str(data["panel_path"]),
            engine_version=str(data["engine_version"]),
            source_label=str(data["source_label"]),
            confidence_rule=dict(data.get("confidence_rule") or DEFAULT_CONFIDENCE_RULE),
            items=items,
            excluded_sources=list(data.get("excluded_sources") or ["ResBot", "resbot"]),
            notes=list(data.get("notes") or []),
        )

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: Path) -> CurriculumManifest:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)
