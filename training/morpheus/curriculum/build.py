"""Build a curriculum manifest from panel trajectories and new map seeds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from arena.paths import REPO_ROOT
from arena.records.store import engine_version as current_engine_version
from arena.records.trajectories import (
    read_trajectory,
    require_same_era,
    verify_trajectory,
)
from training.morpheus.corpus.coverage import load_source_index
from training.morpheus.corpus.panel import load_panel
from training.morpheus.curriculum.classify import (
    classify_trajectory_prefixes,
    sample_prefixes_for_build,
)
from training.morpheus.curriculum.confidence import DEFAULT_CONFIDENCE_RULE
from training.morpheus.curriculum.definitions import (
    CLASS_FULL_START,
    SOURCE_FIXED_PANEL,
    SOURCE_FULL_START,
    is_banned_source,
)
from training.morpheus.curriculum.schema import CurriculumItem, CurriculumManifest


def _relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def iter_panel_trajectories(trajectories_dir: Path) -> list[Path]:
    return sorted(Path(trajectories_dir).glob("*.traj.jsonl.gz"))


def load_corpus_game_meta(directory: Path) -> dict[str, dict[str, Any]]:
    """Per-game metadata from corpus-index.json (seat, player names, labels)."""
    path = Path(directory) / "corpus-index.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    games = data.get("games") or {}
    return {str(gid): dict(meta) for gid, meta in games.items()}


def resolve_sample_seat(
    *,
    game_id: str | None,
    traj_header: dict[str, Any],
    game_meta: dict[str, Any] | None,
    require_seat: bool,
) -> int | None:
    """
    Resolve the training seat for a trajectory.

    Prefer corpus-index ``sample_seat`` / ``queried_player``; else match
    ``queried_player`` against ``bot_a`` / ``bot_b``.
    """
    meta = game_meta or {}
    if meta.get("sample_seat") is not None:
        seat = int(meta["sample_seat"])
        if seat not in (0, 1):
            raise ValueError(f"invalid sample_seat {seat} for {game_id}")
        return seat

    queried = meta.get("queried_player")
    bot_a = str(meta.get("bot_a") or traj_header.get("bot_a") or "")
    bot_b = str(meta.get("bot_b") or traj_header.get("bot_b") or "")
    if queried:
        if queried == bot_a:
            return 0
        if queried == bot_b:
            return 1
        raise ValueError(
            f"queried_player {queried!r} not in seats ({bot_a!r}, {bot_b!r}) "
            f"for {game_id}"
        )
    if require_seat:
        raise ValueError(
            f"sample_seat required but missing for game {game_id}; "
            "corpus-index needs sample_seat or queried_player"
        )
    return None


def make_full_start_items(
    *,
    engine: str,
    seeds: list[int],
    source_label: str = SOURCE_FULL_START,
) -> list[CurriculumItem]:
    """Class-5 items: empty prefixes on new competition map seeds."""
    items: list[CurriculumItem] = []
    for seed in seeds:
        items.append(
            CurriculumItem.build(
                class_id=CLASS_FULL_START,
                engine_version=engine,
                map_seed=int(seed),
                source_label=source_label,
                prefix_len=0,
                game_id=None,
                trajectory_relpath=None,
                decisive=None,
                outcome=None,
                sample_seat=None,
            )
        )
    return items


def build_curriculum(
    *,
    panel_path: Path,
    trajectories_dir: Path | None = None,
    trajectories_dirs: Sequence[Path] | None = None,
    output: Path,
    full_start_seeds: list[int] | None = None,
    full_start_count: int = 8,
    max_games: int | None = None,
    class_ids: Sequence[int] | None = None,
    require_sample_seat: bool = False,
    skip_verify: bool = False,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """
    Classify reachable prefixes, reject ResBot sources, write the manifest.

    Verifies each source trajectory's digests before extracting prefixes.
    ``class_ids`` keeps only those classes (e.g. ``[1]`` for the scraped pilot).
    ``require_sample_seat`` forces every non-class-5 item to carry a seat.
    """
    root = repo_root or REPO_ROOT
    panel = load_panel(panel_path)
    default_label = str(panel.get("source_label") or SOURCE_FIXED_PANEL)
    if is_banned_source(default_label):
        raise ValueError(f"panel source_label is banned: {default_label}")

    dirs: list[Path] = []
    if trajectories_dirs:
        dirs.extend(Path(d) for d in trajectories_dirs)
    if trajectories_dir is not None:
        dirs.append(Path(trajectories_dir))
    if not dirs:
        raise ValueError("at least one trajectories directory is required")

    allowed_classes = (
        {int(c) for c in class_ids} if class_ids is not None else None
    )
    engine = current_engine_version()

    items: list[CurriculumItem] = []
    verify_failures: list[str] = []
    skipped_banned = 0
    skipped_class = 0
    trajectories_scanned = 0

    for traj_dir in dirs:
        index = load_source_index(traj_dir)
        game_meta = load_corpus_game_meta(traj_dir)
        paths = iter_panel_trajectories(traj_dir)
        if max_games is not None:
            remaining = int(max_games) - trajectories_scanned
            if remaining <= 0:
                break
            paths = paths[:remaining]

        for path in paths:
            trajectories_scanned += 1
            traj = read_trajectory(path)
            label = index.get(traj.game_id) or default_label
            if is_banned_source(label):
                skipped_banned += 1
                continue
            # Raw ResBot seat names stay out unless this is an explicit reconstruction.
            from training.morpheus.scraped_rebuild.source import is_reconstruction_source

            bots = f"{traj.header.get('bot_a', '')} {traj.header.get('bot_b', '')}".lower()
            if "resbot" in bots and not is_reconstruction_source(label):
                skipped_banned += 1
                continue

            require_same_era(traj, engine=engine)
            if not skip_verify:
                report = verify_trajectory(traj, engine=engine)
                if not report.ok:
                    verify_failures.append(report.summary())
                    continue

            classified = classify_trajectory_prefixes(traj)
            selected = sample_prefixes_for_build(classified)
            if allowed_classes is not None:
                before = len(selected)
                selected = [row for row in selected if row.class_id in allowed_classes]
                skipped_class += before - len(selected)
            end = traj.end
            winner = str(end.get("winner", "draw"))
            decisive = winner in ("a", "b")
            rel = _relpath(path, root)
            meta = game_meta.get(traj.game_id) or {}
            try:
                sample_seat = resolve_sample_seat(
                    game_id=traj.game_id,
                    traj_header=traj.header,
                    game_meta=meta,
                    require_seat=require_sample_seat,
                )
            except ValueError as exc:
                verify_failures.append(str(exc))
                continue

            for row in selected:
                items.append(
                    CurriculumItem.build(
                        class_id=row.class_id,
                        engine_version=traj.engine_version,
                        map_seed=int(traj.seed),
                        source_label=label,
                        prefix_len=int(row.turn),
                        game_id=traj.game_id,
                        trajectory_relpath=rel,
                        pre_contact_distance=row.pre_contact_distance,
                        first_contact_turn=row.first_contact_turn,
                        first_sight_turn=row.first_sight_turn,
                        outcome=winner if winner in ("a", "b", "draw") else "draw",
                        decisive=decisive,
                        sample_seat=sample_seat,
                    )
                )

    if full_start_seeds is None:
        # Deterministic pilot seeds derived from the panel round seed.
        base = int(panel.get("round_seed") or 7)
        full_start_seeds = [base + 10_000 + i for i in range(int(full_start_count))]
    if allowed_classes is None or CLASS_FULL_START in allowed_classes:
        items.extend(
            make_full_start_items(engine=engine, seeds=list(full_start_seeds))
        )

    notes = [
        "Classes 1–4 come from verified trajectories.",
        "Class 5 uses new competition map seeds with empty prefixes.",
        "Raw ResBot sources are rejected; *_reconstructions labels are allowed.",
        "Confidence rule selected before the main training run.",
    ]
    if allowed_classes is not None:
        notes.append(f"class_ids filter={sorted(allowed_classes)}")
    if require_sample_seat:
        notes.append("sample_seat required on every trajectory item")
    if str(panel.get("kind") or "") == "provenance":
        notes.append("panel is a provenance stub (not a heuristic match panel)")

    manifest = CurriculumManifest(
        panel_name=str(panel["name"]),
        panel_path=_relpath(Path(panel_path), root),
        engine_version=engine,
        source_label=default_label,
        confidence_rule=dict(DEFAULT_CONFIDENCE_RULE),
        items=items,
        notes=notes,
    )
    if verify_failures:
        manifest.notes.append(f"verify_failures={len(verify_failures)}")

    manifest.write(output)
    class_counts: dict[str, int] = {}
    for item in items:
        class_counts[str(item.class_id)] = class_counts.get(str(item.class_id), 0) + 1

    source_labels = sorted({i.source_label for i in items})
    return {
        "output": str(output),
        "item_count": len(items),
        "class_counts": class_counts,
        "source_labels": source_labels,
        "trajectories_scanned": trajectories_scanned,
        "trajectories_dirs": [str(d) for d in dirs],
        "skipped_banned": skipped_banned,
        "skipped_class": skipped_class,
        "verify_failures": verify_failures,
        "confidence_rule": manifest.confidence_rule["name"],
        "ok": not verify_failures,
    }
