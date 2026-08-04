"""Build a curriculum manifest from panel trajectories and new map seeds."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
            )
        )
    return items


def build_curriculum(
    *,
    panel_path: Path,
    trajectories_dir: Path,
    output: Path,
    full_start_seeds: list[int] | None = None,
    full_start_count: int = 8,
    max_games: int | None = None,
    skip_verify: bool = False,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """
    Classify reachable prefixes, reject ResBot sources, write the manifest.

    Verifies each source trajectory's digests before extracting prefixes.
    """
    root = repo_root or REPO_ROOT
    panel = load_panel(panel_path)
    default_label = str(panel.get("source_label") or SOURCE_FIXED_PANEL)
    if is_banned_source(default_label):
        raise ValueError(f"panel source_label is banned: {default_label}")

    index = load_source_index(trajectories_dir)
    engine = current_engine_version()
    paths = iter_panel_trajectories(trajectories_dir)
    if max_games is not None:
        paths = paths[: int(max_games)]

    items: list[CurriculumItem] = []
    verify_failures: list[str] = []
    skipped_banned = 0

    for path in paths:
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
        end = traj.end
        winner = str(end.get("winner", "draw"))
        decisive = winner in ("a", "b")
        rel = _relpath(path, root)

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
                )
            )

    if full_start_seeds is None:
        # Deterministic pilot seeds derived from the panel round seed.
        base = int(panel.get("round_seed") or 7)
        full_start_seeds = [base + 10_000 + i for i in range(int(full_start_count))]
    items.extend(
        make_full_start_items(engine=engine, seeds=list(full_start_seeds))
    )

    manifest = CurriculumManifest(
        panel_name=str(panel["name"]),
        panel_path=_relpath(Path(panel_path), root),
        engine_version=engine,
        source_label=default_label,
        confidence_rule=dict(DEFAULT_CONFIDENCE_RULE),
        items=items,
        notes=[
            "Classes 1–4 come from verified panel trajectories.",
            "Class 5 uses new competition map seeds with empty prefixes.",
            "ResBot sources are rejected.",
            "Confidence rule selected before the main training run.",
        ],
    )
    if verify_failures:
        manifest.notes.append(f"verify_failures={len(verify_failures)}")

    manifest.write(output)
    class_counts: dict[str, int] = {}
    for item in items:
        class_counts[str(item.class_id)] = class_counts.get(str(item.class_id), 0) + 1

    return {
        "output": str(output),
        "item_count": len(items),
        "class_counts": class_counts,
        "trajectories_scanned": len(paths),
        "skipped_banned": skipped_banned,
        "verify_failures": verify_failures,
        "confidence_rule": manifest.confidence_rule["name"],
        "ok": not verify_failures,
    }
