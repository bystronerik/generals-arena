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
from training.morpheus.curriculum.stratify import (
    GameCandidate,
    StratifyResult,
    candidate_from_meta,
    stratify_global_wdl,
)


def _relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def iter_panel_trajectories(trajectories_dir: Path) -> list[Path]:
    return sorted(Path(trajectories_dir).glob("*.traj.jsonl.gz"))


def discover_trajectory_dirs(
    parent: Path,
    *,
    reconstructions_only: bool = True,
) -> list[Path]:
    """Child dirs under ``parent`` that hold trajectories or a corpus-index."""
    parent = Path(parent)
    if not parent.is_dir():
        raise FileNotFoundError(f"trajectories parent missing: {parent}")
    found: list[Path] = []
    for child in sorted(parent.iterdir()):
        if not child.is_dir():
            continue
        if reconstructions_only and not child.name.endswith("-reconstructions"):
            continue
        has_traj = any(child.glob("*.traj.jsonl.gz"))
        has_index = (child / "corpus-index.json").is_file()
        if has_traj or has_index:
            found.append(child)
    return found


def load_corpus_game_meta(directory: Path) -> dict[str, dict[str, Any]]:
    """Per-game metadata from corpus-index.json (seat, player names, labels)."""
    path = Path(directory) / "corpus-index.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    games = data.get("games") or {}
    return {str(gid): dict(meta) for gid, meta in games.items()}


def load_corpus_round_meta(directory: Path) -> dict[str, Any]:
    """Round-level fields from corpus-index.json."""
    path = Path(directory) / "corpus-index.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "player": data.get("player"),
        "outcome": data.get("outcome"),
        "source_label": data.get("source_label"),
        "round": data.get("round"),
    }


def collect_game_candidates(trajectories_dirs: Sequence[Path]) -> list[GameCandidate]:
    """Index reconstruction games from corpus-index without reading traj bodies."""
    out: list[GameCandidate] = []
    for traj_dir in trajectories_dirs:
        traj_dir = Path(traj_dir)
        round_meta = load_corpus_round_meta(traj_dir)
        game_meta = load_corpus_game_meta(traj_dir)
        paths = {p.name.replace(".traj.jsonl.gz", ""): p for p in iter_panel_trajectories(traj_dir)}
        # Prefer index entries; fall back to path stems when index is partial.
        ids = sorted(set(game_meta) | set(paths))
        for gid in ids:
            path = paths.get(gid) or (traj_dir / f"{gid}.traj.jsonl.gz")
            if not path.is_file():
                continue
            meta = game_meta.get(gid) or {}
            cand = candidate_from_meta(
                path=path,
                game_id=gid,
                meta=meta,
                round_player=str(round_meta.get("player") or "") or None,
                round_outcome=str(round_meta.get("outcome") or "") or None,
            )
            if cand is not None:
                out.append(cand)
    return out


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
    """Class-5 items: empty prefixes on new competition map seeds.

    Emits both seats with distinct ``item_id`` suffixes (``_s0`` / ``_s1``)
    so materialize can write one sample per seat without id collisions.
    """
    items: list[CurriculumItem] = []
    for seed in seeds:
        for seat in (0, 1):
            base = CurriculumItem.build(
                class_id=CLASS_FULL_START,
                engine_version=engine,
                map_seed=int(seed),
                source_label=source_label,
                prefix_len=0,
                game_id=None,
                trajectory_relpath=None,
                decisive=None,
                outcome=None,
                sample_seat=int(seat),
            )
            items.append(
                CurriculumItem.from_dict(
                    {**base.to_dict(), "item_id": f"{base.item_id}_s{seat}"}
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
    stratify_global_wdl_flag: bool = False,
    top_win_players: Sequence[str] | None = None,
    top_win_fraction: float = 0.5,
    max_pairs: int | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """
    Classify reachable prefixes, reject ResBot sources, write the manifest.

    Verifies each source trajectory's digests before extracting prefixes.
    ``class_ids`` keeps only those classes (e.g. ``[1]`` for the scraped pilot).
    ``require_sample_seat`` forces every non-class-5 item to carry a seat.
    ``stratify_global_wdl_flag`` keeps equal wins/losses before classify.
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

    stratify_report: StratifyResult | None = None
    path_filter: set[Path] | None = None
    if stratify_global_wdl_flag:
        players = list(top_win_players) if top_win_players else list(
            panel.get("top_win_players") or []
        )
        if not players:
            raise ValueError(
                "stratify_global_wdl requires --top-win-players or panel.top_win_players"
            )
        if top_win_players is not None:
            fraction = float(top_win_fraction)
        elif panel.get("top_win_fraction") is not None:
            fraction = float(panel["top_win_fraction"])
        else:
            fraction = float(top_win_fraction)
        candidates = collect_game_candidates(dirs)
        stratify_report = stratify_global_wdl(
            candidates,
            top_win_players=players,
            top_win_fraction=fraction,
            max_pairs=max_pairs,
        )
        path_filter = {c.path.resolve() for c in stratify_report.kept}

    items: list[CurriculumItem] = []
    verify_failures: list[str] = []
    skipped_banned = 0
    skipped_class = 0
    skipped_stratify = 0
    trajectories_scanned = 0

    for traj_dir in dirs:
        index = load_source_index(traj_dir)
        game_meta = load_corpus_game_meta(traj_dir)
        paths = iter_panel_trajectories(traj_dir)
        if path_filter is not None:
            filtered: list[Path] = []
            for path in paths:
                if path.resolve() in path_filter:
                    filtered.append(path)
                else:
                    skipped_stratify += 1
            paths = filtered
        if max_games is not None:
            remaining = int(max_games) - trajectories_scanned
            if remaining <= 0:
                break
            paths = paths[:remaining]

        for path in paths:
            trajectories_scanned += 1
            if trajectories_scanned == 1 or trajectories_scanned % 50 == 0:
                print(
                    f"[curriculum build] scanned={trajectories_scanned} "
                    f"items={len(items)} failures={len(verify_failures)}",
                    flush=True,
                )
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
    if stratify_report is not None:
        notes.extend(stratify_report.notes)

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
    result: dict[str, Any] = {
        "output": str(output),
        "item_count": len(items),
        "class_counts": class_counts,
        "source_labels": source_labels,
        "trajectories_scanned": trajectories_scanned,
        "trajectories_dirs": [str(d) for d in dirs],
        "skipped_banned": skipped_banned,
        "skipped_class": skipped_class,
        "skipped_stratify": skipped_stratify,
        "verify_failures": verify_failures,
        "confidence_rule": manifest.confidence_rule["name"],
        "ok": not verify_failures,
    }
    if stratify_report is not None:
        result["stratify"] = stratify_report.to_dict()
    return result

