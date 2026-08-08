#!/usr/bin/env python3
"""Materialize curriculum items into a Part 14 *.sample.npz buffer.

One reconstruct pass per game_id continues belief across prefixes (shared
seat seed). Rematerialize after the seed contract change; do not reuse
``*.sample.npz`` built under the old per-prefix seed.

Usage:
    python scripts/morpheus_materialize.py \\
      --manifest training/morpheus/manifests/scraped-classes13.json \\
      --output data/morpheus/trainer/buffer

    python scripts/morpheus_materialize.py \\
      --manifest training/morpheus/manifests/scraped-classes13.json \\
      --output data/morpheus/trainer/buffer --max-items 32
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_OUTPUT = REPO / "data" / "morpheus" / "trainer" / "buffer"


def _sample_already_present(
    sample_id: str,
    *,
    output: Path,
    skip_existing: bool,
    skip_existing_dirs: list[Path],
    suffix: str,
) -> bool:
    """True when skip is on and the sample exists under output or lookup dirs."""
    if not skip_existing:
        return False
    name = f"{sample_id}{suffix}"
    if (output / name).is_file():
        return True
    return any((Path(d) / name).is_file() for d in skip_existing_dirs)


def _materialize_game_group(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Worker: materialize all items that share one game_id in one reconstruct pass."""
    from training.morpheus.curriculum.reconstruct import iter_prefix_reconstructions
    from training.morpheus.curriculum.schema import CurriculumItem
    from training.morpheus.trainer.buffer import SAMPLE_SUFFIX, write_sample
    from training.morpheus.trainer.sample import build_train_sample_from_recon
    from arena.records.trajectories import read_trajectory

    root = Path(payload["repo"])
    output = Path(payload["output"])
    n_particles = int(payload["n_particles"])
    skip_existing = bool(payload["skip_existing"])
    skip_existing_dirs = [Path(p) for p in (payload.get("skip_existing_dirs") or [])]
    items = [CurriculumItem.from_dict(d) for d in payload["items"]]

    written: list[str] = []
    skipped = 0
    failures: list[str] = []
    newly = 0
    traj = None
    terminal_cache: dict[str, Any] = {}
    if items and items[0].trajectory_relpath:
        path = root / items[0].trajectory_relpath
        if path.is_file():
            traj = read_trajectory(path)

    pending: list[CurriculumItem] = []
    class5_pending: list[CurriculumItem] = []
    for item in items:
        if int(item.class_id) == 5:
            seats = (
                (0, 1)
                if item.sample_seat is None
                else (int(item.sample_seat),)
            )
            for seat in seats:
                sample_id = (
                    str(item.item_id)
                    if item.sample_seat is not None
                    else f"{item.item_id}_s{seat}"
                )
                if _sample_already_present(
                    sample_id,
                    output=output,
                    skip_existing=skip_existing,
                    skip_existing_dirs=skip_existing_dirs,
                    suffix=SAMPLE_SUFFIX,
                ):
                    written.append(sample_id)
                    skipped += 1
                    continue
                # Clone with explicit seat for the worker path below.
                row = CurriculumItem.from_dict(
                    {**item.to_dict(), "sample_seat": int(seat), "item_id": sample_id}
                )
                class5_pending.append(row)
            continue
        if item.sample_seat is None:
            failures.append(f"{item.item_id}: missing sample_seat")
            continue
        sample_id = str(item.item_id)
        if _sample_already_present(
            sample_id,
            output=output,
            skip_existing=skip_existing,
            skip_existing_dirs=skip_existing_dirs,
            suffix=SAMPLE_SUFFIX,
        ):
            written.append(sample_id)
            skipped += 1
            continue
        pending.append(item)

    if class5_pending:
        from training.morpheus.trainer.sample import build_full_start_train_sample

        for item in class5_pending:
            try:
                sample = build_full_start_train_sample(
                    item,
                    seat=int(item.sample_seat),
                    n_particles=n_particles,
                )
                write_sample(
                    output,
                    sample,
                    sample_id=str(item.item_id),
                    class_id=str(item.class_id),
                )
                written.append(str(item.item_id))
                newly += 1
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{item.item_id}: {exc}")

    if not pending:
        return {
            "game_id": payload.get("game_id"),
            "written": written,
            "skipped": skipped,
            "newly": newly,
            "failures": failures,
        }

    if traj is None:
        for item in pending:
            failures.append(f"{item.item_id}: missing trajectory for train sample")
        return {
            "game_id": payload.get("game_id"),
            "written": written,
            "skipped": skipped,
            "newly": newly,
            "failures": failures,
        }

    try:
        for item, recon in iter_prefix_reconstructions(
            pending,
            trajectories_root=root,
            traj=traj,
            n_particles=n_particles,
            update_belief_flag=True,
            engine=pending[0].engine_version,
        ):
            sample_id = str(item.item_id)
            try:
                sample = build_train_sample_from_recon(
                    item,
                    recon,
                    traj=traj,
                    terminal_cache=terminal_cache,
                )
                write_sample(
                    output,
                    sample,
                    sample_id=sample_id,
                    class_id=str(item.class_id),
                )
                written.append(sample_id)
                newly += 1
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{item.item_id}: {exc}")
    except Exception as exc:  # noqa: BLE001
        for item in pending:
            failures.append(f"{item.item_id}: {exc}")

    return {
        "game_id": payload.get("game_id"),
        "written": written,
        "skipped": skipped,
        "newly": newly,
        "failures": failures,
    }


def materialize_manifest(
    *,
    manifest_path: Path,
    output: Path,
    max_items: int | None = None,
    skip_existing: bool = True,
    skip_existing_dirs: list[Path] | None = None,
    n_particles: int = 4,
    class_ids: set[int] | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
    repo_root: Path | None = None,
) -> dict:
    from training.morpheus.curriculum.schema import CurriculumManifest
    from training.morpheus.trainer.buffer import write_buffer_index

    root = repo_root or REPO
    manifest = CurriculumManifest.load(manifest_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lookup_dirs = [Path(p) for p in (skip_existing_dirs or [])]

    items = list(manifest.items)
    if class_ids is not None:
        items = [it for it in items if int(it.class_id) in class_ids]
    if max_items is not None:
        items = items[: int(max_items)]

    by_game: dict[str, list] = defaultdict(list)
    for item in items:
        key = str(item.game_id or item.item_id)
        by_game[key].append(item)

    game_ids = sorted(by_game.keys())
    n_shards = max(1, int(shard_count))
    idx = int(shard_index)
    if idx < 0 or idx >= n_shards:
        raise ValueError(f"shard_index {idx} out of range for shard_count {n_shards}")
    if n_shards > 1:
        game_ids = [gid for i, gid in enumerate(game_ids) if i % n_shards == idx]

    payloads = []
    for game_id in game_ids:
        group = by_game[game_id]
        payloads.append(
            {
                "repo": str(root),
                "output": str(output),
                "n_particles": int(n_particles),
                "skip_existing": bool(skip_existing),
                "skip_existing_dirs": [str(p) for p in lookup_dirs],
                "game_id": game_id,
                "items": [it.to_dict() for it in group],
            }
        )

    written: list[str] = []
    skipped = 0
    newly = 0
    failures: list[str] = []
    t0 = time.perf_counter()
    print(
        f"[materialize] shard={idx}/{n_shards} games={len(payloads)} "
        f"items_in_shard={sum(len(p['items']) for p in payloads)}",
        flush=True,
    )

    for i, payload in enumerate(payloads, start=1):
        result = _materialize_game_group(payload)
        written.extend(result["written"])
        skipped += int(result["skipped"])
        newly += int(result["newly"])
        failures.extend(result["failures"])
        if i % 1 == 0 or i == len(payloads):
            print(
                f"[materialize] shard={idx} games={i}/{len(payloads)} "
                f"new={newly} skipped={skipped} failures={len(failures)}",
                flush=True,
            )

    # When sharding, only the final merge pass should rewrite the global index.
    written_sorted = sorted(set(written))
    if n_shards == 1:
        write_buffer_index(output, written_sorted)
    wall_s = time.perf_counter() - t0
    report = {
        "ok": not failures,
        "manifest": str(manifest_path),
        "output": str(output),
        "requested": len(items),
        "games": len(payloads),
        "written_or_present": len(written_sorted),
        "newly_written": newly,
        "skipped_existing": skipped,
        "failures": failures[:50],
        "failure_count": len(failures),
        "shard_index": idx,
        "shard_count": n_shards,
        "wall_s": wall_s,
    }
    report_path = output / (
        "materialize_report.json"
        if n_shards == 1
        else f"materialize_report_shard{idx}.json"
    )
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def merge_buffer_index(output: Path) -> Path:
    """Rewrite buffer_index.json from all *.sample.npz under output."""
    from training.morpheus.trainer.buffer import iter_sample_paths, write_buffer_index

    ids = [p.stem for p in iter_sample_paths(output)]
    return write_buffer_index(output, sorted(ids))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--n-particles", type=int, default=4)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--merge-index-only",
        action="store_true",
        help="only rewrite buffer_index.json from existing npz files",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        help="optional class id filter (e.g. --classes 1)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="rewrite samples even when the npz already exists",
    )
    args = parser.parse_args(argv)
    if args.merge_index_only:
        path = merge_buffer_index(args.output)
        print(json.dumps({"ok": True, "index": str(path)}, indent=2))
        return 0
    class_ids = {int(x) for x in args.classes} if args.classes else None
    report = materialize_manifest(
        manifest_path=args.manifest,
        output=args.output,
        max_items=args.max_items,
        skip_existing=not bool(args.no_skip_existing),
        n_particles=args.n_particles,
        class_ids=class_ids,
        shard_index=int(args.shard_index),
        shard_count=int(args.shard_count),
        repo_root=REPO,
    )
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, indent=2))
    if report.get("failures"):
        print(json.dumps({"failures_head": report["failures"]}, indent=2))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
