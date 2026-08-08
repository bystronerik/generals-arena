#!/usr/bin/env python3
"""Modal CPU materialize: fan-out curriculum items into a Part 14 buffer.

Upload trajectories + manifest once, then run one-core shards:

    modal volume put morpheus-training data/trajectories /morpheus/trajectories
    modal volume put morpheus-training \\
      training/morpheus/manifests/scraped-classes13.json \\
      /morpheus/manifests/scraped-classes13.json

    # Purge orphans not in the manifest, then write only missing IDs:
    modal run scripts/morpheus_modal_materialize.py \\
      --shards 96 --n-particles 4

Smoke (2 shards, 32 items):

    modal run scripts/morpheus_modal_materialize.py \\
      --shards 2 --n-particles 4 --max-items 32

Writes shard-private samples under ``/vol/morpheus/buffer_shards/<i>/``, then
merges into ``/vol/morpheus/buffer`` for Part 14 train. With
``skip_existing=True`` (default), a sample is skipped when it already exists
in ``/vol/morpheus/buffer`` or in the shard dir. ``purge_first`` (default)
deletes buffer samples whose ids are not in the manifest.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import modal


def _parse_classes(raw: str | None) -> set[int] | None:
    """Parse comma-separated class ids; empty means no filter."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    return {int(part.strip()) for part in text.split(",") if part.strip()}

REPO = Path(__file__).resolve().parents[1]

VOLUME = modal.Volume.from_name("morpheus-training", create_if_missing=True)

IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==2.4.6",
        "jax==0.11.0",
        "jaxlib==0.11.0",
        "torch==2.6.0",
    )
    .add_local_dir(
        str(REPO / "competition-module"),
        remote_path="/root/competition-module",
        copy=True,
    )
    .add_local_dir(str(REPO / "arena"), remote_path="/root/arena", copy=True)
    .add_local_dir(str(REPO / "bots"), remote_path="/root/bots", copy=True)
    .add_local_dir(str(REPO / "training"), remote_path="/root/training", copy=True)
    .add_local_dir(str(REPO / "scripts"), remote_path="/root/scripts", copy=True)
    .run_commands("pip install -e /root/competition-module --no-deps")
    .env({"PYTHONPATH": "/root:/root/scripts"})
)

app = modal.App("morpheus-materialize")

ROOT = Path("/root")
VOL_TRAJ = Path("/vol/morpheus/trajectories")
VOL_BUFFER = Path("/vol/morpheus/buffer")
VOL_SHARDS = Path("/vol/morpheus/buffer_shards")
LINK_TRAJ = ROOT / "data" / "trajectories"


def _ensure_trajectories_link() -> None:
    """Point /root/data/trajectories at the volume tree for trajectory_relpath."""
    if not VOL_TRAJ.is_dir():
        raise FileNotFoundError(
            f"missing trajectories on volume: {VOL_TRAJ} "
            "(run: modal volume put morpheus-training data/trajectories "
            "/morpheus/trajectories)"
        )
    LINK_TRAJ.parent.mkdir(parents=True, exist_ok=True)
    if LINK_TRAJ.is_symlink():
        if LINK_TRAJ.resolve() == VOL_TRAJ.resolve():
            return
        LINK_TRAJ.unlink()
    elif LINK_TRAJ.exists():
        raise RuntimeError(
            f"{LINK_TRAJ} exists and is not a symlink to {VOL_TRAJ}; refuse to replace"
        )
    LINK_TRAJ.symlink_to(VOL_TRAJ)


def _resolve_manifest(manifest_vol_path: str) -> Path:
    path = Path(manifest_vol_path)
    if path.is_file():
        return path
    raise FileNotFoundError(
        f"missing manifest on volume: {path} "
        "(run: modal volume put morpheus-training "
        "training/morpheus/manifests/scraped-classes13.json "
        "/morpheus/manifests/scraped-classes13.json)"
    )


@app.function(
    image=IMAGE,
    cpu=1,
    memory=512,
    timeout=6 * 3600,
    volumes={"/vol": VOLUME},
)
def materialize_shard(
    shard_index: int,
    shard_count: int,
    n_particles: int,
    max_items: int,
    skip_existing: bool,
    manifest_vol_path: str,
    classes: str = "",
) -> dict:
    """One CPU shard: materialize a game-id partition into a private dir."""
    sys.path.insert(0, "/root/scripts")
    from morpheus_materialize import materialize_manifest

    _ensure_trajectories_link()
    manifest = _resolve_manifest(manifest_vol_path)
    out = VOL_SHARDS / str(int(shard_index))
    out.mkdir(parents=True, exist_ok=True)
    class_ids = _parse_classes(classes)

    report = materialize_manifest(
        manifest_path=manifest,
        output=out,
        max_items=None if int(max_items) <= 0 else int(max_items),
        skip_existing=bool(skip_existing),
        skip_existing_dirs=[VOL_BUFFER],
        n_particles=int(n_particles),
        class_ids=class_ids,
        shard_index=int(shard_index),
        shard_count=int(shard_count),
        repo_root=ROOT,
    )
    VOLUME.commit()
    return {
        "ok": bool(report.get("ok")),
        "shard_index": int(shard_index),
        "shard_count": int(shard_count),
        "newly_written": int(report.get("newly_written") or 0),
        "skipped_existing": int(report.get("skipped_existing") or 0),
        "failure_count": int(report.get("failure_count") or 0),
        "written_or_present": int(report.get("written_or_present") or 0),
        "wall_s": float(report.get("wall_s") or 0.0),
        "output": str(out),
    }


@app.function(
    image=IMAGE,
    cpu=1,
    memory=512,
    timeout=60 * 60,
    volumes={"/vol": VOLUME},
)
def merge_buffer() -> dict:
    """Copy shard samples into /vol/morpheus/buffer and rewrite the index."""
    sys.path.insert(0, "/root/scripts")
    from morpheus_materialize import merge_buffer_index
    from training.morpheus.trainer.buffer import SAMPLE_SUFFIX

    VOL_BUFFER.mkdir(parents=True, exist_ok=True)
    moved = 0
    if VOL_SHARDS.is_dir():
        for shard_dir in sorted(VOL_SHARDS.iterdir()):
            if not shard_dir.is_dir():
                continue
            for src in shard_dir.glob(f"*{SAMPLE_SUFFIX}"):
                dest = VOL_BUFFER / src.name
                if dest.exists():
                    # Prefer the shard copy when re-merging; replace in place.
                    dest.unlink()
                shutil.move(str(src), str(dest))
                moved += 1

    index_path = merge_buffer_index(VOL_BUFFER)
    sample_count = len(list(VOL_BUFFER.glob(f"*{SAMPLE_SUFFIX}")))
    VOLUME.commit()
    return {
        "ok": True,
        "moved": moved,
        "sample_count": sample_count,
        "index": str(index_path),
        "buffer": str(VOL_BUFFER),
    }


@app.function(
    image=IMAGE,
    cpu=1,
    memory=512,
    timeout=60 * 30,
    volumes={"/vol": VOLUME},
)
def purge_orphans(manifest_vol_path: str) -> dict:
    """Delete buffer samples whose ids are not in the volume manifest."""
    sys.path.insert(0, "/root/scripts")
    from morpheus_materialize import merge_buffer_index
    from training.morpheus.curriculum.schema import CurriculumManifest
    from training.morpheus.trainer.buffer import SAMPLE_SUFFIX

    manifest = CurriculumManifest.load(_resolve_manifest(manifest_vol_path))
    keep: set[str] = set()
    for item in manifest.items:
        if int(item.class_id) == 5 and item.sample_seat is None:
            keep.add(f"{item.item_id}_s0")
            keep.add(f"{item.item_id}_s1")
        else:
            keep.add(str(item.item_id))

    VOL_BUFFER.mkdir(parents=True, exist_ok=True)
    deleted = 0
    kept = 0
    for path in list(VOL_BUFFER.glob(f"*{SAMPLE_SUFFIX}")):
        sample_id = path.name[: -len(SAMPLE_SUFFIX)]
        if sample_id in keep:
            kept += 1
            continue
        path.unlink()
        deleted += 1

    index_path = merge_buffer_index(VOL_BUFFER)
    sample_count = len(list(VOL_BUFFER.glob(f"*{SAMPLE_SUFFIX}")))
    VOLUME.commit()
    return {
        "ok": True,
        "kept": kept,
        "deleted": deleted,
        "sample_count": sample_count,
        "manifest_ids": len(keep),
        "index": str(index_path),
    }


@app.local_entrypoint()
def main(
    shards: int = 96,
    n_particles: int = 4,
    max_items: int = 0,
    skip_existing: bool = True,
    manifest: str = "/vol/morpheus/manifests/scraped-classes13.json",
    classes: str = "",
    purge_first: bool = True,
) -> None:
    """Fan out materialize shards, then merge into the Part 14 buffer.

    ``classes`` is a comma-separated filter (e.g. ``4,5``). Empty keeps all
    classes. Filtered runs leave other class files in ``/vol/morpheus/buffer``.
    When ``purge_first`` is true, delete buffer samples absent from the manifest.
    """
    n_shards = max(1, int(shards))
    class_ids = _parse_classes(classes)
    print(
        json.dumps(
            {
                "shards": n_shards,
                "n_particles": int(n_particles),
                "max_items": int(max_items) if int(max_items) > 0 else None,
                "skip_existing": bool(skip_existing),
                "manifest": str(manifest),
                "classes": sorted(class_ids) if class_ids is not None else None,
                "purge_first": bool(purge_first),
            },
            indent=2,
        ),
        flush=True,
    )
    purge_report = None
    if bool(purge_first):
        purge_report = purge_orphans.remote(str(manifest))
        print(json.dumps({"purge": purge_report}, indent=2), flush=True)

    args = [
        (
            i,
            n_shards,
            int(n_particles),
            int(max_items),
            bool(skip_existing),
            str(manifest),
            str(classes or ""),
        )
        for i in range(n_shards)
    ]
    shard_reports = list(materialize_shard.starmap(args))
    failures = [r for r in shard_reports if not r.get("ok")]
    merge_report = merge_buffer.remote()
    summary = {
        "ok": not failures and bool(merge_report.get("ok")),
        "shards": n_shards,
        "shard_failure_count": len(failures),
        "newly_written": sum(int(r.get("newly_written") or 0) for r in shard_reports),
        "skipped_existing": sum(
            int(r.get("skipped_existing") or 0) for r in shard_reports
        ),
        "failure_count": sum(int(r.get("failure_count") or 0) for r in shard_reports),
        "purge": purge_report,
        "merge": merge_report,
        "shard_failures_head": failures[:8],
    }
    print(json.dumps(summary, indent=2))
    if not summary["ok"]:
        raise SystemExit(1)
