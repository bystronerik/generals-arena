"""Bounded pilot learning loop: reconstruct → loss → AdamW → checkpoint."""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from arena.paths import REPO_ROOT
from training.morpheus.curriculum.schema import CurriculumManifest
from training.morpheus.objective.config import load_pilot_objective_bundle
from training.morpheus.pilot.batch import PilotSample, build_pilot_sample
from training.morpheus.pilot.checkpoint import load_pilot_checkpoint, save_pilot_checkpoint
from training.morpheus.pilot.step import pilot_train_step

DEFAULT_OBJECTIVE = REPO_ROOT / "training/morpheus/configs/pilot-objective.json"
DEFAULT_MANIFEST = REPO_ROOT / "training/morpheus/manifests/pilot-class1-scraped.json"


def _log(msg: str) -> None:
    """Flush immediately so Modal shows progress during long reconstruct phases."""
    print(f"[pilot_learn] {msg}", flush=True)


def _ensure_bot_path() -> None:
    bot = REPO_ROOT / "bots" / "morpheus"
    for entry in (REPO_ROOT, REPO_ROOT / "bots", bot):
        s = str(entry)
        if s not in sys.path:
            sys.path.insert(0, s)


def load_pilot_learn_config(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("pilot learn config must be an object")
    return data


def _select_items(manifest: CurriculumManifest, *, max_items: int) -> list:
    items = [i for i in manifest.items if i.class_id == 1 and i.sample_seat is not None]
    if not items:
        raise ValueError("manifest has no class-1 items with sample_seat")
    # Interleave source labels so the smoke batch covers wins and losses.
    by_label: dict[str, list] = {}
    for item in items:
        by_label.setdefault(item.source_label, []).append(item)
    ordered: list = []
    buckets = [list(v) for v in by_label.values()]
    while buckets and len(ordered) < int(max_items):
        next_buckets = []
        for bucket in buckets:
            if not bucket:
                continue
            ordered.append(bucket.pop(0))
            if bucket:
                next_buckets.append(bucket)
            if len(ordered) >= int(max_items):
                break
        buckets = next_buckets
    return ordered[: int(max_items)]


def _materialize_samples(
    items: Sequence,
    *,
    repo_root: Path,
    n_particles: int,
) -> list[PilotSample]:
    cache: dict[str, dict[str, Any]] = {}
    samples: list[PilotSample] = []
    total = len(items)
    _log(f"materialize start: {total} items n_particles={n_particles}")
    t_all = time.perf_counter()
    for index, item in enumerate(items, start=1):
        t_item = time.perf_counter()
        _log(
            f"materialize {index}/{total} begin "
            f"item_id={item.item_id} seat={item.sample_seat} "
            f"prefix_len={item.prefix_len} source={item.source_label}"
        )
        samples.append(
            build_pilot_sample(
                item,
                trajectories_root=repo_root,
                n_particles=n_particles,
                terminal_cache=cache,
            )
        )
        _log(
            f"materialize {index}/{total} done "
            f"item_s={time.perf_counter() - t_item:.2f}"
        )
    _log(
        f"materialize complete: {len(samples)} samples "
        f"wall_s={time.perf_counter() - t_all:.2f}"
    )
    return samples


def run_pilot_learn(
    config_path: Path | str,
    *,
    output_dir: Path,
    repo_root: Path | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """
    Thin learning smoke: a few AdamW steps on class-1 samples, one checkpoint.

    Success: finite loss that decreases vs step 0; checkpoint reloads.
    """
    t_run = time.perf_counter()
    root = repo_root or REPO_ROOT
    _log(f"start config={config_path} output_dir={output_dir} repo_root={root}")

    cfg = load_pilot_learn_config(config_path)
    objective_path = Path(cfg.get("objective") or DEFAULT_OBJECTIVE)
    if not objective_path.is_file():
        objective_path = root / objective_path
    manifest_path = Path(cfg.get("manifest") or DEFAULT_MANIFEST)
    if not manifest_path.is_file():
        manifest_path = root / manifest_path

    _log(f"load objective={objective_path}")
    bundle = load_pilot_objective_bundle(objective_path)
    objective = bundle["training"]
    _log(f"load manifest={manifest_path}")
    manifest = CurriculumManifest.load(manifest_path)

    max_items = int(cfg.get("max_items") or 8)
    steps = int(cfg.get("steps") or 50)
    batch_size = int(cfg.get("batch_size") or 2)
    lr = float(cfg.get("learning_rate") or 1e-3)
    weight_decay = float(cfg.get("weight_decay") or 1e-4)
    n_particles = int(cfg.get("n_particles") or 4)
    seed = int(cfg.get("seed") or 0)
    n_blocks = int(cfg.get("n_blocks") or 12)
    log_every = int(cfg.get("log_every_steps") or max(1, steps // 10))

    _log(
        f"config max_items={max_items} steps={steps} batch_size={batch_size} "
        f"lr={lr} n_particles={n_particles} n_blocks={n_blocks} seed={seed} "
        f"objective={objective.name}"
    )

    items = _select_items(manifest, max_items=max_items)
    labels = sorted({i.source_label for i in items})
    _log(f"selected {len(items)} items sources={labels}")

    samples = _materialize_samples(items, repo_root=root, n_particles=n_particles)
    if not samples:
        raise RuntimeError("no pilot samples materialized")

    _ensure_bot_path()
    from network import make_model

    torch.manual_seed(seed)
    np.random.default_rng(seed)
    cuda_available = bool(torch.cuda.is_available())
    if device is None:
        device = "cuda" if cuda_available else "cpu"
    dev = torch.device(device)
    _log(
        f"device request={device} resolved={dev} "
        f"cuda_available={cuda_available} "
        f"cuda_device_count={torch.cuda.device_count() if cuda_available else 0}"
    )
    if str(dev).startswith("cuda") and not cuda_available:
        raise RuntimeError("requested cuda but torch.cuda.is_available() is False")

    _log(f"build model n_blocks={n_blocks}")
    model = make_model(seed=seed, n_blocks=n_blocks).to(dev)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_params = sum(p.numel() for p in model.parameters())
    _log(f"model ready params={n_params}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t_train = time.perf_counter()
    history: list[dict[str, Any]] = []
    loss0: float | None = None
    last_loss = float("nan")

    _log(f"train start steps={steps} batch_size={batch_size}")
    rng = np.random.default_rng(seed)
    for step in range(steps):
        if len(samples) <= batch_size:
            batch = list(samples)
        else:
            idxs = rng.choice(len(samples), size=batch_size, replace=False)
            batch = [samples[int(i)] for i in idxs]
        total, terms = pilot_train_step(
            model=model,
            optimizer=optimizer,
            samples=batch,
            objective=objective,
            device=dev,
        )
        if not math.isfinite(total):
            raise RuntimeError(f"non-finite loss at step {step}: {total}")
        if loss0 is None:
            loss0 = total
            _log(f"train step 0 loss={total:.6f}")
        last_loss = total
        history.append({"step": step, "loss": total, "terms": terms.to_dict()["terms"]})
        if step > 0 and (step % log_every == 0 or step == steps - 1):
            _log(
                f"train step {step}/{steps - 1} loss={total:.6f} "
                f"elapsed_s={time.perf_counter() - t_train:.2f}"
            )

    train_s = time.perf_counter() - t_train
    _log(f"train complete wall_s={train_s:.2f} loss_final={last_loss}")

    ckpt_dir = output_dir / "checkpoint"
    _log(f"checkpoint write {ckpt_dir}")
    save_pilot_checkpoint(
        model,
        ckpt_dir,
        meta={
            "seed": seed,
            "n_blocks": n_blocks,
            "steps": steps,
            "objective": objective.name,
            "manifest": str(manifest_path),
            "promotable_main_run": False,
        },
    )
    # Reload check.
    _log("checkpoint reload check")
    reloaded, meta = load_pilot_checkpoint(ckpt_dir)
    reloaded_ok = True
    try:
        reloaded.eval()
        with torch.no_grad():
            x = torch.as_tensor(samples[0].tensor, dtype=torch.float32).unsqueeze(0)
            _ = reloaded(x.to(dev) if next(reloaded.parameters()).is_cuda else x)
    except Exception as exc:  # noqa: BLE001
        reloaded_ok = False
        _log(f"checkpoint reload FAILED: {exc}")
    _ = meta

    decreased = loss0 is not None and last_loss < loss0
    wall_s = time.perf_counter() - t_run
    _log(
        f"done ok_candidate={reloaded_ok and math.isfinite(last_loss) and decreased} "
        f"loss_step0={loss0} loss_final={last_loss} decreased={decreased} "
        f"reloadable={reloaded_ok} wall_s={wall_s:.2f} "
        f"a100_hours={wall_s / 3600.0:.6f}"
    )
    report = {
        "status": "pilot_learn",
        "ok": bool(reloaded_ok and math.isfinite(last_loss) and decreased),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "objective": str(objective_path),
            "manifest": str(manifest_path),
            "max_items": max_items,
            "steps": steps,
            "batch_size": batch_size,
            "learning_rate": lr,
            "n_particles": n_particles,
            "seed": seed,
            "n_blocks": n_blocks,
            "device": str(dev),
            "cuda_available": cuda_available,
        },
        "bundle": {
            "selected_candidate": bundle["selected_candidate"],
            "promotable_main_run": False,
            "status": bundle["status"],
        },
        "sample_count": len(samples),
        "source_labels": sorted({s.source_label for s in samples}),
        "loss_step0": loss0,
        "loss_final": last_loss,
        "loss_decreased": decreased,
        "history": history,
        "checkpoint": str(ckpt_dir),
        "checkpoint_reloadable": reloaded_ok,
        "train_wall_s": train_s,
        "wall_s": wall_s,
        "a100_hours": wall_s / 3600.0,
        "accounting_line": "learning_curve_pilot",
        "notes": [
            "Thin pilot smoke only. Not Part 14. Not a Part 13 yes.",
            "A100 wall time charges to learning_curve_pilot accounting notes.",
        ],
    }
    return report
