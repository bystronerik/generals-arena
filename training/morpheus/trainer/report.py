"""Inspect / download helpers for Part 14 Modal entry points."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from training.morpheus.trainer.loop import inspect_run


def write_inspect_report(run_dir: Path, output: Path | None = None) -> dict[str, Any]:
    report = inspect_run(run_dir)
    path = Path(output) if output is not None else Path(run_dir) / "inspect.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def download_checkpoint(
    run_dir: Path,
    checkpoint_id: str,
    dest: Path,
) -> Path:
    """Copy an immutable checkpoint directory to a local destination."""
    src = Path(run_dir) / checkpoint_id
    if not src.is_dir():
        raise FileNotFoundError(f"checkpoint missing: {src}")
    dest = Path(dest)
    if dest.exists():
        raise FileExistsError(f"refusing to overwrite destination {dest}")
    shutil.copytree(src, dest)
    return dest
