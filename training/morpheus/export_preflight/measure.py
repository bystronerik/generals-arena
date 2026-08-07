"""Run the sandbox export preflight measurement."""
from __future__ import annotations

import platform
import tempfile
from pathlib import Path
from typing import Any

import torch

from training.morpheus.export_preflight.candidates import probe_all_candidates
from training.morpheus.export_preflight.fixtures import BATCH_SHAPES
from training.morpheus.export_preflight.model import (
    architecture_summary,
    make_probe,
)
from training.morpheus.export_preflight.pins import pin_audit


def run_export_preflight(
    *,
    requirements_path: Path | None = None,
    seed: int = 0,
    n_blocks: int = 12,
    trunk_channels: int | None = None,
    expansion: int | None = None,
    work_dir: Path | None = None,
    seat: str = "local",
) -> dict[str, Any]:
    """Measure every sandbox-available export candidate on this host."""
    pins = pin_audit(requirements_path)
    size_kwargs = {}
    if trunk_channels is not None:
        size_kwargs["trunk_channels"] = trunk_channels
    if expansion is not None:
        size_kwargs["expansion"] = expansion
    model = make_probe(seed=seed, n_blocks=n_blocks, **size_kwargs)
    arch = architecture_summary(model)

    own_tmpdir = work_dir is None
    root = (
        Path(tempfile.mkdtemp(prefix="morpheus-export-preflight-"))
        if own_tmpdir
        else work_dir
    )
    assert root is not None
    root.mkdir(parents=True, exist_ok=True)

    candidates = probe_all_candidates(
        seed=seed,
        n_blocks=n_blocks,
        trunk_channels=trunk_channels,
        expansion=expansion,
        work_dir=root,
    )
    return {
        "part": "00b-sandbox-export-preflight",
        "host": {
            "seat": seat,
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "quantized_engine": torch.backends.quantized.engine,
            "supported_qengines": list(
                torch.backends.quantized.supported_engines
            ),
        },
        "pins": pins,
        "architecture": arch,
        "batch_shapes": list(BATCH_SHAPES),
        "batch_roles": {
            "1": "root",
            "4": "leaf",
            "64": "enemy_proposal",
        },
        "candidates": candidates,
        "work_dir": str(root),
    }
