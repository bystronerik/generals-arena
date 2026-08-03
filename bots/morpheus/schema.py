"""Versioned Morpheus model schema: manifest fields and army-bin edges.

Part 12 imports ``ARMY_BIN_EDGES`` for auxiliary labels — do not define a second
copy elsewhere.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

MANIFEST_VERSION = "1"
TENSOR_SCHEMA_VERSION = "morpheus-tensor-v1"
ACTION_SCHEMA_VERSION = "morpheus-action-v1"
ARCHITECTURE_VERSION = "morpheus-net-v1"

ARMY_SCALE_DEFAULT = 4096.0
N_ARMY_BINS = 16


def logarithmic_army_bin_edges(
    army_max: float = ARMY_SCALE_DEFAULT,
    n_bins: int = N_ARMY_BINS,
) -> tuple[float, ...]:
    """``n_bins`` logarithmic bins over ``[0, army_max]``; returns ``n_bins + 1`` edges."""
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if army_max <= 0:
        raise ValueError("army_max must be positive")
    edges = [0.0]
    log_max = math.log(army_max)
    for i in range(1, n_bins + 1):
        edges.append(float(math.exp(log_max * i / n_bins)))
    edges[-1] = army_max
    return tuple(edges)


ARMY_BIN_EDGES: tuple[float, ...] = logarithmic_army_bin_edges()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    *,
    weights_path: Path,
    architecture: dict[str, Any],
    quantization: dict[str, Any],
    training_run: dict[str, Any],
    runtime: str,
    artifact_file: str,
    army_scale: float = ARMY_SCALE_DEFAULT,
    army_bin_edges: tuple[float, ...] = ARMY_BIN_EDGES,
) -> dict[str, Any]:
    """Assemble a versioned manifest dict for a frozen export directory."""
    return {
        "manifest_version": MANIFEST_VERSION,
        "tensor_schema": TENSOR_SCHEMA_VERSION,
        "action_schema": ACTION_SCHEMA_VERSION,
        "architecture_version": ARCHITECTURE_VERSION,
        "architecture": architecture,
        "quantization": quantization,
        "training_run": training_run,
        "runtime": runtime,
        "artifact_file": artifact_file,
        "army_scale": army_scale,
        "army_bin_edges": list(army_bin_edges),
        "weights_sha256": sha256_file(weights_path),
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())
