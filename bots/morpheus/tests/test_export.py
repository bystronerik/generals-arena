"""Part 04 — export adapters, manifest, and inference load."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.morpheus

from export import (
    DEFAULT_ARTIFACT_FILE,
    MANIFEST_NAME,
    export_from_checkpoint,
    load_checkpoint_state,
)
from inference import load_session, validate_manifest
from network import BOARD, IN_CHANNELS, make_model
from schema import (
    ACTION_SCHEMA_VERSION,
    ARCHITECTURE_VERSION,
    ARMY_BIN_EDGES,
    TENSOR_SCHEMA_VERSION,
    build_manifest,
    read_manifest,
    sha256_file,
    write_manifest,
)

REPO = Path(__file__).resolve().parents[3]
TINY_CHECKPOINT = REPO / "training/morpheus/tests/fixtures/tiny-checkpoint"


@pytest.fixture
def tiny_checkpoint(tmp_path: Path) -> Path:
    if TINY_CHECKPOINT.is_dir() and (TINY_CHECKPOINT / "state_dict.pt").is_file():
        return TINY_CHECKPOINT
    ckpt_dir = tmp_path / "tiny-checkpoint"
    ckpt_dir.mkdir()
    model = make_model(seed=0, n_blocks=3)
    torch.save(model.state_dict(), ckpt_dir / "state_dict.pt")
    (ckpt_dir / "meta.json").write_text(json.dumps({"seed": 0, "n_blocks": 3}) + "\n")
    return ckpt_dir


def test_export_writes_manifest_and_loads(tiny_checkpoint: Path, tmp_path: Path):
    out = tmp_path / "export"
    result = export_from_checkpoint(tiny_checkpoint, out)
    assert result.manifest_path.is_file()
    assert result.artifact_path.is_file()
    manifest = read_manifest(result.manifest_path)
    assert manifest["tensor_schema"] == TENSOR_SCHEMA_VERSION
    assert manifest["action_schema"] == ACTION_SCHEMA_VERSION
    assert manifest["architecture_version"] == ARCHITECTURE_VERSION
    assert tuple(manifest["army_bin_edges"]) == ARMY_BIN_EDGES
    assert manifest["weights_sha256"] == sha256_file(result.artifact_path)

    session = load_session(out)
    x = torch.randn(1, IN_CHANNELS, BOARD, BOARD)
    out_fwd = session.forward(x)
    assert out_fwd.policy.shape == (1, 9, BOARD, BOARD)
    assert out_fwd.wdl_logits.shape == (1, 3)


def test_manifest_validation_rejects_bad_digest(tmp_path: Path, tiny_checkpoint: Path):
    out = tmp_path / "export"
    export_from_checkpoint(tiny_checkpoint, out)
    manifest = read_manifest(out / MANIFEST_NAME)
    manifest["weights_sha256"] = "deadbeef"
    write_manifest(out / MANIFEST_NAME, manifest)
    with pytest.raises(ValueError, match="SHA-256"):
        validate_manifest(manifest, out / DEFAULT_ARTIFACT_FILE)


def test_checkpoint_round_trip(tmp_path: Path):
    ckpt = tmp_path / "state"
    ckpt.mkdir()
    model = make_model(seed=7, n_blocks=4)
    torch.save(model.state_dict(), ckpt / "state_dict.pt")
    (ckpt / "meta.json").write_text(json.dumps({"seed": 7, "n_blocks": 4}) + "\n")
    loaded = load_checkpoint_state(ckpt)
    assert loaded["meta"]["n_blocks"] == 4
    from export import build_model_from_checkpoint

    restored = build_model_from_checkpoint(loaded)
    x = torch.randn(2, IN_CHANNELS, BOARD, BOARD)
    with torch.no_grad():
        assert torch.allclose(model(x).policy, restored(x).policy)
