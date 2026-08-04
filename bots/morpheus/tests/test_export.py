"""Part 04 — export adapters, manifest, and inference load."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.morpheus

from export import (
    DEFAULT_ARTIFACT_FILE,
    MANIFEST_NAME,
    ONLINE_PARITY_LIMITS,
    POLICY_ARTIFACT_FILE,
    POLICY_WDL_ARTIFACT_FILE,
    assert_online_parity,
    export_from_checkpoint,
    load_checkpoint_state,
)
from inference import load_session, validate_manifest
from network import BOARD, IN_CHANNELS, N_ARMY_BINS, make_model
from schema import (
    ACTION_SCHEMA_VERSION,
    ARCHITECTURE_VERSION,
    ARMY_BIN_EDGES,
    MANIFEST_VERSION,
    TENSOR_SCHEMA_VERSION,
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
    model = make_model(seed=0, n_blocks=12)
    torch.save(model.state_dict(), ckpt_dir / "state_dict.pt")
    (ckpt_dir / "meta.json").write_text(json.dumps({"seed": 0, "n_blocks": 12}) + "\n")
    return ckpt_dir


def test_export_writes_manifest_and_loads(tiny_checkpoint: Path, tmp_path: Path):
    out = tmp_path / "export"
    result = export_from_checkpoint(tiny_checkpoint, out)
    assert result.manifest_path.is_file()
    assert result.artifact_path.is_file()
    assert result.policy_artifact_path.is_file()
    assert result.policy_wdl_artifact_path.is_file()
    manifest = read_manifest(result.manifest_path)
    assert manifest["manifest_version"] == MANIFEST_VERSION
    assert manifest["tensor_schema"] == TENSOR_SCHEMA_VERSION
    assert manifest["action_schema"] == ACTION_SCHEMA_VERSION
    assert manifest["architecture_version"] == ARCHITECTURE_VERSION
    assert tuple(manifest["army_bin_edges"]) == ARMY_BIN_EDGES
    assert manifest["weights_sha256"] == sha256_file(result.artifact_path)
    assert "engine" in manifest["quantization"]
    assert manifest["architecture"]["n_army_bins"] == N_ARMY_BINS
    assert manifest["online_entry_points"]["policy"] == POLICY_ARTIFACT_FILE
    assert manifest["online_entry_points"]["policy_wdl"] == POLICY_WDL_ARTIFACT_FILE
    assert manifest["online_weights_sha256"]["policy"] == sha256_file(
        result.policy_artifact_path
    )
    assert_online_parity(result.online_float_to_export_mae["policy"])
    assert_online_parity(result.online_float_to_export_mae["policy_wdl"])

    session = load_session(out)
    assert session.policy_module is not None
    assert session.policy_wdl_module is not None
    x = torch.randn(1, IN_CHANNELS, BOARD, BOARD)
    out_fwd = session.forward(x)
    assert out_fwd.policy.shape == (1, 9, BOARD, BOARD)
    assert out_fwd.wdl_logits.shape == (1, 3)
    assert out_fwd.hidden_owner.shape == (1, 1, BOARD, BOARD)
    assert out_fwd.enemy_army_bins.shape == (1, N_ARMY_BINS, BOARD, BOARD)
    assert out_fwd.enemy_general.shape == (1, 1, BOARD, BOARD)
    assert out_fwd.hidden_castle.shape == (1, 1, BOARD, BOARD)
    assert out_fwd.land_margin.shape == (1, 1)
    assert out_fwd.army_margin.shape == (1, 1)
    assert out_fwd.castle_margin.shape == (1, 1)
    assert out_fwd.turns_to_termination.shape == (1, 1)
    # Aux heads must come from the full export, not silent zeros.
    assert float(out_fwd.hidden_owner.abs().sum()) + float(
        out_fwd.enemy_army_bins.abs().sum()
    ) > 0.0

    policy, pass_logit = session.forward_policy(x)
    assert policy.shape == (1, 9, BOARD, BOARD)
    assert pass_logit.shape == (1, 1)
    policy2, pass2, wdl = session.forward_policy_wdl(x)
    assert policy2.shape == (1, 9, BOARD, BOARD)
    assert pass2.shape == (1, 1)
    assert wdl.shape == (1, 3)


def test_manifest_validation_rejects_bad_digest(tmp_path: Path, tiny_checkpoint: Path):
    out = tmp_path / "export"
    export_from_checkpoint(tiny_checkpoint, out)
    manifest = read_manifest(out / MANIFEST_NAME)
    manifest["weights_sha256"] = "deadbeef"
    write_manifest(out / MANIFEST_NAME, manifest)
    with pytest.raises(ValueError, match="SHA-256"):
        validate_manifest(manifest, out / DEFAULT_ARTIFACT_FILE)


def test_manifest_validation_rejects_bad_version(tmp_path: Path, tiny_checkpoint: Path):
    out = tmp_path / "export"
    export_from_checkpoint(tiny_checkpoint, out)
    manifest = read_manifest(out / MANIFEST_NAME)
    manifest["manifest_version"] = "999"
    with pytest.raises(ValueError, match="manifest version"):
        validate_manifest(manifest, out / DEFAULT_ARTIFACT_FILE)


def test_manifest_validation_rejects_architecture_drift(
    tmp_path: Path, tiny_checkpoint: Path
):
    out = tmp_path / "export"
    export_from_checkpoint(tiny_checkpoint, out)
    manifest = read_manifest(out / MANIFEST_NAME)
    manifest["architecture"]["in_channels"] = 7
    with pytest.raises(ValueError, match="in_channels"):
        validate_manifest(manifest, out / DEFAULT_ARTIFACT_FILE)


def test_load_rejects_unsupported_qengine(tmp_path: Path, tiny_checkpoint: Path):
    out = tmp_path / "export"
    export_from_checkpoint(tiny_checkpoint, out)
    manifest = read_manifest(out / MANIFEST_NAME)
    manifest["quantization"]["engine"] = "not-a-real-engine"
    write_manifest(out / MANIFEST_NAME, manifest)
    with pytest.raises(ValueError, match="quantized engine"):
        load_session(out)


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


def test_online_parity_limits_reject_excess():
    with pytest.raises(ValueError, match="policy_mae"):
        assert_online_parity(
            {"policy_mae": ONLINE_PARITY_LIMITS["policy_mae"] + 0.1},
        )
