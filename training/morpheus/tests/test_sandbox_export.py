"""Part 00b sandbox export preflight — local gate.

Runs under `-m morpheus`. Proves the probe architecture, candidate inventory,
static 8-bit FX path (when the host qengine exists), and report decision logic.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from training.morpheus.export_preflight.candidates import (
    RSS_BOUND_BYTES,
    try_safetensors_weight_only_int8,
    try_torch_fx_static,
    try_torch_jit_float,
)
from training.morpheus.export_preflight.fixtures import (
    BATCH_ENEMY_PROPOSAL,
    BATCH_LEAF,
    BATCH_ROOT,
    BATCH_SHAPES,
    all_batch_fixtures,
)
from training.morpheus.export_preflight.measure import run_export_preflight
from training.morpheus.export_preflight.model import (
    BOARD,
    IN_CHANNELS,
    N_BLOCKS,
    TRUNK_CHANNELS,
    architecture_summary,
    make_probe,
    parameter_count,
)
from training.morpheus.export_preflight.pins import (
    JUDGE_REQUIREMENTS,
    SANDBOX_REQUIREMENTS,
    pin_audit,
    versions_match,
)
from training.morpheus.export_preflight.report import (
    build_report,
    decide_pass,
    write_report,
)

pytestmark = pytest.mark.morpheus


def test_sandbox_and_judge_pins_agree():
    audit = pin_audit()
    assert SANDBOX_REQUIREMENTS.is_file()
    assert JUDGE_REQUIREMENTS.is_file()
    assert audit["judge_matches_sandbox"] is True
    assert audit["sandbox_pins"]["torch"] == "2.13.0"
    assert audit["sandbox_pins"]["safetensors"] == "0.8.0"
    assert "onnxruntime" in audit["unavailable_in_sandbox"]


def test_versions_match_accepts_cpu_local_tag():
    assert versions_match("2.13.0", "2.13.0") is True
    assert versions_match("2.13.0+cpu", "2.13.0") is True
    assert versions_match("2.12.0+cpu", "2.13.0") is False
    assert versions_match(None, "2.13.0") is False


def test_probe_architecture_shapes():
    model = make_probe(seed=0, n_blocks=3)
    x = torch.randn(2, IN_CHANNELS, BOARD, BOARD)
    with torch.no_grad():
        out = model(x)
        policy, pass_logit, wdl = out.policy, out.pass_logit, out.wdl_logits
    assert policy.shape == (2, 9, BOARD, BOARD)
    assert pass_logit.shape == (2, 1)
    assert wdl.shape == (2, 3)
    summary = architecture_summary(make_probe(seed=0, n_blocks=N_BLOCKS))
    assert summary["trunk_channels"] == TRUNK_CHANNELS
    assert summary["n_blocks"] == N_BLOCKS
    assert parameter_count(make_probe(seed=0)) > 100_000


def test_batch_fixtures_cover_root_leaf_enemy():
    fixtures = all_batch_fixtures(seed=1)
    assert [f.batch for f in fixtures] == [
        BATCH_ROOT,
        BATCH_LEAF,
        BATCH_ENEMY_PROPOSAL,
    ]
    assert list(BATCH_SHAPES) == [1, 4, 64]
    assert [f.name for f in fixtures] == ["root", "leaf", "enemy_proposal"]


def test_torch_fx_static_qnnpack_when_available(tmp_path: Path):
    supported = list(torch.backends.quantized.supported_engines)
    if "qnnpack" not in supported:
        pytest.skip(f"qnnpack not in {supported}")
    result = try_torch_fx_static(
        engine="qnnpack", seed=0, n_blocks=3, work_dir=tmp_path
    )
    assert result["error"] is None
    assert result["exported"] is True
    assert result["reloaded"] is True
    assert result["executed"] is True
    assert result["has_static_8bit_ops"] is True
    assert result["viable"] is True
    assert any(op.startswith("GroupNorm:") for op in result["fallback_operators"])
    assert result["float_to_export_error"] is not None
    assert result["serialized_bytes"] > 0
    assert result["peak_rss_bytes"] < RSS_BOUND_BYTES
    for name in ("root", "leaf", "enemy_proposal"):
        assert result["batch_results"][name]["ok"] is True


def test_float_and_weight_only_are_not_viable(tmp_path: Path):
    float_result = try_torch_jit_float(
        seed=0, n_blocks=2, work_dir=tmp_path
    )
    assert float_result["executed"] is True
    assert float_result["has_static_8bit_ops"] is False
    assert float_result["viable"] is False

    weight_result = try_safetensors_weight_only_int8(
        seed=0, n_blocks=2, work_dir=tmp_path
    )
    assert weight_result["executed"] is True
    assert weight_result["has_static_8bit_ops"] is False
    assert weight_result["viable"] is False
    assert "entire_forward:float32_after_dequant" in weight_result[
        "fallback_operators"
    ]


def test_unavailable_engine_is_reported(tmp_path: Path):
    supported = list(torch.backends.quantized.supported_engines)
    missing = next(
        (e for e in ("fbgemm", "x86") if e not in supported),
        None,
    )
    if missing is None:
        pytest.skip("all x86 engines available on this host")
    result = try_torch_fx_static(
        engine=missing, seed=0, n_blocks=2, work_dir=tmp_path
    )
    assert result["exported"] is False
    assert result["viable"] is False
    assert f"qengine:{missing}" in result["unsupported_operators"]


def test_decide_pass_and_report(tmp_path: Path):
    # Prefer a short probe in the unit gate; full 12-block run is the script.
    n_blocks = 3 if "qnnpack" in torch.backends.quantized.supported_engines else 2
    result = run_export_preflight(
        seed=0, n_blocks=n_blocks, work_dir=tmp_path / "work"
    )
    decision = decide_pass(result)
    if "qnnpack" in torch.backends.quantized.supported_engines:
        assert decision["verdict"] == "yes"
        assert "torch_fx_static_qnnpack" in decision["viable_candidates"]
    report = build_report(result)
    json_path = tmp_path / "morpheus-export-preflight.json"
    md_path = tmp_path / "morpheus-export-preflight.md"
    write_report(report, json_path=json_path, md_path=md_path)
    loaded = json.loads(json_path.read_text())
    assert loaded["part"] == "00b-sandbox-export-preflight"
    assert "Verdict:" in md_path.read_text()
    # Weight-only and float paths must not be counted as the static answer.
    names = {c["name"]: c for c in loaded["candidates"]}
    assert names["torch_jit_float32"]["viable"] is False
    assert names["safetensors_weight_only_int8"]["viable"] is False


def test_decide_pass_fails_without_viable_candidate():
    fake = {
        "batch_shapes": [1, 4, 64],
        "pins": {
            "installed_vs_sandbox": {
                "torch": {"match": True, "sandbox": "2.13.0", "installed": "2.13.0"}
            }
        },
        "candidates": [
            {
                "name": "torch_jit_float32",
                "viable": False,
                "has_static_8bit_ops": False,
                "fallback_operators": [],
            }
        ],
    }
    decision = decide_pass(fake)
    assert decision["verdict"] == "no"
    assert decision["checks"]["any_static_8bit_viable"] is False
