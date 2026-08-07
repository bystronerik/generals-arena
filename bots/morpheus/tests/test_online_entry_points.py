"""Part 09a Phase 5 — dedicated online export entry points."""
from __future__ import annotations

import json
import platform
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.morpheus

from export import (  # noqa: E402
    ONLINE_PARITY_LIMITS,
    POLICY_OUTPUT_NAMES,
    POLICY_WDL_OUTPUT_NAMES,
    assert_online_parity,
    export_from_checkpoint,
    export_static_int8,
)
from inference import load_session  # noqa: E402
from network import BOARD, IN_CHANNELS, make_model  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
TINY_CHECKPOINT = REPO / "training/morpheus/tests/fixtures/tiny-checkpoint"


def _supported() -> list[str]:
    try:
        return list(torch.backends.quantized.supported_engines)
    except Exception:  # noqa: BLE001
        return []


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


def test_float_online_entry_points_match_full_heads():
    model = make_model(seed=3, n_blocks=3)
    x = torch.randn(2, IN_CHANNELS, BOARD, BOARD)
    with torch.no_grad():
        full = model(x)
        policy, pass_logit = model.forward_policy(x)
        p2, pass2, wdl = model.forward_policy_wdl(x)
    assert torch.allclose(policy, full.policy)
    assert torch.allclose(pass_logit, full.pass_logit)
    assert torch.allclose(p2, full.policy)
    assert torch.allclose(pass2, full.pass_logit)
    assert torch.allclose(wdl, full.wdl_logits)


def test_qnnpack_online_entry_parity(tiny_checkpoint: Path, tmp_path: Path):
    if "qnnpack" not in _supported():
        pytest.skip(f"qnnpack not in {_supported()}")
    out = tmp_path / "qnnpack-online"
    result = export_from_checkpoint(tiny_checkpoint, out, fmt="int8", qengine="qnnpack")
    assert result.qengine == "qnnpack"
    for name in POLICY_OUTPUT_NAMES:
        key = f"{name}_mae"
        assert key in result.online_float_to_export_mae["policy"]
        assert result.online_float_to_export_mae["policy"][key] <= ONLINE_PARITY_LIMITS[key]
    for name in POLICY_WDL_OUTPUT_NAMES:
        key = f"{name}_mae"
        assert key in result.online_float_to_export_mae["policy_wdl"]
        assert (
            result.online_float_to_export_mae["policy_wdl"][key]
            <= ONLINE_PARITY_LIMITS[key]
        )
    assert_online_parity(result.online_float_to_export_mae["policy"])
    assert_online_parity(result.online_float_to_export_mae["policy_wdl"])

    session = load_session(out)
    x = torch.randn(4, IN_CHANNELS, BOARD, BOARD)
    policy, pass_logit = session.forward_policy(x)
    policy2, pass2, wdl = session.forward_policy_wdl(x)
    assert policy.shape == (4, 9, BOARD, BOARD)
    assert pass_logit.shape == (4, 1)
    assert policy2.shape == (4, 9, BOARD, BOARD)
    assert pass2.shape == (4, 1)
    assert wdl.shape == (4, 3)


@pytest.mark.parametrize("engine", ["fbgemm", "x86"])
def test_x86_online_entry_parity_when_available(
    engine: str, tiny_checkpoint: Path, tmp_path: Path
):
    """fbgemm / x86 run on pinned Linux cores; skip on Darwin qnnpack-only hosts."""
    if engine not in _supported():
        pytest.skip(
            f"{engine} not in {_supported()} on {platform.platform()}; "
            "run on pinned Linux (Modal export preflight) to exercise this path"
        )
    out = tmp_path / f"{engine}-online"
    # Short trunk for unit time; full 12-block export is the Modal/script seat.
    model = make_model(seed=0, n_blocks=3)
    result = export_static_int8(model, out, qengine=engine, seed=0)
    assert result.qengine == engine
    assert_online_parity(result.online_float_to_export_mae["policy"])
    assert_online_parity(result.online_float_to_export_mae["policy_wdl"])
    session = load_session(out)
    x = torch.randn(1, IN_CHANNELS, BOARD, BOARD)
    policy, pass_logit = session.forward_policy(x)
    _, _, wdl = session.forward_policy_wdl(x)
    assert policy.shape[0] == 1
    assert pass_logit.shape == (1, 1)
    assert wdl.shape == (1, 3)
