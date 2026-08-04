#!/usr/bin/env python3
"""Modal CPU seat for Part 09a Phase 5: fbgemm / x86 online entry points.

Usage:
    modal run scripts/morpheus_modal_online_entry_points.py

Exports policy-only and policy+WDL static 8-bit graphs on pinned Linux cores
for each available x86 quantized engine, checks soft parity limits, and writes:

    docs/research/measurements/morpheus-online-entry-points.{json,md}

Does not use a GPU and does not spend A100 budget.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import modal

REPO = Path(__file__).resolve().parents[1]

TORCH_PIN = "torch==2.13.0"
NUMPY_PIN = "numpy==2.4.6"
SAFETENSORS_PIN = "safetensors==0.8.0"
_SANDBOX_REST = (
    NUMPY_PIN,
    "scipy==1.18.0",
    "pandas==3.0.5",
    "scikit-learn==1.9.0",
    "jax==0.11.0",
    "jaxlib==0.11.0",
    "numba==0.66.0",
    "networkx==3.6.1",
    SAFETENSORS_PIN,
    "gymnasium==1.3.0",
)

DEFAULT_JSON = (
    REPO / "docs/research/measurements/morpheus-online-entry-points.json"
)
DEFAULT_MD = REPO / "docs/research/measurements/morpheus-online-entry-points.md"

IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        TORCH_PIN,
        index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install(*_SANDBOX_REST)
    .env({"PYTHONPATH": "/root:/root/bots/morpheus"})
    .add_local_dir(
        str(REPO / "bots" / "morpheus"),
        remote_path="/root/bots/morpheus",
        copy=True,
    )
    .add_local_dir(
        str(REPO / "training" / "morpheus" / "tests" / "fixtures"),
        remote_path="/root/training/morpheus/tests/fixtures",
        copy=True,
    )
    .add_local_file(
        str(REPO / "requirements-sandbox.txt"),
        remote_path="/root/requirements-sandbox.txt",
        copy=True,
    )
)

app = modal.App("morpheus-online-entry-points")


def _probe_engine(
    *,
    engine: str,
    seed: int,
    n_blocks: int,
    work_dir: Path,
) -> dict[str, Any]:
    import platform

    import torch

    from export import (  # noqa: E402 — remote PYTHONPATH
        ONLINE_PARITY_LIMITS,
        assert_online_parity,
        export_static_int8,
    )
    from inference import load_session  # noqa: E402
    from network import BOARD, IN_CHANNELS, make_model  # noqa: E402

    supported = list(torch.backends.quantized.supported_engines)
    row: dict[str, Any] = {
        "engine": engine,
        "supported": engine in supported,
        "exported": False,
        "reloaded": False,
        "parity_ok": False,
        "online_float_to_export_mae": None,
        "error": None,
        "host": platform.platform(),
    }
    if engine not in supported:
        row["error"] = f"engine {engine!r} not in supported_engines={supported}"
        return row

    out = work_dir / engine
    out.mkdir(parents=True, exist_ok=True)
    try:
        model = make_model(seed=seed, n_blocks=n_blocks)
        result = export_static_int8(model, out, qengine=engine, seed=seed)
        row["exported"] = True
        row["qengine"] = result.qengine
        row["online_float_to_export_mae"] = result.online_float_to_export_mae
        row["full_float_to_export_mae"] = result.float_to_export_mae
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
        row["reloaded"] = True
        row["parity_ok"] = True
        row["parity_limits"] = dict(ONLINE_PARITY_LIMITS)
    except Exception as exc:  # noqa: BLE001 — remote probe
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


@app.function(image=IMAGE, cpu=1, memory=2048, timeout=60 * 30)
def run_online_entry_points_cpu(
    seed: int = 0,
    n_blocks: int = 12,
) -> dict[str, Any]:
    """Export and check online entry points for fbgemm and x86 on Linux CPU."""
    import platform
    import tempfile

    import torch

    work = Path(tempfile.mkdtemp(prefix="morpheus-online-entry-"))
    supported = list(torch.backends.quantized.supported_engines)
    engines = ("fbgemm", "x86")
    rows = [
        _probe_engine(
            engine=engine,
            seed=seed,
            n_blocks=n_blocks,
            work_dir=work,
        )
        for engine in engines
    ]
    passed = [r["engine"] for r in rows if r.get("parity_ok")]
    failed = [r["engine"] for r in rows if not r.get("parity_ok")]
    verdict = "yes" if passed and not failed else ("partial" if passed else "no")
    return {
        "part": "09a-phase5-online-entry-points",
        "seat": "modal-cpu",
        "seed": seed,
        "n_blocks": n_blocks,
        "host": {
            "platform": platform.platform(),
            "supported_qengines": supported,
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "engines": rows,
        "decision": {
            "verdict": verdict,
            "passed_engines": passed,
            "failed_engines": failed,
            "criterion": (
                "fbgemm and x86 online policy / policy+WDL exports reload and "
                "pass ONLINE_PARITY_LIMITS on pinned Linux CPU cores"
            ),
        },
    }


def _write_report(result: dict[str, Any]) -> tuple[Path, Path]:
    DEFAULT_JSON.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    decision = result.get("decision") or {}
    host = result.get("host") or {}
    lines = [
        "# Morpheus online entry points (Modal Linux CPU)",
        "",
        f"Part 09a Phase 5 seat: `{result.get('seat')}`",
        "",
        f"**Verdict:** `{decision.get('verdict')}`",
        "",
        f"- seed: `{result.get('seed')}`",
        f"- n_blocks: `{result.get('n_blocks')}`",
        f"- platform: `{host.get('platform')}`",
        f"- torch: `{host.get('torch')}`",
        f"- supported_qengines: `{host.get('supported_qengines')}`",
        f"- passed: `{decision.get('passed_engines')}`",
        f"- failed: `{decision.get('failed_engines')}`",
        "",
        "## Engines",
        "",
        "| Engine | Supported | Exported | Reloaded | Parity OK | Error |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in result.get("engines") or []:
        lines.append(
            "| `{engine}` | {supported} | {exported} | {reloaded} | "
            "{parity} | {error} |".format(
                engine=row.get("engine"),
                supported="yes" if row.get("supported") else "no",
                exported="yes" if row.get("exported") else "no",
                reloaded="yes" if row.get("reloaded") else "no",
                parity="yes" if row.get("parity_ok") else "no",
                error=row.get("error") or "",
            )
        )
    lines.append("")
    for row in result.get("engines") or []:
        lines.append(f"### `{row.get('engine')}`")
        lines.append("")
        lines.append(
            f"- online_float_to_export_mae: `{row.get('online_float_to_export_mae')}`"
        )
        lines.append("")
    DEFAULT_MD.write_text("\n".join(lines) + "\n")
    return DEFAULT_JSON, DEFAULT_MD


@app.local_entrypoint()
def main(seed: int = 0, n_blocks: int = 12) -> None:
    t0 = time.perf_counter()
    result = run_online_entry_points_cpu.remote(seed=seed, n_blocks=n_blocks)
    wall_s = time.perf_counter() - t0
    result.setdefault("host", {})
    result["host"]["modal_wall_s"] = wall_s
    json_path, md_path = _write_report(result)
    decision = result.get("decision") or {}
    print(f"verdict={decision.get('verdict')}")
    print(f"passed={decision.get('passed_engines')}")
    print(f"failed={decision.get('failed_engines')}")
    print(f"supported_qengines={(result.get('host') or {}).get('supported_qengines')}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"wall_s={wall_s:.1f}")
    if decision.get("verdict") != "yes":
        raise SystemExit(1)
