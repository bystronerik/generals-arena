"""Read and compare competition-sandbox dependency pins."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SANDBOX_REQUIREMENTS = REPO_ROOT / "requirements-sandbox.txt"
JUDGE_REQUIREMENTS = (
    REPO_ROOT / "competition-module" / "competition" / "requirements.txt"
)

_PIN_RE = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s#]+)")


def parse_requirements(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _PIN_RE.match(stripped)
        if match:
            pins[match.group(1).lower()] = match.group(2)
    return pins


def sandbox_pins(
    requirements_path: Path | None = None,
) -> dict[str, str]:
    return parse_requirements(requirements_path or SANDBOX_REQUIREMENTS)


def judge_pins() -> dict[str, str]:
    return parse_requirements(JUDGE_REQUIREMENTS)


def installed_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name, module_name in (
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("jax", "jax"),
        ("safetensors", "safetensors"),
        ("numba", "numba"),
    ):
        try:
            mod = __import__(module_name)
            versions[name] = getattr(mod, "__version__", None)
        except ImportError:
            versions[name] = None
    return versions


def versions_match(installed: str | None, pin: str | None) -> bool:
    """True when installed equals the pin, or pin with a local tag (e.g. +cpu)."""
    if installed is None or pin is None:
        return False
    return installed == pin or installed.startswith(f"{pin}+")


def pin_audit(requirements_path: Path | None = None) -> dict[str, Any]:
    sandbox = sandbox_pins(requirements_path)
    judge = judge_pins()
    installed = installed_versions()
    mismatches = {
        pkg: {"sandbox": sandbox[pkg], "judge": judge[pkg]}
        for pkg in sandbox
        if pkg in judge and sandbox[pkg] != judge[pkg]
    }
    installed_vs_sandbox = {
        pkg: {
            "sandbox": sandbox.get(pkg),
            "installed": installed.get(pkg),
            "match": versions_match(installed.get(pkg), sandbox.get(pkg)),
        }
        for pkg in ("torch", "safetensors", "numpy", "jax")
        if pkg in sandbox
    }
    return {
        "sandbox_path": str(requirements_path or SANDBOX_REQUIREMENTS),
        "judge_path": str(JUDGE_REQUIREMENTS),
        "sandbox_pins": sandbox,
        "judge_matches_sandbox": not mismatches,
        "sandbox_vs_judge_mismatches": mismatches,
        "installed_vs_sandbox": installed_vs_sandbox,
        "unavailable_in_sandbox": [
            "onnxruntime",
            "onnx",
            "tensorrt",
            "torchao",
            "openvino",
        ],
    }
