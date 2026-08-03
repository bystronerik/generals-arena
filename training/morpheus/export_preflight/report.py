"""Build and write the Morpheus sandbox export preflight report."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JSON = (
    REPO_ROOT / "docs/research/measurements/morpheus-export-preflight.json"
)
DEFAULT_MD = REPO_ROOT / "docs/research/measurements/morpheus-export-preflight.md"


def decide_pass(result: dict[str, Any]) -> dict[str, Any]:
    """Apply Part 00b exit criterion.

    Answer yes if at least one pinned runtime exports, reloads, and executes
    the required random-weight graph with static 8-bit ops, bounded memory,
    and a recorded parity error. Fallback operators must be listed, not hidden.
    """
    candidates = result.get("candidates") or []
    viable = [c for c in candidates if c.get("viable")]
    pins = result.get("pins") or {}
    torch_pin = (pins.get("installed_vs_sandbox") or {}).get("torch") or {}
    torch_ok = bool(torch_pin.get("match"))
    checks = {
        "sandbox_torch_pin_match": torch_ok,
        "any_static_8bit_viable": bool(viable),
        "batch_shapes_exercised": bool(result.get("batch_shapes") == [1, 4, 64]),
        "fallbacks_reported": all("fallback_operators" in c for c in candidates),
    }
    passed = all(
        (
            checks["sandbox_torch_pin_match"],
            checks["any_static_8bit_viable"],
            checks["batch_shapes_exercised"],
        )
    )
    note = (
        "At least one sandbox runtime exports, reloads, and executes a static "
        "8-bit graph for the probe path."
        if passed
        else (
            "No sandbox-only static 8-bit path succeeded. Part 04 is blocked "
            "until the network or deployment design changes."
        )
    )
    return {
        "pass": passed,
        "verdict": "yes" if passed else "no",
        "checks": checks,
        "viable_candidates": [c["name"] for c in viable],
        "note": note,
    }


def build_report(result: dict[str, Any]) -> dict[str, Any]:
    decision = decide_pass(result)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "part": "00b-sandbox-export-preflight",
        "decision": decision,
        "host": result.get("host"),
        "pins": result.get("pins"),
        "architecture": result.get("architecture"),
        "batch_shapes": result.get("batch_shapes"),
        "batch_roles": result.get("batch_roles"),
        "candidates": result.get("candidates"),
    }


def render_markdown(report: dict[str, Any]) -> str:
    d = report["decision"]
    lines = [
        "# Morpheus sandbox export preflight",
        "",
        f"> Verdict: **{d['verdict']}** — {d['note']}",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Checks",
        "",
        "| Check | Result |",
        "| --- | --- |",
    ]
    for name, ok in d["checks"].items():
        lines.append(f"| `{name}` | {'yes' if ok else 'no'} |")
    lines += [
        "",
        f"Viable candidates: `{d.get('viable_candidates')}`",
        "",
        "## Host",
        "",
        f"```json\n{json.dumps(report.get('host'), indent=2)}\n```",
        "",
        "## Architecture probe",
        "",
        f"```json\n{json.dumps(report.get('architecture'), indent=2)}\n```",
        "",
        "## Batch shapes",
        "",
        f"- shapes: `{report.get('batch_shapes')}`",
        f"- roles: `{report.get('batch_roles')}`",
        "",
        "## Candidates",
        "",
        "| Name | Viable | Static 8-bit | Exported | Reloaded | Executed | "
        "Serialized B | Peak RSS | Error |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for c in report.get("candidates") or []:
        lines.append(
            "| `{name}` | {viable} | {static} | {exported} | {reloaded} | "
            "{executed} | {size} | {rss} | {err} |".format(
                name=c.get("name"),
                viable="yes" if c.get("viable") else "no",
                static="yes" if c.get("has_static_8bit_ops") else "no",
                exported="yes" if c.get("exported") else "no",
                reloaded="yes" if c.get("reloaded") else "no",
                executed="yes" if c.get("executed") else "no",
                size=c.get("serialized_bytes"),
                rss=c.get("peak_rss_bytes"),
                err=(c.get("error") or "").replace("|", "/")[:80] or "—",
            )
        )
    lines += ["", "## Candidate detail", ""]
    for c in report.get("candidates") or []:
        lines += [
            f"### `{c.get('name')}`",
            "",
            f"- runtime: `{c.get('runtime')}`",
            f"- quantization_format: `{c.get('quantization_format')}`",
            f"- fallback_operators: `{c.get('fallback_operators')}`",
            f"- unsupported_operators: `{c.get('unsupported_operators')}`",
            f"- float_to_export_error: `{c.get('float_to_export_error')}`",
            f"- batch_results: `{json.dumps(c.get('batch_results'), sort_keys=True)}`",
            f"- notes: `{c.get('notes')}`",
            "",
        ]
    pins = report.get("pins") or {}
    lines += [
        "## Pin audit",
        "",
        f"- judge matches sandbox: "
        f"`{pins.get('judge_matches_sandbox')}`",
        f"- installed vs sandbox: "
        f"`{json.dumps(pins.get('installed_vs_sandbox'), sort_keys=True)}`",
        f"- unavailable packages (not probed): "
        f"`{pins.get('unavailable_in_sandbox')}`",
        "",
    ]
    return "\n".join(lines)


def write_report(
    report: dict[str, Any],
    *,
    json_path: Path = DEFAULT_JSON,
    md_path: Path | None = None,
) -> tuple[Path, Path]:
    if md_path is None:
        md_path = json_path.with_suffix(".md") if json_path != DEFAULT_JSON else DEFAULT_MD
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    md_path.write_text(render_markdown(report))
    return json_path, md_path
