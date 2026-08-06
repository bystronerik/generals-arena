"""Build and write the Morpheus JAX preflight measurement report."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.morpheus.jax_preflight.parity import compare_snapshots

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JSON = REPO_ROOT / "docs/research/measurements/morpheus-jax-preflight.json"
DEFAULT_MD = REPO_ROOT / "docs/research/measurements/morpheus-jax-preflight.md"


def _looks_like_a100_40gb(gpu_result: dict[str, Any]) -> bool:
    """Accept A100 with >=40 GB. Modal may upgrade A100→80 GB at no cost."""
    device = gpu_result.get("device") or {}
    nvidia = gpu_result.get("nvidia") or {}
    haystack = " ".join(
        str(x)
        for x in (
            device.get("device_str"),
            device.get("device_kind"),
            nvidia.get("gpu_name"),
            nvidia.get("memory_total"),
        )
        if x
    ).upper()
    has_a100 = "A100" in haystack
    compact = haystack.replace(" ", "")
    has_40_or_more = (
        "40GB" in compact
        or "80GB" in compact
        or "40960" in compact
        or "81920" in compact
        or "40GI" in compact
        or "80GI" in compact
    )
    # memory_total from nvidia-smi is often like "40960 MiB" or "81920 MiB"
    mem = str(nvidia.get("memory_total") or "").replace(",", "").upper()
    if "40960" in mem or "81920" in mem or "40" in mem or "80" in mem or "81" in mem:
        has_40_or_more = True
    return has_a100 and has_40_or_more and device.get("platform") == "gpu"


def _parity_table(
    cpu_result: dict[str, Any],
    gpu_result: dict[str, Any],
) -> list[dict[str, Any]]:
    cpu_by_name = {
        f["name"]: f["snapshot"] for f in cpu_result.get("parity_fixtures", [])
    }
    rows = []
    for gf in gpu_result.get("parity_fixtures", []):
        name = gf["name"]
        cpu_snap = cpu_by_name.get(name)
        if cpu_snap is None:
            rows.append(
                {
                    "name": name,
                    "match": False,
                    "fields": {},
                    "error": "missing on CPU",
                }
            )
            continue
        # Rehydrate lists to arrays inside compare via np.asarray
        cmp = compare_snapshots(cpu_snap, gf["snapshot"])
        rows.append({"name": name, **cmp})
    return rows


def decide_pass(
    cpu_result: dict[str, Any],
    gpu_result: dict[str, Any],
    parity_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Apply Part 00 exit criterion."""
    if parity_rows is None:
        parity_rows = _parity_table(cpu_result, gpu_result)
    a100 = _looks_like_a100_40gb(gpu_result)
    compiled = bool(gpu_result.get("compiled_ok")) and bool(
        cpu_result.get("compiled_ok")
    )
    modifiers_ok = bool(
        gpu_result.get("modifiers", {}).get("build_castles")
        and gpu_result.get("modifiers", {}).get("deathtouch_turn") == 800
        and gpu_result.get("modifiers", {}).get("mode") == "competition"
    )
    parity_ok = bool(parity_rows) and all(r.get("match") for r in parity_rows)
    passed = a100 and compiled and modifiers_ok and parity_ok
    return {
        "pass": passed,
        "verdict": "yes" if passed else "no",
        "checks": {
            "a100_40gb": a100,
            "compiled": compiled,
            "competition_modifiers": modifiers_ok,
            "cpu_gpu_parity": parity_ok,
        },
        "self_play_backend_if_no": "cpu",
        "note": (
            "GPU self-play transitions are permitted."
            if passed
            else "Select CPU self-play; reserve the A100 for learning."
        ),
    }


def build_report(
    cpu_result: dict[str, Any],
    gpu_result: dict[str, Any],
    *,
    a100_hours: float | None = None,
    wall_s: float | None = None,
) -> dict[str, Any]:
    parity_rows = _parity_table(cpu_result, gpu_result)
    decision = decide_pass(cpu_result, gpu_result, parity_rows)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "part": "00-modal-jax-preflight",
        "decision": decision,
        "parity": parity_rows,
        "cpu": _strip_for_report(cpu_result),
        "gpu": _strip_for_report(gpu_result),
        "accounting": {
            "a100_hours": a100_hours,
            "wall_s": wall_s,
            "budget_hours": 48,
        },
    }


def _strip_for_report(result: dict[str, Any]) -> dict[str, Any]:
    out = dict(result)
    # Keep fixture names only in the top-level parity table; drop bulky grids.
    fixtures = out.get("parity_fixtures") or []
    out["parity_fixtures"] = [
        {
            "name": f["name"],
            "winner": f["snapshot"].get("winner"),
            "time": f["snapshot"].get("time"),
            "army_totals": f["snapshot"].get("army_totals"),
            "land_totals": f["snapshot"].get("land_totals"),
            "castles_true": int(sum(sum(row) for row in f["snapshot"].get("castles", []))),
        }
        for f in fixtures
    ]
    return out


def render_markdown(report: dict[str, Any]) -> str:
    d = report["decision"]
    gpu = report["gpu"]
    cpu = report["cpu"]
    lines = [
        "# Morpheus JAX preflight",
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
        "## GPU device",
        "",
        f"- platform: `{gpu.get('device', {}).get('platform')}`",
        f"- device: `{gpu.get('device', {}).get('device_str')}`",
        f"- nvidia: `{gpu.get('nvidia')}`",
        f"- versions: `{gpu.get('versions')}`",
        "",
        "## Throughput",
        "",
        "| Seat | Cold compile (s) | Warm steps/s | Peak device bytes | H2D (s) | D2H (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, result in (("cpu", cpu), ("gpu", gpu)):
        transfer = result.get("transfer") or {}
        lines.append(
            "| `{label}` | {cold:.3f} | {warm:,.0f} | {peak} | {h2d:.4f} | {d2h:.4f} |".format(
                label=label,
                cold=float(result.get("cold_compile_s") or 0),
                warm=float(result.get("warm_steps_per_s") or 0),
                peak=result.get("device_memory_peak_bytes"),
                h2d=float(transfer.get("host_to_device_s") or 0),
                d2h=float(transfer.get("device_to_host_s") or 0),
            )
        )
    lines += [
        "",
        f"Config: `{gpu.get('config')}`",
        "",
        "## CPU/GPU parity",
        "",
        "| Fixture | Match |",
        "| --- | --- |",
    ]
    for row in report["parity"]:
        lines.append(f"| `{row['name']}` | {'yes' if row.get('match') else 'no'} |")
    acct = report.get("accounting") or {}
    lines += [
        "",
        "## Accounting",
        "",
        f"- A100 hours (this run): `{acct.get('a100_hours')}`",
        f"- Wall seconds: `{acct.get('wall_s')}`",
        f"- Budget hours: `{acct.get('budget_hours')}`",
        "",
        "## Competition modifiers (GPU seat)",
        "",
        f"```json\n{json.dumps(gpu.get('modifiers'), indent=2)}\n```",
        "",
    ]
    return "\n".join(lines)


def write_report(
    report: dict[str, Any],
    *,
    json_path: Path = DEFAULT_JSON,
    md_path: Path = DEFAULT_MD,
) -> tuple[Path, Path]:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    md_path.write_text(render_markdown(report))
    return json_path, md_path
