"""Build and write the Part 13 Modal compute-gate report."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
DEFAULT_JSON = REPO / "docs/research/measurements/morpheus-modal-qualification.json"
DEFAULT_MD = REPO / "docs/research/measurements/morpheus-modal-qualification.md"


def render_markdown(report: dict[str, Any]) -> str:
    d = report["decision"]
    layout = report.get("selected_layout") or {}
    rates = report.get("rates") or {}
    acct = report.get("a100_accounting") or {}
    cadence = report.get("cadence") or {}
    lines = [
        "# Morpheus Modal compute qualification",
        "",
        f"> Verdict: **{d['verdict']}** — {d.get('note', '')}",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Selected layout",
        "",
        f"- self-play backend: `{layout.get('backend')}`",
        f"- physical CPU cores per game: `{layout.get('physical_cores_per_game')}`",
        f"- seat searches: `{layout.get('seat_search')}`",
        f"- games per worker (measured batch): `{layout.get('completed_games')}`",
        f"- concurrent games per container: `{layout.get('concurrent_games')}`",
        f"- worker containers paired with one A100: `{report.get('workers_per_a100')}`",
        "",
        "## Throughput",
        "",
        f"- games/hour (aggregate): `{rates.get('aggregate_games_per_hour')}`",
        f"- positions/hour: `{rates.get('positions_supply_per_hour')}`",
        f"- games/worker-hour: `{rates.get('games_per_worker_hour')}`",
        f"- mean game latency (s): `{layout.get('mean_game_latency_s')}`",
        "",
        "## Checkpoint cadence",
        "",
        f"- games/checkpoint: `{rates.get('games_per_checkpoint')}`",
        f"- checkpoint count: `{rates.get('checkpoint_count')}`",
        f"- useful cadence found: `{cadence.get('useful_found')}`",
        f"- selected cadence: `{cadence.get('selected_name')}`",
        "",
        "## A100 accounting",
        "",
        "| Line | Hours |",
        "| --- | ---: |",
    ]
    for name, hours in (acct.get("lines") or {}).items():
        lines.append(f"| `{name}` | {hours} |")
    lines += [
        f"| **total** | **{acct.get('total_a100_hours')}** |",
        f"| budget | {acct.get('budget_hours')} |",
        "",
        f"- fits budget: `{acct.get('fits_budget')}`",
        f"- CPU hours (measured qualification): `{report.get('cpu_hours_total')}`",
        "",
        "## Checks",
        "",
        "| Check | Result |",
        "| --- | --- |",
    ]
    for name, ok in (d.get("checks") or {}).items():
        lines.append(f"| `{name}` | {'yes' if ok else 'no'} |")
    if d.get("fallback"):
        lines += [
            "",
            "## Fallback",
            "",
            f"```json\n{json.dumps(d['fallback'], indent=2)}\n```",
            "",
        ]
    lines += [
        "",
        "## Layout ranking",
        "",
        "```json",
        json.dumps(report.get("layout_ranking") or [], indent=2),
        "```",
        "",
        "## Cadence candidates",
        "",
        "```json",
        json.dumps(cadence.get("candidates") or [], indent=2),
        "```",
        "",
    ]
    return "\n".join(lines)


def write_report(
    report: dict[str, Any],
    *,
    json_path: Path = DEFAULT_JSON,
    md_path: Path = DEFAULT_MD,
) -> tuple[Path, Path]:
    json_path = Path(json_path)
    md_path = Path(md_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    md_path.write_text(render_markdown(report))
    return json_path, md_path


def stamp_generated(report: dict[str, Any]) -> dict[str, Any]:
    out = dict(report)
    out["generated_at"] = datetime.now(timezone.utc).isoformat()
    out["part"] = "13-modal-compute-gate"
    return out
