"""Write Morpheus pilot learning measurement reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_pilot_report(
    report: dict[str, Any],
    *,
    json_path: Path,
    md_path: Path,
) -> None:
    json_path = Path(json_path)
    md_path = Path(md_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_to_markdown(report), encoding="utf-8")


def _to_markdown(report: dict[str, Any]) -> str:
    ok = "yes" if report.get("ok") else "no"
    lines = [
        "# Morpheus pilot class-1 learn",
        "",
        f"> Smoke ok: **{ok}** — thin Modal/local learning path on scraped class-1 items.",
        "",
        f"Generated: `{report.get('generated_at')}`",
        "",
        "## Result",
        "",
        f"- loss step 0: `{report.get('loss_step0')}`",
        f"- loss final: `{report.get('loss_final')}`",
        f"- loss decreased: `{report.get('loss_decreased')}`",
        f"- checkpoint reloadable: `{report.get('checkpoint_reloadable')}`",
        f"- sample count: `{report.get('sample_count')}`",
        f"- source labels: `{report.get('source_labels')}`",
        f"- wall_s: `{report.get('wall_s')}`",
        f"- a100_hours (wall): `{report.get('a100_hours')}`",
        f"- accounting line: `{report.get('accounting_line')}`",
        "",
        "## Config",
        "",
    ]
    cfg = report.get("config") or {}
    for key in sorted(cfg):
        lines.append(f"- `{key}`: `{cfg[key]}`")
    lines.extend(
        [
            "",
            "## Notes",
            "",
        ]
    )
    for note in report.get("notes") or []:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)
