"""Build and write the Morpheus bootstrap corpus measurement report."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from arena.paths import REPO_ROOT

DEFAULT_JSON = (
    REPO_ROOT / "docs" / "research" / "measurements" / "morpheus-bootstrap-corpus.json"
)
DEFAULT_MD = (
    REPO_ROOT / "docs" / "research" / "measurements" / "morpheus-bootstrap-corpus.md"
)

# Soft floors for Parts 03 / 05 / 10 to start their measurements. Absent data
# stays named in missing_classes; these floors only gate the part exit.
MIN_DECISIVE = 10
MIN_FORCED_MISMATCH = 5
MIN_TRAJECTORIES = 20


def decide_pass(result: dict[str, Any]) -> dict[str, Any]:
    """Apply Part 00c exit criterion."""
    panel = result.get("panel") or {}
    members = panel.get("members") or []
    coverage = result.get("coverage") or {}
    verify = result.get("verify") or {}
    events = coverage.get("events") or {}
    missing = coverage.get("missing_classes") or {}
    modes = coverage.get("modes") or {}
    engines = coverage.get("engine_versions") or {}
    labels = coverage.get("source_labels") or {}

    hashes_named = bool(members) and all(
        m.get("bot_id") and m.get("content_hash") and m.get("role") for m in members
    )
    roles = {str(m.get("role")) for m in members}
    roles_ok = {"anchor", "heuristic", "research"}.issubset(roles)

    competition_only = set(modes.keys()) <= {"competition"} and bool(modes.get("competition"))
    one_era = len(engines) == 1 and bool(engines)
    all_verified = (
        int(verify.get("fail_count", 1)) == 0
        and int(verify.get("ok_count", 0)) == int(coverage.get("trajectory_count", -1))
        and int(verify.get("ok_count", 0)) > 0
    )
    source_labeled = bool(labels) and "unlabeled" not in labels
    decisive_ok = int(events.get("decisive", 0)) >= MIN_DECISIVE
    mismatch_ok = int(events.get("forced_mismatch_eligible", 0)) >= MIN_FORCED_MISMATCH
    count_ok = int(coverage.get("trajectory_count", 0)) >= MIN_TRAJECTORIES
    # Event classes that Part 05 / 10 need must not be absent.
    critical_events_present = all(
        int(events.get(name, 0)) > 0
        for name in ("contact", "sight", "decisive", "forced_mismatch_eligible")
    )

    checks = {
        "panel_hashes_named": hashes_named,
        "panel_roles_complete": roles_ok,
        "competition_mode_only": competition_only,
        "single_engine_era": one_era,
        "all_trajectories_verified": all_verified,
        "source_labels_present": source_labeled,
        "enough_trajectories": count_ok,
        "enough_decisive": decisive_ok,
        "enough_forced_mismatch": mismatch_ok,
        "critical_events_present": critical_events_present,
    }
    passed = all(checks.values())
    note = (
        "Panel names exact registered hashes; all games are competition mode; "
        "every trajectory verifies in one engine era; decisive and "
        "forced-mismatch cases are present for Parts 03, 05, and 10."
        if passed
        else (
            "Corpus gate failed. Missing classes stay named in the report; "
            "do not treat absent coverage as zero."
        )
    )
    return {
        "pass": passed,
        "verdict": "yes" if passed else "no",
        "checks": checks,
        "thresholds": {
            "min_trajectories": MIN_TRAJECTORIES,
            "min_decisive": MIN_DECISIVE,
            "min_forced_mismatch": MIN_FORCED_MISMATCH,
        },
        "missing_classes": missing,
        "note": note,
    }


def build_report(result: dict[str, Any]) -> dict[str, Any]:
    decision = result.get("decision") or decide_pass(result)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "part": "00c-measurement-corpus",
        "decision": decision,
        "panel": result.get("panel"),
        "trajectories_dir": result.get("trajectories_dir"),
        "trajectory_files": result.get("trajectory_files"),
        "engine_checkout": result.get("engine_checkout"),
        "verify": result.get("verify"),
        "coverage": result.get("coverage"),
    }


def render_markdown(report: dict[str, Any]) -> str:
    d = report["decision"]
    coverage = report.get("coverage") or {}
    events = coverage.get("events") or {}
    missing = d.get("missing_classes") or coverage.get("missing_classes") or {}
    lines = [
        "# Morpheus bootstrap measurement corpus",
        "",
        f"> Verdict: **{d['verdict']}** — {d['note']}",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Checks",
        "",
        "| Check | Result |",
        "| --- | --- |",
    ]
    for name, ok in (d.get("checks") or {}).items():
        lines.append(f"| `{name}` | {'yes' if ok else 'no'} |")

    lines.extend(
        [
            "",
            "## Panel",
            "",
        ]
    )
    panel = report.get("panel") or {}
    if panel:
        lines.append(f"Name: `{panel.get('name')}`")
        lines.append(f"Selection date: `{panel.get('selection_date')}`")
        lines.append(f"Rating era: `{panel.get('rating_era')}`")
        lines.append(f"Source label: `{panel.get('source_label')}`")
        lines.extend(
            [
                "",
                "| Bot | Hash | Role | Rating | Decisive |",
                "| --- | --- | --- | ---: | ---: |",
            ]
        )
        for m in panel.get("members") or []:
            lines.append(
                f"| `{m.get('bot_id')}` | `{m.get('content_hash')}` | "
                f"{m.get('role')} | {m.get('rating')} | {m.get('decisive_games')} |"
            )
    else:
        lines.append("(no panel attached)")

    lines.extend(
        [
            "",
            "## Coverage",
            "",
            f"Trajectories: {coverage.get('trajectory_count', 0)}",
            f"Modes: `{coverage.get('modes')}`",
            f"Engine versions: `{coverage.get('engine_versions')}`",
            f"Source labels: `{coverage.get('source_labels')}`",
            "",
            "### Events",
            "",
            "| Event | Count |",
            "| --- | ---: |",
        ]
    )
    for name, count in events.items():
        lines.append(f"| `{name}` | {count} |")

    lines.extend(
        [
            "",
            "### Board sizes",
            "",
            "```json",
            json.dumps(coverage.get("board_sizes") or {}, indent=2, sort_keys=True),
            "```",
            "",
            "### Turn bands",
            "",
            "```json",
            json.dumps(coverage.get("turn_bands") or {}, indent=2, sort_keys=True),
            "```",
            "",
            "### Outcomes",
            "",
            "```json",
            json.dumps(coverage.get("outcomes") or {}, indent=2, sort_keys=True),
            "```",
            "",
            "## Missing classes",
            "",
        ]
    )
    if missing:
        for key, values in missing.items():
            lines.append(f"- `{key}`: {values}")
    else:
        lines.append("None named.")

    verify = report.get("verify") or {}
    lines.extend(
        [
            "",
            "## Verify",
            "",
            f"ok={verify.get('ok_count')} fail={verify.get('fail_count')} "
            f"era_mismatch={verify.get('era_mismatch_count')}",
            "",
            f"Trajectories dir: `{report.get('trajectories_dir')}`",
            "",
            "Machine-readable: [`morpheus-bootstrap-corpus.json`](morpheus-bootstrap-corpus.json)",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(
    result: dict[str, Any],
    *,
    json_path: Path | None = None,
) -> tuple[Path, Path]:
    report = build_report(result)
    out_json = (json_path or DEFAULT_JSON).resolve()
    out_md = out_json.with_suffix(".md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")
    return out_json, out_md
