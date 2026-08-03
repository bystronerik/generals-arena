"""Verify Morpheus corpus trajectories and build the coverage report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from arena.records.store import engine_version
from arena.records.trajectories import EraMismatch, read_trajectory, verify_trajectory
from training.morpheus.corpus.coverage import (
    build_coverage,
    coverage_to_dict,
    iter_trajectory_paths,
)
from training.morpheus.corpus.panel import load_panel
from training.morpheus.corpus.report import decide_pass, write_report


def verify_corpus(
    *,
    trajectories_dir: Path,
    report_path: Path,
    panel_path: Path | None = None,
    scan_events: bool = True,
) -> dict[str, Any]:
    """
    Replay-verify every trajectory, classify coverage, write the measurement report.
    """
    directory = trajectories_dir.resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"trajectory directory missing: {directory}")

    panel = load_panel(panel_path) if panel_path is not None else None
    default_label = str(panel["source_label"]) if panel else "fixed_panel"
    current_engine = engine_version()

    verify_ok: list[str] = []
    verify_fail: list[dict[str, Any]] = []
    era_mismatch: list[str] = []

    paths = list(iter_trajectory_paths(directory))
    for path in paths:
        traj = read_trajectory(path)
        try:
            report = verify_trajectory(traj, engine=current_engine)
        except EraMismatch as exc:
            era_mismatch.append(str(exc))
            verify_fail.append(
                {
                    "game_id": traj.game_id,
                    "ok": False,
                    "mismatches": [str(exc)],
                }
            )
            continue
        if report.ok:
            verify_ok.append(traj.game_id)
        else:
            verify_fail.append(
                {
                    "game_id": report.game_id,
                    "ok": False,
                    "mismatches": list(report.mismatches),
                }
            )

    coverage = build_coverage(
        directory,
        default_source_label=default_label,
        scan_events=scan_events,
    )
    coverage.verify_failures = [f["game_id"] for f in verify_fail]

    result = {
        "trajectories_dir": str(directory),
        "trajectory_files": len(paths),
        "engine_checkout": current_engine,
        "verify": {
            "ok_count": len(verify_ok),
            "fail_count": len(verify_fail),
            "era_mismatch_count": len(era_mismatch),
            "failures": verify_fail,
            "era_mismatches": era_mismatch,
        },
        "coverage": coverage_to_dict(coverage),
        "panel": (
            {
                "name": panel["name"],
                "selection_date": panel["selection_date"],
                "rating_era": panel["rating_era"],
                "source_label": panel["source_label"],
                "members": [
                    {
                        "bot_id": m["bot_id"],
                        "content_hash": m["content_hash"],
                        "role": m["role"],
                        "rating": m["rating"],
                        "decisive_games": m["decisive_games"],
                        "evidence_entity": m.get("evidence_entity"),
                    }
                    for m in panel["members"]
                ],
            }
            if panel
            else None
        ),
    }
    result["decision"] = decide_pass(result)
    write_report(result, json_path=report_path)
    return result
