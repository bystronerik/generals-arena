"""Part 00c measurement corpus — panel schema, coverage, exit gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arena.records.trajectories import TrajectoryRecorder, read_trajectory
from training.morpheus.corpus.coverage import (
    CoverageReport,
    classify_trajectory,
    turn_band_for,
)
from training.morpheus.corpus.panel import (
    PanelError,
    load_panel,
    panel_run_scripts,
    validate_panel_against_checkout,
)
from training.morpheus.corpus.report import decide_pass

REPO = Path(__file__).resolve().parents[3]
PANEL_PATH = REPO / "scripts" / "configs" / "morpheus" / "bootstrap-panel.json"


def _write_panel(tmp_path: Path, **overrides) -> Path:
    base = {
        "name": "test-panel",
        "selection_date": "2026-08-04",
        "rating_era": "era",
        "round_seed": 7,
        "seat_policy": "alternate",
        "games_per_pair": 2,
        "source_label": "fixed_panel",
        "members": [
            {
                "bot_id": "cm_expander",
                "content_hash": "aaaaaaaaaaaa",
                "role": "anchor",
                "rating": 1500.0,
                "decisive_games": 100,
            },
            {
                "bot_id": "fog_scout",
                "content_hash": "bbbbbbbbbbbb",
                "role": "heuristic",
                "rating": 1800.0,
                "decisive_games": 50,
            },
            {
                "bot_id": "aegis",
                "content_hash": "cccccccccccc",
                "role": "heuristic",
                "rating": 2000.0,
                "decisive_games": 50,
            },
            {
                "bot_id": "boom",
                "content_hash": "dddddddddddd",
                "role": "heuristic",
                "rating": 2100.0,
                "decisive_games": 50,
            },
            {
                "bot_id": "macaria",
                "content_hash": "eeeeeeeeeeee",
                "role": "research",
                "rating": 2200.0,
                "decisive_games": 50,
            },
        ],
    }
    base.update(overrides)
    path = tmp_path / "panel.json"
    path.write_text(json.dumps(base), encoding="utf-8")
    return path


def test_load_panel_requires_roles_and_five_members(tmp_path):
    path = _write_panel(tmp_path)
    panel = load_panel(path)
    assert panel["name"] == "test-panel"
    assert len(panel["members"]) == 5


def test_panel_rejects_missing_research_role(tmp_path):
    path = _write_panel(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["members"] = [m for m in data["members"] if m["role"] != "research"]
    data["members"].append(
        {
            "bot_id": "metro",
            "content_hash": "ffffffffffff",
            "role": "heuristic",
            "rating": 1900.0,
            "decisive_games": 10,
        }
    )
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(PanelError, match="research"):
        load_panel(path)


def test_panel_rejects_classic_duel(tmp_path):
    path = _write_panel(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["members"][1]["bot_id"] = "classic_duel"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(PanelError, match="classic_duel"):
        load_panel(path)


def test_panel_rejects_random_seat_policy(tmp_path):
    path = _write_panel(tmp_path, seat_policy="random")
    with pytest.raises(PanelError, match="alternate"):
        load_panel(path)


def test_turn_band_for_boundaries():
    assert turn_band_for(1) == "1-200"
    assert turn_band_for(200) == "1-200"
    assert turn_band_for(201) == "201-400"
    assert turn_band_for(800) == "601-800"
    assert turn_band_for(801) == "801-1000"
    assert turn_band_for(1200) == "1001-1200"
    assert turn_band_for(0) is None


def test_ownership_contact_accepts_engine_bool_stack():
    import numpy as np

    from training.morpheus.corpus.coverage import _ownership_contact

    own = np.zeros((2, 3, 3), dtype=bool)
    own[0, 0, 0] = True
    own[1, 0, 1] = True
    assert _ownership_contact(own) is True
    far = np.zeros((2, 3, 3), dtype=bool)
    far[0, 0, 0] = True
    far[1, 2, 2] = True
    assert _ownership_contact(far) is False


def test_missing_classes_names_absent_events_not_as_zero():
    report = CoverageReport()
    missing = report.missing_classes()
    assert "board_size" in missing
    assert "18x18" in missing["board_size"]
    assert "events" in missing
    assert "contact" in missing["events"]
    assert "decisive" in missing["events"]
    assert report.contact == 0
    assert report.decisive == 0


def test_classify_without_events_uses_header_only(tmp_path):
    rec = TrajectoryRecorder(
        game_id="g1",
        seed=0,
        mode="competition",
        round_name="r",
        engine_version="era",
        bot_a="a",
        bot_b="b",
        directory=tmp_path,
    )
    rec.set_dims(19, 21)
    rec.record_turn(1, (0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (2, 2), (10, 10))
    rec.finish(winner="a", turns=450, terminated=True, truncated=False)
    traj = classify_trajectory(
        read_trajectory(rec.write()),
        source_label="fixed_panel",
        scan_events=False,
    )
    assert traj.board_size == "19x21"
    assert traj.turn_band == "401-600"
    assert traj.outcome == "a"
    assert traj.decisive is True
    assert traj.deathtouch is False
    assert traj.contact is False
    assert traj.source_label == "fixed_panel"


def test_decide_pass_fails_when_critical_events_absent():
    result = {
        "panel": {
            "members": [
                {"bot_id": "cm_expander", "content_hash": "h", "role": "anchor"},
                {"bot_id": "fog_scout", "content_hash": "h", "role": "heuristic"},
                {"bot_id": "macaria", "content_hash": "h", "role": "research"},
            ]
        },
        "verify": {"ok_count": 30, "fail_count": 0},
        "coverage": {
            "trajectory_count": 30,
            "modes": {"competition": 30},
            "engine_versions": {"era": 30},
            "source_labels": {"fixed_panel": 30},
            "events": {
                "contact": 0,
                "sight": 0,
                "castle": 0,
                "deathtouch": 0,
                "decisive": 20,
                "forced_mismatch_eligible": 0,
            },
            "missing_classes": {"events": ["contact", "sight"]},
        },
    }
    decision = decide_pass(result)
    assert decision["verdict"] == "no"
    assert decision["checks"]["critical_events_present"] is False
    assert "contact" in decision["missing_classes"]["events"]


def test_decide_pass_yes_on_complete_corpus():
    result = {
        "panel": {
            "members": [
                {"bot_id": "cm_expander", "content_hash": "h1", "role": "anchor"},
                {"bot_id": "fog_scout", "content_hash": "h2", "role": "heuristic"},
                {"bot_id": "macaria", "content_hash": "h3", "role": "research"},
            ]
        },
        "verify": {"ok_count": 30, "fail_count": 0},
        "coverage": {
            "trajectory_count": 30,
            "modes": {"competition": 30},
            "engine_versions": {"era": 30},
            "source_labels": {"fixed_panel": 30},
            "events": {
                "contact": 12,
                "sight": 8,
                "castle": 3,
                "deathtouch": 2,
                "decisive": 18,
                "forced_mismatch_eligible": 12,
            },
            "missing_classes": {"board_size": ["18x20"]},
        },
    }
    decision = decide_pass(result)
    assert decision["verdict"] == "yes"
    assert decision["missing_classes"]["board_size"] == ["18x20"]


@pytest.mark.morpheus
def test_committed_bootstrap_panel_matches_checkout():
    panel = load_panel(PANEL_PATH)
    assert panel["name"] == "morpheus-bootstrap"
    scripts = panel_run_scripts(panel)
    assert len(scripts) >= 5
    problems = validate_panel_against_checkout(panel)
    assert problems == [], problems
