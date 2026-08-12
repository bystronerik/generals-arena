"""State-file round trip for resumable training (cheap default suite).

No JAX imports: ``training.joe.state`` is stdlib-only, so these add no
measurable time to the 12 s budget (AGENTS.md tester).
"""

from __future__ import annotations

import json
import os

import pytest

from training.joe.state import (
    STATE_FILENAME,
    STATE_SCHEMA,
    check_curriculum_stage,
    read_state,
    write_state,
)

FILES = {"full": "joe-x_1234.eqx", "ema": "joe-x_ema_1234.eqx"}


def test_round_trip(tmp_path):
    written = write_state(str(tmp_path), "joe-x", 1234, 2, 0.61, "sha",
                          files=FILES)
    state = read_state(str(tmp_path))
    assert state == written
    assert state["schema"] == STATE_SCHEMA
    assert state["global_step"] == 1234
    assert state["curriculum_stage"] == 2
    assert state["last_eval_wr"] == pytest.approx(0.61)
    assert state["files"] == FILES
    # Atomic write leaves no tmp file behind
    assert os.listdir(tmp_path) == [STATE_FILENAME]


def test_missing_returns_none(tmp_path):
    assert read_state(str(tmp_path)) is None


def test_stale_tmp_file_is_ignored(tmp_path):
    # A kill mid-write leaves state.json.tmp; the reader never sees it and
    # the next write still lands atomically.
    (tmp_path / (STATE_FILENAME + ".tmp")).write_text("{ garbage")
    assert read_state(str(tmp_path)) is None
    write_state(str(tmp_path), "joe-x", 2, 0, 0.0, "sha", files=FILES)
    assert read_state(str(tmp_path))["global_step"] == 2


def test_legacy_v1_returns_none(tmp_path):
    (tmp_path / STATE_FILENAME).write_text(json.dumps(
        {"run_name": "joe-x", "iteration": 500, "engine_sha": "sha",
         "time": 0.0}))
    assert read_state(str(tmp_path)) is None


def test_future_schema_raises(tmp_path):
    (tmp_path / STATE_FILENAME).write_text(json.dumps({"schema": 3}))
    with pytest.raises(ValueError, match="schema 3"):
        read_state(str(tmp_path))


def test_overwrite_replaces_previous(tmp_path):
    write_state(str(tmp_path), "joe-x", 2, 0, 0.0, "sha", files=FILES)
    write_state(str(tmp_path), "joe-x", 4, 1, 0.5, "sha",
                files={"full": "joe-x_4.eqx", "ema": "joe-x_ema_4.eqx"})
    state = read_state(str(tmp_path))
    assert state["global_step"] == 4
    assert state["curriculum_stage"] == 1


def test_stage_guard():
    check_curriculum_stage(0, 1)
    check_curriculum_stage(2, 3)
    with pytest.raises(ValueError, match="out of range"):
        check_curriculum_stage(3, 3)
    with pytest.raises(ValueError, match="out of range"):
        check_curriculum_stage(-1, 3)
