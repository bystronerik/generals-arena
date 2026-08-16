"""State-file round trip for resumable training (cheap default suite).

No JAX imports: ``training.joe.state`` is stdlib-only, so these add no
measurable time to the 15 s budget (AGENTS.md, "Test suite budget").
"""

from __future__ import annotations

import json
import os

import pytest

from training.joe.state import (
    STATE_FILENAME,
    STATE_SCHEMA,
    check_curriculum_stage,
    prune_checkpoints,
    read_state,
    write_state,
)

FILES = {"full": "joe-x_1234.eqx", "ema": "joe-x_ema_1234.eqx"}
RUN = "joe-x"


def _touch(dirpath, *names):
    for name in names:
        (dirpath / name).write_bytes(b"weights")


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


# ---- prune_checkpoints ----


def test_prune_removes_only_steps_below_the_confirmed_one(tmp_path):
    _touch(tmp_path,
           f"{RUN}_1000.eqx", f"{RUN}_ema_1000.eqx",
           f"{RUN}_1500.eqx", f"{RUN}_ema_1500.eqx",
           f"{RUN}_2000.eqx", f"{RUN}_ema_2000.eqx")
    write_state(str(tmp_path), RUN, 2000, 0, 0.0, "sha",
                files={"full": f"{RUN}_2000.eqx",
                       "ema": f"{RUN}_ema_2000.eqx"})
    removed = prune_checkpoints(str(tmp_path), RUN, 1500)
    assert sorted(removed) == [f"{RUN}_1000.eqx", f"{RUN}_ema_1000.eqx"]
    left = set(os.listdir(tmp_path))
    # The confirmed step itself stays, and so does everything after it: R2
    # holds 1500, but 2000 is still the only copy of that set.
    assert {f"{RUN}_1500.eqx", f"{RUN}_ema_1500.eqx",
            f"{RUN}_2000.eqx", f"{RUN}_ema_2000.eqx"} <= left


@pytest.mark.parametrize("confirmed", [None, 0, -1])
def test_prune_does_nothing_until_an_upload_confirms(tmp_path, confirmed):
    # A failed upload, or a hook with no bucket behind it, must never cost a
    # local file — that is the whole safety property.
    names = {f"{RUN}_1000.eqx", f"{RUN}_ema_1000.eqx"}
    _touch(tmp_path, *names)
    assert prune_checkpoints(str(tmp_path), RUN, confirmed) == []
    assert set(os.listdir(tmp_path)) == names


def test_prune_keeps_unnumbered_artifacts_and_other_runs(tmp_path):
    keep = {f"{RUN}_ema.eqx", f"{RUN}_final.eqx", f"{RUN}_ema_final.eqx",
            "metrics.jsonl", "manifest.json", "joe-other_1000.eqx"}
    _touch(tmp_path, f"{RUN}_1000.eqx", *keep)
    assert prune_checkpoints(str(tmp_path), RUN, 9999) == [f"{RUN}_1000.eqx"]
    assert set(os.listdir(tmp_path)) == keep


def test_prune_never_removes_the_files_state_json_names(tmp_path):
    # Defensive: even if the confirmed step somehow runs ahead of state.json,
    # the resume set survives, so state.json can never point at nothing.
    _touch(tmp_path, f"{RUN}_1000.eqx", f"{RUN}_ema_1000.eqx")
    write_state(str(tmp_path), RUN, 1000, 0, 0.0, "sha",
                files={"full": f"{RUN}_1000.eqx",
                       "ema": f"{RUN}_ema_1000.eqx"})
    assert prune_checkpoints(str(tmp_path), RUN, 5000) == []
    assert read_state(str(tmp_path))["global_step"] == 1000
