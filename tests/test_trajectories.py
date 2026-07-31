"""Trajectory files: round trip, atomic writes, and the series they expose."""
from __future__ import annotations

import gzip
import json

import pytest

from arena.records.trajectories import (
    FORMAT_VERSION,
    Trajectory,
    TrajectoryError,
    TrajectoryRecorder,
    gzip_into_place,
    read_trace,
    read_trajectory,
    round_trajectory_dir,
    trace_path,
    trajectory_path,
    write_jsonl_gz,
)


def _recorder(tmp_path, *, turns: int = 30, digest_every: int = 10) -> TrajectoryRecorder:
    rec = TrajectoryRecorder(
        game_id="20260101T000000Z_a_vs_b_s7_abcd1234",
        seed=7,
        mode="competition",
        round_name="roundT",
        engine_version="9e3b9d1",
        bot_a="metro",
        bot_b="smoke",
        H=19,
        W=21,
        directory=tmp_path,
        digest_every=digest_every,
    )
    for turn in range(1, turns + 1):
        rec.record_turn(
            turn,
            (0, 1, 2, turn % 4, 0),
            (1, 0, 0, 0, 0),
            (turn, turns - turn),
            (turn * 3, turn * 2),
        )
    rec.finish(winner="a", turns=turns, terminated=True, truncated=False)
    return rec


def test_write_then_read_reproduces_every_frame(tmp_path):
    path = _recorder(tmp_path).write()
    traj = read_trajectory(path)

    assert traj.header["v"] == FORMAT_VERSION
    assert (traj.game_id, traj.seed, traj.engine_version) == (
        "20260101T000000Z_a_vs_b_s7_abcd1234",
        7,
        "9e3b9d1",
    )
    assert (traj.header["H"], traj.header["W"]) == (19, 21)
    assert len(traj.frames) == 30
    assert traj.frames[0].turn == 1
    assert traj.frames[0].action_a == (0, 1, 2, 1, 0)
    assert traj.frames[-1].land == (30, 0)
    assert traj.end == {"winner": "a", "turns": 30, "terminated": True, "truncated": False}


def test_the_file_lands_where_the_game_id_says(tmp_path):
    path = _recorder(tmp_path).write()
    assert path == trajectory_path("20260101T000000Z_a_vs_b_s7_abcd1234", tmp_path)
    assert path.name.endswith(".traj.jsonl.gz")


def test_actions_are_stored_as_applied_not_as_validated(tmp_path):
    """Invalid moves stay in: resolving them is the engine's job, and a replay
    that dropped them would not be the same game."""
    rec = TrajectoryRecorder(
        game_id="g", seed=0, mode="competition", round_name="r",
        engine_version="e", bot_a="a", bot_b="b", H=3, W=3, directory=tmp_path,
    )
    rec.record_turn(1, (0, 99, 99, 3, 0), (1, 0, 0, 0, 0), (1, 1), (2, 2))
    rec.finish(winner="draw", turns=1, terminated=False, truncated=True)
    assert read_trajectory(rec.write()).frames[0].action_a == (0, 99, 99, 3, 0)


def test_series_are_readable_per_seat(tmp_path):
    traj = read_trajectory(_recorder(tmp_path, turns=4).write())
    assert traj.series("land", "a") == (1, 2, 3, 4)
    assert traj.series("land", "b") == (3, 2, 1, 0)
    assert traj.series("army", "b") == (2, 4, 6, 8)
    # The margin is one A-perspective quantity; its sign says whose it is.
    assert traj.series("land_margin", "a") == (-2, 0, 2, 4)
    assert traj.series("land_margin", "b") == (-2, 0, 2, 4)


def test_an_unknown_series_name_raises(tmp_path):
    traj = read_trajectory(_recorder(tmp_path, turns=2).write())
    with pytest.raises(ValueError, match="mountains"):
        traj.series("mountains", "a")


def test_digests_are_recorded_on_the_declared_cadence(tmp_path):
    class _FakeState:
        armies = [[1, 2], [3, 4]]
        ownership = [[[True, False], [False, True]]]
        castles = [[False, True], [True, False]]

    rec = TrajectoryRecorder(
        game_id="g", seed=0, mode="competition", round_name="r",
        engine_version="e", bot_a="a", bot_b="b", H=2, W=2,
        directory=tmp_path, digest_every=2,
    )
    for turn in (1, 2, 3, 4):
        rec.record_turn(turn, (0,) * 5, (0,) * 5, (1, 1), (1, 1), state=_FakeState())
    rec.finish(winner="draw", turns=4, terminated=False, truncated=True, state=_FakeState())

    traj = read_trajectory(rec.write())
    assert sorted(traj.digests) == [2, 4]
    assert all(d.startswith("sha256:") for d in traj.digests.values())
    assert len(traj.frames) == 4  # digest lines are not frames


def test_an_unfinished_trajectory_refuses_to_write(tmp_path):
    rec = TrajectoryRecorder(
        game_id="g", seed=0, mode="competition", round_name="r",
        engine_version="e", bot_a="a", bot_b="b", H=3, W=3, directory=tmp_path,
    )
    rec.record_turn(1, (0,) * 5, (0,) * 5, (1, 1), (1, 1))
    with pytest.raises(TrajectoryError, match="never finished"):
        rec.write()


def test_a_file_without_an_end_line_is_refused(tmp_path):
    path = tmp_path / "truncated.traj.jsonl.gz"
    write_jsonl_gz(
        [{"v": FORMAT_VERSION, "game_id": "g", "seed": 0, "engine_version": "e"},
         {"t": 1, "a": [0] * 5, "b": [0] * 5, "land": [1, 1], "army": [1, 1]}],
        path,
    )
    with pytest.raises(TrajectoryError, match="no end line"):
        read_trajectory(path)


def test_a_future_format_version_is_refused_rather_than_guessed(tmp_path):
    path = tmp_path / "future.traj.jsonl.gz"
    write_jsonl_gz([{"v": FORMAT_VERSION + 1}, {"end": {}}], path)
    with pytest.raises(TrajectoryError, match="format v"):
        read_trajectory(path)


def test_a_failed_write_leaves_no_partial_file(tmp_path, monkeypatch):
    """A killed worker must leave the whole trajectory or nothing at all."""
    import arena.records.trajectories as module

    path = tmp_path / "boom.traj.jsonl.gz"
    monkeypatch.setattr(module.os, "replace", _explode)
    with pytest.raises(OSError):
        write_jsonl_gz([{"v": FORMAT_VERSION}, {"end": {}}], path)

    assert not path.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def _explode(*args, **kwargs):
    raise OSError("disk went away")


def test_identical_trajectories_are_byte_identical(tmp_path):
    a = write_jsonl_gz([{"v": 1}, {"end": {}}], tmp_path / "a.jsonl.gz")
    b = write_jsonl_gz([{"v": 1}, {"end": {}}], tmp_path / "b.jsonl.gz")
    assert a.read_bytes() == b.read_bytes()


# --- probe traces -----------------------------------------------------------


def test_gzip_into_place_compresses_the_runners_plain_jsonl(tmp_path):
    raw = tmp_path / "trace.jsonl"
    raw.write_text(
        '{"t":1,"phase":"open"}\n{"t":2,"phase":"expand"}\n', encoding="utf-8"
    )
    destination = trace_path("g", "a", tmp_path)

    assert gzip_into_place(raw, destination) == destination
    assert not raw.exists()  # the runner's scratch file is consumed
    assert read_trace(destination) == {"phase": ["open", "expand"]}


def test_a_bot_with_no_probe_leaves_no_trace_file(tmp_path):
    raw = tmp_path / "empty.jsonl"
    raw.write_text("", encoding="utf-8")
    assert gzip_into_place(raw, trace_path("g", "b", tmp_path)) is None
    assert not trace_path("g", "b", tmp_path).exists()

    assert gzip_into_place(tmp_path / "never-written.jsonl", tmp_path / "x.gz") is None


def test_trace_series_are_grouped_by_key(tmp_path):
    path = tmp_path / "t.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for turn, phase in enumerate(["open", "raid"], start=1):
            handle.write(json.dumps({"t": turn, "phase": phase, "strikes": turn}) + "\n")
    assert read_trace(path) == {"phase": ["open", "raid"], "strikes": [1, 2]}


# --- layout -----------------------------------------------------------------


def test_round_directory_follows_the_game_record_naming_rule(tmp_path):
    assert round_trajectory_dir("round6", root=tmp_path) == tmp_path / "round6"
    with pytest.raises(ValueError, match="invalid round name"):
        round_trajectory_dir("../escape")


def test_trace_paths_are_seat_scoped(tmp_path):
    assert trace_path("g", "a", tmp_path).name == "g.trace.a.jsonl.gz"
    assert trace_path("g", "b", tmp_path).name == "g.trace.b.jsonl.gz"
    with pytest.raises(ValueError, match="seat"):
        trace_path("g", "c", tmp_path)


def test_the_reader_returns_a_frozen_trajectory(tmp_path):
    traj = read_trajectory(_recorder(tmp_path, turns=2).write())
    assert isinstance(traj, Trajectory)
    with pytest.raises(AttributeError):
        traj.frames = ()  # type: ignore[misc]
