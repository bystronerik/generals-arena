"""What may record, what may be spawned instrumented, and where files land."""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from arena.matches import classic, competition, loop
from arena.records.fingerprint import BOTS_DIR
from arena.records.store import GAMES_DIR
from arena.records.trajectories import TRAJECTORIES_DIR, round_trajectory_dir

REPO_ROOT = Path(__file__).resolve().parent.parent


# --- spawn selection --------------------------------------------------------


def test_without_a_trace_the_command_is_exactly_bash_run_sh():
    """Unrecorded matches run byte-for-byte what they have always run."""
    run_sh = BOTS_DIR / "metro" / "run.sh"
    command, cwd = loop.agent_command(run_sh, None)
    assert command == ["bash", str(run_sh)]
    assert cwd == run_sh.parent


def test_with_a_trace_the_command_is_the_arena_runner(tmp_path):
    run_sh = BOTS_DIR / "metro" / "run.sh"
    command, cwd = loop.agent_command(run_sh, tmp_path / "a.jsonl")
    assert command[1:4] == ["-m", "arena.instrument.runner", str(run_sh.parent)]
    assert command[-2:] == ["--trace", str(tmp_path / "a.jsonl")]
    # The runner imports `arena`, so it cannot run from the bot's directory.
    assert cwd == REPO_ROOT


def test_only_bots_with_a_probe_are_traceable():
    """`cm_*` wrappers have no probe, so they are never spawned instrumented."""
    assert loop.has_probe(BOTS_DIR / "metro" / "run.sh")
    assert not loop.has_probe(BOTS_DIR / "cm_random" / "run.sh")
    assert not loop.has_probe(BOTS_DIR / "smoke" / "run.sh")


# --- scope ------------------------------------------------------------------


def test_the_classic_path_passes_no_recorder():
    """
    Classic results never enter `data/games/`, and they never record either.

    Same fencing, one reason: a classic match is a different ruleset, so
    anything derived from it must not sit beside competition data.
    """
    source = inspect.getsource(classic)
    assert "recorder" not in source
    assert "Trajectory" not in source
    assert "RecordRequest" not in source


def test_classic_run_takes_no_record_argument():
    assert "record" not in inspect.signature(classic.run_classic_match).parameters


def test_the_competition_path_is_the_only_recorder_constructor():
    """One construction site is what makes the scope guard checkable at all."""
    constructors = [
        path
        for path in (REPO_ROOT / "arena").rglob("*.py")
        if "TrajectoryRecorder(" in path.read_text(encoding="utf-8")
        and path.name != "trajectories.py"
    ]
    assert constructors == [REPO_ROOT / "arena" / "matches" / "competition.py"]


def test_the_remote_path_does_not_import_the_match_loop():
    for path in (REPO_ROOT / "arena" / "remote").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "arena.matches.loop" not in source, path
        assert "trajectories" not in source, path


def test_recording_defaults_to_off():
    assert competition.run_competition_match.__defaults__ is None
    signature = inspect.signature(competition.run_competition_match)
    assert signature.parameters["record"].default is None
    assert inspect.signature(loop.run_stdio_match).parameters["recorder"].default is None


# --- layout -----------------------------------------------------------------


def test_trajectories_live_in_their_own_root_not_beside_games():
    assert TRAJECTORIES_DIR == REPO_ROOT / "data" / "trajectories"
    for reserved in ("games", "classic_games", "remote_games"):
        assert not TRAJECTORIES_DIR.is_relative_to(REPO_ROOT / "data" / reserved)
    assert not GAMES_DIR.is_relative_to(TRAJECTORIES_DIR)


def test_a_round_directory_cannot_escape_the_trajectory_root():
    assert round_trajectory_dir("round6").parent == TRAJECTORIES_DIR
    with pytest.raises(ValueError, match="invalid round name"):
        round_trajectory_dir("../games")
