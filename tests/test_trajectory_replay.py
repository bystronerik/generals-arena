"""Replay: the era guard, the verify report, and the real acceptance test."""
from __future__ import annotations

from pathlib import Path

import pytest

from arena.records.trajectories import (
    EraMismatch,
    TrajectoryRecorder,
    read_trajectory,
    require_same_era,
    verify_trajectory,
)


def _canned(tmp_path, engine: str = "engine-a"):
    rec = TrajectoryRecorder(
        game_id="g", seed=0, mode="competition", round_name="r",
        engine_version=engine, bot_a="a", bot_b="b", directory=tmp_path,
    )
    rec.set_dims(3, 3)
    rec.record_turn(1, (1,) * 5, (1,) * 5, (1, 1), (1, 1))
    rec.finish(winner="draw", turns=1, terminated=False, truncated=True)
    return read_trajectory(rec.write())


# --- era guard (cheap: no engine needed) ------------------------------------


def test_replaying_across_an_engine_bump_is_refused(tmp_path):
    """
    A rules change moves what a state transitions to.

    Replaying an old trajectory under a new engine would produce a different
    game while looking like a successful reconstruction — the same reason
    records from two eras never pool in the fit.
    """
    traj = _canned(tmp_path, engine="old-sha")
    with pytest.raises(EraMismatch, match="old-sha"):
        require_same_era(traj, engine="new-sha")


def test_the_matching_era_passes(tmp_path):
    require_same_era(_canned(tmp_path, engine="same"), engine="same")


def test_verify_refuses_before_it_replays_anything(tmp_path):
    with pytest.raises(EraMismatch):
        verify_trajectory(_canned(tmp_path, engine="old"), engine="new")


# --- acceptance: record a real match, replay it, assert -----------------------


@pytest.mark.replay
def test_a_recorded_match_replays_state_equal(tmp_path):
    """
    The recorder's acceptance test (recorder plan §2.5).

    A game reconstructs iff all four hold: the per-turn land/army scalars match
    on every turn, every recorded state digest matches, the turn count matches,
    and the outcome matches. `verify_trajectory` checks all four and reports
    each failure with the turn it happened on.
    """
    from arena.matches.competition import RecordRequest, run_competition_match
    from arena.records.store import engine_version
    from arena.records.trajectories import trajectory_path

    bots = Path(__file__).resolve().parent.parent / "bots"
    request = RecordRequest(
        game_id="acceptance",
        round_name="acceptance",
        engine_version=engine_version(),
        directory=tmp_path,
    )
    result = run_competition_match(
        bots / "metro" / "run.sh", bots / "blitz" / "run.sh", seed=13, record=request
    )

    traj = read_trajectory(trajectory_path("acceptance", tmp_path))
    report = verify_trajectory(traj)

    assert report.ok, report.summary()
    assert report.turns == result.turns
    assert report.digests_checked >= 1
    assert traj.end["winner"] == result.winner


@pytest.mark.replay
def test_a_tampered_action_makes_the_replay_diverge(tmp_path):
    """The check has teeth: change one recorded move and verification fails."""
    from arena.matches.competition import RecordRequest, run_competition_match
    from arena.records.store import engine_version
    from arena.records.trajectories import (
        Trajectory,
        TurnFrame,
        trajectory_path,
    )

    bots = Path(__file__).resolve().parent.parent / "bots"
    run_competition_match(
        bots / "metro" / "run.sh",
        bots / "blitz" / "run.sh",
        seed=13,
        record=RecordRequest(
            game_id="tampered",
            round_name="acceptance",
            engine_version=engine_version(),
            directory=tmp_path,
        ),
    )

    traj = read_trajectory(trajectory_path("tampered", tmp_path))
    frames = list(traj.frames)
    victim = frames[len(frames) // 2]
    frames[len(frames) // 2] = TurnFrame(
        turn=victim.turn,
        action_a=(1, 0, 0, 0, 0),  # a pass where the bot actually moved
        action_b=victim.action_b,
        land=victim.land,
        army=victim.army,
    )
    tampered = Trajectory(
        header=traj.header, frames=tuple(frames), digests=traj.digests, end=traj.end
    )

    assert not verify_trajectory(tampered).ok
