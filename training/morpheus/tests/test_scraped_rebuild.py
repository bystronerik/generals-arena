"""Scraped replay rebuild — labels, encoding, skip unresolved, verify gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.morpheus

from training.morpheus.curriculum.definitions import is_banned_source
from training.morpheus.scraped_rebuild.encode import (
    PASS_ACTION5,
    action_dict_to_action5,
    direction_index,
    tick_to_joint_action5,
)
from training.morpheus.scraped_rebuild.infer import InferenceResult
from training.morpheus.scraped_rebuild.source import (
    is_reconstruction_source,
    source_label_for,
)
from training.morpheus.scraped_rebuild.write_trajectory import (
    write_trajectory_from_inference,
)

REPO = Path(__file__).resolve().parents[3]
KUBIC_REPLAYS = REPO / "competition-replays" / "Kubic"


def test_source_label_for_player():
    assert source_label_for("ResBot") == "ResBot_reconstructions"
    assert source_label_for("erik.bystron") == "erik.bystron_reconstructions"
    assert source_label_for("ResBot_reconstructions") == "ResBot_reconstructions"
    assert is_reconstruction_source("ResBot_reconstructions")
    assert is_reconstruction_source("fixed_panel") is False


def test_ban_allows_reconstruction_suffix():
    assert is_banned_source("resbot") is True
    assert is_banned_source("scraped_resbot") is True
    assert is_banned_source("ResBot_reconstructions") is False
    assert is_banned_source("Kubic_reconstructions") is False


def test_encode_pass_move_build():
    assert action_dict_to_action5({"kind": "pass"}) == PASS_ACTION5
    assert direction_index((2, 2), (2, 3)) == 3  # right
    move = action_dict_to_action5(
        {"kind": "move", "src": [2, 2], "dst": [2, 3], "split": 1}
    )
    assert move == (0, 2, 2, 3, 1)
    build = action_dict_to_action5({"kind": "build", "cell": [4, 5], "cost": 35})
    assert build == (2, 4, 5, 0, 0)
    a, b = tick_to_joint_action5(
        {"t": 1, "p0": {"kind": "pass"}, "p1": {"kind": "pass"}}
    )
    assert a == PASS_ACTION5 and b == PASS_ACTION5


def test_write_skips_unresolved(tmp_path: Path):
    inference = InferenceResult(
        match_id="bad",
        players=("A", "B"),
        winner=-1,
        outcome="draw",
        rows=18,
        cols=18,
        seed=1,
        total_ticks=3,
        unresolved=[2],
        ambiguous=[],
        ticks=[
            {"t": 1, "p0": {"kind": "pass"}, "p1": {"kind": "pass"}},
            {"t": 2, "p0": {"kind": "unresolved"}, "p1": {"kind": "unresolved"}},
        ],
    )
    result = write_trajectory_from_inference(
        inference, player="ResBot", directory=tmp_path, force=True
    )
    assert result.ok is False
    assert result.reason == "unresolved"
    assert not list(tmp_path.glob("*.traj.jsonl.gz"))


def test_write_skips_missing_seed(tmp_path: Path):
    inference = InferenceResult(
        match_id="noseed",
        players=("A", "B"),
        winner=0,
        outcome="win",
        rows=18,
        cols=18,
        seed=None,
        total_ticks=1,
        unresolved=[],
        ambiguous=[],
        ticks=[{"t": 1, "p0": {"kind": "pass"}, "p1": {"kind": "pass"}}],
    )
    result = write_trajectory_from_inference(
        inference, player="X", directory=tmp_path, force=True
    )
    assert result.ok is False
    assert result.reason == "missing_seed"


@pytest.mark.skipif(
    not (KUBIC_REPLAYS / "win").is_dir(),
    reason="Kubic scraped replays absent",
)
def test_rebuild_one_kubic_game_or_skip(tmp_path: Path):
    from arena.instrument.replay.loader import iter_replay_paths, load_replay
    from training.morpheus.scraped_rebuild.rebuild import rebuild_player

    candidates = []
    for folder, path in iter_replay_paths("Kubic", root=KUBIC_REPLAYS.parent):
        replay = load_replay(path, "Kubic", folder)
        if replay.is_forfeit or replay.seed is None:
            continue
        candidates.append(replay.total_ticks)
        if len(candidates) >= 3:
            break
    assert candidates, "expected at least one non-forfeit Kubic replay"

    report = rebuild_player(
        player="Kubic",
        replays=KUBIC_REPLAYS,
        output=tmp_path / "rounds",
        report_path=tmp_path / "report.json",
        max_games=1,
        force=True,
        repo_root=REPO,
    )
    root = tmp_path / "rounds"
    assert report.scanned >= 1
    assert report.source_label == "Kubic_reconstructions"
    assert (
        report.kept
        + report.skipped_unresolved
        + report.skipped_verify
        + report.skipped_forfeit
        + report.skipped_other
        + report.skipped_outcome
        == report.scanned
    )
    assert report.kept >= 1, report.to_dict()
    from arena.records.trajectories import read_trajectory, verify_trajectory

    game_id = report.kept_game_ids[0]
    assert sum(report.kept_by_outcome.values()) == report.kept
    # Exactly the rounds that kept something, named per player and result.
    assert set(report.rounds) == {
        o for o, n in report.kept_by_outcome.items() if n
    }
    outcome = next(iter(report.rounds))
    round_dir = root / f"Kubic-{outcome}-reconstructions"
    assert Path(report.rounds[outcome]) == round_dir
    assert (round_dir / "corpus-index.json").is_file()

    index = json.loads((round_dir / "corpus-index.json").read_text())
    assert index["outcome"] == outcome
    assert index["round"] == round_dir.name
    meta = index["games"][game_id]
    assert meta["queried_outcome"] == outcome

    # Rounds are flat, and the round root holds no stray index.
    path = round_dir / f"{game_id}.traj.jsonl.gz"
    assert path.is_file()
    assert not (root / "corpus-index.json").exists()

    traj = read_trajectory(path)
    assert verify_trajectory(traj).ok
    assert traj.header["round"] == round_dir.name


@pytest.mark.skipif(
    not (KUBIC_REPLAYS / "win").is_dir(),
    reason="Kubic scraped replays absent",
)
def test_rebuild_outcome_filter_and_keep(tmp_path: Path):
    from arena.instrument.replay.loader import iter_replay_paths, load_replay
    from training.morpheus.scraped_rebuild.rebuild import rebuild_player

    # Pick an outcome that exists for Kubic.
    outcomes = set()
    for folder, path in iter_replay_paths("Kubic", root=KUBIC_REPLAYS.parent):
        replay = load_replay(path, "Kubic", folder)
        if replay.is_forfeit or replay.seed is None:
            continue
        outcomes.add(replay.outcome)
        if len(outcomes) >= 1:
            break
    assert outcomes
    wanted = next(iter(outcomes))

    report = rebuild_player(
        player="Kubic",
        replays=KUBIC_REPLAYS,
        output=tmp_path / "rounds",
        report_path=tmp_path / "report.json",
        keep=1,
        outcome=wanted,
        force=True,
        repo_root=REPO,
    )
    assert report.outcome_filter == wanted
    assert report.keep_target == 1
    assert report.kept == 1
    assert report.scanned >= 1
    round_dir = tmp_path / "rounds" / f"Kubic-{wanted}-reconstructions"
    # A filtered run writes that one round and no empty siblings.
    assert set(report.rounds) == {wanted}
    assert sorted(p.name for p in (tmp_path / "rounds").iterdir()) == [round_dir.name]
    # Outcome filter must not count folder-only skips as kept.
    for game_id in report.kept_game_ids:
        meta = json.loads((round_dir / "corpus-index.json").read_text())["games"][game_id]
        assert meta["queried_outcome"] == wanted
        assert meta["sample_seat"] in (0, 1)
        assert (round_dir / f"{game_id}.traj.jsonl.gz").is_file()
    assert report.kept_by_outcome[wanted] == 1


def test_round_dir_name_and_validation(tmp_path: Path):
    from arena.records.trajectories import round_trajectory_dir
    from training.morpheus.scraped_rebuild.write_trajectory import (
        round_dir_name,
        round_directory,
        safe_player_name,
    )

    assert round_dir_name("Kubic", "win") == "Kubic-win-reconstructions"
    assert round_directory(tmp_path, "Kubic", "draw") == (
        tmp_path / "Kubic-draw-reconstructions"
    )
    # Spaces and slashes cannot appear in a round name.
    assert safe_player_name("Non-Linear Slob") == "Non-Linear_Slob"
    assert safe_player_name("a/b") == "a_b"
    assert round_dir_name("Non-Linear Slob", "lose") == (
        "Non-Linear_Slob-lose-reconstructions"
    )
    # Every generated name must satisfy the arena's own round-name rule.
    for player in ("Non-Linear Slob", "erik.bystron", "__", "a/b"):
        for outcome in ("win", "lose", "draw"):
            round_trajectory_dir(round_dir_name(player, outcome))

    with pytest.raises(ValueError, match="outcome"):
        round_dir_name("Kubic", "loss")


def test_rebuild_rejects_bad_outcome():
    from training.morpheus.scraped_rebuild.rebuild import rebuild_player

    with pytest.raises(ValueError, match="outcome"):
        rebuild_player(
            player="Kubic",
            output=REPO / "data" / "trajectories" / "_unused",
            outcome="sideA",
            keep=1,
        )
