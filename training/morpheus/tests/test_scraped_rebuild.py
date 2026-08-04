"""Scraped replay rebuild — labels, encoding, skip unresolved, verify gate."""

from __future__ import annotations

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
        output=tmp_path / "kubic-reconstructions",
        report_path=tmp_path / "report.json",
        max_games=1,
        force=True,
        repo_root=REPO,
    )
    assert report.scanned >= 1
    assert report.source_label == "Kubic_reconstructions"
    assert (tmp_path / "kubic-reconstructions" / "corpus-index.json").is_file()
    assert (
        report.kept
        + report.skipped_unresolved
        + report.skipped_verify
        + report.skipped_forfeit
        + report.skipped_other
        == report.scanned
    )
    assert report.kept >= 1, report.to_dict()
    from arena.records.trajectories import read_trajectory, verify_trajectory

    traj = read_trajectory(
        tmp_path
        / "kubic-reconstructions"
        / f"{report.kept_game_ids[0]}.traj.jsonl.gz"
    )
    assert verify_trajectory(traj).ok
