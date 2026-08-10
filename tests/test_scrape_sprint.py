"""Tests for scripts/scrape_sprint.py: blob grouping, paths, and the sidecar.

Offline only. The network path is the submodule's retry policy, reused as-is;
what is new here is turning a results asset into one job per distinct blob.
"""
from __future__ import annotations

import gzip
import json

import pytest

from scripts.scrape_sprint import (
    Job,
    blob_stem,
    decode_replay,
    pair_dirname,
    plan_jobs,
    sidecar,
)

BLOB = "https://blob.example/sprint-replays/aaaa%7Cbbbb/1816681097-0.json.gz"


def row(**overrides):
    base = {
        "pair": "aaaa|bbbb",
        "p0_key": "aaaa",
        "p1_key": "bbbb",
        "p0_name": "Mattz",
        "p1_name": "Kubic",
        "seed": 1816681097,
        "a_side": 0,
        "winner": "p1",
        "turns": 346,
        "stage": "final",
        "replay_gz": BLOB,
    }
    base.update(overrides)
    return base


def test_blob_stem_drops_both_suffixes():
    assert blob_stem(BLOB) == "1816681097-0"


@pytest.mark.parametrize(
    "url",
    [
        "https://blob.example/x/%2E%2E%2F%2E%2E%2Fetc%2Fpasswd.json.gz",
        "https://blob.example/x/.json.gz",
        "https://blob.example/x/",
    ],
)
def test_blob_stem_rejects_escapes(url):
    assert blob_stem(url) is None


def test_pair_dirname_is_stable_across_sides():
    """`p0_key`/`p1_key` swap between the two sides' rows; `pair` does not."""
    side_a = row()
    side_b = row(p0_key="bbbb", p1_key="aaaa", p0_name="Kubic", p1_name="Mattz")
    assert pair_dirname(side_a) == pair_dirname(side_b) == "aaaa_bbbb"


@pytest.mark.parametrize("pair", ["aaaa", "", "aaaa|", "../evil|bbbb"])
def test_pair_dirname_rejects_unusable(pair):
    assert pair_dirname(row(pair=pair)) is None


def test_plan_jobs_groups_stage_duplicates(tmp_path):
    """One blob listed by both stages is one download, not two."""
    matches = [row(stage="final"), row(stage="qualifier")]
    jobs, skipped, unusable = plan_jobs(matches, tmp_path, None)
    assert (len(jobs), skipped, unusable) == (1, 0, 0)
    assert [r["stage"] for r in jobs[0].rows] == ["final", "qualifier"]
    assert jobs[0].replay_path == tmp_path / "aaaa_bbbb" / "1816681097-0.json"
    assert jobs[0].meta_path == tmp_path / "aaaa_bbbb" / "1816681097-0.meta.json"


def test_plan_jobs_skips_what_is_on_disk(tmp_path):
    other = BLOB.replace("1816681097-0", "1816681097-1")
    jobs, _, _ = plan_jobs([row(), row(replay_gz=other)], tmp_path, None)
    assert len(jobs) == 2
    jobs[0].replay_path.write_text("{}")

    jobs, skipped, _ = plan_jobs([row(), row(replay_gz=other)], tmp_path, None)
    assert (len(jobs), skipped) == (1, 1)
    assert jobs[0].replay_path.name == "1816681097-1.json"


def test_plan_jobs_counts_unusable_rows(tmp_path):
    jobs, _, unusable = plan_jobs([row(replay_gz=None), row(pair="aaaa")], tmp_path, None)
    assert (len(jobs), unusable) == (0, 2)


def test_plan_jobs_limit_caps_downloads(tmp_path):
    matches = [row(replay_gz=BLOB.replace("-0", f"-{i}")) for i in range(5)]
    jobs, _, _ = plan_jobs(matches, tmp_path, 2)
    assert len(jobs) == 2


def test_sidecar_nests_rows_and_sorts_stages(tmp_path):
    job = Job(BLOB, tmp_path / "r.json", tmp_path / "r.meta.json", [row(), row(stage="qualifier")])
    got = sidecar(job)
    assert got["stages"] == ["final", "qualifier"]
    assert got["matches"][0]["winner"] == "p1"
    # Nested on purpose: a top-level `winner` here would be read as the
    # leaderboard sidecar's A/B/D by arena.instrument.replay.Meta.
    assert "winner" not in got


def test_decode_replay_accepts_gzip_and_plain():
    payload = json.dumps({"version": 1}).encode()
    assert decode_replay(gzip.compress(payload)) == payload
    assert decode_replay(payload) == payload


def test_decode_replay_rejects_garbage():
    with pytest.raises(json.JSONDecodeError):
        decode_replay(b"not json and not gzip")
