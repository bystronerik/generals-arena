"""
T12: lineage order comes from the registry, never from timestamps or the fit.

Uses a hand-built registry file rather than a git sandbox — what is under test
is the ordering and delta logic, and `tests/test_registry.py` already covers
the hash -> closure -> ref round trip.
"""
from __future__ import annotations

import json

import pytest

from arena.records.ratings.counts import build
from arena.records.ratings.fit import fit_ratings
from arena.records.ratings.lineage import lineage_deltas, lineage_table_lines, steps
from arena.records.ratings.policy import entity_key
from arena.records.registry import Registry

FIRST = "111111111111"
SECOND = "222222222222"
INHERITED = "333333333333"


def _version(content_hash: str, files: list[tuple[str, str]]) -> dict:
    return {
        "content_hash": content_hash,
        "first_seen_at": "2026-01-01T00:00:00Z",
        "git_commit": "a" * 40,
        "git_dirty": False,
        "closure_ref": f"refs/bot-versions/{content_hash}",
        "closure_tree": "b" * 40,
        "files": [{"path": p, "sha256": s, "blob": None} for p, s in files],
    }


@pytest.fixture
def registry(tmp_path) -> Registry:
    """expand_plus: v1 -> v2 (own edit) -> back to v1 (revert) -> v3 (inherited)."""
    payload = {
        "version": 1,
        "bot_id": "expand_plus",
        "versions": [
            _version(FIRST, [("bots/expand_plus/agent.py", "s1"), ("bots/_common/wire.py", "w1")]),
            _version(SECOND, [("bots/expand_plus/agent.py", "s2"), ("bots/_common/wire.py", "w1")]),
            # Only the shared file moved: the fork is real, but it is not an
            # expand_plus experiment.
            _version(INHERITED, [("bots/expand_plus/agent.py", "s1"), ("bots/_common/wire.py", "w2")]),
        ],
        # Deliberately *descending* timestamps: order must come from `seq`.
        "steps": [
            {"seq": 1, "content_hash": FIRST, "at": "2026-03-01T00:00:00Z"},
            {"seq": 2, "content_hash": SECOND, "at": "2026-02-01T00:00:00Z"},
            {"seq": 3, "content_hash": FIRST, "at": "2026-01-15T00:00:00Z",
             "note": "revert to seq 1"},
            {"seq": 4, "content_hash": INHERITED, "at": "2026-01-01T00:00:00Z"},
        ],
    }
    directory = tmp_path / "bot_versions"
    directory.mkdir()
    (directory / "expand_plus.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return Registry(directory)


@pytest.fixture
def fit():
    anchor = entity_key("cm_expander", "cccccccccccc")
    names = {h: entity_key("expand_plus", h) for h in (FIRST, SECOND, INHERITED)}
    tallies = {}
    for content_hash, wins in ((FIRST, 60), (SECOND, 30), (INHERITED, 55)):
        tallies[(anchor, names[content_hash])] = (100 - wins, wins, 20)
        tallies[(names[content_hash], anchor)] = (wins, 100 - wins, 20)
    return fit_ratings(build(tallies), anchor=anchor)


def test_steps_are_ordered_by_seq_not_by_timestamp(registry):
    ordered = steps("expand_plus", registry)
    assert [s.seq for s in ordered] == [1, 2, 3, 4]
    assert [s.at for s in ordered] == sorted([s.at for s in ordered], reverse=True)


def test_a_revert_reports_against_the_previous_step_not_the_previous_hash(registry, fit):
    rows = lineage_deltas("expand_plus", fit, registry)
    revert = rows[2]

    assert revert.seq == 3
    assert revert.content_hash == FIRST
    assert revert.previous_seq == 2
    assert revert.previous_hash == SECOND
    assert revert.note == "revert to seq 1"
    # seq 1 and seq 3 are the same entity, so the delta is exactly the negative
    # of seq 2's, not zero.
    assert revert.delta.value == pytest.approx(-rows[1].delta.value, abs=1e-9)
    assert revert.delta.value > 0  # seq 2 was the weaker program


def test_the_first_step_has_no_delta(registry, fit):
    first = lineage_deltas("expand_plus", fit, registry)[0]
    assert first.previous_seq is None
    assert first.delta is None
    assert first.p_stronger is None


def test_a_step_whose_own_directory_did_not_change_is_labelled_inherited(registry, fit):
    """q4: editing a bot that `proteus` imports forks proteus's lineage."""
    rows = lineage_deltas("expand_plus", fit, registry)
    assert [r.inherited for r in rows] == [False, False, False, True]
    assert rows[3].content_hash == INHERITED


def test_unrated_versions_are_reported_without_a_delta(registry):
    anchor = entity_key("cm_expander", "cccccccccccc")
    only_first = entity_key("expand_plus", FIRST)
    empty = fit_ratings(
        build({(anchor, only_first): (10, 10, 5), (only_first, anchor): (10, 10, 5)}),
        anchor=anchor,
    )
    rows = lineage_deltas("expand_plus", empty, registry)
    assert [r.rated for r in rows] == [True, False, True, False]
    assert rows[1].delta is None
    assert rows[2].delta is None  # its predecessor is unrated


def test_lineage_table_renders_every_step(registry, fit):
    lines = lineage_table_lines("expand_plus", fit, registry)
    assert lines[0].startswith("| Step |")
    assert len(lines) == 2 + 4
    assert "inherited" in lines[-1]
    assert "revert to seq 1" in lines[-2]


def test_an_unregistered_bot_renders_a_placeholder(registry, fit):
    assert lineage_table_lines("never_ran", fit, registry) == [
        "_no registered versions for `never_ran`_"
    ]
    assert lineage_deltas("never_ran", fit, registry) == []
