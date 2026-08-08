"""
The committed smoke slice of the parity corpus: is it still what M1 will read?

This is the shore end of the parity harness. The Rust binary does not exist
yet, so there is nothing to compare *against* — what this file defends is the
contract the comparison will rest on: which fields a frame carries, which
strata are represented, and that the recorded RNG stream is complete.

When `morpheus-rs parity` lands (plan §5), the checks below stay and the
binary's output is compared frame by frame beside them. A format drift caught
here is cheap; the same drift caught after a Rust port is written against the
old shape is not.

The full corpus lives under `data/morpheus/morpheus-rs/` and is derived data.
Regenerate both with `scripts/morpheus_rs_baseline.py`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arena.instrument.capture import read_frames

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "parity-smoke.jsonl.gz"

# 9 policy channels + pass over a 21x21 board (artifact/manifest.json).
ACTION_SPACE = 9 * 21 * 21 + 1


@pytest.fixture(scope="module")
def frames() -> list[dict]:
    if not FIXTURE.is_file():
        pytest.skip(f"no parity smoke slice at {FIXTURE}")
    return read_frames(FIXTURE)


def test_the_slice_is_small_enough_to_commit(frames):
    """
    Bounded on purpose. Proving parity is the full corpus's job; this one has
    to stay cheap enough that CI runs it on every change.
    """
    assert 1 <= len(frames) <= 20
    assert FIXTURE.stat().st_size < 1_000_000


def test_every_frame_carries_the_per_turn_contract(frames):
    for frame in frames:
        assert frame["t"] >= 1
        assert frame["seat"] in (0, 1)
        assert len(frame["action"]) == 5
        assert len(frame["prev_action"]) == 5
        obs = frame["obs"]
        assert obs["type"].shape == obs["owner"].shape == obs["army"].shape
        assert obs["type"].shape == (obs["H"], obs["W"])
        assert frame["timing"]["move_ms"] >= 0
        assert frame["timing"]["component_ms"]


def test_the_recorded_draw_stream_is_complete(frames):
    """
    Replay in Rust consumes exactly this stream. A generator call the proxy did
    not record means the stream is short, and a replay that runs off the end
    diverges for a reason that has nothing to do with the port.
    """
    for frame in frames:
        assert frame["rng_unrecorded"] == {}
        for draw in frame["rng_draws"]:
            assert draw["m"] in {"integers", "choice", "random"}
            assert "r" in draw


def test_heavy_frames_carry_what_a_decision_needs(frames):
    heavy = [f for f in frames if f["heavy"]]
    assert heavy, "the slice is cut from heavy frames only"
    for frame in heavy:
        assert frame["play_mask"].dtype == np.bool_
        assert frame["play_mask"].shape == (ACTION_SPACE,)
        assert frame["shaping_scores"].shape == (ACTION_SPACE,)
        assert frame["shaping_scores"].dtype == np.float64
        tensor = frame["root"]["tensor"]
        assert tensor.shape == (1, 49, 21, 21)
        assert tensor.dtype == np.float32
        assert np.isfinite(tensor).all()


def test_belief_snapshots_are_weighted_particle_sets(frames):
    with_belief = [f for f in frames if "belief" in f]
    assert with_belief
    for frame in with_belief:
        belief = frame["belief"]
        assert belief["particles"]
        assert len(belief["weights"]) == len(belief["particles"])
        assert belief["ess"] > 0
        for particle in belief["particles"]:
            state = particle["state"]
            assert state["armies"].dtype == np.int32
            assert state["ownership"].shape[0] == 2
            assert state["general_positions"].shape == (2, 2)


def test_the_strata_span_the_game(frames):
    """
    §5 asks for pre-contact, post-contact, recovery, late-game and first-move
    coverage. A slice that lost a stratum would let a whole regime — the
    deathtouch transition, say — go unchecked while still looking populated.
    """
    strata = {f["stratum"] for f in frames}
    assert "first_move" in strata
    assert any(s.startswith("pre_contact") for s in strata)
    assert any(s.startswith("post_contact") for s in strata)
    assert any("recovery" in s for s in strata)
