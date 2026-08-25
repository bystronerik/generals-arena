"""Crafted synthetic fixtures for corpus blind spots (port-plan §6).

The corpus games start from real boards, so some branches are hit rarely or
never in a sampled frame: the seen-pad-mountain accumulation rule needs an
owned cell *adjacent to the pad border* of a smaller-than-21 board, and the
rule's memory (a pad cell once seen stays a mountain after visibility
leaves). This builds those frames directly and compares the Rust state
machine against the imported JAX oracle, bit for bit.

Marker `joe` (needs jax + the release binary).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from unclejoe_parity_lib import (
    BINARY,
    CELLS,
    N_CHANNELS,
    PAD,
    bits_f32,
    encode_frame,
    f32_bits,
    run_surface,
)

pytestmark = pytest.mark.joe

JOE_DIR = Path(__file__).resolve().parent.parent.parent / "joe"


@pytest.fixture(scope="module")
def oracle():
    if not BINARY.is_file():
        pytest.skip(f"no release binary at {BINARY}")
    if str(JOE_DIR) not in sys.path:
        sys.path.insert(0, str(JOE_DIR))
    import jax.numpy as jnp

    from joe_obs import augment_obs, build_cost_from_raw, init_obs_state

    def run(frames_raw):
        """Replay raw tensors through the JAX state machine, return augs."""
        state = init_obs_state(PAD)
        augs = []
        for raw in frames_raw:
            raw = jnp.asarray(raw)
            cost = build_cost_from_raw(raw)
            aug, state = augment_obs(raw, cost, state)
            augs.append(np.asarray(aug, dtype=np.float32))
        return augs

    return run


def frame(h, w, turn, cells):
    """Build one wire frame. `cells`: {(r, c): (type, owner, army)}."""
    scalars = [turn, 1, 1, 1, 1]
    grids = np.zeros((3, h, w), dtype=np.int64)
    grids[0] = 1  # plain everywhere
    for (r, c), (t, o, a) in cells.items():
        grids[0, r, c] = t
        grids[1, r, c] = o
        grids[2, r, c] = a
    return scalars, grids


def raw_from_frame(h, w, scalars, grids):
    """The Python bot's frame_to_raw on a crafted frame."""
    from _common.wire import Observation
    from agent import frame_to_raw

    obs = Observation(
        H=h, W=w, turn=scalars[0], my_land=scalars[1], my_army=scalars[2],
        opp_land=scalars[3], opp_army=scalars[4],
        type_grid=grids[0].tolist(), owner_grid=grids[1].tolist(),
        army_grid=grids[2].tolist())
    return frame_to_raw(obs)


def test_seen_pad_mountain_rule(oracle):
    """An owned cell in the corner of an 18x19 board makes pad cells visible
    (3x3 pool spills over the border); those pad cells accumulate as
    mountains and must stay mountains after the cell is lost — while
    never-seen pad cells read as structures-in-fog throughout."""
    h, w = 18, 19
    # Turn 1: own general in the far corner, adjacent to both pad borders.
    f1 = frame(h, w, 1, {(17, 18): (4, 1, 5)})
    # Turn 2: corner lost to the opponent — own visibility leaves the pad.
    f2 = frame(h, w, 2, {(17, 18): (4, 2, 3), (0, 0): (4, 1, 2)})
    # Turn 3: still gone, counters keep counting.
    f3 = frame(h, w, 3, {(17, 18): (4, 2, 4), (0, 0): (4, 1, 2)})
    frames = [f1, f2, f3]

    raws = [raw_from_frame(h, w, s, g) for s, g in frames]
    want_augs = oracle(raws)

    # Sanity of the crafted setup, against the oracle itself: pad neighbors
    # of (17, 18) are mountains on every turn (seen once, remembered), and a
    # far pad cell never seen stays structures-in-fog.
    aug1, aug3 = want_augs[0], want_augs[2]
    assert aug1[8, 18, 19] == 1.0 and aug1[8, 18, 18] == 1.0  # pad, seen
    assert aug3[8, 18, 19] == 1.0, "seen pad mountain must persist"
    assert aug1[13, 0, 20] == 1.0 and aug3[13, 0, 20] == 1.0  # never seen
    assert aug1[8, 0, 20] == 0.0

    # The Rust state machine over the same frames.
    case = [h, w, len(frames)]
    for scalars, grids in frames:
        case.extend(encode_frame(scalars, grids))
    got = run_surface("sequence", [case])
    import zlib

    for t, want in enumerate(want_augs):
        want_hash = np.int64(zlib.crc32(np.ascontiguousarray(want, dtype="<f4").tobytes()))
        assert got[t] == want_hash, f"synthetic pad-rule sequence diverges at turn {t}"


def test_pad_visibility_is_not_sticky_for_enemy(oracle):
    """Enemy visibility pools into the pad too (enemy_seen channel), and pad
    cells the *enemy* saw do not become mountains — only own visibility
    triggers the pad-mountain rule."""
    h, w = 18, 19
    f1 = frame(h, w, 1, {(17, 18): (4, 2, 5), (0, 0): (4, 1, 2)})
    f2 = frame(h, w, 2, {(17, 18): (4, 2, 6), (0, 0): (4, 1, 2)})
    frames = [f1, f2]
    raws = [raw_from_frame(h, w, s, g) for s, g in frames]
    want_augs = oracle(raws)

    aug2 = want_augs[1]
    assert aug2[5, 18, 19] == 1.0  # enemy_seen pools into the pad
    assert aug2[8, 18, 19] == 0.0  # but no mountain from enemy visibility
    assert aug2[13, 18, 19] == 1.0  # stays structures-in-fog

    case = [h, w, len(frames)]
    for scalars, grids in frames:
        case.extend(encode_frame(scalars, grids))
    got = run_surface("sequence", [case])
    import zlib

    for t, want in enumerate(want_augs):
        want_hash = np.int64(zlib.crc32(np.ascontiguousarray(want, dtype="<f4").tobytes()))
        assert got[t] == want_hash, f"enemy-pad sequence diverges at turn {t}"
