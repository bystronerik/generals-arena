"""Wire-frame -> 14-channel raw tensor parsing (cheap, numpy only).

The jax-side fidelity against the live engine lives in
``test_wire_fidelity.py`` (marker ``joe``); this file pins the pure-python
channel semantics in the default suite.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_BOT_DIR = Path(__file__).resolve().parent.parent
# Unique module name: several bots expose a bare `agent`, and the default
# suite runs every bot's tests in one process.
_spec = importlib.util.spec_from_file_location("joe_agent", _BOT_DIR / "agent.py")
joe_agent = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(joe_agent)


class FakeObs:
    def __init__(self, H, W, type_grid, owner_grid, army_grid,
                 turn=7, my_land=3, my_army=12, opp_land=4, opp_army=9):
        self.H, self.W = H, W
        self.turn = turn
        self.my_land, self.my_army = my_land, my_army
        self.opp_land, self.opp_army = opp_land, opp_army
        self.type_grid = type_grid
        self.owner_grid = owner_grid
        self.army_grid = army_grid


def _obs():
    # 2x4 board exercising every cell type:
    #   (0,0) my general, (0,1) my plain, (0,2) visible neutral plain,
    #   (0,3) opp castle, (1,0) mountain, (1,1) fog, (1,2) structure-in-fog,
    #   (1,3) visible neutral castle (post-capture_all shape)
    type_grid = [[4, 1, 1, 3],
                 [2, 0, 5, 3]]
    owner_grid = [[1, 1, 0, 2],
                  [0, 0, 0, 0]]
    army_grid = [[10, 2, 0, 6],
                 [0, 0, 0, 1]]
    return FakeObs(2, 4, type_grid, owner_grid, army_grid)


def test_channel_semantics():
    raw = joe_agent.frame_to_raw(_obs())
    assert raw.shape == (14, 2, 4)
    assert raw.dtype == np.float32

    np.testing.assert_array_equal(raw[0], [[10, 2, 0, 6], [0, 0, 0, 1]])   # armies
    np.testing.assert_array_equal(raw[1], [[1, 0, 0, 0], [0, 0, 0, 0]])    # generals
    np.testing.assert_array_equal(raw[2], [[0, 0, 0, 1], [0, 0, 0, 1]])    # castles
    np.testing.assert_array_equal(raw[3], [[0, 0, 0, 0], [1, 0, 0, 0]])    # mountains
    # neutral: visible, unowned, passable — plain (0,2) and neutral castle
    # (1,3); NOT the mountain, NOT fog cells, NOT owned cells
    np.testing.assert_array_equal(raw[4], [[0, 0, 1, 0], [0, 0, 0, 1]])
    np.testing.assert_array_equal(raw[5], [[1, 1, 0, 0], [0, 0, 0, 0]])    # owned
    np.testing.assert_array_equal(raw[6], [[0, 0, 0, 1], [0, 0, 0, 0]])    # opponent
    np.testing.assert_array_equal(raw[7], [[0, 0, 0, 0], [0, 1, 0, 0]])    # fog
    np.testing.assert_array_equal(raw[8], [[0, 0, 0, 0], [0, 0, 1, 0]])    # structures in fog


def test_scalar_channels_broadcast():
    raw = joe_agent.frame_to_raw(_obs())
    for ch, val in ((9, 3), (10, 12), (11, 4), (12, 9), (13, 7)):
        assert (raw[ch] == val).all(), f"channel {ch}"
