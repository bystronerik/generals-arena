"""Tests for competition-module benchmark agent adapters."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BOTS_DIR = REPO_ROOT / "bots"
if str(BOTS_DIR) not in sys.path:
    sys.path.insert(0, str(BOTS_DIR))

from _common.cm_adapter import wire_to_cm_obs
from _common.wire import Observation
from arena.bot_api import from_competition_remote_obs
from generals.core.observation import Observation as CmObservation


def _wire_fixture() -> Observation:
    H, W = 3, 3
    type_grid = [
        [0, 2, 1],
        [1, 4, 1],
        [1, 1, 0],
    ]
    owner_grid = [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ]
    army_grid = [
        [0, 0, 0],
        [0, 5, 0],
        [0, 0, 0],
    ]
    return Observation(
        H=H,
        W=W,
        turn=7,
        my_land=1,
        my_army=5,
        opp_land=0,
        opp_army=0,
        type_grid=type_grid,
        owner_grid=owner_grid,
        army_grid=army_grid,
    )


def _cm_fixture() -> CmObservation:
    H, W = 3, 3
    armies = np.zeros((H, W), dtype=np.int32)
    armies[1, 1] = 5
    generals = np.zeros((H, W), dtype=bool)
    generals[1, 1] = True
    mountains = np.zeros((H, W), dtype=bool)
    mountains[0, 1] = True
    fog = np.zeros((H, W), dtype=bool)
    fog[0, 0] = True
    fog[2, 2] = True
    owned = np.zeros((H, W), dtype=bool)
    owned[1, 1] = True
    return CmObservation(
        armies=armies,
        generals=generals,
        castles=np.zeros((H, W), dtype=bool),
        mountains=mountains,
        neutral_cells=np.zeros((H, W), dtype=bool),
        owned_cells=owned,
        opponent_cells=np.zeros((H, W), dtype=bool),
        fog_cells=fog,
        structures_in_fog=np.zeros((H, W), dtype=bool),
        owned_land_count=1,
        owned_army_count=5,
        opponent_land_count=0,
        opponent_army_count=0,
        timestep=7,
    )


def test_wire_to_cm_obs_matches_from_competition_remote_obs():
    wire = _wire_fixture()
    cm_from_wire = wire_to_cm_obs(wire)
    unified = from_competition_remote_obs(_cm_fixture())
    assert int(cm_from_wire.timestep) == unified.turn
    assert int(cm_from_wire.owned_land_count) == unified.my_land
    assert bool(np.asarray(cm_from_wire.mountains)[0, 1])
    assert bool(np.asarray(cm_from_wire.generals)[1, 1])
    assert bool(np.asarray(cm_from_wire.fog_cells)[0, 0])
    assert bool(np.asarray(cm_from_wire.fog_cells)[2, 2])
    assert bool(np.asarray(cm_from_wire.owned_cells)[1, 1])
    assert int(np.asarray(cm_from_wire.armies)[1, 1]) == 5


@pytest.mark.parametrize(
    "bot_name",
    ["cm_random", "cm_expander", "cm_hunter", "cm_harvester"],
)
def test_benchmark_agents_return_valid_action(bot_name: str):
    bot_dir = BOTS_DIR / bot_name
    sys.path.insert(0, str(bot_dir))
    try:
        from agent import Agent  # type: ignore[import-not-found]
    finally:
        if str(bot_dir) in sys.path:
            sys.path.remove(str(bot_dir))

    agent = Agent(player_id=0, H=3, W=3)
    action = agent.act(_wire_fixture())
    assert len(action) == 5
    p, r, c, d, s = action
    assert p in (0, 1)
    assert 0 <= r < 3 and 0 <= c < 3
    assert 0 <= d <= 3
    assert s in (0, 1)
