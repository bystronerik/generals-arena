"""Tests for unified bot API mapping."""
from __future__ import annotations

import logging
import sys
import types

import numpy as np
import pytest

from arena.bot_api import (
    PASS,
    ArenaAgent,
    StrategySession,
    from_competition_remote_obs,
    from_game_state,
    load_strategy_class,
    to_client_move,
    translate_action_for_remote,
)
from generals.core.observation import Observation as RemoteObservation
from generals_client.state import (
    TILE_EMPTY,
    TILE_FOG,
    TILE_FOG_OBSTACLE,
    TILE_MOUNTAIN,
    GameState,
)


def test_translate_build_to_pass():
    assert translate_action_for_remote((2, 1, 2, 0, 0)) == PASS


def test_from_competition_remote_obs_mapping():
    H, W = 3, 3
    armies = np.zeros((H, W), dtype=np.int32)
    armies[1, 1] = 5
    generals = np.zeros((H, W), dtype=bool)
    generals[1, 1] = True
    mountains = np.zeros((H, W), dtype=bool)
    mountains[0, 1] = True
    fog = np.zeros((H, W), dtype=bool)
    fog[2, 2] = True
    owned = np.zeros((H, W), dtype=bool)
    owned[1, 1] = True

    obs = RemoteObservation(
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
    unified = from_competition_remote_obs(obs)
    assert unified.H == 3 and unified.W == 3 and unified.turn == 7
    assert unified.type_grid[0][1] == 2
    assert unified.type_grid[1][1] == 4
    assert unified.type_grid[2][2] == 0
    assert unified.owner_grid[1][1] == 1
    assert unified.army_grid[1][1] == 5


def _make_game_state() -> GameState:
    state = GameState(
        {
            "playerIndex": 0,
            "replay_id": "test",
            "usernames": ["[Bot] a", "human"],
        }
    )
    # 3x3 map: width=3 height=3 size=9
    # layout: [w,h, armies x9, terrain x9]
    armies = [1, 0, 0, 0, 5, 0, 0, 0, 0]
    terrain = [
        TILE_FOG,
        TILE_MOUNTAIN,
        TILE_EMPTY,
        TILE_EMPTY,
        0,
        TILE_EMPTY,
        TILE_FOG_OBSTACLE,
        TILE_EMPTY,
        TILE_EMPTY,
    ]
    flat = [3, 3] + armies + terrain
    state.apply_update(
        {
            "turn": 4,
            "map_diff": [0, len(flat)] + flat,
            "cities_diff": [0, 1, 7],
            "generals": [4, -1],
            "scores": [
                {"i": 0, "tiles": 2, "total": 6},
                {"i": 1, "tiles": 1, "total": 1},
            ],
        }
    )
    return state


def test_from_game_state_mapping():
    state = _make_game_state()
    obs = from_game_state(state)
    assert obs.H == 3 and obs.W == 3 and obs.turn == 4
    assert obs.my_land == 2 and obs.my_army == 6
    assert obs.opp_land == 1 and obs.opp_army == 1
    assert obs.type_grid[0][0] == 0
    assert obs.type_grid[0][1] == 2
    assert obs.type_grid[1][1] == 4
    assert obs.type_grid[2][0] == 5
    assert obs.type_grid[2][1] == 3
    assert obs.owner_grid[1][1] == 1
    assert obs.army_grid[1][1] == 5
    assert obs.army_grid[0][0] == 0


def test_to_client_move_and_pass():
    state = _make_game_state()
    assert to_client_move(PASS, state) is None
    assert to_client_move((0, 1, 1, 0, 0), state) == (4, 1)
    assert to_client_move((0, 1, 1, 0, 1), state) == (4, 1, True)


def test_strategy_session_smoke():
    session = StrategySession("smoke")
    obs = from_game_state(_make_game_state())
    action = session.act(obs)
    assert len(action) == 5


def test_arena_agent_protocol_smoke():
    agent_cls = load_strategy_class("smoke")
    agent = agent_cls(player_id=0, H=3, W=3)
    assert isinstance(agent, ArenaAgent)


def _write_bot_with_helper(bots_root, name: str, value: str) -> None:
    bot_dir = bots_root / name
    bot_dir.mkdir(parents=True)
    (bot_dir / "helper.py").write_text(f"VALUE = {value!r}\n")
    (bot_dir / "agent.py").write_text(
        "from helper import VALUE\n"
        "\n"
        "class Agent:\n"
        "    def __init__(self, player_id, H, W):\n"
        "        self.value = VALUE\n"
        "\n"
        "    def act(self, obs):\n"
        "        return (1, 0, 0, 0, 0)\n"
    )


def test_colliding_sibling_modules_stay_private(tmp_path, monkeypatch):
    """Two bots shipping a same-named sibling each get their own copy."""
    import arena.bot_api as bot_api

    _write_bot_with_helper(tmp_path / "bots", "iso_alpha", "alpha-impl")
    _write_bot_with_helper(tmp_path / "bots", "iso_beta", "beta-impl")
    monkeypatch.setattr(bot_api, "REPO_ROOT", tmp_path)

    alpha_cls = load_strategy_class("iso_alpha")
    beta_cls = load_strategy_class("iso_beta")

    assert alpha_cls(player_id=0, H=1, W=1).value == "alpha-impl"
    assert beta_cls(player_id=0, H=1, W=1).value == "beta-impl"
    # The private sibling must not linger for the next load to alias.
    assert "helper" not in sys.modules

    sys.modules.pop("_arena_bot_iso_alpha_agent", None)
    sys.modules.pop("_arena_bot_iso_beta_agent", None)


def test_preloaded_foreign_sibling_module_raises(tmp_path, monkeypatch):
    """A foreign module already holding a sibling's name fails loudly."""
    import arena.bot_api as bot_api

    _write_bot_with_helper(tmp_path / "bots", "iso_gamma", "gamma-impl")
    monkeypatch.setattr(bot_api, "REPO_ROOT", tmp_path)

    foreign = types.ModuleType("helper")
    foreign.__file__ = "/somewhere/else/helper.py"
    monkeypatch.setitem(sys.modules, "helper", foreign)

    with pytest.raises(ImportError, match="helper"):
        load_strategy_class("iso_gamma")


def test_strategy_session_records_first_fault_traceback(caplog):
    class BrokenAgent:
        def __init__(self, player_id, H, W):
            pass

        def act(self, obs):
            raise RuntimeError("boom")

    session = StrategySession("smoke")
    session._strategy_class = BrokenAgent
    obs = from_game_state(_make_game_state())

    with caplog.at_level(logging.ERROR):
        action = session.act(obs)

    assert action == PASS
    assert session.faults == 1
    stats = session.session_stats()
    assert "first_fault_traceback" in stats
    assert "RuntimeError: boom" in stats["first_fault_traceback"]
    assert "StrategySession fault" in caplog.text

    session.act(obs)
    assert session.faults == 2
    assert stats["first_fault_traceback"] == session.session_stats()["first_fault_traceback"]
