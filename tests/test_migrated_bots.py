"""Compatibility and decision tests for the bots migrated from generals-bot.

Each migrated bot gets the arena smoke pair (protocol + session) plus a few
pure decision tests on hand-rolled boards.
"""
from __future__ import annotations

import pytest

from arena.instrument.probes import load_probe, probe_extras
from arena.bot_api import ArenaAgent, StrategySession, load_strategy_class
from arena.records.fingerprint import BOTS_DIR
from test_common_tactics import _grid, make_obs

MIGRATED_BOTS = ["blitz", "boom", "metro", "aegis", "proteus"]


def _basic_obs(turn=10):
    types = _grid(5, 5, fill=1)
    types[0][0] = 4
    types[4][4] = 0
    owner = _grid(5, 5)
    owner[0][0] = 1
    army = _grid(5, 5)
    army[0][0] = 5
    return make_obs(types, owner, army, turn=turn)


@pytest.mark.parametrize("name", MIGRATED_BOTS)
def test_agent_protocol(name):
    agent_cls = load_strategy_class(name)
    agent = agent_cls(player_id=0, H=5, W=5)
    assert isinstance(agent, ArenaAgent)
    action = agent.act(_basic_obs())
    assert len(action) == 5


@pytest.mark.parametrize("name", MIGRATED_BOTS)
def test_strategy_session(name):
    session = StrategySession(name)
    action = session.act(_basic_obs())
    assert len(action) == 5
    assert session.faults == 0


@pytest.mark.parametrize("name", MIGRATED_BOTS)
def test_probe_reads_the_agent_without_being_part_of_it(name):
    """Introspection moved out of the closure; the probe reads it from outside."""
    agent_cls = load_strategy_class(name)
    agent = agent_cls(player_id=0, H=5, W=5)
    agent.act(_basic_obs())

    assert not hasattr(agent, "telemetry_extras")
    extras = probe_extras(load_probe(BOTS_DIR / name), agent)
    assert isinstance(extras, dict) and extras


# ------------------------------------------------------------------- blitz
def test_blitz_finishing_move_takes_visible_general():
    agent_cls = load_strategy_class("blitz")
    agent = agent_cls(player_id=0, H=3, W=3)
    types = _grid(3, 3, fill=1)
    types[0][0] = 4  # our general
    types[2][2] = 4  # enemy general, visible
    owner = _grid(3, 3)
    owner[0][0] = 1
    owner[2][1] = 1
    owner[2][2] = 2
    army = _grid(3, 3)
    army[0][0] = 2
    army[2][1] = 10
    army[2][2] = 3
    obs = make_obs(types, owner, army, turn=100)
    action = agent.act(obs)
    # (2,1) -> right (direction 3) onto the enemy general.
    assert action == (0, 2, 1, 3, 0)
    assert probe_extras(load_probe(BOTS_DIR / "blitz"), agent)["phase"] == "finish"


def test_blitz_chain_launch_rule():
    import importlib

    blitz_agent = importlib.import_module("blitz.agent")
    cfg = blitz_agent.BlitzConfig()
    types = _grid(3, 3, fill=1)
    types[0][0] = 4
    owner = _grid(3, 3)
    owner[0][0] = 1
    army = _grid(3, 3)

    # 2*(army-1) >= turns_left: at turn 40 (10 left), a 6-stack launches...
    army[0][0] = 6
    obs = make_obs(types, owner, army, turn=40)
    assert blitz_agent.should_launch(obs, (0, 0), cfg)
    # ...but at turn 20 (30 left) it keeps accumulating.
    obs_early = make_obs(types, owner, army, turn=20)
    assert not blitz_agent.should_launch(obs_early, (0, 0), cfg)


def test_blitz_defends_home_when_threatened():
    agent_cls = load_strategy_class("blitz")
    agent = agent_cls(player_id=0, H=3, W=3)
    types = _grid(3, 3, fill=1)
    types[0][0] = 4
    owner = _grid(3, 3)
    owner[0][0] = 1
    owner[0][1] = 2  # 15-stack adjacent to our 2-army general
    owner[2][2] = 1  # our field army is too far to count as reinforcement
    army = _grid(3, 3)
    army[0][0] = 2
    army[0][1] = 15
    army[2][2] = 10
    obs = make_obs(types, owner, army, turn=100)
    action = agent.act(obs)
    # Deficit is real (no reinforcement can arrive in time), so blitz pulls
    # its field stack home: (2,2) starts walking toward the general.
    assert probe_extras(load_probe(BOTS_DIR / "blitz"), agent)["phase"] == "defend"
    assert action[0] == 0 and (action[1], action[2]) == (2, 2)


# -------------------------------------------------------------------- boom
def _boom():
    import importlib

    return importlib.import_module("boom.agent")


def test_boom_guard_locks_bank_under_threat():
    boom = _boom()
    types = _grid(5, 5, fill=1)
    types[0][0] = 4
    owner = _grid(5, 5)
    owner[0][0] = 1
    owner[0][2] = 2  # 20-stack two cells from the 5-army general
    owner[4][4] = 1  # field army too far to walk home in time
    army = _grid(5, 5)
    army[0][0] = 5
    army[0][2] = 20
    army[4][4] = 30
    obs = make_obs(types, owner, army, turn=100)
    mem = boom.BoomMemory()
    mem.observe(obs)
    params = boom.BoomParams()
    need = boom.guard_need(obs, mem.model, params)
    assert need > 5  # threat-scaled guard exceeds what is at home
    assert not boom.general_free(obs, mem.model, params, need)
    assert boom.threatened(obs, mem.model, params, need)


def test_boom_builds_castle_from_idle_surplus():
    boom = _boom()
    H = W = 9
    types = _grid(H, W, fill=1)
    types[0][0] = 4
    owner = [[1] * W for _ in range(H)]  # whole map owned: no expansion left
    army = [[1] * W for _ in range(H)]
    army[4][4] = 120  # a big idle stack far from the general
    obs = make_obs(types, owner, army, turn=100)
    mem = boom.BoomMemory()
    mem.observe(obs)
    params = boom.BoomParams()
    need = boom.guard_need(obs, mem.model, params)
    action = boom.build_move(obs, mem, boom.spare_army(obs, need), params)
    # The 120-stack cell covers cost + keep outright: build fires there.
    assert action == (2, 4, 4, 0, 0)


def test_metro_builds_castle_when_funded():
    import importlib

    metro = importlib.import_module("metro.agent")
    H = W = 9
    types = _grid(H, W, fill=1)
    types[0][0] = 4
    owner = [[1] * W for _ in range(H)]  # whole map owned, no enemies
    army = [[1] * W for _ in range(H)]
    army[8][8] = 80  # a funded stack on a cheap far cell
    obs = make_obs(types, owner, army, turn=100)
    core = metro.MetroCore(player_id=0, H=H, W=W)
    core.memory.observe(obs)
    strategy = core.strategy
    action = strategy._castle_programme(obs, core.memory, (0, 0))
    # The site scorer picks the funded far cell (cheap + forward is moot with
    # no enemy anchor beyond the mirror) or walks toward one; either way the
    # programme is active. Drive a few turns and expect a build to fire.
    for _ in range(30):
        if action is not None and action[0] == 2:
            break
        action = core.decide(obs)
    assert action is not None and action[0] == 2


def test_metro_prioritizes_enemy_castle_capture():
    import importlib

    metro = importlib.import_module("metro.agent")
    H = W = 7
    types = _grid(H, W, fill=1)
    types[0][0] = 4
    types[6][6] = 3  # enemy castle, far outside the defend radius
    owner = _grid(H, W)
    owner[0][0] = 1
    owner[6][5] = 1
    owner[6][6] = 2
    army = _grid(H, W)
    army[0][0] = 5
    army[6][5] = 20
    army[6][6] = 10
    obs = make_obs(types, owner, army, turn=200)
    core = metro.MetroCore(player_id=0, H=H, W=W)
    action = core.decide(obs)
    # Cash-in outranks expansion/pressure: (6,5) takes the castle rightward.
    assert action == (0, 6, 5, 3, 0)


def test_aegis_intercepts_over_expanding():
    import importlib

    aegis = importlib.import_module("aegis.agent")
    types = _grid(5, 5, fill=1)
    types[0][0] = 4
    owner = _grid(5, 5)
    owner[0][0] = 1
    owner[1][1] = 1
    owner[1][2] = 2  # incoming 6-stack, two steps from home
    army = _grid(5, 5)
    army[0][0] = 3
    army[1][1] = 10
    army[1][2] = 6
    obs = make_obs(types, owner, army, turn=100)
    core = aegis.AegisCore(player_id=0, H=5, W=5)
    action = core.decide(obs)
    # Local superiority: (1,1) kills the stack instead of expanding.
    assert action == (0, 1, 1, 3, 0)
    assert core.memory.phase == "intercept"


def test_aegis_perimeter_prefers_compact_tiles():
    import importlib

    aegis = importlib.import_module("aegis.agent")
    # Home at (1,1); two claimable neutrals: (0,1) sits against the map edge
    # (1 wall) and next to two owned cells; (1,2) is open ground.
    types = _grid(3, 3, fill=1)
    types[1][1] = 4
    owner = _grid(3, 3)
    owner[1][1] = 1
    owner[0][0] = 1
    army = _grid(3, 3)
    army[1][1] = 5
    army[0][0] = 2
    obs = make_obs(types, owner, army, turn=10)
    dist_home = aegis.multi_bfs(obs, [(1, 1)])
    move = aegis.perimeter_step(obs, (1, 1), dist_home, reserve=0)
    assert move is not None
    dest = (move[1] + [(-1, 0), (1, 0), (0, -1), (0, 1)][move[3]][0],
            move[2] + [(-1, 0), (1, 0), (0, -1), (0, 1)][move[3]][1])
    assert dest == (0, 1)  # the compact, wall-backed cell wins


def test_aegis_wave_estimate_uses_mobile_army():
    import importlib

    aegis = importlib.import_module("aegis.agent")
    cfg = aegis.AegisConfig()
    mem = aegis.AegisMemory()
    types = _grid(3, 3, fill=1)
    obs = make_obs(
        types, _grid(3, 3), _grid(3, 3),
        turn=100, opp_army=100, opp_land=40, my_army=50, my_land=20,
    )
    # mobile = 60; wave = max(seen*1.0, 0.10*60, 8) capped by 100-39.
    assert aegis.wave_estimate(obs, mem, cfg) == 8
    mem.model.biggest_enemy_stack = 30
    assert aegis.wave_estimate(obs, mem, cfg) == 30


def test_boom_endgame_latch_force_commits_late():
    boom = _boom()
    model = type(boom.BoomMemory().model)()
    params = boom.BoomParams()
    types = _grid(3, 3, fill=1)
    obs = make_obs(types, _grid(3, 3), _grid(3, 3), turn=960)
    # Even with expansion available and no army lead, turn >= 950 commits.
    assert boom.endgame_ready(obs, model, params, expansion_available=True)
