"""Strike rung: exclusive tip feed while the tip is below TIP_AT_SIGHT_FLOOR."""
from __future__ import annotations

import pytest

from _imports import agent_on, sosipolis_imports
from boards import BoardFixture, T_GENERAL, action_dst, action_src, plain

TURN = 200
TIP = (2, 2)
FEEDER = (0, 1)
ENEMY_GENERAL = (5, 5)


def _strike_board(tip_army: int) -> BoardFixture:
    """Enemy general visible but far; tip at (2,2); one feeder stack at (0,1)."""
    types, owner, army = plain(6, 6)
    types[0][0] = T_GENERAL
    owner[0][0] = 1
    army[0][0] = 1
    owner[FEEDER[0]][FEEDER[1]] = 1
    army[FEEDER[0]][FEEDER[1]] = 3
    owner[TIP[0]][TIP[1]] = 1
    army[TIP[0]][TIP[1]] = tip_army
    # Far enough that no stack is adjacent, and rich enough to refuse a kill.
    types[ENEMY_GENERAL[0]][ENEMY_GENERAL[1]] = T_GENERAL
    owner[ENEMY_GENERAL[0]][ENEMY_GENERAL[1]] = 2
    army[ENEMY_GENERAL[0]][ENEMY_GENERAL[1]] = 50
    return BoardFixture(
        label=f"strike_tip{tip_army}",
        turn=TURN,
        types=types,
        owner=owner,
        army=army,
        own_general=(0, 0),
    )


def _tip_dist(cell) -> int:
    return abs(cell[0] - TIP[0]) + abs(cell[1] - TIP[1])


def test_sight_floor_is_ten():
    with sosipolis_imports():
        from params import PARAMS

        assert PARAMS.TIP_AT_SIGHT_FLOOR == 10


def test_tip_below_sight_floor_feeds():
    with sosipolis_imports():
        fixture = _strike_board(9)
        agent = agent_on(fixture)

        move = agent.act(fixture.obs())

        assert agent.phase == "strike", agent.phase
        assert agent.branch == "tip_feed", agent.branch
        assert agent.tip == TIP
        src, dst = action_src(move), action_dst(move)
        assert src is not None and dst is not None
        assert src != TIP
        assert _tip_dist(dst) < _tip_dist(src)


def test_tip_at_sight_floor_marches_under_mcts():
    with sosipolis_imports():
        fixture = _strike_board(10)
        agent = agent_on(fixture)

        agent.act(fixture.obs())

        assert agent.phase == "strike", agent.phase
        assert agent.branch == "mcts", agent.branch
        assert agent.tip == TIP


@pytest.mark.parametrize("tip_army, expect_below", [(9, True), (10, False)])
def test_tip_below_sight_floor_predicate(tip_army, expect_below):
    with sosipolis_imports():
        from components.tip import tip_below_sight_floor
        from params import PARAMS

        obs = _strike_board(tip_army).obs()
        assert tip_below_sight_floor(obs, TIP, PARAMS) is expect_below
