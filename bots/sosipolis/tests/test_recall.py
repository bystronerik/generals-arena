"""Recall rung: imminent adjacent loss first, then RECALL_PROX_D proximity.

Both sub-rules report `branch == "recall"`; the tests separate them by board.
"""
from __future__ import annotations

import pytest

from _imports import agent_on, sosipolis_imports
from boards import BoardFixture, T_GENERAL, action_dst, action_src, plain

IMMINENT_TURN = 200
PROXIMITY_TURN = 80  # past OPEN_END (50), before CASTLE_START_TURN (116)
HOME = (0, 0)


def _imminent_board(
    *, gen_army: int, threat_army: int, counter_army: int, feed_army: int
) -> BoardFixture:
    """Enemy at (0,1) next to home at (0,0); counter at (1,1); feeder at (1,0)."""
    H, W = 5, 5
    types, owner, army = plain(H, W)
    types[0][0] = T_GENERAL
    owner[0][0] = 1
    army[0][0] = gen_army
    owner[0][1] = 2
    army[0][1] = threat_army
    owner[1][1] = 1
    army[1][1] = counter_army
    owner[1][0] = 1
    army[1][0] = feed_army
    return BoardFixture(
        label=f"imminent_{gen_army}_{threat_army}_{counter_army}",
        turn=IMMINENT_TURN,
        types=types,
        owner=owner,
        army=army,
        own_general=HOME,
    )


def _proximity_board(*, enemy_col: int) -> BoardFixture:
    """One enemy tile on row 0 at manhattan `enemy_col` from home; tip at (4,0)."""
    H, W = 6, 6
    types, owner, army = plain(H, W)
    types[0][0] = T_GENERAL
    owner[0][0] = 1
    army[0][0] = 1
    owner[0][enemy_col] = 2
    army[0][enemy_col] = 2
    owner[4][0] = 1
    army[4][0] = 20
    return BoardFixture(
        label=f"proximity_d{enemy_col}",
        turn=PROXIMITY_TURN,
        types=types,
        owner=owner,
        army=army,
        own_general=HOME,
    )


def _home_dist(cell) -> int:
    return abs(cell[0] - HOME[0]) + abs(cell[1] - HOME[1])


# ---------------------------------------------------------------------------
# Imminent adjacent loss
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label, gen_army, threat_army, counter_army, expect_src, expect_dst",
    [
        # Counter stack wins the trade (8 - 1 > 5): capture the threat.
        ("counterattack", 1, 5, 8, (1, 1), (0, 1)),
        # No winning capture (3 - 1 <= 5): reinforce the general instead.
        ("reinforce_home", 1, 5, 3, (1, 0), (0, 0)),
        # Threat cannot take the general (5 - 1 < 5): rule does not arm.
        ("not_armed", 5, 5, 8, None, None),
    ],
)
def test_imminent_loss_move_cases(
    label, gen_army, threat_army, counter_army, expect_src, expect_dst
):
    with sosipolis_imports():
        from components.threat import imminent_loss_move

        fixture = _imminent_board(
            gen_army=gen_army,
            threat_army=threat_army,
            counter_army=counter_army,
            feed_army=4,
        )
        agent = agent_on(fixture)
        obs = fixture.obs()
        agent.state.update(obs)

        move = imminent_loss_move(obs, agent.state)
        if expect_src is None:
            assert move is None, label
        else:
            assert move is not None, label
            assert action_src(move) == expect_src, label
            assert action_dst(move) == expect_dst, label


@pytest.mark.parametrize(
    "label, counter_army, expect_src, expect_dst",
    [
        ("counterattack", 8, (1, 1), (0, 1)),
        ("reinforce_home", 3, (1, 0), (0, 0)),
    ],
)
def test_act_reports_recall_for_imminent_loss(
    label, counter_army, expect_src, expect_dst
):
    with sosipolis_imports():
        fixture = _imminent_board(
            gen_army=1, threat_army=5, counter_army=counter_army, feed_army=4
        )
        agent = agent_on(fixture)

        move = agent.act(fixture.obs())

        assert agent.branch == "recall", (label, agent.branch)
        assert action_src(move) == expect_src, label
        assert action_dst(move) == expect_dst, label
        assert agent.recall_fired == 1, label


# ---------------------------------------------------------------------------
# RECALL_PROX_D proximity
# ---------------------------------------------------------------------------


def test_recall_prox_d_is_three():
    with sosipolis_imports():
        from params import PARAMS

        assert PARAMS.RECALL_PROX_D == 3


def test_enemy_at_prox_d_pulls_tip_home():
    with sosipolis_imports():
        fixture = _proximity_board(enemy_col=3)
        agent = agent_on(fixture)

        move = agent.act(fixture.obs())

        assert agent.branch == "recall", agent.branch
        src, dst = action_src(move), action_dst(move)
        assert src is not None and dst is not None
        assert _home_dist(dst) < _home_dist(src)


def test_enemy_beyond_prox_d_leaves_turn_to_mcts():
    with sosipolis_imports():
        fixture = _proximity_board(enemy_col=4)
        agent = agent_on(fixture)

        agent.act(fixture.obs())

        assert agent.branch == "mcts", agent.branch
        assert agent.recall_fired == 0
