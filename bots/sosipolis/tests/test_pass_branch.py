"""Pass rung: no leave-1 move exists, so the turn passes before recall runs."""
from __future__ import annotations

from _imports import agent_on, sosipolis_imports
from boards import BoardFixture, T_GENERAL, plain

TURN = 200


def _all_stacks_army_one() -> BoardFixture:
    """Every owned cell holds army 1; an enemy stack sits next to home."""
    H, W = 4, 4
    types, owner, army = plain(H, W)
    types[0][0] = T_GENERAL
    owner[0][0] = 1
    army[0][0] = 1
    owner[1][0] = 1
    army[1][0] = 1
    owner[0][1] = 2
    army[0][1] = 5
    return BoardFixture(
        label="no_leave1_move",
        turn=TURN,
        types=types,
        owner=owner,
        army=army,
        own_general=(0, 0),
    )


def test_no_leave1_move_passes_before_recall():
    with sosipolis_imports():
        from components.conveyor import has_leave1_move
        from params import PASS

        fixture = _all_stacks_army_one()
        obs = fixture.obs()
        assert not has_leave1_move(obs)

        agent = agent_on(fixture)
        move = agent.act(obs)

        assert agent.branch == "pass", agent.branch
        assert move == PASS
        assert move == (1, 0, 0, 0, 0)
        assert agent.recall_fired == 0
