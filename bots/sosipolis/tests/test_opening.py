"""Opening rung: forced passes to turn 2, first step at turn 3, end at OPEN_END."""
from __future__ import annotations

from _imports import agent_on, sosipolis_imports
from boards import BoardFixture, T_GENERAL, T_MOUNTAIN, action_dst, action_src, plain

ONLY_EXIT = (1, 2)


def _one_exit_board(turn: int) -> BoardFixture:
    """General at (1,1) with mountains on three sides; (1,2) is the only exit."""
    types, owner, army = plain(3, 3)
    types[1][1] = T_GENERAL
    owner[1][1] = 1
    army[1][1] = 5
    for r, c in ((0, 1), (2, 1), (1, 0)):
        types[r][c] = T_MOUNTAIN
    return BoardFixture(
        label=f"one_exit_t{turn}",
        turn=turn,
        types=types,
        owner=owner,
        army=army,
        own_general=(1, 1),
    )


def _open_board(turn: int) -> BoardFixture:
    types, owner, army = plain(5, 5)
    types[0][0] = T_GENERAL
    owner[0][0] = 1
    army[0][0] = 10
    return BoardFixture(
        label=f"open_t{turn}",
        turn=turn,
        types=types,
        owner=owner,
        army=army,
        own_general=(0, 0),
    )


def test_turn_two_passes_under_opening():
    with sosipolis_imports():
        from params import PASS

        fixture = _one_exit_board(2)
        agent = agent_on(fixture)

        move = agent.act(fixture.obs())

        assert agent.branch == "opening", agent.branch
        assert move == PASS


def test_turn_three_takes_the_only_neutral_neighbour():
    with sosipolis_imports():
        fixture = _one_exit_board(3)
        agent = agent_on(fixture)

        move = agent.act(fixture.obs())

        assert agent.branch == "opening", agent.branch
        assert action_src(move) == (1, 1)
        assert action_dst(move) == ONLY_EXIT


def test_open_end_is_the_last_opening_turn():
    with sosipolis_imports():
        from params import PARAMS

        assert PARAMS.OPEN_END == 50

        at_end = _open_board(50)
        last = agent_on(at_end)
        last.act(at_end.obs())
        assert last.branch == "opening", last.branch

        # Past OPEN_END the opening script is done. What follows is the
        # ordinary post-opening economy — `expand` on a wave tick with a
        # neutral in reach, `mcts` otherwise — but never `opening` again.
        past_end = _open_board(51)
        after = agent_on(past_end)
        after.act(past_end.obs())
        assert after.branch in ("expand", "mcts"), after.branch
        assert after.branch != "opening"
