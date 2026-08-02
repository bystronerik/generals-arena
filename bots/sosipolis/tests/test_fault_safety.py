"""Fault safety: any exception inside `_act` degrades to a legal pass."""
from __future__ import annotations

from _imports import agent_on, sosipolis_imports
from boards import BoardFixture, T_GENERAL, plain

TURN = 200


def _simple_board() -> BoardFixture:
    types, owner, army = plain(5, 5)
    types[0][0] = T_GENERAL
    owner[0][0] = 1
    army[0][0] = 5
    return BoardFixture(
        label="fault_probe",
        turn=TURN,
        types=types,
        owner=owner,
        army=army,
        own_general=(0, 0),
    )


def _boom(_obs):
    raise RuntimeError("injected fault")


def test_exception_in_act_returns_pass():
    """`brain` binds `has_leave1_move` at import, so patch it on `brain`."""
    with sosipolis_imports():
        import brain
        from params import PASS

        fixture = _simple_board()
        agent = agent_on(fixture)

        original = brain.has_leave1_move
        brain.has_leave1_move = _boom
        try:
            move = agent.act(fixture.obs())
        finally:
            brain.has_leave1_move = original

        assert agent.branch == "fault", agent.branch
        assert move == PASS


def test_same_board_without_the_fault_does_not_report_fault():
    with sosipolis_imports():
        fixture = _simple_board()
        agent = agent_on(fixture)

        agent.act(fixture.obs())

        assert agent.branch != "fault", agent.branch
