"""Castle rung gates: CASTLE_START_TURN, gather residue, funding, strike lock-out.

The site sits 7 steps from the general, so the RULES §03 distance surcharge is
zero and the build cost is exactly BUILD_BASE_COST.
"""
from __future__ import annotations

from _imports import agent_on, sosipolis_imports
from boards import BoardFixture, T_GENERAL, plain

SITE = (0, 7)
SITE_ARMY = 36  # BUILD_BASE_COST (35) + CASTLE_KEEP (1)
MY_LAND = 25  # override: must clear CASTLE_MIN_LAND (20)


def _castle_board(turn: int) -> BoardFixture:
    """Home general at (0,0); one funded plain stack at (0,7), no enemy in sight."""
    types, owner, army = plain(8, 8)
    types[0][0] = T_GENERAL
    owner[0][0] = 1
    army[0][0] = 1
    owner[SITE[0]][SITE[1]] = 1
    army[SITE[0]][SITE[1]] = SITE_ARMY
    return BoardFixture(
        label=f"castle_t{turn}",
        turn=turn,
        types=types,
        owner=owner,
        army=army,
        own_general=(0, 0),
    )


def test_build_cost_has_no_surcharge_at_distance_seven():
    with sosipolis_imports():
        from components.economy import build_cost
        from params import PARAMS

        assert PARAMS.CASTLE_START_TURN == 116
        assert PARAMS.BUILD_BASE_COST == 35
        obs = _castle_board(116).obs(my_land=MY_LAND)
        assert build_cost(obs, SITE[0], SITE[1], PARAMS) == 35


def test_gather_residue_before_start_turn_does_not_build():
    with sosipolis_imports():
        fixture = _castle_board(115)
        assert 10 <= fixture.turn % 50 <= 27

        agent = agent_on(fixture)
        agent.act(fixture.obs(my_land=MY_LAND))

        assert agent.branch == "mcts", agent.branch


def test_funded_site_on_gather_residue_builds_at_start_turn():
    with sosipolis_imports():
        fixture = _castle_board(116)
        assert 10 <= fixture.turn % 50 <= 27

        agent = agent_on(fixture)
        move = agent.act(fixture.obs(my_land=MY_LAND))

        assert agent.branch == "castle", agent.branch
        assert move[0] == 2
        assert (move[1], move[2]) == SITE


def test_strike_phase_blocks_the_castle_programme():
    with sosipolis_imports():
        from components.clock import Deadline
        from components.economy import Economy
        from params import PARAMS

        fixture = _castle_board(116)
        agent = agent_on(fixture)
        obs = fixture.obs(my_land=MY_LAND)
        agent.state.update(obs)

        economy = Economy(PARAMS)
        assert economy.decide(obs, agent.state, Deadline(50.0)) is not None

        agent.state.phase = "strike"
        assert economy.decide(obs, agent.state, Deadline(50.0)) is None
