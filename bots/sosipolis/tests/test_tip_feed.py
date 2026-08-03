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


def test_tip_at_sight_floor_marches_at_the_general():
    """At the floor the tip stops feeding and closes on the remembered cell.

    The rung that owns this is a deterministic near-BFS march, not the search:
    StrikeMCTS spent its whole budget on every strike tick, so the same
    position played differently depending on machine load.
    """
    with sosipolis_imports():
        fixture = _strike_board(10)
        agent = agent_on(fixture)

        move = agent.act(fixture.obs())

        assert agent.phase == "strike", agent.phase
        assert agent.branch != "tip_feed", agent.branch
        assert agent.tip == TIP
        src, dst = action_src(move), action_dst(move)
        assert src == TIP
        before = abs(src[0] - ENEMY_GENERAL[0]) + abs(src[1] - ENEMY_GENERAL[1])
        after = abs(dst[0] - ENEMY_GENERAL[0]) + abs(dst[1] - ENEMY_GENERAL[1])
        assert after < before


@pytest.mark.parametrize("tip_army, expect_below", [(9, True), (10, False)])
def test_tip_below_sight_floor_predicate(tip_army, expect_below):
    with sosipolis_imports():
        from components.tip import tip_below_sight_floor
        from params import PARAMS

        obs = _strike_board(tip_army).obs()
        assert tip_below_sight_floor(obs, TIP, PARAMS) is expect_below


# ---------------------------------------------------------------------------
# Contact rung: the gather half of the clock has to actually gather
# ---------------------------------------------------------------------------

CONTACT_TURN = 210  # residue 10 — inside GATHER_PHASE_LO..HI
WAVE_TURN = 230  # residue 30 — outside it


def _contact_board(turn: int, tip_army: int, spare: int, spares: int) -> BoardFixture:
    """Contact board with the army *dispersed*: the tip is the biggest stack,
    but most of our army sits on `spares` smaller cells elsewhere. That is the
    real shape — median 3% of our army on the tip at the closest approach.
    """
    types, owner, army = plain(8, 8)
    types[0][0] = T_GENERAL
    owner[0][0] = 1
    army[0][0] = 1
    owner[TIP[0]][TIP[1]] = 1
    army[TIP[0]][TIP[1]] = tip_army
    placed = 0
    for r in range(8):
        for c in range(8):
            if placed >= spares:
                break
            if (r, c) in ((0, 0), TIP) or (r, c) == (7, 7):
                continue
            owner[r][c] = 1
            army[r][c] = spare
            placed += 1
    owner[7][7] = 2  # enemy land only — phase is contact, not strike
    army[7][7] = 4
    return BoardFixture(
        label=f"contact_t{turn}_tip{tip_army}",
        turn=turn,
        types=types,
        owner=owner,
        army=army,
        own_general=(0, 0),
    )


def test_contact_gather_tick_feeds_a_thin_tip():
    """The feed rung used to require strike phase, which needs the general
    *sighted* — never in 8 of 20 games. So through the whole contact phase
    nothing concentrated army: measured at the tip's closest approach to their
    general, median tip 7 against a median 260 on our own board.
    """
    with sosipolis_imports():
        fixture = _contact_board(CONTACT_TURN, tip_army=5, spare=4, spares=12)
        agent = agent_on(fixture)

        move = agent.act(fixture.obs())

        assert agent.state.phase == "contact", agent.state.phase
        assert agent.branch == "tip_feed", agent.branch
        # Feeding moves army toward the tip, never away from it.
        assert _tip_dist(action_dst(move)) < _tip_dist(action_src(move))


def test_contact_feed_bar_is_a_share_of_our_army_not_a_constant():
    """A fixed bar capped the tail: arrivals of 50+ fell from 6 games to 1, and
    those are the games that end with a dead general. The tip here already
    clears every fixed threshold in the file and must still be fed.
    """
    with sosipolis_imports():
        from components.tip import total_owned_army
        from params import PARAMS

        fixture = _contact_board(CONTACT_TURN, tip_army=30, spare=20, spares=10)
        agent = agent_on(fixture)
        obs = fixture.obs()

        assert 30 >= PARAMS.CONTACT_ASSAULT_STACK
        assert 30 < PARAMS.CONTACT_FEED_FRAC * total_owned_army(obs)

        agent.act(obs)
        assert agent.branch == "tip_feed", agent.branch


def test_contact_wave_tick_does_not_feed():
    """Feeding belongs to the gather half of the mod-50 clock, not the wave."""
    with sosipolis_imports():
        fixture = _contact_board(WAVE_TURN, tip_army=5, spare=4, spares=12)
        agent = agent_on(fixture)

        agent.act(fixture.obs())

        assert agent.branch != "tip_feed", agent.branch
