"""Priority ladder order in `Agent._act`: lethal kill outranks imminent defense.

The board is built so that both rungs fire. `imminent_loss_move` is asserted
non-None first, so a passing kill assertion cannot be vacuous.
"""
from __future__ import annotations

from _imports import agent_on, sosipolis_imports
from boards import BoardFixture, T_GENERAL, action_dst, action_src, plain

TURN = 300  # past OPEN_END so the opening rung cannot answer the turn


def _both_rungs_armed(*, with_enemy_general: bool) -> BoardFixture:
    """Home under adjacent lethal threat; a second stack can take the general."""
    H, W = 5, 5
    types, owner, army = plain(H, W)
    types[0][0] = T_GENERAL
    owner[0][0] = 1
    army[0][0] = 1
    # Adjacent enemy that can capture the general (5 - 1 >= 1).
    owner[0][1] = 2
    army[0][1] = 5
    # Counterattacker for the defense rung (8 - 1 > 5).
    owner[1][1] = 1
    army[1][1] = 8
    if with_enemy_general:
        types[4][4] = T_GENERAL
        owner[4][4] = 2
        army[4][4] = 3
    # Killer for the lethal rung (6 - 1 >= 3 + FINISH_MARGIN).
    owner[4][3] = 1
    army[4][3] = 6
    return BoardFixture(
        label="kill_over_defense" if with_enemy_general else "defense_only",
        turn=TURN,
        types=types,
        owner=owner,
        army=army,
        own_general=(0, 0),
    )


def test_imminent_defense_is_armed_on_this_board():
    """Non-vacuity guard for the kill-preempts-defense test below."""
    with sosipolis_imports():
        from components.threat import imminent_loss_move

        fixture = _both_rungs_armed(with_enemy_general=True)
        agent = agent_on(fixture)
        obs = fixture.obs()
        agent.state.update(obs)

        defense = imminent_loss_move(obs, agent.state)
        assert defense is not None
        assert action_src(defense) == (1, 1)
        assert action_dst(defense) == (0, 1)


def test_kill_preempts_imminent_defense():
    """Rung 1 (kill) must win over rung 3 (recall) on the same observation."""
    with sosipolis_imports():
        fixture = _both_rungs_armed(with_enemy_general=True)
        agent = agent_on(fixture)

        move = agent.act(fixture.obs())

        assert agent.branch == "kill", agent.branch
        assert action_src(move) == (4, 3)
        assert action_dst(move) == (4, 4)


def test_defense_fires_when_no_kill_exists():
    """Same board without the enemy general: the defense rung answers."""
    with sosipolis_imports():
        fixture = _both_rungs_armed(with_enemy_general=False)
        agent = agent_on(fixture)

        move = agent.act(fixture.obs())

        assert agent.branch == "recall", agent.branch
        assert action_src(move) == (1, 1)
        assert action_dst(move) == (0, 1)
        assert agent.recall_fired == 1
