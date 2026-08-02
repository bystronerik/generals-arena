"""resolve_objective: remembered general, contact waypoint, hunt, fog frontier."""
from __future__ import annotations

import pytest

from _imports import sosipolis_imports
from boards import make_obs, plain, probe_macro

HOME = (2, 2)
WAYPOINT = (0, 4)
HUNT = (4, 4)
GENERAL = (0, 0)


def _obs():
    types, owner, army = plain(5, 5)
    owner[HOME[0]][HOME[1]] = 1
    army[HOME[0]][HOME[1]] = 5
    return make_obs(types, owner, army, turn=120)


def _state(obs, *, phase, enemy_general, waypoint, hunt):
    from components.contact_mcts import ContactCommitment
    from params import PARAMS
    from state import GameState

    state = GameState(obs.H, obs.W, PARAMS)
    state.memory.own_general = HOME
    state.memory.enemy_general = enemy_general
    state.memory.hunt_cell = hunt
    state.phase = phase
    if waypoint is not None:
        state.contact_commitment = ContactCommitment(
            macro=probe_macro(kind="cluster", waypoint=waypoint),
            committed_turn=obs.turn,
            last_score=1.0,
        )
    return state


@pytest.mark.parametrize(
    "label, phase, enemy_general, waypoint, hunt, expected",
    [
        ("general_beats_commitment", "contact", GENERAL, WAYPOINT, HUNT, GENERAL),
        ("contact_uses_waypoint", "contact", None, WAYPOINT, HUNT, WAYPOINT),
        ("search_uses_hunt", "search", None, None, HUNT, HUNT),
        ("search_ignores_commitment", "search", None, WAYPOINT, HUNT, HUNT),
    ],
)
def test_objective_priority(label, phase, enemy_general, waypoint, hunt, expected):
    with sosipolis_imports():
        from components.conveyor import resolve_objective

        obs = _obs()
        state = _state(
            obs,
            phase=phase,
            enemy_general=enemy_general,
            waypoint=waypoint,
            hunt=hunt,
        )
        assert resolve_objective(obs, state) == expected, label


def test_contact_without_commitment_falls_back_to_fog_frontier():
    """No waypoint: the objective is an unowned cell next to own land."""
    with sosipolis_imports():
        from components.army import neighbors
        from components.conveyor import resolve_objective

        obs = _obs()
        state = _state(
            obs, phase="contact", enemy_general=None, waypoint=None, hunt=None
        )
        cell = resolve_objective(obs, state)
        assert cell is not None
        assert obs.owner_grid[cell[0]][cell[1]] != 1
        own_adjacent = [
            (nr, nc)
            for nr, nc in neighbors(obs.H, obs.W, cell[0], cell[1])
            if obs.owner_grid[nr][nc] == 1
        ]
        assert own_adjacent == [HOME]
