"""GameState.update only: phase transitions, per-phase clearing, land_at_50."""
from __future__ import annotations

import pytest

from _imports import sosipolis_imports
from boards import T_FOG, T_GENERAL, make_obs, plain

H = W = 4
HOME = (0, 0)
ENEMY = (3, 3)


def _obs(turn: int, *, enemy_land: bool = False, enemy_general: bool = False, **ov):
    types, owner, army = plain(H, W)
    types[HOME[0]][HOME[1]] = T_GENERAL
    owner[HOME[0]][HOME[1]] = 1
    army[HOME[0]][HOME[1]] = 5
    if enemy_land or enemy_general:
        owner[ENEMY[0]][ENEMY[1]] = 2
        army[ENEMY[0]][ENEMY[1]] = 4
    if enemy_general:
        types[ENEMY[0]][ENEMY[1]] = T_GENERAL
    return make_obs(types, owner, army, turn, **ov)


def _fogged_general_obs(turn: int):
    """Same board with the enemy general cell reported as fog (type 0)."""
    types, owner, army = plain(H, W)
    types[HOME[0]][HOME[1]] = T_GENERAL
    owner[HOME[0]][HOME[1]] = 1
    army[HOME[0]][HOME[1]] = 5
    types[ENEMY[0]][ENEMY[1]] = T_FOG
    return make_obs(types, owner, army, turn)


def _state():
    from params import PARAMS
    from state import GameState

    return GameState(H, W, PARAMS)


# ---------------------------------------------------------------------------
# Phase progression
# ---------------------------------------------------------------------------


def test_phase_walks_search_contact_strike_and_holds_through_fog():
    with sosipolis_imports():
        state = _state()

        state.update(_obs(10))
        assert state.phase == "search"

        state.update(_obs(20, enemy_land=True))
        assert state.phase == "contact"

        state.update(_obs(30, enemy_general=True))
        assert state.phase == "strike"
        assert state.memory.enemy_general == ENEMY

        state.update(_fogged_general_obs(40))
        assert state.phase == "strike"
        assert state.memory.enemy_general == ENEMY


# ---------------------------------------------------------------------------
# Per-phase clearing
# ---------------------------------------------------------------------------


def test_search_clears_strike_tip_and_contact_commitment():
    with sosipolis_imports():
        state = _state()
        state.strike_tip = (1, 1)
        state.contact_commitment = object()

        state.update(_obs(10))

        assert state.phase == "search"
        assert state.strike_tip is None
        assert state.contact_commitment is None


def test_strike_clears_contact_commitment_but_keeps_the_tip():
    with sosipolis_imports():
        state = _state()
        state.strike_tip = (1, 1)
        state.contact_commitment = object()

        state.update(_obs(30, enemy_general=True))

        assert state.phase == "strike"
        assert state.contact_commitment is None
        assert state.strike_tip == (1, 1)


@pytest.mark.parametrize(
    "label, owner_code, head_army, expect_head",
    [
        ("unowned", 0, 5, None),
        ("army_one", 1, 1, None),
        ("owned_stack", 1, 5, (2, 2)),
    ],
)
def test_chain_head_survives_only_on_a_movable_owned_stack(
    label, owner_code, head_army, expect_head
):
    with sosipolis_imports():
        types, owner, army = plain(H, W)
        types[HOME[0]][HOME[1]] = T_GENERAL
        owner[HOME[0]][HOME[1]] = 1
        army[HOME[0]][HOME[1]] = 5
        owner[2][2] = owner_code
        army[2][2] = head_army
        obs = make_obs(types, owner, army, 30)

        state = _state()
        state.chain_head = (2, 2)
        state.update(obs)

        assert state.chain_head == expect_head, label


# ---------------------------------------------------------------------------
# land_at_50
# ---------------------------------------------------------------------------


def test_land_at_50_latches_once_at_open_end():
    with sosipolis_imports():
        state = _state()

        state.update(_obs(49, my_land=11))
        assert state.land_at_50 is None

        state.update(_obs(50, my_land=17))
        assert state.land_at_50 == 17

        state.update(_obs(51, my_land=99))
        assert state.land_at_50 == 17
