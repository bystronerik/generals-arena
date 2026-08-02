"""SearchMCTS._score_expand: gather and wave read the same board differently.

`sections.score_cell` is stubbed to 0.0 so every expected number is a sum of
named params, and the recall gate is kept disarmed (no enemy tile within
RECALL_PROX_D of home) so `defense_score_delta` adds nothing.
"""
from __future__ import annotations

import pytest

from _imports import sosipolis_imports
from boards import T_GENERAL, make_obs, plain


def _search_state(H: int, W: int, *, home=None, muster=None, clock: str = "wave"):
    """GameState with memory seeded by hand; never calls state.update()."""
    from params import PARAMS
    from state import GameState

    st = GameState(H, W, PARAMS)
    st.memory.own_general = home
    st.memory._seeded = True
    st.muster = muster
    st.clock_phase = clock
    # Flat prior: keep every expected score a sum of named params.
    st.sections.score_cell = lambda r, c: 0.0
    return st


def _search_mcts():
    from components.search_mcts import SearchMCTS
    from params import PARAMS

    return SearchMCTS(PARAMS)


# ---------------------------------------------------------------------------
# gather clock
# ---------------------------------------------------------------------------


def test_gather_refuses_any_non_own_destination():
    """Gather never leaves own land, even on a capture the wave would take."""
    with sosipolis_imports():
        H = W = 7
        types, owner, army = plain(H, W)
        owner[3][3] = 1
        army[3][3] = 5
        obs = make_obs(types, owner, army, turn=100)
        state = _search_state(H, W, home=(0, 0))
        mcts = _search_mcts()

        gather = mcts._score_expand(obs, state, 3, 3, 3, 4, None, "gather")
        wave = mcts._score_expand(obs, state, 3, 3, 3, 4, None, "wave")
        assert gather == -1.0
        assert wave > 0.0


def test_gather_rewards_the_muster_closing_step():
    """Closing on the muster is worth exactly +60 over the same step away."""
    with sosipolis_imports():
        H = W = 7
        types, owner, army = plain(H, W)
        owner[3][3] = 1
        army[3][3] = 5
        owner[3][4] = 1
        owner[4][3] = 1
        obs = make_obs(types, owner, army, turn=100)
        state = _search_state(H, W, home=(0, 0), muster=(0, 6), clock="gather")
        mcts = _search_mcts()

        closing = mcts._score_expand(obs, state, 3, 3, 3, 4, None, "gather")
        away = mcts._score_expand(obs, state, 3, 3, 4, 3, None, "gather")
        assert closing == pytest.approx(5.0 + 60.0 + 2.0)
        assert away == pytest.approx(5.0 + 2.0)
        assert closing - away == pytest.approx(60.0)


def test_gather_rewards_draining_a_structure():
    """A general or castle source is worth +40 over a plain source."""
    with sosipolis_imports():
        H = W = 7
        types, owner, army = plain(H, W)
        types[0][0] = T_GENERAL
        owner[0][0] = 1
        army[0][0] = 5
        owner[0][1] = 1
        owner[3][3] = 1
        army[3][3] = 5
        owner[3][4] = 1
        obs = make_obs(types, owner, army, turn=100)
        # No muster: isolate the structure term from the closing term.
        state = _search_state(H, W, home=(0, 0), muster=None, clock="gather")
        mcts = _search_mcts()

        structure = mcts._score_expand(obs, state, 0, 0, 0, 1, None, "gather")
        plain_src = mcts._score_expand(obs, state, 3, 3, 3, 4, None, "gather")
        assert structure == pytest.approx(5.0 + 40.0 + 2.0)
        assert plain_src == pytest.approx(5.0 + 2.0)
        assert structure - plain_src == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# wave clock
# ---------------------------------------------------------------------------


def test_wave_pays_the_fog_bonus_only_for_unseen_land():
    """Unseen neutral land gets SEARCH_FOG_BONUS * 0.4; seen land gets 0.24x."""
    with sosipolis_imports():
        from params import PARAMS

        H = W = 7
        types, owner, army = plain(H, W)
        owner[3][3] = 1
        army[3][3] = 10
        # Second owned neighbour keeps (3,4) out of the corridor branch.
        owner[2][4] = 1
        army[2][4] = 1
        obs = make_obs(types, owner, army, turn=100)
        state = _search_state(H, W, home=(0, 0))
        mcts = _search_mcts()

        unseen = mcts._score_expand(obs, state, 3, 3, 3, 4, None, "wave")
        state.memory.ever_seen[3][4] = True
        seen = mcts._score_expand(obs, state, 3, 3, 3, 4, None, "wave")

        assert unseen == pytest.approx(
            10.0 + PARAMS.SEARCH_LAND_BONUS + PARAMS.SEARCH_FOG_BONUS * 0.4
        )
        assert seen == pytest.approx(
            10.0 + PARAMS.SEARCH_LAND_BONUS + PARAMS.SEARCH_LAND_BONUS * 0.24
        )
        assert unseen > seen


def test_wave_halves_a_non_closing_corridor_step():
    """One owned neighbour and no closure halves the score; closure exempts it."""
    with sosipolis_imports():
        from params import PARAMS

        H = W = 7
        types, owner, army = plain(H, W)
        owner[3][3] = 1
        army[3][3] = 10
        obs = make_obs(types, owner, army, turn=100)
        state = _search_state(H, W, home=(0, 0))
        mcts = _search_mcts()
        hunt = (6, 3)

        base = 10.0 + PARAMS.SEARCH_LAND_BONUS + PARAMS.SEARCH_FOG_BONUS * 0.4
        away = base - PARAMS.SEARCH_HUNT_BONUS * 0.25
        toward = base + PARAMS.SEARCH_HUNT_BONUS + PARAMS.SEARCH_HUNT_BONUS * 0.4

        corridor_step = mcts._score_expand(obs, state, 3, 3, 2, 3, hunt, "wave")
        closing_step = mcts._score_expand(obs, state, 3, 3, 4, 3, hunt, "wave")

        assert corridor_step == pytest.approx(away * 0.5)
        assert closing_step == pytest.approx(toward)
        assert corridor_step < closing_step


def test_wave_refuses_an_unwinnable_enemy_tile():
    """Leave-1 must beat the defender; equal army is refused with -1.0."""
    with sosipolis_imports():
        from params import PARAMS

        H = W = 7
        types, owner, army = plain(H, W)
        owner[3][3] = 1
        army[3][3] = 5
        owner[3][4] = 2
        army[3][4] = 4
        obs = make_obs(types, owner, army, turn=100)
        state = _search_state(H, W, home=(0, 0))
        mcts = _search_mcts()

        assert mcts._score_expand(obs, state, 3, 3, 3, 4, None, "wave") == -1.0

        army[3][3] = 6
        win_obs = make_obs(types, owner, army, turn=100)
        assert mcts._score_expand(
            win_obs, state, 3, 3, 3, 4, None, "wave"
        ) == pytest.approx(6.0 + PARAMS.SEARCH_ENEMY_BONUS + 4.0)


def test_wave_scenes_keep_the_recall_gate_disarmed():
    """Guard: no defense delta leaks into the expected sums above."""
    with sosipolis_imports():
        from components.threat import recall_armed
        from params import PARAMS

        H = W = 7
        types, owner, army = plain(H, W)
        owner[3][3] = 1
        army[3][3] = 5
        owner[3][4] = 2
        army[3][4] = 4
        obs = make_obs(types, owner, army, turn=100)
        state = _search_state(H, W, home=(0, 0))

        assert recall_armed(obs, state, PARAMS) is False
