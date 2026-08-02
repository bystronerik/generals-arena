"""StrikeMCTS dispatch: the four early returns and the root-move shape.

`search` is only called for scenes that return before the UCT loop, so a
10 s deadline costs nothing. Root-move shape is checked by calling
`_root_moves` directly.
"""
from __future__ import annotations

from _imports import sosipolis_imports
from boards import (
    T_GENERAL,
    action_dst,
    action_src,
    make_obs,
    plain,
)


PASS_ACTION = (1, 0, 0, 0, 0)


def _strike_state(
    H: int,
    W: int,
    *,
    home,
    enemy_general,
    clock: str = "wave",
    strike_tip=None,
    muster=None,
):
    """GameState with memory seeded by hand; never calls state.update()."""
    from params import PARAMS
    from state import GameState

    st = GameState(H, W, PARAMS)
    st.memory.own_general = home
    st.memory.enemy_general = enemy_general
    st.memory._seeded = True
    if enemy_general is not None:
        st.memory.known_owner[enemy_general[0]][enemy_general[1]] = 2
        st.phase = "strike"
    st.clock_phase = clock
    st.strike_tip = strike_tip
    st.muster = muster
    return st


def _strike_mcts():
    from components.strike_mcts import StrikeMCTS
    from params import PARAMS

    return StrikeMCTS(PARAMS)


# ---------------------------------------------------------------------------
# search: early returns before the UCT loop
# ---------------------------------------------------------------------------


def test_search_passes_without_known_enemy_general():
    with sosipolis_imports():
        from components.clock import Deadline

        H = W = 5
        types, owner, army = plain(H, W)
        types[0][0] = T_GENERAL
        owner[0][0] = 1
        army[0][0] = 5
        obs = make_obs(types, owner, army, turn=100)
        state = _strike_state(H, W, home=(0, 0), enemy_general=None)
        mcts = _strike_mcts()

        assert mcts.search(obs, state, Deadline(10_000)) == PASS_ACTION
        assert mcts.stats.iterations == 0


def test_search_takes_the_kill_before_anything_else():
    """Adjacent stack with the finish margin captures the general."""
    with sosipolis_imports():
        from components.clock import Deadline

        H = W = 5
        types, owner, army = plain(H, W)
        types[0][0] = T_GENERAL
        owner[0][0] = 1
        army[0][0] = 3
        types[2][2] = T_GENERAL
        owner[2][2] = 2
        army[2][2] = 8
        owner[2][1] = 1
        army[2][1] = 12
        obs = make_obs(types, owner, army, turn=572)
        state = _strike_state(H, W, home=(0, 0), enemy_general=(2, 2))
        mcts = _strike_mcts()

        move = mcts.search(obs, state, Deadline(10_000))
        assert action_src(move) == (2, 1)
        assert action_dst(move) == (2, 2)
        assert mcts.stats.iterations == 0


def test_search_defends_imminent_loss_when_no_kill_exists():
    """An adjacent enemy that can take home outranks the march."""
    with sosipolis_imports():
        from components.clock import Deadline

        H = W = 7
        types, owner, army = plain(H, W)
        types[0][0] = T_GENERAL
        owner[0][0] = 1
        army[0][0] = 1
        owner[0][1] = 2
        army[0][1] = 5
        owner[1][1] = 1
        army[1][1] = 10
        types[6][6] = T_GENERAL
        owner[6][6] = 2
        army[6][6] = 50
        obs = make_obs(types, owner, army, turn=100)
        state = _strike_state(H, W, home=(0, 0), enemy_general=(6, 6))
        mcts = _strike_mcts()

        assert mcts._kill_shot(obs, (6, 6)) is None
        move = mcts.search(obs, state, Deadline(10_000))
        assert action_src(move) == (1, 1)
        assert action_dst(move) == (0, 1)
        assert mcts.stats.iterations == 0


def test_search_feeds_a_tip_below_the_sight_floor():
    """Below TIP_AT_SIGHT_FLOOR the turn is one exclusive feed move."""
    with sosipolis_imports():
        from components.clock import Deadline
        from components.tip import tip_below_sight_floor
        from params import PARAMS

        H = W = 7
        types, owner, army = plain(H, W)
        types[0][0] = T_GENERAL
        owner[0][0] = 1
        army[0][0] = 1
        owner[0][3] = 1
        army[0][3] = PARAMS.TIP_AT_SIGHT_FLOOR - 2
        owner[0][1] = 1
        army[0][1] = 5
        types[6][6] = T_GENERAL
        owner[6][6] = 2
        army[6][6] = 3
        obs = make_obs(types, owner, army, turn=100)
        state = _strike_state(H, W, home=(0, 0), enemy_general=(6, 6))
        mcts = _strike_mcts()

        move = mcts.search(obs, state, Deadline(10_000))
        assert state.strike_tip == (0, 3)
        assert tip_below_sight_floor(obs, (0, 3), PARAMS) is True
        assert action_src(move) == (0, 1)
        assert action_dst(move) == (0, 2)
        assert mcts.stats.root_moves == 1
        assert mcts.stats.iterations == 0


# ---------------------------------------------------------------------------
# _root_moves
# ---------------------------------------------------------------------------


def _feed_board():
    """Tip at (3,3) with one feeder at (3,1); enemy general at (6,6)."""
    H = W = 7
    types, owner, army = plain(H, W)
    types[0][0] = T_GENERAL
    owner[0][0] = 1
    army[0][0] = 3
    owner[3][1] = 1
    army[3][1] = 20
    owner[3][2] = 1
    army[3][2] = 1
    owner[3][3] = 1
    army[3][3] = 40
    types[6][6] = T_GENERAL
    owner[6][6] = 2
    army[6][6] = 5
    return H, W, types, owner, army


def test_root_moves_gather_clock_feeds_even_when_the_tip_is_ready():
    """Gather consolidates: no root leaves the tip, every dst is own land."""
    with sosipolis_imports():
        H, W, types, owner, army = _feed_board()
        obs = make_obs(types, owner, army, turn=100)
        tip = (3, 3)
        state = _strike_state(
            H,
            W,
            home=(0, 0),
            enemy_general=(6, 6),
            clock="gather",
            strike_tip=tip,
            muster=tip,
        )
        mcts = _strike_mcts()

        roots = mcts._root_moves(
            obs, state, goal=(6, 6), tip=tip, ready=True, clock="gather"
        )
        assert roots
        assert all(action_src(a) != tip for a, _ in roots)
        assert {(action_src(a), action_dst(a)) for a, _ in roots} == {
            ((3, 1), (3, 2))
        }
        for action, _ in roots:
            dr, dc = action_dst(action)
            assert obs.owner_grid[dr][dc] == 1


def test_root_moves_wave_clock_feeds_while_the_tip_is_not_ready():
    """An underfed tip stays put; the roots are feeder steps only."""
    with sosipolis_imports():
        H, W, types, owner, army = _feed_board()
        obs = make_obs(types, owner, army, turn=100)
        tip = (3, 3)
        state = _strike_state(
            H,
            W,
            home=(0, 0),
            enemy_general=(6, 6),
            clock="wave",
            strike_tip=tip,
            muster=tip,
        )
        mcts = _strike_mcts()

        roots = mcts._root_moves(
            obs, state, goal=(6, 6), tip=tip, ready=False, clock="wave"
        )
        assert roots
        assert all(action_src(a) != tip for a, _ in roots)


def test_root_moves_wave_clock_marches_a_ready_tip():
    """Ready tip steps toward the general and only takes on-path captures."""
    with sosipolis_imports():
        from components.threat import recall_armed
        from params import PARAMS

        H = W = 7
        types, owner, army = plain(H, W)
        types[0][0] = T_GENERAL
        owner[0][0] = 1
        army[0][0] = 1
        tip = (3, 3)
        owner[3][3] = 1
        army[3][3] = 40
        owner[3][2] = 2
        army[3][2] = 3
        types[6][6] = T_GENERAL
        owner[6][6] = 2
        army[6][6] = 5
        obs = make_obs(types, owner, army, turn=100)
        state = _strike_state(
            H,
            W,
            home=(0, 0),
            enemy_general=(6, 6),
            clock="wave",
            strike_tip=tip,
            muster=tip,
        )
        mcts = _strike_mcts()
        assert recall_armed(obs, state, PARAMS) is False

        roots = mcts._root_moves(
            obs, state, goal=(6, 6), tip=tip, ready=True, clock="wave"
        )
        assert roots
        assert {action_src(a) for a, _ in roots} == {tip}
        dsts = {action_dst(a) for a, _ in roots}
        # (4,3) and (3,4) close on the general; (3,2) is a retreating enemy
        # capture the tip still wins. The retreating neutral (2,3) never
        # reaches the root set — the tip-neighbour guard and the wave clock
        # prune both reject it.
        assert dsts == {(4, 3), (3, 4), (3, 2)}
        assert (2, 3) not in dsts
