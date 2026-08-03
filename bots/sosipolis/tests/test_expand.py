"""Compact expansion: land is won by transit, not by search.

The general makes one army every other turn, so the number of cells we can
own by t=50 is capped near 24 and each unit gets about one spare turn of
travel. These pin the rules that keep the frontier inside that budget.
"""
from __future__ import annotations

from _imports import sosipolis_imports
from boards import T_CASTLE, T_MOUNTAIN, action_dst, action_src, make_obs, plain


def _state(H, W, *, home, chain_head=None, seen=()):
    from params import PARAMS
    from state import GameState

    st = GameState(H, W, PARAMS)
    st.memory.own_general = home
    st.memory._seeded = True
    st.chain_head = chain_head
    for cell in seen:
        st.memory.ever_seen[cell[0]][cell[1]] = True
    return st


def _board(H, W, *, own=(), mountains=(), castles=()):
    types, owner, army = plain(H, W)
    for (r, c), a in own:
        owner[r][c] = 1
        army[r][c] = a
    for r, c in mountains:
        types[r][c] = T_MOUNTAIN
    for r, c in castles:
        types[r][c] = T_CASTLE
    return types, owner, army


# ---------------------------------------------------------------------------
# what counts as economy
# ---------------------------------------------------------------------------


def test_neutral_castles_are_not_economy():
    """A neutral castle costs ~35 — never the cheap cell to take."""
    with sosipolis_imports():
        from components.expand import capture_move
        from params import PARAMS

        types, owner, army = _board(
            5, 5, own=[((2, 2), 9)], castles=[(2, 3)]
        )
        obs = make_obs(types, owner, army, turn=20)
        st = _state(5, 5, home=(2, 2))

        move = capture_move(obs, st, PARAMS)
        assert move is not None
        assert action_dst(move) != (2, 3)


def test_a_stack_that_cannot_win_the_cell_does_not_try():
    with sosipolis_imports():
        from components.expand import capture_move
        from params import PARAMS

        types, owner, army = _board(5, 5, own=[((2, 2), 1)])
        obs = make_obs(types, owner, army, turn=20)
        st = _state(5, 5, home=(2, 2))

        assert capture_move(obs, st, PARAMS) is None


# ---------------------------------------------------------------------------
# transit is the cost being minimised
# ---------------------------------------------------------------------------


def test_capture_prefers_the_cell_nearest_the_general():
    """Two takeable cells, one beside home and one at the end of a tendril.

    Hunting the far one is what cost 30 of 47 opening turns: the only stack
    that can take it has to walk the whole corridor first.
    """
    with sosipolis_imports():
        from components.expand import capture_move
        from params import PARAMS

        own = [((4, 0), 9)] + [((4, c), 2) for c in range(1, 5)]
        types, owner, army = _board(6, 7, own=own)
        obs = make_obs(types, owner, army, turn=20)
        st = _state(6, 7, home=(4, 0))

        move = capture_move(obs, st, PARAMS)
        assert move is not None
        # (3,0) is one step from home; (4,5) is five steps out along the arm.
        assert action_dst(move) == (3, 0)


def test_chain_continues_before_anything_nearer_home():
    """A live snake takes a cell a turn and has already paid its transit."""
    with sosipolis_imports():
        from components.expand import capture_move
        from params import PARAMS

        own = [((4, 0), 3)] + [((4, c), 1) for c in range(1, 4)] + [((4, 4), 9)]
        types, owner, army = _board(6, 7, own=own)
        obs = make_obs(types, owner, army, turn=20)
        st = _state(6, 7, home=(4, 0), chain_head=(4, 4))

        move = capture_move(obs, st, PARAMS)
        assert move is not None
        assert action_src(move) == (4, 4)


def test_approach_walks_the_biggest_stack_toward_the_nearest_neutral():
    with sosipolis_imports():
        from components.expand import approach_move
        from params import PARAMS

        # Everything owned except a pocket behind a wall gap on the right.
        own = [((r, c), 1) for r in range(3) for c in range(4)]
        own = [x for x in own if x[0] != (0, 0)] + [((0, 0), 20)]
        types, owner, army = _board(3, 6, own=own)
        obs = make_obs(types, owner, army, turn=20)
        st = _state(3, 6, home=(0, 0))

        move = approach_move(obs, st, PARAMS)
        assert move is not None
        assert action_src(move) == (0, 0)
        # Steps onto our own land, toward the neutral column at c=4.
        assert action_dst(move) in {(0, 1), (1, 0)}


def test_expand_takes_a_cell_before_walking_toward_one():
    with sosipolis_imports():
        from components.expand import expand_move
        from params import PARAMS

        types, owner, army = _board(5, 5, own=[((2, 2), 9)])
        obs = make_obs(types, owner, army, turn=20)
        st = _state(5, 5, home=(2, 2))

        move = expand_move(obs, st, PARAMS)
        assert move is not None
        assert obs.owner_grid[action_dst(move)[0]][action_dst(move)[1]] == 0


def test_exclude_keeps_the_assault_tip_out_of_the_economy():
    with sosipolis_imports():
        from components.expand import expand_move
        from params import PARAMS

        types, owner, army = _board(5, 5, own=[((2, 2), 9), ((0, 0), 4)])
        obs = make_obs(types, owner, army, turn=20)
        st = _state(5, 5, home=(2, 2))

        move = expand_move(obs, st, PARAMS, exclude=(2, 2))
        assert move is not None
        assert action_src(move) != (2, 2)


# ---------------------------------------------------------------------------
# long hauls
# ---------------------------------------------------------------------------


def test_far_haul_capture_only_fires_beyond_the_haul_threshold():
    with sosipolis_imports():
        from components.expand import far_haul_capture
        from params import PARAMS

        # One movable stack, so the only question is how far its haul is.
        types, owner, army = _board(3, 20, own=[((1, 19), 9)])
        obs = make_obs(types, owner, army, turn=100)
        st = _state(3, 20, home=(1, 0))

        # Muster at (1,0): 19 steps of walking, and neutrals right here.
        move = far_haul_capture(obs, st, PARAMS, (1, 0))
        assert move is not None
        assert action_src(move) == (1, 19)

        # Muster next door: the haul is short, so keep gathering instead.
        assert far_haul_capture(obs, st, PARAMS, (1, 18)) is None


def test_far_haul_capture_leaves_the_muster_and_chain_alone():
    with sosipolis_imports():
        from components.expand import far_haul_capture
        from params import PARAMS

        types, owner, army = _board(3, 20, own=[((1, 19), 9)])
        obs = make_obs(types, owner, army, turn=100)
        st = _state(3, 20, home=(1, 0), chain_head=(1, 19))

        assert far_haul_capture(obs, st, PARAMS, (1, 0)) is None
