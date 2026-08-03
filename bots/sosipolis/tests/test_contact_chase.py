"""ContactMCTS live-army chase and axis preference.

`_live_army_target` and `_prefer_axis_macro` are called directly; the one
root-move scene uses a single-row corridor so the tip has exactly one legal
step and the chase prior is readable without a UCT tree.

Axis geometry for the `_prefer_axis_macro` scenes: home ``(0, 0)`` and first
contact ``(0, 5)`` give ``t = c / 5`` and lateral ``= r``. First contact at
turn 40 is at or below ``CONTACT_AXIS_EARLY_TURN``, so axis confidence is
1.0 and a cell is on axis when ``c >= 3`` and ``r <= 5``.
"""
from __future__ import annotations

from _imports import sosipolis_imports
from boards import (
    T_FOG,
    T_GENERAL,
    T_MOUNTAIN,
    action_dst,
    action_src,
    corridor,
    make_obs,
    plain,
    probe_macro,
    seeded_contact_mcts,
)


ON_AXIS = (0, 8)
OFF_AXIS = (9, 1)
OFF_AXIS_2 = (10, 2)


def _contact_state(
    H: int,
    W: int,
    *,
    home,
    first_contact,
    first_contact_turn: int = 40,
    candidates=(),
    clock: str = "wave",
    mountains=(),
):
    """GameState with memory seeded by hand; never calls state.update()."""
    from params import PARAMS
    from state import GameState

    st = GameState(H, W, PARAMS)
    mem = st.memory
    mem.own_general = home
    mem.first_contact = first_contact
    mem.first_contact_turn = first_contact_turn
    mem.candidates = set(candidates)
    mem._seeded = True
    if first_contact is not None:
        mem.enemy_seen.add(first_contact)
        mem.primary_path.add(first_contact)
        mem.known_owner[first_contact[0]][first_contact[1]] = 2
    for r, c in mountains:
        mem.known_type[r][c] = T_MOUNTAIN
    st.clock_phase = clock
    return st


# ---------------------------------------------------------------------------
# _live_army_target
# ---------------------------------------------------------------------------


def _chase_board(*, fog_cells=(), enemies=()):
    H = W = 9
    types, owner, army = plain(H, W)
    for r, c in fog_cells:
        types[r][c] = T_FOG
    for (r, c), a in enemies:
        owner[r][c] = 2
        army[r][c] = a
    return H, W, types, owner, army


def test_live_army_target_follows_the_source_ray_not_the_far_corner():
    """Home (0,0), enemy (4,0): the ray runs straight down column 0.

    (7,0) sits on it, 7 from home; (4,6) is 10 from home but 6 cells off the
    line. Distance from home alone picked (4,6) — a corner the enemy never
    came from.
    """
    with sosipolis_imports():
        H, W, types, owner, army = _chase_board(
            fog_cells=[(7, 0), (4, 6)], enemies=[((4, 0), 3)]
        )
        obs = make_obs(types, owner, army, turn=200)
        state = _contact_state(H, W, home=(0, 0), first_contact=(4, 0))
        mcts = seeded_contact_mcts()

        target = mcts._live_army_target(obs, state)
        assert target == (7, 0)
        assert obs.type_grid[target[0]][target[1]] == T_FOG


def test_live_army_target_breaks_the_equidistant_arc_by_direction():
    """The arc behind the front is equidistant from home; the ray decides.

    Seed 1 turn 67 in miniature: every candidate is the same distance from
    home and the same distance behind the enemy, so the old key fell through
    to the coordinate tuple and always took the highest row.
    """
    with sosipolis_imports():
        H, W, types, owner, army = _chase_board(
            fog_cells=[(8, 2), (6, 4)], enemies=[((4, 2), 3)]
        )
        obs = make_obs(types, owner, army, turn=200)
        state = _contact_state(H, W, home=(2, 0), first_contact=(4, 2))
        mcts = seeded_contact_mcts()

        # Both are 8 from home and 4 behind the front, so the old key tied and
        # took the higher row. (6,4) continues the home→enemy line.
        assert {abs(r - 2) + abs(c - 0) for r, c in [(8, 2), (6, 4)]} == {8}
        assert mcts._live_army_target(obs, state) == (6, 4)


def test_live_army_target_restricts_fog_to_belief_candidates():
    """A non-empty candidate set filters the fog pool before the max."""
    with sosipolis_imports():
        H, W, types, owner, army = _chase_board(
            fog_cells=[(2, 4), (4, 7)], enemies=[((4, 4), 3)]
        )
        obs = make_obs(types, owner, army, turn=200)
        state = _contact_state(
            H, W, home=(8, 0), first_contact=(4, 4), candidates={(2, 4)}
        )
        mcts = seeded_contact_mcts()

        assert mcts._live_army_target(obs, state) == (2, 4)


def test_live_army_target_falls_back_to_farthest_enemy_tile():
    """With no fog in reach the tip walks onto the enemy tile far from home."""
    with sosipolis_imports():
        H, W, types, owner, army = _chase_board(
            enemies=[((0, 0), 1), ((4, 4), 9)]
        )
        obs = make_obs(types, owner, army, turn=200)
        state = _contact_state(H, W, home=(8, 8), first_contact=(4, 4))
        mcts = seeded_contact_mcts()

        target = mcts._live_army_target(obs, state)
        # (0,0) is 16 from home and outranks the bigger stack at (4,4) (8).
        assert target == (0, 0)
        assert obs.owner_grid[target[0]][target[1]] == 2


def test_live_army_target_is_none_without_visible_enemy():
    with sosipolis_imports():
        H, W, types, owner, army = _chase_board(fog_cells=[(2, 4)])
        obs = make_obs(types, owner, army, turn=200)
        state = _contact_state(H, W, home=(8, 0), first_contact=None)
        mcts = seeded_contact_mcts()

        assert mcts._live_army_target(obs, state) is None


# ---------------------------------------------------------------------------
# _prefer_axis_macro
# ---------------------------------------------------------------------------


def _axis_state():
    return _contact_state(12, 12, home=(0, 0), first_contact=(0, 5))


def test_axis_scene_confidence_and_membership():
    """Early first contact gives confidence 1.0; pin the on/off axis cells."""
    with sosipolis_imports():
        from params import PARAMS

        state = _axis_state()
        mcts = seeded_contact_mcts()

        assert state.memory.first_contact_turn <= PARAMS.CONTACT_AXIS_EARLY_TURN
        assert mcts._axis_confidence(state.memory) == 1.0
        assert mcts._on_hunt_axis(state, ON_AXIS) is True
        assert mcts._on_hunt_axis(state, OFF_AXIS) is False
        assert mcts._on_hunt_axis(state, OFF_AXIS_2) is False


def test_prefer_axis_macro_chase_beats_higher_scoring_cluster():
    """A visible army outranks abstract fog even off axis and far behind."""
    with sosipolis_imports():
        state = _axis_state()
        mcts = seeded_contact_mcts()
        cluster = probe_macro(kind="cluster", waypoint=ON_AXIS, score=0.9)
        chase = probe_macro(kind="chase", waypoint=OFF_AXIS, score=0.1)

        assert mcts._prefer_axis_macro(state, [cluster, chase]) is chase


def test_prefer_axis_macro_prefers_on_axis_chase():
    """Inside the chase pool the on-axis waypoint still wins."""
    with sosipolis_imports():
        state = _axis_state()
        mcts = seeded_contact_mcts()
        off = probe_macro(kind="chase", waypoint=OFF_AXIS, score=0.9)
        on = probe_macro(kind="chase", waypoint=ON_AXIS, score=0.1)

        assert mcts._prefer_axis_macro(state, [off, on]) is on


def test_prefer_axis_macro_on_axis_path_beats_off_axis_cluster():
    """With no chase, the on-axis pool is taken before any score compare."""
    with sosipolis_imports():
        state = _axis_state()
        mcts = seeded_contact_mcts()
        path = probe_macro(kind="path", waypoint=ON_AXIS, score=0.2)
        cluster = probe_macro(kind="cluster", waypoint=OFF_AXIS, score=0.9)

        assert mcts._prefer_axis_macro(state, [path, cluster]) is path


def test_prefer_axis_macro_falls_back_to_full_pool_when_none_on_axis():
    """No chase, no on-axis macro, no path: the best score wins."""
    with sosipolis_imports():
        state = _axis_state()
        mcts = seeded_contact_mcts()
        cluster = probe_macro(kind="cluster", waypoint=OFF_AXIS, score=0.9)
        frontier = probe_macro(kind="frontier", waypoint=OFF_AXIS_2, score=0.4)

        assert mcts._prefer_axis_macro(state, [cluster, frontier]) is cluster


# ---------------------------------------------------------------------------
# Chase prior inside _root_moves
# ---------------------------------------------------------------------------


CORRIDOR_ROW = 2
CORRIDOR_TIP = (CORRIDOR_ROW, 3)
CORRIDOR_TIP_ARMY = 20
CORRIDOR_FOG = (CORRIDOR_ROW, 8)


def _corridor_scene(*, enemy_visible: bool):
    """One passable row: home, tip, an enemy tile, and fog behind the front."""
    H, W = 5, 9
    row = CORRIDOR_ROW
    types, owner, army = corridor(H, W, row=row)
    types[row][0] = T_GENERAL
    owner[row][0] = 1
    army[row][0] = 1
    owner[row][3] = 1
    army[row][3] = CORRIDOR_TIP_ARMY
    if enemy_visible:
        owner[row][6] = 2
        army[row][6] = 3
    types[row][8] = T_FOG

    mountains = [
        (r, c) for r in range(H) for c in range(W) if types[r][c] == T_MOUNTAIN
    ]
    obs = make_obs(types, owner, army, turn=200)
    state = _contact_state(
        H,
        W,
        home=(row, 0),
        first_contact=(row, 6),
        clock="wave",
        mountains=mountains,
    )
    state.strike_tip = CORRIDOR_TIP
    return obs, state


def test_wave_root_moves_march_the_tip_at_the_chase_target():
    """Corridor scene: every surviving root is the tip step toward the fog."""
    with sosipolis_imports():
        from params import PARAMS

        obs, state = _corridor_scene(enemy_visible=True)
        mcts = seeded_contact_mcts()

        chase = mcts._live_army_target(obs, state)
        assert chase == CORRIDOR_FOG

        roots = mcts._root_moves(
            obs,
            state,
            hunt=chase,
            tip=CORRIDOR_TIP,
            assault_ready=True,
            clock="wave",
        )
        assert roots
        assert {action_src(a) for a, _ in roots} == {CORRIDOR_TIP}
        assert {action_dst(a) for a, _ in roots} == {(CORRIDOR_ROW, 4)}
        assert max(p for _, p in roots) >= PARAMS.CONTACT_CHASE_STEP_BONUS


def test_chase_step_outranks_the_plain_hunt_step():
    """Same march, same target: a visible enemy raises the tip-step prior."""
    with sosipolis_imports():
        chase_obs, chase_state = _corridor_scene(enemy_visible=True)
        quiet_obs, quiet_state = _corridor_scene(enemy_visible=False)
        mcts = seeded_contact_mcts()

        assert mcts._live_army_target(chase_obs, chase_state) == CORRIDOR_FOG
        assert mcts._live_army_target(quiet_obs, quiet_state) is None

        def top_prior(obs, state):
            roots = mcts._root_moves(
                obs,
                state,
                hunt=CORRIDOR_FOG,
                tip=CORRIDOR_TIP,
                assault_ready=True,
                clock="wave",
            )
            assert {action_dst(a) for a, _ in roots} == {(CORRIDOR_ROW, 4)}
            return max(p for _, p in roots)

        assert top_prior(chase_obs, chase_state) > top_prior(quiet_obs, quiet_state)
