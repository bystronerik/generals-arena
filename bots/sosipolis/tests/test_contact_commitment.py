"""ContactMCTS sticky commitment: hold, switch, invalidate, publish.

Every scene calls the commitment helpers directly with a pre-seeded belief
cache, so no UCT tree and no wall-clock budget can change the verdict.

Axis geometry used by the hold/switch scenes: home ``(0, 0)`` and first
contact ``(0, 5)``. The axis vector is ``(0, 5)``, so for a cell ``(r, c)``
the projection is ``t = c / 5`` and the lateral offset is ``r``. Waypoint A
``(0, 8)`` and waypoint B ``(2, 9)`` are both on the hunt axis
(``t >= 0.55``, lateral ``<= 5``) and neither is far off axis
(lateral ``< CONTACT_AXIS_RECOVER_LATERAL``), so the axis-recover branch
never fires and only the hold/switch rule decides.
"""
from __future__ import annotations

import pytest

from _imports import sosipolis_imports
from boards import (
    T_MOUNTAIN,
    make_obs,
    plain,
    probe_macro,
    seeded_contact_mcts,
)


HOME = (0, 0)
FIRST_CONTACT = (0, 5)
WAYPOINT_A = (0, 8)
WAYPOINT_B = (2, 9)
TURN = 200


def _contact_state(
    H: int,
    W: int,
    *,
    home=HOME,
    first_contact=FIRST_CONTACT,
    first_contact_turn: int = 40,
    candidates=(),
    clock: str = "wave",
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
    st.clock_phase = clock
    return st


def _commit_scene(*, clock: str, age: int, b_score: float, last_score: float = 0.4):
    """Committed on A; a better macro B is offered. Returns the new commitment."""
    from components.contact_mcts import ContactCommitment

    H = W = 12
    types, owner, army = plain(H, W)
    obs = make_obs(types, owner, army, turn=TURN)
    state = _contact_state(
        H, W, candidates={WAYPOINT_A, WAYPOINT_B}, clock=clock
    )
    mcts = seeded_contact_mcts(belief={WAYPOINT_A: 0.5, WAYPOINT_B: 0.5})

    macro_a = probe_macro(kind="cluster", waypoint=WAYPOINT_A, score=0.3)
    macro_b = probe_macro(kind="cluster", waypoint=WAYPOINT_B, score=b_score)
    mcts.commitment = ContactCommitment(macro_a, TURN - age, last_score, 0)

    # Tip sits on the axis, away from A, with no visible enemy nearby.
    tip = (0, 6)
    chosen = mcts._commit_macro(obs, state, [macro_a, macro_b], tip)
    return mcts, state, chosen


# ---------------------------------------------------------------------------
# Geometry guards for the hold/switch scenes
# ---------------------------------------------------------------------------


def test_hold_switch_scene_geometry_is_on_axis():
    """Both waypoints stay on axis so only the hold/switch rule can decide."""
    with sosipolis_imports():
        H = W = 12
        state = _contact_state(H, W, candidates={WAYPOINT_A, WAYPOINT_B})
        mcts = seeded_contact_mcts(belief={WAYPOINT_A: 0.5, WAYPOINT_B: 0.5})

        assert mcts._axis_confidence(state.memory) == 1.0
        for waypoint in (WAYPOINT_A, WAYPOINT_B, (0, 6)):
            assert mcts._on_hunt_axis(state, waypoint) is True, waypoint
            assert mcts._far_off_axis(state, waypoint) is False, waypoint


# ---------------------------------------------------------------------------
# _commit_macro: gather hold, min turns, switch threshold
# ---------------------------------------------------------------------------


def test_gather_clock_holds_committed_waypoint():
    """CONTACT_HOLD_GATHER blocks a switch while the next wave rebuilds."""
    with sosipolis_imports():
        _, _, chosen = _commit_scene(clock="gather", age=40, b_score=0.99)
        assert chosen.macro.waypoint == WAYPOINT_A
        assert chosen.switch_count == 0


def test_wave_below_commit_min_turns_holds():
    """A commitment younger than CONTACT_COMMIT_MIN_TURNS cannot be replaced."""
    with sosipolis_imports():
        from params import PARAMS

        age = PARAMS.CONTACT_COMMIT_MIN_TURNS - 1
        _, _, chosen = _commit_scene(clock="wave", age=age, b_score=0.99)
        assert chosen.macro.waypoint == WAYPOINT_A
        assert chosen.switch_count == 0


def test_wave_switches_above_ratio_and_margin():
    """Aged commitment yields when the challenger clears ratio * last + margin."""
    with sosipolis_imports():
        from params import PARAMS

        last_score = 0.4
        threshold = (
            last_score * PARAMS.CONTACT_SWITCH_RATIO + PARAMS.CONTACT_SWITCH_MARGIN
        )
        b_score = threshold + 0.05
        _, _, chosen = _commit_scene(
            clock="wave",
            age=PARAMS.CONTACT_COMMIT_MIN_TURNS,
            b_score=b_score,
            last_score=last_score,
        )
        assert chosen.macro.waypoint == WAYPOINT_B
        assert chosen.committed_turn == TURN
        assert chosen.switch_count == 1


def test_wave_holds_just_below_switch_threshold():
    """One notch under the same threshold keeps the original waypoint."""
    with sosipolis_imports():
        from params import PARAMS

        last_score = 0.4
        threshold = (
            last_score * PARAMS.CONTACT_SWITCH_RATIO + PARAMS.CONTACT_SWITCH_MARGIN
        )
        b_score = threshold - 0.05
        _, _, chosen = _commit_scene(
            clock="wave",
            age=PARAMS.CONTACT_COMMIT_MIN_TURNS,
            b_score=b_score,
            last_score=last_score,
        )
        assert chosen.macro.waypoint == WAYPOINT_A
        assert chosen.switch_count == 0


# ---------------------------------------------------------------------------
# _commitment_invalid truth table
# ---------------------------------------------------------------------------


def _run_commitment_invalid(
    *,
    waypoint,
    tip,
    belief=None,
    candidates=(),
    ever_seen=(),
    mountains=(),
    candidate_cells=None,
) -> bool:
    from components.contact_mcts import ContactCommitment

    H = W = 12
    types, owner, army = plain(H, W)
    state = _contact_state(H, W, candidates=candidates)
    for r, c in mountains:
        types[r][c] = T_MOUNTAIN
        state.memory.known_type[r][c] = T_MOUNTAIN
    for r, c in ever_seen:
        state.memory.ever_seen[r][c] = True
    obs = make_obs(types, owner, army, turn=TURN)

    mcts = seeded_contact_mcts(belief=dict(belief or {}))
    macro = probe_macro(
        kind="cluster",
        waypoint=waypoint,
        candidate_cells=(
            candidate_cells
            if candidate_cells is not None
            else frozenset({waypoint})
        ),
    )
    cur = ContactCommitment(macro, TURN - 10, 0.4, 0)
    return mcts._commitment_invalid(obs, state, cur, tip)


# Expected values read off components/contact_mcts.py::_commitment_invalid.
INVALID_TRUTH_TABLE = [
    # name, kwargs, expected
    (
        "off_board_waypoint",
        # Bounds check fires first.
        dict(waypoint=(0, 99), tip=(0, 6)),
        True,
    ),
    (
        "mountain_waypoint",
        # is_passable_belief False.
        dict(waypoint=(3, 3), tip=(0, 6), mountains=[(3, 3)]),
        True,
    ),
    (
        "tip_on_waypoint_zero_reveal",
        # Explicit early False: _commit_macro owns the tip-arrival case.
        dict(waypoint=(3, 3), tip=(3, 3)),
        False,
    ),
    (
        "spent_seen_no_reveal_not_candidate",
        # Scouted and exhausted.
        dict(waypoint=(3, 3), tip=(0, 6), ever_seen=[(3, 3)]),
        True,
    ),
    (
        "cluster_pruned_but_reveal_positive",
        # Members gone from candidates, waypoint still reveals belief.
        dict(
            waypoint=(3, 3),
            tip=(0, 6),
            belief={(3, 3): 1.0},
            candidate_cells=frozenset({(9, 9)}),
        ),
        False,
    ),
    (
        "cluster_pruned_reveal_positive_tip_adjacent",
        # Same as above with the tip already on the line: reveal still wins,
        # so the two-step proximity rule below must not fire.
        dict(
            waypoint=(3, 3),
            tip=(3, 4),
            belief={(3, 3): 1.0},
            candidate_cells=frozenset({(9, 9)}),
        ),
        False,
    ),
    (
        "cluster_pruned_waypoint_still_candidate",
        # No reveal left, but the waypoint itself is still a general candidate.
        dict(
            waypoint=(3, 3),
            tip=(3, 4),
            candidates={(3, 3)},
            candidate_cells=frozenset({(9, 9)}),
        ),
        False,
    ),
    (
        "cluster_pruned_tip_adjacent",
        # Members gone, no reveal, tip already within two steps.
        dict(
            waypoint=(3, 3),
            tip=(3, 4),
            candidate_cells=frozenset({(9, 9)}),
        ),
        True,
    ),
    (
        "cluster_pruned_tip_far_unseen",
        # Members gone, no reveal, tip elsewhere and waypoint never seen:
        # keep the direction until the tip arrives.
        dict(
            waypoint=(3, 3),
            tip=(0, 0),
            candidate_cells=frozenset({(9, 9)}),
        ),
        False,
    ),
    (
        "healthy_commitment",
        dict(
            waypoint=(3, 3),
            tip=(0, 6),
            belief={(3, 3): 1.0},
            candidates={(3, 3)},
            candidate_cells=frozenset({(3, 3)}),
        ),
        False,
    ),
]


@pytest.mark.parametrize(
    "name, kwargs, expected",
    INVALID_TRUTH_TABLE,
    ids=[row[0] for row in INVALID_TRUTH_TABLE],
)
def test_commitment_invalid_truth_table(name, kwargs, expected):
    with sosipolis_imports():
        assert _run_commitment_invalid(**kwargs) is expected, name


# ---------------------------------------------------------------------------
# prepare_contact
# ---------------------------------------------------------------------------


def test_prepare_contact_clears_commitment_on_general_latch():
    """A known enemy general ends the hunt: StrikeMCTS owns the turn."""
    with sosipolis_imports():
        from components.clock import Deadline
        from components.contact_mcts import ContactCommitment

        H = W = 12
        types, owner, army = plain(H, W)
        obs = make_obs(types, owner, army, turn=TURN)
        state = _contact_state(H, W, candidates={WAYPOINT_A})
        state.memory.enemy_general = (1, 1)

        mcts = seeded_contact_mcts(belief={WAYPOINT_A: 1.0})
        macro = probe_macro(kind="cluster", waypoint=WAYPOINT_A)
        mcts.commitment = ContactCommitment(macro, TURN - 10, 0.4, 0)
        state.contact_commitment = mcts.commitment

        assert mcts.prepare_contact(obs, state, Deadline(10_000)) is None
        assert mcts.commitment is None
        assert state.contact_commitment is None


def test_prepare_contact_returns_none_without_enemy_land():
    """No first contact and no remembered enemy land: nothing to probe."""
    with sosipolis_imports():
        from components.clock import Deadline

        H = W = 12
        types, owner, army = plain(H, W)
        obs = make_obs(types, owner, army, turn=TURN)
        state = _contact_state(H, W, first_contact=None)

        mcts = seeded_contact_mcts()
        assert state.enemy_land_known() is False
        assert mcts.prepare_contact(obs, state, Deadline(10_000)) is None
        assert mcts.commitment is None
        assert state.contact_commitment is None


def test_prepare_contact_republishes_objective_when_deadline_expired():
    """An expired budget must not drop the commitment already in hand."""
    with sosipolis_imports():
        from components.clock import Deadline
        from components.contact_mcts import ContactCommitment

        H = W = 12
        types, owner, army = plain(H, W)
        obs = make_obs(types, owner, army, turn=TURN)
        state = _contact_state(H, W, candidates={WAYPOINT_A})
        state.objective = None
        state.contact_commitment = None

        mcts = seeded_contact_mcts(belief={WAYPOINT_A: 1.0})
        macro = probe_macro(kind="cluster", waypoint=WAYPOINT_A)
        cur = ContactCommitment(macro, TURN - 10, 0.4, 0)
        mcts.commitment = cur

        out = mcts.prepare_contact(obs, state, Deadline(0))
        assert out is cur
        assert state.contact_commitment is cur
        assert state.objective == WAYPOINT_A


# ---------------------------------------------------------------------------
# Candidate tour: nearest reachable candidate, re-picked as they are cleared
# ---------------------------------------------------------------------------


def test_tour_targets_the_nearest_reachable_candidate():
    """The tour never commits to a cell it cannot reach.

    The belief-argmax waypoint could, and nothing released it: a commitment is
    dropped only on arrival or on vision, both of which need us to get there.
    Seed 1 marched at the corner (0,19) for 853 turns without coming closer
    than 7, while the true general sat in the candidate set the whole game.
    """
    with sosipolis_imports():
        from components.clock import Deadline
        from params import PARAMS

        assert PARAMS.CONTACT_TOUR

        H = W = 12
        near, far = (0, 7), (11, 11)
        types, owner, army = plain(H, W)
        owner[0][0] = 1
        army[0][0] = 30  # the tip, at home
        obs = make_obs(types, owner, army, turn=TURN)
        state = _contact_state(H, W, candidates={near, far})
        state.strike_tip = HOME

        mcts = seeded_contact_mcts(belief={near: 0.1, far: 0.9})
        commit = mcts.prepare_contact(obs, state, Deadline(10_000))

        # `far` carries nine times the belief; the tour still clears `near`
        # first, because an arrival is what eliminates candidates.
        assert commit is not None
        assert commit.macro.kind == "tour", commit.macro.kind
        assert commit.macro.waypoint == near, commit.macro.waypoint
        assert state.objective == near


def test_tour_repicks_once_a_candidate_is_cleared():
    """Clearing the near candidate moves the tour on to the next one."""
    with sosipolis_imports():
        from components.clock import Deadline

        H = W = 12
        near, far = (0, 7), (11, 11)
        types, owner, army = plain(H, W)
        owner[0][0] = 1
        army[0][0] = 30
        obs = make_obs(types, owner, army, turn=TURN)
        state = _contact_state(H, W, candidates={near, far})
        state.strike_tip = HOME

        mcts = seeded_contact_mcts(belief={near: 0.1, far: 0.9})
        assert mcts.prepare_contact(obs, state, Deadline(10_000)).macro.waypoint == near

        state.memory.candidates = {far}
        mcts._cache.belief = {far: 0.9}
        commit = mcts.prepare_contact(obs, state, Deadline(10_000))

        assert commit.macro.waypoint == far, commit.macro.waypoint
