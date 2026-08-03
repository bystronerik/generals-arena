"""The kill that is already on must not wait behind the sight floor.

`TIP_AT_SIGHT_FLOOR` is an operating level measured off Kubic, not a rule
about whether a stack can take a general. Traced on seed 6: the tip stood two
cells from a general holding two army, with seven, and spent six turns feeding
while the general grew to seven — then the tip moved to a bigger stack
fourteen cells away and the general reached twenty.
"""
from __future__ import annotations

from _imports import sosipolis_imports
from boards import T_GENERAL, corridor, make_obs, plain


def _scene(H, W, *, tip, tip_army, gen, gen_army, corridor_row=None, between=()):
    if corridor_row is None:
        types, owner, army = plain(H, W)
    else:
        types, owner, army = corridor(H, W, row=corridor_row)
    owner[tip[0]][tip[1]] = 1
    army[tip[0]][tip[1]] = tip_army
    types[gen[0]][gen[1]] = T_GENERAL
    owner[gen[0]][gen[1]] = 2
    army[gen[0]][gen[1]] = gen_army
    for (r, c), (o, a) in between:
        owner[r][c] = o
        army[r][c] = a
    return make_obs(types, owner, army, turn=200)


def test_two_step_kill_is_affordable():
    """Seed 6 in miniature: 7 army, general holds 2, two steps of own land."""
    with sosipolis_imports():
        from components.tip import can_finish_now, finish_cost_exact
        from params import PARAMS

        obs = _scene(
            5, 9, tip=(2, 1), tip_army=7, gen=(2, 3), gen_army=2,
            corridor_row=2, between=[((2, 2), (1, 1))],
        )
        # base = FINISH_MARGIN + their 2; the own-land step in between is free.
        assert finish_cost_exact(obs, (2, 1), (2, 3), PARAMS) == PARAMS.FINISH_MARGIN + 2
        assert can_finish_now(obs, (2, 1), (2, 3), PARAMS) is True


def test_exact_cost_is_cheaper_than_the_march_planner():
    """The planner's per-hop buffer is what priced this kill out of reach."""
    with sosipolis_imports():
        from components.tip import finish_cost_exact, path_finish_need
        from params import PARAMS

        obs = _scene(
            5, 9, tip=(2, 1), tip_army=7, gen=(2, 3), gen_army=2,
            corridor_row=2, between=[((2, 2), (1, 1))],
        )
        assert finish_cost_exact(obs, (2, 1), (2, 3), PARAMS) < path_finish_need(
            obs, (2, 1), (2, 3), PARAMS
        )


def test_a_stack_that_loses_the_fight_is_not_ready():
    with sosipolis_imports():
        from components.tip import can_finish_now
        from params import PARAMS

        obs = _scene(
            5, 9, tip=(2, 1), tip_army=4, gen=(2, 3), gen_army=9,
            corridor_row=2, between=[((2, 2), (1, 1))],
        )
        assert can_finish_now(obs, (2, 1), (2, 3), PARAMS) is False


def test_enemy_tiles_on_the_path_are_charged():
    with sosipolis_imports():
        from components.tip import finish_cost_exact
        from params import PARAMS

        clear = _scene(
            5, 9, tip=(2, 1), tip_army=20, gen=(2, 3), gen_army=2,
            corridor_row=2, between=[((2, 2), (1, 1))],
        )
        held = _scene(
            5, 9, tip=(2, 1), tip_army=20, gen=(2, 3), gen_army=2,
            corridor_row=2, between=[((2, 2), (2, 5))],
        )
        assert finish_cost_exact(held, (2, 1), (2, 3), PARAMS) > finish_cost_exact(
            clear, (2, 1), (2, 3), PARAMS
        )


def test_unreachable_goal_is_never_finishable():
    with sosipolis_imports():
        from components.tip import can_finish_now
        from params import PARAMS

        # Tip walled off on its own corridor row; general sits on another.
        obs = _scene(5, 9, tip=(2, 1), tip_army=99, gen=(0, 8), gen_army=1,
                     corridor_row=2)
        assert can_finish_now(obs, (2, 1), (0, 8), PARAMS) is False


def test_no_tip_and_no_goal_are_handled():
    with sosipolis_imports():
        from components.tip import can_finish_now
        from params import PARAMS

        obs = _scene(5, 9, tip=(2, 1), tip_army=7, gen=(2, 3), gen_army=2,
                     corridor_row=2)
        assert can_finish_now(obs, None, (2, 3), PARAMS) is False
        assert can_finish_now(obs, (2, 1), None, PARAMS) is False
