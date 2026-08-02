"""Sosipolis decision logic on hand-built board fixtures.

No live match. Fixtures live beside this file (`boards.py`) and encode the
turn-572 adjacent-kill miss plus MapMemory latch regressions.
"""
from __future__ import annotations

import pytest

from _imports import agent_on, sosipolis_imports
from boards import (
    FIXTURES,
    action_dst,
    action_src,
    adjacent_kill,
    adjacent_kill_reported,
    contact_seed_only,
    contact_then_off_axis_general,
    same_turn_on_axis_general,
)


# ---------------------------------------------------------------------------
# MapMemory latch
# ---------------------------------------------------------------------------


def test_off_axis_enemy_general_latches_from_fixture():
    """Contact east, then sight general north — must latch (t-shadow fix)."""
    with sosipolis_imports():
        from components.map_memory import MapMemory

        prime = contact_seed_only()
        sight = contact_then_off_axis_general()
        mem = MapMemory(prime.H, prime.W, min_general_distance=17)
        mem.update(prime.obs())
        assert mem.first_contact == (7, 3)
        assert mem.own_general == (7, 0)
        assert mem.enemy_general is None

        mem.update(sight.obs())
        assert mem.enemy_general == (1, 0)


def test_same_turn_on_axis_general_latches():
    """Projection must not block latch on the first sight turn."""
    with sosipolis_imports():
        from components.map_memory import MapMemory

        fixture = same_turn_on_axis_general()
        mem = MapMemory(fixture.H, fixture.W, min_general_distance=17)
        mem.own_general = fixture.own_general
        mem.first_contact = fixture.first_contact
        mem.first_contact_turn = fixture.first_contact_turn
        mem.primary_path.add(fixture.first_contact)
        mem.enemy_seen.add(fixture.first_contact)
        mem._seeded = True

        mem.update(fixture.obs())
        assert mem.enemy_general == (5, 4)


# ---------------------------------------------------------------------------
# Kill shot / act priority
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label, stack, gen_army, turn, expect_kill",
    [
        ("reported_12v8", 12, 8, 572, True),
        ("exact_10v8", 10, 8, 572, True),
        ("refuse_9v8", 9, 8, 572, False),
        ("deathtouch_2v99", 2, 99, 800, True),
        ("pre_dt_2v99", 2, 99, 799, False),
    ],
)
def test_kill_shot_thresholds(label, stack, gen_army, turn, expect_kill):
    with sosipolis_imports():
        fixture = adjacent_kill(
            stack_army=stack,
            general_army=gen_army,
            turn=turn,
            label=label,
        )
        agent = agent_on(fixture)
        obs = fixture.obs()
        agent.state.update(obs)
        assert agent.state.memory.enemy_general == (2, 2)
        kill = agent._kill_shot(obs)
        if expect_kill:
            assert kill is not None, label
            assert action_src(kill) == (2, 1)
            assert action_dst(kill) == (2, 2)
        else:
            assert kill is None, label


def test_act_takes_reported_adjacent_kill():
    """Full priority ladder: act() must capture before gather/MCTS/castle."""
    with sosipolis_imports():
        fixture = adjacent_kill_reported()
        agent = agent_on(fixture)
        obs = fixture.obs()
        move = agent.act(obs)
        assert agent.branch == "kill", agent.branch
        assert action_src(move) == (2, 1)
        assert action_dst(move) == (2, 2)


def test_act_kills_on_gather_clock_residue():
    """Turn 572 is gather (res 22); lethal capture must still win the turn."""
    with sosipolis_imports():
        from components.conveyor import mod50_phase
        from params import PARAMS

        fixture = adjacent_kill_reported()
        assert fixture.turn % 50 == 22
        assert mod50_phase(fixture.turn, PARAMS) == "gather"
        agent = agent_on(fixture)
        move = agent.act(fixture.obs())
        assert agent.branch == "kill"
        assert action_dst(move) == (2, 2)


def test_act_kills_visible_general_without_prior_latch():
    """Belt-and-suspenders: visible type-4 enemy even if memory was empty."""
    with sosipolis_imports():
        fixture = adjacent_kill_reported()
        agent = agent_on(fixture)
        agent.state.memory.enemy_general = None
        agent.state.phase = "contact"
        move = agent.act(fixture.obs())
        assert agent.state.memory.enemy_general == (2, 2)
        assert agent.branch == "kill"
        assert action_dst(move) == (2, 2)


def test_fixture_catalog_covers_named_scenes():
    """Guard: named scenes stay registered for other tests / docs."""
    for name in (
        "off_axis_general_sight",
        "kill_12v8_t572",
        "same_turn_on_axis_general",
        "refuse_9v8",
        "deathtouch_2v99",
    ):
        assert name in FIXTURES, name
