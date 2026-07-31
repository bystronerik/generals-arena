"""Offline tests for proteus classification and switch hysteresis.

Translation of the generals-bot adaptive test suite onto the grid-native
OpponentModel (fed with UnifiedObservation literals instead of GameStates).
"""
from __future__ import annotations

from _common.oppmodel import OpponentModel
from proteus.classifier import Classification, classify
from proteus.switcher import Switcher
from test_common_tactics import _grid, make_obs


def feed(model, turns, opp_land_fn):
    """Feed a 20x1 strip: enemy at the far end (distance 19, never 'near')."""
    W = 20
    types = [[1] * W]
    types[0][0] = 4
    owner = [[0] * W]
    owner[0][0] = 1
    owner[0][W - 1] = 2
    army = [[0] * W]
    army[0][0] = 5
    army[0][W - 1] = 10
    for t in turns:
        n = opp_land_fn(t)
        obs = make_obs(
            types, owner, army, turn=t,
            my_land=5, my_army=10, opp_land=n, opp_army=2 * n,
        )
        model.update(obs)


def test_classifier_unknown_early():
    m = OpponentModel()
    assert classify(m, 10).label == "unknown"


def test_classifier_boomer_on_fast_growth():
    m = OpponentModel()
    # Opponent grows a cell every 2 turns — very fast expansion.
    feed(m, range(1, 200), lambda t: 1 + t // 2)
    c = classify(m, 199)
    assert c.label == "boomer"
    assert c.confidence > 0.4


def test_classifier_citier_on_castles_seen():
    m = OpponentModel()
    feed(m, range(1, 100), lambda t: 5)
    m.enemy_castles_seen = 2
    c = classify(m, 99)
    assert c.label == "citier"


def test_switcher_needs_streak_for_offensive_switches():
    # turtler -> counter "boom"; from default "blitz" this is a normal switch
    sw = Switcher(default="blitz", streak_needed=3, cooldown=0)
    c = Classification("turtler", 0.9)
    assert sw.update(c, 100) == "blitz"
    assert sw.update(c, 101) == "blitz"
    assert sw.update(c, 102) == "boom"  # third consecutive proposal switches
    assert sw.label == "turtler"


def test_switcher_low_confidence_never_switches():
    sw = Switcher(default="boom", streak_needed=2, cooldown=0)
    c = Classification("rusher", 0.2)
    for t in range(100, 120):
        assert sw.update(c, t) == "boom"


def test_switch_into_defense_is_fast(monkeypatch):
    # The defense fast-path, exercised via a counter that maps to the
    # defensive strategy (no production label does today).
    import proteus.switcher as sw_mod

    monkeypatch.setitem(sw_mod.COUNTER, "rusher", "aegis")
    sw = Switcher(default="boom", streak_needed=12, defense_streak=3)
    rush = Classification("rusher", 0.9)
    sw.update(rush, 100)
    sw.update(rush, 101)
    assert sw.update(rush, 102) == "aegis"  # defense_streak, not streak_needed


def test_leaving_defense_is_slow_between_rush_waves(monkeypatch):
    import proteus.switcher as sw_mod

    monkeypatch.setitem(sw_mod.COUNTER, "rusher", "aegis")
    sw = Switcher(default="boom", streak_needed=4, defense_streak=2,
                  leave_defense_streak=30, cooldown=0)
    rush = Classification("rusher", 0.9)
    boom = Classification("boomer", 0.9)
    sw.update(rush, 100)
    assert sw.update(rush, 101) == "aegis"
    # Classifier flips to boomer (rebuild phase) -> counter blitz, but
    # leaving defense needs a 30-turn streak, so we stay put for a while.
    for t in range(102, 131):
        assert sw.update(boom, t) == "aegis"
    assert sw.update(boom, 131) == "blitz"  # 30th consecutive proposal


def test_switcher_interleaved_proposals_reset_streak():
    sw = Switcher(default="blitz", streak_needed=3, cooldown=0)
    turtler = Classification("turtler", 0.9)
    unknown = Classification("unknown", 0.0)
    sw.update(turtler, 1)
    sw.update(turtler, 2)
    sw.update(unknown, 3)  # resets streak
    sw.update(turtler, 4)
    assert sw.update(turtler, 5) == "blitz"
    assert sw.update(turtler, 6) == "boom"


def test_proteus_agent_shares_one_model():
    from arena.bot_api import load_strategy_class

    agent_cls = load_strategy_class("proteus")
    agent = agent_cls(player_id=0, H=5, W=5)
    types = _grid(5, 5, fill=1)
    types[0][0] = 4
    owner = _grid(5, 5)
    owner[0][0] = 1
    army = _grid(5, 5)
    army[0][0] = 5
    obs = make_obs(types, owner, army, turn=10)
    action = agent.act(obs)
    assert len(action) == 5
    # All four cores hold the same model instance, updated exactly once.
    models = {id(core.memory.opp if hasattr(core.memory, "opp") else core.memory.model)
              for core in agent.cores.values()}
    assert models == {id(agent.model)}
    assert len(agent.model.turns) == 1
    assert agent.active_strategy == "blitz"
