"""Offline tests for proteus classification, signals and switch hysteresis.

All three units are pure: `HomePressure` folds observations, `classify` is a
function of (model, turn, pressure), and `Switcher` is a function of
classifications. Nothing here starts an engine.
"""
from __future__ import annotations

from _common.oppmodel import OpponentModel
from proteus.classifier import (
    AGGRESSOR,
    CASTLE_STRUCTURES,
    DUEL_ARMY,
    DUEL_DEADLINE,
    ECONOMY,
    EVIDENCE_TURN,
    UNKNOWN,
    Classification,
    classify,
    structure_estimate,
)
from proteus.signals import HomePressure
from proteus.switcher import COUNTER, Switcher
from test_common_tactics import _grid, make_obs


def strip_obs(turn, *, enemy_army=10, enemy_col=19, W=20):
    """A 1xW corridor: our general at col 0, one enemy cell at `enemy_col`.

    Distance from the general is exactly `enemy_col`, so a test picks how
    close the enemy stands by choosing the column.
    """
    types = [[1] * W]
    types[0][0] = 4
    owner = [[0] * W]
    owner[0][0] = 1
    owner[0][enemy_col] = 2
    army = [[0] * W]
    army[0][0] = 5
    army[0][enemy_col] = enemy_army
    return make_obs(types, owner, army, turn=turn)


def feed(model, turns, opp_land_fn):
    """Feed the corridor with a scripted opponent land curve."""
    for t in turns:
        n = opp_land_fn(t)
        obs = strip_obs(t)
        obs.my_land, obs.my_army = 5, 10
        obs.opp_land, obs.opp_army = n, 2 * n
        model.update(obs)


# --------------------------------------------------------------- signals
def test_pressure_ignores_a_far_stack():
    p = HomePressure()
    for t in range(1, 40):
        p.update(strip_obs(t, enemy_army=99, enemy_col=19))
    assert p.max_stack_near == 0
    assert p.duel_turn is None
    assert p.turns_near == 0


def test_pressure_latches_a_near_fist():
    p = HomePressure()
    p.update(strip_obs(10, enemy_army=40, enemy_col=5))
    assert p.max_stack_near == 40
    assert p.duel_turn == 10
    # Latched: it stays after the stack leaves.
    for t in range(11, 30):
        p.update(strip_obs(t, enemy_army=99, enemy_col=19))
    assert p.max_stack_near == 40
    assert p.duel_turn == 10


def test_pressure_is_idempotent_within_a_turn():
    p = HomePressure()
    obs = strip_obs(10, enemy_army=40, enemy_col=5)
    p.update(obs)
    p.update(obs)
    assert p.turns_near == 1


def test_pressure_conjunction_is_what_the_old_rule_missed():
    """A far fist plus a near nibble must not read as a fist at the door.

    This is the exact false positive the shipped `deep_incursion` test had:
    it ANDed a running-min distance with a running-max stack that never
    co-occurred.
    """
    p = HomePressure()
    p.update(strip_obs(10, enemy_army=99, enemy_col=19))  # big, but far
    p.update(strip_obs(11, enemy_army=2, enemy_col=3))    # near, but tiny
    assert p.max_stack_near == 2
    assert p.duel_turn is None


# ------------------------------------------------------------ classifier
def test_unknown_before_evidence_turn():
    m = OpponentModel()
    assert classify(m, EVIDENCE_TURN - 1).label == UNKNOWN


def test_no_fist_reads_as_economy_once_the_window_closes():
    m = OpponentModel()
    feed(m, range(1, DUEL_DEADLINE + 40), lambda t: 1 + t // 2)
    c = classify(m, DUEL_DEADLINE + 20, HomePressure())
    assert c.label == ECONOMY
    assert c.confidence >= 0.6


def test_silence_before_the_deadline_is_not_evidence():
    """"No fist yet" and "no fist coming" are the same observation until
    the window a fist would have arrived in has closed."""
    m = OpponentModel()
    feed(m, range(1, DUEL_DEADLINE), lambda t: 1 + t // 2)
    assert classify(m, DUEL_DEADLINE - 1, HomePressure()).label == UNKNOWN
    assert classify(m, DUEL_DEADLINE, HomePressure()).label == UNKNOWN


def test_economy_confidence_grows_after_the_deadline():
    m = OpponentModel()
    feed(m, range(1, DUEL_DEADLINE + 300), lambda t: 1 + t // 2)
    early = classify(m, DUEL_DEADLINE + 10, HomePressure()).confidence
    late = classify(m, DUEL_DEADLINE + 200, HomePressure()).confidence
    assert early < late == 1.0


def test_a_timely_fist_reads_as_aggressor_before_the_deadline():
    """An attack is actionable the moment it lands; only silence must wait."""
    m = OpponentModel()
    feed(m, range(1, 200), lambda t: 1 + t // 2)
    p = HomePressure()
    p.update(strip_obs(150, enemy_army=DUEL_ARMY, enemy_col=5))
    c = classify(m, 199, p)
    assert c.label == AGGRESSOR


def test_a_bigger_fist_is_more_confident():
    m = OpponentModel()
    feed(m, range(1, 200), lambda t: 1 + t // 2)
    small, big = HomePressure(), HomePressure()
    small.update(strip_obs(150, enemy_army=DUEL_ARMY, enemy_col=5))
    big.update(strip_obs(150, enemy_army=DUEL_ARMY + 40, enemy_col=5))
    assert classify(m, 199, big).confidence > classify(m, 199, small).confidence


def test_a_late_fist_is_a_counterattack_not_a_rush():
    """Past the deadline a fist is aegis countering or late_rush committing.

    Both want boom, and without the cut aegis latches as an aggressor in most
    games (its counter window opens around turn 400).
    """
    m = OpponentModel()
    feed(m, range(1, DUEL_DEADLINE + 200), lambda t: 1 + t // 2)
    p = HomePressure()
    p.update(strip_obs(DUEL_DEADLINE + 1, enemy_army=99, enemy_col=5))
    assert classify(m, DUEL_DEADLINE + 100, p).label == ECONOMY


def test_a_latched_fist_outranks_a_saturated_economy_score():
    """The economy score reaches 1.0 by turn 200; a fist must still win."""
    m = OpponentModel()
    feed(m, range(1, 600), lambda t: 1 + t // 2)
    p = HomePressure()
    p.update(strip_obs(120, enemy_army=DUEL_ARMY, enemy_col=5))
    assert classify(m, 599, p).label == AGGRESSOR


def test_classify_is_pure_in_its_arguments():
    """Same inputs, same answer — no hidden state between calls."""
    m = OpponentModel()
    feed(m, range(1, 200), lambda t: 1 + t // 2)
    p = HomePressure()
    p.update(strip_obs(150, enemy_army=30, enemy_col=5))
    first = classify(m, 199, p)
    assert (first.label, first.confidence) == (
        classify(m, 199, p).label, classify(m, 199, p).confidence)


# ------------------------------------------------- structure estimate
def _army_series(model, structures, turns=120, start=100):
    """Opponent army growing at `structures` per two turns (RULES.md §04)."""
    army = start
    for t in range(1, turns):
        if t % 50 != 0 and t % 2 == 0:
            army += structures
        model.turns.append(t)
        model.opp_army.append(army)
        model.opp_land.append(10)
        model.my_land.append(10)
        model.my_army.append(50)


def test_structure_estimate_recovers_the_count_from_the_aggregate():
    """General alone against general plus three castles, no vision at all."""
    bare, rich = OpponentModel(), OpponentModel()
    _army_series(bare, 1)   # general only
    _army_series(rich, 4)   # general + 3 castles
    assert structure_estimate(bare) == 1
    assert structure_estimate(rich) == 4
    assert structure_estimate(bare) < CASTLE_STRUCTURES <= structure_estimate(rich)


def test_structure_estimate_survives_combat_losses():
    """Combat only ever lowers total army, so a high quantile ignores it."""
    m = OpponentModel()
    army = 100
    for t in range(1, 160):
        if t % 50 != 0 and t % 2 == 0:
            army += 3                      # general + 2 castles
        if t % 7 == 0:
            army -= 25                     # a wave dies every seventh turn
        m.turns.append(t)
        m.opp_army.append(army)
        m.opp_land.append(10)
        m.my_land.append(10)
        m.my_army.append(50)
    assert structure_estimate(m) == 3


def test_structure_estimate_needs_a_window():
    m = OpponentModel()
    _army_series(m, 1, turns=8)
    assert structure_estimate(m) is None


# --------------------------------------------------------------- switcher
def test_every_label_maps_to_a_constructed_core():
    from arena.bot_api import load_strategy_class

    agent = load_strategy_class("proteus")(player_id=0, H=5, W=5)
    for label, strategy in COUNTER.items():
        target = agent.switcher.default if strategy == "default" else strategy
        assert target in agent.cores, f"{label} -> {target} is unreachable"


def test_leaving_the_spine_needs_a_long_streak():
    sw = Switcher(default="blitz", leave_spine_streak=3, cooldown=0)
    c = Classification(ECONOMY, 0.9)
    assert sw.update(c, 100) == "blitz"
    assert sw.update(c, 101) == "blitz"
    assert sw.update(c, 102) == "boom"
    assert sw.label == ECONOMY


def test_returning_to_the_spine_is_fast():
    sw = Switcher(default="blitz", leave_spine_streak=2,
                  return_spine_streak=1, cooldown=0)
    econ, aggr = Classification(ECONOMY, 0.9), Classification(AGGRESSOR, 0.9)
    sw.update(econ, 100)
    assert sw.update(econ, 101) == "boom"
    assert sw.update(aggr, 102) == "blitz"  # one turn, not leave_spine_streak


def test_a_single_aggressor_turn_cancels_a_drift_to_economy():
    sw = Switcher(default="blitz", leave_spine_streak=4, cooldown=0)
    econ, aggr = Classification(ECONOMY, 0.9), Classification(AGGRESSOR, 0.9)
    sw.update(econ, 1)
    sw.update(econ, 2)
    sw.update(econ, 3)
    sw.update(aggr, 4)   # proposal == current -> streak resets
    sw.update(econ, 5)
    sw.update(econ, 6)
    assert sw.update(econ, 7) == "blitz"
    assert sw.update(econ, 8) == "boom"


def test_low_confidence_never_switches():
    sw = Switcher(default="blitz", leave_spine_streak=2, cooldown=0)
    c = Classification(ECONOMY, 0.2)
    for t in range(100, 130):
        assert sw.update(c, t) == "blitz"


def test_cooldown_gates_leaving_but_not_returning():
    sw = Switcher(default="blitz", leave_spine_streak=1,
                  return_spine_streak=1, cooldown=50)
    econ, aggr = Classification(ECONOMY, 0.9), Classification(AGGRESSOR, 0.9)
    assert sw.update(econ, 100) == "boom"
    assert sw.update(aggr, 101) == "blitz"      # return ignores the cooldown
    assert sw.update(econ, 102) == "blitz"      # leaving again is gated
    assert sw.update(econ, 151) == "boom"       # 50 turns after the last switch


def test_unknown_holds_the_spine():
    sw = Switcher(default="blitz")
    for t in range(0, 200):
        assert sw.update(Classification(UNKNOWN, 0.0), t) == "blitz"
    assert sw.history == []


# ------------------------------------------------------------------ agent
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
    # Both cores hold the same model instance, updated exactly once.
    models = {id(core.memory.opp if hasattr(core.memory, "opp") else core.memory.model)
              for core in agent.cores.values()}
    assert models == {id(agent.model)}
    assert len(agent.model.turns) == 1
    assert agent.active_strategy == "blitz"


def test_probe_keys_are_all_declared():
    """A probe key with no schema entry fails the recorded match, not here."""
    from arena.instrument.probes import load_probe, probe_extras
    from arena.records.telemetry_schema import validate_extras
    from arena.bot_api import load_strategy_class
    from arena.paths import REPO_ROOT

    agent = load_strategy_class("proteus")(player_id=0, H=5, W=5)
    types = _grid(5, 5, fill=1)
    types[0][0] = 4
    owner = _grid(5, 5)
    owner[0][0] = 1
    army = _grid(5, 5)
    army[0][0] = 5
    agent.act(make_obs(types, owner, army, turn=10))

    probe = load_probe(REPO_ROOT / "bots" / "proteus")
    assert probe is not None
    validate_extras(probe_extras(probe, agent))  # raises on an undeclared key
