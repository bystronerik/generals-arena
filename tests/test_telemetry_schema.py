"""The telemetry registry: typed coercion, loud unknowns, and reducers."""
from __future__ import annotations

import pytest

from arena.instrument.probes import load_probe
from arena.records.fingerprint import BOTS_DIR
from arena.records.telemetry_schema import (
    TELEMETRY_SCHEMA,
    Kind,
    UnknownTelemetryKey,
    coerce,
    metric_keys,
    reduce_series,
    validate_extras,
)


# --- kinds ------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, raw, expected",
    [
        ("enemy_general_sighted", 1, True),
        ("enemy_general_sighted", 0, False),
        ("enemy_general_sighted", "1", True),
        ("strikes", 7, 7),
        ("strikes", "7", 7),
        ("phase", "expand", "expand"),
    ],
)
def test_each_kind_coerces_to_its_declared_type(name, raw, expected):
    got = coerce(name, raw)
    assert got == expected and type(got) is type(expected)


@pytest.mark.parametrize(
    "name, raw",
    [
        ("enemy_general_sighted", 2),  # BOOL01 is 0/1, not "any int"
        ("phase", "two words"),  # TOKEN is \S+
        ("phase", ""),
        ("strikes", "many"),
    ],
)
def test_a_value_that_does_not_fit_its_kind_raises(name, raw):
    with pytest.raises(ValueError):
        coerce(name, raw)


def test_an_undeclared_key_raises_and_names_itself():
    """The old path stored it as a string and moved on. This is the fix."""
    with pytest.raises(UnknownTelemetryKey, match="brand_new_counter"):
        validate_extras({"phase": "expand", "brand_new_counter": 3})


def test_validate_extras_types_a_whole_probe_dict():
    assert validate_extras({"phase": "raid", "strikes": "3", "was_attacked": 1}) == {
        "phase": "raid",
        "strikes": 3,
        "was_attacked": True,
    }


# --- probes and the schema agree --------------------------------------------


def test_every_probe_key_is_declared():
    """A probe is importable without a game, so this costs nothing to check."""
    probed = sorted(p.parent.name for p in BOTS_DIR.glob("*/probe.py"))
    assert probed, "no bot carries a probe; this check would be vacuous"

    for bot_id in probed:
        probe = load_probe(BOTS_DIR / bot_id)
        source = (BOTS_DIR / bot_id / "probe.py").read_text(encoding="utf-8")
        assert probe is not None and "def extras" in source
        for name in _emitted_keys(source):
            assert name in TELEMETRY_SCHEMA, f"{bot_id} emits undeclared {name!r}"


def _emitted_keys(source: str) -> set[str]:
    """Quoted dict keys in a probe body — enough to catch an undeclared one."""
    import re

    return set(re.findall(r'"(\w+)":', source))


def test_no_probe_still_reports_a_first_sighting_turn():
    """It became the `first_turn_true` reducer over the sighting series."""
    assert "first_sighting_turn" not in TELEMETRY_SCHEMA
    for path in BOTS_DIR.glob("*/probe.py"):
        # The agent attribute may still be read; emitting it as a key is what
        # the reducer replaced.
        assert "first_sighting_turn" not in _emitted_keys(path.read_text(encoding="utf-8"))


# --- reducers ---------------------------------------------------------------


def test_int_reducers_on_a_hand_computed_series():
    # guard: final, mean, max
    assert reduce_series("guard", [1, 5, 3]) == {"": 3, "_mean": 3.0, "_max": 5}


def test_argmax_reports_the_first_peak_one_based():
    assert reduce_series("land", [2, 9, 9, 4])["_argmax_turn"] == 2


def test_auc_is_the_discrete_area():
    assert reduce_series("land_margin", [1, -2, 3])["_auc"] == 2


def test_first_turn_true_uses_the_kind_not_python_truthiness():
    """A negative margin is truthy to Python but means "behind"."""
    assert reduce_series("land_margin", [-5, -1, 4, 2])["_first_turn"] == 3
    assert reduce_series("enemy_general_sighted", [0, 0, 1])["_first_turn"] == 3


def test_first_turn_true_is_absent_when_the_predicate_never_fires():
    assert "_first_turn" not in reduce_series("enemy_general_sighted", [0, 0, 0])
    assert "_first_turn" not in reduce_series("land_margin", [-3, -1, 0])


def test_an_empty_series_reduces_to_nothing():
    """Not measured, so absent — never zero."""
    assert reduce_series("guard", []) == {}
    assert metric_keys("land", [], seat="a") == {}


def test_a_single_turn_series_is_well_defined():
    assert reduce_series("guard", [4]) == {"": 4, "_mean": 4.0, "_max": 4}


def test_token_series_keeps_only_the_last_frame():
    assert reduce_series("phase", ["open", "expand", "raid"]) == {"": "raid"}


def test_metric_key_names_carry_the_reducer_and_the_seat():
    assert metric_keys("guard", [1, 5, 3], seat="b") == {
        "guard_b": 3,
        "guard_mean_b": 3.0,
        "guard_max_b": 5,
    }


def test_engine_series_reduce_under_their_own_names():
    assert set(metric_keys("land", [1, 2, 3], seat="a")) == {
        "land_mean_a",
        "land_max_a",
        "land_argmax_turn_a",
    }


def test_schema_entries_are_complete():
    for name, key in TELEMETRY_SCHEMA.items():
        assert key.name == name
        assert isinstance(key.kind, Kind)
        assert key.meaning.strip(), name
        assert key.reducers, name
