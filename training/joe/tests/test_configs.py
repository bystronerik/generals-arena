"""Frozen configs vs GeneralsEnv(mode="competition") — drift protection.

Plan section 4: curriculum stages construct the env from explicit kwargs, so
the final stage must be bit-identical to the mode preset. This test pins the
config <-> preset agreement and fails if the preset gains, loses, or changes
a key without the configs following.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.joe

import yaml

from generals.core.env import _MODE_PRESETS

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

# Preset keys that carry no information under build-castles rules: neutral
# castles are generated then stripped, so their count and value never matter.
IRRELEVANT_PRESET_KEYS = {"num_castles_range", "castle_val_range"}

# Preset keys the YAML deliberately does not carry. perfect_info is pinned
# here instead: training must keep fog of war.
KEYS_PINNED_IN_TEST = {"perfect_info": False}

EXPECTED_PRESET_KEYS = {
    "min_grid_size", "max_grid_size", "pad_to", "truncation", "perfect_info",
    "mountain_density_range", "num_castles_range", "min_generals_distance",
    "castle_val_range", "build_castles", "deathtouch_turn",
}


def _load(tier):
    with open(CONFIG_DIR / f"{tier}.yaml") as f:
        return yaml.safe_load(f)


def test_preset_key_set_is_known():
    preset = _MODE_PRESETS["competition"]
    assert set(preset) == EXPECTED_PRESET_KEYS, (
        "GeneralsEnv competition preset changed shape; re-check "
        "training/joe/configs/*.yaml against it (plan section 4)"
    )
    for key, val in KEYS_PINNED_IN_TEST.items():
        assert preset[key] == val


@pytest.mark.parametrize("tier", ["S", "M", "M7", "M7F4"])
def test_config_env_matches_preset(tier):
    cfg = _load(tier)
    preset = _MODE_PRESETS["competition"]
    assert cfg["min_grid_size"] == preset["min_grid_size"]
    assert cfg["max_grid_size"] == preset["max_grid_size"]
    assert cfg["pad_to"] == preset["pad_to"]
    assert cfg["truncation"] == preset["truncation"]
    assert (cfg["mountain_density_min"], cfg["mountain_density_max"]) == \
        preset["mountain_density_range"]
    assert cfg["build_castles"] == preset["build_castles"]
    assert cfg["deathtouch_turn"] == preset["deathtouch_turn"]


@pytest.mark.parametrize("tier", ["S", "M", "M7", "M7F4"])
def test_final_curriculum_stage_equals_preset(tier):
    cfg = _load(tier)
    final = cfg["curriculum"][-1]
    preset = _MODE_PRESETS["competition"]
    assert final["min_generals_distance"] == preset["min_generals_distance"]
    # The preset leaves the max unset; the final stage must too.
    assert final["max_generals_distance"] is None
    assert "max_generals_distance" not in preset


@pytest.mark.parametrize("tier", ["S", "M", "M7", "M7F4"])
def test_network_matches_plan(tier):
    cfg = _load(tier)
    assert cfg["pad_to"] % cfg["patch_size"] == 0
    assert cfg["value_loss"] == "ce"
    assert (cfg["v_min"], cfg["v_max"]) == (-1.0, 1.0)
    assert cfg["hl_sigma"] == 0.04  # NOT the released default 0.75 (plan section 2)
