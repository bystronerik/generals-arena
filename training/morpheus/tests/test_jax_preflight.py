"""Part 00 Modal JAX preflight — local gate (no Modal / A100 required).

Runs under `-m morpheus`. Proves fixtures, competition compile, scan
throughput measurement, and report decision logic on the host device.
"""
from __future__ import annotations

import json
from pathlib import Path

import jax
import pytest

from training.morpheus.jax_preflight.fixtures import (
    all_parity_fixtures,
    castle_build_fixture,
    deathtouch_chase_defense_fixture,
    deathtouch_fixture,
)
from training.morpheus.jax_preflight.measure import run_preflight_on_device
from training.morpheus.jax_preflight.parity import (
    compare_snapshots,
    snapshots_match,
    state_info_snapshot,
)
from training.morpheus.jax_preflight.report import (
    build_report,
    decide_pass,
    write_report,
)
from training.morpheus.jax_preflight.transition import (
    compile_scan_transition,
    make_action_sequence,
    make_competition_env,
    make_pool_and_states,
    step_one,
)

pytestmark = pytest.mark.morpheus


@pytest.fixture(scope="module")
def competition_env_and_pool():
    env = make_competition_env(pool_size=16)
    pool, _ = make_pool_and_states(env, seed=0, num_envs=1)
    return env, pool


def test_competition_modifiers_pinned(competition_env_and_pool):
    env, _ = competition_env_and_pool
    assert env.mode == "competition"
    assert env.build_castles is True
    assert env.deathtouch_turn == 800
    assert env.pad_to == 21
    assert (env.min_grid_size, env.max_grid_size) == (18, 21)


def test_fixtures_are_padded_to_21():
    for fixture in all_parity_fixtures():
        assert fixture.state.armies.shape == (21, 21)
        assert fixture.actions.shape == (2, 5)


def test_castle_build_fixture(competition_env_and_pool):
    env, pool = competition_env_and_pool
    fixture = castle_build_fixture()
    last_state, info = step_one(env, pool, fixture.state, fixture.actions)
    assert bool(last_state.castles[0, 3])
    assert int(info.winner) == -1


def test_deathtouch_fixture(competition_env_and_pool):
    env, pool = competition_env_and_pool
    fixture = deathtouch_fixture()
    last_state, info = step_one(env, pool, fixture.state, fixture.actions)
    assert int(info.winner) == 0
    assert bool(info.is_done)
    assert int(last_state.winner) == 0


def test_deathtouch_chase_defense_fixture(competition_env_and_pool):
    env, pool = competition_env_and_pool
    fixture = deathtouch_chase_defense_fixture()
    _last_state, info = step_one(env, pool, fixture.state, fixture.actions)
    assert int(info.winner) == -1
    assert not bool(info.is_done)


def test_snapshot_parity_identical():
    env = make_competition_env(pool_size=16)
    pool, _ = make_pool_and_states(env, seed=1, num_envs=1)
    fixture = castle_build_fixture()
    last_state, info = step_one(env, pool, fixture.state, fixture.actions)
    snap = state_info_snapshot(last_state, info)
    assert snapshots_match(snap, snap)
    cmp = compare_snapshots(snap, snap)
    assert cmp["match"] is True
    assert all(cmp["fields"].values())


def test_vmap_scan_compiles_and_runs():
    env = make_competition_env(pool_size=16)
    pool, states = make_pool_and_states(env, seed=2, num_envs=4)
    actions = make_action_sequence(seed=3, num_envs=4, num_steps=8)
    scan_fn = compile_scan_transition(env, pool)
    final, infos = scan_fn(states, actions)
    jax.block_until_ready(final.armies)
    assert final.armies.shape[0] == 4
    assert infos.army.shape[0] == 8  # scan length


def test_run_preflight_on_device_local_smoke():
    result = run_preflight_on_device(
        role="cpu",
        seed=0,
        num_envs=4,
        scan_steps=4,
        pool_size=16,
        warm_reps=1,
    )
    assert result["compiled_ok"] is True
    assert result["modifiers"]["mode"] == "competition"
    assert result["warm_steps_per_s"] > 0
    names = {f["name"] for f in result["parity_fixtures"]}
    assert names == {
        "castle_build",
        "deathtouch",
        "deathtouch_chase_defense",
        "pass_noop",
    }


def test_decide_pass_requires_a100_and_parity(tmp_path: Path):
    base = run_preflight_on_device(
        role="cpu",
        seed=0,
        num_envs=2,
        scan_steps=2,
        pool_size=16,
        warm_reps=1,
    )
    cpu = dict(base)
    gpu_fail = dict(base)
    gpu_fail["role"] = "gpu"
    # Same snapshots → parity ok, but not an A100.
    decision = decide_pass(cpu, gpu_fail)
    assert decision["verdict"] == "no"
    assert decision["checks"]["a100_40gb"] is False

    gpu_ok = dict(base)
    gpu_ok["role"] = "gpu"
    gpu_ok["device"] = {
        "platform": "gpu",
        "device_str": "cuda:0",
        "device_kind": "NVIDIA A100-SXM4-40GB",
    }
    gpu_ok["nvidia"] = {
        "gpu_name": "NVIDIA A100-SXM4-40GB",
        "memory_total": "40960 MiB",
        "driver_version": "550.54.15",
        "cuda_version": "12.4",
    }
    decision_ok = decide_pass(cpu, gpu_ok)
    assert decision_ok["checks"]["a100_40gb"] is True
    assert decision_ok["checks"]["cpu_gpu_parity"] is True
    assert decision_ok["verdict"] == "yes"

    report = build_report(cpu, gpu_ok, a100_hours=0.01, wall_s=36.0)
    json_path = tmp_path / "morpheus-jax-preflight.json"
    md_path = tmp_path / "morpheus-jax-preflight.md"
    write_report(report, json_path=json_path, md_path=md_path)
    loaded = json.loads(json_path.read_text())
    assert loaded["decision"]["verdict"] == "yes"
    assert "Verdict: **yes**" in md_path.read_text()


def test_decide_pass_fails_on_parity_mismatch():
    cpu = run_preflight_on_device(
        role="cpu", seed=0, num_envs=2, scan_steps=2, pool_size=16, warm_reps=1
    )
    gpu = dict(cpu)
    gpu["role"] = "gpu"
    gpu["device"] = {
        "platform": "gpu",
        "device_str": "cuda:0",
        "device_kind": "NVIDIA A100-SXM4-40GB",
    }
    gpu["nvidia"] = {
        "gpu_name": "NVIDIA A100-SXM4-40GB",
        "memory_total": "40960 MiB",
    }
    # Corrupt one fixture snapshot.
    gpu["parity_fixtures"] = [
        dict(f) for f in gpu["parity_fixtures"]
    ]
    bad = dict(gpu["parity_fixtures"][0])
    snap = dict(bad["snapshot"])
    snap["winner"] = 99
    bad["snapshot"] = snap
    gpu["parity_fixtures"][0] = bad
    decision = decide_pass(cpu, gpu)
    assert decision["verdict"] == "no"
    assert decision["checks"]["cpu_gpu_parity"] is False
