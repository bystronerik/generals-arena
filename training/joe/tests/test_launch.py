"""Launcher helpers: offer query, smoke overrides, run-name guard."""

from __future__ import annotations

from pathlib import Path

import pytest

from training.joe.launch import (
    H100_GPU_NAMES,
    SMOKE_OVERRIDES,
    apply_smoke,
    code_object_name,
    destroy_in_progress,
    env_var_names,
    find_vastai_bin,
    instance_label,
    offer_query,
    offer_unavailable,
    resolve_gpu_names,
    validate_run_name,
    vastai_cli_error,
)


def test_offer_query_single_and_h100_list():
    q = offer_query(["RTX_4090"], 1)
    assert q.startswith("gpu_name=RTX_4090")
    assert "num_gpus=1" in q
    assert "verified=true" in q
    assert "rentable=true" in q
    qh = offer_query(H100_GPU_NAMES, 2)
    assert qh.startswith("gpu_name in [H100_SXM, H100_NVL, H100_PCIE]")
    assert "num_gpus=2" in qh


def test_resolve_gpu_aliases():
    assert resolve_gpu_names("h100") == list(H100_GPU_NAMES)
    assert resolve_gpu_names("4090")[0] == "RTX_4090"
    assert resolve_gpu_names("H100_SXM") == ["H100_SXM"]
    assert resolve_gpu_names("RTX_4090,RTX_4090_D") == \
        ["RTX_4090", "RTX_4090_D"]


def test_smoke_overrides_keep_run_name_and_shrink_iters():
    out = apply_smoke({"run_name": "joe-S", "num_iters": 50_000,
                       "num_envs": 2048})
    assert out["run_name"] == "joe-S"
    assert out["num_iters"] == SMOKE_OVERRIDES["num_iters"]
    assert out["num_envs"] == SMOKE_OVERRIDES["num_envs"]
    # minibatch must divide 2 * num_envs * num_steps
    n = 2 * out["num_envs"] * out["num_steps"]
    assert n % out["minibatch_size"] == 0


def test_run_name_and_code_object_name():
    assert validate_run_name("joe-S-vast-smoke-20260813") == \
        "joe-S-vast-smoke-20260813"
    with pytest.raises(ValueError):
        validate_run_name("joe S")
    with pytest.raises(ValueError):
        validate_run_name("")
    assert code_object_name("abc", False, "deadbeef") == "abc.tar.gz"
    assert code_object_name("abc", True, "deadbeefcafebabe").startswith(
        "abc-dirty-deadbeefcafe")
    assert instance_label("joe-x") == "joe-x"
    assert instance_label("run-1") == "joe-run-1"


def test_find_vastai_bin_order(tmp_path):
    path_bin = tmp_path / "on-path" / "vastai"
    path_bin.parent.mkdir()
    path_bin.write_text("#!/bin/sh\n")
    path_bin.chmod(0o755)
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("")
    sibling = venv_python.parent / "vastai"
    sibling.write_text("#!/bin/sh\n")
    sibling.chmod(0o755)
    repo = tmp_path / "repo"
    repo_bin = repo / ".venv" / "bin" / "vastai"
    repo_bin.parent.mkdir(parents=True)
    repo_bin.write_text("#!/bin/sh\n")
    repo_bin.chmod(0o755)

    assert Path(find_vastai_bin(
        path_which=str(path_bin), executable=str(venv_python),
        repo=repo)).resolve() == path_bin.resolve()
    assert Path(find_vastai_bin(
        executable=str(venv_python), repo=repo)).resolve() == sibling.resolve()
    assert Path(find_vastai_bin(repo=repo)).resolve() == repo_bin.resolve()
    assert find_vastai_bin() is None


def test_env_var_names_flat_and_wrapped():
    assert env_var_names({}) == set()
    assert env_var_names(None) == set()
    assert env_var_names({"R2_ENDPOINT_URL": "*****",
                          "R2_ACCESS_KEY_ID": "*****"}) == {
        "R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID"}
    assert env_var_names({"success": True, "secrets": {
        "R2_ENDPOINT_URL": "*****"}}) == {"R2_ENDPOINT_URL"}


def test_vastai_cli_error_exit_zero_http_error():
    assert vastai_cli_error("{}", "", 0) is None
    err = vastai_cli_error(
        "",
        '{"error": true, "status_code": 401, '
        '"msg": "Authorization Error. Your key lacks the api.secrets '
        'route access which is required for this action."}',
        0)
    assert err is not None
    assert "401" in err
    assert "api.secrets" in err
    assert "manage-keys" in err
    line = vastai_cli_error(
        "", "Failed with error 401: Authorization Error. "
        "Your key lacks the api.secrets route access which is "
        "required for this action.\n  Sent key from /x\n", 0)
    assert line.startswith("Failed with error 401")
    assert "Sent key" not in line.split("\n")[0]
    assert vastai_cli_error(
        "Failed to create environment variable: existing_key", "", 0) == \
        "Failed to create environment variable: existing_key"
    assert vastai_cli_error(
        "Environment variable created successfully.", "", 0) is None


def test_destroy_in_progress_states():
    assert destroy_in_progress({"cur_state": "destroying"})
    assert destroy_in_progress({"next_state": "destroyed"})
    assert destroy_in_progress({"intended_status": "deleted"})
    assert destroy_in_progress({"cur_state": "running",
                                "next_state": "Destroyed"})
    assert not destroy_in_progress({"cur_state": "running",
                                    "next_state": "running",
                                    "intended_status": "running"})
    assert not destroy_in_progress({"cur_state": None})
    assert not destroy_in_progress({})


def test_offer_unavailable_detects_410():
    assert offer_unavailable(
        "vastai error 410: error 410/3907: no_such_ask  "
        "Instance type 39184694 is no longer available.")
    assert offer_unavailable("no_such_ask")
    assert not offer_unavailable("vastai error 401: api.secrets")
