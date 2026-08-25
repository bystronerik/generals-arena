"""GCP launcher helpers: VM names, env file, create args, bootstrap."""

from __future__ import annotations

import pytest

from pathlib import Path

from training.joe.gcp_launch import (
    REMOTE_ENV_FILE,
    REMOTE_ONSTART,
    create_args,
    deliver_command,
    instance_status,
    render_env_file,
    vm_name,
)

STARTUP = Path(__file__).resolve().parents[3] / "scripts" / "joe_gcp_startup.sh"


def test_vm_name_lowercases_and_prefixes():
    assert vm_name("joe-X16-gcp-20260825") == "joe-x16-gcp-20260825"
    assert vm_name("MyRun.v2") == "joe-myrun-v2"


def test_vm_name_collapses_and_truncates():
    name = vm_name("joe--A__" + "x" * 80)
    assert len(name) <= 63
    assert "--" not in name
    assert name.startswith("joe-")


def test_vm_name_rejects_unusable():
    with pytest.raises(ValueError):
        vm_name("...")


def test_render_env_file_sorted_and_quoted():
    text = render_env_file({"B_KEY": "two words", "A_KEY": "v1"})
    assert text == "A_KEY='v1'\nB_KEY='two words'\n"


@pytest.mark.parametrize("value", ["a\nb", "a'b"])
def test_render_env_file_rejects_unsafe_values(value):
    with pytest.raises(ValueError):
        render_env_file({"KEY": value})


def test_render_env_file_rejects_bad_key():
    with pytest.raises(ValueError):
        render_env_file({"lower": "v"})


def test_create_args_spot_stops_not_deletes():
    args = create_args("joe-vm", zone="us-central1-b", startup_script="/s.sh")
    assert "--provisioning-model=SPOT" in args
    assert "--instance-termination-action=STOP" in args
    assert "--maintenance-policy=TERMINATE" in args
    assert "--metadata=install-nvidia-driver=True" in args
    assert "--metadata-from-file=startup-script=/s.sh" in args


def test_create_args_on_demand_has_no_spot_flags():
    args = create_args("joe-vm", zone="us-central1-b", startup_script="/s.sh",
                       spot=False)
    assert not any("SPOT" in a for a in args)
    assert "--maintenance-policy=TERMINATE" in args


def test_deliver_command_writes_root_only():
    cmd = deliver_command(REMOTE_ENV_FILE)
    assert f"tee {REMOTE_ENV_FILE}" in cmd
    assert f"chmod 600 {REMOTE_ENV_FILE}" in cmd


def test_startup_script_waits_kills_and_sources():
    """The metadata startup script (no secrets) polls for the delivered
    files, waits for the driver, kills a stale trainer, and runs onstart.
    No command substitution: a login shell must never pre-expand it."""
    text = STARTUP.read_text()
    assert REMOTE_ENV_FILE in text
    assert REMOTE_ONSTART in text
    assert "pkill -f training.joe.vast_boot" in text
    assert "nvidia-smi" in text
    assert "R2_" not in text
    # Metadata startup scripts have no HOME; onstart's set -u needs one.
    assert "export HOME=/root" in text


def test_instance_status():
    assert instance_status({"status": "terminated"}) == "TERMINATED"
    assert instance_status(None) == ""
