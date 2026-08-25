#!/usr/bin/env python3
"""Local CLI for Joe training on Google Cloud (G4 / RTX PRO 6000).

Subcommands: launch, status, resume, destroy, watch, zones.

Same shape as ``scripts/joe_vast_train.py``: pack the checkout, upload it
to R2, create a VM whose bootstrap converges on ``joe/<run_name>/``. The
VM runs the unchanged ``scripts/joe_vast_onstart.sh`` and
``training.joe.vast_boot``; R2 stays the only durable state, so a run can
move between vast.ai and GCP with a plain resume on the other launcher.

    python scripts/joe_gcp_train.py zones
    python scripts/joe_gcp_train.py launch --tier M7F4 --zone us-central1-b \\
        --run-name joe-... --overrides '{...}'
    python scripts/joe_gcp_train.py status --run-name ...
    python scripts/joe_gcp_train.py resume --run-name ...
    python scripts/joe_gcp_train.py watch --run-name ...
    python scripts/joe_gcp_train.py destroy --run-name ... --purge-r2

Prerequisites: ``gcloud`` installed and authenticated (``gcloud auth
login``), a project selected (``gcloud config set project <id>``), and
GPU quota — spot uses ``PREEMPTIBLE_NVIDIA_RTX_PRO_6000_GPUS``, on-demand
the ``NVIDIA_RTX_PRO_6000`` family quota (docs/engine/joe-gcp-train.md).

R2 secrets travel over SSH into a root-only env file on the VM — never
through VM metadata, the gcloud command line, or this repo. Spot VMs stop
on preemption; ``resume`` restarts the same VM warm, and ``watch`` does
that automatically. Never pipe output through tail/head — redirect to a
file (AGENTS.md).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ONSTART = REPO / "scripts" / "joe_vast_onstart.sh"
STARTUP = REPO / "scripts" / "joe_gcp_startup.sh"
sys.path.insert(0, str(REPO))

from training.joe.gcp_launch import (  # noqa: E402
    DEFAULT_BOOT_DISK_GB,
    DEFAULT_IMAGE_FAMILY,
    DEFAULT_IMAGE_PROJECT,
    DEFAULT_MACHINE,
    REMOTE_ENV_FILE,
    REMOTE_ONSTART,
    create_args,
    deliver_command,
    instance_status,
    render_env_file,
    vm_name,
)
from training.joe.launch import apply_smoke, code_object_name, validate_run_name  # noqa: E402
from training.joe.pack import engine_sha, pack_checkout, repo_git_state  # noqa: E402
from training.joe.store import LAUNCH_SCHEMA, R2Store, load_dotenv  # noqa: E402

REQUIRED_R2 = ("R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")
SSH_ATTEMPTS = 12
SSH_RETRY_S = 20


def gcloud(*args, parse=True, check=True, quiet=True, input_text=None):
    """Run gcloud. Secrets never appear on this command line; delivered
    file contents travel through ``input_text`` (stdin)."""
    cmd = ["gcloud", *[str(a) for a in args]]
    if quiet:
        cmd.append("--quiet")
    if parse:
        cmd.append("--format=json")
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          input=input_text)
    if proc.returncode != 0:
        if check:
            raise SystemExit(
                f"gcloud failed ({proc.returncode}):\n{proc.stderr.strip()}")
        return None
    if not parse:
        return proc.stdout
    out = proc.stdout.strip()
    return json.loads(out) if out else None


def gcloud_ssh(vm, zone, command, check=True, input_text=None):
    return gcloud("compute", "ssh", vm, f"--zone={zone}",
                  f"--command={command}", parse=False, check=check,
                  input_text=input_text)


def gcloud_ssh_retry(vm, zone, command, input_text=None):
    """SSH with retries: first boot reboots for the driver install, SSH
    flaps while the guest agent propagates keys, and the IAP tunnel used
    by no-external-IP VMs drops connections under a preemption."""
    for attempt in range(SSH_ATTEMPTS):
        out = gcloud_ssh(vm, zone, command, check=False,
                         input_text=input_text)
        if out is not None:
            return out
        print(f"ssh not ready (attempt {attempt + 1}/{SSH_ATTEMPTS}); "
              f"retrying in {SSH_RETRY_S}s", flush=True)
        time.sleep(SSH_RETRY_S)
    raise SystemExit(f"could not reach {vm} over SSH")


def _require_r2_env():
    load_dotenv()
    missing = [k for k in REQUIRED_R2 if not os.environ.get(k)]
    if missing:
        raise SystemExit("missing R2 env vars (gitignored .env): "
                         + ", ".join(missing))


def _load_cfg_dict(tier, smoke, run_name, overrides):
    import yaml

    path = REPO / "training" / "joe" / "configs" / f"{tier}.yaml"
    with open(path) as f:
        cfg_dict = yaml.safe_load(f)
    if smoke:
        cfg_dict = apply_smoke(cfg_dict)
    if run_name:
        cfg_dict["run_name"] = run_name
    if overrides:
        cfg_dict.update(json.loads(overrides))
    validate_run_name(cfg_dict["run_name"])
    return cfg_dict


def _upload_code_and_launch(store, cfg_dict, *, tier, zone, machine, spot,
                            image_family, image_project, boot_disk_gb):
    import yaml

    run_name = cfg_dict["run_name"]
    git_sha, dirty = repo_git_state(REPO)
    engine = engine_sha(REPO)
    print(f"packing {run_name} (tier {tier}, engine {engine[:12]}, "
          f"git {git_sha[:12]}{' dirty' if dirty else ''})", flush=True)
    with tempfile.TemporaryDirectory(prefix="joe-gcp-") as td:
        tar_path = Path(td) / "code.tar.gz"
        meta = pack_checkout(REPO, tar_path, git_sha=git_sha)
        code_name = code_object_name(meta["git_sha"], dirty, meta["sha256"])
        code_key = store.key(run_name, "code", code_name)
        print(f"uploading {code_key} ({meta['size']} bytes, "
              f"{meta['n_files']} files)", flush=True)
        ref = store.upload_file(code_key, str(tar_path))

        cfg_path = Path(td) / "config.yaml"
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg_dict, f, default_flow_style=False)
        store.upload_run_file(run_name, "config.yaml", str(cfg_path))

    launch = {
        "schema": LAUNCH_SCHEMA,
        "run_name": run_name,
        "git_sha": meta["git_sha"],
        "git_dirty": dirty,
        "engine_sha": engine,
        "code_key": code_key,
        "code_sha256": ref["sha256"],
        "code_size": ref["size"],
        "tier": tier,
        "provider": "gcp",
        "zone": zone,
        "machine_type": machine,
        "spot": bool(spot),
        "image_family": image_family,
        "image_project": image_project,
        "boot_disk_gb": int(boot_disk_gb),
        "vm_name": vm_name(run_name),
        "created": time.time(),
    }
    store.put_launch(run_name, launch)
    print(f"wrote launch.json for {run_name}", flush=True)
    return launch


def _describe(vm, zone):
    return gcloud("compute", "instances", "describe", vm, f"--zone={zone}",
                  check=False)


def _deliver_files(vm, zone):
    """Deliver the env file and onstart over SSH stdin, one time.

    The env file holds the R2 secrets and lands root-only on the VM (the
    same place vast's onstart copies them: the instance environment).
    ``CONTAINER_ID`` carries the VM name so the R2 lease works unchanged.
    Both files persist on the boot disk; the metadata startup script
    polls for them and starts the trainer, so no further SSH is needed —
    not even after a spot preemption.
    """
    env = {k: os.environ[k] for k in REQUIRED_R2}
    if os.environ.get("R2_BUCKET"):
        env["R2_BUCKET"] = os.environ["R2_BUCKET"]
    env.update({
        "RUN_NAME": os.environ["RUN_NAME"],
        "JOE_ROOT": "/workspace/joe",
        "CONTAINER_ID": vm,
    })
    gcloud_ssh_retry(vm, zone, deliver_command(REMOTE_ENV_FILE),
                     input_text=render_env_file(env))
    gcloud_ssh_retry(vm, zone, deliver_command(REMOTE_ONSTART),
                     input_text=ONSTART.read_text())
    print(f"delivered env + onstart to {vm}; the startup script takes it "
          f"from here (log: /root/joe-startup.log)", flush=True)


def _create_and_bootstrap(store, launch):
    run_name = launch["run_name"]
    vm = launch["vm_name"]
    zone = launch["zone"]
    args = create_args(
        vm, zone=zone, startup_script=str(STARTUP),
        machine=launch["machine_type"], spot=launch["spot"],
        image_family=launch["image_family"],
        image_project=launch["image_project"],
        boot_disk_gb=launch["boot_disk_gb"])
    gcloud(*args)
    store.put_instance(run_name, vm, extra={"zone": zone, "provider": "gcp"})
    os.environ["RUN_NAME"] = run_name
    _deliver_files(vm, zone)


def cmd_launch(args):
    _require_r2_env()
    smoke = bool(args.smoke)
    tier = args.tier or ("S" if smoke else "M")
    stamp = time.strftime("%Y%m%d-%H%M")
    run_name = args.run_name or (
        f"joe-{tier}-gcp-smoke-{stamp}" if smoke else f"joe-{tier}-gcp-{stamp}")
    validate_run_name(run_name)
    cfg_dict = _load_cfg_dict(tier, smoke, run_name, args.overrides)

    store = R2Store.from_env()
    if store.read_launch(run_name) is not None and not args.force:
        raise SystemExit(
            f"launch.json already exists for {run_name}; "
            f"use resume, or destroy --purge-r2, or pass --force")
    launch = _upload_code_and_launch(
        store, cfg_dict, tier=tier, zone=args.zone, machine=args.machine,
        spot=not args.on_demand, image_family=args.image_family,
        image_project=args.image_project, boot_disk_gb=args.boot_disk_gb)
    if args.pack_only:
        print(f"pack-only: code and launch.json are in R2 for {run_name}",
              flush=True)
        return
    _create_and_bootstrap(store, launch)
    print(f"launched {run_name} on VM {launch['vm_name']} ({args.zone})",
          flush=True)


def cmd_status(args):
    _require_r2_env()
    run_name = validate_run_name(args.run_name)
    store = R2Store.from_env()
    launch = store.read_launch(run_name)
    if launch is None:
        raise SystemExit(f"no launch.json for {run_name}")
    vm, zone = launch["vm_name"], launch["zone"]
    desc = _describe(vm, zone)
    print(f"vm {vm} ({zone}): "
          f"{instance_status(desc) if desc else 'NOT FOUND'}", flush=True)
    boot = store.read_boot(run_name)
    if boot:
        age = time.time() - boot["time"]
        print(f"boot.json: phase={boot['phase']} age={age:.0f}s", flush=True)
    hb = store.read_heartbeat(run_name)
    if hb:
        age = time.time() - float(hb["time"])
        print(f"heartbeat: instance={hb.get('instance_id')} age={age:.0f}s",
              flush=True)
    latest = store.resolve_latest(run_name)
    step = latest["state"]["global_step"] if latest else None
    print(f"latest checkpoint step: {step}", flush=True)


def cmd_resume(args):
    _require_r2_env()
    run_name = validate_run_name(args.run_name)
    store = R2Store.from_env()
    launch = store.read_launch(run_name)
    if launch is None:
        raise SystemExit(f"no launch.json for {run_name}; run launch first")
    vm, zone = launch["vm_name"], launch["zone"]
    os.environ["RUN_NAME"] = run_name
    desc = _describe(vm, zone)
    if desc is None:
        print(f"vm {vm} not found; creating a replacement", flush=True)
        _create_and_bootstrap(store, launch)
        return
    status = instance_status(desc)
    if status == "TERMINATED":
        # SSH-free recovery: the delivered files persist on the boot
        # disk and the metadata startup script reruns on boot.
        print(f"vm {vm} is stopped (spot preemption); starting it",
              flush=True)
        gcloud("compute", "instances", "start", vm, f"--zone={zone}")
    elif status == "RUNNING":
        # Hard reboot reruns the startup script, which restarts the
        # trainer from the last R2 checkpoint. Also SSH-free.
        print(f"vm {vm} is running; resetting it to restart the trainer",
              flush=True)
        gcloud("compute", "instances", "reset", vm, f"--zone={zone}")
    else:
        raise SystemExit(f"vm {vm} is {status}; wait or destroy first")
    store.put_instance(run_name, vm, extra={"zone": zone, "provider": "gcp"})
    print(f"resumed {run_name} on VM {vm}", flush=True)


def cmd_watch(args):
    """Restart the trainer whenever the spot VM lands in TERMINATED.

    One line per check; safe under a Monitor. The R2 heartbeat is the
    liveness signal for the trainer itself; this loop only heals the VM.
    """
    _require_r2_env()
    run_name = validate_run_name(args.run_name)
    store = R2Store.from_env()
    launch = store.read_launch(run_name)
    if launch is None:
        raise SystemExit(f"no launch.json for {run_name}")
    vm, zone = launch["vm_name"], launch["zone"]
    while True:
        desc = _describe(vm, zone)
        status = instance_status(desc) if desc else "NOT_FOUND"
        hb = store.read_heartbeat(run_name)
        hb_age = time.time() - float(hb["time"]) if hb else None
        print(f"watch: vm={status} heartbeat_age="
              f"{f'{hb_age:.0f}s' if hb_age is not None else 'none'}",
              flush=True)
        if status == "TERMINATED":
            # SSH-free: files persist on disk, the startup script reruns.
            print("watch: restarting stopped spot VM", flush=True)
            gcloud("compute", "instances", "start", vm, f"--zone={zone}")
        time.sleep(args.interval)


def cmd_destroy(args):
    _require_r2_env()
    run_name = validate_run_name(args.run_name)
    store = R2Store.from_env()
    launch = store.read_launch(run_name)
    if launch is not None:
        vm, zone = launch["vm_name"], launch["zone"]
        if _describe(vm, zone) is not None:
            print(f"deleting VM {vm} ({zone})", flush=True)
            gcloud("compute", "instances", "delete", vm, f"--zone={zone}")
        else:
            print(f"vm {vm} already gone", flush=True)
    if args.purge_r2:
        deleted = store.delete_prefix(run_name)
        print(f"purged {deleted} R2 objects under joe/{run_name}/",
              flush=True)


def cmd_zones(_args):
    """Zones that offer the G4 accelerator, for --zone selection."""
    rows = gcloud("compute", "accelerator-types", "list",
                  "--filter=name~rtx-pro-6000")
    for row in rows or []:
        print(f"{row.get('zone', '?').rsplit('/', 1)[-1]}  "
              f"{row.get('name')}", flush=True)
    if not rows:
        print("no rtx-pro-6000 accelerator types visible; check project "
              "and auth", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    launch = sub.add_parser("launch")
    launch.add_argument("--tier", default="")
    launch.add_argument("--run-name", default="")
    launch.add_argument("--zone", required=True)
    launch.add_argument("--machine", default=DEFAULT_MACHINE)
    launch.add_argument("--on-demand", action="store_true",
                        help="on-demand instead of spot (quota permitting)")
    launch.add_argument("--image-family", default=DEFAULT_IMAGE_FAMILY)
    launch.add_argument("--image-project", default=DEFAULT_IMAGE_PROJECT)
    launch.add_argument("--boot-disk-gb", type=int,
                        default=DEFAULT_BOOT_DISK_GB)
    launch.add_argument("--smoke", action="store_true")
    launch.add_argument("--overrides", default="",
                        help="JSON object of Config field overrides")
    launch.add_argument("--pack-only", action="store_true")
    launch.add_argument("--force", action="store_true")
    launch.set_defaults(fn=cmd_launch)

    status = sub.add_parser("status")
    status.add_argument("--run-name", required=True)
    status.set_defaults(fn=cmd_status)

    resume = sub.add_parser("resume")
    resume.add_argument("--run-name", required=True)
    resume.set_defaults(fn=cmd_resume)

    watch = sub.add_parser("watch")
    watch.add_argument("--run-name", required=True)
    watch.add_argument("--interval", type=int, default=60)
    watch.set_defaults(fn=cmd_watch)

    destroy = sub.add_parser("destroy")
    destroy.add_argument("--run-name", required=True)
    destroy.add_argument("--purge-r2", action="store_true")
    destroy.set_defaults(fn=cmd_destroy)

    zones = sub.add_parser("zones")
    zones.set_defaults(fn=cmd_zones)

    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
