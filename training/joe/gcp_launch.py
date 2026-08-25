"""Pure helpers for the GCP (Compute Engine) Joe launcher.

No gcloud, no network, no JAX. The script ``scripts/joe_gcp_train.py``
calls these; the cheap suite tests them. Mirror of ``training/joe/launch.py``
(the vast.ai helpers) for the Google Cloud G4 path.

The trainer side is unchanged: the VM runs the same
``scripts/joe_vast_onstart.sh`` and ``training.joe.vast_boot``, with the R2
bucket as the only durable state. ``CONTAINER_ID`` carries the VM name so
the R2 lease/heartbeat identity works exactly as on vast.
"""

import re

# g4-standard-48 = 1x RTX PRO 6000 Blackwell Server Edition (96 GB GDDR7),
# 48 vCPU, 180 GB RAM. Measured for this silicon class 2026-08-24:
# docs/research/measurements/joe-x16-pilots.md.
DEFAULT_MACHINE = "g4-standard-48"
DEFAULT_BOOT_DISK_GB = 200
# Deep Learning VM image with a Blackwell-capable driver (R580). Family
# checked 2026-08-25; when it rotates, list the current ones
# (docs/engine/joe-gcp-train.md):
#   gcloud compute images list --project deeplearning-platform-release \
#     --filter="family~common" --format="value(family)" | sort -u
DEFAULT_IMAGE_FAMILY = "common-cu129-ubuntu-2404-nvidia-580"
DEFAULT_IMAGE_PROJECT = "deeplearning-platform-release"

# GCE VM names: lowercase RFC1035 label, at most 63 chars.
_VM_NAME_RE = re.compile(r"^[a-z]([-a-z0-9]{0,61}[a-z0-9])?$")
_ENV_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")

REMOTE_ENV_FILE = "/root/joe-env"
REMOTE_ONSTART = "/root/joe-onstart.sh"


def vm_name(run_name):
    """A valid GCE VM name derived from the run name.

    Lowercase, invalid characters become ``-``, runs of ``-`` collapse,
    and the result is prefixed with ``joe-`` unless it already starts
    with a letter-led ``joe`` label. Truncated to 63 characters.
    """
    name = re.sub(r"[^a-z0-9-]+", "-", run_name.lower())
    name = re.sub(r"-{2,}", "-", name).strip("-")
    if not name:
        raise ValueError(f"cannot derive a GCE VM name from {run_name!r}")
    if not name.startswith("joe"):
        name = f"joe-{name}"
    name = name[:63].rstrip("-")
    if not _VM_NAME_RE.match(name):
        raise ValueError(f"cannot derive a GCE VM name from {run_name!r}")
    return name


def render_env_file(env):
    """The ``KEY=value`` lines the bootstrap sources on the VM.

    Refuses keys that are not upper-snake and values that contain a
    newline: the file is sourced by ``sh`` with ``set -a`` and a broken
    line would silently drop or mangle a secret.
    """
    lines = []
    for key in sorted(env):
        value = str(env[key])
        if not _ENV_KEY_RE.match(key):
            raise ValueError(f"bad env var name {key!r}")
        if "\n" in value or "\r" in value:
            raise ValueError(f"env var {key} contains a newline")
        if "'" in value:
            raise ValueError(f"env var {key} contains a single quote")
        lines.append(f"{key}='{value}'")
    return "\n".join(lines) + "\n"


def create_args(vm, *, zone, startup_script, machine=DEFAULT_MACHINE,
                spot=True, image_family=DEFAULT_IMAGE_FAMILY,
                image_project=DEFAULT_IMAGE_PROJECT,
                boot_disk_gb=DEFAULT_BOOT_DISK_GB):
    """Arguments for ``gcloud compute instances create``.

    Spot VMs stop (not delete) on preemption so ``resume`` can restart
    the same VM with its warm disk. GPU VMs cannot live-migrate, so the
    maintenance policy is TERMINATE either way. ``install-nvidia-driver``
    is the Deep Learning VM image's first-boot driver hook.

    ``startup_script`` is the local path of the secret-free boot wrapper
    (``scripts/joe_gcp_startup.sh``); it reruns on every boot, so a spot
    restart recovers without SSH.
    """
    args = [
        "compute", "instances", "create", vm,
        f"--zone={zone}",
        f"--machine-type={machine}",
        f"--image-family={image_family}",
        f"--image-project={image_project}",
        f"--boot-disk-size={int(boot_disk_gb)}GB",
        # G4 machine types refuse pd-* disks; Hyperdisk is required.
        "--boot-disk-type=hyperdisk-balanced",
        "--maintenance-policy=TERMINATE",
        "--metadata=install-nvidia-driver=True",
        f"--metadata-from-file=startup-script={startup_script}",
        f"--labels=joe-vm={vm}",
    ]
    if spot:
        args += [
            "--provisioning-model=SPOT",
            "--instance-termination-action=STOP",
            "--no-restart-on-failure",
        ]
    return args


def deliver_command(remote_path):
    """The remote command that writes one delivered file from SSH stdin.

    ``tee`` from stdin instead of scp: the small write survives a flaky
    tunnel that drops scp's data channel (measured 2026-08-25 over IAP),
    and nothing secret touches the command line or VM metadata. The
    startup script polls for these files and starts the trainer itself.
    """
    return (f"sudo bash -c 'tee {remote_path} >/dev/null "
            f"&& chmod 600 {remote_path}'")


def instance_status(desc):
    """RUNNING / TERMINATED / ... from a ``describe --format=json`` dict."""
    return str((desc or {}).get("status", "")).upper()
