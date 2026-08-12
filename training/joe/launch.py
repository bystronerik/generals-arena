"""Pure helpers for the vast.ai Joe launcher.

No vastai CLI, no network, no JAX. The script
``scripts/joe_vast_train.py`` calls these; the cheap suite tests them.
The CLI itself is the pip console script (``pip install vastai``), which
lives next to the venv interpreter when PATH does not include ``.venv/bin``.
"""

import json
from pathlib import Path

# Any H100 variant — interchangeable for this run (vast plan §7).
H100_GPU_NAMES = ("H100_SXM", "H100_NVL", "H100_PCIE")
# Phase 2 micro-smoke: 24 GB class. Full M training stays off these cards.
SMOKE_GPU_NAMES = ("RTX_4090", "RTX_4090_D")

GPU_ALIASES = {
    "h100": H100_GPU_NAMES,
    "4090": SMOKE_GPU_NAMES,
    "rtx4090": SMOKE_GPU_NAMES,
    "rtx_4090": SMOKE_GPU_NAMES,
}

# S-config-sized net plus a few iterations, shrunk so a 24 GB card does not
# OOM (num_envs=2048 is the M/S training default and needs ~80 GB).
SMOKE_OVERRIDES = {
    "num_iters": 6,
    "eval_every": 2,
    "eval_games": 64,
    "ckpt_every": 2,
    "save_every": 2,
    "num_envs": 256,
    "num_steps": 64,
    "minibatch_size": 256,
    "pool_size": 2000,
}

DEFAULT_IMAGE = "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime"
SMOKE_DISK_GB = 40
FULL_DISK_GB = 80

RUN_NAME_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                     "0123456789._-")


def resolve_gpu_names(gpu):
    """``gpu`` is an alias (h100, 4090) or a comma-separated vast gpu_name list."""
    gpu = gpu.strip()
    for alias, names in GPU_ALIASES.items():
        if gpu.lower() == alias:
            return list(names)
    names = [n.strip() for n in gpu.split(",") if n.strip()]
    if not names:
        raise ValueError("empty --gpu")
    return names


def offer_query(gpu_names, num_gpus=1):
    """vastai search-offers query. Names use underscore form (RTX_4090)."""
    gpu_names = list(gpu_names)
    if len(gpu_names) == 1:
        gpu = f"gpu_name={gpu_names[0]}"
    else:
        inner = ", ".join(gpu_names)
        gpu = f"gpu_name in [{inner}]"
    return (f"{gpu} num_gpus={num_gpus} verified=true rentable=true")


def offer_unavailable(exc):
    """True when create lost the race: the ask was rented or taken down."""
    text = str(exc).lower()
    return ("no_such_ask" in text or "no longer available" in text
            or "error 410" in text or " 410:" in text or "410/" in text)


def instance_label(run_name):
    """vast.ai instance label used to find the run's machine later."""
    if run_name.startswith("joe-"):
        return run_name[:64]
    return f"joe-{run_name}"[:64]


def find_vastai_bin(path_which=None, executable=None, repo=None):
    """Locate the pip-installed ``vastai`` console script.

    Order: ``PATH`` (``shutil.which``), the directory of the running
    interpreter (venv ``bin/``), then ``<repo>/.venv/bin/vastai``.
    """
    candidates = []
    if path_which:
        candidates.append(path_which)
    exe_dir = Path(executable).resolve().parent if executable else None
    if exe_dir is not None:
        candidates.append(str(exe_dir / "vastai"))
    if repo is not None:
        candidates.append(str(Path(repo) / ".venv" / "bin" / "vastai"))
    seen = set()
    for c in candidates:
        if not c or c in seen:
            continue
        seen.add(c)
        p = Path(c)
        if p.is_file():
            return str(p.resolve())
    return None


SECRETS_HINT = (
    "The vast.ai API key needs the secrets permission (api.secrets). "
    "Create a key with that access at https://console.vast.ai/manage-keys/ "
    "then run: vastai set api-key <key>"
)

_UNWRAP_KEYS = ("secrets", "env_vars", "results", "data")


def env_var_names(raw):
    """Env-var names from ``vastai show env-vars --raw`` (values ignored)."""
    if raw is None:
        return set()
    if isinstance(raw, dict):
        for key in _UNWRAP_KEYS:
            inner = raw.get(key)
            if isinstance(inner, (dict, list)) and inner:
                return env_var_names(inner)
        # A flat {NAME: masked_value} map from the CLI --raw path.
        return {str(k) for k in raw.keys() if k not in ("success", "error")}
    names = set()
    for item in raw:
        if isinstance(item, str):
            names.add(item)
        elif isinstance(item, dict):
            name = item.get("name") or item.get("key") or item.get("var")
            if name:
                names.add(str(name))
    return names


def vastai_cli_error(stdout, stderr, returncode=0):
    """Error text if the CLI failed.

    vastai 1.5.3 exits 0 on HTTPError and writes the failure to stderr
    (JSON when ``--raw``, a ``Failed with error N:`` line otherwise).
    """
    err = (stderr or "").strip()
    out = (stdout or "").strip()
    for blob in (err, out):
        if not blob:
            continue
        payload = None
        try:
            payload = json.loads(blob)
        except json.JSONDecodeError:
            for line in blob.splitlines():
                line = line.strip()
                if line.startswith("{") and '"error"' in line:
                    try:
                        payload = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        pass
        if isinstance(payload, dict) and payload.get("error"):
            msg = (f"vastai error {payload.get('status_code')}: "
                   f"{payload.get('msg')}")
            if "api.secrets" in msg:
                msg = msg + "\n" + SECRETS_HINT
            return msg
    if "Failed with error" in err:
        first = err.split("\n", 1)[0]
        if "api.secrets" in first:
            first = first + "\n" + SECRETS_HINT
        return first
    for blob in (err, out):
        if blob.startswith("Failed to create") or \
                blob.startswith("Failed to update") or \
                blob.startswith("Failed to delete"):
            return blob.split("\n", 1)[0]
    if returncode:
        return err or out or f"vastai failed ({returncode})"
    return None


def validate_run_name(run_name):
    if not run_name or any(c not in RUN_NAME_CHARS for c in run_name):
        raise ValueError(
            f"run_name {run_name!r} must be non-empty [A-Za-z0-9._-]")
    return run_name


def apply_smoke(cfg_dict):
    out = dict(cfg_dict)
    out.update(SMOKE_OVERRIDES)
    return out


def code_object_name(git_sha, git_dirty, tar_sha256):
    """Immutable R2 object basename under ``code/``."""
    if git_dirty:
        return f"{git_sha}-dirty-{tar_sha256[:12]}.tar.gz"
    return f"{git_sha}.tar.gz"
