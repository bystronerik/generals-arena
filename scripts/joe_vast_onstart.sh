#!/bin/sh
# Idempotent vast.ai onstart for Joe training.
#
# Parameterized only by RUN_NAME and the injected R2_* account env vars
# (docs/research/strategies/joe-vast-training-plan.md §3). Runs on every
# boot, including a same-machine resume after an outbid.
#
# POSIX preamble: vast.ai may invoke onstart with sh -c. Re-exec bash
# so `set -o pipefail` and process substitution work. The first echo
# must stay outside the heredoc so `vastai logs` shows it even if bash
# is missing.
echo "joe onstart starting run=${RUN_NAME:-UNSET} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ ! -x /bin/bash ]; then
  echo "joe onstart needs /bin/bash" >&2
  exit 1
fi
# Write the bash body to a file so nested python heredocs keep stdin.
# `bash -s` would consume this script from stdin and break those.
JOE_ONSTART_FILE="${TMPDIR:-/tmp}/joe-onstart-$$.sh"
cat > "${JOE_ONSTART_FILE}" <<'JOE_ONSTART'
set -euo pipefail

export RUN_NAME="${RUN_NAME:?RUN_NAME is required}"
export JOE_ROOT="${JOE_ROOT:-/workspace/joe}"
export REPO_DIR="${REPO_DIR:-${JOE_ROOT}/repo}"
export CKPT_DIR="${CKPT_DIR:-${JOE_ROOT}/ckpt}"
export JOE_BOOT_ID="${JOE_BOOT_ID:-$(date +%Y%m%d-%H%M%S)-$$}"

# Visible in `vastai logs` even if later redirect fails.
echo "joe onstart starting run=${RUN_NAME} $(date -u +%Y-%m-%dT%H:%M:%SZ)"

mkdir -p "${JOE_ROOT}/logs" "${CKPT_DIR}" "${REPO_DIR}" "${JOE_ROOT}/jax-cache"
export JOE_TRAIN_LOG="${JOE_ROOT}/logs/train-${JOE_BOOT_ID}.log"
touch "${JOE_TRAIN_LOG}"

# Account env vars are visible to onstart; copy them so later SSH sessions
# see them too (vast.ai docker-environment note).
env | grep _ >> /etc/environment || true

# Keep a copy on stdout (`vastai logs`) and in the boot log file.
exec > >(tee -a "${JOE_TRAIN_LOG}") 2>&1
echo "joe onstart boot=${JOE_BOOT_ID} run=${RUN_NAME} $(date -u +%Y-%m-%dT%H:%M:%SZ)"

ensure_python() {
  venv="${JOE_ROOT}/venv"
  if [ -x "${venv}/bin/python" ] && \
     "${venv}/bin/python" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)'; then
    PYTHON="${venv}/bin/python"
    return
  fi
  echo "creating Python 3.12 venv at ${venv} (jax 0.11.0 needs >=3.12)"
  if ! command -v curl >/dev/null 2>&1; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y curl
  fi
  export PATH="${HOME}/.local/bin:${PATH}"
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
  fi
  uv python install 3.12
  uv venv --python 3.12 --seed "${venv}"
  PYTHON="${venv}/bin/python"
}

if [ -n "${JOE_PYTHON:-}" ]; then
  PYTHON="${JOE_PYTHON}"
else
  ensure_python
fi
export JOE_PYTHON="${PYTHON}"
echo "python=${PYTHON}"
"${PYTHON}" - <<'PY'
import sys
print("python_version", sys.version.split()[0], flush=True)
if sys.version_info < (3, 12):
    sys.exit("Joe needs Python >= 3.12 (jax==0.11.0); got " + sys.version)
PY

need_mod() {
  local mod="$1"
  "${PYTHON}" - "$mod" <<'PY'
import importlib, sys
importlib.import_module(sys.argv[1])
PY
}

need_train_stack() {
  "${PYTHON}" - <<'PY'
import sys
try:
    import boto3  # noqa: F401
    import jax
    import numpy
except Exception:
    sys.exit(1)
if jax.__version__ != "0.11.0":
    sys.exit(1)
if numpy.__version__ != "2.4.6":
    sys.exit(1)
sys.exit(0)
PY
}

put_boot() {
  local phase="$1"
  "${PYTHON}" - "$phase" <<'PY'
import json, os, sys, time
phase = sys.argv[1]
missing = [k for k in ("R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID",
                       "R2_SECRET_ACCESS_KEY") if not os.environ.get(k)]
if missing:
    sys.exit("missing R2 env vars: " + ", ".join(missing))
import boto3
bucket = os.environ.get("R2_BUCKET", "joe-training")
run = os.environ["RUN_NAME"]
client = boto3.client(
    "s3",
    endpoint_url=os.environ["R2_ENDPOINT_URL"],
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    region_name="auto",
)
body = {
    "schema": 1,
    "phase": phase,
    "time": time.time(),
    "boot_id": os.environ.get("JOE_BOOT_ID", ""),
    "python": sys.executable,
    "python_version": sys.version.split()[0],
}
client.put_object(
    Bucket=bucket,
    Key=f"joe/{run}/logs/boot.json",
    Body=json.dumps(body, indent=2).encode(),
)
print(f"boot.json phase={phase}", flush=True)
PY
}

if ! need_mod boto3; then
  echo "installing boto3"
  "${PYTHON}" -m pip install --upgrade pip
  "${PYTHON}" -m pip install boto3
fi
put_boot installing_pip_stack

if ! need_train_stack; then
  echo "installing pinned pip stack"
  "${PYTHON}" -m pip install --upgrade pip
  "${PYTHON}" -m pip install \
    "numpy==2.4.6" \
    "jax[cuda12]==0.11.0" \
    equinox \
    optax \
    pyyaml \
    boto3
else
  echo "pinned pip stack already present"
fi

put_boot fetching_code
echo "fetching code tarball from R2"
"${PYTHON}" - <<'PY'
import hashlib, json, os, tarfile, sys
from pathlib import Path

missing = [k for k in ("R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID",
                       "R2_SECRET_ACCESS_KEY") if not os.environ.get(k)]
if missing:
    sys.exit("missing R2 env vars: " + ", ".join(missing))

import boto3

bucket = os.environ.get("R2_BUCKET", "joe-training")
run = os.environ["RUN_NAME"]
repo = Path(os.environ["REPO_DIR"])
root = Path(os.environ["JOE_ROOT"])
client = boto3.client(
    "s3",
    endpoint_url=os.environ["R2_ENDPOINT_URL"],
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    region_name="auto",
)
launch_key = f"joe/{run}/launch.json"
try:
    body = client.get_object(Bucket=bucket, Key=launch_key)["Body"].read()
except Exception as e:
    sys.exit(f"failed to read {launch_key}: {e}")
launch = json.loads(body)
code_key = launch["code_key"]
want_sha = launch["code_sha256"]
marker = repo / ".joe-code-sha"
if marker.exists() and marker.read_text().strip() == want_sha:
    print(f"code already unpacked ({want_sha[:12]})")
    sys.exit(0)

print(f"downloading s3://{bucket}/{code_key}")
data = client.get_object(Bucket=bucket, Key=code_key)["Body"].read()
got = hashlib.sha256(data).hexdigest()
if got != want_sha:
    sys.exit(f"code checksum mismatch: {got} != {want_sha}")
tar_path = root / "code.tar.gz"
tar_path.write_bytes(data)
repo.mkdir(parents=True, exist_ok=True)
with tarfile.open(tar_path, "r:gz") as tf:
    dest = os.path.abspath(repo)
    for member in tf.getmembers():
        target = os.path.abspath(os.path.join(dest, member.name))
        if target != dest and not target.startswith(dest + os.sep):
            sys.exit(f"unsafe tar member {member.name!r}")
    try:
        tf.extractall(repo, filter="data")
    except TypeError:
        tf.extractall(repo)
marker.write_text(want_sha)
print(f"unpacked {code_key} ({len(data)} bytes)")
PY

export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_DIR}"
echo "installing competition-module (editable, no deps)"
"${PYTHON}" -m pip install -e "${REPO_DIR}/competition-module" --no-deps

put_boot starting_train
echo "starting training.joe.vast_boot"
exec "${PYTHON}" -m training.joe.vast_boot
JOE_ONSTART
exec /bin/bash "${JOE_ONSTART_FILE}"
