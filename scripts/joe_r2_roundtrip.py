#!/usr/bin/env python3
"""Phase 1 verification: one real laptop round-trip against R2.

Runs the vast plan's Phase 1 check (docs/research/strategies/
joe-vast-training-plan.md) against the live bucket: upload a tiny run dir,
kill the uploader mid-checkpoint on purpose, confirm ``latest.json`` still
resolves to the previous complete set, and resume-load with checksum
verification. Cleans up its own throwaway run prefix afterwards.

Credentials come from ``R2_ENDPOINT_URL`` / ``R2_ACCESS_KEY_ID`` /
``R2_SECRET_ACCESS_KEY`` in the environment, falling back to the gitignored
``.env`` at the repo root. Costs: a few KB of storage, a handful of PUTs.

    .venv/bin/python scripts/joe_r2_roundtrip.py
"""

import os
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from training.joe.state import write_state  # noqa: E402
from training.joe.store import R2Store  # noqa: E402


def load_dotenv():
    """Fill missing R2_* vars from the repo's gitignored .env, if present."""
    path = os.path.join(REPO, ".env")
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip().removeprefix("export ").strip()
            if key.startswith("R2_") and key not in os.environ:
                os.environ[key] = value.strip().strip("'\"")


class KilledMidCheckpoint(Exception):
    pass


class FailingClient:
    """Wraps the real client; dies after ``allow`` PUTs — the injected
    preemption between the checkpoint blobs and the latest.json PUT."""

    def __init__(self, client, allow):
        self._client = client
        self._allow = allow
        self.exceptions = client.exceptions

    def put_object(self, **kwargs):
        if self._allow <= 0:
            raise KilledMidCheckpoint(kwargs["Key"])
        self._allow -= 1
        return self._client.put_object(**kwargs)

    def __getattr__(self, name):
        return getattr(self._client, name)


def make_set(run_dir, run_name, step):
    files = {"full": f"{run_name}_{step}.eqx",
             "ema": f"{run_name}_ema_{step}.eqx"}
    for basename in files.values():
        with open(os.path.join(run_dir, basename), "wb") as f:
            f.write(os.urandom(4096))
    return write_state(run_dir, run_name, step, 0, 0.0, "roundtrip",
                       files=files)


def main():
    load_dotenv()
    store = R2Store.from_env()
    run_name = f"roundtrip-{time.strftime('%Y%m%d-%H%M%S')}"
    print(f"Bucket {store.bucket}, run prefix {store.key(run_name)}")

    try:
        with tempfile.TemporaryDirectory() as run_dir:
            # 1. Upload a complete set
            state1 = make_set(run_dir, run_name, 1)
            store.upload_checkpoint(run_dir, state1)
            assert store.resolve_latest(run_name)["state"]["global_step"] == 1
            print("1. uploaded step-1 set; latest resolves to step 1")

            # 2. Kill mid-checkpoint: blobs land, latest.json never does
            state2 = make_set(run_dir, run_name, 2)
            real_client = store.client
            store.client = FailingClient(real_client, allow=2)
            try:
                store.upload_checkpoint(run_dir, state2)
                raise SystemExit("FAIL: injected kill did not fire")
            except KilledMidCheckpoint as e:
                print(f"2. killed mid-checkpoint (before PUT of {e})")
            finally:
                store.client = real_client
            latest = store.resolve_latest(run_name)
            assert latest["state"]["global_step"] == 1, latest
            print("   latest still resolves to the step-1 set")

            # 3. Resume-load with checksum verification
            with tempfile.TemporaryDirectory() as dest:
                store.download_checkpoint(latest, dest)
                got = sorted(os.listdir(dest))
                print(f"3. resume-load verified checksums: {got}")

            # 4. The retry completes and moves latest forward
            store.upload_checkpoint(run_dir, state2)
            assert store.resolve_latest(run_name)["state"]["global_step"] == 2
            print("4. retried upload; latest now resolves to step 2")

            # 5. Heartbeat
            store.put_heartbeat(run_name, "laptop-roundtrip")
            assert store.read_heartbeat(run_name)["instance_id"] == \
                "laptop-roundtrip"
            print("5. heartbeat round trip ok")
    finally:
        resp = store.client.list_objects_v2(
            Bucket=store.bucket, Prefix=store.key(run_name) + "/")
        keys = [o["Key"] for o in resp.get("Contents", [])]
        for key in keys:
            store.client.delete_object(Bucket=store.bucket, Key=key)
        print(f"Cleaned up {len(keys)} objects")

    print("PASS")


if __name__ == "__main__":
    main()
