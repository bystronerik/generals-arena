"""Instance-side boot for a vast.ai Joe run.

Onstart unpacks the code tarball, then runs ``python -m training.joe.vast_boot``.
This module: checks the heartbeat lease, restores ``latest.json`` if present,
starts the heartbeat + log uploaders, and calls ``training.joe.main.run`` with
the R2 ``CheckpointUploader`` as ``on_checkpoint``. No vastai CLI.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
import traceback

from training.joe.store import (
    CheckpointUploader,
    R2Store,
    check_lease,
)


HEARTBEAT_S = 60
LOG_UPLOAD_S = 60


def _instance_id():
    return (os.environ.get("CONTAINER_ID")
            or os.environ.get("VAST_CONTAINERLABEL")
            or socket.gethostname())


class _IntervalUploader(threading.Thread):
    """Daemon thread: call ``fn`` every ``interval_s`` until ``stop()``.

    The event is ``_stop_event``, not ``_stop``: ``threading.Thread._stop``
    is a real method the stdlib calls from inside ``join()``, so shadowing
    it with an Event makes every ``stop()`` raise ``TypeError`` and hides
    whatever error was being cleaned up after.
    """

    def __init__(self, fn, interval_s, name):
        super().__init__(name=name, daemon=True)
        self._fn = fn
        self._interval_s = interval_s
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            try:
                self._fn()
            except Exception:
                print(f"{self.name} failed (will retry)", flush=True)
                traceback.print_exc()
            if self._stop_event.wait(self._interval_s):
                break

    def stop(self):
        self._stop_event.set()
        self.join(timeout=self._interval_s + 5)
        try:
            self._fn()
        except Exception:
            traceback.print_exc()


def _upload_log(store, run_name, log_path, boot_id):
    if not log_path or not os.path.exists(log_path):
        return
    rel = f"logs/train-{boot_id}.log"
    store.upload_run_file(run_name, rel, log_path)


def apply_env_overrides(cfg, overrides_json):
    """Apply the ``JOE_CONFIG_OVERRIDES`` JSON to a loaded Config.

    The R2 ``config.yaml`` stays authoritative and untouched; this hook
    exists for per-platform keys whose meaning depends on the machine —
    ``num_envs`` and ``minibatch_size`` are PER DEVICE, so the same run
    needs half the values on a 2-GPU host to keep the recipe identical.
    Never override schedule or network keys mid-run through this hook.
    """
    if not overrides_json:
        return cfg
    import json

    from training.joe.config import Config

    data = json.loads(overrides_json)
    print(f"Applying JOE_CONFIG_OVERRIDES: "
          f"{ {k: data[k] for k in sorted(data)} }", flush=True)
    return Config.from_dict({**cfg.to_dict(), **data},
                            source="JOE_CONFIG_OVERRIDES")


def fetch_config(store, run_name, ckpt_dir):
    """Always fetch the run's ``config.yaml`` from R2; return its path.

    The launch upload is authoritative. The local file must never win: an
    adopted instance's ckpt_dir can hold another run's ``config.yaml``
    (``main.run`` writes one there), and a stale config builds the wrong
    network template. Measured 2026-08-18: the joe-M7 step-0 seed crashed
    at deserialization against a depth-5 template read from the base run's
    leftover config. A missing R2 config fails loudly instead of falling
    back to whatever is on disk.
    """
    cfg_path = os.path.join(ckpt_dir, "config.yaml")
    store.download_run_file(run_name, "config.yaml", cfg_path)
    return cfg_path


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    del argv  # parameterized only by env vars (vast plan §3)

    run_name = os.environ.get("RUN_NAME")
    if not run_name:
        raise SystemExit("RUN_NAME is not set")
    joe_root = os.environ.get("JOE_ROOT", "/workspace/joe")
    ckpt_dir = os.environ.get("CKPT_DIR", os.path.join(joe_root, "ckpt"))
    log_path = os.environ.get("JOE_TRAIN_LOG", "")
    boot_id = os.environ.get("JOE_BOOT_ID", str(int(time.time())))
    instance_id = _instance_id()

    os.makedirs(ckpt_dir, exist_ok=True)
    cache_dir = os.path.join(joe_root, "jax-cache")
    os.makedirs(cache_dir, exist_ok=True)

    store = R2Store.from_env()
    reason = check_lease(store.read_heartbeat(run_name), instance_id)
    if reason:
        raise SystemExit(f"Refusing to start: {reason}")

    store.put_heartbeat(run_name, instance_id)
    print(f"Lease acquired: instance {instance_id} run {run_name}", flush=True)

    launch = store.read_launch(run_name)
    if launch is None:
        raise SystemExit(f"no launch.json under joe/{run_name}/")

    latest = store.resolve_latest(run_name)
    if latest is not None:
        state = latest["state"]
        print(f"Restoring checkpoint set at global step "
              f"{state['global_step']}, curriculum stage "
              f"{state['curriculum_stage']}", flush=True)
        store.download_checkpoint(latest, ckpt_dir)
    else:
        print("No latest.json; fresh start", flush=True)

    # metrics.jsonl is append-local and mirrored as a whole file, so a
    # boot on a machine with a stale (or empty) local copy would upload
    # that copy over the run's history. Seed the local file from R2 so
    # appends continue the union. Measured 2026-08-26: a cross-platform
    # handover cost the metrics rows of the previous platform's stretch.
    try:
        store.download_run_file(run_name, "logs/metrics.jsonl",
                                os.path.join(ckpt_dir, "metrics.jsonl"))
        print("Fetched metrics.jsonl for append continuity", flush=True)
    except FileNotFoundError:
        pass

    cfg_path = fetch_config(store, run_name, ckpt_dir)

    heartbeat = _IntervalUploader(
        lambda: store.put_heartbeat(run_name, instance_id),
        HEARTBEAT_S, name="joe-heartbeat")
    heartbeat.start()
    log_thread = None
    if log_path:
        log_thread = _IntervalUploader(
            lambda: _upload_log(store, run_name, log_path, boot_id),
            LOG_UPLOAD_S, name="joe-log-upload")
        log_thread.start()

    import jax
    jax.config.update("jax_compilation_cache_dir", cache_dir)

    from training.joe.config import Config
    from training.joe.main import run

    cfg = Config.from_yaml(cfg_path)
    cfg = apply_env_overrides(cfg, os.environ.get("JOE_CONFIG_OVERRIDES", ""))
    if cfg.run_name != run_name:
        print(f"NOTE: config run_name={cfg.run_name!r}; "
              f"overriding to {run_name!r}", flush=True)
        object.__setattr__(cfg, "run_name", run_name)

    # Frozen-reference eval files (ckpt_dir-relative). A missing R2 object
    # raises FileNotFoundError here — the config asked for a reference, so
    # starting without one would silently drop the eval_ref curve.
    for rel in (cfg.eval_ref_checkpoint, cfg.eval_ref_config):
        if rel:
            dest = os.path.join(ckpt_dir, rel)
            store.download_run_file(run_name, rel, dest)
            print(f"Fetched eval reference file {rel}", flush=True)

    uploader = CheckpointUploader(store, ckpt_dir, run_name)
    engine_sha = launch.get("engine_sha") or None
    try:
        run(cfg, ckpt_dir, engine_sha=engine_sha, on_checkpoint=uploader)
    finally:
        uploader.wait()
        heartbeat.stop()
        if log_thread is not None:
            log_thread.stop()
        print("vast_boot finished", flush=True)


if __name__ == "__main__":
    main()
