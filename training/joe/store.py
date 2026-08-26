"""R2 (S3-compatible) storage for vast.ai training runs.

Durable state for interruptible runs lives in one bucket under
``joe/<run_name>/`` (docs/research/strategies/joe-vast-training-plan.md,
section 2). S3 PUTs are atomic per object, so the only corruption risk is a
pointer to an incomplete set. ``upload_checkpoint`` removes that risk by
ordering: step-named blobs first, verify each against a local SHA-256, and
only then PUT ``state/latest.json`` referencing the exact keys and
checksums. A kill anywhere before the last PUT leaves ``latest.json`` on
the previous complete set.

No JAX or training-loop imports: the module is unit-testable against a fake
S3 client in the cheap default suite. ``training.joe.state`` is stdlib-only.
boto3 is imported lazily and only by ``R2Store.from_env``.
"""

import hashlib
import json
import os
import threading
import time
import traceback

from training.joe.state import STATE_FILENAME, read_state

DEFAULT_BUCKET = "joe-training"
DEFAULT_PREFIX = "joe"
LATEST_SCHEMA = 1
LAUNCH_SCHEMA = 1
LEASE_STALE_S = 5 * 60

# Run-dir file -> remote key (relative to the run prefix). Start files are
# written once at run start; mutable files re-upload on every checkpoint.
# config.yaml is deliberately absent: the R2 copy from launch is
# authoritative (vast_boot.fetch_config), and the ckpt_dir copy is the
# EFFECTIVE config after JOE_CONFIG_OVERRIDES. Measured 2026-08-26: a
# Modal 2-GPU boot uploaded its per-device num_envs=1024 config over the
# launch config, and the next single-GPU boot trained at half the recipe.
RUN_START_FILES = {"manifest.json": "manifest.json",
                   "hparams.json": "hparams.json"}
MUTABLE_RUN_FILES = {"metrics.jsonl": "logs/metrics.jsonl"}


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dotenv(path=None):
    """Fill missing ``R2_*`` vars from a gitignored ``.env``, if present.

    Does not override variables already in the environment. Looks at
    ``JOE_DOTENV``, then ``<repo>/.env`` next to ``training/``.
    """
    if path is None:
        path = os.environ.get("JOE_DOTENV")
        if not path:
            here = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(here, "..", "..", ".env")
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


def check_lease(heartbeat, instance_id, now=None, stale_s=LEASE_STALE_S):
    """None if this instance may start; a reason string if it must refuse.

    A heartbeat newer than ``stale_s`` from a *different* instance id means
    another writer holds the run. The same id (this machine rebooted) or a
    stale heartbeat is fine. ``resume`` writes the new instance id into the
    heartbeat before the replacement boots, so destroy-then-create does not
    trip this guard.
    """
    if heartbeat is None:
        return None
    now = time.time() if now is None else now
    age = now - float(heartbeat["time"])
    if age > stale_s:
        return None
    held = str(heartbeat.get("instance_id", ""))
    if held == str(instance_id):
        return None
    return (f"lease held by instance {held} "
            f"(heartbeat age {age:.0f}s < {stale_s}s)")


class R2Store:
    """Bucket operations for one run prefix; takes any S3-shaped client."""

    def __init__(self, client, bucket=DEFAULT_BUCKET, prefix=DEFAULT_PREFIX):
        self.client = client
        self.bucket = bucket
        self.prefix = prefix

    @classmethod
    def from_env(cls):
        """Real client from the environment (never from committed files).

        Needs ``R2_ENDPOINT_URL``, ``R2_ACCESS_KEY_ID``,
        ``R2_SECRET_ACCESS_KEY``; ``R2_BUCKET`` overrides the default.
        Locally these live in the gitignored ``.env`` / shell env; on
        vast.ai they arrive as account-level encrypted env vars.
        """
        import boto3

        load_dotenv()
        missing = [k for k in ("R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID",
                               "R2_SECRET_ACCESS_KEY") if not os.environ.get(k)]
        if missing:
            raise RuntimeError(f"Missing env vars for R2: {', '.join(missing)}")
        client = boto3.client(
            "s3",
            endpoint_url=os.environ["R2_ENDPOINT_URL"],
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            region_name="auto",
        )
        return cls(client, bucket=os.environ.get("R2_BUCKET", DEFAULT_BUCKET))

    def key(self, run_name, *parts):
        return "/".join((self.prefix, run_name) + parts)

    # -- low-level objects --

    def _put_bytes(self, key, body, metadata=None):
        self.client.put_object(Bucket=self.bucket, Key=key, Body=body,
                               Metadata=metadata or {})

    def _put_json(self, key, obj):
        self._put_bytes(key, json.dumps(obj, indent=2).encode())

    def _get_json(self, key):
        try:
            resp = self.client.get_object(Bucket=self.bucket, Key=key)
        except self.client.exceptions.NoSuchKey:
            return None
        return json.loads(resp["Body"].read())

    def upload_file(self, key, local_path):
        """PUT one file with size + SHA-256 in object metadata, then verify.

        Returns ``{"key", "size", "sha256"}`` — the reference a pointer
        object stores so a reader can verify without trusting the blob.
        """
        size = os.path.getsize(local_path)
        sha = _sha256_file(local_path)
        with open(local_path, "rb") as f:
            self._put_bytes(key, f, metadata={"sha256": sha,
                                              "size": str(size)})
        head = self.client.head_object(Bucket=self.bucket, Key=key)
        if head["ContentLength"] != size or \
                head["Metadata"].get("sha256") != sha:
            raise IOError(
                f"Upload verification failed for {key}: remote "
                f"size={head['ContentLength']} sha={head['Metadata'].get('sha256')}, "
                f"local size={size} sha={sha}")
        return {"key": key, "size": size, "sha256": sha}

    # -- checkpoint sets --

    def upload_checkpoint(self, ckpt_dir, state):
        """Ordered upload of one complete checkpoint set; returns the
        ``latest.json`` dict.

        ``state`` is the v2 ``state.json`` dict; its ``files`` map names the
        exact local checkpoint files. Step-named keys are new on every call,
        so nothing existing is touched until the final ``latest.json`` PUT.
        """
        run_name = state["run_name"]
        step = state["global_step"]
        objects = {}
        for role, basename in state["files"].items():
            key = self.key(run_name, "checkpoints", basename)
            objects[role] = self.upload_file(
                key, os.path.join(ckpt_dir, basename))

        state_key = self.key(run_name, "state", f"state-{step}.json")
        self._put_json(state_key, state)

        latest = {
            "schema": LATEST_SCHEMA,
            "state": state,
            "state_key": state_key,
            "objects": objects,
            "time": time.time(),
        }
        self._put_json(self.key(run_name, "state", "latest.json"), latest)
        return latest

    def resolve_latest(self, run_name):
        """The last complete checkpoint set, or None for a fresh run."""
        return self._get_json(self.key(run_name, "state", "latest.json"))

    def download_checkpoint(self, latest, dest_dir):
        """Download the referenced set into ``dest_dir``, verify checksums,
        and write ``state.json`` so ``main.run()`` resumes from it."""
        os.makedirs(dest_dir, exist_ok=True)
        for role, ref in latest["objects"].items():
            resp = self.client.get_object(Bucket=self.bucket, Key=ref["key"])
            data = resp["Body"].read()
            sha = hashlib.sha256(data).hexdigest()
            if len(data) != ref["size"] or sha != ref["sha256"]:
                raise IOError(
                    f"Checksum mismatch for {ref['key']}: got "
                    f"size={len(data)} sha={sha}, expected "
                    f"size={ref['size']} sha={ref['sha256']}")
            basename = ref["key"].rsplit("/", 1)[-1]
            path = os.path.join(dest_dir, basename)
            tmp = path + ".tmp"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, path)

        state_path = os.path.join(dest_dir, STATE_FILENAME)
        tmp = state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(latest["state"], f, indent=2)
        os.replace(tmp, state_path)
        return dest_dir

    def download_file(self, key, dest_path, expected=None):
        """Download one object; verify size/sha if ``expected`` is a ref dict."""
        try:
            resp = self.client.get_object(Bucket=self.bucket, Key=key)
        except self.client.exceptions.NoSuchKey:
            raise FileNotFoundError(key)
        data = resp["Body"].read()
        if expected is not None:
            sha = hashlib.sha256(data).hexdigest()
            if len(data) != expected["size"] or sha != expected["sha256"]:
                raise IOError(
                    f"Checksum mismatch for {key}: got "
                    f"size={len(data)} sha={sha}, expected "
                    f"size={expected['size']} sha={expected['sha256']}")
        parent = os.path.dirname(dest_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = dest_path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, dest_path)
        return dest_path

    # -- run files, launch pointer, and lease --

    def upload_run_file(self, run_name, rel, local_path):
        """Whole-file PUT of a small mutable run file (logs, manifest).

        The body is a bytes snapshot, never the open handle. These files
        are appended to while the upload runs — the logger flushes a
        metrics row per iteration, and the boot script tees the train log
        — and botocore fixes ``Content-Length`` when it prepares the
        request but then streams a handle to EOF. A write that lands
        mid-send makes the body outrun the header, and R2 rejects the whole
        PUT with ``IncompleteBody``. Measured 2026-08-18: that error on a
        68 MB ``metrics.jsonl`` cost the joe-M7 step-500 checkpoint.
        """
        with open(local_path, "rb") as f:
            data = f.read()
        self._put_bytes(self.key(run_name, *rel.split("/")), data)

    def download_run_file(self, run_name, rel, dest_path):
        return self.download_file(self.key(run_name, *rel.split("/")), dest_path)

    def put_launch(self, run_name, launch):
        """Write the launch pointer (code key, engine SHA, instance shape)."""
        if launch.get("schema", LAUNCH_SCHEMA) != LAUNCH_SCHEMA:
            raise ValueError(f"unsupported launch schema {launch.get('schema')}")
        self._put_json(self.key(run_name, "launch.json"), launch)
        return launch

    def read_launch(self, run_name):
        return self._get_json(self.key(run_name, "launch.json"))

    def put_instance(self, run_name, instance_id, extra=None):
        obj = {"instance_id": str(instance_id), "time": time.time()}
        if extra:
            obj.update(extra)
        self._put_json(self.key(run_name, "lease", "instance.json"), obj)
        return obj

    def read_instance(self, run_name):
        return self._get_json(self.key(run_name, "lease", "instance.json"))

    def put_heartbeat(self, run_name, instance_id):
        self._put_json(self.key(run_name, "lease", "heartbeat.json"),
                       {"instance_id": str(instance_id), "time": time.time()})

    def read_heartbeat(self, run_name):
        return self._get_json(self.key(run_name, "lease", "heartbeat.json"))

    def put_boot(self, run_name, phase, extra=None):
        """Onstart progress marker. Written before the long JAX pip install."""
        obj = {"schema": 1, "phase": str(phase), "time": time.time()}
        if extra:
            obj.update(extra)
        self._put_json(self.key(run_name, "logs", "boot.json"), obj)
        return obj

    def read_boot(self, run_name):
        return self._get_json(self.key(run_name, "logs", "boot.json"))

    def delete_prefix(self, run_name):
        """Delete every object under ``joe/<run_name>/``. Smoke cleanup only."""
        prefix = self.key(run_name) + "/"
        token = None
        deleted = 0
        while True:
            kw = {"Bucket": self.bucket, "Prefix": prefix}
            if token:
                kw["ContinuationToken"] = token
            resp = self.client.list_objects_v2(**kw)
            for obj in resp.get("Contents", []):
                self.client.delete_object(Bucket=self.bucket, Key=obj["Key"])
                deleted += 1
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")
        return deleted


class CheckpointUploader:
    """Background ``on_checkpoint`` hook: mirror the run dir to R2.

    Each call joins any in-flight upload of the previous checkpoint (so the
    ordered protocol holds), re-reads ``state.json``, and uploads in a
    thread — the training loop never blocks on the network, and an upload
    that raises is swallowed so R2 can never stop training.

    The stages are independent and the checkpoint set goes first: each
    stage catches its own failure, so an upload of a log can never discard
    the durable state.

    A failed upload prints loudly and ``latest.json`` keeps pointing at the
    last verified set. The next checkpoint re-reads ``state.json`` and
    uploads whatever it names *then*, so recovery skips the missed step
    rather than replaying it — only the newest set has to be durable for a
    resume. ``__call__`` returns the confirmed step so the caller can prune
    local files behind it.
    """

    def __init__(self, store, ckpt_dir, run_name):
        self.store = store
        self.ckpt_dir = ckpt_dir
        self.run_name = run_name
        self._thread = None
        self._uploaded_step = -1
        self._start_files_done = False

    def __call__(self):
        """Start the newest upload; return the last step R2 has verified.

        ``wait()`` runs first, so the returned step already accounts for the
        upload the previous call started. It is the newest global step whose
        blobs are all in the bucket and checksum-verified, which is exactly
        the step ``prune_checkpoints`` may delete below. A failed upload
        leaves it where it was, so nothing local is pruned until a later
        upload succeeds.
        """
        self.wait()
        confirmed = self._uploaded_step
        self._thread = threading.Thread(target=self._upload_once, daemon=True)
        self._thread.start()
        return confirmed

    def wait(self):
        """Block until the in-flight upload (if any) finishes."""
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _upload_run_files(self, files):
        for name, rel in files.items():
            path = os.path.join(self.ckpt_dir, name)
            if os.path.exists(path):
                self.store.upload_run_file(self.run_name, rel, path)

    def _stage(self, what, fn):
        """Run one upload stage; report a failure and continue.

        Each stage stands alone so a broken one cannot skip the stages
        behind it. Measured 2026-08-18: ``metrics.jsonl`` failed with
        ``IncompleteBody`` before the joe-M7 step-500 blobs were attempted,
        and that step never reached the bucket.
        """
        try:
            fn()
            return True
        except Exception:
            print(f"R2 upload failed ({what}; will retry on the next "
                  f"checkpoint); latest.json still points at step "
                  f"{self._uploaded_step}", flush=True)
            traceback.print_exc()
            return False

    def _upload_checkpoint_set(self):
        state = read_state(self.ckpt_dir)
        if state is None or state["global_step"] <= self._uploaded_step:
            return
        self.store.upload_checkpoint(self.ckpt_dir, state)
        self._uploaded_step = state["global_step"]

    def _upload_terminal_artifacts(self):
        """The two ``*_final.eqx`` files, written just before the last hook."""
        for suffix in ("final.eqx", "ema_final.eqx"):
            path = os.path.join(self.ckpt_dir, f"{self.run_name}_{suffix}")
            if os.path.exists(path):
                self.store.upload_file(
                    self.store.key(self.run_name, "checkpoints",
                                   os.path.basename(path)), path)

    def _upload_start_files(self):
        self._upload_run_files(RUN_START_FILES)
        self._start_files_done = True

    def _upload_once(self):
        # Durable state first, logs last: a checkpoint set is the only
        # thing a resume needs, so nothing cheaper may stand in front of it.
        self._stage("checkpoint", self._upload_checkpoint_set)
        self._stage("final artifacts", self._upload_terminal_artifacts)
        if not self._start_files_done:
            self._stage("start files", self._upload_start_files)
        self._stage("run logs",
                    lambda: self._upload_run_files(MUTABLE_RUN_FILES))
