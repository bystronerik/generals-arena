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

# Run-dir file -> remote key (relative to the run prefix). Start files are
# written once at run start; mutable files re-upload on every checkpoint.
RUN_START_FILES = {"manifest.json": "manifest.json",
                   "config.yaml": "config.yaml",
                   "hparams.json": "hparams.json"}
MUTABLE_RUN_FILES = {"metrics.jsonl": "logs/metrics.jsonl"}


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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

    # -- run files and lease --

    def upload_run_file(self, run_name, rel, local_path):
        """Whole-file PUT of a small mutable run file (logs, manifest)."""
        with open(local_path, "rb") as f:
            self._put_bytes(self.key(run_name, *rel.split("/")), f)

    def put_heartbeat(self, run_name, instance_id):
        self._put_json(self.key(run_name, "lease", "heartbeat.json"),
                       {"instance_id": str(instance_id), "time": time.time()})

    def read_heartbeat(self, run_name):
        return self._get_json(self.key(run_name, "lease", "heartbeat.json"))


class CheckpointUploader:
    """Background ``on_checkpoint`` hook: mirror the run dir to R2.

    Each call joins any in-flight upload of the previous checkpoint (so the
    ordered protocol holds), re-reads ``state.json``, and uploads in a
    thread — the training loop never blocks on the network. A failed upload
    prints loudly and is retried on the next checkpoint; ``latest.json``
    keeps pointing at the last verified set in the meantime.
    """

    def __init__(self, store, ckpt_dir, run_name):
        self.store = store
        self.ckpt_dir = ckpt_dir
        self.run_name = run_name
        self._thread = None
        self._uploaded_step = -1
        self._start_files_done = False

    def __call__(self):
        self.wait()
        self._thread = threading.Thread(target=self._upload_once, daemon=True)
        self._thread.start()

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

    def _upload_once(self):
        try:
            if not self._start_files_done:
                self._upload_run_files(RUN_START_FILES)
                self._start_files_done = True
            self._upload_run_files(MUTABLE_RUN_FILES)

            state = read_state(self.ckpt_dir)
            if state is not None and \
                    state["global_step"] > self._uploaded_step:
                self.store.upload_checkpoint(self.ckpt_dir, state)
                self._uploaded_step = state["global_step"]

            # Terminal artifacts (written once, just before the final hook)
            for suffix in ("final.eqx", "ema_final.eqx"):
                path = os.path.join(self.ckpt_dir,
                                    f"{self.run_name}_{suffix}")
                if os.path.exists(path):
                    self.store.upload_file(
                        self.store.key(self.run_name, "checkpoints",
                                       os.path.basename(path)), path)
        except Exception:
            print(f"R2 upload failed (will retry on the next checkpoint); "
                  f"latest.json still points at step {self._uploaded_step}",
                  flush=True)
            traceback.print_exc()
