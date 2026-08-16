"""R2 store logic against a fake S3 client (cheap default suite).

No JAX and no network: ``training.joe.store`` only needs an object with the
boto3 client shape. The fake keeps objects in a dict and records the PUT
order, which is the property the ordered upload protocol lives on:
``latest.json`` is always the last write, so a kill anywhere earlier leaves
it pointing at the previous complete set.
"""

from __future__ import annotations

import hashlib
import io
import json
import os

import pytest

from training.joe.state import write_state
from training.joe.store import (
    CheckpointUploader,
    R2Store,
    check_lease,
)


class NoSuchKey(Exception):
    pass


class FakeExceptions:
    NoSuchKey = NoSuchKey


class FakeS3Client:
    """The subset of the boto3 S3 client the store uses."""

    def __init__(self):
        self.objects = {}   # key -> (bytes, metadata)
        self.put_order = []
        self.fail_after_puts = None

    def put_object(self, Bucket, Key, Body, Metadata=None):
        if self.fail_after_puts is not None and \
                len(self.put_order) >= self.fail_after_puts:
            raise IOError("injected network failure")
        data = Body.read() if hasattr(Body, "read") else Body
        self.objects[Key] = (data, dict(Metadata or {}))
        self.put_order.append(Key)

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise NoSuchKey(Key)
        data, metadata = self.objects[Key]
        return {"ContentLength": len(data), "Metadata": metadata}

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise NoSuchKey(Key)
        data, _ = self.objects[Key]
        return {"Body": io.BytesIO(data)}

    def delete_object(self, Bucket, Key):
        self.objects.pop(Key, None)

    def list_objects_v2(self, Bucket, Prefix="", ContinuationToken=None):
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        return {"Contents": [{"Key": k} for k in keys], "IsTruncated": False}

    exceptions = FakeExceptions


def make_ckpt_dir(tmp_path, run_name, step, stage=0, wr=0.0):
    """A run dir holding one full checkpoint set + its state.json."""
    files = {"full": f"{run_name}_{step}.eqx",
             "ema": f"{run_name}_ema_{step}.eqx"}
    for basename in files.values():
        (tmp_path / basename).write_bytes(
            f"weights-{basename}".encode() * 100)
    state = write_state(str(tmp_path), run_name, step, stage, wr, "sha",
                        files=files)
    return state


@pytest.fixture
def store():
    return R2Store(FakeS3Client(), bucket="test-bucket")


def test_upload_checkpoint_round_trip(store, tmp_path):
    state = make_ckpt_dir(tmp_path, "joe-x", 100, stage=1, wr=0.5)
    latest = store.upload_checkpoint(str(tmp_path), state)

    resolved = store.resolve_latest("joe-x")
    assert resolved == json.loads(json.dumps(latest))
    assert resolved["state"]["global_step"] == 100
    assert resolved["state"]["curriculum_stage"] == 1
    assert set(resolved["objects"]) == {"full", "ema"}
    for ref in resolved["objects"].values():
        data, _ = store.client.objects[ref["key"]]
        assert len(data) == ref["size"]
        assert hashlib.sha256(data).hexdigest() == ref["sha256"]


def test_latest_json_is_written_last(store, tmp_path):
    state = make_ckpt_dir(tmp_path, "joe-x", 100)
    store.upload_checkpoint(str(tmp_path), state)
    assert store.client.put_order[-1] == "joe/joe-x/state/latest.json"
    # The immutable copies precede it
    assert "joe/joe-x/state/state-100.json" in store.client.put_order[:-1]


def test_kill_mid_upload_preserves_previous_latest(store, tmp_path):
    state = make_ckpt_dir(tmp_path, "joe-x", 100)
    store.upload_checkpoint(str(tmp_path), state)

    # Second set; the client dies after the blobs, before latest.json
    state2 = make_ckpt_dir(tmp_path, "joe-x", 200)
    store.client.fail_after_puts = len(store.client.put_order) + 2
    with pytest.raises(IOError):
        store.upload_checkpoint(str(tmp_path), state2)

    resolved = store.resolve_latest("joe-x")
    assert resolved["state"]["global_step"] == 100
    for ref in resolved["objects"].values():
        assert ref["key"] in store.client.objects


def test_resolve_latest_absent_returns_none(store):
    assert store.resolve_latest("joe-x") is None


def test_download_checkpoint_restores_run_dir(store, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    state = make_ckpt_dir(src, "joe-x", 100, stage=2, wr=0.7)
    latest = store.upload_checkpoint(str(src), state)

    dest = tmp_path / "dest"
    store.download_checkpoint(latest, str(dest))
    for basename in state["files"].values():
        assert (dest / basename).read_bytes() == (src / basename).read_bytes()
    restored = json.loads((dest / "state.json").read_text())
    assert restored["global_step"] == 100
    assert restored["curriculum_stage"] == 2


def test_download_rejects_corrupted_object(store, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    state = make_ckpt_dir(src, "joe-x", 100)
    latest = store.upload_checkpoint(str(src), state)

    key = latest["objects"]["full"]["key"]
    data, meta = store.client.objects[key]
    store.client.objects[key] = (data[:-1] + b"X", meta)
    with pytest.raises(IOError, match="Checksum mismatch"):
        store.download_checkpoint(latest, str(tmp_path / "dest"))


def test_upload_verification_catches_truncated_put(store, tmp_path):
    class TruncatingClient(FakeS3Client):
        def put_object(self, Bucket, Key, Body, Metadata=None):
            data = Body.read() if hasattr(Body, "read") else Body
            self.objects[Key] = (data[:-1], dict(Metadata or {}))
            self.put_order.append(Key)

    store = R2Store(TruncatingClient(), bucket="test-bucket")
    state = make_ckpt_dir(tmp_path, "joe-x", 100)
    with pytest.raises(IOError, match="Upload verification failed"):
        store.upload_checkpoint(str(tmp_path), state)


def test_heartbeat_round_trip(store):
    assert store.read_heartbeat("joe-x") is None
    store.put_heartbeat("joe-x", "instance-42")
    hb = store.read_heartbeat("joe-x")
    assert hb["instance_id"] == "instance-42"
    assert hb["time"] > 0


def test_boot_json_round_trip(store):
    assert store.read_boot("joe-x") is None
    rec = store.put_boot("joe-x", "installing_pip_stack",
                         extra={"python_version": "3.12.0"})
    got = store.read_boot("joe-x")
    assert got["phase"] == "installing_pip_stack"
    assert got["python_version"] == "3.12.0"
    assert got["time"] == rec["time"]


def test_uploader_mirrors_checkpoints_and_run_files(store, tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "config.yaml").write_text("run_name: joe-x")
    (tmp_path / "metrics.jsonl").write_text('{"step": 1}\n')
    uploader = CheckpointUploader(store, str(tmp_path), "joe-x")

    # First hook call: run start, no checkpoint yet
    uploader()
    uploader.wait()
    assert "joe/joe-x/manifest.json" in store.client.objects
    assert "joe/joe-x/config.yaml" in store.client.objects
    assert "joe/joe-x/logs/metrics.jsonl" in store.client.objects
    assert store.resolve_latest("joe-x") is None

    # A checkpoint lands; the next call uploads the set
    make_ckpt_dir(tmp_path, "joe-x", 100)
    uploader()
    uploader.wait()
    assert store.resolve_latest("joe-x")["state"]["global_step"] == 100

    # No new state: nothing re-uploaded except the mutable run files
    puts_before = len(store.client.put_order)
    uploader()
    uploader.wait()
    new_puts = store.client.put_order[puts_before:]
    assert new_puts == ["joe/joe-x/logs/metrics.jsonl"]

    # A newer step uploads again and moves latest forward
    make_ckpt_dir(tmp_path, "joe-x", 200)
    uploader()
    uploader.wait()
    assert store.resolve_latest("joe-x")["state"]["global_step"] == 200
    # The step-100 blobs were never touched again
    assert store.client.put_order.count("joe/joe-x/checkpoints/joe-x_100.eqx") == 1


def test_uploader_failure_retries_next_checkpoint(store, tmp_path, capsys):
    make_ckpt_dir(tmp_path, "joe-x", 100)
    uploader = CheckpointUploader(store, str(tmp_path), "joe-x")

    store.client.fail_after_puts = 0
    uploader()
    uploader.wait()
    assert store.resolve_latest("joe-x") is None
    assert "R2 upload failed" in capsys.readouterr().out

    store.client.fail_after_puts = None
    uploader()
    uploader.wait()
    assert store.resolve_latest("joe-x")["state"]["global_step"] == 100


def test_uploader_returns_the_confirmed_step(store, tmp_path):
    # The return value is the floor prune_checkpoints deletes below, so it
    # reports the step the *previous* call verified, never the one this call
    # is about to start uploading.
    make_ckpt_dir(tmp_path, "joe-x", 100)
    uploader = CheckpointUploader(store, str(tmp_path), "joe-x")
    assert uploader() == -1
    make_ckpt_dir(tmp_path, "joe-x", 200)
    assert uploader() == 100
    assert uploader() == 200


def test_confirmed_step_holds_still_while_uploads_fail(store, tmp_path,
                                                       capsys):
    # A stalled bucket must never move the prune floor: every checkpoint
    # from the last verified step onward stays on disk until R2 comes back.
    make_ckpt_dir(tmp_path, "joe-x", 100)
    uploader = CheckpointUploader(store, str(tmp_path), "joe-x")
    uploader()
    uploader.wait()

    store.client.fail_after_puts = 0
    for step in (200, 300, 400):
        make_ckpt_dir(tmp_path, "joe-x", step)
        assert uploader() == 100
    assert "R2 upload failed" in capsys.readouterr().out
    assert store.resolve_latest("joe-x")["state"]["global_step"] == 100

    # Recovery uploads the newest state and skips the missed steps outright
    store.client.fail_after_puts = None
    uploader()
    uploader.wait()
    assert store.resolve_latest("joe-x")["state"]["global_step"] == 400
    assert uploader() == 400
    for skipped in ("joe-x_200.eqx", "joe-x_300.eqx"):
        assert f"joe/joe-x/checkpoints/{skipped}" not in store.client.objects


def test_uploader_uploads_final_artifacts(store, tmp_path):
    state = make_ckpt_dir(tmp_path, "joe-x", 100)
    (tmp_path / "joe-x_final.eqx").write_bytes(b"final" * 100)
    (tmp_path / "joe-x_ema_final.eqx").write_bytes(b"ema-final" * 100)
    uploader = CheckpointUploader(store, str(tmp_path), "joe-x")
    uploader()
    uploader.wait()
    assert "joe/joe-x/checkpoints/joe-x_final.eqx" in store.client.objects
    assert "joe/joe-x/checkpoints/joe-x_ema_final.eqx" in store.client.objects
    assert store.resolve_latest("joe-x")["state"] == \
        json.loads(json.dumps(state))


def test_launch_pointer_round_trip(store):
    launch = {"schema": 1, "run_name": "joe-x", "code_key": "joe/joe-x/code/a.tar.gz",
              "code_sha256": "abc", "engine_sha": "def"}
    store.put_launch("joe-x", launch)
    assert store.read_launch("joe-x") == launch
    assert store.read_launch("missing") is None


def test_download_file_verifies_checksum(store, tmp_path):
    src = tmp_path / "blob.bin"
    src.write_bytes(b"hello-r2")
    ref = store.upload_file("joe/joe-x/code/x.tar.gz", str(src))
    dest = tmp_path / "out.bin"
    store.download_file(ref["key"], str(dest), expected=ref)
    assert dest.read_bytes() == b"hello-r2"
    store.client.objects[ref["key"]] = (b"tampered!", {"sha256": ref["sha256"]})
    with pytest.raises(IOError, match="Checksum mismatch"):
        store.download_file(ref["key"], str(tmp_path / "bad.bin"), expected=ref)


def test_download_missing_file_raises(store, tmp_path):
    with pytest.raises(FileNotFoundError):
        store.download_file("joe/nope", str(tmp_path / "x"))


def test_instance_and_delete_prefix(store, tmp_path):
    store.put_instance("joe-x", "42", extra={"offer_id": "9"})
    rec = store.read_instance("joe-x")
    assert rec["instance_id"] == "42"
    assert rec["offer_id"] == "9"
    src = tmp_path / "a.bin"
    src.write_bytes(b"aa")
    store.upload_file("joe/joe-x/code/a.tar.gz", str(src))
    store.put_launch("joe-x", {"schema": 1, "run_name": "joe-x"})
    n = store.delete_prefix("joe-x")
    assert n >= 3
    assert store.read_launch("joe-x") is None
    assert store.read_instance("joe-x") is None
    assert all(not k.startswith("joe/joe-x/") for k in store.client.objects)


def test_lease_same_instance_or_stale_allows():
    now = 10_000.0
    assert check_lease(None, "a", now=now) is None
    assert check_lease({"instance_id": "a", "time": now - 1}, "a",
                       now=now) is None
    assert check_lease({"instance_id": "b", "time": now - 900}, "a",
                       now=now) is None


def test_lease_fresh_other_instance_refuses():
    now = 10_000.0
    reason = check_lease({"instance_id": "b", "time": now - 10}, "a",
                         now=now)
    assert reason is not None
    assert "b" in reason
