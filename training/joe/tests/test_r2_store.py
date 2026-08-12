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
from training.joe.store import CheckpointUploader, R2Store


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
