#!/usr/bin/env python3
"""Build the prebuilt x86_64 binary that joe-rs submissions ship, and prove it.

    .venv/bin/modal run scripts/joe_rs_modal_static_build.py

Writes `bots/joe-rs/tools/submission/joe-rs-x86_64-linux-gnu` and its sidecar
`.json`, then `bots/joe-rs/tools/package_submission.py --force` ships them.

**Run this after any change to `bots/joe-rs/src/` or `Cargo.lock`.** The binary
is derived state living outside the rated closure — `tools/` is skipped by the
content-hash walk — so nothing else notices when it goes stale. The sidecar
records the content hash it was built from and the packager refuses to ship a
binary whose sidecar disagrees with the tree.

Why a prebuilt binary at all
----------------------------

The judge runs `build.sh` before `run.sh`. Compiling joe-rs's 93 vendored
crates there costs 71-131 s, 342 MB of disk and 1.6 GB of memory, against
morpheus-rs's 17 s and near-nothing — and joe-rs was rejected from the
tournament for crashing in evaluation rounds while morpheus-rs passed, with no
failure reproducible in any rehearsal this repo can run. A prebuilt binary
turns intake into a chmod, which does not narrow the cause so much as delete
the whole class.

Why glibc 2.31 rather than static musl
--------------------------------------

Measured, not preferred. Both run 1,602 recorded turns correctly; musl costs
4x per forward pass because this network allocates hard and musl's allocator is
slow:

| build | p50 | p99 | max | startup |
| --- | --- | --- | --- | --- |
| musl static | 65.2 ms | 79.5 ms | 92.7 ms | 762 ms |
| glibc 2.31 | 16.0 ms | 19.4 ms | 23.1 ms | 169 ms |

The budget is 150 ms a move (RULES.md §08). musl would survive it and spend
half the budget doing so. An old-glibc build is the ordinary way to ship a
portable Linux binary: linked against Debian bullseye's 2.31, it runs on any
newer glibc, and `run.sh` self-tests and falls back to a passing seat if the
runtime is somehow older.

Verification runs in an image with **no Rust and no network**, so portability
is proved rather than assumed: `selfcheck`, then a full 1,602-turn game.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import modal

app = modal.App("joe-rs-static-build")
VOL = modal.Volume.from_name("joe-rs-static", create_if_missing=True)

TARGET = "x86_64-unknown-linux-gnu"
RUSTFLAGS = "-C target-cpu=x86-64-v3"
BINARY_NAME = "joe-rs-x86_64-linux-gnu"

# Debian bullseye: glibc 2.31. A binary linked here runs on every newer glibc,
# which is the whole point of not building on the current image.
BUILD_IMAGE = (
    modal.Image.from_registry("debian:bullseye", add_python="3.12")
    .apt_install("curl", "build-essential")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
        "| sh -s -- -y --default-toolchain 1.97.1 --profile minimal"
    )
)
# No Rust. If the binary needs anything the build image provided, it fails here.
VERIFY_IMAGE = modal.Image.debian_slim(python_version="3.12")

if modal.is_local():
    REPO = Path(__file__).resolve().parents[1]
    BOT = REPO / "bots" / "joe-rs"
    GAME = REPO / "data" / "joe" / "joe-rs-parity" / "games" / "synthetic-long.in.log"
    BUILD_IMAGE = (
        BUILD_IMAGE.add_local_dir(str(BOT / "src"), remote_path="/src/src", copy=True)
        .add_local_file(str(BOT / "Cargo.toml"), remote_path="/src/Cargo.toml", copy=True)
        .add_local_file(str(BOT / "Cargo.lock"), remote_path="/src/Cargo.lock", copy=True)
        # A network exists while the image builds; the compile below is offline.
        .run_commands("cd /src && /root/.cargo/bin/cargo fetch --locked")
    )
    VERIFY_IMAGE = (
        VERIFY_IMAGE.add_local_dir(str(BOT / "artifact"), remote_path="/bot/artifact", copy=True)
        .add_local_file(str(GAME), remote_path="/game.in.log", copy=True)
    )


@app.function(image=BUILD_IMAGE, cpu=4, memory=(4096, 4096), timeout=60 * 45,
              volumes={"/vol": VOL})
def build() -> dict:
    import os

    env = {**os.environ, "PATH": f"/root/.cargo/bin:{os.environ['PATH']}",
           "RUSTFLAGS": RUSTFLAGS}
    began = time.time()
    proc = subprocess.run(
        ["cargo", "build", "--release", "--offline", "--locked", "--target", TARGET],
        cwd="/src", env=env, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return {"returncode": proc.returncode, "stderr_tail": proc.stderr[-2500:]}

    binary = Path("/src/target") / TARGET / "release" / "joe-rs"
    data = binary.read_bytes()
    Path(f"/vol/{BINARY_NAME}").write_bytes(data)
    Path(f"/vol/{BINARY_NAME}").chmod(0o755)
    VOL.commit()
    ldd = subprocess.run(["ldd", str(binary)], capture_output=True, text=True)
    return {
        "returncode": 0,
        "seconds": round(time.time() - began, 1),
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "ldd": (ldd.stdout + ldd.stderr).strip()[:400],
    }


@app.function(image=VERIFY_IMAGE, cpu=1, memory=(2048, 2048), timeout=60 * 40,
              volumes={"/vol": VOL}, block_network=True)
def verify() -> dict:
    import os
    import shutil

    root = Path("/bot")
    shutil.copy(f"/vol/{BINARY_NAME}", root / "joe-rs")
    (root / "joe-rs").chmod(0o755)
    env = {**os.environ, "JOE_RS_ARTIFACT": str(root / "artifact")}
    out: dict = {"rust_present": shutil.which("cargo") is not None}

    check = subprocess.run([str(root / "joe-rs"), "selfcheck"], env=env,
                           capture_output=True, text=True, timeout=300)
    out["selfcheck_returncode"] = check.returncode
    out["selfcheck"] = dict(
        line.partition(" ")[::2] for line in check.stdout.splitlines() if " " in line
    )

    frames = Path("/game.in.log").read_text()
    n_frames = sum(1 for ln in frames.splitlines()
                   if ln and ln[0].isdigit() and len(ln.split()) == 5)
    play = subprocess.run([str(root / "joe-rs")], env=env, input=frames,
                          capture_output=True, text=True, timeout=900)
    replies = [ln for ln in play.stdout.strip().splitlines() if ln.strip()]
    bench = subprocess.run([str(root / "joe-rs"), "bench"], env=env,
                           input=frames, capture_output=True, text=True, timeout=60 * 30)
    out.update(
        {
            "game_returncode": play.returncode,
            "frames": n_frames,
            "replies": len(replies),
            "bench": bench.stdout.strip(),
        }
    )
    return out


@app.local_entrypoint()
def main() -> None:
    built = build.remote()
    print("BUILD", json.dumps(built, indent=2)[:1200])
    if built["returncode"] != 0:
        raise SystemExit("build failed")

    checked = verify.remote()
    print("VERIFY", json.dumps(checked, indent=2)[:1600])
    if checked["selfcheck_returncode"] != 0 or checked["replies"] != checked["frames"]:
        raise SystemExit("verification failed; the binary was not written locally")

    repo = Path(__file__).resolve().parents[1]
    dest = repo / "bots" / "joe-rs" / "tools" / "submission" / BINARY_NAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"".join(VOL.read_file(BINARY_NAME)))
    dest.chmod(0o755)

    sys.path.insert(0, str(repo))
    from arena.records.fingerprint import bot_content_hash

    sidecar = {
        "content_hash": bot_content_hash(repo / "bots" / "joe-rs" / "run.sh"),
        "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
        "size_bytes": dest.stat().st_size,
        "target": TARGET,
        "glibc": "2.31 (Debian bullseye)",
        "rustc": "1.97.1",
        "rustflags": RUSTFLAGS,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bench": checked.get("bench", ""),
        "note": "Rebuild after any change to bots/joe-rs/src/ or Cargo.lock.",
    }
    dest.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print(f"\nwrote {dest.relative_to(repo)} ({sidecar['size_bytes']} B)")
    print(f"wrote {dest.with_suffix('.json').relative_to(repo)} "
          f"at content hash {sidecar['content_hash']}")
    print(f"RESULT selfcheck_rc={checked['selfcheck_returncode']} "
          f"game={checked['replies']}/{checked['frames']} {checked.get('bench','')}")
