#!/usr/bin/env python3
"""unclejoe: same-host interleaved A/B for forward-pass optimisations.

    python scripts/unclejoe_modal_forward_ab.py --stage <worktree> --as <arm-name>
    modal run scripts/unclejoe_modal_forward_ab.py > /tmp/uj_ab.log 2>&1 &
    modal app list          # then check startup a few minutes in
    modal app logs <app-id>

Writes docs/research/measurements/unclejoe-forward-ab-modal.json.

The fleet spans 1.72x for identical bytes
([unclejoe-forward-steps](../docs/research/measurements/unclejoe-forward-steps.md)),
so a before/after taken in two different containers measures the fleet, not
the change. This builds every arm in ONE container and runs them
**interleaved**, base first and base again last, for `--rounds` rounds. The
ratio that counts is arm-to-base inside one container; the round-to-round
spread of the base arm is the noise floor that ratio has to clear.

Arms are staged locally before the run:

    python scripts/unclejoe_modal_forward_ab.py --stage /path/to/worktree --as ff-fused

`base` is staged automatically from the committed tree at HEAD. Staging only
copies the crate sources — the 118 MB weights are shared, unpacked once in
the image from `model.packed`.

Correctness is a gate, not a metric: every arm replays the recorded wire log
in play mode and its replies must be byte-identical to base's. An arm that
changes a reply is reported as a failure whatever its latency did, because
on x86 the dispatch picks AVX2 or AVX-512 tiles the arm64 dev box never runs.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
MEASUREMENTS = REPO / "docs" / "research" / "measurements"
BOT = REPO / "bots" / "unclejoe"
IN_LOG = REPO / "data" / "joe" / "joe-rs-parity" / "games" / "synthetic-long.in.log"
ARMS = REPO / "data" / "unclejoe_ab_arms"          # gitignored staging area

CRATE_FILES = ["Cargo.toml", "Cargo.lock", "rust-toolchain.toml"]
REMOTE_ARMS = "/root/arms"
ARTIFACT = "/root/artifact"


def stage(src: Path, name: str) -> None:
    """Copy one crate's sources into the staging area as arm `name`."""
    if not (src / "src" / "main.rs").exists():
        src = src / "bots" / "unclejoe"
    if not (src / "src" / "main.rs").exists():
        sys.exit(f"no unclejoe crate at {src}")
    dst = ARMS / name
    if dst.exists():
        shutil.rmtree(dst)
    (dst / ".cargo").mkdir(parents=True)
    shutil.copytree(src / "src", dst / "src")
    (dst / "tests" / "fixtures").mkdir(parents=True)
    shutil.copy2(BOT / "tests" / "fixtures" / "rans_vectors.rs",
                 dst / "tests" / "fixtures" / "rans_vectors.rs")
    for f in CRATE_FILES:
        shutil.copy2(src / f, dst / f)
    shutil.copy2(src / ".cargo" / "config.toml", dst / ".cargo" / "config.toml")
    print(f"staged {src} as arm {name!r} -> {dst}")


def stage_base() -> None:
    """Stage the committed HEAD tree, so base is never a dirty worktree."""
    dst = ARMS / "base"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    out = subprocess.run(
        ["git", "archive", "--format=tar", "HEAD:bots/unclejoe"],
        cwd=REPO, check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(dst)], input=out, check=True)
    for junk in ("artifact", "tools", "run.sh"):
        p = dst / junk
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
    print(f"staged HEAD as arm 'base' -> {dst}")


def arm_names() -> list[str]:
    if not ARMS.exists():
        sys.exit("no staged arms; run --stage first")
    names = sorted(p.name for p in ARMS.iterdir() if (p / "src" / "main.rs").exists())
    if "base" not in names:
        sys.exit("arm 'base' missing; re-run to stage HEAD")
    return ["base"] + [n for n in names if n != "base"]


def _clean_targets() -> None:
    """Drop any `target/` under the staging area before the image is built.

    The image compiles every arm itself. A local build tree would be
    uploaded, ignored, and would make the arms differ by nothing but upload
    size.
    """
    if ARMS.exists():
        for t in ARMS.glob("*/target"):
            shutil.rmtree(t)


_clean_targets()

IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "build-essential")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
        "| sh -s -- -y --profile minimal --default-toolchain 1.97.1 "
        "--component rustfmt --component clippy "
        "--target x86_64-unknown-linux-musl"
    )
    .env({"PATH": "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin"})
    .add_local_dir(str(ARMS), REMOTE_ARMS, copy=True)
    .add_local_file(str(BOT / "artifact" / "manifest.json"),
                    f"{ARTIFACT}/manifest.json", copy=True)
    .add_local_file(str(BOT / "artifact" / "model.packed"),
                    f"{ARTIFACT}/model.packed", copy=True)
    .add_local_file(str(IN_LOG), "/root/synthetic-long.in.log", copy=True)
    # Build every arm in the image, so no measurement pays a compile and all
    # containers run the same bytes for every arm.
    .run_commands(
        f"for a in {REMOTE_ARMS}/*/; do echo \"building $a\"; "
        f"(cd \"$a\" && cargo build --release) || exit 1; done",
        f"cd {REMOTE_ARMS}/base && JOE_RS_ARTIFACT={ARTIFACT} "
        f"./target/release/unclejoe unpack-artifact",
    )
)

app = modal.App("unclejoe-forward-ab")


@app.function(
    image=IMAGE, cpu=1, memory=4096, timeout=60 * 60, single_use_containers=True)
def ab(spec: dict) -> dict:
    import os
    import platform
    import time

    arms: list[str] = spec["arms"]
    rounds: int = spec["rounds"]

    for var in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS",
    ):
        os.environ[var] = "1"

    out: dict = {
        "replica": spec["replica"],
        "platform": platform.platform(),
        "arms": arms,
        "rounds": rounds,
        "cpu_ident": "",
        "cpu_mhz": "",
        "avx512f": False,
        "avx2": False,
    }
    ident: dict[str, str] = {}
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        if key in ("vendor_id", "cpu family", "model", "stepping") and key not in ident:
            ident[key] = value
        elif key == "cpu MHz" and not out["cpu_mhz"]:
            out["cpu_mhz"] = value
        elif key == "flags" and not out["avx2"]:
            flags = set(value.split())
            out["avx2"] = "avx2" in flags and "fma" in flags
            out["avx512f"] = "avx512f" in flags
    out["cpu_ident"] = " ".join(
        f"{k}={ident[k]}" for k in ("vendor_id", "cpu family", "model", "stepping")
        if k in ident)

    env = dict(os.environ)
    env["JOE_RS_ARTIFACT"] = ARTIFACT

    def run(arm: str, args: list[str]):
        with open("/root/synthetic-long.in.log", "rb") as f:
            return subprocess.run(
                [f"{REMOTE_ARMS}/{arm}/target/release/unclejoe", *args],
                stdin=f, capture_output=True, text=True, env=env)

    # Correctness first: an arm that changes a reply is a failure whatever it
    # did to the clock, and there is no point timing it.
    out["replies"] = {}
    base_replies = None
    for arm in arms:
        played = run(arm, [])
        if arm == "base":
            base_replies = played.stdout
        out["replies"][arm] = {
            "rc": played.returncode,
            "bytes": len(played.stdout),
            "identical_to_base": played.stdout == base_replies,
        }
        check = subprocess.run(
            [f"{REMOTE_ARMS}/{arm}/target/release/unclejoe", "selfcheck"],
            capture_output=True, text=True, env=env)
        out["replies"][arm]["selfcheck_rc"] = check.returncode
        out["replies"][arm]["runtime_gemm"] = next(
            (ln.split()[1] for ln in check.stdout.splitlines()
             if ln.startswith("runtime_gemm")), "?")

    # Interleaved timing: base, arm1, ..., base, arm1, ... The base arm's own
    # round-to-round spread is the noise floor every ratio has to clear.
    order = [a for _ in range(rounds) for a in arms] + ["base"]
    out["schedule"] = order
    samples: dict[str, list[dict]] = {a: [] for a in arms}
    for arm in order:
        t0 = time.time()
        r = run(arm, ["bench", "--forward-stages"])
        if r.returncode != 0:
            out["error"] = f"{arm}: {r.stderr[-2000:]}"
            return out
        lines = [ln for ln in r.stdout.strip().splitlines() if ln.strip()]
        head = lines[0].split()
        blobs = [json.loads(ln) for ln in lines[1:]]
        steps = next(b for b in blobs if "steps" in b)
        samples[arm].append({
            "wall_s": round(time.time() - t0, 1),
            "turns": int(head[1]),
            "p50": float(head[head.index("p50") + 1]),
            "p90": float(head[head.index("p90") + 1]),
            "p99": float(head[head.index("p99") + 1]),
            "forward_mean": steps["steps"]["forward"]["mean"],
            "uninstrumented_forward_mean": steps["uninstrumented_forward_mean_ms"],
            "step_means": {k: v["mean"] for k, v in steps["steps"].items()},
        })
    out["samples"] = samples
    return out


@app.local_entrypoint()
def main(containers: int = 2, rounds: int = 2) -> None:
    arms = arm_names()
    print(f"arms: {arms}; {containers} containers x {rounds} interleaved rounds")
    specs = [{"replica": i, "arms": arms, "rounds": rounds} for i in range(containers)]
    results = sorted(ab.map(specs), key=lambda r: r["replica"])

    for r in results:
        print(f"\n===== replica {r['replica']}: {r['cpu_ident']} @ {r['cpu_mhz']} MHz "
              f"(avx512f={r['avx512f']}) =====")
        if "error" in r:
            print("  ERROR:", r["error"])
            continue
        for arm in r["arms"]:
            rep = r["replies"][arm]
            ok = "same replies" if rep["identical_to_base"] else "REPLIES DIFFER"
            print(f"  {arm:<16} {ok:<16} selfcheck_rc={rep['selfcheck_rc']} "
                  f"gemm={rep['runtime_gemm']}")
        base = [s["forward_mean"] for s in r["samples"]["base"]]
        base_mean = sum(base) / len(base)
        spread = (max(base) - min(base)) / base_mean * 100 if base_mean else 0
        print(f"  base forward mean {base_mean:.3f} ms over {len(base)} runs, "
              f"round-to-round spread {spread:.2f}% (the noise floor)")
        for arm in r["arms"]:
            if arm == "base":
                continue
            vals = [s["forward_mean"] for s in r["samples"][arm]]
            mean = sum(vals) / len(vals)
            print(f"  {arm:<16} forward mean {mean:.3f} ms  "
                  f"ratio {base_mean / mean:.4f}x  ({100 * (base_mean - mean) / base_mean:+.2f}%)")

    MEASUREMENTS.mkdir(parents=True, exist_ok=True)
    path = MEASUREMENTS / "unclejoe-forward-ab-modal.json"
    path.write_text(json.dumps({"results": results}, indent=2) + "\n")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", type=Path, help="worktree or crate dir to stage as an arm")
    ap.add_argument("--as", dest="name", help="arm name for --stage")
    args = ap.parse_args()
    ARMS.mkdir(parents=True, exist_ok=True)
    stage_base()
    if args.stage:
        if not args.name:
            sys.exit("--stage needs --as <arm-name>")
        if args.name == "base":
            sys.exit("'base' is reserved for the committed HEAD tree")
        stage(args.stage.resolve(), args.name)
    print("arms staged:", arm_names())
