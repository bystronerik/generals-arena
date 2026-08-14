#!/usr/bin/env python3
"""Build and run the joe-rs submission zip on Linux x86, with no network.

    python bots/joe-rs/tools/package_submission.py --force
    modal run scripts/joe_rs_modal_submission_smoke.py

Writes docs/research/measurements/joe-rs-sandbox-smoke.{json,md}.

Milestone J5.4 of docs/bots/joe-rs/packaging.md. joe-rs gets **no vendor
probe**, unlike morpheus: that probe exists because morpheus ships an empty
`vendor/`, where `--offline` cannot fail for want of anything to resolve. joe-rs
vendors 93 crates, so the shipped archive is the probe.

Four things this proves that a macOS packaging run cannot:

1. **Source replacement against a real graph** — 93 crates resolved out of
   `vendor/` with no network, and the build scripts in `libc`, `candle-core`,
   `gemm-*` and `half` compiling and *running* at intake.
2. **The compile flag actually applies.** `.cargo/config.toml` scopes
   `target-cpu=x86-64-v3` to `x86_64-unknown-linux-gnu`, a target a macOS host
   never selects — so a macOS build exercises none of it, and the wrong-cwd
   failure mode that cost morpheus a 49x pessimisation is invisible there.
3. **The 2 GB cap at build time.** `lto = "fat"` with `codegen-units = 1` over
   this graph on one core has an unmeasured peak RSS and an unmeasured wall
   time; RULES.md §08 sets 2 GB per bot and documents no intake timeout. This
   is the only place either can be observed, which is packaging.md's risk P2.
4. **A different toolchain installation than the one that vendored.**

What it cannot tell us: whether generals.bot's own image agrees. Modal is a
proxy, the same caveat the morpheus rehearsal carries.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
MEASUREMENTS = REPO / "docs" / "research" / "measurements"

# `joe-rs-<content_hash>.zip`, resolved rather than hardcoded: the name moves
# with the program, which is the point of putting the hash in it. A stale path
# here would smoke-test whatever the last build happened to leave behind.
#
# **Local only, and the guard is load-bearing.** Modal re-imports this module
# *inside the container* to find the function, so every module-level statement
# runs twice — once here, once there, where `arena` does not exist because only
# this file is mounted. Without the guard the container dies at import with
# `ModuleNotFoundError: No module named 'arena'` before a single line of the
# smoke runs (AGENTS.md "Modal jobs": a job that dies at import looks exactly
# like one that is working).
if modal.is_local():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from arena.bundle import default_output_path

    sys.path.insert(0, str(REPO / "bots" / "joe-rs" / "tools"))
    import importlib.util

    _tool = REPO / "bots" / "joe-rs" / "tools" / "package_submission.py"
    _spec = importlib.util.spec_from_file_location("joe_rs_package_submission", _tool)
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _module
    _spec.loader.exec_module(_module)

    SUBMISSION_ZIP = default_output_path("joe-rs")
    # The packager's own frames, passed to the container as an argument rather
    # than copied into this file. A second hand-written copy of a 21x21 board
    # would drift from the one the smoke actually enforces.
    SMOKE_INPUT = _module.SMOKE_INPUT
else:
    SUBMISSION_ZIP = Path("/root/joe-rs-submission.zip")
    SMOKE_INPUT = ""

# Rust installed at *image build* time, where a network is allowed — the
# judge's sandbox likewise has a toolchain preinstalled and no network at
# intake. The version tracks the sandbox's 1.97 stable.
IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "build-essential", "unzip")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
        "| sh -s -- -y --profile minimal --default-toolchain 1.97.1"
    )
    .env({"PATH": "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin"})
    .add_local_file(str(SUBMISSION_ZIP), "/root/joe-rs-submission.zip", copy=True)
)

app = modal.App("joe-rs-sandbox-smoke")


def _memory_peak_bytes() -> int | None:
    """The cgroup's own high-water mark, when the sandbox exposes one.

    `ru_maxrss` is the largest *single* child, and a parallel cargo build runs
    several rustc processes at once — so the cgroup total is the number that
    decides an OOM kill. Under gVisor these files are often absent, which is
    why the report falls back to `ru_maxrss`: with `lto = "fat"` and
    `codegen-units = 1` the peak is the final single-threaded link, so the
    largest child is a good estimate of the whole.
    """
    for path in (
        "/sys/fs/cgroup/memory.peak",
        "/sys/fs/cgroup/memory/memory.max_usage_in_bytes",
        "/sys/fs/cgroup/memory/memory.memsw.max_usage_in_bytes",
    ):
        try:
            return int(Path(path).read_text().strip())
        except (OSError, ValueError):
            continue
    return None


def _cpu_quota() -> str:
    """What the container is actually allowed, not what `os.cpu_count()` says.

    `os.cpu_count()` reports the *host's* processors through gVisor, so a
    `cpu=1` function still sees a couple of dozen. A build timed under that is
    not the one-core number RULES.md §08 describes, which is why the report
    below carries a deliberate `-j1` build beside the default one.
    """
    try:
        raw = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        return "unlimited" if raw[0] == "max" else f"{int(raw[0]) / int(raw[1]):.2f} cpu"
    except (OSError, ValueError, IndexError):
        pass
    try:
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text().strip())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text().strip())
        return "unlimited" if quota < 0 else f"{quota / period:.2f} cpu"
    except (OSError, ValueError):
        return "unknown"


@app.function(image=IMAGE, cpu=1, memory=2048, timeout=60 * 45, block_network=True)
def smoke_sandbox(smoke_input: str) -> dict:
    """The submission zip, one core, 2 GB, no network."""
    import os
    import platform
    import resource
    import shutil
    import subprocess
    import time
    import zipfile

    host = {
        "kernel": platform.release(),
        "arch": platform.machine(),
        # Reported side by side on purpose: the first is the host's count
        # leaking through gVisor, the second is what this container may use.
        "online_cpus_reported": os.cpu_count(),
        "cpu_quota": _cpu_quota(),
        "memory_limit_mb": 2048,
        "network_blocked": True,
    }
    try:
        host["rustc"] = subprocess.run(
            ["rustc", "--version"], capture_output=True, text=True
        ).stdout.strip()
    except OSError as exc:
        host["rustc"] = f"<unavailable: {exc}>"

    # Proof that block_network means what it says, rather than a claim in a
    # docstring: a fetch that would succeed on any connected host.
    probe = subprocess.run(
        ["curl", "-sS", "--max-time", "8", "https://static.crates.io/"],
        capture_output=True,
        text=True,
    )
    host["network_probe_returncode"] = probe.returncode
    host["network_probe_stderr"] = probe.stderr.strip()[:300]

    root = "/tmp/submission"
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root, exist_ok=True)
    with zipfile.ZipFile("/root/joe-rs-submission.zip") as zf:
        zf.extractall(root)
        infos = zf.infolist()
    for name in ("run.sh", "build.sh"):
        path = os.path.join(root, name)
        if os.path.exists(path):
            os.chmod(path, 0o755)

    result: dict = {
        "variant": "submission",
        "files": len(infos),
        "unpacked_bytes": sum(i.file_size for i in infos),
        "zip_bytes": os.path.getsize("/root/joe-rs-submission.zip"),
        "vendored_files": sum(1 for i in infos if i.filename.startswith("vendor/")),
    }

    def _build(where: str, jobs: int | None) -> dict:
        """One intake build, timed and measured for peak memory.

        `jobs=1` is the conservative arm of P2. `os.cpu_count()` leaks the
        host's processor count through gVisor, so cargo's default parallelism
        is whatever the host offers rather than the one core RULES.md §08
        describes — and a wall time measured that way would understate the
        judge's by a large factor. `CARGO_BUILD_JOBS=1` puts a real upper
        bound under it, which is the number the tripwire should be read
        against.
        """
        env = dict(os.environ)
        if jobs is not None:
            env["CARGO_BUILD_JOBS"] = str(jobs)
        before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024
        t0 = time.perf_counter()
        proc = subprocess.run(
            ["bash", os.path.join(where, "build.sh")],
            cwd=where,
            capture_output=True,
            text=True,
            env=env,
        )
        after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024
        return {
            "jobs": jobs or "default",
            "returncode": proc.returncode,
            "seconds": round(time.perf_counter() - t0, 2),
            # `lto = "fat"` with `codegen-units = 1` ends in one single-threaded
            # link, so the largest child is where the peak lives. An OOM kill
            # would show as a non-zero return code with a truncated link step.
            "child_maxrss_bytes": max(before, after),
            "cgroup_peak_bytes": _memory_peak_bytes(),
            "stdout_tail": proc.stdout[-1500:],
            "stderr_tail": proc.stderr[-1500:],
        }

    # The default-parallelism build first, because it is the one whose binary
    # gets run below; then a cold `-j1` rebuild in a second extract for P2's
    # bound. Two extracts rather than a `cargo clean`, so the second build
    # starts from exactly the state the judge would.
    builds = [_build(root, None)]
    build_default = builds[0]
    result["builds"] = builds
    result["build_returncode"] = build_default["returncode"]
    result["build_seconds"] = build_default["seconds"]
    result["build_stdout_tail"] = build_default["stdout_tail"]
    result["build_stderr_tail"] = build_default["stderr_tail"]
    result["build_child_maxrss_bytes"] = build_default["child_maxrss_bytes"]
    result["cgroup_peak_bytes"] = build_default["cgroup_peak_bytes"]
    if build_default["returncode"] != 0:
        result["ok"] = False
        return {"host": host, "results": [result]}

    single = "/tmp/submission-j1"
    shutil.rmtree(single, ignore_errors=True)
    with zipfile.ZipFile("/root/joe-rs-submission.zip") as zf:
        zf.extractall(single)
    for name in ("run.sh", "build.sh"):
        os.chmod(os.path.join(single, name), 0o755)
    builds.append(_build(single, 1))
    shutil.rmtree(single, ignore_errors=True)

    play = subprocess.run(
        ["bash", os.path.join(root, "run.sh")],
        cwd=root,
        input=smoke_input,
        capture_output=True,
        text=True,
        timeout=300,
    )
    lines = [line for line in play.stdout.strip().splitlines() if line.strip()]
    result["reply_lines"] = lines
    result["run_returncode"] = play.returncode
    result["run_stderr_tail"] = play.stderr[-1500:]

    # The same two questions the packager's smoke asks, for the same reasons.
    # A bundle that cannot find its artifact exits 1 and replies nothing; a
    # bundle that runs but decides nothing replies `1 0 0 0 0` to frames where
    # passing is the wrong answer.
    well_formed = len(lines) == 2 and all(len(line.split()) == 5 for line in lines)
    result["all_pass"] = bool(lines) and all(line.split()[0] == "1" for line in lines)
    result["ok"] = play.returncode == 0 and well_formed and not result["all_pass"]

    # The exe-relative fallback in `main.rs::artifact_dir`, exercised on the
    # judge's own shape: the generated launcher exports JOE_RS_ARTIFACT, so
    # this runs the binary directly to reach the branch the launcher hides.
    env = {k: v for k, v in os.environ.items() if k != "JOE_RS_ARTIFACT"}
    fallback = subprocess.run(
        [os.path.join(root, "target", "release", "joe-rs")],
        cwd="/",
        input=smoke_input,
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    fallback_lines = [ln for ln in fallback.stdout.strip().splitlines() if ln.strip()]
    result["fallback_returncode"] = fallback.returncode
    result["fallback_reply_lines"] = fallback_lines
    result["fallback_ok"] = fallback.returncode == 0 and len(fallback_lines) == 2

    return {"host": host, "results": [result]}


def _mib(value: int | None) -> str:
    return "—" if value is None else f"{value / 2**20:.0f} MiB"


def _markdown(report: dict) -> str:
    host = report["host"]
    r = report["results"][0]
    blocked = host["network_probe_returncode"] != 0
    lines = [
        "# joe-rs — submission rehearsal on Linux x86, offline",
        "",
        "Milestone J5.4 of [the packaging plan](../../bots/joe-rs/packaging.md).",
        "joe-rs ships 93 vendored crates, so unlike the morpheus rehearsal there is",
        "no separate `vendor-probe` variant: the submitted archive is itself the",
        "proof that offline source replacement resolves a real graph.",
        "",
        f"- Container: `@app.function(cpu=1, memory=2048, block_network=True)`, "
        f"{host['arch']}, kernel {host['kernel']}",
        f"- CPU: quota `{host.get('cpu_quota', '?')}`, but `os.cpu_count()` reports "
        f"{host.get('online_cpus_reported', '?')} — the host's count leaking through "
        "gVisor, which is why the `-j1` row below exists",
        f"- Toolchain: `{host.get('rustc', '?')}`",
        f"- Network probe (curl to crates.io): exit {host['network_probe_returncode']}"
        + (
            " — blocked, as intended"
            if blocked
            else " — **reachable, so this run does not prove the offline path**"
        ),
        "",
        "| verdict | files | vendored | unpacked | zip |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| {'ok' if r.get('ok') else 'FAILED'} | {r['files']} | {r['vendored_files']} | "
        f"{r['unpacked_bytes']} B | {r['zip_bytes']} B |",
        "",
        "## P2 — the intake build under one core and 2 GB",
        "",
        "`lto = \"fat\"` with `codegen-units = 1` links the whole graph in one rustc",
        "invocation, so the memory peak lands at the end of the build and an OOM kill",
        "would appear as a failed build with a truncated link step. The tripwire in",
        "packaging.md is an OOM kill or a build over 15 minutes.",
        "",
        "Two builds, because cargo's default parallelism here is the host's and not",
        "the judge's. The `-j1` row is the conservative bound and the one the",
        "tripwire should be read against.",
        "",
        "| cargo jobs | exit | wall s | peak RSS (largest child) | cgroup peak |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for b in r.get("builds", []):
        lines.append(
            f"| {b['jobs']} | {b['returncode']} | {b['seconds']} | "
            f"{_mib(b.get('child_maxrss_bytes'))} | {_mib(b.get('cgroup_peak_bytes'))} |"
        )
    lines += [
        "",
        "`cgroup peak` is blank when the sandbox does not expose the counter, which",
        "gVisor generally does not. The RSS column is the largest single child rather",
        "than the cgroup total — for a fat-LTO build those nearly coincide, because",
        "the last link is one process and the biggest one.",
        "",
    ]
    slowest = max((b["seconds"] for b in r.get("builds", [])), default=0)
    if slowest > 900:
        lines += [
            "**The build exceeded 15 minutes.** P2's fallback is `lto = \"thin\"`, which",
            "changes the shipped binary — so the latency figure must be re-measured",
            "before it ships.",
            "",
        ]
    lines += [
        "## What the bundle replied",
        "",
        "Two 21x21 frames where our general sits on plentiful army with four empty",
        "plains around it. A pass is the wrong answer there, which is what makes",
        "the replies evidence rather than a formality: a joe-rs seat that cannot",
        "load its artifact exits 1 and answers nothing, and one that runs without",
        "deciding answers `1 0 0 0 0`.",
        "",
        "| path | exit | replies |",
        "| --- | ---: | --- |",
        f"| generated `run.sh` (exports `JOE_RS_ARTIFACT`) | {r.get('run_returncode', '?')} | "
        + (" / ".join(f"`{ln}`" for ln in r.get("reply_lines", [])) or "—")
        + " |",
        f"| binary directly, variable unset, cwd `/` | {r.get('fallback_returncode', '?')} | "
        + (" / ".join(f"`{ln}`" for ln in r.get("fallback_reply_lines", [])) or "—")
        + " |",
        "",
        "The second row is the exe-relative branch of `main.rs::artifact_dir`, which",
        "the launcher normally hides. Both paths resolving is why the `export` line",
        "is insurance rather than a dependency.",
        "",
    ]
    if not r.get("ok"):
        lines += [
            "## Failure",
            "",
            "```",
            (r.get("build_stderr_tail") or r.get("run_stderr_tail") or "").strip(),
            "```",
            "",
        ]
    return "\n".join(lines) + "\n"


@app.local_entrypoint()
def main(output: str = "") -> None:
    report = smoke_sandbox.remote(SMOKE_INPUT)
    out_json = Path(output) if output else MEASUREMENTS / "joe-rs-sandbox-smoke.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    out_json.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    print(f"wrote {out_json}")
    print(f"wrote {out_json.with_suffix('.md')}")
    r = report["results"][0]
    for b in r.get("builds", []):
        print(
            f"build jobs={b['jobs']}: exit={b['returncode']} {b['seconds']}s "
            f"peak={_mib(b.get('child_maxrss_bytes'))}"
        )
    print(f"submission: ok={r.get('ok')} replies={r.get('reply_lines')}")
    if not r.get("ok"):
        raise SystemExit(1)
