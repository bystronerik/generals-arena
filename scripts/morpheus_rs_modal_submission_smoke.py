#!/usr/bin/env python3
"""M0.5: run both morpheus-rs submission zips on Linux x86, with no network.

    python bots/morpheus-rs/tools/package_submission.py --force
    modal run scripts/morpheus_rs_modal_submission_smoke.py

Writes docs/research/measurements/morpheus-rs-sandbox-smoke.{json,md}.

Milestone M0.5 of docs/bots/morpheus-rs/rewrite-plan.md is "submit a trivial
Rust bot and see whether the sandbox builds and runs it". Only the account
holder can submit to generals.bot, so this is the rehearsal that can be
automated — and it is the part that actually carries information, because the
things M0.5 is trying to falsify are all environmental:

- does `build.sh` work with **no network at all** (`block_network=True`, which
  is stricter than `cargo --offline` asserting it needs none),
- does the vendored tree build on a *different* toolchain installation than the
  one that packaged it,
- does the static musl binary run on x86-64 Linux, which the arm64 macOS
  packaging host can build but cannot execute,
- do both stay inside the file-count and size limits after extraction.

What it cannot tell us: whether generals.bot's own image agrees. Modal is a
proxy — same caveat as the M0 CPU probe, and the reason R4's fallback variant
exists at all.
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
BUNDLES = REPO / "data" / "bundles"
MEASUREMENTS = REPO / "docs" / "research" / "measurements"

VENDORED_ZIP = BUNDLES / "morpheus-rs-vendored.zip"
STATIC_ZIP = BUNDLES / "morpheus-rs-static.zip"

# Rust installed at *image build* time, where network is allowed — the judge's
# sandbox likewise has a toolchain preinstalled and no network at intake. The
# version tracks the sandbox's 1.97 stable (rewrite-plan §1).
IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "build-essential", "unzip")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
        "| sh -s -- -y --profile minimal --default-toolchain 1.97.1"
    )
    .env({"PATH": "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin"})
    .add_local_file(str(VENDORED_ZIP), "/root/morpheus-rs-vendored.zip", copy=True)
    .add_local_file(str(STATIC_ZIP), "/root/morpheus-rs-static.zip", copy=True)
)

app = modal.App("morpheus-rs-sandbox-smoke")

# One handshake and two frames, same script the packager uses locally.
SMOKE_INPUT = "0 2 2\n" + ("1 1 1 1 1\n1 1\n1 1\n1 0\n0 0\n1 0\n0 0\n" * 2)


def _smoke_one(zip_path: str, variant: str) -> dict:
    import os
    import shutil
    import subprocess
    import time
    import zipfile

    root = f"/tmp/{variant}"
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)
        infos = zf.infolist()

    for name in ("run.sh", "build.sh"):
        path = os.path.join(root, name)
        if os.path.exists(path):
            os.chmod(path, 0o755)

    result: dict = {
        "variant": variant,
        "files": len(infos),
        "unpacked_bytes": sum(i.file_size for i in infos),
        "zip_bytes": os.path.getsize(zip_path),
    }

    t0 = time.perf_counter()
    build = subprocess.run(
        ["bash", os.path.join(root, "build.sh")],
        cwd=root,
        capture_output=True,
        text=True,
    )
    result["build_returncode"] = build.returncode
    result["build_seconds"] = round(time.perf_counter() - t0, 2)
    result["build_stderr_tail"] = build.stderr[-1500:]
    if build.returncode != 0:
        result["ok"] = False
        return result

    play = subprocess.run(
        ["bash", os.path.join(root, "run.sh")],
        cwd=root,
        input=SMOKE_INPUT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    lines = [line for line in play.stdout.strip().splitlines() if line.strip()]
    result["reply_lines"] = lines
    result["run_returncode"] = play.returncode
    result["run_stderr_tail"] = play.stderr[-1500:]
    result["ok"] = (
        play.returncode == 0
        and len(lines) == 2
        and all(len(line.split()) == 5 for line in lines)
    )
    return result


@app.function(image=IMAGE, cpu=1, memory=2048, timeout=60 * 20, block_network=True)
def smoke_sandbox() -> dict:
    """Both variants, one core, no network — as close to intake as we get."""
    import os
    import platform
    import subprocess

    facts = {
        "kernel": platform.release(),
        "arch": platform.machine(),
        "online_cpus": os.cpu_count(),
        "network_blocked": True,
    }
    try:
        facts["rustc"] = subprocess.run(
            ["rustc", "--version"], capture_output=True, text=True
        ).stdout.strip()
    except OSError as exc:
        facts["rustc"] = f"<unavailable: {exc}>"

    # Proof that block_network means what it says, rather than a claim in a
    # docstring: a fetch that would succeed on any connected host.
    probe = subprocess.run(
        ["curl", "-sS", "--max-time", "8", "https://static.crates.io/"],
        capture_output=True,
        text=True,
    )
    facts["network_probe_returncode"] = probe.returncode
    facts["network_probe_stderr"] = probe.stderr.strip()[:300]

    return {
        "host": facts,
        "results": [
            _smoke_one("/root/morpheus-rs-vendored.zip", "vendored"),
            _smoke_one("/root/morpheus-rs-static.zip", "static"),
        ],
    }


def _markdown(report: dict) -> str:
    host = report["host"]
    lines = [
        "# Morpheus-rs M0.5 — submission rehearsal on Linux x86, offline",
        "",
        "Milestone M0.5 of [the rewrite plan](../../bots/morpheus-rs/rewrite-plan.md):",
        "prove the offline build path, the file-count budget, and binary",
        "compatibility before any real porting starts.",
        "",
        f"- Container: `@app.function(cpu=1, block_network=True)`, "
        f"{host['arch']}, kernel {host['kernel']}, {host['online_cpus']} cpus visible",
        f"- Toolchain: `{host.get('rustc', '?')}`",
        f"- Network probe (curl to crates.io): exit {host['network_probe_returncode']}"
        + (" — blocked, as intended" if host["network_probe_returncode"] != 0 else
           " — **reachable, so this run does not prove the offline path**"),
        "",
        "Modal stands in for the generals.bot sandbox, which cannot be probed.",
        "A green result here is necessary, not sufficient — which is why R4's",
        "static-binary fallback is built and smoked on every packaging run.",
        "",
        "| variant | build | build s | files | unpacked | zip | replies |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in report["results"]:
        replies = " / ".join(f"`{line}`" for line in r.get("reply_lines", [])) or "—"
        lines.append(
            f"| {r['variant']} | {'ok' if r.get('ok') else 'FAILED'} | "
            f"{r.get('build_seconds', 0)} | {r['files']} | "
            f"{r['unpacked_bytes']} B | {r['zip_bytes']} B | {replies} |"
        )
    failed = [r for r in report["results"] if not r.get("ok")]
    lines += [
        "",
        "Both variants build with no network and answer the protocol."
        if not failed
        else "**Failures:** " + ", ".join(r["variant"] for r in failed) + ".",
        "",
        "The judge's limits are 50 MB zipped, 512 MB unpacked, 10,000 files. The",
        "file count is the binding one for a Rust bot (`cargo vendor` over a fat",
        "graph clears it easily), and a zero-dependency crate vendors to nothing —",
        "which is why the dependency budget is reviewed at every `cargo add`.",
        "",
    ]
    for r in report["results"]:
        if not r.get("ok"):
            lines += [
                f"### {r['variant']} failure",
                "",
                "```",
                (r.get("build_stderr_tail") or r.get("run_stderr_tail") or "").strip(),
                "```",
                "",
            ]
    return "\n".join(lines) + "\n"


@app.local_entrypoint()
def main(output: str = "") -> None:
    report = smoke_sandbox.remote()
    out_json = (
        Path(output) if output else MEASUREMENTS / "morpheus-rs-sandbox-smoke.json"
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    out_json.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    print(f"wrote {out_json}")
    print(f"wrote {out_json.with_suffix('.md')}")
    for r in report["results"]:
        print(f"{r['variant']}: ok={r.get('ok')} replies={r.get('reply_lines')}")
    if any(not r.get("ok") for r in report["results"]):
        raise SystemExit(1)
