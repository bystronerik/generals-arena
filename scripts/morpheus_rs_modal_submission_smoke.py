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
- does an offline build work when there is genuinely something to resolve —
  the two shipping variants have no dependencies, so `--offline` cannot fail
  for them; the `vendor-probe` zip carries a transitive graph and a build
  script precisely to close that gap,
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
# Built by tools/vendor_probe.py. Carries a real transitive dependency graph
# with a build script, which the two shipping variants deliberately do not —
# so this is the only zip here that actually exercises offline registry
# resolution. See the probe's module docstring for why that gap mattered.
PROBE_ZIP = BUNDLES / "morpheus-rs-vendor-probe.zip"

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
    .add_local_file(str(PROBE_ZIP), "/root/morpheus-rs-vendor-probe.zip", copy=True)
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

    # `build.sh` runs `morpheus-rs selfcheck` for the two shipping variants
    # (M8). Its key/value lines are the only x86 evidence that the submitted
    # bytes are a working bot rather than a well-formed one: a seat that cannot
    # load its weights, its knobs, or a hardware FMA still answers every frame
    # with a legal skip, which is what this smoke used to score as a pass.
    facts = {}
    for line in build.stdout.splitlines():
        head, _, tail = line.partition(" ")
        if head in {
            "hardware_fma",
            "weights_sha256",
            "checkpoint_id",
            "load_ms",
            "warmup_ms",
            "init_ms",
            "decision",
            "decide_ms",
            "selfcheck",
        }:
            facts[head] = tail.strip()
    result["selfcheck"] = facts
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
    # One selfcheck is a check; five are a measurement. The first x86 run put
    # the static variant's decision at 11.0 ms against the vendored variant's
    # 4.7 ms on the same host — musl's allocator against glibc's, plausibly,
    # or a cold sample. Either it is real and R4's fallback does not inherit
    # M7's latency qualification, or it is noise; asserting either from one
    # sample is what this project keeps writing down as the mistake.
    if variant in {"vendored", "static"}:
        repeats = []
        for _ in range(4):
            again = subprocess.run(
                ["bash", os.path.join(root, "build.sh")],
                cwd=root,
                capture_output=True,
                text=True,
            )
            sample = {}
            for line in again.stdout.splitlines():
                head, _, tail = line.partition(" ")
                if head in {"warmup_ms", "decide_ms", "load_ms"}:
                    sample[head] = float(tail)
            if sample:
                repeats.append(sample)
        result["selfcheck_repeats"] = repeats

    spoke_protocol = (
        play.returncode == 0
        and len(lines) == 2
        and all(len(line.split()) == 5 for line in lines)
    )
    # `vendor-probe` is a scratch crate, not this bot; it has no selfcheck to
    # pass. The two shipping variants must have one, and it must have passed.
    needs_selfcheck = variant in {"vendored", "static"}
    result["ok"] = spoke_protocol and (
        not needs_selfcheck or facts.get("selfcheck") == "ok"
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
            _smoke_one("/root/morpheus-rs-vendor-probe.zip", "vendor-probe"),
        ],
    }


def _markdown(report: dict) -> str:
    host = report["host"]
    lines = [
        "# Morpheus-rs — submission rehearsal on Linux x86, offline",
        "",
        "Milestone M0.5 of [the rewrite plan](../../bots/morpheus-rs/rewrite-plan.md)",
        "proved the offline build path, the file-count budget, and binary",
        "compatibility before any real porting started. Re-run at **M8** against the",
        "finished bot, where the interesting column is no longer *replies* but",
        "*selfcheck*: a seat that cannot load its weights, its knobs, or a hardware",
        "FMA still answers every frame with a well-formed skip, so a green",
        "replies column proved nothing about this bot at all.",
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
        "| variant | verdict | build s | files | unpacked | zip | replies |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in report["results"]:
        replies = " / ".join(f"`{line}`" for line in r.get("reply_lines", [])) or "—"
        lines.append(
            f"| {r['variant']} | {'ok' if r.get('ok') else 'FAILED'} | "
            f"{r.get('build_seconds', 0)} | {r['files']} | "
            f"{r['unpacked_bytes']} B | {r['zip_bytes']} B | {replies} |"
        )

    checked = [r for r in report["results"] if r.get("selfcheck")]
    if checked:
        lines += [
            "",
            "## What the binary says about itself, on x86",
            "",
            "`build.sh` runs `morpheus-rs selfcheck` at intake and aborts on a",
            "non-zero exit. It loads the weights against their manifest digest,",
            "resolves `deployment.json` rather than falling back to the placeholder",
            "knobs, constructs the playing seat with its warmup and thread-count",
            "invariant, and decides one hand-built frame — a general on thirteen",
            "army with four empty neighbours, where a skip is the only wrong answer.",
            "",
            "`warmup_ms` is 26 forwards, so it doubles as a per-forward probe.",
            "Divide the column by 26: a build that lost its FMA would read fifty",
            "times what it does, which is the failure this whole check exists for.",
            "",
            "| variant | fma | load ms | warmup ms | decision | decide ms | verdict |",
            "| --- | --- | ---: | ---: | --- | ---: | --- |",
        ]
        for r in checked:
            f = r["selfcheck"]
            lines.append(
                f"| {r['variant']} | {f.get('hardware_fma', '?')} | "
                f"{f.get('load_ms', '?')} | {f.get('warmup_ms', '?')} | "
                f"`{f.get('decision', '?')}` | {f.get('decide_ms', '?')} | "
                f"{f.get('selfcheck', 'absent')} |"
            )

        repeated = [r for r in report["results"] if r.get("selfcheck_repeats")]
        if repeated:
            lines += [
                "",
                "Five runs each, because one is a check and not a measurement —",
                "the two variants differ by their libc, and the fallback inheriting",
                "M7's latency qualification is an assumption until it is measured.",
                "",
                "| variant | warmup ms (5 runs) | decide ms (5 runs) |",
                "| --- | --- | --- |",
            ]
            for r in repeated:
                first = r["selfcheck"]
                warm = [float(first["warmup_ms"])] + [
                    s["warmup_ms"] for s in r["selfcheck_repeats"] if "warmup_ms" in s
                ]
                dec = [float(first["decide_ms"])] + [
                    s["decide_ms"] for s in r["selfcheck_repeats"] if "decide_ms" in s
                ]
                lines.append(
                    f"| {r['variant']} | "
                    + ", ".join(f"{v:.1f}" for v in warm)
                    + " | "
                    + ", ".join(f"{v:.1f}" for v in dec)
                    + " |"
                )
    failed = [r for r in report["results"] if not r.get("ok")]
    lines += [
        "",
        "Every variant builds with no network and answers the protocol."
        if not failed
        else "**Failures:** " + ", ".join(r["variant"] for r in failed) + ".",
        "",
        "`vendor-probe` is not a shipping variant. The two that are have no",
        "dependencies, so their `--offline` build proves less than it appears",
        "to — with nothing to resolve, `--offline` cannot fail. The probe adds",
        "a real transitive graph (`sha2` → `digest` → `block-buffer` →",
        "`generic-array` → `typenum`) and a build script, so registry",
        "replacement and intake-time build scripts are exercised before M3",
        "depends on ninety crates.",
        "",
        "The judge's limits are 50 MB zipped, 512 MB unpacked, 10,000 files. The",
        "file count is the binding one for a Rust bot, though not as binding as",
        "this line used to claim: M2 measured `sha2` at 565 files and candle-core",
        "at 3,888, against a cap of 10,000. A zero-dependency crate vendors to",
        "nothing, which is why the shipped variants need the probe above to",
        "exercise offline resolution at all.",
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
