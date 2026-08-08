#!/usr/bin/env python3
"""M0 CPU-feature probe: what x86 does a one-core Linux container actually have?

    modal run scripts/morpheus_rs_modal_cpu_probe.py
    modal run scripts/morpheus_rs_modal_cpu_probe.py --samples 8

Writes docs/research/measurements/morpheus-rs-cpu-probe.{json,md}.

Milestone M0 of docs/bots/morpheus-rs/rewrite-plan.md. The Rust binary's
`target-cpu` has to be chosen before any SIMD work starts, and choosing it
wrong is a crash on the judge's host, not a slowdown. So: sample several
one-core containers, read `/proc/cpuinfo` and `lscpu`, derive the x86-64
microarchitecture level each host supports, and set `target-cpu` to the
highest level **every** observed host clears.

**Modal is a proxy, not the target.** The generals.bot sandbox cannot be probed
— no network, no visible logs — so this measures "a one-core x86 Linux server
container" and nothing stronger. Two consequences the plan (§9, R4) fixes here
rather than later: the compile target stays conservative, and any hand-written
SIMD path does runtime feature detection instead of trusting the compile flag.

Modal's own SDKs are Python/Go/JS only, so this is a Python probe for a Rust
decision; it follows the repo's other `scripts/morpheus_modal_*.py` entry
points. Nothing here builds or runs Rust — that starts at M0.5.
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
MEASUREMENTS = REPO / "docs" / "research" / "measurements"

# The x86-64 psABI microarchitecture levels, as the feature sets `rustc`'s
# `-C target-cpu=x86-64-v{2,3,4}` and gcc's `-march=x86-64-v{2,3,4}` require.
# Names are the /proc/cpuinfo spellings, which differ from the ISA names
# (`pni` is SSE3, `sse4_1`/`sse4_2` carry underscores, `f16c` is `f16c`).
X86_64_LEVELS: dict[str, tuple[str, ...]] = {
    "x86-64-v2": ("pni", "sse4_1", "sse4_2", "ssse3", "popcnt", "cx16"),
    "x86-64-v3": (
        "avx",
        "avx2",
        "bmi1",
        "bmi2",
        "f16c",
        "fma",
        "abm",
        "movbe",
        "xsave",
    ),
    "x86-64-v4": ("avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl"),
}

# Features worth calling out on their own, whatever level they imply: these are
# the ones a hand-written kernel would branch on.
NOTABLE = ("avx", "avx2", "avx512f", "fma", "f16c", "avx_vnni", "amx_tile")

IMAGE = modal.Image.debian_slim(python_version="3.12").apt_install("util-linux")

app = modal.App("morpheus-rs-cpu-probe")


def _read(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"<unreadable: {exc}>"


def _cpuinfo_fields(text: str) -> tuple[dict[str, str], set[str]]:
    """First processor block's key/value pairs, plus its flag set."""
    fields: dict[str, str] = {}
    flags: set[str] = set()
    for line in text.splitlines():
        if not line.strip():
            if fields:  # first block is enough; the rest repeat
                break
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key == "flags":
            flags = set(value.split())
        elif key and key not in fields:
            fields[key] = value
    return fields, flags


def derive_level(flags: set[str]) -> tuple[str, dict[str, list[str]]]:
    """
    The highest x86-64 level these flags satisfy, plus what is missing above it.

    Levels are cumulative, so the walk stops at the first unmet level: a host
    with AVX-512 but no BMI2 is not v4, it is v1, and reporting it as v4 would
    produce a binary that dies on an illegal instruction.
    """
    missing: dict[str, list[str]] = {}
    level = "x86-64"
    for name in ("x86-64-v2", "x86-64-v3", "x86-64-v4"):
        absent = sorted(f for f in X86_64_LEVELS[name] if f not in flags)
        missing[name] = absent
        if absent:
            break
        level = name
    return level, missing


@app.function(image=IMAGE, cpu=1, memory=2048, timeout=60 * 5)
def probe_one(index: int) -> dict:
    """One single-core container's CPU identity. `cpu=1` mirrors the judge."""
    import os
    import platform
    import subprocess
    import time

    cpuinfo = _read("/proc/cpuinfo")
    fields, flags = _cpuinfo_fields(cpuinfo)
    level, missing = derive_level(flags)

    def _run(*command: str) -> str:
        try:
            out = subprocess.run(
                command, capture_output=True, text=True, timeout=30
            )
            return (out.stdout or out.stderr).strip()
        except (OSError, subprocess.SubprocessError) as exc:
            return f"<unavailable: {exc}>"

    return {
        "index": index,
        "sampled_at": time.time(),
        "hostname": platform.node(),
        "kernel": platform.release(),
        "arch": platform.machine(),
        "model_name": fields.get("model name", ""),
        "vendor_id": fields.get("vendor_id", ""),
        "cpu_family": fields.get("cpu family", ""),
        "model": fields.get("model", ""),
        "stepping": fields.get("stepping", ""),
        "cpu_mhz": fields.get("cpu MHz", ""),
        "cache_size": fields.get("cache size", ""),
        # `cpu=1` is a scheduling reservation, not an isolation guarantee, so
        # both numbers are recorded: what the container is allowed to use and
        # what the host has.
        "online_cpus": os.cpu_count(),
        "affinity_cpus": len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "x86_64_level": level,
        "missing_for_next_level": missing,
        "notable_flags": {name: (name in flags) for name in NOTABLE},
        "flags": sorted(flags),
        "lscpu": _run("lscpu"),
        "lscpu_caches": _run("lscpu", "-C"),
    }


def _consensus(samples: list[dict]) -> dict:
    """
    The compile target: the highest level *every* sampled host supports.

    The minimum, not the mode. One host without AVX2 in a fleet of twenty is
    not noise to be averaged away — it is the host that would crash.
    """
    order = ["x86-64", "x86-64-v2", "x86-64-v3", "x86-64-v4"]
    levels = [s["x86_64_level"] for s in samples]
    floor = min(levels, key=order.index) if levels else "x86-64"
    models = sorted({s["model_name"] for s in samples if s["model_name"]})
    common = set.intersection(*(set(s["flags"]) for s in samples)) if samples else set()
    varies = sorted(
        set.union(*(set(s["flags"]) for s in samples)) - common
    ) if samples else []
    # How many *distinct CPUs* the run actually saw, which is not the same as
    # how many containers it launched. Six containers on one fleet generation
    # is one observation repeated six times; reporting it as six would make a
    # narrow sample look like evidence of fleet-wide uniformity.
    identities = sorted(
        {
            "{vendor_id} family {cpu_family} model {model} stepping {stepping}".format(
                **{k: s.get(k) or "?" for k in
                   ("vendor_id", "cpu_family", "model", "stepping")}
            )
            for s in samples
        }
    )
    return {
        "samples": len(samples),
        "levels_observed": sorted(set(levels), key=order.index),
        "target_cpu": floor,
        "unanimous": len(set(levels)) <= 1,
        "cpu_models": models,
        "distinct_cpu_identities": identities,
        "flags_varying_across_hosts": varies,
        "notable_flags_everywhere": sorted(f for f in NOTABLE if f in common),
    }


def _identity(sample: dict) -> str:
    return "{vendor_id} family {cpu_family} model {model} stepping {stepping}".format(
        **{
            k: sample.get(k) or "?"
            for k in ("vendor_id", "cpu_family", "model", "stepping")
        }
    )


def _markdown(report: dict) -> str:
    from collections import Counter

    c = report["consensus"]
    runs = report.get("runs") or [
        {"sampled_at": report["sampled_at"], "samples": report["samples"]}
    ]
    lines = [
        "# Morpheus-rs M0 CPU-feature probe (Modal, one core)",
        "",
        "Milestone M0 of [the rewrite plan](../../bots/morpheus-rs/rewrite-plan.md).",
        "Chooses the Rust binary's `target-cpu`.",
        "",
        f"- Last sampled: {report['sampled_at']}",
        f"- Containers: {c['samples']} × `@app.function(cpu=1)`, "
        f"over {len(runs)} run(s)",
        f"- **Compile target: `{c['target_cpu']}`**"
        + (
            ""
            if c["unanimous"]
            else " — the *floor*: hosts at a higher level were also seen"
        ),
        "",
        "## Why this is a proxy",
        "",
        "The generals.bot sandbox has no network and no visible logs, so it "
        "cannot be probed directly. Modal stands in for \"a one-core x86 Linux "
        "server container\" and nothing stronger. The compile target is "
        "therefore the floor of what was observed, and any hand-written SIMD "
        "path must do runtime feature detection rather than trust the flag "
        "(rewrite-plan §9, R4).",
        "",
        "## Hosts seen",
        "",
        "| cpu identity | level | containers | cache | avx512f |",
        "| --- | --- | ---: | --- | --- |",
    ]
    every = [s for run in runs for s in run["samples"]]
    counts = Counter(_identity(s) for s in every)
    first: dict[str, dict] = {}
    for s in every:
        first.setdefault(_identity(s), s)
    for identity, count in counts.most_common():
        s = first[identity]
        lines.append(
            f"| {identity} | `{s['x86_64_level']}` | {count} | "
            f"{s['cache_size'] or '?'} | "
            f"{'yes' if s['notable_flags'].get('avx512f') else 'no'} |"
        )
    lines += [
        "",
        "### Per run",
        "",
        "| run | sampled | containers | levels |",
        "| ---: | --- | ---: | --- |",
    ]
    for n, run in enumerate(runs, start=1):
        levels = sorted({s["x86_64_level"] for s in run["samples"]})
        lines.append(
            f"| {n} | {run['sampled_at']} | {len(run['samples'])} | "
            + ", ".join(f"`{level}`" for level in levels)
            + " |"
        )
    lines += [
        "",
        f"**Distinct CPU identities: {len(counts)}** across {c['samples']} "
        "containers. "
        + (
            "One fleet generation, repeated — the sample bounds nothing about "
            "hosts these runs did not land on."
            if len(counts) <= 1
            else "Containers launched together land on the same generation, so "
            "variety comes from running the probe again later, not from asking "
            "for more containers at once."
        ),
        "",
        "Containers report more visible CPUs than the `cpu=1` reservation, and "
        "the sandbox masks the model name and cache topology, so neither is "
        "evidence about the judge's host.",
        "",
        "## Feature availability",
        "",
        "| flag | present on every sampled host |",
        "| --- | --- |",
    ]
    everywhere = set(c["notable_flags_everywhere"])
    for name in NOTABLE:
        lines.append(f"| `{name}` | {'yes' if name in everywhere else 'no'} |")
    varying = c["flags_varying_across_hosts"]
    lines += [
        "",
        (
            "No CPU flag varied across the sampled hosts."
            if not varying
            else "Flags present on some hosts and not others: "
            + ", ".join(f"`{f}`" for f in varying[:40])
            + ("…" if len(varying) > 40 else "")
            + ". A binary may not assume any of these."
        ),
        "",
    ]
    return "\n".join(lines) + "\n"


def _previous_runs(path: Path) -> list[dict]:
    """
    Earlier runs from the report this one is about to rewrite.

    Runs accumulate rather than replace, because one `modal run` samples one
    placement decision: ten containers launched together landed on ten hosts of
    the same fleet generation, and a run an hour later landed on a different
    one with a different feature set. Overwriting would let the compile target
    swing with whichever fleet answered last — exactly the mistake that ships a
    binary using instructions half the fleet lacks.
    """
    if not path.is_file():
        return []
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(previous.get("runs"), list):
        return list(previous["runs"])
    if isinstance(previous.get("samples"), list):  # pre-accumulation format
        return [
            {
                "sampled_at": previous.get("sampled_at", "unknown"),
                "samples": previous["samples"],
            }
        ]
    return []


@app.local_entrypoint()
def main(samples: int = 5, output: str = "", fresh: bool = False) -> None:
    import datetime

    out_json = Path(output) if output else MEASUREMENTS / "morpheus-rs-cpu-probe.json"
    results = sorted(probe_one.map(range(samples)), key=lambda s: s["index"])
    runs = [] if fresh else _previous_runs(out_json)
    runs.append(
        {
            "sampled_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "samples": results,
        }
    )
    every_sample = [s for run in runs for s in run["samples"]]
    report = {
        "sampled_at": runs[-1]["sampled_at"],
        "provider": "modal",
        "note": (
            "Modal is a proxy for the generals.bot sandbox, which cannot be "
            "probed. Compile target is the floor over every sampled host, "
            "across runs; SIMD paths must still detect features at runtime."
        ),
        "consensus": _consensus(every_sample),
        "runs": runs,
        "samples": results,
    }

    out_md = out_json.with_suffix(".md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(_markdown(report), encoding="utf-8")
    print(f"target_cpu={report['consensus']['target_cpu']}")
    print(f"unanimous={report['consensus']['unanimous']}")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
