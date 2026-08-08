#!/usr/bin/env python3
"""M0 x86 latency baseline: the same game suite, on a one-core Linux container.

    modal run scripts/morpheus_rs_modal_baseline.py
    modal run scripts/morpheus_rs_modal_baseline.py --games 8 --control-games 2

Writes docs/research/measurements/morpheus-rs-baseline-modal.{json,md}.

Milestone M0 of docs/bots/morpheus-rs/rewrite-plan.md, second half. The M3 Pro
numbers from `scripts/morpheus_rs_baseline.py` are the development baseline;
these are the ones that carry deployment authority, because the competition
host is one x86 core running Linux and an arm64 laptop is not a proxy for it.

Same code path as the local run: `scripts/morpheus_rs_baseline.py capture`
followed by `report`, inside `@app.function(cpu=1)`. Only two things differ,
both forced by the container:

- **No git**, so the version registry cannot append a lineage step. `capture`
  runs with `--no-register` and passes content hashes computed from the source
  closure instead, which keeps the games correctly attributed.
- **Nothing is written back** but the report. The corpus itself stays local;
  a container's captures would be discarded with the container, and the corpus
  that matters for parity is the one recorded beside the oracle's own host.

The same proxy caveat as the CPU probe applies: this is *a* one-core x86 Linux
container, not the judge's sandbox, which cannot be measured directly.
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
MEASUREMENTS = REPO / "docs" / "research" / "measurements"

# The sandbox pin set (requirements-sandbox.txt), minus what the arena's match
# path never imports. torch runs the bot; jax builds the competition board.
IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("bash", "git")
    .pip_install("numpy==2.4.6", "jax==0.11.0", "jaxlib==0.11.0")
    .pip_install("torch==2.13.0", index_url="https://download.pytorch.org/whl/cpu")
    .add_local_dir(str(REPO / "competition-module"), "/root/competition-module", copy=True)
    .add_local_dir(str(REPO / "arena"), "/root/arena", copy=True)
    .add_local_dir(str(REPO / "bots"), "/root/bots", copy=True)
    .add_local_dir(str(REPO / "scripts"), "/root/scripts", copy=True)
    .run_commands("pip install -e /root/competition-module --no-deps")
    .env({"PYTHONPATH": "/root", "PYTHONUNBUFFERED": "1"})
)

app = modal.App("morpheus-rs-modal-baseline")


@app.function(image=IMAGE, cpu=1, memory=4096, timeout=60 * 60 * 4)
def run_baseline(
    games: int, control_games: int, seed: int, panel: list[str], engine: str
) -> dict:
    """Play the suite and aggregate it, all inside the one-core container."""
    import os
    import subprocess
    import sys
    import time

    sys.path.insert(0, "/root")

    # `arena.paths.REPO_ROOT` is derived from the package location, so the
    # container layout has to look like a checkout: /root/{arena,bots,scripts}.
    from arena.paths import REPO_ROOT

    assert Path(REPO_ROOT) == Path("/root"), REPO_ROOT

    argv = [
        sys.executable,
        "/root/scripts/morpheus_rs_baseline.py",
        "capture",
        "--games",
        str(games),
        "--control-games",
        str(control_games),
        "--seed",
        str(seed),
        "--round",
        "morpheus-rs-m0-modal",
        "--no-register",
        # The image copies the submodule's files, not its git metadata, so the
        # engine SHA travels as a value rather than being re-derived from a
        # checkout that no longer knows what it is.
        "--engine-version",
        engine,
        "--panel",
        *panel,
    ]
    t0 = time.perf_counter()
    capture = subprocess.run(argv, capture_output=True, text=True)
    capture_s = time.perf_counter() - t0

    report_path = Path("/root/report.json")
    report = subprocess.run(
        [
            sys.executable,
            "/root/scripts/morpheus_rs_baseline.py",
            "report",
            "--round",
            "morpheus-rs-m0-modal",
            "--output-json",
            str(report_path),
            "--output-md",
            "/root/report.md",
        ],
        capture_output=True,
        text=True,
    )

    payload: dict = {
        "capture_returncode": capture.returncode,
        "report_returncode": report.returncode,
        "capture_wall_s": round(capture_s, 1),
        # Tails only: a full game log is megabytes and the useful part of a
        # failure is always at the end.
        "capture_stdout_tail": capture.stdout[-4000:],
        "capture_stderr_tail": capture.stderr[-4000:],
        "report_stderr_tail": report.stderr[-4000:],
        "cpuinfo_model": "",
        "online_cpus": os.cpu_count(),
    }
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                payload["cpuinfo_model"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    if report_path.is_file():
        payload["report"] = json.loads(report_path.read_text())
    return payload


def _markdown(payload: dict, report: dict) -> str:
    m = report["measured"]
    shipped = report["shipped_offline_p99_ms"]
    control = report.get("control") or {}
    lines = [
        "# Morpheus-rs M0 baseline — one x86 core (Modal)",
        "",
        "Milestone M0 of [the rewrite plan](../../bots/morpheus-rs/rewrite-plan.md).",
        "The deployment-authority half: the competition host is one x86 core "
        "running Linux, so the arm64 laptop numbers in "
        "[the local baseline](morpheus-rs-baseline.md) do not settle latency.",
        "",
        f"- Oracle: `{report['oracle']['bot_id']}@{report['oracle']['content_hash']}`",
        f"- Container: `@app.function(cpu=1)`, {payload.get('online_cpus')} cpus "
        f"visible, `{payload.get('cpuinfo_model') or 'model masked by the sandbox'}`",
        f"- Games: {m['games']} · frames: {m['frames']} "
        f"({m['normal_frames']} normal moves) · suite wall "
        f"{payload.get('capture_wall_s')} s",
        "",
        "Modal is a proxy for \"a one-core x86 Linux container\", not the "
        "generals.bot sandbox, which cannot be measured directly.",
        "",
        "## Per-move wall time",
        "",
        "| series | n | p50 | p99 | p99.9 | max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, key in (("all moves", "move_ms"), ("normal moves", "normal_move_ms")):
        s = m[key]
        lines.append(
            f"| {label} | {s['n']} | {s['p50']} | {s['p99']} | "
            f"{s['p999']} | {s['max']} |"
        )
    if control.get("games"):
        c = control["normal_move_ms"]
        lines.append(
            f"| control, uncaptured | {c['n']} | {c['p50']} | {c['p99']} | "
            f"{c['p999']} | {c['max']} |"
        )
    lines += [
        "",
        f"Normal moves over the judge's 150 ms limit: "
        f"**{m['moves_over_judge_limit']}** "
        f"({m['moves_over_judge_limit_frac'] * 100:.2f}%)"
        + (
            f"; uncaptured control "
            f"{control['moves_over_judge_limit_frac'] * 100:.2f}%."
            if control.get("games")
            else "."
        ),
        "",
        "## Per-component p99 — measured against shipped",
        "",
        "`shipped` is the `offline_p99_ms` table in "
        "`bots/morpheus/deployment.json`, calibrated on the M3 Pro for a "
        "configuration whose qualification verdict is *no*. These are the "
        "numbers M7 has to re-derive against.",
        "",
        "| component | measured p99 ms | shipped p99 ms | ratio |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, measured in m["offline_p99_ms"].items():
        was = shipped.get(name)
        if was:
            lines.append(f"| {name} | {measured:.3f} | {was:.3f} | {measured / was:.2f}x |")
        else:
            lines.append(f"| {name} | {measured:.3f} | — | — |")
    s = m["completed_simulations_normal"]
    lines += [
        "",
        "## Search throughput",
        "",
        f"Completed simulations per normal move: p50 **{s['p50']:.0f}**, "
        f"p99 {s['p99']:.0f}, max {s['max']:.0f}, mean {s['mean']:.2f}.",
        "",
        f"Belief update *and* root inference both completed on "
        f"{m['belief_plus_root_ok_frac'] * 100:.1f}% of turns; the belief "
        f"update was deferred on {m['recovery_frac'] * 100:.1f}%.",
        "",
    ]
    return "\n".join(lines) + "\n"


@app.local_entrypoint()
def main(
    games: int = 8,
    control_games: int = 2,
    seed: int = 2000,
    panel: str = "cm_expander,macaria,castle_builder,metro",
) -> None:
    import sys

    sys.path.insert(0, str(REPO))
    from arena.records.store import engine_version

    payload = run_baseline.remote(
        games,
        control_games,
        seed,
        [p for p in panel.split(",") if p],
        engine_version(),
    )
    out_json = MEASUREMENTS / "morpheus-rs-baseline-modal.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_json}")

    report = payload.get("report")
    if payload["capture_returncode"] != 0 or not report:
        print(f"capture rc={payload['capture_returncode']} report rc={payload['report_returncode']}")
        print(payload["capture_stderr_tail"][-2000:])
        raise SystemExit(1)

    out_md = out_json.with_suffix(".md")
    out_md.write_text(_markdown(payload, report), encoding="utf-8")
    m = report["measured"]
    print(f"wrote {out_md}")
    print(
        f"games={m['games']} normal_move_p99_ms={m['normal_move_ms']['p99']} "
        f"over_limit={m['moves_over_judge_limit_frac']} "
        f"sims_p50={m['completed_simulations_normal']['p50']}"
    )
