#!/usr/bin/env python3
"""Morpheus-rs M6: the Rust bot's own latency, against M0's Python baseline.

Milestone M6 of docs/bots/morpheus-rs/rewrite-plan.md, success criterion 2 and
4 (§11): per-move wall time, completed simulations per normal move, the
per-component p99 table, and the first-move load + warmup + init budget — all
measured on the *same host and the same schedule* as the M0 baseline this
compares against.

    # ~20 games, serial, traces under data/morpheus/morpheus-rs/m6/
    python scripts/morpheus_rs_m6.py play --games 20

    # aggregate into the published report
    python scripts/morpheus_rs_m6.py report

Serial on purpose. Both bots are deadline-driven, so a parallel schedule
measures the scheduler as much as the bot, and M0's baseline was serial: a
comparison against it has to be too.

The trace is the Rust bot's own, written when `MORPHEUS_RS_TRACE` is set —
`arena.instrument.runner` can only probe an `Agent` it constructs in-process,
which a subprocess binary is not. Keys mirror `bots/morpheus/probe.py`, so the
two bots' numbers are the same quantities.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from arena.records.fingerprint import bot_content_hash  # noqa: E402

MEASUREMENTS = REPO / "docs" / "research" / "measurements"
DEFAULT_ROUND = "morpheus-rs-m6"
DEFAULT_TRACES = REPO / "data" / "morpheus" / "morpheus-rs" / "m6"
BASELINE = MEASUREMENTS / "morpheus-rs-baseline.json"

# The M0 panel, unchanged: the rating anchor, two research bots, and two
# heuristics that force different phases. Changing it would compare two
# different sets of games rather than two bots.
DEFAULT_PANEL = ("cm_expander", "macaria", "sosipolis", "castle_builder", "metro")

COMPONENTS = (
    "particle_transitions",
    "leaf_batch",
    "enemy_prior_batch",
    "root_inference",
    "belief_proposal",
    "belief_tensor",
    "selection",
    "backup",
    "hashing",
    "reply",
)

JUDGE_LIMIT_MS = 150.0


def _nearest_rank(values: Sequence[float], q: float) -> float:
    """The estimator `runtime.py` deploys, so the report quotes what it acts on."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(-(-q * len(ordered) // 1))))
    return float(ordered[rank - 1])


def _stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "p50": 0.0, "p99": 0.0, "p999": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "n": len(values),
        "p50": round(_nearest_rank(values, 0.50), 3),
        "p99": round(_nearest_rank(values, 0.99), 3),
        "p999": round(_nearest_rank(values, 0.999), 3),
        "max": round(max(values), 3),
        "mean": round(statistics.fmean(values), 3),
    }


def _host_facts() -> dict[str, Any]:
    facts = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }
    try:
        facts["cpu"] = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        facts["cpu"] = platform.processor()
    return facts


def _schedule(games: int, panel: Sequence[str], seed0: int) -> list[dict[str, Any]]:
    """M0's schedule exactly: round-robin panel, seat flipping, seeds from `seed0`."""
    return [
        {
            "game": i,
            "opponent": panel[i % len(panel)],
            "seed": seed0 + i,
            "bot_seat": "a" if i % 2 == 0 else "b",
        }
        for i in range(games)
    ]


# ------------------------------------------------------------------- playing


def cmd_play(args: argparse.Namespace) -> int:
    from arena.matches.run_match import run_and_store

    traces = Path(args.traces)
    traces.mkdir(parents=True, exist_ok=True)
    bot = REPO / "bots" / "morpheus-rs" / "run.sh"
    content_hash = bot_content_hash(bot)
    print(f"[m6] morpheus-rs@{content_hash}")

    panel = tuple(args.panel) if args.panel else DEFAULT_PANEL
    schedule = _schedule(args.games, panel, args.seed)
    played: list[dict[str, Any]] = []
    for row in schedule:
        opponent = REPO / "bots" / row["opponent"] / "run.sh"
        a, b = (bot, opponent) if row["bot_seat"] == "a" else (opponent, bot)
        # One trace file per game. The path carries the game index rather than
        # the pid so a rerun overwrites its own file instead of accumulating
        # traces from runs that measured different code.
        trace = traces / f"game{row['game']:03d}.jsonl"
        os.environ["MORPHEUS_RS_TRACE"] = str(trace)
        print(
            f"[m6] game {row['game'] + 1}/{len(schedule)} vs {row['opponent']} "
            f"seed={row['seed']} seat={row['bot_seat']}",
            flush=True,
        )
        record = run_and_store(a, b, seed=row["seed"], round_name=args.round)
        played.append(
            {
                **row,
                "game_id": record.game_id,
                "winner": record.winner,
                "turns": record.turns,
                "truncated": record.truncated,
                "trace": trace.name,
            }
        )
    os.environ.pop("MORPHEUS_RS_TRACE", None)

    manifest = {
        "round": args.round,
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bot": {"bot_id": "morpheus-rs", "content_hash": content_hash},
        "host": _host_facts(),
        "panel": list(panel),
        "games": played,
    }
    path = traces / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[m6] wrote {path}")
    return 0


# ----------------------------------------------------------------- reporting


def _aggregate(files: Iterable[Path]) -> dict[str, Any]:
    move_ms: list[float] = []
    normal_move_ms: list[float] = []
    component: dict[str, list[float]] = {name: [] for name in COMPONENTS}
    # Per *call*, for the three search components whose call count the trace
    # carries. The totals above are per turn, and this bot runs more search per
    # turn than the Python does, so a total-versus-total ratio charges the Rust
    # side for the extra work it managed to fit. Dividing by the calls is the
    # only comparison that answers "is the kernel faster".
    per_call: dict[str, list[float]] = {}
    CALL_KEYS = {
        "selection": "search_selection_calls",
        "leaf_batch": "search_leaf_batch_calls",
        "enemy_prior_batch": "search_enemy_prior_calls",
    }
    sims: list[float] = []
    forwards: list[float] = []
    fallback: Counter[str] = Counter()
    startup: list[tuple[float, float, float]] = []
    over_judge = 0
    frames = 0
    games = 0
    recovery = 0
    belief_root_ok = 0

    for path in files:
        games += 1
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            frames += 1
            ms = float(row["move_ms"])
            move_ms.append(ms)
            if row["t"] == 1:
                startup.append(
                    (
                        float(row.get("load_ms", 0.0)),
                        float(row.get("warmup_ms", 0.0)),
                        float(row.get("init_ms", 0.0)),
                    )
                )
            else:
                normal_move_ms.append(ms)
                sims.append(float(row["completed_simulations"]))
                forwards.append(float(row["forward_equivalents"]))
                if ms > JUDGE_LIMIT_MS:
                    over_judge += 1
            for name, value in (row.get("components") or {}).items():
                # Turns where the component ran, not every turn. M0's
                # aggregation walks the Python's `component_ms` *dict*, which
                # only holds a key when that component was timed — so its p99
                # for anything that does not run every turn is over the turns
                # it did. The Rust side carries all ten slots always, and
                # counting the zeros would compare two different questions:
                # `belief_tensor` runs once per game (both bots time only
                # `first_move_setup` with it), so against 10,974 zeros its p99
                # is 0 and against 20 real values it is 2.4 ms.
                if float(value) > 0.0:
                    component.setdefault(name, []).append(float(value))
            for name, key in CALL_KEYS.items():
                calls = int(row.get(key, 0))
                total = float((row.get("components") or {}).get(name, 0.0))
                if calls > 0:
                    per_call.setdefault(name, []).append(total / calls)
            fallback[str(row["fallback_level"])] += 1
            recovery += int(row["recovery"])
            belief_root_ok += int(row["belief_plus_root_ok"])

    normal = max(len(normal_move_ms), 1)
    return {
        "games": games,
        "frames": frames,
        "normal_frames": len(normal_move_ms),
        "move_ms": _stats(move_ms),
        "normal_move_ms": _stats(normal_move_ms),
        "moves_over_judge_limit": over_judge,
        "moves_over_judge_limit_frac": round(over_judge / normal, 5),
        "completed_simulations_normal": _stats(sims),
        "forward_equivalents_normal": _stats(forwards),
        "offline_p99_ms": {
            name: round(_nearest_rank(component.get(name, []), 0.99), 4)
            for name in COMPONENTS
            if component.get(name)
        },
        "per_call_p99_ms": {
            name: round(_nearest_rank(values, 0.99), 4)
            for name, values in sorted(per_call.items())
        },
        "fallback_level": dict(fallback.most_common()),
        "recovery_frames": recovery,
        "belief_plus_root_ok_frac": round(belief_root_ok / max(frames, 1), 5),
        "first_move": {
            "load_ms": _stats([v[0] for v in startup]),
            "warmup_ms": _stats([v[1] for v in startup]),
            "init_ms": _stats([v[2] for v in startup]),
            "total_ms": _stats([sum(v) for v in startup]),
        },
    }


def _baseline() -> dict[str, Any] | None:
    """M0's Python numbers, from the captured arm of the same schedule.

    The captured arm rather than the control: it is the one with the full
    per-component table, and M0 bounded what measuring cost at 1.01x at p99.
    """
    if not BASELINE.is_file():
        return None
    return json.loads(BASELINE.read_text(encoding="utf-8")).get("measured")


def _markdown(report: dict[str, Any], manifest: dict[str, Any]) -> str:
    base = report["baseline"]
    rust = report["rust"]
    out: list[str] = []
    out.append("# Morpheus-rs M6 — the Rust bot's latency, on M0's schedule\n")
    out.append(
        "Milestone M6 of [the rewrite plan](../../bots/morpheus-rs/rewrite-plan.md), "
        "success criteria 2 and 4. The Python numbers are M0's, re-quoted from "
        "[the baseline](morpheus-rs-baseline.md) — same host, same panel, same "
        "seeds, same serial schedule, both bots timing themselves with their own "
        "clock.\n"
    )
    out.append(f"- Bot: `morpheus-rs@{manifest['bot']['content_hash']}`")
    out.append(f"- Host: {manifest['host'].get('cpu') or manifest['host']['machine']}")
    out.append(f"- Measured: {manifest['measured_at']}")
    out.append(
        f"- Games: {rust['games']} · frames: {rust['frames']} "
        f"({rust['normal_frames']} normal moves)\n"
    )

    out.append("## Per-move wall time\n")
    out.append("| bot | n | p50 | p99 | p99.9 | max |")
    out.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for label, stats in (
        ("morpheus (M0)", base and base.get("normal_move_ms")),
        ("morpheus-rs", rust["normal_move_ms"]),
    ):
        if not stats:
            continue
        out.append(
            f"| {label} | {stats['n']} | {stats['p50']} | {stats['p99']} | "
            f"{stats['p999']} | {stats['max']} |"
        )
    out.append("")
    over = rust["moves_over_judge_limit"]
    out.append(
        f"Normal moves over the judge's 150 ms limit: **{over}** "
        f"({rust['moves_over_judge_limit_frac'] * 100:.2f}%)."
    )
    if base:
        out.append(
            f"The Python bot's figure on the same schedule was "
            f"{base['moves_over_judge_limit']} "
            f"({base['moves_over_judge_limit_frac'] * 100:.2f}%).\n"
        )

    out.append("## Per-component p99\n")
    out.append(
        "Both columns are **per turn, over the turns the component ran** — "
        "which is what `deployment.json`'s `offline_p99_ms` means and what the "
        "admission controller forecasts from. Two things it is not. It is not "
        "a per-call speedup: the search components run once per simulation and "
        "this bot completes more simulations per turn, so a total-versus-total "
        "ratio charges it for the extra work it managed to fit; the three whose "
        "call count the trace carries are normalized below. And `belief_tensor` "
        "is a first-move cost in both bots — both time only `first_move_setup` "
        "with it — so its row is twenty values per arm, not ten thousand.\n"
    )
    out.append("| component | morpheus-rs p99 ms | morpheus (M0) p99 ms | ratio |")
    out.append("| --- | ---: | ---: | ---: |")
    base_components = (base or {}).get("offline_p99_ms", {})
    for name in COMPONENTS:
        mine = rust["offline_p99_ms"].get(name)
        theirs = base_components.get(name)
        if mine is None:
            continue
        ratio = f"{theirs / mine:.1f}x" if (theirs and mine) else "—"
        out.append(f"| {name} | {mine:.4f} | {theirs if theirs else '—'} | {ratio} |")
    out.append("")

    per_call = rust.get("per_call_p99_ms") or {}
    base_calls = (base or {}).get("component_calls_mean", {})
    if per_call:
        out.append("### Per call, where the trace counts calls\n")
        out.append(
            "The Python side divides its per-turn p99 by its *mean* calls per "
            "turn, which is the only figure M0 recorded — so its column is an "
            "estimate and the Rust column is not.\n"
        )
        out.append(
            "| component | morpheus-rs p99 ms/call | morpheus (M0) est. ms/call | ratio |"
        )
        out.append("| --- | ---: | ---: | ---: |")
        for name, mine in sorted(per_call.items()):
            total = base_components.get(name)
            calls = base_calls.get(name)
            theirs = (total / calls) if (total and calls) else None
            ratio = f"{theirs / mine:.1f}x" if (theirs and mine) else "—"
            out.append(
                f"| {name} | {mine:.4f} | "
                f"{f'{theirs:.4f}' if theirs else '—'} | {ratio} |"
            )
        out.append("")

    out.append("## Search throughput\n")
    sims = rust["completed_simulations_normal"]
    out.append(
        f"Completed simulations per normal move: p50 **{sims['p50']:.0f}**, "
        f"p99 {sims['p99']:.0f}, max {sims['max']:.0f}, mean {sims['mean']:.2f}."
    )
    if base:
        bsims = base["completed_simulations_normal"]
        out.append(
            f"The Python bot: p50 {bsims['p50']:.0f}, p99 {bsims['p99']:.0f}, "
            f"mean {bsims['mean']:.2f}.\n"
        )
    out.append(
        f"Belief update *and* root inference both completed on "
        f"{rust['belief_plus_root_ok_frac'] * 100:.1f}% of turns"
        + (
            f", against {base['belief_plus_root_ok_frac'] * 100:.1f}% for the Python.\n"
            if base
            else ".\n"
        )
    )
    out.append("| degradation level | turns |")
    out.append("| --- | ---: |")
    for level, count in rust["fallback_level"].items():
        out.append(f"| {level} | {count} |")
    out.append("")

    first = rust["first_move"]
    out.append("## First-move budget\n")
    out.append(
        "Success criterion 4: load + warmup + init has to fit "
        "`first_move_limit_ms` with the same reserve discipline as today.\n"
    )
    out.append("| stage | p50 ms | p99 ms | max ms |")
    out.append("| --- | ---: | ---: | ---: |")
    for name in ("load_ms", "warmup_ms", "init_ms", "total_ms"):
        s = first[name]
        out.append(f"| {name[:-3]} | {s['p50']} | {s['p99']} | {s['max']} |")
    out.append("")
    return "\n".join(out) + "\n"


def cmd_report(args: argparse.Namespace) -> int:
    traces = Path(args.traces)
    manifest_path = traces / "manifest.json"
    if not manifest_path.is_file():
        print(f"no manifest at {manifest_path}; run `play` first", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = sorted(traces.glob("game*.jsonl"))
    if not files:
        print(f"no traces under {traces}", file=sys.stderr)
        return 1
    report = {"rust": _aggregate(files), "baseline": _baseline(), "manifest": manifest}
    MEASUREMENTS.mkdir(parents=True, exist_ok=True)
    json_path = MEASUREMENTS / "morpheus-rs-m6-latency.json"
    json_path.write_text(
        json.dumps({k: v for k, v in report.items() if k != "baseline"}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    md_path = MEASUREMENTS / "morpheus-rs-m6-latency.md"
    md_path.write_text(_markdown(report, manifest), encoding="utf-8")
    print(f"[m6] wrote {json_path}\n[m6] wrote {md_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    play = sub.add_parser("play", help="serial instrumented games")
    play.add_argument("--games", type=int, default=20)
    play.add_argument("--seed", type=int, default=1000, help="first match seed")
    play.add_argument("--round", default=DEFAULT_ROUND)
    play.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    play.add_argument("--panel", nargs="*", default=None)
    play.set_defaults(func=cmd_play)

    report = sub.add_parser("report", help="aggregate traces into the report")
    report.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    report.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
