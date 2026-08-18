#!/usr/bin/env python3
"""Morpheus-rs M0: the frozen Python bot's latency baseline and parity corpus.

Milestone M0 of docs/bots/morpheus-rs/rewrite-plan.md. Everything the Rust
rewrite is measured against starts here: the per-component p99 table that
`deployment.json` currently asserts from a stale calibration, the per-move
wall-time distribution, how many simulations a normal move actually completes,
and the recorded frames a Rust port replays in parity mode.

    # ~20 instrumented competition games, capture armed (slow; run detached)
    python scripts/morpheus_rs_baseline.py capture --games 20

    # aggregate into the published measurement report
    python scripts/morpheus_rs_baseline.py report

    # cut the committed smoke slice out of the derived corpus
    python scripts/morpheus_rs_baseline.py fixtures

`capture` writes derived data only: games under `data/games/`, trajectories and
captures under `data/morpheus/morpheus-rs/<round>/`. The corpus is far too
large to commit; `fixtures` is what produces the handful of frames that ride
with the repo.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from arena.instrument.capture import (  # noqa: E402
    CAPTURE_MODULE_ENV,
    read_frames,
    write_frames,
)
from arena.records.fingerprint import bot_content_hash  # noqa: E402

CAPTURE_MODULE = REPO / "bots" / "morpheus-rs" / "tools" / "capture_morpheus.py"
DEFAULT_ROUND = "morpheus-rs-m0"
DEFAULT_CORPUS = REPO / "data" / "morpheus" / "morpheus-rs"
MEASUREMENTS = REPO / "docs" / "research" / "measurements"
DEFAULT_FIXTURE = (
    REPO / "bots" / "morpheus-rs" / "tests" / "fixtures" / "parity-smoke.jsonl.gz"
)

# Varied panel per rewrite-plan §5: the rating anchor, one research bot, and
# two heuristics whose play forces different phases (castle economy, expansion
# pressure). Seats alternate across the schedule so neither side's opening is
# over-represented in the corpus.
DEFAULT_PANEL = ("cm_expander", "macaria", "castle_builder", "metro")

# The `deployment.json` components, in the order the shipped table lists them.
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

# The judge's hard per-move limit (RULES.md); the shipped internal deadline is
# `normal_deadline_ms` in bots/morpheus/deployment.json.
JUDGE_LIMIT_MS = 150.0


# ------------------------------------------------------------------ helpers


def _nearest_rank(values: Sequence[float], q: float) -> float:
    """
    Nearest-rank percentile — the same estimator `runtime.py` deploys.

    Deliberately not `statistics.quantiles`, which interpolates: the bot's
    admission control forecasts from nearest rank, so a baseline computed any
    other way would not be the number the controller is acting on.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(-(-q * len(ordered) // 1))))
    return float(ordered[rank - 1])


def _stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 4),
        "p50": round(_nearest_rank(values, 0.50), 4),
        "p99": round(_nearest_rank(values, 0.99), 4),
        "p999": round(_nearest_rank(values, 0.999), 4),
        "max": round(max(values), 4),
    }


def _capture_files(corpus_dir: Path) -> list[Path]:
    return sorted(corpus_dir.rglob("*.capture.*.jsonl.gz"))


def _label(path: Path) -> str:
    """Repo-relative when it can be — a corpus may live outside the checkout."""
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _host_facts() -> dict[str, Any]:
    facts = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "processor": platform.processor(),
    }
    if sys.platform == "darwin":
        try:
            facts["cpu_brand"] = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return facts


# ------------------------------------------------------------------ capture


def _schedule(games: int, panel: Sequence[str], seed0: int) -> list[dict[str, Any]]:
    """
    One row per game: opponent, seed, and which seat morpheus takes.

    Round-robin over the panel with the seat flipping every game, so a panel
    of five and twenty games gives each opponent two games on each side. Seeds
    are consecutive from `seed0`, which makes the schedule reproducible and
    the games independent boards.
    """
    rows = []
    for i in range(games):
        opponent = panel[i % len(panel)]
        rows.append(
            {
                "game": i,
                "opponent": opponent,
                "seed": seed0 + i,
                "morpheus_seat": "a" if i % 2 == 0 else "b",
            }
        )
    return rows


def _play(
    schedule: Sequence[dict[str, Any]],
    *,
    round_name: str,
    directory: Path,
    label: str,
    register: bool = True,
    engine: str | None = None,
) -> list[dict[str, Any]]:
    from arena.matches.run_match import run_and_store

    morpheus = REPO / "bots" / "morpheus" / "run.sh"
    played: list[dict[str, Any]] = []
    for row in schedule:
        opponent = REPO / "bots" / row["opponent"] / "run.sh"
        a, b = (
            (morpheus, opponent)
            if row["morpheus_seat"] == "a"
            else (opponent, morpheus)
        )
        # Hashing needs only the source closure; *registering* needs a git
        # sandbox, which a Modal container running the same suite does not
        # have. Passing the hashes explicitly keeps the games correctly
        # attributed without asking for a lineage step that only the developer
        # machine can legitimately append.
        hashes = (
            {}
            if register
            else {
                "bot_a_content_hash": bot_content_hash(a),
                "bot_b_content_hash": bot_content_hash(b),
            }
        )
        print(
            f"[m0] {label} game {row['game'] + 1}/{len(schedule)} "
            f"vs {row['opponent']} seed={row['seed']} "
            f"morpheus_seat={row['morpheus_seat']}",
            flush=True,
        )
        record = run_and_store(
            a,
            b,
            seed=row["seed"],
            round_name=round_name,
            record_trajectory=True,
            trajectories_dir=directory,
            engine=engine,
            **hashes,
        )
        played.append(
            {
                **row,
                "game_id": record.game_id,
                "winner": record.winner,
                "turns": record.turns,
                "truncated": record.truncated,
            }
        )
    return played


def cmd_capture(args: argparse.Namespace) -> int:
    import os

    corpus_dir = Path(args.corpus) / args.round
    control_dir = Path(args.corpus) / f"{args.round}-control"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    panel = tuple(args.panel) if args.panel else DEFAULT_PANEL

    morpheus = REPO / "bots" / "morpheus" / "run.sh"
    oracle_hash = bot_content_hash(morpheus)
    print(f"[m0] oracle morpheus@{oracle_hash}")

    schedule = _schedule(args.games, panel, args.seed)

    # Control first, with capture *disarmed*, over the head of the same
    # schedule — same opponents, same seeds, same seats. Without it the
    # baseline cannot separate what the bot costs from what measuring it
    # costs, and every later "the Rust bot is N times faster" rests on a
    # number nobody bounded. Traces alone are enough: `move_ms` comes from
    # the agent's own clock and rides the ordinary probe.
    control: list[dict[str, Any]] = []
    if args.control_games > 0:
        os.environ.pop(CAPTURE_MODULE_ENV, None)
        control_dir.mkdir(parents=True, exist_ok=True)
        control = _play(
            schedule[: args.control_games],
            round_name=f"{args.round}-control",
            directory=control_dir,
            label="control",
            register=not args.no_register,
            engine=args.engine_version or None,
        )

    # Armed for the child processes the match loop spawns. Set here rather than
    # asked of the caller so a capture run cannot be started half-armed.
    os.environ[CAPTURE_MODULE_ENV] = str(CAPTURE_MODULE)
    if args.stride is not None:
        os.environ["MORPHEUS_CAPTURE_STRIDE"] = str(args.stride)

    played = _play(
        schedule,
        round_name=args.round,
        directory=corpus_dir,
        label="capture",
        register=not args.no_register,
        engine=args.engine_version or None,
    )

    manifest = {
        "round": args.round,
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "oracle": {"bot_id": "morpheus", "content_hash": oracle_hash},
        "host": _host_facts(),
        "panel": list(panel),
        "games": played,
        "control_games": control,
    }
    manifest_path = corpus_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[m0] wrote {manifest_path}")
    return 0


# ------------------------------------------------------------------- report


def _aggregate(files: Iterable[Path]) -> dict[str, Any]:
    """
    Fold every captured frame into the report's distributions.

    Read with `arrays=False`: the report only ever touches scalars, and
    decoding several hundred megabytes of tensors to compute a median move
    time would dominate the run for nothing.
    """
    move_ms: list[float] = []
    normal_move_ms: list[float] = []
    component: dict[str, list[float]] = defaultdict(list)
    component_calls: dict[str, list[int]] = defaultdict(list)
    sims_normal: list[int] = []
    forwards_normal: list[int] = []
    strata: Counter[str] = Counter()
    heavy_strata: Counter[str] = Counter()
    fallback: Counter[str] = Counter()
    recovery = 0
    belief_root_ok = 0
    over_judge = 0
    frames = 0
    games = 0
    rng_methods: Counter[str] = Counter()
    rng_unrecorded: Counter[str] = Counter()

    for path in files:
        games += 1
        for frame in read_frames(path, arrays=False):
            frames += 1
            timing = frame["timing"]
            counters = frame["counters"]
            ms = float(timing["move_ms"])
            move_ms.append(ms)
            first_move = frame["t"] == 1
            if not first_move:
                normal_move_ms.append(ms)
                sims_normal.append(int(counters["completed_simulations"]))
                forwards_normal.append(int(counters["forward_equivalents"]))
                if ms > JUDGE_LIMIT_MS:
                    over_judge += 1
            for name, value in timing["component_ms"].items():
                component[name].append(float(value))
            for name, value in (timing.get("component_calls") or {}).items():
                component_calls[name].append(int(value))
            strata[frame["stratum"]] += 1
            if frame["heavy"]:
                heavy_strata[frame["stratum"]] += 1
            fallback[str(counters["fallback_level"])] += 1
            recovery += int(counters["recovery"])
            belief_root_ok += int(counters["belief_plus_root_ok"])
            for draw in frame.get("rng_draws", ()):
                rng_methods[draw["m"]] += 1
            for name, count in (frame.get("rng_unrecorded") or {}).items():
                rng_unrecorded[name] += count

    normal_frames = max(len(normal_move_ms), 1)
    return {
        "games": games,
        "frames": frames,
        "normal_frames": len(normal_move_ms),
        "move_ms": _stats(move_ms),
        "normal_move_ms": _stats(normal_move_ms),
        "moves_over_judge_limit": over_judge,
        "moves_over_judge_limit_frac": round(over_judge / normal_frames, 5),
        "completed_simulations_normal": _stats([float(v) for v in sims_normal]),
        "forward_equivalents_normal": _stats([float(v) for v in forwards_normal]),
        "offline_p99_ms": {
            name: round(_nearest_rank(component.get(name, []), 0.99), 4)
            for name in COMPONENTS
            if component.get(name)
        },
        "component_ms": {name: _stats(v) for name, v in sorted(component.items())},
        "component_calls_mean": {
            name: round(statistics.fmean(v), 3)
            for name, v in sorted(component_calls.items())
        },
        "strata": dict(strata.most_common()),
        "heavy_strata": dict(heavy_strata.most_common()),
        "fallback_level": dict(fallback.most_common()),
        "recovery_frames": recovery,
        "recovery_frac": round(recovery / max(frames, 1), 5),
        "belief_plus_root_ok_frac": round(belief_root_ok / max(frames, 1), 5),
        "rng_draws": dict(rng_methods.most_common()),
        "rng_draws_per_turn": round(sum(rng_methods.values()) / max(frames, 1), 2),
        "rng_unrecorded": dict(rng_unrecorded.most_common()),
    }


def _control_move_ms(
    control_dir: Path, games: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """
    Morpheus's per-move wall time from the uncaptured control games.

    Read from each game's probe trace on the seat the manifest says morpheus
    played — `move_ms` is the agent's own measurement either way, so the only
    difference from the captured arm is whether the capture ran at all.
    """
    from arena.records.trajectories import read_trace, trace_path

    move_ms: list[float] = []
    sims: list[float] = []
    found = 0
    for row in games:
        path = trace_path(row["game_id"], row["morpheus_seat"], control_dir)
        if not path.is_file():
            continue
        found += 1
        series = read_trace(path)
        move_ms.extend(float(v) for v in series.get("move_ms", [])[1:])
        sims.extend(float(v) for v in series.get("completed_simulations", [])[1:])
    return {
        "games": found,
        "normal_move_ms": _stats(move_ms),
        "completed_simulations_normal": _stats(sims),
        "moves_over_judge_limit_frac": round(
            sum(1 for v in move_ms if v > JUDGE_LIMIT_MS) / max(len(move_ms), 1), 5
        ),
    }


def _deployment() -> dict[str, Any]:
    return json.loads(
        (REPO / "bots" / "morpheus" / "deployment.json").read_text(encoding="utf-8")
    )


def _deployment_field(name: str, fallback: float) -> float:
    try:
        return float(_deployment()[name])
    except (KeyError, TypeError, ValueError):
        return fallback


def _shipped_p99() -> dict[str, float]:
    return dict(_deployment().get("offline_p99_ms") or {})


def _markdown(report: dict[str, Any]) -> str:
    m = report["measured"]
    shipped = report["shipped_offline_p99_ms"]
    lines: list[str] = []
    add = lines.append

    add("# Morpheus-rs M0 baseline — the Python oracle, re-measured")
    add("")
    add(
        f"Milestone M0 of [the rewrite plan](../../bots/morpheus-rs/rewrite-plan.md). "
        f"Every latency claim the rewrite is judged against starts from this table."
    )
    add("")
    add(f"- Oracle: `{report['oracle']['bot_id']}@{report['oracle']['content_hash']}`")
    add(f"- Host: {report['host'].get('cpu_brand', report['host']['platform'])}")
    add(f"- Captured: {report['captured_at']}")
    add(
        f"- Games: {m['games']} · frames: {m['frames']} "
        f"({m['normal_frames']} normal moves)"
    )
    add("")

    add("## Per-move wall time")
    add("")
    add("| series | n | p50 | p99 | p99.9 | max |")
    add("| --- | ---: | ---: | ---: | ---: | ---: |")
    for label, key in (("all moves", "move_ms"), ("normal moves", "normal_move_ms")):
        s = m[key]
        add(
            f"| {label} | {s['n']} | {s['p50']} | {s['p99']} | "
            f"{s['p999']} | {s['max']} |"
        )
    add("")
    add(
        f"Normal moves over the judge's {JUDGE_LIMIT_MS:.0f} ms limit: "
        f"**{m['moves_over_judge_limit']}** "
        f"({m['moves_over_judge_limit_frac'] * 100:.2f}%)."
    )
    add("")

    control = report.get("control") or {}
    if control.get("games"):
        c = control["normal_move_ms"]
        cm = m["normal_move_ms"]
        add("### Control: the same games, uncaptured")
        add("")
        add(
            f"{control['games']} game(s) over the head of the same schedule — "
            "same opponents, same seeds, same seats — played with capture "
            "disarmed. `move_ms` is the agent's own clock in both arms, so the "
            "gap is what measuring costs, and it bounds how much of the table "
            "above to discount."
        )
        add("")
        add("| series | n | p50 | p99 | p99.9 | max |")
        add("| --- | ---: | ---: | ---: | ---: | ---: |")
        add(
            f"| control (uncaptured) | {c['n']} | {c['p50']} | {c['p99']} | "
            f"{c['p999']} | {c['max']} |"
        )
        add(
            f"| captured | {cm['n']} | {cm['p50']} | {cm['p99']} | "
            f"{cm['p999']} | {cm['max']} |"
        )
        add("")
        if c["p99"]:
            add(
                f"Capture overhead at p99: **{cm['p99'] / c['p99']:.2f}x** "
                f"({cm['p99'] - c['p99']:+.1f} ms). Over-limit moves: "
                f"{control['moves_over_judge_limit_frac'] * 100:.2f}% control vs "
                f"{m['moves_over_judge_limit_frac'] * 100:.2f}% captured."
            )
            add("")

    add("## Per-component p99 — measured against shipped")
    add("")
    add(
        "`shipped` is the `offline_p99_ms` table in `bots/morpheus/deployment.json`, "
        "which the plan quotes only for shape: it was fitted on a host and a "
        "configuration whose qualification verdict is *no*."
    )
    add("")
    add("| component | measured p99 ms | shipped p99 ms | ratio |")
    add("| --- | ---: | ---: | ---: |")
    for name in COMPONENTS:
        measured = m["offline_p99_ms"].get(name)
        if measured is None:
            continue
        was = shipped.get(name)
        if was:
            add(f"| {name} | {measured:.3f} | {was:.3f} | {measured / was:.2f}x |")
        else:
            add(f"| {name} | {measured:.3f} | — | — |")
    add("")

    # The number the rewrite's whole thesis rests on: how much of the internal
    # deadline is gone before search starts. Quoted from measured p99s rather
    # than restated from the plan, which took it from the stale shipped table.
    p99 = m["offline_p99_ms"]
    belief = sum(
        p99.get(name, 0.0)
        for name in ("belief_proposal", "belief_tensor", "particle_transitions")
    )
    root = p99.get("root_inference", 0.0)
    deadline = _deployment_field("normal_deadline_ms", 140.0)
    add("### Where the turn goes")
    add("")
    add(
        f"Belief update ({belief:.1f} ms at p99) plus root inference "
        f"({root:.1f} ms) is **{belief + root:.1f} ms** against an internal "
        f"deadline of {deadline:.0f} ms. Search gets whatever is left, which "
        + (
            "is nothing at p99 — the deadline is already spent before a single "
            "simulation is admitted."
            if belief + root >= deadline
            else f"is {deadline - belief - root:.1f} ms at p99."
        )
    )
    add("")
    add(
        "That is the rewrite's premise stated in measured numbers, and it is "
        "why `particle_transitions` is the first kernel ported (plan §7), not "
        "inference."
    )
    add("")

    add("## Search throughput")
    add("")
    s = m["completed_simulations_normal"]
    add(
        f"Completed simulations per normal move: p50 **{s['p50']:.0f}**, "
        f"p99 {s['p99']:.0f}, max {s['max']:.0f}, mean {s['mean']:.2f}."
    )
    f = m["forward_equivalents_normal"]
    add(
        f"Forward equivalents per normal move: p50 {f['p50']:.0f}, "
        f"p99 {f['p99']:.0f}, max {f['max']:.0f}."
    )
    add("")
    add(
        f"Belief update *and* root inference both completed on "
        f"{m['belief_plus_root_ok_frac'] * 100:.1f}% of turns; the belief update "
        f"was deferred on {m['recovery_frac'] * 100:.1f}%."
    )
    add("")
    add("| degradation level | turns |")
    add("| --- | ---: |")
    for level, count in m["fallback_level"].items():
        add(f"| {level} | {count} |")
    add("")

    add("## Parity corpus")
    add("")
    add(
        "Frames are captured for every turn; the heavy payload (belief "
        "particles, root tensor, shaping scores) rides a per-stratum stride. "
        "Format: [parity-corpus.md](../../bots/morpheus-rs/parity-corpus.md)."
    )
    add("")
    add("| stratum | frames | heavy |")
    add("| --- | ---: | ---: |")
    for stratum, count in m["strata"].items():
        add(f"| {stratum} | {count} | {m['heavy_strata'].get(stratum, 0)} |")
    add("")
    add(
        f"RNG draws recorded: {sum(m['rng_draws'].values())} "
        f"({m['rng_draws_per_turn']} per turn) — "
        + ", ".join(f"`{k}` {v}" for k, v in m["rng_draws"].items())
        + "."
    )
    unrecorded = m["rng_unrecorded"]
    add("")
    add(
        "Unrecorded generator calls: **none** — the whole draw stream is "
        "replayable."
        if not unrecorded
        else "Unrecorded generator calls: **"
        + ", ".join(f"{k} ({v})" for k, v in unrecorded.items())
        + "** — the replay stream is incomplete until these are recorded."
    )
    add("")

    add("## Read alongside")
    add("")
    add(
        "- [one x86 core](morpheus-rs-baseline-modal.md) — the same suite on "
        "Linux, which is the shape the competition host has and this one does "
        "not"
    )
    add(
        "- [CPU-feature probe](morpheus-rs-cpu-probe.md) — the compile target "
        "these measurements do not settle"
    )
    add(
        "- [parity corpus format](../../bots/morpheus-rs/parity-corpus.md) — "
        "what the captured frames contain and what they cannot prove"
    )
    add("")
    add(
        "Every number here is this host's. Morpheus is deadline-driven, so its "
        "play is a function of how fast the machine under it happens to be — "
        "these are not seeds anyone can replay into the same games."
    )
    add("")
    return "\n".join(lines) + "\n"


def cmd_report(args: argparse.Namespace) -> int:
    corpus_dir = Path(args.corpus) / args.round
    files = _capture_files(corpus_dir)
    if not files:
        print(f"[m0] no captures under {corpus_dir}", file=sys.stderr)
        return 1

    manifest_path = corpus_dir / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    report = {
        "round": args.round,
        "captured_at": manifest.get("captured_at", "unknown"),
        "oracle": manifest.get(
            "oracle",
            {
                "bot_id": "morpheus",
                "content_hash": bot_content_hash(
                    REPO / "bots" / "morpheus" / "run.sh"
                ),
            },
        ),
        "host": manifest.get("host", _host_facts()),
        "panel": manifest.get("panel", []),
        "capture_files": [_label(p) for p in files],
        "shipped_offline_p99_ms": _shipped_p99(),
        "measured": _aggregate(files),
        "control": _control_move_ms(
            Path(args.corpus) / f"{args.round}-control",
            manifest.get("control_games", []),
        ),
    }

    json_out = Path(args.output_json)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_out = Path(args.output_md)
    md_out.write_text(_markdown(report), encoding="utf-8")
    print(f"[m0] wrote {json_out}")
    print(f"[m0] wrote {md_out}")

    m = report["measured"]
    print(
        json.dumps(
            {
                "games": m["games"],
                "frames": m["frames"],
                "normal_move_p99_ms": m["normal_move_ms"]["p99"],
                "over_judge_limit_frac": m["moves_over_judge_limit_frac"],
                "sims_p50": m["completed_simulations_normal"]["p50"],
            },
            indent=2,
        )
    )
    return 0


# ----------------------------------------------------------------- fixtures


def cmd_fixtures(args: argparse.Namespace) -> int:
    """
    Cut the committed smoke slice: one heavy frame per stratum, per §5.

    The full corpus is derived data measured in tens of megabytes. What ships
    with the repo is the slice CI can run in under a second — enough to catch a
    Rust port that breaks a whole surface, not enough to prove parity, which is
    what `tools/run_parity.sh` over the full corpus is for.
    """
    corpus_dir = Path(args.corpus) / args.round
    files = _capture_files(corpus_dir)
    if not files:
        print(f"[m0] no captures under {corpus_dir}", file=sys.stderr)
        return 1

    per_stratum = max(1, int(args.per_stratum))
    chosen: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in files:
        for frame in read_frames(path, arrays=False):
            if not frame["heavy"]:
                continue
            bucket = chosen[frame["stratum"]]
            if len(bucket) < per_stratum:
                frame["source"] = path.name
                bucket.append(frame)

    frames = [f for stratum in sorted(chosen) for f in chosen[stratum]]
    out = Path(args.output)
    written = write_frames(frames, out)
    print(
        json.dumps(
            {
                "output": _label(out),
                "frames": written,
                "bytes": out.stat().st_size,
                "strata": {k: len(v) for k, v in sorted(chosen.items())},
            },
            indent=2,
        )
    )
    return 0


# --------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--round", default=DEFAULT_ROUND)
        p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)

    capture = sub.add_parser(
        "capture", help="play instrumented competition games with capture armed"
    )
    common(capture)
    capture.add_argument("--games", type=int, default=20)
    capture.add_argument("--seed", type=int, default=1000, help="first match seed")
    capture.add_argument(
        "--panel", nargs="*", default=None, help=f"default: {' '.join(DEFAULT_PANEL)}"
    )
    capture.add_argument(
        "--stride",
        type=int,
        default=None,
        help="turns between heavy frames within a stratum (default: module's 8)",
    )
    capture.add_argument(
        "--no-register",
        action="store_true",
        help="skip the version registry (needs git); hash the closure directly",
    )
    capture.add_argument(
        "--engine-version",
        default="",
        help="competition-module SHA, when the checkout has no git metadata",
    )
    capture.add_argument(
        "--control-games",
        type=int,
        default=4,
        help="uncaptured games over the same seeds, to bound capture overhead",
    )
    capture.set_defaults(func=cmd_capture)

    report = sub.add_parser("report", help="aggregate captures into the M0 report")
    common(report)
    report.add_argument(
        "--output-json", type=Path, default=MEASUREMENTS / "morpheus-rs-baseline.json"
    )
    report.add_argument(
        "--output-md", type=Path, default=MEASUREMENTS / "morpheus-rs-baseline.md"
    )
    report.set_defaults(func=cmd_report)

    fixtures = sub.add_parser("fixtures", help="cut the committed smoke slice")
    common(fixtures)
    fixtures.add_argument("--output", type=Path, default=DEFAULT_FIXTURE)
    fixtures.add_argument("--per-stratum", type=int, default=2)
    fixtures.set_defaults(func=cmd_fixtures)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
