#!/usr/bin/env python3
"""Morpheus-rs M7: measure a knob configuration, then qualify one.

Milestone M7 of docs/bots/morpheus-rs/rewrite-plan.md — the first *qualified*
deployment either bot will have had. §8's procedure, in order:

    # 1. per-call costs at a handful of configs, to fit the cost model
    python scripts/morpheus_rs_m7.py sweep --probe --games 2

    # 2. predict the whole grid from those, and shortlist what fits
    python scripts/morpheus_rs_m7.py model

    # 3. the feasibility gate on a shortlisted config, for real
    python scripts/morpheus_rs_m7.py sweep --config n16-s32-b4-d4 --games 20

Every run here is **unrated**: it drives `matchup.py` directly, so no game
record is stored and no registry step is appended. A configuration is part of
the content hash, so a rated round needs a separate bot directory per arm —
that is step 4 of §8 and it is not this script's job.

## How a config is applied

`bots/morpheus-rs/deployment.json` is overwritten for the duration of the run
and restored in a `finally`, the same shape `tools/mutation_check.py` uses on
source files. Two reasons it is safe here: `run.sh` decides staleness from
`crates/**` and the Cargo files only, so a config change never triggers a
rebuild; and nothing rated runs while it is swapped.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BOT = REPO / "bots" / "morpheus-rs"
DEPLOYMENT = BOT / "deployment.json"
MATCHUP = REPO / "competition-module" / "competition" / "matchup.py"
MEASUREMENTS = REPO / "docs" / "research" / "measurements"
DEFAULT_OUT = REPO / "data" / "morpheus" / "morpheus-rs" / "m7"

# The interpreter the bots are launched with. A checkout has a venv holding
# torch and jax; a Modal container has neither the path nor the need, because
# the image installs into the system interpreter this script is already running
# under. `run.sh` reads `$PYTHON`, so getting this wrong is the silent
# "agent closed stdout unexpectedly" failure.
VENV_PYTHON = REPO / ".venv" / "bin" / "python"
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.is_file() else sys.executable

JUDGE_LIMIT_MS = 150.0

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

# The grid §8 names. `pending_leaf_batch` is included even though the engine
# has no batch axis — "batch 4" is four sequential forwards (M3) — because it
# still decides how much work happens between two admission checks, and that
# is what can overrun a deadline.
GRID = {
    "n_particles": (8, 16, 32, 64),
    "target_simulations": (16, 32, 64),
    "pending_leaf_batch": (2, 4, 8),
    "search_depth": (2, 4, 8, 16),
}

# The configs the cost model is fitted from: one axis moved at a time off the
# parity knobs, so each coefficient has a lever that only it responds to.
PROBE_CONFIGS = (
    {"n_particles": 8, "target_simulations": 16, "pending_leaf_batch": 4, "search_depth": 2},
    {"n_particles": 16, "target_simulations": 16, "pending_leaf_batch": 4, "search_depth": 2},
    {"n_particles": 32, "target_simulations": 16, "pending_leaf_batch": 4, "search_depth": 2},
    {"n_particles": 64, "target_simulations": 16, "pending_leaf_batch": 4, "search_depth": 2},
    {"n_particles": 8, "target_simulations": 32, "pending_leaf_batch": 4, "search_depth": 2},
    {"n_particles": 8, "target_simulations": 64, "pending_leaf_batch": 4, "search_depth": 2},
    {"n_particles": 8, "target_simulations": 16, "pending_leaf_batch": 4, "search_depth": 8},
    {"n_particles": 8, "target_simulations": 16, "pending_leaf_batch": 8, "search_depth": 2},
)

# Two opponents, not five: the sweep measures cost, and cost is a property of
# the position rather than of who produced it. One expander and one strong
# research bot give short games and long ones without spending five times the
# wall clock per config.
SWEEP_PANEL = ("cm_expander", "macaria")


def slug(config: dict[str, int]) -> str:
    return (
        f"n{config['n_particles']}-s{config['target_simulations']}"
        f"-b{config['pending_leaf_batch']}-d{config['search_depth']}"
    )


def parse_slug(text: str) -> dict[str, int]:
    parts = dict(zip("nsbd", text.split("-")))
    return {
        "n_particles": int(parts["n"][1:]),
        "target_simulations": int(parts["s"][1:]),
        "pending_leaf_batch": int(parts["b"][1:]),
        "search_depth": int(parts["d"][1:]),
    }


def _nearest_rank(values: Sequence[float], q: float) -> float:
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
        "p50": round(_nearest_rank(values, 0.50), 4),
        "p99": round(_nearest_rank(values, 0.99), 4),
        "p999": round(_nearest_rank(values, 0.999), 4),
        "max": round(max(values), 4),
        "mean": round(statistics.fmean(values), 4),
    }


def host_facts() -> dict[str, Any]:
    facts = {"platform": platform.platform(), "machine": platform.machine()}
    try:
        facts["cpu"] = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        facts["cpu"] = platform.processor()
    return facts


# ------------------------------------------------------------------- playing


def apply_config(config: dict[str, int], extra: dict[str, Any] | None = None) -> dict:
    """Overwrite `deployment.json` with these knobs; return the original."""
    original = json.loads(DEPLOYMENT.read_text(encoding="utf-8"))
    updated = dict(original)
    updated.update(config)
    if extra:
        updated.update(extra)
    # `max_proposal_batch` bounds the proposal's batching and is meaningless
    # above the particle count; the parity file has them equal and a sweep that
    # raised one without the other would measure a cap, not a configuration.
    updated["max_proposal_batch"] = int(config.get("n_particles", updated["n_particles"]))
    DEPLOYMENT.write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return original


def restore_config(original: dict) -> None:
    DEPLOYMENT.write_text(json.dumps(original, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def play(trace: Path, opponent: str, seed: int, seat_a: bool) -> dict[str, Any]:
    """One unrated competition match, with this bot's trace armed."""
    env = dict(os.environ)
    env["MORPHEUS_RS_TRACE"] = str(trace)
    env["PYTHON"] = PYTHON
    env["PATH"] = f"{Path.home() / '.cargo' / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    mine = str(BOT / "run.sh")
    theirs = str(REPO / "bots" / opponent / "run.sh")
    a, b = (mine, theirs) if seat_a else (theirs, mine)
    result = subprocess.run(
        [PYTHON, str(MATCHUP), a, b,
         "--mode", "competition", "--seed", str(seed)],
        capture_output=True, text=True, env=env, cwd=str(REPO),
    )
    tail = (result.stdout or "").strip().splitlines()[-3:]
    return {"opponent": opponent, "seed": seed, "seat": "a" if seat_a else "b",
            "returncode": result.returncode, "tail": tail}


def cmd_sweep(args: argparse.Namespace) -> int:
    # Absolute, always. `MORPHEUS_RS_TRACE` is read by the *bot*, which
    # `matchup.py` spawns with cwd set to the bot's own directory — so a
    # relative `--out` writes the traces under `bots/morpheus-rs/` and this
    # script then reports "no sweep directories" over a run that went fine.
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.probe:
        configs = list(PROBE_CONFIGS)
    elif args.config:
        configs = [parse_slug(c) for c in args.config]
    else:
        print("give --probe or --config <slug>", file=sys.stderr)
        return 2

    original = json.loads(DEPLOYMENT.read_text(encoding="utf-8"))
    try:
        for config in configs:
            name = slug(config)
            directory = out / name
            directory.mkdir(parents=True, exist_ok=True)
            apply_config(config, extra=args.extra)
            played = []
            for i in range(args.games):
                opponent = SWEEP_PANEL[i % len(SWEEP_PANEL)]
                trace = directory / f"game{i:03d}.jsonl"
                print(f"[m7] {name} game {i + 1}/{args.games} vs {opponent}", flush=True)
                played.append(play(trace, opponent, args.seed + i, i % 2 == 0))
            (directory / "manifest.json").write_text(
                json.dumps(
                    {
                        "config": config,
                        "extra": args.extra or {},
                        "games": played,
                        "host": host_facts(),
                        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    finally:
        restore_config(original)
        print("[m7] deployment.json restored")
    return 0


# ----------------------------------------------------------------- measuring


def read_traces(directory: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(directory.glob("game*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize(directory: Path) -> dict[str, Any] | None:
    rows = read_traces(directory)
    if not rows:
        return None
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    normal = [r for r in rows if r["t"] > 1]
    move_ms = [float(r["move_ms"]) for r in normal]
    per_turn: dict[str, list[float]] = {}
    per_call: dict[str, list[float]] = {}
    calls: dict[str, list[float]] = {}
    for r in normal:
        comps = r.get("components") or {}
        cs = r.get("calls") or {}
        for name, value in comps.items():
            value = float(value)
            n = int(cs.get(name, 0))
            if value > 0.0:
                per_turn.setdefault(name, []).append(value)
            if n > 0:
                per_call.setdefault(name, []).append(value / n)
                calls.setdefault(name, []).append(float(n))
    over = [v for v in move_ms if v > JUDGE_LIMIT_MS]
    return {
        "config": manifest.get("config", (rows[0].get("config") or {})),
        "games": len(list(directory.glob("game*.jsonl"))),
        "turns": len(normal),
        "move_ms": _stats(move_ms),
        "over_judge_limit": len(over),
        "completed_simulations": _stats([float(r["completed_simulations"]) for r in normal]),
        "belief_plus_root_ok_frac": round(
            sum(int(r["belief_plus_root_ok"]) for r in normal) / max(len(normal), 1), 5
        ),
        "per_turn_p99_ms": {k: round(_nearest_rank(v, 0.99), 4) for k, v in sorted(per_turn.items())},
        "per_call_p99_ms": {k: round(_nearest_rank(v, 0.99), 4) for k, v in sorted(per_call.items())},
        "calls_mean": {k: round(statistics.fmean(v), 3) for k, v in sorted(calls.items())},
        "fallback_level": {
            level: sum(1 for r in normal if r["fallback_level"] == level)
            for level in sorted({r["fallback_level"] for r in normal})
        },
    }


def cmd_arm(args: argparse.Namespace) -> int:
    """Materialize a candidate configuration as its own bot directory.

    A configuration is part of the content hash, so two knob settings are two
    rated entities and a decision arm needs both to exist at once. This is the
    `bots/macaria_base/` pattern from the macaria hunt: a copy that lives only
    for the duration of the comparison, registered like any other bot, removed
    afterwards with its registry entry left behind as provenance.

    `target/` is not copied — it is 337 MB of build output and `run.sh` rebuilds
    on a cold tree anyway, which costs ninety seconds once.
    """
    import shutil

    name = args.name
    dest = REPO / "bots" / name
    if dest.exists():
        if not args.force:
            print(f"{dest} exists; pass --force to replace it", file=sys.stderr)
            return 2
        shutil.rmtree(dest)
    shutil.copytree(
        BOT, dest,
        ignore=shutil.ignore_patterns("target", "__pycache__", "data", "*.pyc"),
    )
    config = parse_slug(args.config)
    original = json.loads((dest / "deployment.json").read_text(encoding="utf-8"))
    updated = dict(original)
    updated.update(config)
    if args.extra:
        updated.update(args.extra)
    updated["max_proposal_batch"] = int(config["n_particles"])
    updated["belief_limitation_note"] = (
        f"M7 decision arm, config {args.config}"
        + (f" with {json.dumps(args.extra)}" if args.extra else "")
        + ". A temporary sibling of bots/morpheus-rs so both knob settings can "
        "be rated in one round; removed once the contrast is read."
    )
    (dest / "deployment.json").write_text(
        json.dumps(updated, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {dest}")
    print(f"  config: {json.dumps(config)}")
    if args.extra:
        print(f"  extra:  {json.dumps(args.extra)}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    out = Path(args.out)
    rows = []
    for directory in sorted(p for p in out.iterdir() if p.is_dir()):
        summary = summarize(directory)
        if summary:
            rows.append((directory.name, summary))
    if not rows:
        print(f"no sweep directories under {out}", file=sys.stderr)
        return 1
    print(f"{'config':<20} {'turns':>6} {'move p50':>9} {'p99':>7} {'p99.9':>7} "
          f"{'max':>7} {'>150':>5} {'sims p50':>9} {'belief ok':>10}")
    for name, s in rows:
        m = s["move_ms"]
        print(f"{name:<20} {s['turns']:>6} {m['p50']:>9.1f} {m['p99']:>7.1f} "
              f"{m['p999']:>7.1f} {m['max']:>7.1f} {s['over_judge_limit']:>5} "
              f"{s['completed_simulations']['p50']:>9.1f} "
              f"{s['belief_plus_root_ok_frac'] * 100:>9.1f}%")
    path = out / "summary.json"
    path.write_text(json.dumps({name: s for name, s in rows}, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sweep = sub.add_parser("sweep", help="play games at one or more configs")
    sweep.add_argument("--probe", action="store_true", help="the cost-model probe set")
    sweep.add_argument("--config", nargs="*", default=None, help="config slugs, e.g. n16-s32-b4-d4")
    sweep.add_argument("--games", type=int, default=2)
    sweep.add_argument("--seed", type=int, default=5000)
    sweep.add_argument("--out", type=Path, default=DEFAULT_OUT)
    sweep.add_argument("--extra", type=json.loads, default=None,
                       help="extra deployment.json overrides as JSON")
    sweep.set_defaults(func=cmd_sweep)

    arm = sub.add_parser("arm", help="materialize a config as a sibling bot dir")
    arm.add_argument("--name", required=True, help="bot id, e.g. morpheus-rs-s32")
    arm.add_argument("--config", required=True, help="config slug")
    arm.add_argument("--extra", type=json.loads, default=None)
    arm.add_argument("--force", action="store_true")
    arm.set_defaults(func=cmd_arm)

    report = sub.add_parser("report", help="summarize every swept config")
    report.add_argument("--out", type=Path, default=DEFAULT_OUT)
    report.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
