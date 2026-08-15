#!/usr/bin/env python3
"""joe-net N0: does the search budget survive a 21 ms forward?

Phase N0 of docs/bots/morpheus-rs/joe-net-plan.md — the kill gate that runs
before any of joe's network is ported. Three commands, in the order §5.4
names them:

    # N0.2 — what the shipped bot spends its forwards on, per turn and per call
    python scripts/joe_net_n0.py costs data/morpheus/morpheus-rs/joe-net-n0/baseline/n8-s16-b4-d8

    # N0.4 — turn those costs into the spike's deployment.json overrides,
    #        with the two network components repriced at joe's measured forward
    python scripts/joe_net_n0.py spike-config <baseline-dir> --forward-ms 23.01

    # the write-up's numbers, both arms side by side
    python scripts/joe_net_n0.py report --baseline <dir> --spike <dir>

The costs come from morpheus-rs's own trace (`MORPHEUS_RS_TRACE`), which
`scripts/morpheus_rs_m7.py sweep` arms per game. **Per call, never per
turn**: the admission controller charges a component every time it runs, so
a per-turn total at sixteen simulations is sixteen times the number the
controller actually forecasts for a single leaf batch.

The forward figure comes from `joe-rs bench --stages` on one x86 core
(scripts/joe_rs_modal_stages.py), because the whole point of N0 is to ask
what the controller does at a cost this binary cannot yet produce.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

COMPONENTS = (
    "belief_tensor",
    "belief_proposal",
    "particle_transitions",
    "hashing",
    "root_inference",
    "leaf_batch",
    "enemy_prior_batch",
    "backup",
    "reply",
    "selection",
)

CONSUMERS = ("belief_proposal", "root", "enemy_prior", "leaf_batch")

# `normal_deadline_ms` in the shipped deployment.json, and nothing subtracted.
#
# Not `normal_deadline_ms - reserve_ms`, which is what the plan assumes and what
# the file's own naming implies. **`reserve_ms` is inert in both bots**: it is
# parsed, stored on `RuntimeConfig`, carried through `to_runtime_config`, and
# read by nothing. `deadline_for_turn` is `turn_start + normal_deadline_ms`.
# So the controller spends the whole 140 ms and the 10 ms of safety margin the
# config claims to hold back is not held back — which is the same defect as the
# inert `min_simulations` the plan already records, and worth about half a
# forward to N0's arithmetic. The spike confirms it: a median-belief turn
# reports a virtual `move_ms` of exactly 140.
USABLE_MS = 140.0


def nearest_rank(values: Sequence[float], q: float) -> float:
    """The estimator the controller deploys, so the report quotes what it acts on."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(-(-q * len(ordered) // 1))))
    return float(ordered[rank - 1])


def stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "p50": 0.0, "p99": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "n": len(values),
        "p50": round(nearest_rank(values, 0.50), 4),
        "p99": round(nearest_rank(values, 0.99), 4),
        "max": round(max(values), 4),
        "mean": round(statistics.fmean(values), 4),
    }


def read_traces(directory: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(directory.glob("game*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"no game*.jsonl traces under {directory}")
    return rows


def summarize(directory: Path) -> dict[str, Any]:
    rows = read_traces(directory)
    # The first move carries load + warmup + belief allocation and answers a
    # different question; every deadline number here is about a normal move.
    normal = [r for r in rows if r["t"] > 1]
    per_call: dict[str, list[float]] = {}
    per_turn: dict[str, list[float]] = {}
    calls: dict[str, list[float]] = {}
    forwards: dict[str, list[float]] = {name: [] for name in CONSUMERS}
    sims = Counter()
    fallback = Counter()

    for row in normal:
        components = row.get("components") or {}
        call_counts = row.get("calls") or {}
        for name in COMPONENTS:
            value = float(components.get(name, 0.0))
            n = int(call_counts.get(name, 0))
            if value > 0.0:
                per_turn.setdefault(name, []).append(value)
            if n > 0:
                per_call.setdefault(name, []).append(value / n)
                calls.setdefault(name, []).append(float(n))
        by_consumer = row.get("forward_by_consumer") or {}
        for name in CONSUMERS:
            forwards[name].append(float(by_consumer.get(name, 0)))
        sims[int(row["completed_simulations"])] += 1
        fallback[str(row["fallback_level"])] += 1

    manifest_path = directory / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    has_consumers = any(
        "forward_by_consumer" in row for row in normal
    )
    return {
        "directory": str(directory),
        "config": manifest.get("config", {}),
        "extra": manifest.get("extra", {}),
        "host": manifest.get("host", {}),
        "measured_at": manifest.get("measured_at", ""),
        "games": len(list(directory.glob("game*.jsonl"))),
        "normal_turns": len(normal),
        "move_ms": stats([float(r["move_ms"]) for r in normal]),
        "completed_simulations": stats(
            [float(r["completed_simulations"]) for r in normal]
        ),
        "completed_simulations_hist": dict(sorted(sims.items())),
        # Evidence for the widening freeze, which the trace does not record
        # directly: a root that never widens keeps a small tree.
        "tree_size": stats([float(r["tree_size"]) for r in normal]),
        "fallback_level": dict(fallback.most_common()),
        "forward_equivalents": stats(
            [float(r["forward_equivalents"]) for r in normal]
        ),
        "forward_by_consumer_mean": (
            {name: round(statistics.fmean(v), 3) for name, v in forwards.items()}
            if has_consumers
            else None
        ),
        "forward_by_consumer_total": (
            {name: int(sum(v)) for name, v in forwards.items()} if has_consumers else None
        ),
        "per_turn_p99_ms": {
            name: round(nearest_rank(v, 0.99), 4) for name, v in sorted(per_turn.items())
        },
        "per_call_p99_ms": {
            name: round(nearest_rank(v, 0.99), 4) for name, v in sorted(per_call.items())
        },
        # The belief is bimodal by two orders of magnitude — `filter_step` and
        # the `recover_belief` it may trigger are charged to one component, as
        # `runtime/controller.rs` says at the top of the file. A spike that
        # charges p99 every turn therefore models a recovery on every turn, so
        # the whole quantile ladder is kept and the arms bracket it.
        "per_call_quantiles_ms": {
            name: {
                f"p{int(q * 100)}": round(nearest_rank(v, q), 4)
                for q in (0.10, 0.50, 0.90, 0.99)
            }
            for name, v in sorted(per_call.items())
        },
        "calls_mean": {
            name: round(statistics.fmean(v), 3) for name, v in sorted(calls.items())
        },
    }


def spike_overrides(
    baseline: dict[str, Any], forward_ms: float, batch: int, quantile: str = "p99"
) -> dict[str, Any]:
    """The `deployment.json` keys that reprice the two network components.

    Every non-network component keeps its measured per-call cost, so the spike
    changes exactly one thing: what a forward costs. `enemy_prior_batch` and
    `belief_tensor` go to zero because the port deletes both (§3.4, §3.2) — not
    because they are free.

    `quantile` picks *which* measured cost the non-network components get. A
    charged clock advances by the forecast, so a table of p99s does not model a
    p99 turn — it models a run in which every turn is a p99 turn, and the belief
    is bimodal enough that this is a different question. Run it at `p50` and at
    `p99` and the answer is bracketed instead of asserted.
    """
    quantiles = baseline["per_call_quantiles_ms"]
    fixed = {
        name: values.get(quantile, 0.0) for name, values in quantiles.items()
    }
    for name in COMPONENTS:
        fixed.setdefault(name, 0.0)
    fixed["root_inference"] = round(forward_ms, 4)
    fixed["leaf_batch"] = round(forward_ms * batch, 4)
    fixed["enemy_prior_batch"] = 0.0
    fixed["belief_tensor"] = 0.0
    return {
        "fixed_forecasts_ms": fixed,
        "charge_fixed_forecasts": True,
        "pending_leaf_batch": batch,
    }


def projection(
    forward_ms: float, baseline: dict[str, Any], batch: int, quantile: str = "p99"
) -> dict[str, Any]:
    """§5.4 step 3, from measured numbers instead of the plan's estimates.

    Admission is gated on the *forecast* and the clock advances by the *cost*,
    and the two are not the same number on a real turn: the controller forecasts
    a component's p99 and then spends whatever it spends. So the gate below uses
    the p99 and the spending uses `quantile`. Setting both to p99 reproduces
    what the charged-clock spike does, which is how the model is checked.
    """
    gate = baseline["per_call_p99_ms"]
    spend = {
        name: values.get(quantile, 0.0)
        for name, values in baseline["per_call_quantiles_ms"].items()
    }
    calls = baseline["calls_mean"]
    # Everything the port keeps and the network does not touch, at the rate the
    # shipped bot runs it. `selection` and `backup` are per simulation and are
    # charged inside the loop below, not here.
    fixed_ms = sum(
        spend.get(name, 0.0) * calls.get(name, 0.0)
        for name in ("belief_proposal", "particle_transitions", "hashing", "reply")
    )
    left = USABLE_MS - fixed_ms
    # `selection` runs more than once per simulation — a re-selection after a
    # missing enemy prior is another call — so it is charged at its measured
    # rate rather than once.
    per_sim_spend = (
        spend.get("selection", 0.0) * calls.get("selection", 0.0) / max(calls.get("backup", 1.0), 1e-9)
        + spend.get("backup", 0.0)
        + forward_ms
    )
    after_root = left - forward_ms
    sims = 0
    remaining = after_root
    # `can_admit` needs the whole batch's forecast to fit before any of it runs.
    batch_gate = gate.get("selection", 0.0) + forward_ms * batch
    while remaining >= batch_gate:
        remaining -= per_sim_spend * batch
        sims += batch
    return {
        "usable_ms": USABLE_MS,
        "spend_quantile": quantile,
        "non_network_ms": round(fixed_ms, 3),
        "left_for_inference_ms": round(left, 3),
        "forward_ms": forward_ms,
        "after_root_ms": round(after_root, 3),
        "per_sim_spend_ms": round(per_sim_spend, 4),
        "batch_gate_ms": round(batch_gate, 4),
        "pending_leaf_batch": batch,
        "projected_simulations": sims,
    }


def cmd_costs(args: argparse.Namespace) -> int:
    print(json.dumps(summarize(Path(args.directory)), indent=2))
    return 0


def cmd_spike_config(args: argparse.Namespace) -> int:
    baseline = summarize(Path(args.baseline))
    overrides = spike_overrides(baseline, args.forward_ms, args.batch, args.quantile)
    if args.projection:
        print(
            json.dumps(
                projection(args.forward_ms, baseline, args.batch, args.quantile),
                indent=2,
            )
        )
        return 0
    # One line, so it can be pasted straight into `--extra`.
    print(json.dumps(overrides))
    return 0


def _table(rows: Iterable[Sequence[Any]], header: Sequence[str]) -> list[str]:
    out = ["| " + " | ".join(header) + " |"]
    out.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return out


def cmd_report(args: argparse.Namespace) -> int:
    baseline = summarize(Path(args.baseline))
    arms = [("baseline", baseline)]
    for label, directory in args.spike or []:
        arms.append((label, summarize(Path(directory))))
    payload = {
        "forward_ms": args.forward_ms,
        "arms": {label: arm for label, arm in arms},
        "projection": {
            f"batch{b}-{q}": projection(args.forward_ms, baseline, b, q)
            for b in (1, 2, 4)
            for q in ("p50", "p90", "p99")
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")

    lines = ["", "## Completed simulations per normal move", ""]
    lines += _table(
        [
            (
                label,
                arm["normal_turns"],
                arm["completed_simulations"]["p50"],
                arm["completed_simulations"]["p99"],
                arm["completed_simulations"]["mean"],
            )
            for label, arm in arms
        ],
        ("arm", "turns", "p50", "p99", "mean"),
    )
    lines += ["", "## Degrade level", ""]
    levels = sorted({k for _, arm in arms for k in arm["fallback_level"]})
    lines += _table(
        [
            tuple([label] + [arm["fallback_level"].get(k, 0) for k in levels])
            for label, arm in arms
        ],
        tuple(["arm"] + levels),
    )
    print("\n".join(lines))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    costs = sub.add_parser("costs", help="per-call component costs from a trace dir")
    costs.add_argument("directory")
    costs.set_defaults(func=cmd_costs)

    spike = sub.add_parser("spike-config", help="the --extra JSON for the N0.4 spike")
    spike.add_argument("baseline")
    spike.add_argument("--forward-ms", type=float, required=True)
    spike.add_argument("--batch", type=int, default=4)
    spike.add_argument("--quantile", default="p99", choices=("p10", "p50", "p90", "p99"),
                       help="which measured cost the non-network components get")
    spike.add_argument("--projection", action="store_true",
                       help="print §5.4 step 3's arithmetic instead of the overrides")
    spike.set_defaults(func=cmd_spike_config)

    report = sub.add_parser("report", help="both arms, side by side")
    report.add_argument("--baseline", required=True)
    report.add_argument("--spike", nargs=2, action="append", metavar=("LABEL", "DIR"))
    report.add_argument("--forward-ms", type=float, required=True)
    report.add_argument("--out", default="docs/research/measurements/joe-net-n0.json")
    report.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
