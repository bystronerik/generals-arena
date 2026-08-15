#!/usr/bin/env python3
"""The M4 exit-gate measurement: how much faster is the belief update?

    python bots/morpheus-joe/tools/bench_belief.py --smoke
    python bots/morpheus-joe/tools/bench_belief.py --corpus data/morpheus/morpheus-rs/morpheus-rs-m0

rewrite-plan §4 asks M4 for "measured particle-transitions time at n=8 (expect
>=10x vs Python)". This runs both implementations over the **same recorded
beliefs** and reports the distribution across them.

Two deliberate choices about what is being timed:

* **Same inputs, not same games.** The M0 baseline measured
  `particle_transitions` under live play, where the deadline controller decides
  how often the update runs at all. That number answers "what does a turn
  cost"; this one answers "what does the kernel cost", which is the one a port
  can be held to. The report quotes both.
* **Minimum per case, distribution across cases.** Repeating one belief and
  taking a p99 measures the machine's scheduling noise. Taking the minimum per
  belief and the p50/p99 *across* beliefs measures the code against the range of
  inputs it actually sees.

Both sides run the pair the runtime charges separately — `belief_proposal`
(`propose_enemy_actions`) and `particle_transitions` (`filter_step`) — because
that is how `deployment.json` budgets them and how M7 will re-derive them.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
BOT_DIR = TOOLS_DIR.parent
REPO = BOT_DIR.parents[1]
for entry in (REPO, REPO / "bots", REPO / "bots" / "morpheus", BOT_DIR / "tests"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import morpheus_joe_parity_cases as pc  # noqa: E402

BINARY = BOT_DIR / "target" / "release" / "morpheus-joe"


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def pick(q: float) -> float:
        return ordered[max(1, int(np.ceil(q * len(ordered)))) - 1]

    return {
        "n": len(ordered),
        "p50": pick(0.50),
        "p99": pick(0.99),
        "max": ordered[-1],
        "mean": float(np.mean(ordered)),
    }


def python_timings(cases: list, iters: int):
    """Minimum wall time per belief, for the two blocks the runtime charges.

    The update block is `filter_step` **and** `recover_belief` when nothing
    survived — copied from `runtime.py`, where both sit inside one
    `particle_transitions` charge. Splitting them here would produce a
    flattering number that no deployment budget ever sees.
    """
    from belief import filter_step
    from proposal import propose_enemy_actions
    from recovery import recover_belief

    propose_ms: list[float] = []
    update_ms: list[list[float]] = []
    survivors: list[list[int]] = []
    for case in cases:
        belief, my_action, memory = case["belief"], case["my_action"], case["memory"]
        rng = np.random.default_rng(0x5EED)
        propose_enemy_actions(belief, rng, policy=None, max_proposal_batch=8)

        best_propose = float("inf")
        for _ in range(iters):
            t0 = time.perf_counter()
            propose_enemy_actions(belief, rng, policy=None, max_proposal_batch=8)
            best_propose = min(best_propose, (time.perf_counter() - t0) * 1e3)
        propose_ms.append(best_propose)

        enemy_actions = propose_enemy_actions(
            belief, rng, policy=None, max_proposal_batch=8
        )
        per_target, per_target_survivors = [], []
        for real_obs in case["targets"]:
            best = float("inf")
            kept = 0
            for _ in range(iters):
                t1 = time.perf_counter()
                nxt = filter_step(belief, my_action, real_obs, enemy_actions, rng)
                if not any(p.weight > 0.0 for p in nxt.particles):
                    nxt = recover_belief(
                        belief, my_action, real_obs, memory, rng, policy=None
                    )
                best = min(best, (time.perf_counter() - t1) * 1e3)
                kept = sum(1 for p in nxt.particles if p.weight > 0.0)
            per_target.append(best)
            per_target_survivors.append(kept)
        update_ms.append(per_target)
        survivors.append(per_target_survivors)
    return propose_ms, update_ms, survivors


def rust_timings(cases: list, iters: int):
    stream: list[int] = [len(cases)]
    for case in cases:
        stream += pc.encode_belief_state(case["belief"])
        stream += list(case["my_action"])
        stream += pc.encode_memory(case["memory"])
        stream += [len(case["targets"])]
        for real_obs in case["targets"]:
            stream += pc.encode_observation(real_obs)

    result = subprocess.run(
        [str(BINARY), "bench-belief", str(iters)],
        input=" ".join(str(v) for v in stream),
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"bench-belief failed: {result.stderr[-2000:]}")
    propose_ms, update_ms, survivors = [], [], []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [int(v) for v in line.split()]
        propose_ms.append(fields[1] / 1e6)
        update_ms.append([v / 1e6 for v in fields[2::2]])
        survivors.append(fields[3::2])
    return propose_ms, update_ms, survivors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=pc.DEFAULT_CORPUS)
    parser.add_argument("--smoke", action="store_true", help="use the committed slice")
    parser.add_argument("--limit", type=int, default=0, help="cap heavy frames")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None, help="write a JSON report")
    args = parser.parse_args(argv)

    if not BINARY.is_file():
        print(f"no release binary at {BINARY}", file=sys.stderr)
        return 2
    paths = (
        [pc.SMOKE_FIXTURE]
        if args.smoke
        else sorted(Path(args.corpus).rglob("*.capture.*.jsonl.gz"))
    )
    if not paths:
        print(f"no captures under {args.corpus}", file=sys.stderr)
        return 1

    from observe import emit_observation
    from transition import transition

    frames = pc.load_frames(paths, limit=args.limit)
    cases = []
    for frame in frames:
        belief = pc.belief_from_frame(frame)
        if belief is None or belief.n == 0 or "memory" not in frame:
            continue
        my_action = tuple(int(v) for v in frame["action"])
        # Two target frames per belief, because the update has two very
        # different costs. Advancing the leading particle gives a frame at
        # least one particle explains — the healthy path, all filter. The
        # frame's own observation refutes every particle — the collapse path,
        # which is where recovery runs and where the M0 baseline's p99 lives.
        enemy_actions = [
            pc._enemy_action_for(belief, p, i) for i, p in enumerate(belief.particles)
        ]
        joint = np.zeros((2, 5), dtype=np.int32)
        joint[belief.seat] = np.asarray(my_action, np.int32)
        joint[belief.enemy_seat] = np.asarray(enemy_actions[0], np.int32)
        nxt, _ = transition(belief.particles[0].state, joint)
        cases.append(
            {
                "belief": belief,
                "my_action": my_action,
                "memory": pc._memory_from_capture(frame["memory"]),
                "targets": [
                    emit_observation(nxt, belief.seat, as_arrays=True),
                    pc._obs_from_capture(frame["obs"]),
                ],
            }
        )

    if not cases:
        print("no usable beliefs in the frames", file=sys.stderr)
        return 1
    sizes = sorted({c["belief"].n for c in cases})
    print(f"{len(cases)} belief(s), particle counts {sizes}, {args.iters} iteration(s) each")

    py_propose, py_update, py_kept = python_timings(cases, args.iters)
    rs_propose, rs_update, rs_kept = rust_timings(cases, args.iters)

    # A target where the two sides disagree about survival is not a timing
    # result, it is a parity failure wearing one. Say so rather than reporting
    # a ratio between two different computations.
    mismatched = sum(
        1
        for a, b in zip(py_kept, rs_kept)
        for x, y in zip(a, b)
        if x != y
    )

    series = {"belief_proposal_ms": (py_propose, rs_propose)}
    for index, label in enumerate(("particle_transitions_ms", "particle_transitions_recovery_ms")):
        series[label] = (
            [row[index] for row in py_update],
            [row[index] for row in rs_update],
        )
    recovered = sum(1 for row in rs_kept for k in row[1:2] if k > 0)

    report = {
        "n_beliefs": len(cases),
        "particle_counts": sizes,
        "iters": args.iters,
        "survivor_mismatches": mismatched,
        "collapse_targets_recovered": recovered,
        "python": {k: _quantiles(v[0]) for k, v in series.items()},
        "rust": {k: _quantiles(v[1]) for k, v in series.items()},
        "speedup": {},
    }
    for component in series:
        py, rs = report["python"][component], report["rust"][component]
        report["speedup"][component] = {
            stat: (py[stat] / rs[stat] if rs[stat] else float("inf"))
            for stat in ("p50", "p99", "max", "mean")
        }

    if mismatched:
        print(f"WARNING: {mismatched} target(s) kept a different number of particles")
    print(f"{'component':<34} {'python p50':>11} {'p99':>9} {'rust p50':>10} {'p99':>9} "
          f"{'x p50':>8} {'x p99':>8}")
    for component in series:
        py, rs = report["python"][component], report["rust"][component]
        gain = report["speedup"][component]
        print(
            f"{component:<34} {py['p50']:>11.3f} {py['p99']:>9.3f} "
            f"{rs['p50']:>10.4f} {rs['p99']:>9.4f} "
            f"{gain['p50']:>7.1f}x {gain['p99']:>7.1f}x"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
