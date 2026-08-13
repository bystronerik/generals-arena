#!/usr/bin/env python3
"""Per-move latency of the deployed joe bot over full competition games.

Phase 5 verification (averagejoe-competition-plan.md section 7): spawn the
real ``bots/joe/run.sh`` subprocess, play full games in a competition-shaped
env (preset kwargs, small map pool) against the engine's random agent, and
time every wire round-trip — frame write to action read, the exact cost the
match runner sees. The first move of each process (JIT compile) is reported
separately: the 10 s first-move grace absorbs it in a match.

This measures the local machine. The x86 budget evidence is the Phase 2
Modal benchmark (docs/research/measurements/joe-phase2-cpu-latency.md);
this script exists to catch a deployment-path regression (a stray
recompile, an unpinned thread pool), not to restate the budget.

    .venv/bin/python scripts/joe_bot_latency.py --games 3
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "competition-module"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--bot", default=str(REPO / "bots" / "joe" / "run.sh"))
    args = parser.parse_args()

    import jax.random as jrandom
    import numpy as np

    from competition.protocol import encode_observation
    from generals.core.action import sample_valid_action
    from generals.core.game import get_observation

    from training.joe.env import make_competition_env, preset_min_generals_distance

    # The preset draws H and W independently in 18..21 — 16 size combos, so
    # the pool must be at least 16 to get one board per combo.
    env = make_competition_env(preset_min_generals_distance(), None, pool_size=32)
    key = jrandom.PRNGKey(0)
    key, reset_key = jrandom.split(key)
    pool, _ = env.reset(reset_key)

    times_ms: list[float] = []
    first_moves_ms: list[float] = []
    outcomes: list[str] = []

    for game in range(args.games):
        key, init_key = jrandom.split(key)
        state = env.init_state(init_key)
        H, W = int(state.armies.shape[0]), int(state.armies.shape[1])

        proc = subprocess.Popen(
            [args.bot], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            cwd=str(Path(args.bot).parent), text=True,
            env={"PATH": "/usr/bin:/bin",
                 "PYTHON": str(REPO / ".venv" / "bin" / "python")})
        proc.stdin.write(f"0 {H} {W}\n")
        proc.stdin.flush()

        outcome = "truncated"
        for turn in range(env.truncation):
            obs = get_observation(state, 0)
            frame = encode_observation(obs)
            t0 = time.perf_counter()
            proc.stdin.write(frame)
            proc.stdin.flush()
            reply = proc.stdout.readline()
            dt_ms = (time.perf_counter() - t0) * 1e3
            (first_moves_ms if turn == 0 else times_ms).append(dt_ms)

            import jax.numpy as jnp
            a0 = jnp.array([int(x) for x in reply.split()], dtype=jnp.int32)
            key, k1 = jrandom.split(key)
            a1 = sample_valid_action(k1, get_observation(state, 1))
            timestep, state = env.step(state, jnp.stack([a0, a1]), pool)
            if bool(timestep.terminated | timestep.truncated):
                winner = int(timestep.info.winner)
                outcome = {0: "win", 1: "loss"}.get(winner, "draw")
                break

        proc.stdin.close()
        proc.wait(timeout=15)
        outcomes.append(outcome)
        print(f"game {game + 1}/{args.games}: {outcome} after "
              f"{turn + 1} turns, first move "
              f"{first_moves_ms[-1] / 1e3:.1f} s", flush=True)

    arr = np.array(times_ms)
    print(f"\nmoves timed (excl. first): {arr.size} over {args.games} games "
          f"({', '.join(outcomes)})")
    print(f"first move (startup + JIT): "
          f"{', '.join(f'{x / 1e3:.1f}s' for x in first_moves_ms)}")
    print(f"p50 {np.percentile(arr, 50):.1f} ms | "
          f"p90 {np.percentile(arr, 90):.1f} ms | "
          f"p99 {np.percentile(arr, 99):.1f} ms | "
          f"max {arr.max():.1f} ms")


if __name__ == "__main__":
    main()
