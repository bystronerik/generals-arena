"""Dump joe's per-turn decision detail over a turn window, next to the board.

Used to read the mechanism of a confinement span: what the top five actions were,
how the army sat on the cells joe kept choosing, and what the value head said.

    python scripts/inspect_joe_window.py <traj-dir> <game-id> <seat> <t0> <t1>
"""
from __future__ import annotations

import sys
from pathlib import Path

from arena.records.trajectories import (
    read_jsonl_gz,
    read_trajectory,
    trace_path,
    trajectory_path,
)

SEAT_PLAYER = {"a": 0, "b": 1}


def main(argv: list[str]) -> int:
    traj_dir, game_id, seat = Path(argv[1]), argv[2], argv[3]
    t0, t1 = int(argv[4]), int(argv[5])
    player = SEAT_PLAYER[seat]

    rows = {r["t"]: r for r in read_jsonl_gz(trace_path(game_id, seat, traj_dir))}
    traj = read_trajectory(trajectory_path(game_id, traj_dir))

    import numpy as np
    from arena.records.trajectories import replay_states
    from generals import get_observation

    print(f"{game_id} seat {seat}, turns {t0}-{t1}\n")
    for turn, state, _ in replay_states(traj):
        if not (t0 <= turn <= t1):
            continue
        row = rows.get(turn)
        if row is None:
            continue
        obs = get_observation(state, player)
        armies = np.asarray(obs.armies)
        owned = np.asarray(obs.owned_cells)
        castles = np.asarray(obs.castles)
        mine = armies * owned
        # The biggest stacks joe owns, which is what a greedy policy should move.
        flat = np.argsort(mine, axis=None)[::-1][:3]
        stacks = [(int(i // mine.shape[1]), int(i % mine.shape[1]),
                   int(mine.flat[i])) for i in flat if mine.flat[i] > 1]
        own_castles = [(int(r), int(c)) for r, c in np.argwhere(castles & owned)]
        print(f"t={turn:<4} land={int(owned.sum()):<4} "
              f"v={row['joe_value_milli']:<6} m={row['joe_margin_milli']:<6} "
              f"H={row['joe_entropy_milli']:<5} stacks={stacks} "
              f"castles={own_castles}")
        print(f"       top5: {row['joe_top5']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
