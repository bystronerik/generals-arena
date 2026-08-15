"""Test whether joe's confident moves are moves onto its own structures.

The confinement spans look like a stack walking on and off a self-built castle.
If that is the mechanism, then confidence should be asymmetric: high margin on
the turns that move army *onto* an own structure, near-indifference on the turns
that move it off. This measures that split, and also how much army joe leaves
parked on its general while it does so.

    python scripts/inspect_joe_castle_pull.py <traj-dir> <game-id> <seat> [t0 t1]

`t0 t1` restricts the margin split to one turn window. The asymmetry is a
property of the confinement span, so measuring it over a whole game — most of
which is ordinary play — dilutes it away.
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
# decode_action direction order: UP, DOWN, LEFT, RIGHT.
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def parse(row: dict) -> tuple[str, int, int, int] | None:
    """('move'|'half'|'build', row, col, dir) for the chosen action; None on pass."""
    head = row["joe_top5"].split("|")[0].rsplit(":", 1)[0]
    if head == "pass":
        return None
    kind, rest = head.split("@")
    if kind == "build":
        r, c = rest.split(".")
        return "build", int(r), int(c), -1
    coords, d = rest.split("d")
    r, c = coords.split(".")
    return kind, int(r), int(c), int(d)


def main(argv: list[str]) -> int:
    traj_dir, game_id, seat = Path(argv[1]), argv[2], argv[3]
    t0, t1 = (int(argv[4]), int(argv[5])) if len(argv) > 5 else (0, 10 ** 9)
    player = SEAT_PLAYER[seat]

    rows = {r["t"]: r for r in read_jsonl_gz(trace_path(game_id, seat, traj_dir))
            if t0 <= r["t"] <= t1}
    traj = read_trajectory(trajectory_path(game_id, traj_dir))

    import numpy as np
    from arena.records.trajectories import replay_states
    from generals import get_observation

    onto = []      # margins for moves whose destination is an own structure
    off = []       # margins for moves whose source is an own structure
    neither = []
    general_hoard = []
    prev_state = None

    for turn, state, _ in replay_states(traj):
        row = rows.get(turn)
        # The decision for turn `t` was made on the board *before* it, so the
        # structure map has to come from the previous state, not this one.
        if row is not None and prev_state is not None:
            obs = get_observation(prev_state, player)
            owned = np.asarray(obs.owned_cells)
            structure = (np.asarray(obs.castles) | np.asarray(obs.generals)) & owned
            armies = np.asarray(obs.armies)
            gen = np.argwhere(np.asarray(obs.generals) & owned)
            if len(gen):
                gr, gc = int(gen[0][0]), int(gen[0][1])
                general_hoard.append((turn, int(armies[gr, gc]),
                                      int((armies * owned).sum())))
            act = parse(row)
            if act is not None and act[0] != "build":
                _, r, c, d = act
                dr, dc = DIRS[d]
                dest = (r + dr, c + dc)
                m = row["joe_margin_milli"]
                in_bounds = (0 <= dest[0] < structure.shape[0]
                             and 0 <= dest[1] < structure.shape[1])
                if in_bounds and structure[dest]:
                    onto.append(m)
                elif structure[r, c]:
                    off.append(m)
                else:
                    neither.append(m)
        prev_state = state

    def stat(name: str, xs: list[int]) -> None:
        if not xs:
            print(f"{name:<34} n=0")
            return
        s = sorted(xs)
        big = sum(1 for x in xs if x > 5000)
        print(f"{name:<34} n={len(xs):<5} median={s[len(s) // 2]:<7} "
              f"p90={s[int(len(s) * 0.9)]:<7} share margin>5.0={100.0 * big / len(xs):.0f}%")

    print(f"{game_id} seat {seat}\n")
    print("margin (milli-logits) of the chosen move, split by what it does:")
    stat("moves ONTO an own structure", onto)
    stat("moves OFF an own structure", off)
    stat("moves touching no structure", neither)

    print("\narmy parked on joe's own general vs joe's total army:")
    for turn, on_gen, total in general_hoard[::60]:
        share = f"{100.0 * on_gen / total:.0f}%" if total else "-"
        print(f"  t={turn:<4} general={on_gen:<5} total={total:<6} ({share} idle "
              f"on the general)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
