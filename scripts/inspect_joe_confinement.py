"""How confined joe's play is, and what the board looked like while it was.

`inspect_joe_oscillation.py` measures strict short-period loops. This measures
the broader symptom those loops sit inside: joe acting in one small pocket for
hundreds of turns while the opponent expands. It replays the engine trajectory
so the confinement can be read against ground truth — where the two generals
are, how much of the board joe has seen, and whether the enemy general was ever
inside joe's fog view.

    python scripts/inspect_joe_confinement.py <traj-dir> <game-id> <seat> [band]
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


def cell(row: dict) -> tuple[int, int]:
    """The (row, col) the chosen action acts from; (-1, -1) for a pass."""
    head = row["joe_top5"].split("|")[0].rsplit(":", 1)[0]
    if head == "pass":
        return (-1, -1)
    coords = head.split("@")[1].split("d")[0]
    r, c = coords.split(".")
    return int(r), int(c)


def top5_cells(row: dict) -> list[tuple[int, int]]:
    out = []
    for entry in row["joe_top5"].split("|"):
        head = entry.rsplit(":", 1)[0]
        if head == "pass":
            continue
        r, c = head.split("@")[1].split("d")[0].split(".")
        out.append((int(r), int(c)))
    return out


def main(argv: list[str]) -> int:
    traj_dir, game_id, seat = Path(argv[1]), argv[2], argv[3]
    band = int(argv[4]) if len(argv) > 4 else 50
    player = SEAT_PLAYER[seat]

    rows = list(read_jsonl_gz(trace_path(game_id, seat, traj_dir)))
    traj = read_trajectory(trajectory_path(game_id, traj_dir))
    cells = [cell(r) for r in rows]

    print(f"{game_id} seat {seat} (player {player}), {len(rows)} turns\n")
    print(f"per-{band}-turn band: how confined the acting cells are")
    print(f"{'turns':<12}{'distinct':<10}{'top cell':<14}{'top share':<11}"
          f"{'passes':<8}{'margin med':<12}{'entropy med':<12}{'value med'}")
    for start in range(0, len(rows), band):
        chunk = rows[start:start + band]
        cc = cells[start:start + band]
        moves = [x for x in cc if x != (-1, -1)]
        distinct = len(set(moves))
        top = max(set(moves), key=moves.count) if moves else None
        share = f"{100.0 * moves.count(top) / len(moves):.0f}%" if moves else "-"
        m = sorted(r["joe_margin_milli"] for r in chunk)
        h = sorted(r["joe_entropy_milli"] for r in chunk)
        v = sorted(r["joe_value_milli"] for r in chunk)
        print(f"{chunk[0]['t']}-{chunk[-1]['t']:<8}{distinct:<10}"
              f"{str(top):<14}{share:<11}{len(cc) - len(moves):<8}"
              f"{m[len(m) // 2]:<12}{h[len(h) // 2]:<12}{v[len(v) // 2]}")

    # Are the runner-up actions anywhere else, or the same pocket?
    same_pocket = 0
    for row, chosen in zip(rows, cells):
        if chosen == (-1, -1):
            continue
        alts = top5_cells(row)[1:]
        if alts and all(abs(a[0] - chosen[0]) + abs(a[1] - chosen[1]) <= 2
                        for a in alts):
            same_pocket += 1
    moved = sum(1 for x in cells if x != (-1, -1))
    print(f"\nturns where all top-5 runners-up are within 2 steps of the "
          f"chosen cell: {same_pocket}/{moved} "
          f"({100.0 * same_pocket / moved:.1f}%)")

    # Engine truth: generals, and what joe could see.
    print("\nreplaying engine states for ground truth ...")
    from arena.records.trajectories import replay_states
    import numpy as np
    from generals import get_observation

    gen_rc = None
    opp_gen_rc = None
    sighted = None
    owned_band: list[tuple[int, int]] = []
    for turn, state, _ in replay_states(traj):
        obs = get_observation(state, player)
        generals = np.asarray(obs.generals)
        owned = np.asarray(obs.owned_cells)
        if gen_rc is None:
            own_gen = np.argwhere(generals & owned)
            if len(own_gen):
                gen_rc = tuple(int(x) for x in own_gen[0])
        # The enemy general is visible when a general cell is not ours.
        enemy_gen = np.argwhere(generals & ~owned)
        if len(enemy_gen) and sighted is None:
            sighted = turn
            opp_gen_rc = tuple(int(x) for x in enemy_gen[0])
        if turn % band == 0:
            owned_band.append((turn, int(owned.sum())))

    print(f"joe's own general: {gen_rc}")
    print(f"first turn the enemy general entered joe's view: {sighted}"
          f"  (at {opp_gen_rc})" if sighted else
          f"the enemy general never entered joe's view")
    if gen_rc:
        far = [(abs(c[0] - gen_rc[0]) + abs(c[1] - gen_rc[1]))
               for c in cells if c != (-1, -1)]
        far_sorted = sorted(far)
        print(f"manhattan distance from joe's general to the acting cell: "
              f"med={far_sorted[len(far_sorted) // 2]} max={far_sorted[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
