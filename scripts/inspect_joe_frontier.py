"""What joe's expansion frontier looked like while it was confined.

Distinguishes two very different reasons a greedy policy stops expanding:

- **no frontier**: every cell next to joe's territory is a mountain or an enemy
  tile it cannot afford. Then the shuffling is a symptom and the real failure is
  that joe never gathers a stack big enough to break out.
- **frontier available**: neutral cells sit next to joe's territory and it walks
  past them. Then the policy itself is the failure.

Reports both per band, together with the army joe is holding in reserve — the
biggest stack it owns — since that is what a breakout would have to spend.

    python scripts/inspect_joe_frontier.py <traj-dir> <game-id> <seat> [band]
"""
from __future__ import annotations

import sys
from pathlib import Path

from arena.records.trajectories import read_trajectory, trajectory_path

SEAT_PLAYER = {"a": 0, "b": 1}


def main(argv: list[str]) -> int:
    traj_dir, game_id, seat = Path(argv[1]), argv[2], argv[3]
    band = int(argv[4]) if len(argv) > 4 else 50
    player = SEAT_PLAYER[seat]

    traj = read_trajectory(trajectory_path(game_id, traj_dir))

    import numpy as np
    from arena.records.trajectories import replay_states
    from generals import get_observation

    print(f"{game_id} seat {seat}\n")
    print(f"{'turn':<7}{'land':<7}{'neutral adj':<14}{'enemy adj':<12}"
          f"{'cheapest enemy':<16}{'top stack':<12}{'reserve share'}")
    for turn, state, _ in replay_states(traj):
        if turn % band or turn == 0:
            continue
        obs = get_observation(state, player)
        owned = np.asarray(obs.owned_cells)
        armies = np.asarray(obs.armies)
        mountains = np.asarray(obs.mountains)
        opp = np.asarray(obs.opponent_cells)

        # Cells orthogonally adjacent to something joe owns.
        adj = np.zeros_like(owned)
        adj[1:, :] |= owned[:-1, :]
        adj[:-1, :] |= owned[1:, :]
        adj[:, 1:] |= owned[:, :-1]
        adj[:, :-1] |= owned[:, 1:]
        frontier = adj & ~owned & ~mountains

        neutral_adj = int((frontier & ~opp).sum())
        enemy_adj = int((frontier & opp).sum())
        enemy_costs = armies[frontier & opp]
        cheapest = int(enemy_costs.min()) if enemy_costs.size else -1

        mine = armies * owned
        top = int(mine.max())
        total = int(mine.sum())
        share = f"{100.0 * top / total:.0f}%" if total else "-"
        print(f"{turn:<7}{int(owned.sum()):<7}{neutral_adj:<14}{enemy_adj:<12}"
              f"{cheapest:<16}{top:<12}{share}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
