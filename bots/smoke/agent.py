"""
Minimal smoke strategy — verify stdio + competition mode.

Not competitive. Prefer expanding onto visible neutrals; otherwise take any
legal move; pass if none exist.
"""

# A no-op action — used when no valid move exists.
PASS = (1, 0, 0, 0, 0)

# (dr, dc) offsets for direction codes 0..3
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _is_passable(t):
    # Mountains (2) and fogged-structures (5) are impassable.
    return t != 2 and t != 5


def _is_visible_neutral(owner, cell_type):
    # owner 0 with a visible tile type (not fog / structure-in-fog).
    return owner == 0 and cell_type not in (0, 5)


class Agent:
    """Smoke strategy: expand onto visible neutrals, else any legal move."""

    def __init__(self, player_id, H, W):
        self.player_id = player_id
        self.H = H
        self.W = W

    def act(self, obs):
        expand_move = None
        first_valid = None

        for r in range(obs.H):
            for c in range(obs.W):
                if obs.owner_grid[r][c] != 1:
                    continue
                src_army = obs.army_grid[r][c]
                if src_army <= 1:
                    continue

                for d, (dr, dc) in enumerate(DIRECTIONS):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < obs.H and 0 <= nc < obs.W):
                        continue
                    if not _is_passable(obs.type_grid[nr][nc]):
                        continue

                    move = (0, r, c, d, 0)
                    if first_valid is None:
                        first_valid = move

                    dest_owner = obs.owner_grid[nr][nc]
                    dest_type = obs.type_grid[nr][nc]
                    dest_army = obs.army_grid[nr][nc]

                    if not _is_visible_neutral(dest_owner, dest_type):
                        continue
                    # Need enough army to capture (one stays on source).
                    if src_army <= dest_army + 1:
                        continue

                    if expand_move is None:
                        expand_move = move

        if expand_move is not None:
            return expand_move
        if first_valid is not None:
            return first_valid
        return PASS
