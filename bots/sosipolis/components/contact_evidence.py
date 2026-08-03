"""Fused contact evidence: every contact point votes, not just the first.

The hunt used to key off `first_contact` — one cell, latched for the whole
game. A second and third contact somewhere else could not move the estimate,
so a general reached by a short path through the later contacts stayed ranked
below the fog behind the first one.

Two fields accumulate over *all* remembered enemy land instead:

- **density** — how much enemy land sits around a cell. Their base region is
  thick; the snakes they push into us are one tile wide. Two contact clusters
  near the same candidate add up, which is the fusion the single anchor could
  not do.
- **openness** — how much of the area around a cell we have never seen. Their
  general is in the part of their region we have not scouted, so a candidate
  whose neighbourhood is already revealed is worth less than an equally dense
  one that is not.

Both are window counts over a summed-area table, so a refresh is O(H·W) and a
lookup is O(1) — the whole candidate set costs the same as a few BFS steps.
"""
from __future__ import annotations

from params import Params


Cell = tuple[int, int]


class _WindowField:
    """Summed-area table over a 0/1 grid: window counts in constant time."""

    def __init__(self, H: int, W: int, hot) -> None:
        self.H = H
        self.W = W
        sat = [[0] * (W + 1) for _ in range(H + 1)]
        for r in range(H):
            row = 0
            cur, prev = sat[r + 1], sat[r]
            for c in range(W):
                if hot(r, c):
                    row += 1
                cur[c + 1] = prev[c + 1] + row
        self._sat = sat

    def window(self, r: int, c: int, radius: int) -> tuple[int, int]:
        """(hot count, cells looked at) in the square window around (r, c)."""
        r0 = max(0, r - radius)
        c0 = max(0, c - radius)
        r1 = min(self.H - 1, r + radius)
        c1 = min(self.W - 1, c + radius)
        s = self._sat
        count = s[r1 + 1][c1 + 1] - s[r0][c1 + 1] - s[r1 + 1][c0] + s[r0][c0]
        return count, (r1 - r0 + 1) * (c1 - c0 + 1)


class ContactEvidence:
    """Density / openness fields rebuilt when what we know changes."""

    def __init__(self, params: Params) -> None:
        self.params = params
        self._enemy: _WindowField | None = None
        self._unseen: _WindowField | None = None
        self._enemy_epoch = -1
        self._terrain_epoch = -1

    def refresh(self, mem) -> None:
        """Rebuild on new enemy observations or newly revealed terrain."""
        if (
            self._enemy is not None
            and self._enemy_epoch == mem.enemy_obs_epoch
            and self._terrain_epoch == mem.terrain_epoch
        ):
            return
        seen = mem.enemy_seen
        self._enemy = _WindowField(
            mem.H, mem.W, lambda r, c: (r, c) in seen
        )
        ever = mem.ever_seen
        self._unseen = _WindowField(
            mem.H, mem.W, lambda r, c: not ever[r][c]
        )
        self._enemy_epoch = mem.enemy_obs_epoch
        self._terrain_epoch = mem.terrain_epoch

    def has_evidence(self) -> bool:
        return self._enemy is not None

    def anchor(self, mem, depth: dict[Cell, int]) -> Cell | None:
        """One cell standing for *all* contacts: the deepest thick one.

        `first_contact` answers "where did we meet them first", which stops
        being the right question the moment a second contact shows up
        somewhere else. This answers "which contact region is most like a
        home region and furthest into their side", and it moves as evidence
        accumulates.
        """
        if not mem.enemy_seen:
            return None
        home = mem.own_general
        if home is None:
            return None

        def score(cell: Cell) -> tuple[float, int, Cell]:
            straight = abs(cell[0] - home[0]) + abs(cell[1] - home[1])
            deep = depth.get(cell, straight)
            return ((1.0 + self.density(cell)) * deep, deep, cell)

        return max(mem.enemy_seen, key=score)

    def density(self, cell: Cell) -> float:
        """Share of the window around `cell` we have seen as enemy land."""
        if self._enemy is None:
            return 0.0
        count, area = self._enemy.window(
            cell[0], cell[1], self.params.CONTACT_EVIDENCE_RADIUS
        )
        return count / area if area else 0.0

    def openness(self, cell: Cell) -> float:
        """Share of the window around `cell` we have never seen."""
        if self._unseen is None:
            return 0.0
        count, area = self._unseen.window(
            cell[0], cell[1], self.params.CONTACT_EVIDENCE_RADIUS
        )
        return count / area if area else 0.0
