"""Board sections and P(general) priors from spawn + searched area."""
from __future__ import annotations

from typing import Iterable


Cell = tuple[int, int]


class SectionPrior:
    def __init__(self, H: int, W: int, rows: int, cols: int):
        self.H = H
        self.W = W
        self.rows = rows
        self.cols = cols
        self.n = rows * cols
        self.mass = [1.0 / self.n] * self.n

    def section_of(self, r: int, c: int) -> int:
        sr = min(self.rows - 1, r * self.rows // max(1, self.H))
        sc = min(self.cols - 1, c * self.cols // max(1, self.W))
        return sr * self.cols + sc

    def reset_uniform(self) -> None:
        self.mass = [1.0 / self.n] * self.n

    def seed_from_candidates(
        self,
        candidates: Iterable[Cell],
        own_general: Cell | None,
    ) -> None:
        counts = [0.0] * self.n
        for r, c in candidates:
            counts[self.section_of(r, c)] += 1.0
        if own_general is not None:
            # Soft penalty on own section.
            own_s = self.section_of(*own_general)
            counts[own_s] *= 0.15
        total = sum(counts)
        if total <= 0:
            self.reset_uniform()
            return
        self.mass = [c / total for c in counts]

    def reweight_contact(self, contact: Cell, weight: float) -> None:
        """Shift mass toward the contact section; keep residual elsewhere."""
        s = self.section_of(*contact)
        w = max(0.0, min(1.0, weight))
        new = [(1.0 - w) * m for m in self.mass]
        new[s] += w
        total = sum(new)
        self.mass = [m / total for m in new]

    def prune_to_candidates(self, candidates: Iterable[Cell]) -> None:
        alive = set()
        for r, c in candidates:
            alive.add(self.section_of(r, c))
        new = [self.mass[i] if i in alive else 0.0 for i in range(self.n)]
        total = sum(new)
        if total <= 0:
            self.reset_uniform()
            return
        self.mass = [m / total for m in new]

    def top_section(self) -> int:
        return max(range(self.n), key=lambda i: self.mass[i])

    def score_cell(self, r: int, c: int) -> float:
        return self.mass[self.section_of(r, c)]
