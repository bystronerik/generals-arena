"""Monotonic deadline helpers for the 100 ms move cap."""
from __future__ import annotations

import time


class Deadline:
    __slots__ = ("end",)

    def __init__(self, budget_ms: float):
        self.end = time.monotonic() + max(0.0, budget_ms) / 1000.0

    @classmethod
    def until(cls, end: float) -> "Deadline":
        d = cls.__new__(cls)
        d.end = end
        return d

    def expired(self) -> bool:
        return time.monotonic() >= self.end

    def remaining_ms(self) -> float:
        return max(0.0, (self.end - time.monotonic()) * 1000.0)


def now() -> float:
    return time.monotonic()


def ms_since(start: float) -> int:
    return int((time.monotonic() - start) * 1000.0)
