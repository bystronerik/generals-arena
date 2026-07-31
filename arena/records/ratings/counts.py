"""
Games -> integer sufficient statistics.

This module is where order-independence is won. The likelihood reads the games
only through counts per **ordered** pair (seat matters), and building that
table is integer addition — exact, commutative, associative. So the table is
bit-identical for any input order, and `CountTable.digest` proves it: two runs
over the same games in different orders produce the same digest, or something
is wrong upstream.

Count tables are also exactly additive, which is what makes the per-round cache
(`cache.py`) safe: merging per-round tables in any order gives the same result
as aggregating everything in one pass.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from arena.records.ratings.model import CountArrays
from arena.records.ratings.policy import Policy, entity_key, rejection_reason
from arena.records.registry import Registry
from arena.records.store import GameRecord


@dataclass(frozen=True)
class Cell:
    """Every game played by one ordered pair, as three integers."""

    seat_a: str
    seat_b: str
    wins_a: int
    wins_b: int
    draws: int

    @property
    def games(self) -> int:
        return self.wins_a + self.wins_b + self.draws


@dataclass(frozen=True)
class CountTable:
    entities: tuple[str, ...]  # canonical sorted order
    cells: tuple[Cell, ...]  # sorted by (seat_a, seat_b)
    digest: str  # sha256 over the canonical form
    excluded: Mapping[str, int]  # rejection reason -> games dropped

    @property
    def games(self) -> int:
        return sum(cell.games for cell in self.cells)

    def record_for(self, entity: str) -> tuple[int, int, int]:
        """`(wins, losses, draws)` for one entity across both seats."""
        wins = losses = draws = 0
        for cell in self.cells:
            if cell.seat_a == entity:
                wins += cell.wins_a
                losses += cell.wins_b
                draws += cell.draws
            if cell.seat_b == entity:
                wins += cell.wins_b
                losses += cell.wins_a
                draws += cell.draws
        return wins, losses, draws

    def to_arrays(self) -> CountArrays:
        index = {name: i for i, name in enumerate(self.entities)}
        return CountArrays(
            seat_a=np.array([index[c.seat_a] for c in self.cells], dtype=np.intp),
            seat_b=np.array([index[c.seat_b] for c in self.cells], dtype=np.intp),
            wins_a=np.array([c.wins_a for c in self.cells], dtype=float),
            wins_b=np.array([c.wins_b for c in self.cells], dtype=float),
            draws=np.array([c.draws for c in self.cells], dtype=float),
        )

    def to_dict(self) -> dict:
        return {
            "entities": list(self.entities),
            "cells": [
                [c.seat_a, c.seat_b, c.wins_a, c.wins_b, c.draws] for c in self.cells
            ],
            "digest": self.digest,
            "excluded": dict(self.excluded),
        }

    @classmethod
    def from_dict(cls, data: dict) -> CountTable:
        cells = tuple(
            Cell(seat_a=a, seat_b=b, wins_a=int(wa), wins_b=int(wb), draws=int(wd))
            for a, b, wa, wb, wd in data["cells"]
        )
        return build(
            {(c.seat_a, c.seat_b): (c.wins_a, c.wins_b, c.draws) for c in cells},
            entities=data.get("entities", ()),
            excluded=data.get("excluded", {}),
        )


def connected_components(table: CountTable) -> tuple[tuple[str, ...], ...]:
    """
    Groups of entities linked by actual games, largest first.

    The likelihood only ever sees *differences* `theta_i - theta_j`, and only
    for pairs that played. So it is flat along "add a constant to everybody",
    once per component: with one component the anchor pins that constant, but
    with two the second one is pinned by nothing but the prior. Ratings from
    different components are therefore not comparable at all — see
    `RatingFit.delta`, which refuses to contrast across them.

    Entities with no games are their own singleton component, which is exactly
    right: an entity nobody played cannot be placed against anyone.

    Ordering is canonical (size descending, then first entity name), so the
    result is identical for any input order — the same property the count table
    itself guarantees.
    """
    parent = {name: name for name in table.entities}

    def find(name: str) -> str:
        root = name
        while parent[root] != root:
            parent[root] = parent[parent[root]]
            root = parent[root]
        return root

    for cell in table.cells:
        a, b = find(cell.seat_a), find(cell.seat_b)
        if a != b:
            # Union by name keeps the result independent of cell order.
            low, high = (a, b) if a < b else (b, a)
            parent[high] = low

    grouped: dict[str, list[str]] = {}
    for name in table.entities:
        grouped.setdefault(find(name), []).append(name)
    return tuple(
        tuple(sorted(members))
        for members in sorted(grouped.values(), key=lambda m: (-len(m), min(m)))
    )


def component_index(table: CountTable) -> dict[str, int]:
    """`entity -> component number`, numbered by `connected_components` order."""
    return {
        name: number
        for number, members in enumerate(connected_components(table))
        for name in members
    }


def _digest(entities: tuple[str, ...], cells: tuple[Cell, ...]) -> str:
    """
    A hash of the canonical form: the entity list, then every cell in order.

    Deliberately not a hash of the *file* — the point is to be identical across
    two runs that saw the same games in different orders, and to change when
    the games change.
    """
    sha = hashlib.sha256()
    for name in entities:
        sha.update(name.encode("utf-8"))
        sha.update(b"\0")
    sha.update(b"\n")
    for cell in cells:
        sha.update(
            f"{cell.seat_a}\t{cell.seat_b}\t{cell.wins_a}\t{cell.wins_b}\t{cell.draws}\n".encode(
                "utf-8"
            )
        )
    return f"sha256:{sha.hexdigest()}"


def build(
    tallies: Mapping[tuple[str, str], tuple[int, int, int]],
    *,
    entities: Iterable[str] = (),
    excluded: Mapping[str, int] | None = None,
) -> CountTable:
    """Assemble a canonical `CountTable` from `{(seat_a, seat_b): (w_a, w_b, d)}`."""
    names = set(entities)
    for seat_a, seat_b in tallies:
        names.add(seat_a)
        names.add(seat_b)
    ordered = tuple(sorted(names))
    cells = tuple(
        Cell(seat_a=a, seat_b=b, wins_a=wa, wins_b=wb, draws=wd)
        for (a, b), (wa, wb, wd) in sorted(tallies.items())
    )
    return CountTable(
        entities=ordered,
        cells=cells,
        digest=_digest(ordered, cells),
        excluded=dict(sorted((excluded or {}).items())),
    )


def count_table(
    games: Iterable[GameRecord],
    *,
    policy: Policy,
    registry: Registry | None = None,
    extra_entities: Iterable[str] = (),
) -> CountTable:
    """
    Aggregate eligible games into the canonical integer table.

    `extra_entities` forces entities into the fit that played no eligible game
    — the anchor, most importantly, which must exist even in an empty pool.
    They land at exactly the prior mean with `SE = sigma`.
    """
    tallies: dict[tuple[str, str], list[int]] = {}
    excluded: Counter[str] = Counter()

    for record in games:
        reason = rejection_reason(record, policy, registry)
        if reason is not None:
            excluded[reason] += 1
            continue
        key = (
            entity_key(record.bot_a, record.bot_a_content_hash),
            entity_key(record.bot_b, record.bot_b_content_hash),
        )
        cell = tallies.setdefault(key, [0, 0, 0])
        if record.winner == "a":
            cell[0] += 1
        elif record.winner == "b":
            cell[1] += 1
        else:
            cell[2] += 1

    return build(
        {k: (v[0], v[1], v[2]) for k, v in tallies.items()},
        entities=extra_entities,
        excluded=excluded,
    )


def merge(tables: Iterable[CountTable]) -> CountTable:
    """
    Add count tables together. Order-independent by construction.

    Used to fold per-round caches into one pool without re-reading the games.
    """
    tallies: dict[tuple[str, str], list[int]] = {}
    names: set[str] = set()
    excluded: Counter[str] = Counter()
    for table in tables:
        names.update(table.entities)
        excluded.update(table.excluded)
        for cell in table.cells:
            entry = tallies.setdefault((cell.seat_a, cell.seat_b), [0, 0, 0])
            entry[0] += cell.wins_a
            entry[1] += cell.wins_b
            entry[2] += cell.draws
    return build(
        {k: (v[0], v[1], v[2]) for k, v in tallies.items()},
        entities=names,
        excluded=excluded,
    )
