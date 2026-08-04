"""Tactical candidate helpers and suite loaders for Morpheus search tests.

Mandatory widening set: pass, legal general captures, and legal actions that
directly interact with a visible enemy source.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from action import (
    PASS_INDEX,
    decode_action,
    encode_action,
    legal_mask,
)
from memory import TYPE_GENERAL, VisibleMemory
from transition import DIRECTIONS

Array = np.ndarray
Action5 = tuple[int, int, int, int, int]


def _as_grids(obs) -> tuple[Array, Array, Array]:
    return (
        np.asarray(obs.type_grid, dtype=np.int32),
        np.asarray(obs.owner_grid, dtype=np.int32),
        np.asarray(obs.army_grid, dtype=np.int32),
    )


def general_capture_indices(obs, memory: VisibleMemory, mask: Array) -> list[int]:
    """Legal move indices whose destination is a visible enemy general."""
    types, owners, _ = _as_grids(obs)
    H, W = int(obs.H), int(obs.W)
    out: list[int] = []
    enemy_gen = (types == TYPE_GENERAL) & (owners == 2)
    enemy_gen |= memory.known_enemy_general & (owners == 2)
    for r, c in np.argwhere(enemy_gen):
        r, c = int(r), int(c)
        for d in range(4):
            sr = r - int(DIRECTIONS[d, 0])
            sc = c - int(DIRECTIONS[d, 1])
            if not (0 <= sr < H and 0 <= sc < W):
                continue
            if owners[sr, sc] != 1:
                continue
            for split in (0, 1):
                idx = encode_action((0, sr, sc, d, split))
                if mask[idx]:
                    out.append(idx)
    return out


def visible_enemy_source_interaction_indices(
    obs, memory: VisibleMemory, mask: Array
) -> list[int]:
    """Legal moves onto or from a cell adjacent to a visible enemy army source."""
    types, owners, armies = _as_grids(obs)
    H, W = int(obs.H), int(obs.W)
    enemy_sources = (owners == 2) & (armies >= 1)
    if not np.any(enemy_sources):
        return []
    # Cells that interact: own cells adjacent to enemy, or destinations that
    # are enemy-owned / adjacent attack squares.
    interact_dest = enemy_sources.copy()
    for r, c in np.argwhere(enemy_sources):
        r, c = int(r), int(c)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                interact_dest[nr, nc] = True
    out: list[int] = []
    for idx in np.flatnonzero(mask):
        idx = int(idx)
        if idx == PASS_INDEX:
            continue
        action = decode_action(idx)
        if int(action[0]) != 0:
            continue
        sr, sc, d = int(action[1]), int(action[2]), int(action[3])
        tr = sr + int(DIRECTIONS[d, 0])
        tc = sc + int(DIRECTIONS[d, 1])
        if not (0 <= tr < H and 0 <= tc < W):
            continue
        if interact_dest[tr, tc] or interact_dest[sr, sc]:
            out.append(idx)
    return out


def mandatory_action_indices(obs, memory: VisibleMemory) -> list[int]:
    """Pass + legal general captures + visible enemy-source interactions."""
    mask = legal_mask(obs, memory)
    out: list[int] = []
    seen: set[int] = set()
    for idx in (
        [PASS_INDEX]
        + general_capture_indices(obs, memory, mask)
        + visible_enemy_source_interaction_indices(obs, memory, mask)
    ):
        if idx in seen:
            continue
        if not mask[idx]:
            continue
        seen.add(idx)
        out.append(idx)
    return out


def policy_ordered_candidates(
    prior: Array,
    mask: Array,
    *,
    mandatory: Sequence[int],
    limit: int,
) -> list[int]:
    """Mandatory actions first (stable), then remaining by descending prior."""
    prior = np.asarray(prior, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    out: list[int] = []
    seen: set[int] = set()
    for idx in mandatory:
        idx = int(idx)
        if idx in seen or not mask[idx]:
            continue
        seen.add(idx)
        out.append(idx)
        if len(out) >= limit:
            return out
    scores = np.where(mask, prior, -1.0)
    for idx in np.argsort(-scores):
        idx = int(idx)
        if idx in seen or not mask[idx]:
            continue
        seen.add(idx)
        out.append(idx)
        if len(out) >= limit:
            break
    return out


def load_tactical_suite(path: Path | str) -> list[dict[str, Any]]:
    """Load ``tactical-suite.json`` cases."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = data.get("cases", data)
    if not isinstance(cases, list):
        raise ValueError("tactical suite must contain a list of cases")
    return cases


def suite_path() -> Path:
    return Path(__file__).resolve().parent / "tests" / "fixtures" / "tactical-suite.json"
