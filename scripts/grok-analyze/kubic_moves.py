#!/usr/bin/env python3
"""Infer per-seat move events from consecutive Kubic replay frames.

Replays store states, not actions. Between tick t and t+1 the engine order is:
builds, then moves/combat, then growth. Growth: generals/castles +1 every even
tick (competition), and all owned land +1 every 50 ticks.

This heuristic attributes one primary move per seat when a source cell loses
army and an orthogonal neighbour gains ownership/army consistently with a
full (leave-1) or half move. Ambiguous ticks are marked kind='unknown'.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from arena.instrument.replay.loader import Cell, Replay

ORTH = ((-1, 0), (1, 0), (0, -1), (0, 1))
MoveKind = Literal["full", "half", "build_suspect", "pass", "unknown", "multi"]


@dataclass(frozen=True)
class InferredMove:
    tick: int  # transition from tick -> tick+1
    player: int
    src: Cell | None
    dst: Cell | None
    kind: MoveKind
    src_before: int
    src_after: int
    sent: int | None
    dst_owner_before: int  # -1 neutral, 0/1 player
    dst_owner_after: int
    captured: bool


def _growth_delta(tick_after: int, is_prod: bool) -> int:
    """Army added on a cell after moves resolve, visible in frame tick_after."""
    # Frame index == tick number in our loader; growth applies at end of turn.
    # Competition: production on even turns (turn % 2 == 0) for gen/castle;
    # bulk +1 on turn % 50 == 0 for every owned land.
    # After transition into frame T, growth for turn T has applied if T>0.
    t = tick_after
    delta = 0
    if t > 0 and t % 50 == 0:
        delta += 1
    if is_prod and t > 0 and t % 2 == 0:
        delta += 1
    return delta


def _owned_cells(replay: Replay, tick: int, player: int) -> dict[Cell, int]:
    frame = replay.ticks[tick]
    out: dict[Cell, int] = {}
    for r in range(replay.rows):
        for c in range(replay.cols):
            if frame.owners[r][c] == player:
                out[(r, c)] = frame.armies[r][c]
    return out


def _is_structure(replay: Replay, cell: Cell, tick: int, player: int, castles: set[Cell]) -> bool:
    if cell == replay.generals[player]:
        return True
    return cell in castles


def infer_moves_for_player(
    replay: Replay,
    player: int,
    castles: set[Cell] | None = None,
) -> list[InferredMove]:
    """One InferredMove per tick transition for `player`."""
    castles = castles or set()
    moves: list[InferredMove] = []
    n = len(replay.ticks)
    for t in range(n - 1):
        before = _owned_cells(replay, t, player)
        after_owners = replay.ticks[t + 1].owners
        after_armies = replay.ticks[t + 1].armies
        # Candidates: owned cells whose army dropped more than growth can explain
        # (growth only increases, so any drop means a leave-behind after a send,
        # a build spend, or combat loss on that cell).
        sources: list[tuple[Cell, int, int]] = []
        for cell, army0 in before.items():
            r, c = cell
            if after_owners[r][c] != player:
                # lost the cell — combat on this tile or abandoned? rare as src
                continue
            army1 = after_armies[r][c]
            # Undo growth on source for comparison
            g = _growth_delta(t + 1, _is_structure(replay, cell, t, player, castles))
            adj = army1 - g
            if adj < army0:
                sources.append((cell, army0, army1))

        # Destination candidates: neighbour of a source that gained our ownership
        # or gained army beyond growth.
        classified: InferredMove | None = None
        if not sources:
            classified = InferredMove(
                tick=t,
                player=player,
                src=None,
                dst=None,
                kind="pass",
                src_before=0,
                src_after=0,
                sent=None,
                dst_owner_before=-2,
                dst_owner_after=-2,
                captured=False,
            )
        else:
            candidates: list[InferredMove] = []
            for src, a0, a1 in sources:
                g_src = _growth_delta(t + 1, _is_structure(replay, src, t, player, castles))
                left = a1 - g_src
                sent_est = a0 - left
                if sent_est <= 0:
                    continue
                sr, sc = src
                for dr, dc in ORTH:
                    dst = (sr + dr, sc + dc)
                    rr, cc = dst
                    if not (0 <= rr < replay.rows and 0 <= cc < replay.cols):
                        continue
                    if dst in replay.mountains:
                        continue
                    own0 = replay.ticks[t].owners[rr][cc]
                    arm0 = replay.ticks[t].armies[rr][cc]
                    own1 = after_owners[rr][cc]
                    arm1 = after_armies[rr][cc]
                    # full leave-1: left == 1 (before growth undo already applied)
                    # half: left ~= ceil(a0/2) or floor — try both
                    kind: MoveKind = "unknown"
                    if left == 1 and sent_est == a0 - 1:
                        kind = "full"
                    elif left in {a0 // 2, (a0 + 1) // 2} or sent_est in {a0 // 2, (a0 + 1) // 2}:
                        kind = "half"
                    # Ownership / capture checks
                    captured = own0 != player and own1 == player
                    reinforced = own0 == player and own1 == player
                    if not (captured or reinforced or own1 == player):
                        continue
                    # Build spend looks like army drop with no neighbour gain —
                    # handled when no dst matches.
                    candidates.append(
                        InferredMove(
                            tick=t,
                            player=player,
                            src=src,
                            dst=dst,
                            kind=kind,
                            src_before=a0,
                            src_after=a1,
                            sent=sent_est,
                            dst_owner_before=own0,
                            dst_owner_after=own1,
                            captured=captured,
                        )
                    )
            if not candidates:
                # army spent in place -> likely castle build, or combat on src
                src, a0, a1 = max(sources, key=lambda x: x[1] - (x[2]))
                classified = InferredMove(
                    tick=t,
                    player=player,
                    src=src,
                    dst=None,
                    kind="build_suspect",
                    src_before=a0,
                    src_after=a1,
                    sent=a0 - a1,
                    dst_owner_before=-2,
                    dst_owner_after=-2,
                    captured=False,
                )
            elif len(candidates) == 1:
                classified = candidates[0]
            else:
                # Prefer largest send; mark multi if several distinct src
                srcs = {c.src for c in candidates}
                best = max(candidates, key=lambda m: m.sent or 0)
                if len(srcs) > 1:
                    classified = InferredMove(
                        tick=best.tick,
                        player=best.player,
                        src=best.src,
                        dst=best.dst,
                        kind="multi",
                        src_before=best.src_before,
                        src_after=best.src_after,
                        sent=best.sent,
                        dst_owner_before=best.dst_owner_before,
                        dst_owner_after=best.dst_owner_after,
                        captured=best.captured,
                    )
                else:
                    classified = best
        assert classified is not None
        moves.append(classified)
    return moves


def move_kind_rates(moves: list[InferredMove]) -> dict:
    from collections import Counter

    c = Counter(m.kind for m in moves)
    n = len(moves) or 1
    return {k: {"count": v, "rate": v / n} for k, v in sorted(c.items())}
