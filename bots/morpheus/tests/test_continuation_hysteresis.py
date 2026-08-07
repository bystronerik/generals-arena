"""Commitment hysteresis: continuing last turn's stack gets a bounded bonus.

Without it the chosen source tile jumped >=3 Manhattan on 27% of consecutive
move turns (measured, seed-0 game vs cm_expander) — the shaped-prior argmax
re-tie-broke from scratch every turn.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.morpheus

from action import PASS_INDEX, decode_action
from tactics import (
    CONTINUATION_BONUS,
    heuristic_action_scores,
    move_dest,
    play_mask,
)

from shaping_boards import BOARDS


def _continues(idx: int, prev_dest: tuple[int, int]) -> bool:
    action = decode_action(idx)
    if int(action[0]) != 0:
        return False
    return (int(action[1]), int(action[2])) == prev_dest


@pytest.mark.parametrize("board", ["march", "contact", "corridor"])
def test_bonus_multiplies_only_continuing_positive_scores(board):
    obs, mem = BOARDS[board]()
    mask = play_mask(obs, mem)
    base = heuristic_action_scores(obs, mem, mask)

    # Choose a previous action whose destination sources a positive-score move.
    prev = None
    for idx in np.flatnonzero(base > 0.0):
        action = decode_action(int(idx))
        if int(action[0]) != 0:
            continue
        sr, sc = int(action[1]), int(action[2])
        # Fabricate "we moved onto (sr, sc) last turn" from any neighbor.
        if sr + 1 < int(obs.H):
            prev = (0, sr + 1, sc, 0, 0)  # direction 0 = up in DIRECTIONS
            ends = move_dest(prev)
            if ends is not None and ends[2:] == (sr, sc):
                break
            prev = None
    assert prev is not None, "board must offer a continuing move"
    prev_dest = move_dest(prev)[2:]

    boosted = heuristic_action_scores(obs, mem, mask, prev_action=prev)
    for idx in range(PASS_INDEX):
        if _continues(idx, prev_dest) and base[idx] > 0.0:
            assert boosted[idx] == pytest.approx(base[idx] * CONTINUATION_BONUS)
        else:
            assert boosted[idx] == pytest.approx(base[idx])


def test_zero_score_moves_are_not_resurrected():
    obs, mem = BOARDS["march"]()
    mask = play_mask(obs, mem)
    base = heuristic_action_scores(obs, mem, mask)
    # Any zero-score continuing move must stay zero: the bonus amplifies
    # commitment, it never overrides a ban or a crushed retreat.
    for idx in np.flatnonzero(base == 0.0)[:50]:
        action = decode_action(int(idx))
        if int(action[0]) != 0:
            continue
        sr, sc = int(action[1]), int(action[2])
        if sr + 1 >= int(obs.H):
            continue
        prev = (0, sr + 1, sc, 0, 0)
        ends = move_dest(prev)
        if ends is None or ends[2:] != (sr, sc):
            continue
        boosted = heuristic_action_scores(obs, mem, mask, prev_action=prev)
        assert boosted[int(idx)] == 0.0
        break
