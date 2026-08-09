#!/usr/bin/env python3
"""Tier-1 parity: run a ported surface in both languages over recorded cases.

    python bots/morpheus-rs/tests/parity_cases.py --smoke
    python bots/morpheus-rs/tests/parity_cases.py --corpus data/morpheus/morpheus-rs/morpheus-rs-m0

Milestone M1 of docs/bots/morpheus-rs/rewrite-plan.md, §5 tier 1: transition
next-state, legal masks, observation emission, and the action codec must be
**bit-exact**, not close.

The inputs are real: board states come from the belief particles recorded in
the M0 corpus, so every case is a position morpheus actually reasoned about.
Both implementations then run fresh on those inputs — the corpus supplies
states, not expected answers, so a case can exercise action pairs the recorded
game never played (a build, a deathtouch, a move from an unowned cell).

Wire format is a flat stream of integers, documented in
`crates/core/src/parity.rs`. Layouts here and there are positional and must be
edited together.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

BOT_DIR = Path(__file__).resolve().parents[1]
REPO = BOT_DIR.parents[1]
for entry in (REPO, REPO / "bots", REPO / "bots" / "morpheus", BOT_DIR / "tools"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

# `MORPHEUS_RS_BINARY` lets `tools/mutation_check.py` point the harness at its
# own faster-building profile. The default is, and must stay, the release
# binary: the parity harness proper tests the thing that plays.
BINARY = Path(
    os.environ.get("MORPHEUS_RS_BINARY", BOT_DIR / "target" / "release" / "morpheus-rs")
)
SMOKE_FIXTURE = BOT_DIR / "tests" / "fixtures" / "parity-smoke.jsonl.gz"
DEFAULT_CORPUS = REPO / "data" / "morpheus" / "morpheus-rs" / "morpheus-rs-m0"

PASS = (1, 0, 0, 0, 0)

# Tier 2 (rewrite-plan §5): tensor float planes must agree to 1e-6. Tighter
# than the network cares about, loose enough to survive the last bit of a
# float32 log — and the bit-pattern transport means a failure here is a real
# arithmetic difference, never a formatting one.
FLOAT_TOLERANCE = 1e-6

# §5 budgets 1e-5 MAE for the network heads against TorchScript. Unlike the
# tensor, this one *cannot* be bit-exact: the two engines sum the same products
# in different orders, and the Rust kernels fuse their multiply-adds. So the
# budget is real here — but it is still far looser than what the port achieves,
# and M2's lesson was that a tolerance nothing ever approaches is a check that
# cannot fail. Both are therefore enforced: the specified MAE, and a much
# tighter cap on the worst single element, set from measurement with headroom.
# Measured over the full corpus (1,399 cases): worst MAE 4.05e-6 on
# `pass_logit`, worst single element 1.05e-5 on the auxiliary spatial heads.
# The MAE budget is §5's and it is genuinely tight — 2.5x headroom, not the
# thousandfold slack the tensor's 1e-6 turned out to have. The element cap is
# set from the measurement at ~5x, loose enough for a different vector width
# on x86 and tight enough that a real graph error cannot hide under it.
NET_MAE_TOLERANCE = 1e-5
NET_MAX_ABS_TOLERANCE = 5e-5

# The prior is the one surface where "as tight as achievable" is genuinely
# loose. Both sides softmax in f32 over 3,970 terms — the width is copied from
# the Python deliberately (see `network::wdl_value`) — so the two answers
# differ by the last bit of `exp` and by how the 3,970 terms were summed.
# Measured worst case is a little over one f32 ulp; the cap is set an order of
# magnitude above that, and `worst_observed()` reports what a run actually hit
# so the headroom stays visible rather than becoming folklore. Measured worst
# over the full corpus: 6.56e-7 on a prior mass, 1.19e-7 on the backup value.
PRIOR_TOLERANCE = 5e-6

# The masked softmax was expected to need a tolerance — two implementations of
# `exp` over 3,970 f64 logits — and does not: it agrees to the **last bit**, so
# that is what is enforced, following the tensor's precedent. NumPy's
# vectorized `exp` and the platform libm land on the same double for every
# logit here, and `npsum` reproduces the reduction exactly, which leaves
# nothing to differ.
#
# The figure below is the floor to fall back to, deliberately, if a platform
# ever produces real drift; the failure message reports the max |Δ| and how
# many elements exceed it so that decision comes with a number attached. It is
# not a budget anything is currently measured against.
TOPLEGAL_TOLERANCE = 1e-15

# The three entry points share a trunk. Their common heads must agree exactly —
# not approximately — on each side, because they are the same arithmetic run
# twice; anything else is an entry-point switch that changed what it computed.
ENTRY_POINT_HEADS = ("policy", "pass_logit")


# --- corpus -> python objects ----------------------------------------------


def _state_from_capture(raw: dict) -> object:
    from state import GameState

    return GameState(
        armies=np.asarray(raw["armies"], dtype=np.int32),
        ownership=np.asarray(raw["ownership"], dtype=bool),
        ownership_neutral=np.asarray(raw["ownership_neutral"], dtype=bool),
        generals=np.asarray(raw["generals"], dtype=bool),
        castles=np.asarray(raw["castles"], dtype=bool),
        mountains=np.asarray(raw["mountains"], dtype=bool),
        passable=np.asarray(raw["passable"], dtype=bool),
        general_positions=np.asarray(raw["general_positions"], dtype=np.int32),
        time=int(raw["time"]),
        winner=int(raw["winner"]),
        pool_idx=0,
    )


def _memory_from_capture(raw: dict) -> object:
    from memory import VisibleMemory

    return VisibleMemory(
        H=int(raw["H"]),
        W=int(raw["W"]),
        known_mountain=np.asarray(raw["known_mountain"], dtype=bool),
        known_passable_base=np.asarray(raw["known_passable_base"], dtype=bool),
        known_castle=np.asarray(raw["known_castle"], dtype=bool),
        own_general=np.asarray(raw["own_general"], dtype=bool),
        known_enemy_general=np.asarray(raw["known_enemy_general"], dtype=bool),
        ever_visible=np.asarray(raw["ever_visible"], dtype=bool),
        last_seen_turn=np.asarray(raw["last_seen_turn"], dtype=np.int32),
        remembered_owner=np.asarray(raw["remembered_owner"], dtype=np.int8),
        remembered_army=np.asarray(raw["remembered_army"], dtype=np.int32),
        remembered_was_castle=np.asarray(raw["remembered_was_castle"], dtype=bool),
        remembered_castle_owner=np.asarray(raw["remembered_castle_owner"], dtype=np.int8),
    )


def _obs_from_capture(raw: dict) -> object:
    from observe import ArrayObservation

    return ArrayObservation(
        H=int(raw["H"]),
        W=int(raw["W"]),
        turn=int(raw["turn"]),
        my_land=int(raw["my_land"]),
        my_army=int(raw["my_army"]),
        opp_land=int(raw["opp_land"]),
        opp_army=int(raw["opp_army"]),
        type_grid=np.asarray(raw["type"], dtype=np.int32),
        owner_grid=np.asarray(raw["owner"], dtype=np.int32),
        army_grid=np.asarray(raw["army"], dtype=np.int32),
    )


def load_frames(paths: list[Path], limit: int = 0) -> list[dict]:
    """Heavy frames from capture files, newest-first cap applied per file."""
    from arena.instrument.capture import read_frames

    frames: list[dict] = []
    for path in paths:
        for frame in read_frames(path):
            if not frame.get("heavy"):
                continue
            frames.append(frame)
            if limit and len(frames) >= limit:
                return frames
    return frames


# --- serialization (mirrors crates/core/src/parity.rs) ----------------------


def encode_state(state) -> list[int]:
    h, w = state.armies.shape
    out = [h, w, int(state.time), int(state.winner)]
    out += [int(v) for v in np.asarray(state.general_positions).ravel()]
    out += [int(v) for v in state.armies.ravel()]
    for plane in (
        state.ownership[0],
        state.ownership[1],
        state.ownership_neutral,
        state.generals,
        state.castles,
        state.mountains,
        state.passable,
    ):
        out += [int(bool(v)) for v in np.asarray(plane).ravel()]
    return out


def decode_state(values: list[int]) -> tuple[object, tuple, int]:
    """Inverse of `encode_state`, plus the trailing `GameInfo` seven."""
    from state import GameState

    h, w = values[0], values[1]
    time, winner = values[2], values[3]
    gp = np.asarray(values[4:8], dtype=np.int32).reshape(2, 2)
    n = h * w
    at = 8
    armies = np.asarray(values[at : at + n], dtype=np.int32).reshape(h, w)
    at += n
    planes = []
    for _ in range(7):
        planes.append(np.asarray(values[at : at + n], dtype=bool).reshape(h, w))
        at += n
    info = tuple(values[at : at + 7])
    state = GameState(
        armies=armies,
        ownership=np.stack([planes[0], planes[1]]),
        ownership_neutral=planes[2],
        generals=planes[3],
        castles=planes[4],
        mountains=planes[5],
        passable=planes[6],
        general_positions=gp,
        time=time,
        winner=winner,
        pool_idx=0,
    )
    return state, info, at + 7


def encode_observation(obs) -> list[int]:
    types = np.asarray(obs.type_grid, dtype=np.int32)
    owners = np.asarray(obs.owner_grid, dtype=np.int32)
    armies = np.asarray(obs.army_grid, dtype=np.int32)
    return (
        [
            int(obs.H),
            int(obs.W),
            int(obs.turn),
            int(obs.my_land),
            int(obs.my_army),
            int(obs.opp_land),
            int(obs.opp_army),
        ]
        + [int(v) for v in types.ravel()]
        + [int(v) for v in owners.ravel()]
        + [int(v) for v in armies.ravel()]
    )


def encode_memory(memory) -> list[int]:
    out = [int(memory.H), int(memory.W)]
    for plane in (
        memory.known_mountain,
        memory.known_passable_base,
        memory.known_castle,
        memory.own_general,
        memory.known_enemy_general,
        memory.ever_visible,
        memory.last_seen_turn,
        memory.remembered_owner,
        memory.remembered_army,
        memory.remembered_was_castle,
        memory.remembered_castle_owner,
    ):
        out += [int(v) for v in np.asarray(plane).ravel()]
    return out


# --- synthetic states -------------------------------------------------------


def _blank(h: int, w: int):
    from state import GameState

    return GameState(
        armies=np.zeros((h, w), np.int32),
        ownership=np.zeros((2, h, w), bool),
        ownership_neutral=np.zeros((h, w), bool),
        generals=np.zeros((h, w), bool),
        castles=np.zeros((h, w), bool),
        mountains=np.zeros((h, w), bool),
        passable=np.ones((h, w), bool),
        general_positions=np.array([[0, 0], [h - 1, w - 1]], np.int32),
        time=0,
        winner=-1,
        pool_idx=0,
    )


def synthetic_states() -> list:
    """
    Hand-built states for branches the recorded corpus provably cannot reach.

    Mutation testing found two. Dropping the 50-tick growth survived every
    recorded case, because no sampled state sat at `time % 50 == 49`. Dropping
    the NumPy negative-index wrap in `_determine_move_order` survived too, for
    a subtler reason: the wrap reads row `h-1`, and the competition preset pads
    smaller boards to 21×21 with mountains, so on a padded board that row is
    border and can never be owned.

    Neither gap is a reason to relax the check — they are a reason to stop
    relying on replay alone. These states own the wrap cell deliberately, sit
    on the growth and deathtouch boundaries, and can afford a build.
    """
    states = []

    for time in (0, 1, 48, 49, 98, 99, 799, 800, 801, 1199):
        s = _blank(5, 5)._replace(time=time)
        s.generals[0, 0] = True
        s.ownership[0][0, 0] = True
        s.armies[0, 0] = 12
        s.generals[4, 4] = True
        s.ownership[1][4, 4] = True
        s.armies[4, 4] = 12
        s.ownership[0][0, 1] = True
        s.armies[0, 1] = 6
        s.ownership[1][4, 3] = True
        s.armies[4, 3] = 6
        s.castles[2, 2] = True
        s.ownership[0][2, 2] = True
        s.armies[2, 2] = 40
        s.ownership_neutral[:] = ~(s.ownership[0] | s.ownership[1])
        states.append(s)

    # The wrap cell: `di = -1` resolves to row h-1, column 0. Own it, one seat
    # at a time, so `p0_reinforcing` and `p1_reinforcing` actually differ.
    for owner in (0, 1, 2):
        s = _blank(5, 5)._replace(time=10)
        s.generals[0, 0] = True
        s.ownership[0][0, 0] = True
        s.armies[0, 0] = 9
        s.generals[0, 4] = True
        s.ownership[1][0, 4] = True
        s.armies[0, 4] = 9
        if owner in (0, 2):
            s.ownership[0][4, 0] = True
            s.armies[4, 0] = 3
        if owner in (1, 2):
            cell = (4, 0) if owner == 1 else (4, 1)
            s.ownership[1][cell] = True
            s.armies[cell] = 3
        s.ownership_neutral[:] = ~(s.ownership[0] | s.ownership[1])
        states.append(s)

    # Deathtouch shapes on a corridor: adjacent generals, and both seats one
    # step from the other's general.
    for time in (799, 800):
        s = _blank(1, 4)._replace(time=time)
        s.general_positions[:] = np.array([[0, 0], [0, 3]], np.int32)
        s.generals[0, 0] = True
        s.ownership[0][0, 0] = True
        s.armies[0, 0] = 5
        s.generals[0, 3] = True
        s.ownership[1][0, 3] = True
        s.armies[0, 3] = 5
        s.ownership[0][0, 2] = True
        s.armies[0, 2] = 2
        s.ownership[1][0, 1] = True
        s.armies[0, 1] = 2
        states.append(s)

    # A plain owned cell rich enough to build on, far from any structure.
    # Without one, every build in the synthetic set is refused for being
    # unaffordable or for sitting on a general or castle, and the
    # `known_passable_base` gate — the only thing that stops a build on ground
    # never actually seen — never gets to decide anything.
    s = _blank(7, 7)._replace(time=200)
    s.generals[0, 0] = True
    s.ownership[0][0, 0] = True
    s.armies[0, 0] = 5
    s.generals[6, 6] = True
    s.ownership[1][6, 6] = True
    s.armies[6, 6] = 5
    for cell, army in (((3, 3), 200), ((3, 4), 36), ((3, 2), 34)):
        s.ownership[0][cell] = True
        s.armies[cell] = army
    s.ownership_neutral[:] = ~(s.ownership[0] | s.ownership[1])
    states.append(s)

    # A finished game: every branch downstream of `winner >= 0` must agree.
    s = _blank(3, 3)._replace(winner=0, time=120)
    s.generals[0, 0] = True
    s.ownership[0][0, 0] = True
    s.armies[0, 0] = 4
    states.append(s)

    return states


def memory_for(state, seat: int = 0, *, explored: bool = True):
    """
    A memory for a synthetic state — fully informed, or never explored.

    `explored=False` blanks `known_passable_base`, which is the only thing
    standing between an owned, affordable cell and a build. Mutation testing
    found that gate unreachable otherwise: owned cells in real games have
    always been seen, so deleting the check changed nothing anywhere in the
    corpus. A cell can be owned and still unproven — captured through fog by a
    move whose destination was never in vision — so the branch is real, and
    without this variant nothing would notice if the port lost it.
    """
    from memory import VisibleMemory

    h, w = state.armies.shape
    own = np.asarray(state.ownership[seat])
    passable_base = (np.asarray(state.passable) & ~np.asarray(state.castles)
                     & ~np.asarray(state.generals))
    if not explored:
        passable_base = np.zeros((h, w), bool)
    return VisibleMemory(
        H=h,
        W=w,
        known_mountain=np.asarray(state.mountains).copy(),
        known_passable_base=passable_base,
        known_castle=np.asarray(state.castles).copy(),
        own_general=(np.asarray(state.generals) & own),
        known_enemy_general=(np.asarray(state.generals) & ~own),
        ever_visible=np.ones((h, w), bool),
        last_seen_turn=np.full((h, w), int(state.time), np.int32),
        remembered_owner=np.where(own, 1, 0).astype(np.int8),
        remembered_army=np.asarray(state.armies).copy(),
        remembered_was_castle=np.asarray(state.castles).copy(),
        remembered_castle_owner=np.where(own & np.asarray(state.castles), 1, 0).astype(np.int8),
    )


# --- case generation --------------------------------------------------------


def action_pairs(state, cap: int = 12) -> list[tuple[tuple, tuple]]:
    """
    A deterministic spread of joint actions for one state.

    Includes moves the recorded game never played — from unowned cells, into
    walls, half-splits, builds — because `transition` has to agree on invalid
    input too: it validates internally and silently no-ops, and "silently" is
    exactly where two implementations drift apart without anyone noticing.

    Every generated action keeps its source **and** destination on the board.
    That is not tidiness: `_determine_move_order` indexes with raw
    `row + dr`, and NumPy raises on positive overflow while wrapping negatives.
    Pass (`[1,0,0,0,0]`) is the wrap case and is always included; positive
    overflow is outside the Python's domain, so it is outside the contract.
    """
    h, w = state.armies.shape
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    per_seat: list[list[tuple]] = [[PASS], [PASS]]

    for seat in (0, 1):
        owned = np.argwhere(np.asarray(state.ownership[seat]))
        for r, c in owned[: max(1, cap // 2)]:
            r, c = int(r), int(c)
            for d, (dr, dc) in enumerate(dirs):
                if not (0 <= r + dr < h and 0 <= c + dc < w):
                    continue
                per_seat[seat].append((0, r, c, d, 0))
                per_seat[seat].append((0, r, c, d, 1))
            per_seat[seat].append((2, r, c, 0, 0))  # build, usually unaffordable
        # A move from a cell this seat does not own: must be a silent no-op.
        unowned = np.argwhere(~np.asarray(state.ownership[seat]))
        if len(unowned):
            r, c = (int(x) for x in unowned[0])
            d = next((i for i, (dr, dc) in enumerate(dirs) if 0 <= r + dr < h and 0 <= c + dc < w), 0)
            per_seat[seat].append((0, r, c, d, 0))

    pairs = []
    for i in range(cap):
        a = per_seat[0][i % len(per_seat[0])]
        b = per_seat[1][(i * 3 + 1) % len(per_seat[1])]
        pairs.append((a, b))
    return pairs


def order_pairs(state) -> list[tuple[tuple, tuple]]:
    """
    Action pairs chosen to make *move order* observable.

    Replaying recorded positions barely tests this. Move order only changes the
    board when the two moves interact, so a mutation that broke the NumPy
    negative-index wrap in `_determine_move_order` — the rule that decides who
    resolves first whenever either seat passes — survived 672 end-to-end
    transition cases untouched. These pairs hit the ordering rules directly:
    pass against move (the wrap case), reinforce against capture, and equal and
    unequal source armies for the size tiebreak.
    """
    h, w = state.armies.shape
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    moves: list[list[tuple]] = [[], []]
    for seat in (0, 1):
        for r, c in np.argwhere(np.asarray(state.ownership[seat]))[:6]:
            r, c = int(r), int(c)
            for d, (dr, dc) in enumerate(dirs):
                if 0 <= r + dr < h and 0 <= c + dc < w:
                    moves[seat].append((0, r, c, d, 0))

    pairs = [(PASS, PASS)]
    for a in moves[0][:8]:
        pairs.append((a, PASS))       # only seat 1 passes
        pairs.append((PASS, a))       # only seat 0 passes -> the wrap case
    for a, b in zip(moves[0][:8], moves[1][:8]):
        pairs.append((a, b))
        pairs.append((b, a))
    return pairs


def _decode_memory(values: list[int]) -> dict:
    """Inverse of `encode_memory`, keyed by field name."""
    h, w = values[0], values[1]
    n = h * w
    names = (
        "known_mountain", "known_passable_base", "known_castle", "own_general",
        "known_enemy_general", "ever_visible", "last_seen_turn", "remembered_owner",
        "remembered_army", "remembered_was_castle", "remembered_castle_owner",
    )
    out, at = {}, 2
    for name in names:
        out[name] = np.asarray(values[at : at + n], dtype=np.int64).reshape(h, w)
        at += n
    return out


def encode_belief(belief) -> list[int]:
    """Belief planes as f32 bit patterns; see `Ints::floats` on the Rust side."""
    out: list[int] = []
    for plane in (
        belief.enemy_owner,
        belief.enemy_army_mean,
        belief.enemy_army_std,
        belief.enemy_general,
        belief.enemy_castle_owner,
        belief.enemy_visibility,
    ):
        out += [
            int(v) for v in np.asarray(plane, dtype=np.float32).ravel().view(np.uint32)
        ]
    out.append(int(np.float32(belief.ess_fraction).view(np.uint32)))
    return out


def _belief_summary_for(obs):
    """
    A deterministic, non-trivial belief summary for the tensor cases.

    Zeros would leave seven of the 49 planes constant and prove nothing about
    them — including `belief_owner_entropy`, whose whole shape lives strictly
    between 0 and 1. The values are arbitrary but reproducible, and chosen to
    put probabilities at 0, 1 and points in between so the entropy plane sees
    both its clamped ends and its interior.
    """
    from tensor import BeliefSummary

    h, w = int(obs.H), int(obs.W)
    n = h * w
    ramp = (np.arange(n, dtype=np.float32) % 11) / 10.0
    army = (np.arange(n, dtype=np.float32) % 97) * 3.5
    return BeliefSummary(
        enemy_owner=ramp.reshape(h, w),
        enemy_army_mean=army.reshape(h, w),
        enemy_army_std=(army / 4.0).reshape(h, w),
        enemy_general=(ramp * ramp).reshape(h, w),
        enemy_castle_owner=(1.0 - ramp).reshape(h, w),
        enemy_visibility=((np.arange(n, dtype=np.float32) % 3) / 2.0).reshape(h, w),
        ess_fraction=0.375,
    )


def _memory_pairs(frame, synthetic: bool):
    """`(obs, memory, prev_digest, action)` for the memory/hash/tensor kinds."""
    pairs = []
    if frame is None:
        if not synthetic:
            return pairs
        from observe import emit_observation

        for state in synthetic_states():
            for seat in (0, 1):
                obs = emit_observation(state, seat, as_arrays=True)
                for explored in (True, False):
                    pairs.append((obs, memory_for(state, seat, explored=explored)))
    elif "memory" in frame:
        pairs.append((_obs_from_capture(frame["obs"]), _memory_from_capture(frame["memory"])))

    if frame is None and synthetic:
        pairs.extend(_crafted_memory_pairs())

    out = []
    for i, (obs, memory) in enumerate(pairs):
        # A previous digest that is not all-zero on some cases, so the rolling
        # history is exercised rather than always starting fresh.
        prev = bytes((i * 7 + j) % 256 for j in range(32)) if i % 2 else bytes(32)
        # The half-split is not decoration: without an action carrying
        # `split=1`, the tensor's previous-move-kind plane is constant at 1.0
        # and painting every move as a full one is undetectable. Mutation
        # testing found exactly that.
        action = [
            (0, 1, 1, 3, 0),
            (2, 0, 0, 0, 0),
            (1, 0, 0, 0, 0),
            (0, 1, 1, 3, 1),
        ][i % 4]
        out.append((obs, memory, prev, tuple(action)))
    return out


def _crafted_memory_pairs():
    """
    Observation/memory pairs no emitted board produces.

    `emit_observation` cannot show a remembered castle as plain fog — a castle
    out of sight encodes as type 5, never type 0 — so the rule that a type-0
    frame must *not* erase `known_castle` is unreachable from replay and from
    the synthetic states alike. Mutation testing found it: deleting the rule
    changed nothing anywhere. These pairs are built by hand to reach it.
    """
    from memory import VisibleMemory
    from observe import ArrayObservation

    def blank_memory(h, w):
        return VisibleMemory(
            H=h, W=w,
            known_mountain=np.zeros((h, w), bool),
            known_passable_base=np.zeros((h, w), bool),
            known_castle=np.zeros((h, w), bool),
            own_general=np.zeros((h, w), bool),
            known_enemy_general=np.zeros((h, w), bool),
            ever_visible=np.zeros((h, w), bool),
            last_seen_turn=np.full((h, w), -1, np.int32),
            remembered_owner=np.zeros((h, w), np.int8),
            remembered_army=np.zeros((h, w), np.int32),
            remembered_was_castle=np.zeros((h, w), bool),
            remembered_castle_owner=np.zeros((h, w), np.int8),
        )

    def obs_of(types, owners, armies, turn, h, w):
        return ArrayObservation(
            H=h, W=w, turn=turn,
            my_land=3, my_army=40, opp_land=4, opp_army=55,
            type_grid=np.asarray(types, np.int32).reshape(h, w),
            owner_grid=np.asarray(owners, np.int32).reshape(h, w),
            army_grid=np.asarray(armies, np.int32).reshape(h, w),
        )

    h, w = 1, 6
    pairs = []

    # A remembered castle observed as plain fog. The castle must survive.
    m = blank_memory(h, w)
    m.known_castle[0, 0] = True
    m.ever_visible[0, 0] = True
    m.last_seen_turn[0, 0] = 30
    m.remembered_was_castle[0, 0] = True
    m.remembered_castle_owner[0, 0] = 2
    pairs.append((obs_of([0, 0, 0, 0, 0, 0], [0] * 6, [0] * 6, 90, h, w), m))

    # Every wire type at once, over ground with mixed history: exercises each
    # branch of the terrain inference in one frame.
    m2 = blank_memory(h, w)
    m2.known_passable_base[0, 1] = True   # a structure here is a new castle
    m2.ever_visible[0, 2] = True          # ...and here too
    m2.known_mountain[0, 3] = True
    pairs.append((
        obs_of([5, 5, 5, 1, 3, 4], [0, 0, 0, 1, 2, 1], [0, 0, 0, 9, 44, 1], 120, h, w),
        m2,
    ))

    # A cell seen long ago, so sight age is large but still inside the cap.
    m3 = blank_memory(h, w)
    m3.ever_visible[0, :] = True
    m3.last_seen_turn[0, :] = 5
    m3.remembered_owner[0, :] = np.array([0, 1, 2, 1, 2, 0], np.int8)
    m3.remembered_army[0, :] = np.array([0, 3, 900, 12, 4096, 1], np.int32)
    m3.remembered_was_castle[0, 2] = True
    m3.remembered_castle_owner[0, 2] = 2
    pairs.append((obs_of([0] * 6, [0] * 6, [0] * 6, 1150, h, w), m3))

    return pairs


# --- M4: beliefs, and the recorded RNG stream -------------------------------

# Method codes shared with `read_draws` in crates/core/src/parity.rs.
_DRAW_CODES = {"integers": 0, "choice": 1, "random": 2}


def recording_rng(seed: int):
    """A NumPy generator that logs what it hands out.

    The same `RecordingGenerator` the M0 capture uses on the live bot, pointed
    at a fresh seeded generator here. That reuse is the point: if the recorder
    ever stops covering a draw site the corpus and this harness go blind
    together, which is a single failure to notice rather than two.
    """
    from capture_morpheus import RecordingGenerator

    return RecordingGenerator(np.random.default_rng(seed))


def encode_draws(draws: list[dict]) -> list[int]:
    """`n` then, per draw: `code a b size replace weighted count values…`.

    `size=None` rides as -1. NumPy's scalar form is a different call from
    `size=1` and the Rust `Replay` refuses to accept one for the other, which
    is how draw-site order becomes a checked contract instead of a hope
    (rewrite-plan §5).
    """
    out = [len(draws)]
    for draw in draws:
        method, args, result = draw["m"], draw["a"], draw["r"]
        values = result if isinstance(result, list) else [result]
        if method == "integers":
            a, b = int(args["low"]), int(args["high"])
            replace, weighted = 1, 0
        elif method == "choice":
            a, b = int(args["n"]), 0
            replace = int(bool(args["replace"]))
            weighted = int(bool(args["weighted"]))
        else:  # random
            a = b = 0
            replace, weighted = 1, 0
        size = args.get("size")
        out += [
            _DRAW_CODES[method],
            a,
            b,
            -1 if size is None else int(size),
            replace,
            weighted,
            len(values),
        ]
        if method == "random":
            out += [int(v) for v in np.asarray(values, np.float64).view(np.int64)]
        else:
            out += [int(v) for v in values]
    return out


def _f64_bits(values) -> list[int]:
    return np.asarray(values, dtype=np.float64).ravel().view(np.int64).astype(np.int64).tolist()


def _memory_from_planes(raw: dict, h: int, w: int):
    """The capture stores the eleven planes without H/W; the frame has those."""
    return _memory_from_capture({"H": h, "W": w, **raw})


def encode_particle_state(particle) -> list[int]:
    """`weight has_prev action5 n_history state memory [oldest pairs…]`.

    Histories go out as the oldest state plus the action pairs, exactly as the
    corpus stores them, and both sides rebuild the intermediate boards through
    the transition kernel. Shipping the boards instead would be two thirds of a
    megabyte per turn and would also skip the one property M4's rejuvenation
    gate exists to check.
    """
    prev = particle.enemy_prev_action
    out = _f64_bits([float(particle.weight)])
    out += [1 if prev is not None else 0] + list(prev or PASS)
    out += [len(particle.history)]
    out += encode_state(particle.state)
    out += encode_memory(particle.enemy_memory)
    if particle.history:
        out += encode_state(particle.history[0].state)
        for frame in particle.history:
            out += list(frame.my_action) + list(frame.enemy_action)
    return out


def encode_belief_state(belief) -> list[int]:
    out = [
        int(belief.seat),
        int(bool(belief.collapsed)),
        int(belief.config.n_particles),
        int(belief.n),
    ]
    for particle in belief.particles:
        out += encode_particle_state(particle)
    return out


def _read_state_at(values: list[int], at: int):
    """`decode_state` without the trailing `GameInfo`, at an offset."""
    from state import GameState

    h, w = values[at], values[at + 1]
    time, winner = values[at + 2], values[at + 3]
    gp = np.asarray(values[at + 4 : at + 8], dtype=np.int32).reshape(2, 2)
    n = h * w
    at += 8
    armies = np.asarray(values[at : at + n], dtype=np.int32).reshape(h, w)
    at += n
    planes = []
    for _ in range(7):
        planes.append(np.asarray(values[at : at + n], dtype=bool).reshape(h, w))
        at += n
    return (
        GameState(
            armies=armies,
            ownership=np.stack([planes[0], planes[1]]),
            ownership_neutral=planes[2],
            generals=planes[3],
            castles=planes[4],
            mountains=planes[5],
            passable=planes[6],
            general_positions=gp,
            time=time,
            winner=winner,
            pool_idx=0,
        ),
        at,
    )


def _read_memory_at(values: list[int], at: int):
    h, w = values[at], values[at + 1]
    n = h * w
    at += 2
    raw = {}
    for name in (
        "known_mountain", "known_passable_base", "known_castle", "own_general",
        "known_enemy_general", "ever_visible", "last_seen_turn", "remembered_owner",
        "remembered_army", "remembered_was_castle", "remembered_castle_owner",
    ):
        raw[name] = np.asarray(values[at : at + n], dtype=np.int64).reshape(h, w)
        at += n
    return _memory_from_capture({"H": h, "W": w, **raw}), at


def _read_belief_at(values: list[int], at: int):
    """Inverse of `write_belief` on the Rust side."""
    seat, collapsed, count = values[at], values[at + 1], values[at + 2]
    at += 3
    particles = []
    for _ in range(count):
        weight = float(np.asarray([values[at]], np.int64).view(np.float64)[0])
        has_prev = bool(values[at + 1])
        prev = tuple(values[at + 2 : at + 7])
        history_len = values[at + 7]
        at += 8
        frames = []
        for _ in range(history_len):
            frames.append(
                (tuple(values[at : at + 5]), tuple(values[at + 5 : at + 10]), values[at + 10])
            )
            at += 11
        state, at = _read_state_at(values, at)
        memory, at = _read_memory_at(values, at)
        particles.append(
            {
                "weight": weight,
                "enemy_prev_action": prev if has_prev else None,
                "history_len": history_len,
                "history": frames,
                "state": state,
                "enemy_memory": memory,
            }
        )
    return {"seat": seat, "collapsed": bool(collapsed), "particles": particles}, at


def belief_from_frame(frame: dict):
    """A live `BeliefState` from a captured heavy frame.

    History frames are reconstructed from the oldest recorded state through
    the transition kernel — the same recipe the Rust side follows, so a
    disagreement about a rebuilt board would surface as a rejuvenation
    mismatch rather than as an encoding argument.
    """
    from belief import BeliefConfig, BeliefState, HistoryFrame, Particle
    from observe import emit_observation
    from transition import transition

    snapshot = frame.get("belief")
    if not snapshot or not snapshot["particles"]:
        return None
    h, w = int(frame["obs"]["H"]), int(frame["obs"]["W"])
    seat = int(snapshot["seat"])

    particles = []
    for raw in snapshot["particles"]:
        history = []
        if raw["history_actions"]:
            current = _state_from_capture(raw["history_oldest_state"])
            for step in raw["history_actions"]:
                actions = np.zeros((2, 5), dtype=np.int32)
                actions[seat] = np.asarray(step["mine"], dtype=np.int32)
                actions[1 - seat] = np.asarray(step["enemy"], dtype=np.int32)
                nxt, _ = transition(current, actions)
                history.append(
                    HistoryFrame(
                        state=current,
                        my_action=tuple(int(v) for v in step["mine"]),
                        enemy_action=tuple(int(v) for v in step["enemy"]),
                        observation_after=emit_observation(nxt, seat, as_arrays=True),
                    )
                )
                current = nxt
        prev = raw["enemy_prev_action"]
        particles.append(
            Particle(
                state=_state_from_capture(raw["state"]),
                weight=float(raw["weight"]),
                enemy_memory=_memory_from_planes(raw["enemy_memory"], h, w),
                enemy_prev_action=tuple(int(v) for v in prev) if prev else None,
                history=tuple(history),
            )
        )
    return BeliefState(
        seat=seat,
        particles=particles,
        config=BeliefConfig(n_particles=int(snapshot["n_particles_config"])),
        collapsed=bool(snapshot["collapsed"]),
    )


def _deep_history_belief(seat: int, depth: int):
    """A belief whose particles already hold a full `recovery_lag` window.

    Mutation testing found the gap: `filter_step` appends one frame and drops
    from the *old* end, and swapping that for a truncate is invisible until a
    history is already at the cap. Corpus histories on the smoke slice are one
    or two deep, so nothing could tell the two apart.

    The chain is built with real transitions rather than by fabricating frames,
    so every `observation_after` is one the state actually produced — which is
    what rejuvenation checks against.
    """
    from belief import BeliefConfig, BeliefState, HistoryFrame, Particle
    from memory import empty_memory
    from observe import emit_observation
    from transition import transition

    state = synthetic_states()[0]
    frames = []
    current = state
    for _ in range(depth):
        actions = np.zeros((2, 5), dtype=np.int32)
        actions[seat] = np.asarray(PASS, np.int32)
        actions[1 - seat] = np.asarray(PASS, np.int32)
        nxt, _ = transition(current, actions)
        frames.append(
            HistoryFrame(
                state=current,
                my_action=PASS,
                enemy_action=PASS,
                observation_after=emit_observation(nxt, seat, as_arrays=True),
            )
        )
        current = nxt

    h, w = current.armies.shape
    return BeliefState(
        seat=seat,
        particles=[
            Particle(
                state=current,
                weight=0.5,
                enemy_memory=empty_memory(h, w),
                enemy_prev_action=PASS,
                history=tuple(frames),
            )
            for _ in range(2)
        ],
        config=BeliefConfig(n_particles=2),
        collapsed=False,
    )


def synthetic_beliefs():
    """Hand-built beliefs for shapes the corpus cannot supply.

    The capture only ever holds eight particles, all alive, all with weights
    that already sum to one. That last property hides more than it looks:
    normalizing a normalized set is the identity, so `ess`'s division,
    `normalize_weights`'s clamp, and `summarize_belief`'s rescale are all
    unreachable from replay alone. Mutation testing found every one of them.

    So these deliberately include: an empty set; a single particle; **weights
    that do not sum to one**; a **negative** weight; a particle count below the
    configured one so the resample-to-`n` pad in `filter_step` fires; a
    collapsed set; particles identical except for their previous action, which
    is the only thing the proposal information key would notice; and a
    `recovery_lag`-deep history.
    """
    from belief import BeliefConfig, BeliefState, Particle
    from memory import empty_memory

    out = []
    states = synthetic_states()
    for seat in (0, 1):
        for n_config, weights in (
            (4, []),
            (4, [1.0]),
            (4, [0.5, 0.25, 0.25]),
            (2, [0.4, 0.0, 0.6]),
            (8, [1.0 / 3.0] * 3),
            # Unnormalized: `total` is not 1, so every rescale is observable.
            (4, [3.0, 1.0, 4.0, 1.0]),
            (4, [0.05, 0.05]),
            # A negative weight, which `normalize_weights` must clamp to zero
            # rather than carry through the division.
            (4, [2.0, -1.0, 3.0]),
            # All-zero: the fallback branches in `resample` and
            # `summarize_belief` that avoid dividing by zero.
            (4, [0.0, 0.0]),
        ):
            particles = [
                Particle(
                    state=states[i % len(states)],
                    weight=w,
                    enemy_memory=empty_memory(
                        *states[i % len(states)].armies.shape
                    ),
                    enemy_prev_action=None if i % 2 else (0, 1, 1, 3, 0),
                    history=(),
                )
                for i, w in enumerate(weights)
            ]
            out.append(
                BeliefState(
                    seat=seat,
                    particles=particles,
                    config=BeliefConfig(n_particles=n_config),
                    collapsed=bool(n_config == 2),
                )
            )

        # Same board, different previous action. The information key is the
        # only thing that separates these, and in the uniform proposal it
        # reaches exactly one observable: `n_unique_info_keys`.
        state = states[0]
        h, w = state.armies.shape
        out.append(
            BeliefState(
                seat=seat,
                particles=[
                    Particle(
                        state=state,
                        weight=0.25,
                        enemy_memory=empty_memory(h, w),
                        enemy_prev_action=prev,
                        history=(),
                    )
                    for prev in (None, PASS, (0, 1, 1, 3, 0), (0, 1, 1, 3, 1))
                ],
                config=BeliefConfig(n_particles=4),
                collapsed=False,
            )
        )
        out.append(_deep_history_belief(seat, 8))

        # Eight identical boards at irregular weights. Two mutations needed
        # exactly this and nothing else could supply it: NumPy's pairwise sum
        # only diverges from a sequential one from eight elements up, and the
        # belief planes only round differently in f32 when the weights are not
        # negative powers of two. The corpus is eight particles at 1/8, where
        # both are exact.
        out.append(
            BeliefState(
                seat=seat,
                particles=[
                    Particle(
                        state=state,
                        weight=1.0 / (i + 3.0),
                        enemy_memory=empty_memory(h, w),
                        enemy_prev_action=None,
                        history=(),
                    )
                    for i in range(8)
                ],
                config=BeliefConfig(n_particles=8),
                collapsed=False,
            )
        )
    return out


def _enemy_action_for(belief, particle, offset: int):
    """A legal enemy action for one particle, chosen without an RNG.

    Deterministic on purpose: `filter_step`'s own draw stream should be about
    resampling, not about how the enemy actions were picked, and the mask this
    indexes into is a surface parity already proves bit-exact.
    """
    from action import decode_action, legal_mask
    from belief import as_action5
    from memory import update_memory
    from observe import emit_observation

    enemy_obs = emit_observation(particle.state, belief.enemy_seat, as_arrays=True)
    enemy_mem = update_memory(particle.enemy_memory, enemy_obs)
    legal = np.flatnonzero(legal_mask(enemy_obs, enemy_mem))
    if not len(legal):
        return PASS
    return as_action5(decode_action(int(legal[offset % len(legal)])))


def _reservoir_snapshot(reservoir) -> dict:
    return {
        "admitted_count": int(reservoir.admitted_count),
        "version": int(reservoir.version),
        "particles": [(float(p.weight), p.state) for p in reservoir.particles],
    }


def _replace_from_belief_recording(reservoir_obj, belief, rng) -> None:
    """Run `replace_from_belief` with its private generator logged.

    The Python builds `np.random.default_rng(0)` *inside* the method, so its
    draws never reach the bot's shared stream — a deliberate property, since a
    node's contents must not depend on how much searching came before. That
    also puts them out of reach of the recorder, so the factory is patched for
    the duration and the draws are appended to the same log. Sharing the list
    rather than keeping a second one matters: the Rust side reads one stream in
    call order, and a separate log would silently permit a port that made these
    draws at a different point.
    """
    import reservoir as reservoir_module  # noqa: F401 - patched via numpy
    from capture_morpheus import RecordingGenerator

    inner = RecordingGenerator(np.random.default_rng(0))
    inner.draws = rng.draws
    real = np.random.default_rng
    try:
        np.random.default_rng = lambda seed=None: inner
        reservoir_obj.replace_from_belief(belief)
    finally:
        np.random.default_rng = real


def npsum_vectors() -> list[np.ndarray]:
    """Vectors whose length crosses every branch of NumPy's pairwise sum.

    8 is where it switches to eight interleaved lanes, 128 where it starts
    recursing, and the odd sizes leave a remainder for the scalar tail. The
    values are deliberately not representable as short binary fractions, so a
    reassociated sum lands on a different float.
    """
    sizes = [0, 1, 2, 7, 8, 9, 15, 16, 17, 31, 64, 65, 127, 128, 129, 200, 257]
    out = []
    for size in sizes:
        out.append(np.asarray([1.0 / (i + 3.0) for i in range(size)], np.float64))
        out.append(np.asarray([(i % 11) * 0.1 for i in range(size)], np.float64))
    return out


def argsort_vectors() -> list[np.ndarray]:
    """Score vectors, most of them full of ties.

    A uniform proposal gives every legal action the same probability, so the
    ordering `top_legal_actions` walks is decided entirely by how the sort
    breaks ties. These reproduce that: masks of several densities at the real
    3,970 action space, plus small sizes around the insertion-sort ceiling and
    one all-distinct vector as a control.
    """
    n = 9 * 441 + 1
    out = []
    for density in (0.01, 0.05, 0.2, 0.6):
        rng = np.random.default_rng(int(density * 1000))
        mask = rng.random(n) < density
        mask[-1] = True
        probs = np.zeros(n, np.float64)
        probs[mask] = 1.0 / mask.sum()
        out.append(np.where(mask, probs, -1.0))
    for size in (0, 1, 2, 15, 16, 17, 128, 129):
        out.append(np.asarray([float((i * 37) % 11) for i in range(size)], np.float64))
    out.append(np.asarray([float(i) * 1e-3 for i in range(500)], np.float64))
    return out


# --- the network oracle ------------------------------------------------------

_SESSION = None


def torchscript_session():
    """The frozen TorchScript artifact, loaded once per process.

    This is the *oracle* for the `net` surface — the three modules the Python
    bot actually runs, not a re-implementation. The Rust side reads a
    safetensors conversion of the same weights (`tools/convert_artifact.py`),
    so a mismatch is arithmetic, never a different checkpoint.

    That last sentence is checked rather than asserted. The converted manifest
    records the SHA-256 of the `.pt` it was made from; if the two sides have
    drifted, every head disagrees by a lot and the failure would read like an
    arithmetic catastrophe instead of a stale artifact. Naming the real cause
    costs one hash of a one-megabyte file.
    """
    global _SESSION
    if _SESSION is None:
        import hashlib
        import json

        import torch

        torch.set_num_threads(1)
        art = REPO / "bots" / "morpheus" / "artifact"
        converted = BOT_DIR / "artifact" / "manifest.json"
        if converted.is_file():
            claimed = json.loads(converted.read_text()).get("source_artifact", {})
            source = art / claimed.get("file", "model.pt")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if claimed.get("sha256") != digest:
                raise RuntimeError(
                    f"{converted.relative_to(REPO)} was converted from a "
                    f"{str(claimed.get('sha256'))[:12]} artifact but "
                    f"{source.relative_to(REPO)} is now {digest[:12]}; "
                    "re-run bots/morpheus-rs/tools/convert_artifact.py"
                )
        _SESSION = {
            name: torch.jit.load(str(art / filename), map_location="cpu").eval()
            for name, filename in (
                ("policy", "model_policy.pt"),
                ("policy_wdl", "model_policy_wdl.pt"),
                ("full", "model.pt"),
            )
        }
    return _SESSION


def _network_expectation(tensor: np.ndarray) -> dict:
    """Every head TorchScript produces for one 49×21×21 tensor."""
    import torch

    from export import forward_exported

    session = torchscript_session()
    x = torch.from_numpy(np.ascontiguousarray(tensor, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        p_policy, p_pass = session["policy"](x)
        w_policy, w_pass, w_wdl = session["policy_wdl"](x)
        # The scripted module returns a bare tuple; `forward_exported` is the
        # Python bot's own way of naming those eleven tensors, so the head
        # order in this file is the bot's order and not a second guess at it.
        full = forward_exported(session["full"], x)

    def flat(t) -> np.ndarray:
        return np.asarray(t.detach().numpy(), dtype=np.float32).reshape(-1)

    return {
        "policy_only.policy": flat(p_policy),
        "policy_only.pass_logit": flat(p_pass),
        "policy_wdl.policy": flat(w_policy),
        "policy_wdl.pass_logit": flat(w_pass),
        "policy_wdl.wdl_logits": flat(w_wdl),
        "full.policy": flat(full.policy),
        "full.pass_logit": flat(full.pass_logit),
        "full.wdl_logits": flat(full.wdl_logits),
        "full.hidden_owner": flat(full.hidden_owner),
        "full.enemy_army_bins": flat(full.enemy_army_bins),
        "full.enemy_general": flat(full.enemy_general),
        "full.hidden_castle": flat(full.hidden_castle),
        "full.margins": np.concatenate(
            [
                flat(full.land_margin),
                flat(full.army_margin),
                flat(full.castle_margin),
                flat(full.turns_to_termination),
            ]
        ),
    }


# Head name -> element count, in the order `parity.rs` writes them. Reading the
# Rust stream back is positional, so this table and the `"net"` arm over there
# are one layout described twice and must be edited together.
NET_LAYOUT = (
    ("policy_only.policy", 9 * 441),
    ("policy_only.pass_logit", 1),
    ("policy_wdl.policy", 9 * 441),
    ("policy_wdl.pass_logit", 1),
    ("policy_wdl.wdl_logits", 3),
    ("full.policy", 9 * 441),
    ("full.pass_logit", 1),
    ("full.wdl_logits", 3),
    ("full.hidden_owner", 441),
    ("full.enemy_army_bins", 16 * 441),
    ("full.enemy_general", 441),
    ("full.hidden_castle", 441),
    ("full.margins", 4),
)


def _prior_expectation(
    logits: np.ndarray, mask: np.ndarray, wdl: np.ndarray, from_root: bool
) -> tuple[np.ndarray, float]:
    """`legal_normalized_policy` and `backup_value`, through the real Python."""
    import torch
    from network import backup_value, legal_normalized_policy

    policy = torch.from_numpy(
        np.ascontiguousarray(logits[:-1], dtype=np.float32)
    ).reshape(1, 9, 21, 21)
    pass_logit = torch.from_numpy(
        np.ascontiguousarray(logits[-1:], dtype=np.float32)
    ).reshape(1, 1)
    mask_t = torch.from_numpy(np.ascontiguousarray(mask, dtype=bool)).unsqueeze(0)
    prior = (
        legal_normalized_policy(policy, pass_logit, mask_t)
        .squeeze(0)
        .numpy()
        .astype(np.float64)
    )
    value = float(
        backup_value(
            torch.from_numpy(np.ascontiguousarray(wdl, dtype=np.float32)).reshape(1, 3),
            from_root=from_root,
        ).item()
    )
    return prior, value


def _f32_bits(values: np.ndarray) -> list[int]:
    return np.asarray(values, dtype=np.float32).ravel().view(np.uint32).astype(np.int64).tolist()


def _decode_f32(values: list[int]) -> np.ndarray:
    return np.asarray(values, dtype=np.int64).astype(np.uint32).view(np.float32)


def _decode_f64(values: list[int]) -> np.ndarray:
    return np.asarray(values, dtype=np.int64).view(np.float64)


def build_cases(
    frames: list[dict], kind: str, *, pairs_per_state: int = 12, synthetic: bool = True
) -> tuple[list[int], list]:
    """`(int stream, python-side expectations)` for one parity kind."""
    from action import legal_mask
    from observe import emit_observation
    from transition import transition
    from action import live_build_cost

    stream: list[int] = []
    expected: list = []

    from transition import _determine_move_order

    def _states_of(frame) -> list:
        if frame is None:  # the synthetic block
            return synthetic_states()
        return [
            _state_from_capture(p["state"])
            for p in (frame.get("belief") or {}).get("particles", [])
        ]

    # `None` stands for the synthetic block, appended so every kind sees the
    # branches replay cannot reach.
    for frame in list(frames) + ([None] if synthetic else []):
        if kind in ("transition", "observe", "order"):
            for state in _states_of(frame):
                if kind == "transition":
                    for a, b in action_pairs(state, pairs_per_state):
                        actions = np.asarray([a, b], dtype=np.int32)
                        nxt, info = transition(state, actions)
                        stream += encode_state(state) + list(a) + list(b)
                        expected.append((nxt, info))
                elif kind == "order":
                    for a, b in order_pairs(state):
                        actions = np.asarray([a, b], dtype=np.int32)
                        stream += encode_state(state) + list(a) + list(b)
                        expected.append(int(_determine_move_order(state, actions)))
                else:
                    for seat in (0, 1):
                        obs = emit_observation(state, seat, as_arrays=True)
                        stream += encode_state(state) + [seat]
                        expected.append(obs)
        elif kind in ("mask", "cost"):
            pairs = []
            if frame is None:
                # Synthetic: emit each seat's own view and a memory that knows
                # the board, so builds are priceable and actually reachable.
                for state in synthetic_states():
                    for seat in (0, 1):
                        obs = emit_observation(state, seat, as_arrays=True)
                        for explored in (True, False):
                            pairs.append((obs, memory_for(state, seat, explored=explored)))
            elif "memory" in frame:
                pairs.append((_obs_from_capture(frame["obs"]),
                              _memory_from_capture(frame["memory"])))
            for obs, memory in pairs:
                stream += encode_observation(obs) + encode_memory(memory)
                expected.append(
                    legal_mask(obs, memory) if kind == "mask" else live_build_cost(obs, memory)
                )
        elif kind in ("memory", "hash", "tensor"):
            from hashing import (
                child_edge_key,
                enemy_info_hash_prehashed,
                info_state_key_prehashed,
                memory_digest,
                roll_history_digest,
            )
            from memory import update_memory
            from observe import observation_hash
            from tensor import build_tensor

            for obs, memory, prev_digest, action in _memory_pairs(frame, synthetic):
                stream += encode_observation(obs) + encode_memory(memory)
                if kind == "memory":
                    # Memory *before* the fold goes in; the fold is what is
                    # being checked, so feeding the post-fold memory would
                    # test nothing.
                    expected.append(update_memory(memory, obs))
                elif kind == "hash":
                    stream += list(prev_digest) + list(action)
                    payload = observation_hash(obs)
                    digest = memory_digest(memory)
                    expected.append(
                        digest
                        + info_state_key_prehashed(
                            int(obs.turn), digest, payload, bytes(prev_digest)
                        )
                        + enemy_info_hash_prehashed(payload, digest)
                        + child_edge_key(action, payload)
                        + roll_history_digest(bytes(prev_digest), action, payload)
                    )
                else:
                    belief = _belief_summary_for(obs)
                    stream += encode_belief(belief)
                    stream += [1] + list(action)
                    expected.append(
                        build_tensor(
                            obs, memory, belief=belief, previous_action=action
                        )
                    )
        elif kind in ("net", "prior"):
            from action import legal_mask
            from memory import update_memory
            from tensor import build_tensor

            for obs, memory, _prev_digest, action in _memory_pairs(frame, synthetic):
                # The tensor the network sees is the one the `tensor` surface
                # already proves bit-identical, so a `net` failure is the graph
                # and never its input.
                tensor = np.asarray(
                    build_tensor(
                        obs,
                        memory,
                        belief=_belief_summary_for(obs),
                        previous_action=action,
                    ),
                    dtype=np.float32,
                )
                heads = _network_expectation(tensor)
                if kind == "net":
                    stream += _f32_bits(tensor)
                    expected.append(heads)
                else:
                    # Feed the *Python* logits to both sides. The softmax and
                    # the WDL contraction are what is under test here; running
                    # them on each side's own logits would fold the network's
                    # tolerance into a check meant to be near-exact.
                    logits = np.concatenate(
                        [heads["policy_wdl.policy"], heads["policy_wdl.pass_logit"]]
                    )
                    mask = np.asarray(
                        legal_mask(obs, update_memory(memory, obs)), dtype=bool
                    )
                    wdl = heads["policy_wdl.wdl_logits"]
                    from_root = bool((int(obs.turn) // 2) % 2 == 0)
                    stream += _f32_bits(logits)
                    stream += [int(v) for v in mask]
                    stream += _f32_bits(wdl)
                    stream += [int(from_root)]
                    expected.append(
                        _prior_expectation(logits, mask, wdl, from_root)
                    )
        elif kind == "toplegal":
            from action import legal_mask
            from memory import update_memory
            from proposal import (
                _singleton_probs,
                _softmax_masked,
                top_legal_actions,
            )

            for index, (obs, memory, _prev, _action) in enumerate(
                _memory_pairs(frame, synthetic)
            ):
                mask = np.asarray(legal_mask(obs, update_memory(memory, obs)), dtype=bool)
                # Three shapes of logit vector, because they rank differently.
                # A wide spread makes the ordering about the values; a narrow
                # one puts neighbouring actions within a rounding of each
                # other; a flat one is what the *deployed* uniform proposal
                # actually produces, where the ranking is pure tie-breaking.
                for k, spread in ((4, 6.0), (8, 0.05), (2, 0.0)):
                    logits = spread * np.sin(
                        np.arange(len(mask), dtype=np.float64) * (index + 1) * 0.017
                    )
                    probs = _softmax_masked(logits, mask)
                    stream += _f64_bits(logits)
                    stream += [int(v) for v in mask]
                    stream += [k]
                    expected.append(
                        (
                            probs,
                            int(np.argmax(_singleton_probs(mask))),
                            top_legal_actions(probs, mask, k),
                        )
                    )
        elif kind in ("npsum", "argsort"):
            if frame is not None:
                continue  # pure functions; one pass over the vectors is enough
            vectors = npsum_vectors() if kind == "npsum" else argsort_vectors()
            for values in vectors:
                stream += [len(values)] + _f64_bits(values)
                expected.append(
                    float(values.sum())
                    if kind == "npsum"
                    else [int(i) for i in np.argsort(-values)]
                )
        elif kind == "initbelief":
            from belief import (
                BeliefConfig,
                initialize_belief,
                legal_enemy_general_candidates,
            )
            from observe import emit_observation

            # A first frame is the one thing the corpus records but never
            # feeds back: it stores the beliefs that exist, not the frames that
            # created them. These are real first frames where the capture has
            # one, and every synthetic board's own emission otherwise.
            observations = []
            if frame is None:
                for state in synthetic_states():
                    for seat in (0, 1):
                        observations.append((emit_observation(state, seat, as_arrays=True), seat))
            elif frame.get("stratum") == "first_move" or int(frame["obs"]["turn"]) <= 2:
                observations.append((_obs_from_capture(frame["obs"]), int(frame["seat"])))

            for index, (obs, seat) in enumerate(observations):
                # Distances that straddle the real 17: one that admits most of
                # the board, one that admits none of a small one and forces the
                # fallback, and the shipped value.
                for n_particles, min_distance in ((4, 1), (8, 17), (3, 40)):
                    rng = recording_rng(6000 + index)
                    try:
                        candidates = legal_enemy_general_candidates(
                            obs, min_distance=min_distance
                        )
                    except ValueError:
                        candidates = None
                    try:
                        result = initialize_belief(
                            obs,
                            seat,
                            rng,
                            config=BeliefConfig(
                                n_particles=n_particles,
                                min_general_distance=min_distance,
                            ),
                        )
                    except ValueError:
                        result = None
                    stream += encode_observation(obs)
                    stream += [seat, n_particles, min_distance]
                    stream += encode_draws(rng.take())
                    expected.append((candidates, result))
        elif kind in ("summary", "propose", "filter", "rejuvenate", "maxent"):
            from belief import ess, ess_fraction, filter_step
            from memory import update_memory
            from observe import emit_observation
            from particle_summary import summarize_belief
            from proposal import ProposalTelemetry, propose_enemy_actions, uniform_legal_probs
            from recovery import maximum_entropy_reconstruction, rejuvenate
            from transition import transition

            beliefs = []
            if frame is None:
                beliefs = synthetic_beliefs()
            else:
                belief = belief_from_frame(frame)
                if belief is not None:
                    beliefs = [belief]

            for b_index, belief in enumerate(beliefs):
                if kind == "summary":
                    stream += encode_belief_state(belief)
                    expected.append(
                        (
                            summarize_belief(belief),
                            ess([p.weight for p in belief.particles]),
                            ess_fraction(belief),
                        )
                    )
                elif kind == "propose":
                    rng = recording_rng(1000 + b_index)
                    probs = []
                    for particle in belief.particles:
                        enemy_obs = emit_observation(
                            particle.state, belief.enemy_seat, as_arrays=True
                        )
                        probs.append(
                            uniform_legal_probs(
                                enemy_obs,
                                update_memory(particle.enemy_memory, enemy_obs),
                            )
                        )
                    telemetry = ProposalTelemetry()
                    actions = propose_enemy_actions(
                        belief, rng, policy=None, telemetry=telemetry
                    )
                    stream += encode_belief_state(belief) + encode_draws(rng.take())
                    expected.append(
                        (
                            probs,
                            actions,
                            telemetry.n_singleton_particles,
                            telemetry.n_unique_info_keys,
                        )
                    )
                elif kind == "filter":
                    enemy_actions = [
                        _enemy_action_for(belief, p, i + b_index)
                        for i, p in enumerate(belief.particles)
                    ]
                    my_action = tuple(int(v) for v in frame["action"]) if frame else PASS
                    # Two frames per belief. The first is one the leading
                    # particle can actually explain, so survivors exist and the
                    # normalize/resample tail runs; the second is the turn's own
                    # observation, which almost never survives and is therefore
                    # the total-collapse branch.
                    variants = []
                    if belief.n:
                        actions = np.zeros((2, 5), dtype=np.int32)
                        actions[belief.seat] = np.asarray(my_action, np.int32)
                        actions[belief.enemy_seat] = np.asarray(enemy_actions[0], np.int32)
                        nxt, _ = transition(belief.particles[0].state, actions)
                        advanced = emit_observation(nxt, belief.seat, as_arrays=True)
                        variants.append((enemy_actions, advanced))
                        # The same frame with **one** enemy action for every
                        # particle. On a belief whose particles share a board
                        # that makes all of them survive, which is the only way
                        # `normalize_weights` ever sees more than one weight —
                        # and a mutation swapping its sequential sum for
                        # NumPy's pairwise one is invisible on a single weight.
                        variants.append(([enemy_actions[0]] * belief.n, advanced))
                    if frame is not None:
                        variants.append((enemy_actions, _obs_from_capture(frame["obs"])))
                    for t_index, (chosen, real_obs) in enumerate(variants):
                        rng = recording_rng(2000 + b_index * 8 + t_index)
                        result = filter_step(belief, my_action, real_obs, chosen, rng)
                        stream += encode_belief_state(belief)
                        stream += list(my_action)
                        stream += encode_observation(real_obs)
                        for action in chosen:
                            stream += list(action)
                        stream += encode_draws(rng.take())
                        expected.append(result)
                elif kind == "rejuvenate":
                    if not any(p.history for p in belief.particles):
                        continue
                    rng = recording_rng(3000 + b_index)
                    result = rejuvenate(belief, rng, policy=None)
                    stream += encode_belief_state(belief) + encode_draws(rng.take())
                    expected.append(result)
                else:  # maxent
                    if frame is None or "memory" not in frame:
                        continue
                    obs = _obs_from_capture(frame["obs"])
                    memory = _memory_from_capture(frame["memory"])
                    rng = recording_rng(4000 + b_index)
                    try:
                        result = maximum_entropy_reconstruction(
                            obs, belief.seat, memory, rng, config=belief.config
                        )
                    except ValueError:
                        result = None
                    stream += encode_observation(obs) + [int(belief.seat)]
                    stream += encode_memory(memory)
                    stream += [int(belief.config.n_particles)]
                    stream += encode_draws(rng.take())
                    expected.append(result)
        elif kind == "reservoir":
            from belief import BeliefConfig, BeliefState, Particle
            from memory import empty_memory
            from reservoir import ParticleReservoir

            if frame is not None:
                continue  # nothing here depends on which board is in a particle
            states = synthetic_states()
            # `(8, ...)` is the case where the belief's own particle count
            # binds instead of the capacity — mutation testing found that
            # `min()` unreachable while every capacity was the smaller of the
            # two.
            for capacity, n_arrivals, seat in ((3, 5, 0), (4, 2, 1), (2, 9, 0), (8, 3, 1)):
                arrivals = [
                    Particle(
                        state=states[i % len(states)],
                        weight=0.5 + 0.1 * i,
                        enemy_memory=empty_memory(*states[i % len(states)].armies.shape),
                        enemy_prev_action=None if i % 2 else (0, 1, 1, 3, 0),
                        history=(),
                    )
                    for i in range(n_arrivals)
                ]
                belief = BeliefState(
                    seat=seat,
                    particles=[
                        Particle(
                            state=states[(i + 3) % len(states)],
                            weight=1.0 / 3.0,
                            enemy_memory=empty_memory(
                                *states[(i + 3) % len(states)].armies.shape
                            ),
                            history=(),
                        )
                        for i in range(3)
                    ],
                    config=BeliefConfig(n_particles=6),
                    collapsed=False,
                )

                rng = recording_rng(5000 + capacity)
                reservoir = ParticleReservoir(capacity=capacity)
                for _ in range(2):
                    for particle in arrivals:
                        reservoir.admit(particle, rng)
                after_admit = _reservoir_snapshot(reservoir)
                sampled = reservoir.sample(rng) if reservoir.particles else None
                _replace_from_belief_recording(reservoir, belief, rng)
                after_replace = _reservoir_snapshot(reservoir)

                stream += [capacity, seat, len(arrivals)]
                for particle in arrivals:
                    stream += encode_particle_state(particle)
                stream += encode_belief_state(belief)
                stream += encode_draws(rng.take())
                expected.append((after_admit, sampled, after_replace))
        elif kind == "symmetry":
            from symmetry import (
                SYMMETRIES,
                decode_action as _decode,
                encode_action as _encode,
                transform_action_tuple,
            )
            from action import N_ACTIONS as _N

            if frame is not None:
                continue  # the group is a fixed object; one pass is enough
            for index, sym in enumerate(SYMMETRIES):
                stream += [index]
                out: list[int] = []
                for r in range(21):
                    for c in range(21):
                        nr, nc = sym.transform_rc(r, c)
                        out += [nr, nc]
                out += [sym.transform_dir(d) for d in range(4)]
                for idx in range(_N):
                    act = _decode(idx) if idx != _N - 1 else (1, 0, 0, 0, 0)
                    out.append(_encode(transform_action_tuple(act, sym)))
                expected.append(out)
        else:
            raise ValueError(f"unknown kind {kind!r}")

    return [len(expected)] + stream, expected


# --- running ----------------------------------------------------------------


def run_binary(kind: str, stream: list[int]) -> list[list[int]]:
    if not BINARY.is_file():
        raise FileNotFoundError(BINARY)
    text = " ".join(str(v) for v in stream)
    result = subprocess.run(
        [str(BINARY), "parity", kind],
        input=text,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if result.returncode != 0:
        raise RuntimeError(f"parity {kind} failed: {result.stderr[-2000:]}")
    return [
        [int(t) for t in line.split()]
        for line in result.stdout.splitlines()
        if line.strip()
    ]


# Worst |Δ| seen per float surface since the last `reset_stats()`. A tolerance
# is only meaningful next to the number it is not being reached by; M2 found a
# real precision bug hiding under a budget that never fired.
_WORST: dict[str, float] = {}


def reset_stats() -> None:
    _WORST.clear()


def worst_observed() -> dict[str, float]:
    return dict(_WORST)


def _note(name: str, value: float) -> None:
    if value > _WORST.get(name, -1.0):
        _WORST[name] = value


def _state_difference(label: str, want, got) -> list[str]:
    """Which planes of two `GameState`s disagree, named."""
    from state import states_equal

    if states_equal(want, got):
        return []
    fields = [
        name
        for name in ("armies", "ownership", "ownership_neutral", "generals",
                     "castles", "mountains", "passable")
        if not np.array_equal(getattr(want, name), getattr(got, name))
    ]
    if want.time != got.time:
        fields.append(f"time({got.time}!={want.time})")
    if want.winner != got.winner:
        fields.append(f"winner({got.winner}!={want.winner})")
    if not np.array_equal(want.general_positions, got.general_positions):
        fields.append("general_positions")
    return [f"{label}: state differs in {', '.join(fields) or 'an unnamed field'}"]


def _compare_belief(kind: str, case: int, want, got: list[int], at: int) -> list[str]:
    """Compare a returned `BeliefState` particle by particle.

    Weights are compared **exactly**. They are products and quotients of the
    same numbers on both sides, and the one thing that could make them differ
    — NumPy's pairwise sum — is reproduced deliberately and has its own
    surface. A tolerance here would hide precisely the bug that matters.
    """
    problems: list[str] = []
    mine, _ = _read_belief_at(got, at)
    if mine["seat"] != int(want.seat):
        problems.append(f"{kind}[{case}]: seat {mine['seat']} != {want.seat}")
    if mine["collapsed"] != bool(want.collapsed):
        problems.append(
            f"{kind}[{case}]: collapsed {mine['collapsed']} != {bool(want.collapsed)}"
        )
    if len(mine["particles"]) != want.n:
        return problems + [
            f"{kind}[{case}]: {len(mine['particles'])} particle(s) survived, "
            f"the oracle kept {want.n}"
        ]

    for index, (got_p, want_p) in enumerate(zip(mine["particles"], want.particles)):
        label = f"{kind}[{case}]: particle {index}"
        if got_p["weight"] != float(want_p.weight):
            _note(f"{kind}.weight.max", abs(got_p["weight"] - float(want_p.weight)))
            problems.append(
                f"{label} weight {got_p['weight']!r} != {float(want_p.weight)!r}"
            )
        want_prev = (
            tuple(int(v) for v in want_p.enemy_prev_action)
            if want_p.enemy_prev_action is not None
            else None
        )
        if got_p["enemy_prev_action"] != want_prev:
            problems.append(
                f"{label} previous enemy action {got_p['enemy_prev_action']} != {want_prev}"
            )
        if got_p["history_len"] != len(want_p.history):
            problems.append(
                f"{label} carries {got_p['history_len']} history frame(s), "
                f"the oracle {len(want_p.history)}"
            )
        else:
            # Depth alone is not enough. `_append_history` drops from the old
            # end when the window overflows, and a truncation keeps the same
            # count and the wrong frames.
            want_frames = [
                (
                    tuple(int(v) for v in f.my_action),
                    tuple(int(v) for v in f.enemy_action),
                    int(f.state.time),
                )
                for f in want_p.history
            ]
            if got_p["history"] != want_frames:
                first = next(
                    (
                        j
                        for j, (a, b) in enumerate(zip(got_p["history"], want_frames))
                        if a != b
                    ),
                    0,
                )
                problems.append(
                    f"{label} history differs at frame {first}: "
                    f"{got_p['history'][first]} != {want_frames[first]}"
                )
        problems += _state_difference(label, want_p.state, got_p["state"])
        for name in (
            "known_mountain", "known_passable_base", "known_castle", "own_general",
            "known_enemy_general", "ever_visible", "last_seen_turn",
            "remembered_owner", "remembered_army", "remembered_was_castle",
            "remembered_castle_owner",
        ):
            a = np.asarray(getattr(want_p.enemy_memory, name)).astype(np.int64)
            b = np.asarray(getattr(got_p["enemy_memory"], name)).astype(np.int64)
            if not np.array_equal(a, b):
                diff = np.flatnonzero(a.ravel() != b.ravel())
                problems.append(
                    f"{label} enemy memory {name} differs at {diff.size} cell(s), "
                    f"first {int(diff[0])}"
                )
                break
    return problems


def compare(kind: str, expected: list, actual: list[list[int]]) -> list[str]:
    """Every mismatch, described well enough to act on without a debugger."""
    problems: list[str] = []
    if len(actual) != len(expected):
        return [f"{kind}: {len(actual)} results for {len(expected)} cases"]

    for i, (want, got) in enumerate(zip(expected, actual)):
        if kind == "transition":
            nxt, info = want
            rust_state, rust_info, _ = decode_state(got)
            from state import states_equal

            if not states_equal(nxt, rust_state):
                fields = [
                    name
                    for name in ("armies", "ownership", "ownership_neutral",
                                 "generals", "castles", "mountains", "passable")
                    if not np.array_equal(getattr(nxt, name), getattr(rust_state, name))
                ]
                if nxt.time != rust_state.time:
                    fields.append(f"time({nxt.time}!={rust_state.time})")
                if nxt.winner != rust_state.winner:
                    fields.append(f"winner({nxt.winner}!={rust_state.winner})")
                problems.append(f"transition[{i}]: {', '.join(fields) or 'state'}")
            want_info = (
                int(info.army[0]), int(info.army[1]),
                int(info.land[0]), int(info.land[1]),
                int(bool(info.is_done)), int(info.winner), int(info.time),
            )
            if tuple(rust_info) != want_info:
                problems.append(f"transition[{i}]: info {rust_info} != {want_info}")
        elif kind == "observe":
            h, w = want.H, want.W
            n = h * w
            head = got[:7]
            if head != [h, w, want.turn, want.my_land, want.my_army, want.opp_land, want.opp_army]:
                problems.append(f"observe[{i}]: scalars {head}")
            grids = {
                "type": np.asarray(want.type_grid).ravel(),
                "owner": np.asarray(want.owner_grid).ravel(),
                "army": np.asarray(want.army_grid).ravel(),
            }
            for k, (name, plane) in enumerate(grids.items()):
                chunk = np.asarray(got[7 + k * n : 7 + (k + 1) * n], dtype=np.int32)
                if not np.array_equal(chunk, plane.astype(np.int32)):
                    bad = int(np.argmax(chunk != plane.astype(np.int32)))
                    problems.append(
                        f"observe[{i}]: {name} differs first at cell {bad} "
                        f"({chunk[bad]} != {plane[bad]})"
                    )
        elif kind == "mask":
            want_mask = np.asarray(want, dtype=bool)
            got_mask = np.asarray(got, dtype=bool)
            if got_mask.shape != want_mask.shape:
                problems.append(f"mask[{i}]: length {got_mask.shape} != {want_mask.shape}")
            elif not np.array_equal(want_mask, got_mask):
                diff = np.flatnonzero(want_mask != got_mask)
                problems.append(
                    f"mask[{i}]: {diff.size} action(s) differ, first index {int(diff[0])}"
                )
        elif kind == "order":
            if got != [want]:
                problems.append(f"order[{i}]: seat {got} != {want}")
        elif kind == "memory":
            got_memory = _decode_memory(got)
            for name in (
                "known_mountain", "known_passable_base", "known_castle", "own_general",
                "known_enemy_general", "ever_visible", "last_seen_turn",
                "remembered_owner", "remembered_army", "remembered_was_castle",
                "remembered_castle_owner",
            ):
                a = np.asarray(getattr(want, name)).ravel()
                b = np.asarray(got_memory[name]).ravel()
                if not np.array_equal(a.astype(np.int64), b.astype(np.int64)):
                    diff = np.flatnonzero(a.astype(np.int64) != b.astype(np.int64))
                    problems.append(
                        f"memory[{i}]: {name} differs at {diff.size} cell(s), "
                        f"first {int(diff[0])} ({b[diff[0]]} != {a[diff[0]]})"
                    )
        elif kind == "hash":
            want_bytes = list(want)
            if got != want_bytes:
                names = ["memory", "info_state_key", "enemy_info", "child_edge", "history"]
                for k, name in enumerate(names):
                    lo, hi = k * 32, (k + 1) * 32
                    if got[lo:hi] != want_bytes[lo:hi]:
                        problems.append(
                            f"hash[{i}]: {name} digest differs "
                            f"({bytes(got[lo:hi]).hex()[:16]}… != "
                            f"{bytes(want_bytes[lo:hi]).hex()[:16]}…)"
                        )
        elif kind == "tensor":
            # Tier 2: floats arrive as f32 bit patterns, so the channel is
            # lossless and the tolerance is the only thing being judged.
            from tensor import PLANE_NAMES

            got_tensor = (
                np.asarray(got, dtype=np.uint32).view(np.float32).reshape(49, 21, 21)
            )
            want_tensor = np.asarray(want, dtype=np.float32)
            if got_tensor.shape != want_tensor.shape:
                problems.append(f"tensor[{i}]: shape {got_tensor.shape}")
                continue
            delta = np.abs(got_tensor - want_tensor)
            worst = float(delta.max())
            _note("tensor.max", worst)
            # Tier 2 asks for 1e-6; the port achieves **bit-identical**, so
            # that is what is enforced. The looser figure is not a free pass:
            # at 1e-6 a mutation computing `army_value` in single precision
            # was invisible, which means the specified tolerance cannot tell
            # the two arithmetics apart. Should a platform ever produce real
            # drift here, the message says so and §5's 1e-6 is the floor to
            # fall back to — deliberately, not silently.
            if worst > 0.0:
                plane = int(np.unravel_index(int(delta.argmax()), delta.shape)[0])
                over_spec = int((delta > FLOAT_TOLERANCE).sum())
                problems.append(
                    f"tensor[{i}]: not bit-identical, max |Δ| {worst:.3g} on plane "
                    f"{plane} ({PLANE_NAMES[plane]}); {over_spec} cell(s) also exceed "
                    f"the §5 tolerance {FLOAT_TOLERANCE:g}"
                )
        elif kind == "net":
            offset = 0
            heads: dict[str, np.ndarray] = {}
            for name, count in NET_LAYOUT:
                heads[name] = _decode_f32(got[offset : offset + count])
                offset += count
            if offset != len(got):
                problems.append(f"net[{i}]: {len(got)} values for a {offset}-value layout")
                continue
            for name, _count in NET_LAYOUT:
                a, b = heads[name], want[name]
                delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
                mae = float(delta.mean())
                worst = float(delta.max())
                _note(f"net.{name}.mae", mae)
                _note(f"net.{name}.max", worst)
                if mae > NET_MAE_TOLERANCE or worst > NET_MAX_ABS_TOLERANCE:
                    at = int(delta.argmax())
                    problems.append(
                        f"net[{i}]: {name} MAE {mae:.3g} (budget {NET_MAE_TOLERANCE:g}), "
                        f"max |Δ| {worst:.3g} (cap {NET_MAX_ABS_TOLERANCE:g}) at index "
                        f"{at} ({a[at]!r} != {b[at]!r})"
                    )
            # Same trunk, three exits: the shared heads must be *identical*,
            # on each side independently. A drift here is not float noise, it
            # is an entry point computing something else.
            for side, values in (("rust", heads), ("torchscript", want)):
                for head in ENTRY_POINT_HEADS:
                    a = values[f"policy_only.{head}"]
                    b = values[f"policy_wdl.{head}"]
                    c = values[f"full.{head}"]
                    if not (np.array_equal(a, b) and np.array_equal(b, c)):
                        problems.append(
                            f"net[{i}]: {side} entry points disagree on {head}"
                        )
        elif kind == "prior":
            want_prior, want_value = want
            got_prior = _decode_f64(got[:-1])
            got_value = float(_decode_f64(got[-1:])[0])
            if got_prior.shape != want_prior.shape:
                problems.append(f"prior[{i}]: length {got_prior.shape}")
                continue
            # Illegal actions are not "small", they are zero. A tolerance would
            # hide a mask that leaked probability onto an unplayable move.
            leaked = np.flatnonzero((want_prior == 0.0) & (got_prior != 0.0))
            if leaked.size:
                problems.append(
                    f"prior[{i}]: {leaked.size} illegal action(s) got mass, "
                    f"first index {int(leaked[0])}"
                )
            delta = np.abs(got_prior - want_prior)
            _note("prior.max", float(delta.max()))
            _note("prior.value", abs(got_value - want_value))
            if float(delta.max()) > PRIOR_TOLERANCE:
                at = int(delta.argmax())
                problems.append(
                    f"prior[{i}]: max |Δ| {float(delta.max()):.3g} at action {at} "
                    f"({got_prior[at]!r} != {want_prior[at]!r})"
                )
            if abs(got_value - want_value) > PRIOR_TOLERANCE:
                problems.append(
                    f"prior[{i}]: value {got_value!r} != {want_value!r}"
                )
        elif kind == "symmetry":
            if got != list(want):
                diff = next(
                    (k for k, (a, b) in enumerate(zip(got, want)) if a != b), None
                )
                problems.append(f"symmetry[{i}]: first difference at position {diff}")
        elif kind == "toplegal":
            want_probs, want_single, want_top = want
            count = got[0]
            pairs = got[1 : 1 + 2 * count]
            at = 1 + 2 * count
            indices = np.asarray(pairs[0::2], np.int64)
            values = _decode_f64(pairs[1::2])
            mine = np.flatnonzero(np.asarray(want_probs) != 0.0)
            if not np.array_equal(indices, mine):
                problems.append(
                    f"toplegal[{i}]: softmax has mass on {indices.size} action(s), "
                    f"the oracle on {mine.size}"
                )
            else:
                delta = np.abs(values - np.asarray(want_probs)[mine])
                worst = float(delta.max()) if delta.size else 0.0
                _note("toplegal.softmax.max", worst)
                # Bit-exact, not within a tolerance. This surface was expected
                # to need one and turned out not to, and M2's lesson is that a
                # budget nothing approaches is a check that cannot fail. The
                # message names the fallback figure so a platform that really
                # does drift is a deliberate decision, not a silent pass.
                if worst > 0.0:
                    bad = int(delta.argmax())
                    over = int((delta > TOPLEGAL_TOLERANCE).sum())
                    problems.append(
                        f"toplegal[{i}]: softmax not bit-identical, max |Δ| "
                        f"{worst:.3g} at action {int(indices[bad])}; {over} "
                        f"element(s) also exceed the fallback floor "
                        f"{TOPLEGAL_TOLERANCE:g}"
                    )
            if got[at] != want_single:
                problems.append(
                    f"toplegal[{i}]: singleton picks action {got[at]}, not {want_single}"
                )
            at += 1
            n_top = got[at]
            at += 1
            mine_top = [tuple(got[at + 5 * j : at + 5 * j + 5]) for j in range(n_top)]
            want_top = [tuple(int(v) for v in a) for a in want_top]
            if mine_top != want_top:
                first = next(
                    (j for j, (a, b) in enumerate(zip(mine_top, want_top)) if a != b),
                    min(len(mine_top), len(want_top)),
                )
                problems.append(
                    f"toplegal[{i}]: candidate list differs from position {first} "
                    f"({mine_top[first:first + 2]} != {want_top[first:first + 2]}); "
                    f"{len(mine_top)} vs {len(want_top)} candidate(s)"
                )
        elif kind == "npsum":
            got_sum = float(_decode_f64(got)[0])
            _note("npsum.max", abs(got_sum - want))
            if got_sum != want:
                # Not a tolerance. This function exists to reproduce NumPy's
                # pairwise reduction exactly, because the ESS it feeds decides
                # whether a resample runs, and a resample consumes a draw.
                problems.append(
                    f"npsum[{i}]: {got_sum!r} != {want!r} — NumPy's reduction on "
                    "this host is not the eight-lane pairwise sum the port "
                    "reproduces (rng::npsum)"
                )
        elif kind == "argsort":
            got = got[1:]  # the length prefix; see the Rust arm
            if got != list(want):
                first = next(
                    (k for k, (a, b) in enumerate(zip(got, want)) if a != b), len(got)
                )
                problems.append(
                    f"argsort[{i}]: permutations differ from position {first} "
                    f"({got[first : first + 4]} != {list(want)[first : first + 4]}); "
                    "if this host's NumPy dispatches argsort to x86-simd-sort "
                    "(AVX-512-SKX) the oracle itself has changed tie order — see "
                    "rng::argsort_desc_numpy"
                )
        elif kind == "summary":
            want_summary, want_ess, want_ess_fraction = want
            planes = (
                "enemy_owner", "enemy_army_mean", "enemy_army_std",
                "enemy_general", "enemy_castle_owner", "enemy_visibility",
            )
            at = 0
            for name in planes:
                count = got[at]
                at += 1
                a = _decode_f32(got[at : at + count])
                at += count
                b = np.asarray(getattr(want_summary, name), np.float32).ravel()
                if a.shape != b.shape:
                    problems.append(f"summary[{i}]: {name} length {a.size} != {b.size}")
                    continue
                delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
                worst = float(delta.max()) if delta.size else 0.0
                _note(f"summary.{name}.max", worst)
                if worst > 0.0:
                    cell = int(delta.argmax())
                    problems.append(
                        f"summary[{i}]: {name} not bit-identical, max |Δ| "
                        f"{worst:.3g} at cell {cell} ({a[cell]!r} != {b[cell]!r})"
                    )
            got_fraction_f32 = float(_decode_f32(got[at : at + 1])[0])
            at += 1
            got_ess, got_ess_fraction = _decode_f64(got[at : at + 2])
            # The planes narrow to f32 where the Python's `.astype` does, but
            # ESS is compared in full precision: M2's lesson is that a
            # narrowing can hide an accumulation bug from every check that
            # only looks at the narrowed value.
            for name, a, b in (
                ("ess", float(got_ess), float(want_ess)),
                ("ess_fraction", float(got_ess_fraction), float(want_ess_fraction)),
            ):
                _note(f"summary.{name}.max", abs(a - b))
                if a != b:
                    problems.append(f"summary[{i}]: {name} {a!r} != {b!r}")
            if got_fraction_f32 != np.float32(want_summary.ess_fraction):
                problems.append(
                    f"summary[{i}]: ess_fraction narrowed to "
                    f"{got_fraction_f32!r}, not {np.float32(want_summary.ess_fraction)!r}"
                )
        elif kind == "propose":
            want_probs, want_actions, want_singletons, want_unique_keys = want
            at = 0
            for p_index, probs in enumerate(want_probs):
                count = got[at]
                at += 1
                pairs = got[at : at + 2 * count]
                at += 2 * count
                indices = np.asarray(pairs[0::2], np.int64)
                values = _decode_f64(pairs[1::2])
                mine = np.flatnonzero(np.asarray(probs) != 0.0)
                if not np.array_equal(indices, mine):
                    problems.append(
                        f"propose[{i}]: particle {p_index} has {indices.size} legal "
                        f"action(s), the oracle has {mine.size}"
                    )
                    continue
                # Uniform means `1/n` on both sides from the same integer, so
                # anything but equality is a different legal count or a
                # different division.
                delta = np.abs(values - np.asarray(probs)[mine])
                worst = float(delta.max()) if delta.size else 0.0
                _note("propose.max", worst)
                if worst > 0.0:
                    bad = int(delta.argmax())
                    problems.append(
                        f"propose[{i}]: particle {p_index} probability at action "
                        f"{int(indices[bad])} is {values[bad]!r}, not {np.asarray(probs)[mine][bad]!r}"
                    )
            for p_index, action in enumerate(want_actions):
                mine = tuple(got[at : at + 5])
                at += 5
                if mine != tuple(int(v) for v in action):
                    problems.append(
                        f"propose[{i}]: particle {p_index} sampled {mine} != {tuple(action)}"
                    )
            telemetry = got[at : at + 5]
            consumed = got[at + 5]
            # Every counter, not just the particle count. `n_unique_info_keys`
            # is the *only* observable the proposal information key reaches on
            # the deployed uniform path, so leaving it uncompared made the key
            # itself untested — which is what mutation testing found.
            want_telemetry = (
                len(want_actions),
                want_singletons,
                want_unique_keys,
                0,
                0,
            )
            if tuple(telemetry) != want_telemetry:
                problems.append(
                    f"propose[{i}]: telemetry {tuple(telemetry)} != {want_telemetry} "
                    "(particles, singletons, unique info keys, unique policy "
                    "inputs, policy batches)"
                )
            if consumed != len(want_actions):
                problems.append(
                    f"propose[{i}]: consumed {consumed} draw(s) for "
                    f"{len(want_actions)} particle(s)"
                )
        elif kind == "initbelief":
            want_candidates, want_belief = want
            count = got[0]
            at = 1
            if want_candidates is None:
                if count != -1:
                    problems.append(
                        f"initbelief[{i}]: rust listed {count} candidate(s) where "
                        "the oracle refused the frame"
                    )
            elif count != len(want_candidates):
                problems.append(
                    f"initbelief[{i}]: {count} candidate cell(s), the oracle "
                    f"found {len(want_candidates)}"
                )
                at += 2 * max(count, 0)
            else:
                mine = [tuple(got[at + 2 * j : at + 2 * j + 2]) for j in range(count)]
                at += 2 * count
                theirs = [tuple(int(v) for v in c) for c in want_candidates]
                if mine != theirs:
                    first = next(
                        (j for j, (a, b) in enumerate(zip(mine, theirs)) if a != b),
                        0,
                    )
                    problems.append(
                        f"initbelief[{i}]: prior support differs from position "
                        f"{first} ({mine[first:first + 3]} != {theirs[first:first + 3]})"
                    )
            ok = bool(got[at])
            at += 1
            if ok != (want_belief is not None):
                problems.append(
                    f"initbelief[{i}]: rust {'built' if ok else 'refused'} an "
                    f"initial belief, the oracle "
                    f"{'built' if want_belief is not None else 'refused'} one"
                )
            elif ok:
                problems += _compare_belief(kind, i, want_belief, got, at)
        elif kind in ("filter", "rejuvenate"):
            problems += _compare_belief(kind, i, want, got, 0)
        elif kind == "maxent":
            ok = bool(got[0])
            if ok != (want is not None):
                problems.append(
                    f"maxent[{i}]: rust {'built' if ok else 'refused'} a "
                    f"reconstruction, the oracle "
                    f"{'built' if want is not None else 'refused'} one"
                )
            elif ok:
                problems += _compare_belief(kind, i, want, got, 1)
        elif kind == "reservoir":
            want_admit, want_sampled, want_replace = want
            at = 0
            for label, snapshot in (("after admit", want_admit), ("after replace", want_replace)):
                admitted, version, count = got[at : at + 3]
                at += 3
                if (admitted, version, count) != (
                    snapshot["admitted_count"],
                    snapshot["version"],
                    len(snapshot["particles"]),
                ):
                    problems.append(
                        f"reservoir[{i}]: {label} counters "
                        f"(admitted {admitted}, version {version}, n {count}) != "
                        f"(admitted {snapshot['admitted_count']}, version "
                        f"{snapshot['version']}, n {len(snapshot['particles'])})"
                    )
                for p_index in range(count):
                    weight = float(_decode_f64(got[at : at + 1])[0])
                    at += 1
                    state, at = _read_state_at(got, at)
                    if p_index >= len(snapshot["particles"]):
                        continue
                    want_weight, want_state = snapshot["particles"][p_index]
                    if weight != want_weight:
                        problems.append(
                            f"reservoir[{i}]: {label} resident {p_index} weight "
                            f"{weight!r} != {want_weight!r}"
                        )
                    problems += _state_difference(
                        f"reservoir[{i}]: {label} resident {p_index}", want_state, state
                    )
                if label == "after admit":
                    has_sample = bool(got[at])
                    at += 1
                    if has_sample != (want_sampled is not None):
                        problems.append(f"reservoir[{i}]: sample presence differs")
                    elif has_sample:
                        state, at = _read_state_at(got, at)
                        problems += _state_difference(
                            f"reservoir[{i}]: sampled particle",
                            want_sampled.state,
                            state,
                        )
        elif kind == "cost":
            want_cost = np.asarray(want, dtype=np.int32).ravel()
            got_cost = np.asarray(got, dtype=np.int32)
            if not np.array_equal(want_cost, got_cost):
                diff = np.flatnonzero(want_cost != got_cost)
                problems.append(
                    f"cost[{i}]: {diff.size} cell(s) differ, first {int(diff[0])} "
                    f"({got_cost[diff[0]]} != {want_cost[diff[0]]})"
                )
    return problems


def check(
    kind: str,
    frames: list[dict],
    *,
    pairs_per_state: int = 12,
    batch: int = 25,
) -> tuple[int, list[str]]:
    """
    Run one kind over all frames, in batches.

    Batched because the whole corpus does not fit in one invocation: ~780 heavy
    frames times eight particles times a dozen action pairs is tens of
    thousands of cases, and a board state serializes to about 3,500 integers.
    One stream would be hundreds of megabytes of text through a pipe. Frames
    are independent, so splitting changes nothing about what is checked.
    """
    total = 0
    problems: list[str] = []
    chunks = [frames[i : i + batch] for i in range(0, len(frames), batch)] or [[]]
    for n, chunk in enumerate(chunks):
        # The synthetic block rides the first chunk only; it is appended inside
        # `build_cases` and would otherwise be re-run for every batch.
        stream, expected = build_cases(
            chunk, kind, pairs_per_state=pairs_per_state, synthetic=(n == 0)
        )
        if not expected:
            continue
        total += len(expected)
        for problem in compare(kind, expected, run_binary(kind, stream)):
            problems.append(f"batch {n}: {problem}")
    return total, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--smoke", action="store_true", help="use the committed slice")
    parser.add_argument("--limit", type=int, default=0, help="cap heavy frames")
    parser.add_argument("--pairs", type=int, default=12, help="action pairs per state")
    parser.add_argument(
        "--kinds",
        nargs="*",
        default=[
            "transition", "order", "observe", "mask", "cost",
            "memory", "hash", "tensor", "symmetry", "net", "prior",
            "npsum", "argsort", "summary", "propose", "filter",
            "rejuvenate", "maxent", "reservoir", "toplegal", "initbelief",
        ],
    )
    args = parser.parse_args(argv)

    if args.smoke:
        paths = [SMOKE_FIXTURE]
    else:
        paths = sorted(Path(args.corpus).rglob("*.capture.*.jsonl.gz"))
    if not paths:
        print(f"no captures under {args.corpus}", file=sys.stderr)
        return 1

    frames = load_frames(paths, limit=args.limit)
    print(f"{len(frames)} heavy frame(s) from {len(paths)} file(s)")

    reset_stats()
    failed = False
    for kind in args.kinds:
        count, problems = check(kind, frames, pairs_per_state=args.pairs)
        status = "ok" if not problems else f"{len(problems)} MISMATCH"
        print(f"  {kind:<11} {count:>6} case(s)  {status}")
        for problem in problems[:20]:
            print(f"      {problem}")
        failed = failed or bool(problems)

    stats = worst_observed()
    if stats:
        print("worst observed |\u0394| (the headroom under each budget):")
        for name in sorted(stats):
            print(f"  {name:<34} {stats[name]:.3g}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
