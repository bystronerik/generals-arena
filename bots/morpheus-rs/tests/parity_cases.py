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
import subprocess
import sys
from pathlib import Path

import numpy as np

BOT_DIR = Path(__file__).resolve().parents[1]
REPO = BOT_DIR.parents[1]
for entry in (REPO, REPO / "bots", REPO / "bots" / "morpheus"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

BINARY = BOT_DIR / "target" / "release" / "morpheus-rs"
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
