#!/usr/bin/env python3
"""Tier-1 parity: run a ported surface in both languages over recorded cases.

    python bots/morpheus-joe/tests/morpheus_joe_parity_cases.py --smoke
    python bots/morpheus-joe/tests/morpheus_joe_parity_cases.py --corpus data/morpheus/morpheus-rs/morpheus-rs-m0

Milestone M1 of docs/bots/morpheus-rs/rewrite-plan.md, §5 tier 1: transition
next-state, legal masks, observation emission, and the action codec must be
**bit-exact**, not close.

**Twenty-eight surfaces, and no network among them.** The Python oracle here is
Python morpheus, unchanged, and the corpus is morpheus-rs's own committed
smoke slice — neither is touched by the joe-net port, because none of the
surviving surfaces involves a network. Five went at N1
(docs/bots/morpheus-rs/joe-net-plan.md §8.3, §8.4):

* `tensor`, `net` and `prior` were the **TorchScript** oracle's, and morpheus's
  249,316-parameter CNN is not what this bot runs. Their three `torch.jit.load`
  calls are also where ~7 of this slice's ~11.4 s went, so what is left is
  faster.
* `summary` went with `summarize_belief`, whose only consumer was the 49-plane
  tensor build.
* `decide` — the whole no-search decision, network included — **loses its
  oracle outright**, as an accepted cost. No Python program plays morpheus's
  search over joe's net, and building one would mean porting joe's JAX network
  into Python morpheus for no other purpose.

The retired surfaces' strata stay in `fixtures/parity-smoke.jsonl.gz` as dead
weight; regenerating the corpus to remove them would cost more than the bytes.
N3 adds a **new** `prior` surface with a NumPy/JAX oracle over joe's remap and
value decode, which is where this port's own bugs will live.

The inputs are real: board states come from the belief particles recorded in
the M0 corpus, so every case is a position morpheus actually reasoned about.
Both implementations then run fresh on those inputs — the corpus supplies
states, not expected answers, so a case can exercise action pairs the recorded
game never played (a build, a deathtouch, a move from an unowned cell).

Wire format is a flat stream of integers, documented in
`crates/core/src/parity/`. Layouts here and there are positional and must be
edited together.

The bot-prefixed filename is not decoration. Both this fork and `morpheus-rs`
carry a `tests/` directory with no `__init__.py`, so two modules of the same
basename collide during collection and — worse — the first `import
parity_cases` to win would silently hand the other bot's harness to this one.
Every test module here is prefixed for the same reason; `bots/joe-rs/` and
`bots/unclejoe/` already follow the convention.
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

# `MORPHEUS_JOE_BINARY` lets `tools/mutation_check.py` point the harness at its
# own faster-building profile. The default is, and must stay, the release
# binary: the parity harness proper tests the thing that plays.
BINARY = Path(
    os.environ.get("MORPHEUS_JOE_BINARY", BOT_DIR / "target" / "release" / "morpheus-joe")
)
SMOKE_FIXTURE = BOT_DIR / "tests" / "fixtures" / "parity-smoke.jsonl.gz"
DEFAULT_CORPUS = REPO / "data" / "morpheus" / "morpheus-rs" / "morpheus-rs-m0"

PASS = (1, 0, 0, 0, 0)

# The two float tolerances this harness used to carry — the network heads at
# 1e-5 MAE and the prior at 5e-6 — went with the `net` and `prior` surfaces and
# the TorchScript oracle behind them. **Nothing left here has a float tolerance
# except the one below, which is a floor and not a budget.** joe's forward
# keeps its own bounds in `bots/joe-rs/tests/test_parity.py`, against the
# JAX/eqx oracle, and this fork inherits them transitively by byte-identity
# (joe-net-plan §8.2) rather than restating them.

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


# --- serialization (mirrors crates/core/src/parity/codec.rs) ----------------


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


def _copy_state(state):
    """A `GameState` whose arrays are its own.

    `NamedTuple._replace()` copies the tuple and shares every array in it, so
    two "copies" made that way are one board wearing two names.
    """
    return state._replace(
        armies=np.array(state.armies, np.int32),
        ownership=np.array(state.ownership, bool),
        ownership_neutral=np.array(state.ownership_neutral, bool),
        generals=np.array(state.generals, bool),
        castles=np.array(state.castles, bool),
        mountains=np.array(state.mountains, bool),
        passable=np.array(state.passable, bool),
        general_positions=np.array(state.general_positions, np.int32),
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


# --- positions the corpus cannot reach --------------------------------------
#
# The mutation pass over M5 came back with 63 survivors, and they had one
# shape: every rule gated on a *game phase* — the garrison window, the castle
# window, a live threat, a reachable kill — survived, because the recorded
# frames on the smoke slice never sit in one and `synthetic_states()` was built
# for M1's transition boundaries on a 5x5 board.
#
# These positions are built the other way round: from the rule backwards. Each
# one exists to put exactly one gate in its firing range, and the observation
# and memory are constructed directly rather than emitted from a `GameState`,
# because the tactical layer only ever reads those two and fog would otherwise
# have to be arranged rather than stated.


def _tactical_obs(h, w, turn, *, own=(), enemy=(), neutral_army=(), types=None,
                  my_land=0, my_army=0, opp_land=0, opp_army=0):
    """An observation built cell by cell. `own`/`enemy` are `((r, c), army)`."""
    from observe import ArrayObservation

    type_grid = np.ones((h, w), np.int32)
    owner_grid = np.zeros((h, w), np.int32)
    army_grid = np.zeros((h, w), np.int32)
    for (r, c), kind in (types or {}).items():
        type_grid[r, c] = kind
    for (r, c), army in own:
        owner_grid[r, c] = 1
        army_grid[r, c] = army
    for (r, c), army in enemy:
        owner_grid[r, c] = 2
        army_grid[r, c] = army
    for (r, c), army in neutral_army:
        army_grid[r, c] = army
    return ArrayObservation(
        H=h, W=w, turn=turn,
        my_land=my_land or int((owner_grid == 1).sum()),
        my_army=my_army or int(army_grid[owner_grid == 1].sum()),
        opp_land=opp_land or int((owner_grid == 2).sum()),
        opp_army=opp_army or int(army_grid[owner_grid == 2].sum()),
        type_grid=type_grid, owner_grid=owner_grid, army_grid=army_grid,
    )


def _tactical_memory(obs, *, own_general=None, enemy_general=None, castles=()):
    from memory import VisibleMemory

    h, w = int(obs.H), int(obs.W)
    types = np.asarray(obs.type_grid, np.int32)
    mountain = types == 2
    known_castle = np.zeros((h, w), bool)
    for cell in castles:
        known_castle[cell] = True
    own_gen = np.zeros((h, w), bool)
    if own_general is not None:
        own_gen[own_general] = True
    enemy_gen = np.zeros((h, w), bool)
    if enemy_general is not None:
        enemy_gen[enemy_general] = True
    return VisibleMemory(
        H=h, W=w,
        known_mountain=mountain,
        known_passable_base=~(mountain | known_castle | own_gen | enemy_gen),
        known_castle=known_castle,
        own_general=own_gen,
        known_enemy_general=enemy_gen,
        ever_visible=np.ones((h, w), bool),
        last_seen_turn=np.full((h, w), int(obs.turn), np.int32),
        remembered_owner=np.asarray(obs.owner_grid, np.int32).astype(np.int8),
        remembered_army=np.asarray(obs.army_grid, np.int32).copy(),
        remembered_was_castle=known_castle.copy(),
        remembered_castle_owner=np.where(
            known_castle & (np.asarray(obs.owner_grid) == 1), 1, 0
        ).astype(np.int8),
    )


def _frame_like(seat=0, prev=PASS, recent=(), action=PASS, believed=(), prior=None):
    """A synthetic stand-in for a captured frame, carrying only what M5 reads.

    `believed` is `((r, c), weight)` per particle: the tactical layer's only
    use of a belief is `believed_enemy_general`, and two cells with unequal
    weights is what makes the weighted mode distinguishable from a count.

    `prior` states the unshaped root prior instead of taking the default ramp.
    The blend caps a heuristic at a 10x nudge either way, so a rule that only
    fires when the *network* insists — against a heuristic that scored the
    action at zero — needs a prior whose spread exceeds that cap. A position
    cannot reach one by being arranged; it has to be said.
    """
    particles = [
        {"state": {"general_positions": [[0, 0], list(cell)]}, "weight": weight}
        for cell, weight in believed
    ]
    frame = {
        "seat": seat,
        "prev_action": list(prev),
        "recent_actions": [list(a) for a in recent],
        "action": list(action),
        "belief": {"seat": seat, "particles": particles} if particles else None,
    }
    if prior is not None:
        frame["root"] = {"unshaped_prior": [float(v) for v in prior]}
    return frame


def tactical_positions() -> list[tuple]:
    """`(obs, memory, frame)` for each tactical gate, one gate at a time."""
    out: list[tuple] = []

    # (a) The garrison window with a live, imminent threat. Reaches the
    # threat-aware floor, `general_threat`'s tie rule, both branches of
    # `defend_general_move`, the castle anchor standing down, and the garrison
    # release refusing to ship mid-emergency.
    for garrison, threat_army, threat_at in ((20, 40, (6, 9)), (44, 40, (6, 8))):
        obs = _tactical_obs(
            13, 13, 300,
            own=[((6, 6), garrison), ((6, 4), 30), ((6, 5), 4), ((5, 6), 3),
                 ((6, 8), 50) if threat_at != (6, 8) else ((7, 8), 50)],
            enemy=[(threat_at, threat_army), ((6, 10), 3)],
        )
        out.append((obs, _tactical_memory(obs, own_general=(6, 6)),
                    _frame_like(prev=(0, 6, 5, 3, 0), recent=[(0, 6, 5, 3, 0)])))

    # (b) The castle savings window. One surcharge-free candidate far from every
    # structure, one nearer a visible enemy, and one already holding a pile —
    # the sticky rule, the safe-distance rule and the base-price rule each need
    # a different pair to be decidable. Turn 200 is on the tithe period.
    for site_army in (12, 40):
        obs = _tactical_obs(
            15, 15, 200,
            own=[((1, 1), 15), ((1, 2), 6), ((8, 8), site_army), ((8, 9), 9),
                 ((7, 8), 20), ((1, 12), 30), ((2, 12), 4)],
            enemy=[((1, 14), 5)],
        )
        out.append((obs, _tactical_memory(obs, own_general=(1, 1)),
                    _frame_like(prev=(0, 7, 8, 1, 0))))

    # (c) A reachable kill. The enemy general is visible and thin, an own stack
    # sits three steps out, there is an own army to collect on the way and a
    # neutral to pay for — and a fogged cell on the alternative descent, which
    # is the only thing that makes "refuse to price a fogged cell" decidable.
    obs = _tactical_obs(
        13, 13, 400,
        own=[((6, 6), 14), ((6, 7), 6), ((5, 5), 3), ((6, 2), 40)],
        enemy=[((6, 10), 2), ((6, 9), 1)],
        neutral_army=[((6, 8), 2)],
        types={(6, 10): 4, (5, 8): 0, (5, 9): 5},
    )
    out.append((obs, _tactical_memory(obs, own_general=(6, 2), enemy_general=(6, 10)),
                _frame_like(prev=(0, 6, 6, 3, 0),
                            # Two particles at one cell, one at another with more
                            # weight: the weighted mode and the count mode disagree,
                            # which is the only way to tell them apart.
                            believed=[((2, 2), 0.2), ((2, 2), 0.2), ((6, 10), 0.6)])))

    # (d) The deathtouch regime: any touch wins, so the surplus gate must not
    # suppress it and the capture is forced however thin the attacker.
    obs = _tactical_obs(
        9, 9, 850,
        own=[((4, 3), 3), ((4, 2), 30)],
        enemy=[((4, 4), 99)],
        types={(4, 4): 4},
    )
    out.append((obs, _tactical_memory(obs, own_general=(4, 2), enemy_general=(4, 4)),
                _frame_like(prev=(0, 4, 2, 3, 0))))

    # (e) An oscillation corridor with a real history. Every history-dependent
    # rule — the eight-move window, the enemy-land and new-vision exemptions,
    # commitment hysteresis, the retreat redirect — needs a previous move that
    # is a move, which no synthetic frame had before.
    # Ten, not eight: with exactly eight the last-eight window and the
    # first-eight window are the same slice and the rule is undecidable.
    history = [(0, 4, c, 3, 0) for c in range(1, 11)]
    obs = _tactical_obs(
        11, 11, 250,
        own=[((4, c), 2) for c in range(1, 9)] + [((4, 0), 25), ((4, 9), 12)],
        enemy=[((4, 10), 4)],
    )
    out.append((obs, _tactical_memory(obs, own_general=(4, 0)),
                _frame_like(prev=(0, 4, 8, 3, 0), recent=history,
                            action=(0, 4, 9, 2, 0))))

    # (e2) The two exemptions from the oscillation ban, one per position: a
    # reverse onto enemy land, and a reverse onto a cell that unlocks fog.
    # Without these the ban looks unconditional and deleting either check
    # changes nothing.
    obs = _tactical_obs(
        9, 9, 250,
        own=[((4, 3), 9), ((4, 4), 6)],
        enemy=[((4, 5), 3)],
    )
    out.append((obs, _tactical_memory(obs, own_general=(4, 3)),
                _frame_like(prev=(0, 4, 5, 2, 0), recent=[(0, 4, 5, 2, 0)],
                            action=(0, 4, 4, 3, 0))))
    obs = _tactical_obs(
        9, 9, 250,
        own=[((4, 1), 9), ((4, 2), 6)],
        types={(r, c): 0 for r in range(9) for c in range(5, 9)},
    )
    out.append((obs, _tactical_memory(obs, own_general=(4, 1)),
                _frame_like(prev=(0, 4, 2, 2, 0), recent=[(0, 4, 2, 2, 0)],
                            action=(0, 4, 1, 3, 0))))

    # (f) Pre-contact with a fat pile on the general: the evacuation redirect,
    # the structure-idle score branch, and the pre-contact ban on stacking onto
    # a structure all need this and nothing else.
    obs = _tactical_obs(
        11, 11, 60,
        own=[((5, 5), 40), ((5, 6), 3), ((4, 5), 2)],
        types={(5, 5): 4, (2, 2): 2},
    )
    out.append((obs, _tactical_memory(obs, own_general=(5, 5)),
                _frame_like(prev=(0, 5, 6, 3, 0))))

    # (g) The garrison release: past the release factor, no threat in sight.
    # Turn 351, not 350: on a tithe turn the savings gather returns first and
    # the release below is never reached.
    obs = _tactical_obs(
        11, 11, 351,
        own=[((5, 5), 60), ((5, 6), 4), ((6, 5), 3), ((0, 0), 8)],
        types={(5, 5): 4},
    )
    out.append((obs, _tactical_memory(obs, own_general=(5, 5)),
                _frame_like(prev=(0, 5, 6, 3, 0), action=(0, 5, 6, 3, 0))))

    out.extend(_m6_tactical_positions())
    return out


def _m6_tactical_positions() -> list[tuple]:
    """M6: the positions M5's eleven could not reach.

    M5 left twenty-nine mutations alive, and the diagnosis in
    `parity-harness.md` split them three ways: rules needing **two** gates in
    range at once, rules needing an **exact numeric coincidence**, and
    exemptions **shadowed by the next exemption**. Each position below is
    built for one of those, and the arithmetic that puts the gate on its
    boundary is spelled out rather than tuned by trial — a position that only
    happens to sit on a boundary stops sitting on it the next time a constant
    moves, and would then quietly stop testing anything.
    """
    out: list[tuple] = []

    # (h) The garrison floor's one exemption: a capture that *wins*. Turn 600
    # is past the castle window, so the anchor cannot also fire and confuse
    # which rule moved the mask. Floor is 10 (own total 26 -> the minimum),
    # every full move off the general leaves 1, and the only one that survives
    # the ban is the touch on the enemy general with 19 against its 2.
    obs = _tactical_obs(
        11, 11, 600,
        own=[((5, 5), 20), ((2, 2), 6)],
        enemy=[((5, 6), 2)],
        types={(5, 5): 4, (5, 6): 4},
    )
    out.append((obs, _tactical_memory(obs, own_general=(5, 5), enemy_general=(5, 6)),
                _frame_like(prev=(0, 2, 2, 3, 0), action=(0, 5, 5, 3, 0))))

    # (i) The half-split remainder, on the one garrison where it is decidable.
    # A split leaves `ceil(a/2)`; the mutation leaves `floor(a/2)`. The two
    # differ only on an odd garrison, and only change the ban when the floor
    # falls exactly between them — so the garrison must be `2 * floor - 1`,
    # here 19 against a floor of 10. Full moves leave 1 and are banned in both.
    #
    # The same position is the only one where the release *factor* is
    # decidable: the release wants `2 x floor` = 20 and the mask needs
    # `ceil(a/2) >= floor`, and 19 is the single value that fails the first
    # while passing the second. The chosen action comes from (2, 2) so the
    # release's "not already shipping from the general" gate is open.
    obs = _tactical_obs(
        11, 11, 600,
        own=[((5, 5), 19), ((2, 2), 6), ((2, 3), 1)],
        types={(5, 5): 4},
    )
    out.append((obs, _tactical_memory(obs, own_general=(5, 5)),
                _frame_like(prev=PASS, action=(0, 2, 2, 3, 0))))

    # (j) The castle anchor standing down for a threat: both windows open at
    # once. The site is 18 cells from the general (any nearer and the build
    # surcharge disqualifies it) and 16 from the enemy (the safe-distance rule
    # wants 4), while a 30-army stack two steps from a garrison of 3 is a live
    # threat. Without the stand-down the anchor would pin the savings pile
    # while the general falls.
    obs = _tactical_obs(
        15, 15, 200,
        own=[((1, 1), 3), ((10, 10), 20), ((10, 11), 2), ((11, 10), 1)],
        enemy=[((1, 3), 30)],
        types={(1, 1): 4},
    )
    out.append((obs, _tactical_memory(obs, own_general=(1, 1)),
                _frame_like(prev=(0, 10, 10, 3, 0))))

    # (k) A threat that ties. Arrival is `army - d` and the bar is
    # `garrison + d/2`: 13 army at distance 2 arrives with 11 against a
    # garrison of 10 plus 1. Ties count — the attacker may collect on the way —
    # and only an exact tie can tell `>=` from `>`.
    obs = _tactical_obs(
        11, 11, 600,
        own=[((5, 5), 10), ((2, 2), 5)],
        enemy=[((5, 7), 13)],
        types={(5, 5): 4},
    )
    out.append((obs, _tactical_memory(obs, own_general=(5, 5)),
                _frame_like(prev=(0, 2, 2, 3, 0))))

    # (l) Reinforcement must land *before* the threat, not with it. The threat
    # is three steps out, so the window is `1..=2`; a 3-army tip sits at 1 and
    # a 50-army stack at exactly 3. Off-by-one and the big stack wins the
    # "biggest that arrives in time" test — and arrives one turn too late.
    obs = _tactical_obs(
        13, 13, 600,
        own=[((6, 6), 4), ((6, 5), 3), ((6, 4), 1), ((6, 3), 50)],
        enemy=[((6, 9), 20)],
        types={(6, 6): 4},
    )
    out.append((obs, _tactical_memory(obs, own_general=(6, 6)),
                _frame_like(prev=(0, 6, 5, 3, 0))))

    # (m) A threat inside the defence *radius* (4) but outside the *forced*
    # window (3): radius loitering, which the hard rule must not answer. The
    # reinforcement exists and is legal, so deleting the imminence test
    # commits it.
    obs = _tactical_obs(
        13, 13, 600,
        own=[((6, 6), 4), ((6, 5), 8)],
        enemy=[((6, 10), 30)],
        types={(6, 6): 4},
    )
    out.append((obs, _tactical_memory(obs, own_general=(6, 6)),
                _frame_like(prev=PASS, action=(0, 6, 5, 2, 0))))

    # (n) The kill/defence race, at a tie. The march reaches the enemy general
    # in two steps and the enemy stack reaches ours in two: the tie goes to the
    # kill. Both plans exist and both are legal, so the comparison is the only
    # thing deciding, which is what `<=` against `<` needs.
    obs = _tactical_obs(
        15, 15, 600,
        own=[((7, 2), 3), ((7, 3), 4), ((7, 4), 20)],
        enemy=[((7, 6), 1), ((5, 2), 10)],
        types={(7, 2): 4, (7, 6): 4},
    )
    out.append((obs, _tactical_memory(obs, own_general=(7, 2), enemy_general=(7, 6)),
                _frame_like(prev=(0, 7, 4, 3, 0))))

    # (o) The kill march pricing a fogged cell. Two cells sit one step down the
    # gradient: a fogged one, and a neutral holding 5. A fogged cell *reads* as
    # army 0, so it looks free and would be chosen — which is exactly the
    # doomed march the rule refuses. The general is twelve steps away, outside
    # the kill horizon, so this stack is the only source.
    obs = _tactical_obs(
        13, 13, 600,
        own=[((5, 3), 30), ((0, 0), 5)],
        enemy=[((6, 6), 1)],
        neutral_army=[((6, 3), 5)],
        types={(0, 0): 4, (6, 6): 4, (5, 4): 0},
    )
    out.append((obs, _tactical_memory(obs, own_general=(0, 0), enemy_general=(6, 6)),
                _frame_like(prev=(0, 5, 3, 1, 0))))

    # (p) The same fork with the fog removed and the neutral owned: collecting
    # 5 of our own beats crossing a free empty cell, and the preference is only
    # visible when the own cell is the *expensive*-looking one.
    obs = _tactical_obs(
        13, 13, 600,
        own=[((5, 3), 30), ((6, 3), 5), ((0, 0), 5)],
        enemy=[((6, 6), 1)],
        types={(0, 0): 4, (6, 6): 4},
    )
    out.append((obs, _tactical_memory(obs, own_general=(0, 0), enemy_general=(6, 6)),
                _frame_like(prev=(0, 5, 3, 1, 0))))

    # (q) The tithe yielding to a take. Turn 200 is on the tithe period and
    # inside the castle window, the site holds 20 of the 35 it needs (so the
    # build rule stays quiet), a gather step exists from (10, 8) — and the
    # action under test captures an enemy cell. Never trade a capture for a
    # shuffle.
    obs = _tactical_obs(
        15, 15, 200,
        own=[((1, 1), 12), ((10, 10), 20), ((10, 9), 1), ((10, 8), 9), ((4, 1), 6)],
        enemy=[((5, 1), 2)],
        types={(1, 1): 4},
    )
    out.append((obs, _tactical_memory(obs, own_general=(1, 1)),
                _frame_like(prev=PASS, action=(0, 4, 1, 1, 0))))

    # (r) The release standing down mid-emergency, which needs a threat the
    # mask does not answer for us. A lethal threat forces the play mask's floor
    # above any half of the garrison, so every general move would be banned —
    # and the ban is skipped wholesale when it would leave nothing legal. The
    # general is therefore the only cell that can move at all (the other two
    # hold one army each), the threat loiters at distance 4 so forced defence
    # stays quiet, and 24 against a floor of 10 clears the release factor.
    #
    # The four single-army neighbours are not decoration. The release takes an
    # argmax over the *splits* and the pass redirect over every legal move, so
    # the two agree whenever a split is the best action overall — which is what
    # a bare general produces, and why a first attempt at this position could
    # not tell the rule from its deletion. Surrounding the general makes the
    # eastward full move the clear top and the answers separate.
    obs = _tactical_obs(
        11, 11, 600,
        own=[((5, 5), 24), ((4, 4), 1), ((4, 6), 1), ((6, 4), 1), ((6, 6), 1)],
        enemy=[((5, 9), 40)],
        types={(5, 5): 4},
    )
    out.append((obs, _tactical_memory(obs, own_general=(5, 5)), _frame_like(prev=PASS)))

    # (s) A reverse onto enemy land, with the *next* exemption closed. M5's
    # position for this rule reached the vision exemption first and returned
    # there, so the enemy-land branch was never the reason. Owning both cells
    # flanking the enemy makes its whole 3x3 box already visible, so the
    # reverse reveals nothing and only the enemy-land rule can allow it.
    obs = _tactical_obs(
        9, 9, 250,
        own=[((4, 4), 9), ((4, 6), 5), ((0, 0), 3)],
        enemy=[((4, 5), 3)],
        types={(0, 0): 4},
    )
    out.append((obs, _tactical_memory(obs, own_general=(0, 0)),
                _frame_like(prev=(0, 4, 5, 2, 0), recent=[(0, 4, 5, 2, 0)],
                            action=(0, 4, 4, 3, 0))))

    # (t) A reverse that unlocks vision, onto a cell we do not own. M5's
    # position reversed onto own land, and an owned cell reveals nothing by
    # definition — the count returns zero before the exemption is reached.
    # (4, 3) is neutral and its box still holds three unseen cells.
    obs = _tactical_obs(
        9, 9, 250,
        own=[((4, 1), 9), ((4, 2), 6), ((0, 0), 3)],
        types={(0, 0): 4},
    )
    out.append((obs, _tactical_memory(obs, own_general=(0, 0)),
                _frame_like(prev=(0, 4, 3, 2, 0), recent=[(0, 4, 3, 2, 0)],
                            action=(0, 4, 2, 3, 0))))

    # (u) The oscillation window's *end*. Nine moves, and the one being
    # reversed is the ninth: inside the last eight, outside the first eight.
    # M5's corridor had ten moves but reversed the eighth, which both windows
    # contain — the position was long enough and pointed at the wrong move.
    obs = _tactical_obs(
        11, 11, 250,
        own=[((4, c), 2) for c in range(1, 10)] + [((4, 0), 25)],
        enemy=[((4, 10), 4)],
        types={(4, 0): 4},
    )
    out.append((obs, _tactical_memory(obs, own_general=(4, 0)),
                _frame_like(prev=(0, 4, 9, 3, 0),
                            recent=[(0, 4, c, 3, 0) for c in range(1, 10)],
                            action=(0, 4, 10, 2, 0))))

    # (v) The redirect refusing an own-land retreat. The chosen move retreats
    # onto own land, which is what summons the redirect in the first place —
    # and the prior's argmax is *another* retreat, so the rule is the only
    # thing standing between the two.
    #
    # The prior has to say so explicitly. Shaping scores every own-land
    # non-progressing move at 0.01 against a top of ~309, and the blend clips a
    # heuristic to a 10x nudge either way, so the heuristic term alone spans
    # 100x. A prior that merely *prefers* the retreat loses to that; one that
    # insists by 150x wins, and "the network insists on a retreat" is exactly
    # the case the rule exists to overrule.
    obs = _tactical_obs(
        11, 11, 60,
        own=[((5, 5), 40), ((5, 6), 3), ((5, 7), 9), ((5, 8), 2), ((4, 7), 6),
             ((6, 7), 4), ((4, 5), 2), ((5, 4), 2), ((6, 5), 2)],
        types={(5, 5): 4},
    )
    memory = _tactical_memory(obs, own_general=(5, 5))
    from action import N_ACTIONS, encode_action
    from tactics import play_mask

    live = np.flatnonzero(np.asarray(play_mask(obs, memory), dtype=bool))
    insisted = np.zeros(N_ACTIONS, dtype=np.float64)
    insisted[live] = 1.0
    insisted[encode_action((0, 5, 7, 3, 0))] = 150.0
    insisted /= insisted.sum()
    out.append((obs, memory,
                _frame_like(prev=(0, 5, 6, 3, 0), action=(0, 5, 6, 3, 0),
                            prior=insisted)))

    return out


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

# Method codes shared with `read_draws` in crates/core/src/parity/codec.rs.
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



# --- M5: tactics, search math, and the decision -----------------------------

# rewrite-plan §5 budgets 1e-9 for the heuristic/shaping scores in f64. Both
# sides compute them in f64 throughout and agree to the last bit, so the
# harness enforces bit-exactness and keeps the specified figure as the floor to
# fall back to — M2's precedent, for M2's reason: the budget is a thousand
# times looser than anything a run reaches, so on its own it could not fail.
SHAPING_TOLERANCE = 1e-9

# `matrix` is the one surface where bit-exactness is unreachable *because of
# the oracle*. The Python writes `q_eff @ sigma_enemy`, and NumPy sends `@` on
# f64 to BLAS — Accelerate here, OpenBLAS on the x86 container — whose
# reduction order is the vendor's and matches neither a sequential sum nor
# NumPy's own pairwise one (measured: mean 0.4 ulp, max 3, over 3,000 random
# simplex vectors at the widths the search uses). So the port reduces with
# `npsum` and the tolerance is set from measurement. See `matrix.rs`.
MATRIX_TOLERANCE = 5e-12


def _lite_belief_ints(frame) -> list[int]:
    """`seat n` then per particle `weight_bits gr gc` — see `read_belief_lite`."""
    belief = (frame or {}).get("belief")
    if not belief:
        return [int((frame or {}).get("seat", 0)), 0]
    seat = int(belief["seat"])
    enemy = 1 - seat
    out = [seat, len(belief["particles"])]
    for particle in belief["particles"]:
        gp = np.asarray(particle["state"]["general_positions"], dtype=np.int64)
        out += _f64_bits([float(particle["weight"])])
        out += [int(gp[enemy][0]), int(gp[enemy][1])]
    return out


def _lite_belief_object(frame):
    """The Python-side stand-in the Rust `read_belief_lite` builds."""
    from belief import BeliefConfig, BeliefState, Particle

    belief = (frame or {}).get("belief")
    seat = int(belief["seat"]) if belief else int((frame or {}).get("seat", 0))
    if not belief:
        return BeliefState(seat=seat, particles=[], config=BeliefConfig(), collapsed=False)
    enemy = 1 - seat
    particles = []
    for particle in belief["particles"]:
        gp = np.asarray(particle["state"]["general_positions"], dtype=np.int32)
        state = _blank(1, 1)
        state.general_positions[enemy] = gp[enemy]
        from memory import empty_memory

        particles.append(
            Particle(
                state=state,
                weight=float(particle["weight"]),
                enemy_memory=empty_memory(1, 1),
                enemy_prev_action=None,
                history=(),
            )
        )
    return BeliefState(seat=seat, particles=particles, config=BeliefConfig(), collapsed=False)


def _history_ints(frame) -> list[int]:
    """`has_prev action5 n_recent action5…` from the frame's recorded history."""
    prev, recent = _history_of(frame)
    out = [int(prev is not None)] + [int(v) for v in (prev or PASS)]
    out.append(len(recent))
    for action in recent:
        out += [int(v) for v in action]
    return out


def _history_of(frame):
    """The pre-move history the hard-rule layer saw, as the capture recorded it."""
    prev = tuple(int(v) for v in (frame or {}).get("prev_action", PASS))
    recent = [tuple(int(v) for v in a) for a in (frame or {}).get("recent_actions", [])]
    return prev, recent


def _shaping_knobs(frame):
    """The deployed blend knobs. Fixed, because both bots ship the same file."""
    import math

    del frame
    return 1.0, 1.0, math.log(10.0), 1e-3


def _network_prior_for(frame, obs, memory):
    """The shaped-prior input: the frame's own unshaped root prior when it has
    one, else a deterministic ramp over the play mask.

    A recorded prior is the realistic input; the ramp exists so a frame whose
    root inference was skipped still exercises the blend, and so the blend sees
    a distribution that is not the network's own (a prior proportional to the
    heuristic would hide a blend that ignored one of its two inputs).
    """
    from action import N_ACTIONS

    recorded = (frame or {}).get("root", {}).get("unshaped_prior")
    if recorded is not None:
        return np.asarray(recorded, dtype=np.float64).reshape(-1)
    from tactics import play_mask

    mask = np.asarray(play_mask(obs, memory), dtype=bool)
    prior = np.zeros(N_ACTIONS, dtype=np.float64)
    live = np.flatnonzero(mask)
    if live.size:
        ramp = 1.0 + (np.arange(live.size, dtype=np.float64) % 7)
        prior[live] = ramp / ramp.sum()
    return prior


def matrix_cases() -> list[dict]:
    """Hand-built strategy shapes for the regret cycle.

    Deliberately irregular, for M4's reason: the deployed configuration makes a
    whole class of arithmetic invisible. Equal priors normalize to themselves,
    an all-zero regret vector never exercises the positive branch, and a single
    enemy hash never exercises the particle weighting. Every one of those is a
    case here.
    """
    rng = np.random.default_rng(20260809)
    cases: list[dict] = []
    shapes = [(1, 1, 1), (2, 3, 1), (8, 12, 3), (16, 12, 8), (5, 4, 2)]
    for n_a, n_b, n_h in shapes:
        for variant in range(3):
            if variant == 0:            # everything zero: the fallback branches
                regrets = np.zeros(n_a)
                prior = np.zeros(n_a)
                weights = np.zeros(n_h)
            elif variant == 1:          # uniform: the deployed shape
                regrets = np.zeros(n_a)
                prior = np.full(n_a, 1.0 / n_a)
                weights = np.full(n_h, 1.0 / n_h)
            else:                       # irregular, with negatives to be clipped
                regrets = rng.standard_normal(n_a) * 3.0
                prior = np.abs(rng.standard_normal(n_a))
                weights = rng.standard_normal(n_h)
            if variant == 2 and n_a > 1:
                # One live entry, the rest below the selector's 1e-6 * max
                # cutoff, and the *average strategy* piled on a dead one. This
                # is the only shape in which "drop actions the live prior
                # zeroed" changes the answer; a prior of plain magnitudes never
                # goes near the cutoff.
                prior = np.full(n_a, 1e-12)
                prior[-1] = 1.0
            cases.append(
                {
                    "n_a": n_a,
                    "n_b": n_b,
                    "n_h": n_h,
                    "n": [0, 1, 7, 64, 4096][(n_a + variant) % 5],
                    "regrets": regrets,
                    "prior": prior,
                    "avg_strategy": (
                        np.linspace(9.0, 1.0, n_a)
                        if variant == 2
                        else np.abs(rng.standard_normal(n_a)) * (variant != 0)
                    ),
                    "marginal_visits": np.floor(rng.random(n_a) * 5.0),
                    "first_play": float(rng.standard_normal()),
                    "enemy_weights": weights,
                    "enemy_regrets": [rng.standard_normal(n_b) * 2.0 for _ in range(n_h)],
                    "enemy_priors": [np.abs(rng.standard_normal(n_b)) for _ in range(n_h)],
                    "visits": [np.floor(rng.random((n_a, n_b)) * 3.0) for _ in range(n_h)],
                    "q": [rng.standard_normal((n_a, n_b)) for _ in range(n_h)],
                    "a_idx": (n_a - 1) // 2,
                    "b_idx": (n_b - 1) // 2,
                    "leaf_value": float(rng.standard_normal()),
                }
            )
    return cases


def _matrix_expectation(case: dict) -> list:
    from matrix import (
        aggregate_self_utilities,
        apply_joint_backup,
        effective_q,
        enemy_widening_limit,
        exploration_epsilon,
        matrix_utilities,
        mixed_strategy,
        normalize_average_strategy,
        regret_plus_update,
        select_root_action,
        self_widening_limit,
    )

    n = case["n"]
    out: list = [self_widening_limit(n), enemy_widening_limit(n), exploration_epsilon(n)]
    sigma_self = mixed_strategy(case["regrets"], case["prior"], n)
    out.append(sigma_self)
    enemy_sigmas, q_effs = [], []
    for h in range(case["n_h"]):
        sigma_b = mixed_strategy(case["enemy_regrets"][h], case["enemy_priors"][h], n)
        q_eff = effective_q(case["visits"][h], case["q"][h], case["first_play"])
        out += [sigma_b, q_eff.ravel()]
        enemy_sigmas.append(sigma_b)
        q_effs.append(q_eff)
    u_self, v = aggregate_self_utilities(
        sigma_self, case["enemy_weights"], enemy_sigmas, q_effs
    )
    out += [u_self, v]
    u_h, u_enemy, v_h = matrix_utilities(sigma_self, enemy_sigmas[0], q_effs[0])
    out += [u_h, u_enemy, v_h]
    out.append(regret_plus_update(case["regrets"], u_self, v, maximizing=True))
    out.append(regret_plus_update(case["enemy_regrets"][0], u_enemy, v, maximizing=False))
    visits, value_sum, q = apply_joint_backup(
        visits=case["visits"][0],
        value_sum=np.zeros_like(case["visits"][0]),
        q=case["q"][0],
        a_idx=case["a_idx"],
        b_idx=case["b_idx"],
        leaf_value=case["leaf_value"],
    )
    out += [visits.ravel(), value_sum.ravel(), q.ravel()]
    out.append(normalize_average_strategy(case["avg_strategy"]))
    out.append(
        select_root_action(case["avg_strategy"], case["marginal_visits"], case["prior"])
    )
    return out


def _matrix_stream(case: dict) -> list[int]:
    out = [case["n_a"], case["n_b"], case["n_h"], case["n"]]
    out += _f64_bits(case["regrets"])
    out += _f64_bits(case["prior"])
    out += _f64_bits(case["avg_strategy"])
    out += _f64_bits(case["marginal_visits"])
    out += _f64_bits([case["first_play"]])
    out += _f64_bits(case["enemy_weights"])
    for h in range(case["n_h"]):
        out += _f64_bits(case["enemy_regrets"][h])
        out += _f64_bits(case["enemy_priors"][h])
        out += _f64_bits(np.asarray(case["visits"][h]).ravel())
        out += _f64_bits(np.asarray(case["q"][h]).ravel())
    out += [case["a_idx"], case["b_idx"]]
    out += _f64_bits([case["leaf_value"]])
    return out


def _tactical_pairs(frame, synthetic: bool):
    """`(obs, memory, frame)` for the tactical surfaces.

    Real frames carry their own belief and history. The synthetic block passes
    `None` for the frame, which reduces to an empty belief and a pass history —
    those positions exist to reach turn numbers and board shapes the corpus does
    not (the garrison-floor window, the deathtouch regime, an affordable build).
    """
    pairs = []
    if frame is None:
        if not synthetic:
            return pairs
        from observe import emit_observation

        for state in synthetic_states():
            for seat in (0, 1):
                obs = emit_observation(state, seat, as_arrays=True)
                pairs.append((obs, memory_for(state, seat), None))
        for obs, memory in _crafted_memory_pairs():
            pairs.append((obs, memory, None))
        pairs.extend(tactical_positions())
    elif "memory" in frame:
        pairs.append((_obs_from_capture(frame["obs"]), _memory_from_capture(frame["memory"]), frame))
    return pairs


DIRECTIONS_PY = ((-1, 0), (1, 0), (0, -1), (0, 1))


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
        elif kind in ("memory", "hash"):
            from hashing import (
                child_edge_key,
                enemy_info_hash_prehashed,
                info_state_key_prehashed,
                memory_digest,
                roll_history_digest,
            )
            from memory import update_memory
            from observe import observation_hash

            for obs, memory, prev_digest, action in _memory_pairs(frame, synthetic):
                stream += encode_observation(obs) + encode_memory(memory)
                if kind == "memory":
                    # Memory *before* the fold goes in; the fold is what is
                    # being checked, so feeding the post-fold memory would
                    # test nothing.
                    expected.append(update_memory(memory, obs))
                else:
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
        elif kind in ("propose", "filter", "rejuvenate", "maxent"):
            from belief import filter_step
            from memory import update_memory
            from observe import emit_observation
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
                if kind == "propose":
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
        elif kind in (
            "playmask", "shaping", "candidates", "planners", "constrain", "decide"
        ):
            from action import N_ACTIONS
            from tactics import (
                apply_pre_contact_prior,
                army_concentration,
                believed_enemy_general,
                blend_prior,
                blocks_oscillation,
                castle_build_site,
                castle_tithe_move,
                constrain_nn_action,
                defend_general_move,
                enemy_is_visible,
                enemy_seek_target,
                garrison_floor,
                general_threat,
                heuristic_action_scores,
                kill_plan,
                king_cell,
                known_enemy_general_cell,
                mandatory_action_indices,
                max_threat_arrival,
                own_general_cell,
                path_distance_field,
                play_mask,
                policy_ordered_candidates,
                reveal_count_grid,
                seek_goals,
                structure_idle_army,
                wave_assembly_cell,
            )

            for obs, memory, source in _tactical_pairs(frame, synthetic):
                mask = np.asarray(play_mask(obs, memory), dtype=bool)
                if kind == "playmask":
                    from action import legal_mask as _legal

                    base = np.asarray(_legal(obs, memory), dtype=bool)
                    owners = np.asarray(obs.owner_grid, dtype=np.int32)
                    armies = np.asarray(obs.army_grid, dtype=np.int32)
                    own_total = int(armies[owners == 1].sum())
                    stream += encode_observation(obs) + encode_memory(memory)
                    expected.append(
                        (
                            mask,
                            int(np.sum(base & ~mask)),
                            int(enemy_is_visible(obs, memory)),
                            int(garrison_floor(own_total)),
                            int(max_threat_arrival(obs, memory)),
                        )
                    )
                elif kind == "shaping":
                    belief = _lite_belief_object(source)
                    prev, _recent = _history_of(source)
                    prior = _network_prior_for(source, obs, memory)
                    lam, _lam_post, clip, floor = _shaping_knobs(source)
                    stream += encode_observation(obs) + encode_memory(memory)
                    stream += _lite_belief_ints(source) + _history_ints(source)
                    stream += _f64_bits(prior)
                    stream += _f64_bits([lam, clip, floor])
                    scores = heuristic_action_scores(
                        obs, memory, mask, belief, prev_action=prev
                    )
                    expected.append(
                        (
                            np.asarray(scores, dtype=np.float64),
                            np.asarray(
                                blend_prior(
                                    prior,
                                    scores,
                                    mask,
                                    lam=lam,
                                    log_clip=clip,
                                    floor_frac=floor,
                                ),
                                dtype=np.float64,
                            ),
                        )
                    )
                elif kind == "candidates":
                    from action import legal_mask as _legal

                    prior = _network_prior_for(source, obs, memory)
                    for limit, use_play in ((1, 1), (8, 1), (16, 1), (12, 0)):
                        active = mask if use_play else np.asarray(_legal(obs, memory), dtype=bool)
                        stream += encode_observation(obs) + encode_memory(memory)
                        stream += _f64_bits(prior) + [limit, use_play]
                        mandatory = mandatory_action_indices(obs, memory, mask=active)
                        expected.append(
                            (
                                [int(i) for i in mandatory],
                                [
                                    int(i)
                                    for i in policy_ordered_candidates(
                                        prior, active, mandatory=mandatory, limit=limit
                                    )
                                ],
                            )
                        )
                elif kind == "planners":
                    belief = _lite_belief_object(source)
                    stream += encode_observation(obs) + encode_memory(memory)
                    stream += _lite_belief_ints(source)
                    goals = seek_goals(obs, memory, belief)
                    exclude = own_general_cell(obs, memory)
                    share, max_own, total = army_concentration(obs, exclude=exclude)
                    site = castle_build_site(obs, memory)
                    threat = general_threat(obs, memory)
                    expected.append(
                        {
                            "own_general": own_general_cell(obs, memory),
                            "enemy_general": known_enemy_general_cell(obs, memory),
                            "king": king_cell(obs, exclude=None),
                            "believed": believed_enemy_general(belief),
                            "seek_target": enemy_seek_target(obs, memory, belief),
                            "goals": goals,
                            "assembly": wave_assembly_cell(
                                obs, memory, belief, exclude=exclude
                            ),
                            "share": float(share),
                            "max_own": int(max_own),
                            "total": int(total),
                            "idle": int(structure_idle_army(obs, memory)),
                            "threat_arrival": int(max_threat_arrival(obs, memory)),
                            "site": site,
                            "tithe": castle_tithe_move(obs, memory, site) if site else None,
                            "threat": threat,
                            "defend": defend_general_move(obs, memory, threat)
                            if threat
                            else None,
                            "kill": kill_plan(obs, memory),
                            "reveal": np.asarray(reveal_count_grid(obs), dtype=np.int64),
                            "field": np.asarray(
                                path_distance_field(obs, goals), dtype=np.int64
                            ),
                        }
                    )
                elif kind == "constrain":
                    prev, recent = _history_of(source)
                    prior = _network_prior_for(source, obs, memory)
                    shaped = np.asarray(
                        apply_pre_contact_prior(
                            prior,
                            obs,
                            memory,
                            mask=mask,
                            belief=_lite_belief_object(source),
                            lam=_shaping_knobs(source)[0],
                            log_clip=_shaping_knobs(source)[2],
                            floor_frac=_shaping_knobs(source)[3],
                            prev_action=prev,
                        ),
                        dtype=np.float64,
                    )
                    played = tuple(int(v) for v in (source or {}).get("action", PASS))
                    reveal = np.asarray(reveal_count_grid(obs), dtype=np.int64)
                    # Three chosen actions per position: what the bot played,
                    # a pass (the no-pass hard rule), and the reverse of the
                    # previous move (the oscillation rule). Without the last
                    # two, most corpus frames never reach either branch.
                    reversed_prev = (
                        (0, prev[1] + int(DIRECTIONS_PY[prev[3]][0]),
                         prev[2] + int(DIRECTIONS_PY[prev[3]][1]),
                         (prev[3] ^ 1), 0)
                        if prev[0] == 0
                        else PASS
                    )
                    for chosen, with_prior in (
                        (played, True), (PASS, True), (reversed_prev, True), (played, False),
                    ):
                        stream += encode_observation(obs) + encode_memory(memory)
                        stream += [int(v) for v in chosen]
                        stream += _history_ints(source)
                        stream += [int(with_prior)] + _f64_bits(shaped)
                        expected.append(
                            (
                                tuple(
                                    int(v)
                                    for v in constrain_nn_action(
                                        obs,
                                        memory,
                                        chosen,
                                        prev_action=prev,
                                        recent_actions=tuple(recent),
                                        prior=shaped if with_prior else None,
                                    )
                                ),
                                int(
                                    blocks_oscillation(
                                        chosen, prev, obs, recent_actions=tuple(recent)
                                    )
                                ),
                            )
                        )
        elif kind == "search":
            for obs, memory, belief, cfg, batches, freeze, vary in _search_setups(
                frame, synthetic
            ):
                prior = _search_prior_for(frame, obs, memory)
                case = _search_expectation(
                    obs, memory, belief, prior, 0.25, cfg, batches, freeze, vary,
                    90210 + len(expected),
                )
                expected.append(case)
                stream += encode_observation(obs) + encode_memory(memory)
                stream += encode_belief_state(belief)
                stream += _f64_bits(prior) + _f64_bits([0.25])
                stream += [
                    cfg["depth"], cfg["pending_batch"], cfg["n_particles"],
                    cfg["max_nodes"], cfg["max_enemy_tables"], batches,
                    int(freeze), int(vary),
                ]
                stream += encode_draws(case["draws"])
        elif kind == "runtime":
            if frame is not None:
                continue  # the inputs are synthetic; one pass is enough
            from runtime import highest_prior_legal, nearest_rank_p99

            for samples, prior, mask in runtime_cases():
                stream += [len(samples)] + _f64_bits(samples)
                stream += _f64_bits(prior)
                stream += [int(bool(v)) for v in mask]
                try:
                    p99 = (1, float(nearest_rank_p99(samples)))
                except ValueError:
                    p99 = (0, 0.0)
                expected.append(
                    (p99, tuple(int(v) for v in highest_prior_legal(prior, mask)))
                )
        elif kind == "evict":
            if frame is not None:
                continue  # the table sets are synthetic; one pass is enough
            from tree import InfoNode, SearchTree

            for tables, node_n, max_tables, arriving in evict_cases():
                stream += [len(tables), node_n, max_tables]
                for seed, last_used, touch, pinned in tables:
                    stream += [seed, last_used, touch, int(pinned)]
                stream += [arriving]

                tree = SearchTree(max_nodes=1024, max_enemy_tables=max_tables, seat=0)
                node = InfoNode(
                    key=bytes(32), turn=0, memory_digest=bytes(32),
                    obs_hash=bytes(32), history_digest=bytes(32),
                )
                node.reservoir.capacity = 1
                tree.nodes[node.key] = node
                tree.root = node
                pins = []
                for seed, last_used, touch, pinned in tables:
                    h = bytes([seed]) * 32
                    table = tree.get_or_create_enemy_table(node, h, [1], np.array([1.0]))
                    table.last_used = int(last_used)
                    table.touch_count = int(touch)
                    if pinned:
                        pins.append(h)
                for h in pins:
                    tree.pin_enemy(node, h)
                node.N = int(node_n)
                tree.get_or_create_enemy_table(
                    node, bytes([arriving]) * 32, [1], np.array([1.0])
                )
                expected.append(
                    [
                        (int(t.info_hash[0]), int(t.last_used), int(t.touch_count))
                        for t in node.enemy_tables.values()
                    ]
                )
        elif kind == "matrix":
            if frame is not None:
                continue  # the shapes are synthetic; one pass is enough
            for case in matrix_cases():
                stream += _matrix_stream(case)
                expected.append(_matrix_expectation(case))
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
    for key in _BLAS_FLIPS:
        _BLAS_FLIPS[key] = 0


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


def _first_diff(got: list, want: list):
    for k, (a, b) in enumerate(zip(got, want)):
        if a != b:
            return f"index {k} ({a} != {b})"
    return f"index {min(len(got), len(want))} (length)"


def _compare_planners(i: int, want: dict, got: list[int]) -> list[str]:
    """Every planner answer, named, so a failure says which one moved."""
    problems: list[str] = []
    at = 0

    def cell():
        nonlocal at
        present, r, c = got[at], got[at + 1], got[at + 2]
        at += 3
        return (int(r), int(c)) if present else None

    def action():
        nonlocal at
        present = got[at]
        value = tuple(int(v) for v in got[at + 1 : at + 6])
        at += 6
        return value if present else None

    for name in ("own_general", "enemy_general", "king", "believed", "seek_target"):
        rust = cell()
        if rust != want[name]:
            problems.append(f"planners[{i}]: {name} {rust} != {want[name]}")
    n_goals = got[at]; at += 1
    rust_goals = [(int(got[at + 2 * k]), int(got[at + 2 * k + 1])) for k in range(n_goals)]
    at += 2 * n_goals
    if rust_goals != list(want["goals"]):
        problems.append(
            f"planners[{i}]: seek_goals {len(rust_goals)} vs {len(want['goals'])}; "
            f"first difference at {_first_diff(rust_goals, list(want['goals']))}"
        )
    rust = cell()
    if rust != want["assembly"]:
        problems.append(f"planners[{i}]: wave_assembly_cell {rust} != {want['assembly']}")
    share = float(_decode_f64(got[at : at + 1])[0]); at += 1
    if share != want["share"]:
        problems.append(f"planners[{i}]: army share {share!r} != {want['share']!r}")
    for name in ("max_own", "total", "idle", "threat_arrival"):
        if int(got[at]) != int(want[name]):
            problems.append(f"planners[{i}]: {name} {got[at]} != {want[name]}")
        at += 1

    rust = cell()
    if rust != want["site"]:
        problems.append(f"planners[{i}]: castle_build_site {rust} != {want['site']}")
    rust = action()
    if rust != (tuple(want["tithe"]) if want["tithe"] else None):
        problems.append(f"planners[{i}]: castle_tithe_move {rust} != {want['tithe']}")

    present, tr, tc, td = got[at], got[at + 1], got[at + 2], got[at + 3]
    at += 4
    rust_threat = ((int(tr), int(tc)), int(td)) if present else None
    want_threat = (
        (tuple(want["threat"][0]), int(want["threat"][1])) if want["threat"] else None
    )
    if rust_threat != want_threat:
        problems.append(f"planners[{i}]: general_threat {rust_threat} != {want_threat}")
    rust = action()
    if rust != (tuple(want["defend"]) if want["defend"] else None):
        problems.append(f"planners[{i}]: defend_general_move {rust} != {want['defend']}")

    present, steps, margin = got[at], got[at + 1], got[at + 2]
    at += 3
    rust_kill = (int(steps), int(margin)) if present else None
    first = action()
    want_kill = (int(want["kill"][0]), int(want["kill"][1])) if want["kill"] else None
    want_first = tuple(want["kill"][2]) if want["kill"] else None
    if rust_kill != want_kill or first != want_first:
        problems.append(
            f"planners[{i}]: kill_plan {rust_kill}/{first} != {want_kill}/{want_first}"
        )

    reveal = want["reveal"].ravel()
    got_reveal = np.asarray(got[at : at + reveal.size], dtype=np.int64)
    at += reveal.size
    if not np.array_equal(got_reveal, reveal):
        bad = int(np.argmax(got_reveal != reveal))
        problems.append(
            f"planners[{i}]: reveal_count_grid differs first at cell {bad} "
            f"({got_reveal[bad]} != {reveal[bad]})"
        )
    field = want["field"].ravel()
    got_field = np.asarray(got[at : at + field.size], dtype=np.int64)
    if not np.array_equal(got_field, field):
        bad = int(np.argmax(got_field != field))
        problems.append(
            f"planners[{i}]: path_distance_field differs first at cell {bad} "
            f"({got_field[bad]} != {field[bad]})"
        )
    return problems


def _compare_matrix(i: int, want: list, got: list[int]) -> list[str]:
    """The regret cycle, under the one tolerance the oracle's BLAS forces."""
    problems: list[str] = []
    at = 0
    names = [
        "self_widening_limit", "enemy_widening_limit", "exploration_epsilon",
        "sigma_self",
    ]
    for k, value in enumerate(want):
        if isinstance(value, (int, np.integer)) and k < 2:
            if int(got[at]) != int(value):
                problems.append(f"matrix[{i}]: {names[k]} {got[at]} != {value}")
            at += 1
        elif isinstance(value, (int, np.integer)):
            # the trailing `select_root_action` index
            if int(got[at]) != int(value):
                problems.append(
                    f"matrix[{i}]: select_root_action {got[at]} != {value}"
                )
            at += 1
        elif np.isscalar(value) or isinstance(value, float):
            rust = float(_decode_f64(got[at : at + 1])[0])
            at += 1
            delta = abs(rust - float(value))
            _note("matrix.scalar.max", delta)
            if delta > MATRIX_TOLERANCE:
                problems.append(
                    f"matrix[{i}]: scalar at {k} {rust!r} != {value!r} (|d| {delta:.3g})"
                )
        else:
            arr = np.asarray(value, dtype=np.float64).ravel()
            rust = _decode_f64(got[at : at + arr.size])
            at += arr.size
            delta = np.abs(rust - arr)
            _note("matrix.vector.max", float(delta.max()) if delta.size else 0.0)
            if delta.size and delta.max() > MATRIX_TOLERANCE:
                bad = int(np.argmax(delta))
                problems.append(
                    f"matrix[{i}]: vector at {k} differs at {bad} "
                    f"({rust[bad]!r} != {arr[bad]!r}, |d| {delta.max():.3g})"
                )
    return problems


# How much regret has to be present before a regret-matching branch counts as
# a real disagreement rather than the zero test resolving differently. Set from
# measurement: the flips this harness sees decide on 2.8e-17, half an ulp of the
# first-play value they are derived from.
REGRET_TIE_EPS = 1e-12


class _PairwiseDots:
    """Run the oracle's search with `@` replaced by NumPy's pairwise reduction.

    Not a convenience. `matrix.py` writes `q_eff @ sigma_enemy`, NumPy sends
    that to BLAS, and BLAS's reduction order is the vendor's — so the *oracle*
    is host-dependent there (see `matrix.rs`). On its own that is a 1-ulp
    difference the `matrix` surface measures and tolerates.

    Regret matching plus then amplifies it into a different strategy. Its first
    branch is `sum(max(regret, 0)) <= 0`, and on the first backup of a fresh
    enemy table the true regret is exactly zero, so whether the accumulated
    float lands on 0.0 or on 2.8e-17 decides between "fall back to the prior"
    and "spread uniformly over the positive entries". One ulp becomes a
    qualitatively different mixed strategy, deterministically.

    Running the oracle a second time with the one substitution the port makes
    turns that from an unexplained mismatch into a measured one: the port must
    match *this* oracle to the last bit, and the gap between the two oracles is
    reported as what it is — the search's sensitivity to its own BLAS.
    """

    def __enter__(self):
        import matrix
        import tree

        self.modules = (matrix, tree)
        self.saved = [
            (m, name, getattr(m, name))
            for m in self.modules
            for name in ("matrix_utilities", "aggregate_self_utilities")
            if hasattr(m, name)
        ]
        for m in self.modules:
            if hasattr(m, "matrix_utilities"):
                m.matrix_utilities = _pairwise_matrix_utilities
            if hasattr(m, "aggregate_self_utilities"):
                m.aggregate_self_utilities = _pairwise_aggregate
        return self

    def __exit__(self, *exc):
        for module, name, value in self.saved:
            setattr(module, name, value)
        return False


def _np_pairwise(values) -> float:
    """`np.sum`'s reduction, which is what `rng::npsum` reproduces in Rust."""
    return float(np.add.reduce(np.asarray(values, dtype=np.float64).ravel()))


def _pairwise_matrix_utilities(sigma_self, sigma_enemy, q_eff):
    s = np.asarray(sigma_self, np.float64).ravel()
    b = np.asarray(sigma_enemy, np.float64).ravel()
    q = np.asarray(q_eff, np.float64)
    u_self = np.array([_np_pairwise(q[i] * b) for i in range(len(s))])
    u_enemy = np.array([_np_pairwise(s * q[:, j]) for j in range(len(b))])
    return u_self, u_enemy, _np_pairwise(s * u_self)


def _pairwise_aggregate(sigma_self, enemy_weights, enemy_sigmas, q_eff_list):
    s = np.asarray(sigma_self, np.float64).ravel()
    w = np.maximum(np.asarray(enemy_weights, np.float64).ravel(), 0.0)
    total = _np_pairwise(w)
    if total <= 0.0 and len(w):
        w = np.full(len(w), 1.0 / len(w))
    elif total > 0.0:
        w = w / total
    u_self = np.zeros(len(s), dtype=np.float64)
    for weight, sigma_b, q_eff in zip(w, enemy_sigmas, q_eff_list):
        if weight <= 0.0:
            continue
        u_h, _, _ = _pairwise_matrix_utilities(s, sigma_b, q_eff)
        u_self = u_self + weight * u_h
    return u_self, _np_pairwise(s * u_self)


def _varying_evaluator(*, prior, value):
    """A scripted prior whose *value* depends on the leaf. Mirrors `parity.rs`.

    `ScriptedEvaluator` hands every leaf the same number, and a constant leaf
    value makes the entire enemy mixture unobservable: every `q` entry ends up
    equal, so the weights they are averaged with cannot change an answer. That
    alone hid the enemy-hash cache key, the reservoir's weights and the marginal
    aggregation from every mutation M5 aimed at them.

    The value stays a deterministic function of the leaf's own observation
    payload, so nothing the scripted evaluator exists to exclude comes back —
    no network, no softmax, and the same double on both sides.
    """
    from observe import observation_hash
    from search import ScriptedEvaluator

    class _Varying(ScriptedEvaluator):
        def evaluate(self, obs, memory, belief, *, from_root, shape=False):
            got, _ = super().evaluate(
                obs, memory, belief, from_root=from_root, shape=shape
            )
            total = sum(observation_hash(obs))
            return got, (float(total % 2001) - 1000.0) / 1000.0

    return _Varying(prior=prior, value=value)


def _run_search(obs, memory, belief, prior, value, cfg, batches, freeze, vary, rng):
    """Build a tree from one frame and run it, with every draw recorded.

    `replace_from_belief` builds `np.random.default_rng(0)` inside itself, so
    the factory is patched for the whole run and those draws land in the same
    ordered log — the Rust side reads one stream in call order, and a separate
    log would silently permit a port that made them at a different point.
    """
    from search import ScriptedEvaluator, SearchConfig, SearchController

    controller = SearchController(
        seat=int(belief.seat),
        evaluator=(_varying_evaluator if vary else ScriptedEvaluator)(
            prior=np.asarray(prior, dtype=np.float64), value=value
        ),
        config=SearchConfig(
            depth=cfg["depth"],
            pending_batch=cfg["pending_batch"],
            max_nodes=cfg["max_nodes"],
            max_enemy_tables=cfg["max_enemy_tables"],
            n_particles=cfg["n_particles"],
        ),
        rng=rng,
    )
    from capture_morpheus import RecordingGenerator

    inner = RecordingGenerator(np.random.default_rng(0))
    inner.draws = rng.draws
    real = np.random.default_rng
    try:
        np.random.default_rng = lambda seed=None: inner
        controller.ensure_root(obs, memory, belief)
        for _ in range(batches):
            controller.run_batch(belief, freeze_widening=freeze)
    finally:
        np.random.default_rng = real

    tree = controller.tree
    root = tree.root

    def _tables_of(node):
        out = []
        for table in node.enemy_tables.values():
            out.append(
                {
                    "actions": [int(a) for a in table.actions],
                    "prior": np.asarray(table.prior, dtype=np.float64),
                    "regret": np.asarray(table.regret, dtype=np.float64),
                    "avg": np.asarray(table.avg_strategy, dtype=np.float64),
                    "n_self": int(table.visits.shape[0]),
                    "last_used": int(table.last_used),
                    "touch_count": int(table.touch_count),
                    "visits": np.asarray(table.visits, dtype=np.float64).ravel(),
                    "q": np.asarray(table.q, dtype=np.float64).ravel(),
                }
            )
        return out

    # Every node, in creation order. See the matching comment in `parity.rs`:
    # a leaf value never travels up through a child's own statistics and
    # `Replay` hides a changed distribution, so a root-only comparison cannot
    # see anything a child computes.
    detail = []
    for node in tree.nodes.values():
        detail.append(
            {
                "n": int(node.N),
                "turn": int(node.turn),
                "memory_digest": bytes(node.memory_digest),
                "reservoir": int(node.reservoir.n),
                "actions": [int(a) for a in node.actions],
                "prior": np.asarray(node.prior, dtype=np.float64),
                "regret": np.asarray(node.regret, dtype=np.float64),
                "avg": np.asarray(node.avg_strategy, dtype=np.float64),
                "tables": _tables_of(node),
            }
        )

    tables = []
    for table in root.enemy_tables.values():
        tables.append(
            {
                "actions": [int(a) for a in table.actions],
                "prior": np.asarray(table.prior, dtype=np.float64),
                "regret": np.asarray(table.regret, dtype=np.float64),
                "avg": np.asarray(table.avg_strategy, dtype=np.float64),
                "n_self": int(table.visits.shape[0]),
                "last_used": int(table.last_used),
                "touch_count": int(table.touch_count),
                "visits": np.asarray(table.visits, dtype=np.float64).ravel(),
                "q": np.asarray(table.q, dtype=np.float64).ravel(),
            }
        )
    try:
        index = int(tree.root_action_index())
    except ValueError:
        index = -1
    best = tuple(int(v) for v in controller.best_action()) if root.actions else None
    return {
        "completed": int(tree.completed_simulations),
        "nodes": int(len(tree.nodes)),
        "hits": int(tree.table_hits),
        "misses": int(tree.table_misses),
        "eviction_loss": float(tree.eviction_loss),
        "joint_visits": float(tree.total_joint_visits),
        "root_n": int(root.N),
        "actions": [int(a) for a in root.actions],
        "prior": np.asarray(root.prior, dtype=np.float64),
        "regret": np.asarray(root.regret, dtype=np.float64),
        "avg": np.asarray(root.avg_strategy, dtype=np.float64),
        "tables": tables,
        "marginal": np.asarray(controller.root_marginal_visits(), dtype=np.float64),
        "index": index,
        "best": best,
        "degraded": _degradation_tail(controller, int(tree.completed_simulations)),
        "detail": detail,
        "draws": len(rng.draws),
    }


def _search_expectation(obs, memory, belief, prior, value, cfg, batches, freeze, vary, seed):
    """Both oracles: the shipped one, and the one whose dots the port shares."""
    blas = _run_search(
        obs, memory, belief, prior, value, cfg, batches, freeze, vary, recording_rng(seed)
    )
    rng = recording_rng(seed)
    with _PairwiseDots():
        pairwise = _run_search(
            obs, memory, belief, prior, value, cfg, batches, freeze, vary, rng
        )
    return {"blas": blas, "pairwise": pairwise, "draws": rng.take()}


# The four `select_degraded_action` inputs the surface asks about. The third
# and fourth are the interesting ones: with a real root that says pass, the
# path has to prefer a non-pass policy fallback over emitting pass after a
# search — a branch no corpus frame reaches, because a root with candidates
# almost never picks pass.
_DEGRADATION_PROBES = (
    (0, False, PASS),
    (0, True, (0, 1, 1, 0, 0)),
    (None, True, (0, 1, 1, 0, 0)),
    (None, True, PASS),
)

_LEVEL_ORDER = ("pass", "policy", "visit", "average")


def _degradation_tail(controller, completed: int) -> list[tuple]:
    from runtime import select_degraded_action

    out = []
    for sims, has_root, fallback in _DEGRADATION_PROBES:
        action, level = select_degraded_action(
            completed_simulations=completed if sims is None else sims,
            has_root_result=has_root,
            policy_fallback=fallback,
            search=controller,
        )
        out.append(
            (tuple(int(v) for v in action), _LEVEL_ORDER.index(level.value))
        )
    return out


def evict_cases() -> list[tuple]:
    """`(tables, node_n, max_tables, arriving)` for the retention rule.

    Stated rather than played, for the reason the `runtime` surface exists. The
    retention score is `last_used + 0.25 * ln1p(touch_count)`, and `last_used`
    is an integer, so the touch term can only ever decide a tie. In a real tree
    there are no ties: `last_used` is the node's visit counter at the table's
    last backup, and a backup advances it, so every table at a node carries a
    different one. Measured over the search setups — 680 backups — the harness
    never saw two candidates share a `last_used`.

    Each table is `(seed, last_used, touch_count, pinned)`; the eviction runs
    when the arriving table pushes the node over `max_tables`.
    """
    return [
        # The touch term decides: equal recency, different use. The
        # *well-used* table is listed first on purpose — with the term deleted
        # the two scores tie and the first minimum wins, which is the same
        # answer the term gives in the other order. The order is the test.
        ([(1, 5, 9, 0), (2, 5, 1, 0)], 5, 2, 3),
        # An exact tie in both. `min` keeps the *first*, so the order of the
        # table list is the answer, and `<=` instead of `<` would take the last.
        ([(1, 5, 3, 0), (2, 5, 3, 0)], 5, 2, 3),
        # The lowest score is pinned; the next one goes instead.
        ([(1, 0, 1, 1), (2, 7, 1, 0)], 7, 2, 3),
        # The lowest score is the table that just arrived.
        ([(1, 9, 4, 0), (2, 9, 6, 0)], 0, 2, 3),
        # Two over capacity: the loop evicts twice, lowest first.
        ([(1, 1, 1, 0), (2, 2, 1, 0), (3, 3, 1, 0)], 4, 2, 4),
        # Everything pinned: eviction refuses rather than breaking a pin.
        ([(1, 1, 1, 1), (2, 2, 1, 1)], 3, 1, 3),
        # Nothing to do — one table, one slot.
        ([(1, 4, 2, 0)], 4, 4, 2),
    ]


def runtime_cases() -> list[tuple]:
    """Sample vectors for the two controller functions with no state."""
    from action import N_ACTIONS

    rng = np.random.default_rng(4242)
    cases = []
    for n in (0, 1, 2, 99, 100, 101, 200):
        samples = np.round(rng.random(n) * 50.0, 6) if n else np.zeros(0)
        prior = np.zeros(N_ACTIONS, dtype=np.float64)
        mask = np.zeros(N_ACTIONS, dtype=bool)
        if n:
            # The illegal action carries the largest mass, so a fallback that
            # ignored the mask would pick it. A legal-normalized prior can
            # never look like this, which is why nothing else reaches the mask.
            live = [(n * 13) % (N_ACTIONS - 1), (n * 29) % (N_ACTIONS - 1)]
            mask[live] = True
            prior[live] = [0.1, 0.2]
            prior[(n * 7) % (N_ACTIONS - 1)] = 0.9
        cases.append((samples, prior, mask))
    return cases


def _search_setups(frame, synthetic: bool):
    """`(obs, memory, belief, config, batches, freeze)` per search case.

    Recorded frames run one small, *odd* configuration — depth 3, batches of 2,
    three of them — which crosses the terminal, unexpanded-node and depth-limit
    exits and makes the enemy-prior materialisation run more than once with
    tables already installed.

    The synthetic block runs the configurations a recorded frame cannot
    produce, and it exists because the mutation pass said so: on the smoke
    slice, breaking eviction, the node cap, child reuse, the terminal value or
    the widening freeze changed nothing anywhere. Two enemy tables against four
    particles forces eviction; twenty-four nodes forces the cap's refusal;
    `freeze` is a flag the corpus never sets because the capture host was never
    that far behind its deadline.
    """
    from belief import BeliefConfig, BeliefState, Particle
    from memory import empty_memory
    from observe import emit_observation

    if frame is not None:
        if "belief" not in frame or "memory" not in frame:
            return
        belief = belief_from_frame(frame)
        if belief is None or belief.n == 0:
            return
        yield (
            _obs_from_capture(frame["obs"]),
            _memory_from_capture(frame["memory"]),
            belief,
            {"depth": 3, "pending_batch": 2, "max_nodes": 64,
             "max_enemy_tables": 3,
             "n_particles": int(belief.config.n_particles)},
            3,
            False,
            False,
        )
        return
    if not synthetic:
        return

    for base in synthetic_states():
        h, w = base.armies.shape
        seat = 0
        # Four particles that disagree about the enemy's armies, so their enemy
        # information hashes differ and more than one table is installed —
        # which is the only way eviction and the LRU ever get a decision.
        #
        # M5 wrote this and it did not work. `_replace()` on a NamedTuple copies
        # the *tuple*, not the arrays inside it, so all four "particles" shared
        # one `armies` and the last write won on every one of them: four
        # identical states, one enemy hash, one table, and eviction never
        # reached. That is why every eviction and LRU mutation survived M5 while
        # the comment above claimed otherwise. `_copy_state` is the fix, and the
        # `search` case now asserts the four hashes really do differ.
        particles = []
        for k, weight in enumerate((0.4, 0.3, 0.2, 0.1)):
            state = _copy_state(base)
            enemy_cells = np.argwhere(np.asarray(state.ownership[1]))
            for j, (r, c) in enumerate(enemy_cells):
                state.armies[r, c] = int(base.armies[r, c]) + k * (j + 1)
            particles.append(
                Particle(state=state, weight=weight,
                         enemy_memory=empty_memory(h, w),
                         enemy_prev_action=None, history=())
            )
        belief = BeliefState(
            seat=seat, particles=particles,
            config=BeliefConfig(n_particles=4), collapsed=False,
        )
        obs = emit_observation(base, seat, as_arrays=True)
        memory = memory_for(base, seat)
        for batches, freeze, cap in ((4, False, 24), (2, True, 4096)):
            yield (
                obs, memory, belief,
                {"depth": 6, "pending_batch": 4, "max_nodes": cap,
                 "max_enemy_tables": 2, "n_particles": 4},
                batches,
                freeze,
                False,
            )

    # One *contended* tree, on one board rather than every board, because it is
    # the most expensive case the harness runs: eight distinct enemy views
    # against a two-table cap, eight selections per batch so a whole batch's
    # pins are live at once, and four batches so tables from an earlier batch
    # are still resident when a later one needs the room.
    #
    # Measured on this configuration: seventeen over-capacity evictions, ten of
    # them where the *protected* table carried the lowest retention score and
    # four where a *pinned* one did. Those are the two decisions M5's four
    # particles could never reach — see `_copy_state` above for why they had
    # only one enemy hash between them.
    base = synthetic_states()[0]
    h, w = base.armies.shape
    particles = []
    for k in range(8):
        state = _copy_state(base)
        for j, (r, c) in enumerate(np.argwhere(np.asarray(state.ownership[1]))):
            state.armies[r, c] = int(base.armies[r, c]) + (k + 1) * (j + 3)
        particles.append(
            Particle(state=state, weight=1.0 / 8.0,
                     enemy_memory=empty_memory(h, w),
                     enemy_prev_action=None, history=())
        )
    belief = BeliefState(
        seat=0, particles=particles,
        config=BeliefConfig(n_particles=8), collapsed=False,
    )
    yield (
        emit_observation(base, 0, as_arrays=True),
        memory_for(base, 0),
        belief,
        {"depth": 6, "pending_batch": 8, "max_nodes": 128,
         "max_enemy_tables": 2, "n_particles": 8},
        4,
        False,
        True,
    )

    # And one *fogged* tree, which is a different kind of contention. Every
    # board above is small enough to see whole, so two particles that disagree
    # about the enemy produce two different observations of ours, two different
    # child edges, and two nodes that never meet. Here the generals sit in
    # opposite corners of a 7x7 with nothing else on it: our seat sees a 3x3
    # box, every enemy cell is fogged, and all eight particles therefore walk
    # into the *same* child carrying eight different enemy views.
    #
    # That is what reaches a per-node enemy-hash cache going stale — the cache
    # is keyed on the reservoir version, and only a node whose reservoir grows
    # between two backups while holding more than one enemy hash can tell the
    # key from its absence. The memory fold needs the fog for the same reason:
    # on a board we can already see whole, folding a child's observation into
    # the root's memory adds nothing.
    from memory import empty_memory as _empty_memory, update_memory

    fogged = _blank(7, 7)._replace(time=100)
    fogged.general_positions[:] = np.array([[0, 0], [6, 6]], np.int32)
    for cell, seat_i in (((0, 0), 0), ((6, 6), 1)):
        fogged.generals[cell] = True
        fogged.ownership[seat_i][cell] = True
        fogged.armies[cell] = 8
    fogged.ownership_neutral[:] = ~(fogged.ownership[0] | fogged.ownership[1])
    particles = []
    for k in range(8):
        state = _copy_state(fogged)
        for j, (r, c) in enumerate(np.argwhere(np.asarray(state.ownership[1]))):
            state.armies[r, c] = int(fogged.armies[r, c]) + (k + 1) * (j + 3)
        particles.append(
            Particle(state=state, weight=1.0 / 8.0,
                     enemy_memory=_empty_memory(7, 7),
                     enemy_prev_action=None, history=())
        )
    obs = emit_observation(fogged, 0, as_arrays=True)
    yield (
        obs,
        # The memory a seat actually has on its first turn: nothing but this
        # observation. `memory_for` knows the whole board, which is what made
        # the fold unobservable everywhere else.
        update_memory(_empty_memory(7, 7), obs),
        BeliefState(seat=0, particles=particles,
                    config=BeliefConfig(n_particles=8), collapsed=False),
        {"depth": 6, "pending_batch": 8, "max_nodes": 128,
         "max_enemy_tables": 4, "n_particles": 8},
        6,
        False,
        True,
    )


def _search_prior_for(frame, obs, memory):
    """A scripted prior that is neither uniform nor the network's.

    Uniform would make every candidate ordering a tie and hide the ranking; the
    network's own prior would make the search comparison depend on the 6.6e-7 the
    `prior` surface already measures. A deterministic irregular ramp does
    neither.
    """
    from action import N_ACTIONS, legal_mask

    del frame
    mask = np.asarray(legal_mask(obs, memory), dtype=bool)
    prior = np.zeros(N_ACTIONS, dtype=np.float64)
    live = np.flatnonzero(mask)
    if live.size:
        ramp = 1.0 + ((live * 7919) % 23).astype(np.float64)
        prior[live] = ramp / ramp.sum()
    return prior


# How often, and by how much, the shipped oracle's BLAS dot products change a
# search statistic. Not a port defect: reported so the number exists.
_BLAS_FLIPS: dict[str, int] = {"vectors": 0, "regret_branch": 0, "cases": 0}


def blas_amplification() -> dict[str, int]:
    return dict(_BLAS_FLIPS)


def _tally_blas_amplification(blas: dict, pairwise: dict) -> None:
    _BLAS_FLIPS["cases"] += 1
    pairs = [(blas[k], pairwise[k]) for k in ("prior", "regret", "avg", "marginal")]
    for tb, tp in zip(blas["tables"], pairwise["tables"]):
        pairs += [(tb[k], tp[k]) for k in ("prior", "regret", "avg", "visits", "q")]
    for a, b in pairs:
        a = np.asarray(a, dtype=np.float64).ravel()
        b = np.asarray(b, dtype=np.float64).ravel()
        if a.size != b.size or not np.array_equal(a, b):
            _BLAS_FLIPS["vectors"] += 1
            _note("search.blas_vs_pairwise.max",
                  float(np.abs(a[: b.size] - b[: a.size]).max()) if a.size == b.size else 1.0)
    for tb, tp in zip(blas["tables"], pairwise["tables"]):
        pos_b = float(np.maximum(np.asarray(tb["regret"]), 0.0).sum())
        pos_p = float(np.maximum(np.asarray(tp["regret"]), 0.0).sum())
        if (pos_b <= 0.0) != (pos_p <= 0.0):
            _BLAS_FLIPS["regret_branch"] += 1
            _note("search.regret_branch_decided_on",
                  max(abs(pos_b), abs(pos_p)))


def _compare_search(i: int, case: dict, got: list[int]) -> list[str]:
    """The tree after N batches, statistic by statistic.

    Strict against the pairwise-dot oracle — the port must match it to the last
    bit. The shipped BLAS oracle is compared against that one separately, and
    every place the two differ is tallied as what it is: the search's own
    sensitivity to whichever BLAS NumPy was built against. See `_PairwiseDots`.
    """
    _tally_blas_amplification(case["blas"], case["pairwise"])
    want = case["pairwise"]
    problems: list[str] = []
    at = 0

    def ints(count):
        nonlocal at
        chunk = got[at : at + count]
        at += count
        return chunk

    def floats(count):
        nonlocal at
        chunk = _decode_f64(got[at : at + count])
        at += count
        return chunk

    for name in ("completed", "nodes", "hits", "misses"):
        value = ints(1)[0]
        if int(value) != int(want[name]):
            problems.append(f"search[{i}]: {name} {value} != {want[name]}")
    loss, joint = floats(2)
    if loss != want["eviction_loss"] or joint != want["joint_visits"]:
        problems.append(
            f"search[{i}]: eviction {loss!r}/{joint!r} != "
            f"{want['eviction_loss']!r}/{want['joint_visits']!r}"
        )
    root_n = ints(1)[0]
    if int(root_n) != want["root_n"]:
        problems.append(f"search[{i}]: root N {root_n} != {want['root_n']}")
    n_actions = ints(1)[0]
    actions = [int(v) for v in ints(n_actions)]
    if actions != want["actions"]:
        problems.append(
            f"search[{i}]: root candidates {len(actions)} vs {len(want['actions'])}; "
            f"first difference at {_first_diff(actions, want['actions'])}"
        )
    for name in ("prior", "regret", "avg"):
        arr = want[name]
        rust = floats(arr.size)
        problems += _float_difference(f"search[{i}]: root {name}", arr, rust)
    n_tables = ints(1)[0]
    if int(n_tables) != len(want["tables"]):
        return problems + [
            f"search[{i}]: {n_tables} enemy table(s) != {len(want['tables'])}"
        ]
    for k in range(int(n_tables)):
        table = want["tables"][k]
        n_enemy = ints(1)[0]
        table_actions = [int(v) for v in ints(n_enemy)]
        if table_actions != table["actions"]:
            problems.append(
                f"search[{i}]: table {k} columns {len(table_actions)} vs "
                f"{len(table['actions'])}; first difference at "
                f"{_first_diff(table_actions, table['actions'])}"
            )
        for name in ("prior", "regret", "avg"):
            problems += _float_difference(
                f"search[{i}]: table {k} {name}", table[name], floats(table[name].size)
            )
        n_self, last_used, touch = ints(3)
        if (int(n_self), int(last_used), int(touch)) != (
            table["n_self"], table["last_used"], table["touch_count"],
        ):
            problems.append(
                f"search[{i}]: table {k} shape/LRU ({n_self}, {last_used}, {touch}) "
                f"!= ({table['n_self']}, {table['last_used']}, {table['touch_count']})"
            )
        for name in ("visits", "q"):
            problems += _float_difference(
                f"search[{i}]: table {k} {name}", table[name], floats(table[name].size)
            )
    problems += _float_difference(
        f"search[{i}]: marginal visits", want["marginal"], floats(want["marginal"].size)
    )
    index = ints(1)[0]
    if int(index) != want["index"]:
        problems.append(f"search[{i}]: root action index {index} != {want['index']}")
    present = ints(1)[0]
    best = tuple(int(v) for v in ints(5))
    rust_best = best if present else None
    if rust_best != want["best"]:
        problems.append(f"search[{i}]: best action {rust_best} != {want['best']}")
    for k, (want_action, want_level) in enumerate(want["degraded"]):
        action = tuple(int(v) for v in ints(5))
        level = int(ints(1)[0])
        if (action, level) != (want_action, want_level):
            problems.append(
                f"search[{i}]: degradation probe {k} "
                f"{action}/{_LEVEL_ORDER[level]} != "
                f"{want_action}/{_LEVEL_ORDER[want_level]}"
            )
    n_nodes = int(ints(1)[0])
    if n_nodes != len(want["detail"]):
        return problems + [
            f"search[{i}]: {n_nodes} node(s) in the detail block != "
            f"{len(want['detail'])}"
        ]
    for k, node in enumerate(want["detail"]):
        n, turn = ints(2)
        digest = bytes(int(v) & 0xFF for v in ints(32))
        reservoir = int(ints(1)[0])
        if (int(n), int(turn), reservoir) != (node["n"], node["turn"], node["reservoir"]):
            problems.append(
                f"search[{i}]: node {k} (N, turn, reservoir) "
                f"({n}, {turn}, {reservoir}) != "
                f"({node['n']}, {node['turn']}, {node['reservoir']})"
            )
        if digest != node["memory_digest"]:
            problems.append(
                f"search[{i}]: node {k} memory digest {digest.hex()[:16]} != "
                f"{node['memory_digest'].hex()[:16]}"
            )
        n_a = int(ints(1)[0])
        node_actions = [int(v) for v in ints(n_a)]
        if node_actions != node["actions"]:
            problems.append(
                f"search[{i}]: node {k} candidates {len(node_actions)} vs "
                f"{len(node['actions'])}; first difference at "
                f"{_first_diff(node_actions, node['actions'])}"
            )
        for name in ("prior", "regret", "avg"):
            problems += _float_difference(
                f"search[{i}]: node {k} {name}", node[name], floats(node[name].size)
            )
        n_t = int(ints(1)[0])
        if n_t != len(node["tables"]):
            return problems + [
                f"search[{i}]: node {k} has {n_t} table(s) != {len(node['tables'])}"
            ]
        for j, table in enumerate(node["tables"]):
            n_self, last_used, touch = ints(3)
            if (int(n_self), int(last_used), int(touch)) != (
                table["n_self"], table["last_used"], table["touch_count"],
            ):
                problems.append(
                    f"search[{i}]: node {k} table {j} shape/LRU "
                    f"({n_self}, {last_used}, {touch}) != ({table['n_self']}, "
                    f"{table['last_used']}, {table['touch_count']})"
                )
            n_b = int(ints(1)[0])
            cols = [int(v) for v in ints(n_b)]
            if cols != table["actions"]:
                problems.append(
                    f"search[{i}]: node {k} table {j} columns {len(cols)} vs "
                    f"{len(table['actions'])}; first difference at "
                    f"{_first_diff(cols, table['actions'])}"
                )
            for name in ("prior", "regret", "avg", "visits", "q"):
                problems += _float_difference(
                    f"search[{i}]: node {k} table {j} {name}",
                    table[name], floats(table[name].size),
                )
    consumed = ints(1)[0]
    if int(consumed) != want["draws"]:
        problems.append(
            f"search[{i}]: consumed {consumed} draw(s), the oracle recorded "
            f"{want['draws']}"
        )
    return problems


def _float_difference(label: str, want, got) -> list[str]:
    """Bit-exact, with the tolerance reported rather than applied."""
    want = np.asarray(want, dtype=np.float64).ravel()
    got = np.asarray(got, dtype=np.float64).ravel()
    if got.size != want.size:
        return [f"{label}: length {got.size} != {want.size}"]
    delta = np.abs(got - want)
    _note("search.max", float(delta.max()) if delta.size else 0.0)
    if np.array_equal(got, want):
        return []
    bad = int(np.argmax(delta))
    return [
        f"{label}: not bit-identical; max |d| {delta.max():.3g} at {bad} "
        f"({got[bad]!r} != {want[bad]!r})"
    ]


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
        elif kind == "playmask":
            want_mask, want_removed, want_seen, want_floor, want_threat = want
            got_mask = np.asarray(got[:len(want_mask)], dtype=bool)
            if not np.array_equal(got_mask, want_mask):
                diff = np.flatnonzero(got_mask != want_mask)
                problems.append(
                    f"playmask[{i}]: {diff.size} action(s) differ, first "
                    f"{int(diff[0])} (rust {bool(got_mask[diff[0]])})"
                )
            tail = got[len(want_mask):]
            for name, want_v, got_v in zip(
                ("removed", "enemy_visible", "garrison_floor", "threat_arrival"),
                (want_removed, want_seen, want_floor, want_threat),
                tail,
            ):
                if int(got_v) != int(want_v):
                    problems.append(f"playmask[{i}]: {name} {got_v} != {want_v}")
        elif kind == "shaping":
            want_scores, want_blend = want
            n = want_scores.size
            got_scores = _decode_f64(got[:n])
            got_blend = _decode_f64(got[n : 2 * n])
            for name, wanted, gotten, tol in (
                ("scores", want_scores, got_scores, SHAPING_TOLERANCE),
                ("blend", want_blend, got_blend, SHAPING_TOLERANCE),
            ):
                delta = np.abs(gotten - wanted)
                _note(f"shaping.{name}.max", float(delta.max()) if delta.size else 0.0)
                if not np.array_equal(gotten, wanted):
                    bad = int(np.argmax(delta))
                    over = int(np.sum(delta > tol))
                    problems.append(
                        f"shaping[{i}]: {name} not bit-identical; max |d| "
                        f"{delta.max():.3g} at action {bad} "
                        f"({gotten[bad]!r} != {wanted[bad]!r}), {over} over {tol:g}"
                    )
        elif kind == "candidates":
            want_mandatory, want_candidates = want
            at = 0
            n_m = got[at]; at += 1
            got_mandatory = got[at : at + n_m]; at += n_m
            n_c = got[at]; at += 1
            got_candidates = got[at : at + n_c]
            if got_mandatory != want_mandatory:
                problems.append(
                    f"candidates[{i}]: mandatory {len(got_mandatory)} vs "
                    f"{len(want_mandatory)}; first difference at "
                    f"{_first_diff(got_mandatory, want_mandatory)}"
                )
            if got_candidates != want_candidates:
                problems.append(
                    f"candidates[{i}]: ordered {len(got_candidates)} vs "
                    f"{len(want_candidates)}; first difference at "
                    f"{_first_diff(got_candidates, want_candidates)}"
                )
        elif kind == "planners":
            problems += _compare_planners(i, want, got)
        elif kind == "constrain":
            want_action, want_blocks = want
            got_action = tuple(int(v) for v in got[:5])
            if got_action != want_action:
                problems.append(
                    f"constrain[{i}]: action {got_action} != {want_action}"
                )
            if int(got[5]) != int(want_blocks) or int(got[6]) != int(want_blocks):
                problems.append(
                    f"constrain[{i}]: blocks_oscillation {got[5]}/{got[6]} "
                    f"!= {want_blocks} (plain / reveal-grid)"
                )
        elif kind == "runtime":
            (want_ok, want_p99), want_action = want
            got_ok = int(got[0])
            got_p99 = float(_decode_f64(got[1:2])[0]) if got_ok else 0.0
            if got_ok != want_ok or (want_ok and got_p99 != want_p99):
                problems.append(
                    f"runtime[{i}]: nearest_rank_p99 {got_ok}/{got_p99!r} != "
                    f"{want_ok}/{want_p99!r}"
                )
            got_action = tuple(int(v) for v in got[2:7])
            if got_action != want_action:
                problems.append(
                    f"runtime[{i}]: highest_prior_legal {got_action} != {want_action}"
                )
        elif kind == "evict":
            count = int(got[0])
            survivors = [
                (int(got[1 + 3 * k]), int(got[2 + 3 * k]), int(got[3 + 3 * k]))
                for k in range(count)
            ]
            if survivors != want:
                problems.append(
                    f"evict[{i}]: survivors {survivors} != {want} "
                    f"(seed, last_used, touch_count)"
                )
        elif kind == "matrix":
            problems += _compare_matrix(i, want, got)
        elif kind == "search":
            problems += _compare_search(i, want, got)
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
    # The twenty-eight kinds this harness still owns, which is
    # `test_morpheus_joe_parity_tier1.KINDS`. N1 retired `tensor`, `net`,
    # `prior`, `summary` and `decide` from `build_cases` and left them on this
    # list, so a bare run — the one `tools/mutation_check.py` makes for any
    # mutation that maps to "all surfaces" — died on `unknown kind 'tensor'`
    # and took the mutation gate with it. Found and fixed at N2.
    #
    # `sequence` is deliberately **not** here. Every kind on this list is
    # checked against Python morpheus; `sequence` is checked against joe's own
    # recorded corpus, so it runs from `tests/test_morpheus_joe_bridge.py`
    # instead and `mutation_check.py` dispatches it there.
    parser.add_argument(
        "--kinds",
        nargs="*",
        default=[
            "transition", "order", "observe", "mask", "cost",
            "memory", "hash", "symmetry",
            "npsum", "argsort", "propose", "filter",
            "rejuvenate", "maxent", "reservoir", "toplegal", "initbelief",
            "matrix", "runtime", "evict", "playmask", "candidates", "planners",
            "shaping", "constrain", "search",
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

    flips = blas_amplification()
    if flips["cases"]:
        print(
            f"search: the oracle's BLAS moved {flips['vectors']} statistic "
            f"vector(s) over {flips['cases']} case(s), flipping the "
            f"regret-matching branch on {flips['regret_branch']} enemy table(s)"
        )
    stats = worst_observed()
    if stats:
        print("worst observed |\u0394| (the headroom under each budget):")
        for name in sorted(stats):
            print(f"  {name:<34} {stats[name]:.3g}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
