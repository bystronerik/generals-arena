"""
Is the observation bridge joe's observation pipeline, turn for turn?

N2's gate, and the reason it can be a gate at all is that the fork does not
have to record anything. joe-rs's corpus under `data/joe/joe-rs-parity/games/`
already stores, per turn of every game, a CRC-32 of the augmented tensor that
**joe's JAX pipeline** produced (`all_aug_hash`) and the accumulated
`AugmentedObsState` the game ended on (`final_state_*`). This file replays the
same games through `morpheus-joe parity sequence` — morpheus's `Observation`,
widened, into joe's `augment_obs` — and asserts the digests and the final state
agree bit for bit.

That is a stronger check than it looks. The bridge's failure mode is silent:
a mis-widened grid, a missed `mem::swap`, a temporal window advanced twice or
not at all, all leave a tensor that is in range, plausible, and wrong, and a
frozen net answers it with a confident bad move. Nothing at play time notices.
A per-turn digest over whole games notices on the first turn that differs, and
the state check catches an error that cancels inside a single frame but
accumulates across 500.

**Q10 rides along in a second column.** After §6.2 only morpheus's
`live_build_cost` prices a castle and joe's `build_cost_from_raw` is never
consulted again, so the two can drift with nothing to say so. The surface
counts the cells where they disagree, per turn, with the memory advanced
exactly as the runtime advances it; the gate is zero everywhere. A hand-built
frame would not have found a disagreement that needs a latched castle to
appear.

The second half of the gate is `test_morpheus_joe_wire_replay.py`: same games,
whole binary, replies compared to joe's own. This file localizes a bridge bug;
that one proves there is not one.

Marked `morpheus`, so it is a gate rather than part of the default suite
(`pytest -m morpheus bots/morpheus-joe`): it needs a release binary and the
derived corpus, neither of which the default suite can assume (AGENTS.md,
"Test suite budget").
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.morpheus

BOT_DIR = Path(__file__).resolve().parents[1]
REPO = BOT_DIR.parents[1]
# `MORPHEUS_JOE_BINARY` aims this at the mutation profile's binary, the same
# way `morpheus_joe_parity_cases.py` reads it — `tools/mutation_check.py` needs
# to run this surface, because it is the only one that can see a bridge bug.
BINARY = Path(
    os.environ.get("MORPHEUS_JOE_BINARY", BOT_DIR / "target" / "release" / "morpheus-joe")
)
CORPUS = REPO / "data" / "joe" / "joe-rs-parity" / "games"
# The committed slice joe-rs keeps for a checkout with no derived data. One
# game, and enough to keep this test meaningful in that case.
SMOKE = REPO / "bots" / "joe-rs" / "tests" / "fixtures"

# The plan asks for three whole games. More would cost minutes of `augment_obs`
# for no new failure mode: a bridge that is right for 1,500 consecutive turns
# across three maps is not wrong on the fourth game.
GAMES_WANTED = 3

PAD = 21
CELLS = PAD * PAD
HISTORY = 7
TEMPORAL_WINDOW = 512

# `AugState`'s field order, which is also the order `surfaces/bridge.rs` writes
# it and the order joe-rs's own harness reads it. Positional, like every layout
# in this harness: a reordering here is a silent misread, not an error.
STATE_LAYOUT = (
    ("army_stack", HISTORY * CELLS, "f32"),
    ("enemy_stack", HISTORY * CELLS, "f32"),
    ("last_army", CELLS, "f32"),
    ("last_enemy_army", CELLS, "f32"),
    ("castles", CELLS, "bool"),
    ("generals", CELLS, "bool"),
    ("mountains", CELLS, "bool"),
    ("seen", CELLS, "bool"),
    ("enemy_seen", CELLS, "bool"),
    ("last_enemy_army_seen_value", CELLS, "f32"),
    ("last_enemy_army_seen_timestep", CELLS, "f32"),
    ("opponent_army_history", TEMPORAL_WINDOW, "f32"),
    ("opponent_land_history", TEMPORAL_WINDOW, "f32"),
    ("temporal_step", 1, "int"),
)
STATE_LEN = sum(width for _, width, _ in STATE_LAYOUT)
# The `(2, 512)` buffer handed to the forward, written after the state.
TEMPORAL_LEN = 2 * TEMPORAL_WINDOW


def f32_bits(arr) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32).ravel()


def games() -> list[Path]:
    """The derived corpus when it is there, else joe-rs's committed slice."""
    found = sorted(CORPUS.glob("*.npz")) if CORPUS.is_dir() else []
    if not found:
        found = sorted(SMOKE.glob("*.npz"))
    return found[:GAMES_WANTED]


def parse_in_log(path: Path):
    """Handshake plus frames, straight off the log the engine sent joe."""
    lines = path.read_text().splitlines()
    _, h, w = (int(x) for x in lines[0].split())
    frames = []
    pos, frame_len = 1, 1 + 3 * h
    while pos + frame_len <= len(lines):
        chunk = lines[pos:pos + frame_len]
        scalars = [int(x) for x in chunk[0].split()]
        grids = np.array(
            [[int(x) for x in row.split()] for row in chunk[1:]],
            dtype=np.int64).reshape(3, h, w)
        frames.append(scalars + grids.ravel().tolist())
        pos += frame_len
    return h, w, frames


def run_sequence(cases: list[list[int]]) -> list[np.ndarray]:
    """One line of integers per case, as `parity::run` writes them."""
    stream = [str(len(cases))]
    for case in cases:
        stream.extend(str(v) for v in case)
    proc = subprocess.run(
        [str(BINARY), "parity", "sequence"],
        input="\n".join(stream).encode(),
        capture_output=True,
        cwd=str(BOT_DIR),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"morpheus-joe parity sequence failed: {proc.stderr.decode()[-2000:]}")
    return [np.array(line.split(), dtype=np.int64)
            for line in proc.stdout.decode().splitlines() if line.strip()]


def decode_state(values: np.ndarray) -> dict:
    out, pos = {}, 0
    for field, width, kind in STATE_LAYOUT:
        chunk = values[pos:pos + width]
        assert len(chunk) == width, f"truncated state at {field}"
        pos += width
        if kind == "int":
            out[field] = int(chunk[0])
        elif kind == "bool":
            out[field] = chunk.astype(bool)
        else:
            out[field] = chunk.astype(np.uint32)
    assert pos == STATE_LEN, f"decoded {pos} of {STATE_LEN} state integers"
    return out


@pytest.fixture(scope="module")
def replayed():
    """Every game replayed once; the four tests below read the same run.

    Module-scoped because the replay is the cost — `augment_obs` over ~1,000
    turns — and the four assertions are four readings of one output, not four
    experiments.
    """
    if not BINARY.is_file():
        pytest.skip(
            f"no release binary at {BINARY}; build with "
            f"`cargo build --release --manifest-path {BOT_DIR}/Cargo.toml`")
    found = games()
    if not found:
        pytest.skip(f"no recorded games under {CORPUS} or {SMOKE}")

    cases, meta = [], []
    for npz_path in found:
        in_log = npz_path.parent / f"{npz_path.name.removesuffix('.npz')}.in.log"
        if not in_log.is_file():
            continue
        data = np.load(npz_path)
        h, w, frames = parse_in_log(in_log)
        turns = int(data["turns"])
        assert len(frames) >= turns, f"{npz_path.name}: {len(frames)} frames, {turns} turns"
        case = [h, w, turns]
        for frame in frames[:turns]:
            case.extend(frame)
        cases.append(case)
        meta.append((npz_path.name, data, turns))
    if not cases:
        pytest.skip("no recorded games carry an input log")
    return meta, run_sequence(cases)


def test_the_augmented_tensor_matches_joes_turn_for_turn(replayed):
    """
    Per turn, the CRC-32 of the (39, 21, 21) tensor, against the digest joe's
    own pipeline recorded. Whole games, not sampled frames: the bridge's state
    accumulates, so a bug that first shows on turn 300 is the bug this is for.
    """
    meta, lines = replayed
    total = 0
    for (name, data, turns), got in zip(meta, lines):
        assert len(got) == 2 * turns + STATE_LEN + TEMPORAL_LEN, \
            f"{name}: {len(got)} integers"
        digests = got[0:2 * turns:2]
        want = np.asarray(data["all_aug_hash"])[:turns]
        bad = np.nonzero(digests != want)[0]
        assert bad.size == 0, (
            f"{name}: the augmented tensor diverges first at turn {bad[0]} of "
            f"{turns} ({bad.size} turns differ). Localize inside that turn with "
            f"joe-rs's `obs` surface, which compares the tensor channel by "
            f"channel rather than as one digest.")
        total += turns
    print(f"\n[bridge] {len(meta)} games, {total} turns, every tensor digest equal")


def test_the_accumulated_state_matches_joes_at_the_final_turn(replayed):
    """
    The digest above is a hash of the *output*; this is the carried state
    itself, field by field. They fail differently: a state field that no
    channel reads — `enemy_seen`, the two `last_enemy_army_seen_*` planes — is
    invisible to the digest on the turn it goes wrong and shows up here.
    """
    meta, lines = replayed
    for (name, data, turns), got in zip(meta, lines):
        state = decode_state(got[2 * turns:])
        for field, _, kind in STATE_LAYOUT:
            want = data[f"final_state_{field}"]
            if kind == "int":
                assert state[field] == int(want), f"{name}: temporal_step"
            elif kind == "bool":
                assert np.array_equal(state[field], np.asarray(want).ravel()), \
                    f"{name}: final state {field}"
            else:
                assert np.array_equal(state[field], f32_bits(want)), \
                    f"{name}: final state {field} not bit-exact"


def test_the_forwards_temporal_input_is_the_two_ring_buffers(replayed):
    """
    §6.1's window, checked against itself rather than against a corpus.

    The `(2, 512)` buffer the forward is handed is a copy of the two histories
    above, and the corpus has no separate record of it — so the question is not
    what it contains but whether it is that copy. Halves swapped, or a buffer
    left over from the previous turn, leaves a state the corpus still agrees
    with and a network input that is wrong on every turn. Nothing else in this
    file can see that.
    """
    meta, lines = replayed
    for (name, _, turns), got in zip(meta, lines):
        state = decode_state(got[2 * turns:])
        temporal = got[2 * turns + STATE_LEN:].astype(np.uint32)
        assert len(temporal) == TEMPORAL_LEN, f"{name}: {len(temporal)} temporal integers"
        assert np.array_equal(temporal[:TEMPORAL_WINDOW], state["opponent_army_history"]), \
            f"{name}: the first half of the temporal input is not the army history"
        assert np.array_equal(temporal[TEMPORAL_WINDOW:], state["opponent_land_history"]), \
            f"{name}: the second half of the temporal input is not the land history"


def test_the_two_build_cost_rules_agree_on_every_cell(replayed):
    """
    Q10, closed over real games rather than argued from the formula.

    joe prices the own structures the *frame* shows; morpheus folds in what
    `VisibleMemory` latched — `own_general` and `known_castle`. Owning a cell
    implies seeing it, so the latch should never add a structure the frame does
    not already carry, and this is the check that the "should" is a "does".
    It matters because §6.2 leaves only morpheus's rule on the play path, so a
    disagreement would change which castles the bot believes it can afford and
    say nothing.
    """
    meta, lines = replayed
    for (name, _, turns), got in zip(meta, lines):
        disagreements = got[1:2 * turns:2]
        bad = np.nonzero(disagreements)[0]
        assert bad.size == 0, (
            f"{name}: joe's build_cost_from_raw and morpheus's live_build_cost "
            f"disagree on {disagreements[bad[0]]} cells, first at turn {bad[0]}")
