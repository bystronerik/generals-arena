"""
Parity, smoke slice: the Rust port must match the Python oracle exactly.

The CI-sized half of the harness in `parity_cases.py`. It runs the committed
seven-frame slice plus the synthetic states through every ported surface, in
under a second. The full corpus — tens of thousands of cases — runs from
`tools/run_parity.sh` before a milestone gate.

What "exact" means here is bit-exact — including the 49-plane tensor, whose
tier-2 budget is 1e-6 but which matches to the last bit. Everything downstream
inherits it: the belief filter's likelihood is *observations match or they do
not*, so a fog rule off by one cell turns a correct particle into a rejected
one, a node key that differs is a different node, and a search built on a
transition that rounds differently explores a game nobody is playing.

Skips with a named reason when the release binary is absent — a cold checkout
has not built it yet, and a silently-passing parity test is worse than none.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import parity_cases as pc  # noqa: E402

KINDS = (
    "transition", "order", "observe", "mask", "cost",
    "memory", "hash", "tensor", "symmetry",
)


@pytest.fixture(scope="module")
def frames() -> list[dict]:
    if not pc.BINARY.is_file():
        pytest.skip(
            f"no release binary at {pc.BINARY}; run `cargo build --release "
            f"--manifest-path bots/morpheus-rs/Cargo.toml`"
        )
    if not pc.SMOKE_FIXTURE.is_file():
        pytest.skip(f"no parity smoke slice at {pc.SMOKE_FIXTURE}")
    return pc.load_frames([pc.SMOKE_FIXTURE])


@pytest.mark.parametrize("kind", KINDS)
def test_the_port_matches_the_oracle(frames, kind):
    count, problems = pc.check(kind, frames)
    assert count > 0, f"{kind}: no cases were generated"
    assert not problems, f"{kind}: {len(problems)} mismatch(es)\n" + "\n".join(problems[:10])


def test_the_synthetic_states_reach_what_replay_cannot(frames):
    """
    Guards the coverage the recorded corpus does not have.

    Two mutations survived the full recorded case set: deleting the 50-tick
    army growth (no recorded state sat at `time % 50 == 49`) and deleting the
    NumPy index wrap (it reads row `h-1`, which on a padded board is mountain
    border and never owned). The synthetic states exist to reach both. If this
    ever fails, the parity runs above have quietly stopped proving those.
    """
    states = pc.synthetic_states()
    times = {int(s.time) for s in states}
    assert any((t + 1) % 50 == 0 for t in times), "no 50-tick growth boundary"
    assert any(t >= 800 for t in times), "no deathtouch regime"
    assert any(t < 800 for t in times), "no pre-deathtouch regime"
    assert any(int(s.winner) >= 0 for s in states), "no finished game"

    # The wrap cell is (h-1, 0): some state must have it owned, and some must
    # have the two seats disagreeing about it, or the wrap decides nothing.
    owned = [
        (bool(s.ownership[0][s.armies.shape[0] - 1, 0]),
         bool(s.ownership[1][s.armies.shape[0] - 1, 0]))
        for s in states
    ]
    assert any(a != b for a, b in owned), "no state makes the index wrap decisive"


def test_an_affordable_build_exists_somewhere_in_the_case_set(frames):
    """
    The build gates are only tested when a build is actually reachable.

    `known_passable_base` refuses a build on ground never seen. With no owned,
    affordable, plain cell anywhere in the case set, deleting that check
    changes nothing — which is exactly what mutation testing found before the
    rich-cell state was added.
    """
    from action import legal_mask, PASS_INDEX

    buildable = 0
    for state in pc.synthetic_states():
        from observe import emit_observation

        for seat in (0, 1):
            obs = emit_observation(state, seat, as_arrays=True)
            mask = legal_mask(obs, pc.memory_for(state, seat))
            buildable += int(mask[8 * 441 : PASS_INDEX].any())
    assert buildable, "no synthetic case can legally build"
