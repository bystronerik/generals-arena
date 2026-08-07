"""Part 02 transition kernel — deployment NumPy vs training JAX vs oracle."""
from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.morpheus

# Bot root is already on path via conftest; keep sibling imports working.
from state import GameInfo, from_engine, infos_equal, states_equal  # noqa: E402
from transition import (  # noqa: E402
    DEATHTOUCH_TURN,
    TRUNCATION_TURN,
    at_truncation,
    transition as deploy_transition,
)

REPO = Path(__file__).resolve().parents[3]
COMPETITION = REPO / "competition-module" / "competition"
sys.path.insert(0, str(COMPETITION))
sys.path.insert(0, str(REPO / "competition-module"))

from generals.core.env import GeneralsEnv  # noqa: E402
from generals.core.game import create_initial_state  # noqa: E402
import matchup  # noqa: E402

from training.morpheus.jax_preflight.fixtures import (  # noqa: E402
    all_parity_fixtures,
)
from training.morpheus.transition import (  # noqa: E402
    compile_batched_transition,
    make_competition_transition,
)

PASS = np.array([1, 0, 0, 0, 0], dtype=np.int32)
RIGHT, LEFT, DOWN, UP = 3, 2, 1, 0


def _oracle():
    return matchup.make_transition(GeneralsEnv(mode="competition"))


def _train():
    return make_competition_transition(GeneralsEnv(mode="competition"))


def _info_from_engine(info) -> GameInfo:
    return GameInfo(
        army=np.asarray(info.army).astype(np.int64),
        land=np.asarray(info.land).astype(np.int64),
        is_done=bool(info.is_done),
        winner=int(info.winner),
        time=int(info.time),
    )


def _assert_match(label: str, before, actions_np: np.ndarray):
    """Compare deployment + training adapters to the competition oracle."""
    oracle = _oracle()
    train = _train()
    actions_j = jnp.asarray(actions_np, dtype=jnp.int32)

    want_state, want_info = oracle(before, actions_j)
    train_state, train_info = train(before, actions_j)

    deploy_before = from_engine(before)
    got_state, got_info = deploy_transition(deploy_before, actions_np)

    want = from_engine(want_state)
    want_gi = _info_from_engine(want_info)
    train_as = from_engine(train_state)
    train_gi = _info_from_engine(train_info)

    assert states_equal(got_state, want), f"{label}: deploy state != oracle"
    assert infos_equal(got_info, want_gi), f"{label}: deploy info != oracle"
    assert states_equal(train_as, want), f"{label}: train state != oracle"
    assert infos_equal(train_gi, want_gi), f"{label}: train info != oracle"


def open_board(size: int = 8, time: int = 0):
    grid = jnp.zeros((size, size), dtype=jnp.int32).at[0, 0].set(1).at[0, size - 1].set(2)
    return create_initial_state(grid)._replace(time=jnp.int32(time))


def give(state, player, ij, army):
    i, j = ij
    return state._replace(
        armies=state.armies.at[i, j].set(army),
        ownership=state.ownership.at[player, i, j].set(True),
        ownership_neutral=state.ownership_neutral.at[i, j].set(False),
    )


def move(i, j, d, split=0):
    return np.array([0, i, j, d, split], dtype=np.int32)


def build(i, j):
    return np.array([2, i, j, 0, 0], dtype=np.int32)


@pytest.mark.parametrize("fixture", all_parity_fixtures(), ids=lambda f: f.name)
def test_parity_fixtures_match_oracle(fixture):
    actions = np.asarray(fixture.actions, dtype=np.int32)
    _assert_match(fixture.name, fixture.state, actions)


def test_castle_build_cost_and_remainder():
    state = give(open_board(), 0, (5, 5), 60)
    actions = np.stack([build(5, 5), PASS])
    _assert_match("build_base_cost", state, actions)


def test_deathtouch_wins_and_early_touch_fails():
    late = give(open_board(time=DEATHTOUCH_TURN + 1), 1, (0, 7), 500)
    late = give(late, 0, (0, 6), 3)
    actions = np.stack([move(0, 6, RIGHT), PASS])
    _assert_match("deathtouch_late", late, actions)

    early = give(open_board(time=DEATHTOUCH_TURN - 1), 1, (0, 7), 500)
    early = give(early, 0, (0, 6), 3)
    _assert_match("deathtouch_early", early, actions)


def test_chase_defense_and_mutual_touch():
    s = give(open_board(size=6, time=800), 0, (0, 4), 5)
    s = give(s, 1, (1, 4), 10)
    _assert_match("chase_defense", s, np.stack([move(0, 4, RIGHT), move(1, 4, UP)]))

    m = give(open_board(size=6, time=800), 0, (0, 4), 5)
    m = give(m, 1, (0, 1), 5)
    _assert_match("mutual_touch", m, np.stack([move(0, 4, RIGHT), move(0, 1, LEFT)]))


def test_contested_neutral_smaller_source_first():
    s = open_board()
    s = give(s, 0, (2, 1), 25)
    s = give(s, 1, (2, 3), 40)
    s = s._replace(armies=s.armies.at[2, 2].set(20))
    actions = np.stack([move(2, 1, RIGHT), move(2, 3, LEFT)])
    _assert_match("contested_neutral", s, actions)


def test_half_move_and_invalid_noop():
    s = give(open_board(), 0, (2, 2), 10)
    _assert_match("half_move", s, np.stack([move(2, 2, DOWN, split=1), PASS]))
    # Invalid: move from cell we do not own.
    _assert_match("invalid_source", s, np.stack([move(3, 3, DOWN), PASS]))


def test_growth_on_pass_and_truncation_helper():
    s = open_board(time=1)  # next time becomes 2 → structure growth
    _assert_match("pass_growth", s, np.stack([PASS, PASS]))

    capped = from_engine(open_board(time=TRUNCATION_TURN))
    assert at_truncation(capped)
    assert not at_truncation(from_engine(open_board(time=TRUNCATION_TURN - 1)))


def test_generated_joint_actions_match_oracle():
    """Sweep legal and invalid joints whose move-order destinations stay in bounds."""
    rng = np.random.default_rng(0)
    size = 6
    base = give(open_board(size=size, time=10), 0, (1, 1), 20)
    base = give(base, 1, (1, 4), 15)
    base = give(base, 0, (2, 2), 8)
    base = give(base, 1, (2, 3), 12)

    # Interior cells so every direction stays on the board for reinforce checks.
    cells = [(1, 1), (1, 4), (2, 2), (2, 3), (3, 3), (2, 1)]
    for i in range(40):
        parts = []
        for _seat in range(2):
            kind = int(rng.integers(0, 4))  # 0 move, 1 pass, 2 build, 3 invalid
            if kind == 1:
                parts.append(PASS.copy())
            elif kind == 2:
                r, c = cells[int(rng.integers(0, len(cells)))]
                parts.append(build(r, c))
            elif kind == 0:
                r, c = cells[int(rng.integers(0, len(cells)))]
                d = int(rng.integers(0, 4))
                split = int(rng.integers(0, 2))
                parts.append(move(r, c, d, split))
            else:
                # Wrong owner / empty / unaffordable build — still in-bound.
                r, c = cells[int(rng.integers(0, len(cells)))]
                parts.append(
                    np.array(
                        [
                            int(rng.choice([0, 2])),
                            r,
                            c,
                            int(rng.integers(0, 4)),
                            int(rng.integers(0, 2)),
                        ],
                        dtype=np.int32,
                    )
                )
        _assert_match(f"generated_{i}", base, np.stack(parts))


def test_training_batched_scan_compiles():
    scan_fn, pool, states = compile_batched_transition(num_envs=4, seed=0)
    from training.morpheus.transition import make_action_sequence

    actions_seq = make_action_sequence(seed=0, num_envs=4, num_steps=3)
    final, infos = scan_fn(states, actions_seq)
    assert final.armies.shape[0] == 4
    assert infos.time.shape[0] == 3  # scan length
