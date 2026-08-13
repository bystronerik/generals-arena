"""joe-rs parity drivers (port-plan §6): tier 1 bit-exact, tier 2 tolerance,
tier 3 decision-level, plus the sequence state-accumulation check.

Marker `joe` — excluded from the default suite (needs a release binary and
the fixture corpus). Runs over the full corpus under
`data/joe/joe-rs-parity/games/` when present, else the committed smoke slice
in `tests/fixtures/`.

Tolerances follow the M2 rule: the *achieved* bound over the corpus is the
enforced one; the nominal bound is reported separately.
"""
from __future__ import annotations

import numpy as np
import pytest

from parity_lib import (
    BINARY,
    CELLS,
    N_CHANNELS,
    N_LOGITS,
    PAD,
    STATE_FIELDS,
    STATE_BOOL_FIELDS,
    TEMPORAL_WINDOW,
    corpus_games,
    decode_state,
    encode_frame,
    encode_state,
    f32_bits,
    bits_f32,
    parse_in_log,
    run_surface,
    state_stream_len,
    ulp_distance,
)

pytestmark = pytest.mark.joe

# Channel 21 is `log1p(t)/5` — the one transcendental. The gate below is the
# achieved bound over the corpus plus the exhaustive 0..4096 sweep in
# `test_channel21_log1p_exhaustive`: XLA's f32 log1p and Rust's agree
# bit-for-bit on every input the channel can see, so the pin is zero ULP.
CH21_MAX_ULP = 0

# Tier-2 nominal gate (different GEMM summation orders); the achieved bound
# is asserted in `test_forward_tier2` and recorded in the test output.
LOGIT_TOL_NOMINAL = 1e-4
# Achieved over the 12-game corpus (673 frames, 2026-08-14): max |dlogit|
# 3.052e-5, max |dbin| 4.387e-5, max |dvalue| 9.537e-7. Pinned at the
# measurement per the M2 rule; a corpus refresh that pushes past these is a
# finding, not a reason to relax silently.
LOGIT_TOL_ACHIEVED = 3.1e-5
BIN_TOL_ACHIEVED = 4.4e-5
VALUE_TOL_ACHIEVED = 1e-6

# Tier-3 gate: greedy action equal on >= 99.5% of frames; every divergence
# must be a near-tie inside the tier-2 bound.
DECIDE_AGREE_MIN = 0.995


@pytest.fixture(scope="module")
def games():
    if not BINARY.is_file():
        pytest.skip(f"no release binary at {BINARY} (cargo build --release)")
    found = corpus_games()
    if not found:
        pytest.skip("no parity corpus and no smoke fixtures")
    return found


def load_game(npz_path):
    data = np.load(npz_path)
    in_log = npz_path.with_suffix("").with_suffix("")  # strip .npz
    in_log = npz_path.parent / (npz_path.name.removesuffix(".npz") + ".in.log")
    return data, in_log


def sampled_frames(data, in_log):
    """(turn, scalars, grids, k) for each sampled frame k."""
    _, H, W, frames = parse_in_log(in_log)
    out = []
    for k, t in enumerate(data["sampled_turns"].tolist()):
        scalars, grids = frames[t]
        out.append((t, scalars, grids, k))
    return H, W, out


# ---- Tier 1: bit-exact ----


def test_raw_bit_exact(games):
    for npz_path in games:
        data, in_log = load_game(npz_path)
        H, W, frames = sampled_frames(data, in_log)
        cases = [[H, W] + encode_frame(scalars, grids) for _, scalars, grids, _ in frames]
        got = run_surface("raw", cases)
        want = np.concatenate([f32_bits(data["raw"][k]) for *_ , k in frames])
        assert np.array_equal(got.astype(np.uint32), want), npz_path.name


def test_cost_bit_exact(games):
    for npz_path in games:
        data, in_log = load_game(npz_path)
        H, W, frames = sampled_frames(data, in_log)
        cases = [[H, W] + f32_bits(data["raw"][k]).tolist() for *_, k in frames]
        got = run_surface("cost", cases)
        want = np.concatenate([data["cost"][k].ravel() for *_, k in frames])
        assert np.array_equal(got, want), npz_path.name


def test_masks_bit_exact(games):
    for npz_path in games:
        data, in_log = load_game(npz_path)
        H, W, frames = sampled_frames(data, in_log)
        cases = [[H, W] + f32_bits(data["raw"][k]).tolist() for *_, k in frames]
        got = run_surface("mask", cases)
        want = np.concatenate([
            np.concatenate([
                data["move"][k].astype(np.int64).ravel(),
                data["build"][k].astype(np.int64).ravel(),
            ])
            for *_, k in frames
        ])
        assert np.array_equal(got, want), npz_path.name


def _state_in(data, k) -> dict:
    return {f: data[f"state_{f}"][k] for f in STATE_FIELDS}


def assert_states_equal(got: dict, want: dict, label: str):
    for field in STATE_FIELDS:
        g, w = got[field], want[field]
        if field == "temporal_step":
            assert int(g) == int(w), f"{label}: temporal_step {g} != {w}"
        elif field in STATE_BOOL_FIELDS:
            assert np.array_equal(np.asarray(g, bool), np.asarray(w, bool)), \
                f"{label}: state field {field}"
        else:
            assert np.array_equal(
                f32_bits(g), f32_bits(w)), f"{label}: state field {field} not bit-exact"


def test_obs_bit_exact(games):
    """The augmented tensor and the updated state, per sampled frame, from
    the recorded input state. Channel 21 gets its own ULP gate."""
    for npz_path in games:
        data, in_log = load_game(npz_path)
        H, W, frames = sampled_frames(data, in_log)
        cases = [
            [H, W] + f32_bits(data["raw"][k]).tolist() + encode_state(_state_in(data, k))
            for *_, k in frames
        ]
        got = run_surface("obs", cases)
        per_case = N_CHANNELS * CELLS + state_stream_len()
        assert len(got) == per_case * len(cases)
        for i, (t, _, _, k) in enumerate(frames):
            chunk = got[i * per_case:(i + 1) * per_case]
            aug_got = chunk[:N_CHANNELS * CELLS].astype(np.uint32)
            aug_want = f32_bits(data["aug"][k])
            label = f"{npz_path.name} turn {t}"

            g = aug_got.reshape(N_CHANNELS, CELLS)
            w = aug_want.reshape(N_CHANNELS, CELLS)
            for c in range(N_CHANNELS):
                if c == 21:
                    ulp = ulp_distance(bits_f32(g[c]), bits_f32(w[c]))
                    assert ulp.max() <= CH21_MAX_ULP, \
                        f"{label}: channel 21 max ULP {ulp.max()}"
                else:
                    assert np.array_equal(g[c], w[c]), f"{label}: channel {c}"

            state_got = decode_state(chunk[N_CHANNELS * CELLS:])
            # The recorded *next* state is the input state of the next
            # sampled frame only when frames are adjacent; instead compare
            # against what capture stored per-frame: state after this turn is
            # not stored, so rebuild the comparison from aug + known fields.
            # The stacks and scalar fields are all present inside aug or the
            # next state; the authoritative check is the sequence test. Here
            # we check the fields that aug exposes directly.
            np_state = state_got
            assert np.array_equal(
                f32_bits(np_state["army_stack"]),
                f32_bits(data["aug"][k][25:32])), f"{label}: army_stack vs aug"
            assert np.array_equal(
                f32_bits(np_state["enemy_stack"]),
                f32_bits(data["aug"][k][32:39])), f"{label}: enemy_stack vs aug"


def test_sequence_bit_exact(games):
    """Whole games through the Rust state machine: every turn's augmented
    tensor digest, then the final state, bit-exact (tier 1, sequence-level)."""
    for npz_path in games:
        data, in_log = load_game(npz_path)
        _, H, W, frames = parse_in_log(in_log)
        T = int(data["turns"])
        frames = frames[:T]
        case = [H, W, T]
        for scalars, grids in frames:
            case.extend(encode_frame(scalars, grids))
        got = run_surface("sequence", [case])
        hashes = got[:T]
        want_hashes = data["all_aug_hash"]
        first_bad = np.nonzero(hashes != want_hashes)[0]
        assert first_bad.size == 0, (
            f"{npz_path.name}: sequence diverges first at turn {first_bad[0]} "
            f"of {T} (localize with the obs surface at that turn)")
        final_got = decode_state(got[T:])
        final_want = {f: data[f"final_state_{f}"] for f in STATE_FIELDS}
        assert_states_equal(final_got, final_want, f"{npz_path.name} final state")


def test_channel21_log1p_exhaustive(games):
    """Channel 21 is `log1p(counter) * (1/5)` where the counter is an integer
    turn count. Sweep the whole domain (0..16384 — truncation is 1200, so
    >13x margin) against the jitted JAX computation, bit for bit. This is why
    CH21_MAX_ULP can be 0: the Rust side mirrors XLA's Eigen polynomial, not
    libm (src/xla_math.rs)."""
    import jax
    import jax.numpy as jnp

    counters = np.arange(0.0, 16385.0, dtype=np.float32)
    want = np.asarray(jax.jit(lambda v: jnp.log1p(v) / 5.0)(jnp.asarray(counters)))
    cases = [[len(counters)] + f32_bits(counters).tolist()]
    got = run_surface("log1p", cases)
    mismatch = np.nonzero(got.astype(np.uint32) != f32_bits(want))[0]
    assert mismatch.size == 0, (
        f"{mismatch.size} of {len(counters)} log1p values differ; first at "
        f"counter {counters[mismatch[0]]}: rust "
        f"{bits_f32(got.astype(np.uint32)[mismatch[:1]])[0]!r} vs jax "
        f"{want[mismatch[0]]!r}")


# ---- Tier 2: tolerance ----


def test_forward_tier2(games):
    max_logit_err = 0.0
    max_value_err = 0.0
    max_bin_err = 0.0
    n_frames = 0
    for npz_path in games:
        data, in_log = load_game(npz_path)
        H, W, frames = sampled_frames(data, in_log)
        cases = [
            [H, W]
            + f32_bits(data["aug"][k]).tolist()
            + data["move"][k].astype(np.int64).ravel().tolist()
            + data["build"][k].astype(np.int64).ravel().tolist()
            + f32_bits(data["temporal"][k]).tolist()
            for *_, k in frames
        ]
        got = run_surface("forward", cases)
        per_case = N_LOGITS + 1 + 128
        assert len(got) == per_case * len(cases)
        for i, (t, _, _, k) in enumerate(frames):
            chunk = got[i * per_case:(i + 1) * per_case]
            logits = bits_f32(chunk[:N_LOGITS])
            value = bits_f32(chunk[N_LOGITS:N_LOGITS + 1])[0]
            bins = bits_f32(chunk[N_LOGITS + 1:])
            max_logit_err = max(max_logit_err,
                                float(np.abs(logits - data["logits"][k]).max()))
            max_value_err = max(max_value_err, abs(float(value) - float(data["value"][k])))
            max_bin_err = max(max_bin_err,
                              float(np.abs(bins - data["value_bins"][k]).max()))
            n_frames += 1
    print(f"\n[tier2] {n_frames} frames: max |dlogit| {max_logit_err:.3e}, "
          f"max |dvalue| {max_value_err:.3e}, max |dbin| {max_bin_err:.3e} "
          f"(nominal {LOGIT_TOL_NOMINAL:.0e})")
    assert max_logit_err <= LOGIT_TOL_ACHIEVED
    assert max_bin_err <= BIN_TOL_ACHIEVED
    assert max_value_err <= VALUE_TOL_ACHIEVED


# ---- Tier 3: decision-level ----


def test_decide_tier3(games):
    agree = 0
    total = 0
    divergences = []
    for npz_path in games:
        data, in_log = load_game(npz_path)
        H, W, frames = sampled_frames(data, in_log)
        cases = [
            [H, W]
            + f32_bits(data["aug"][k]).tolist()
            + data["move"][k].astype(np.int64).ravel().tolist()
            + data["build"][k].astype(np.int64).ravel().tolist()
            + f32_bits(data["temporal"][k]).tolist()
            for *_, k in frames
        ]
        got = run_surface("decide", cases)
        per_case = 6
        for i, (t, _, _, k) in enumerate(frames):
            chunk = got[i * per_case:(i + 1) * per_case]
            idx_got = int(chunk[0])
            action_got = chunk[1:6]
            total += 1
            if idx_got == int(data["idx"][k]):
                assert np.array_equal(action_got, data["action"][k])
                agree += 1
            else:
                logits = data["logits"][k]
                top2 = np.partition(logits, -2)[-2:]
                margin = float(top2[1] - top2[0])
                divergences.append((npz_path.name, t, idx_got, int(data["idx"][k]), margin))
    rate = agree / total if total else 0.0
    print(f"\n[tier3] {agree}/{total} greedy actions equal ({rate:.4%})")
    for name, t, got_idx, want_idx, margin in divergences:
        print(f"[tier3]   {name} turn {t}: rust {got_idx} vs jax {want_idx}, "
              f"top-2 margin {margin:.3e}")
    assert rate >= DECIDE_AGREE_MIN, divergences
    for name, t, got_idx, want_idx, margin in divergences:
        assert margin <= LOGIT_TOL_ACHIEVED, (
            f"{name} turn {t}: divergence with top-2 margin {margin:.3e} "
            f"outside the tier-2 bound — a logic bug, not float noise (R3)")
