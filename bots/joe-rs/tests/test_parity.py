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
#
# The achieved gate is **relative, per frame**: each frame's max |delta| over
# that frame's own largest reference activation, maximised across frames.
#
# Why not an absolute pin (which is what this was until 2026-08-14): the
# absolute error tracks activation magnitude, and a stronger checkpoint has
# larger activations. Measured across the step-5000 and step-6000 corpora:
#
#   corpus       max|bin logit|   max |dbin|   |dbin| / scale
#   step 5000    35.18            4.387e-5     1.247e-6
#   step 6000    46.54            5.722e-5     1.230e-6
#
# The step-6000 value head is more confident, so it emits ~1.32x larger bin
# logits and ~1.30x larger absolute error — the same ~10 ULP over a 384-wide
# reduction, and not a precision regression. An absolute pin therefore fails
# on every stronger export for a reason that is not a bug, which is the one
# false alarm the M2 rule cannot afford. The ratio column is the invariant.
#
# Why per frame rather than per corpus: normalising by a corpus-wide maximum
# makes the statistic depend on corpus composition. The committed 13-frame
# smoke slice is early-game, where activations are ~half the full corpus's,
# so the same noise read 1.7e-6 against a corpus-max scale and would have
# failed a bound the full corpus passed.
#
# Pinned with ~30% headroom over the worst measured, which keeps the gate
# ~30x tighter than the nominal absolute contract. Per-frame relative figures
# (2026-08-14, step 6000): full corpus 731 frames logit 2.331e-6, bin
# 2.244e-6; smoke slice logit 1.055e-6, bin 2.120e-6. The cross-checkpoint
# evidence above is the corpus-max ratio, measured on both nets; the
# per-frame statistic itself is measured on step 6000 only. Exceeding these
# is still a finding, not a reason to relax silently — `mutation_check`
# re-confirms this gate kills all 9 planted bugs.
LOGIT_REL_ACHIEVED = 3.0e-6   # worst measured 2.331e-6
BIN_REL_ACHIEVED = 3.0e-6     # worst measured 2.244e-6
# |value| <= 1 by construction (bin_centers span [-1, 1]), so this one is
# already scale-free and stays absolute. It tracks bin sharpness rather than
# bin magnitude: 9.537e-7 at step 5000, 2.205e-6 at step 6000.
VALUE_TOL_ACHIEVED = 2.5e-6

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
    logit_scale = 0.0
    bin_scale = 0.0
    rel_logit = 0.0
    rel_bin = 0.0
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
            # Relative error is computed **per frame**, against that frame's
            # own largest activation, then maximised over frames. Normalising
            # by a corpus-wide maximum instead would make the statistic depend
            # on corpus composition: a short early-game slice has smaller
            # activations, so the same float noise would read as a larger
            # relative error. Masked policy entries (-1e9) are structural,
            # not computed, and are excluded from the scale.
            ref_logits = data["logits"][k]
            live = ref_logits[ref_logits > -1e8]
            f_logit_err = float(np.abs(logits - ref_logits).max())
            f_logit_scale = float(np.abs(live).max())
            max_logit_err = max(max_logit_err, f_logit_err)
            logit_scale = max(logit_scale, f_logit_scale)
            rel_logit = max(rel_logit, f_logit_err / f_logit_scale)

            ref_bins = data["value_bins"][k]
            f_bin_err = float(np.abs(bins - ref_bins).max())
            f_bin_scale = float(np.abs(ref_bins).max())
            max_bin_err = max(max_bin_err, f_bin_err)
            bin_scale = max(bin_scale, f_bin_scale)
            rel_bin = max(rel_bin, f_bin_err / f_bin_scale)

            max_value_err = max(max_value_err, abs(float(value) - float(data["value"][k])))
            n_frames += 1
    print(f"\n[tier2] {n_frames} frames: max |dlogit| {max_logit_err:.3e} "
          f"(per-frame rel {rel_logit:.3e}), max |dvalue| {max_value_err:.3e}, "
          f"max |dbin| {max_bin_err:.3e} (per-frame rel {rel_bin:.3e}) "
          f"[largest scales: logit {logit_scale:.2f}, bin {bin_scale:.2f}; "
          f"nominal abs {LOGIT_TOL_NOMINAL:.0e}]")
    assert rel_logit <= LOGIT_REL_ACHIEVED
    assert rel_bin <= BIN_REL_ACHIEVED
    assert max_value_err <= VALUE_TOL_ACHIEVED
    # Absolute backstop: the engineering contract from the port plan holds
    # regardless of how large activations get.
    assert max_logit_err <= LOGIT_TOL_NOMINAL
    assert max_bin_err <= LOGIT_TOL_NOMINAL


# ---- Tier 3: decision-level ----


def test_decide_tier3(games):
    agree = 0
    total = 0
    logit_scale = 0.0
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
            ref = data["logits"][k]
            logit_scale = max(logit_scale, float(np.abs(ref[ref > -1e8]).max()))
    rate = agree / total if total else 0.0
    print(f"\n[tier3] {agree}/{total} greedy actions equal ({rate:.4%})")
    for name, t, got_idx, want_idx, margin in divergences:
        print(f"[tier3]   {name} turn {t}: rust {got_idx} vs jax {want_idx}, "
              f"top-2 margin {margin:.3e}")
    assert rate >= DECIDE_AGREE_MIN, divergences
    # A divergence is only excusable as float noise if the two top logits sat
    # closer together than the tier-2 forward error could move them. That is
    # an absolute question, so the relative bound is put back on this corpus's
    # own logit scale.
    tie_margin_max = LOGIT_REL_ACHIEVED * logit_scale
    for name, t, got_idx, want_idx, margin in divergences:
        assert margin <= tie_margin_max, (
            f"{name} turn {t}: divergence with top-2 margin {margin:.3e} "
            f"outside the tier-2 bound {tie_margin_max:.3e} — a logic bug, "
            f"not float noise (R3)")
