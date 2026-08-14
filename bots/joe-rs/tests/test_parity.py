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
# absolute error tracks activation magnitude, so an absolute pin fails on a
# checkpoint whose activations grew, for a reason that is not a bug — the one
# false alarm the M2 rule cannot afford.
#
# Why per frame rather than per corpus: normalising by a corpus-wide maximum
# makes the statistic depend on corpus composition. The committed 13-frame
# smoke slice is early-game, where activations are ~half the full corpus's,
# so the same noise read 1.7e-6 against a corpus-max scale and would have
# failed a bound the full corpus passed.
#
# **The 2026-08-14 pins (3.0e-6 / 3.0e-6) were over-fitted to the step-6000
# corpus.** They were set from the worst frame of one frame population, and a
# max over frames does not transfer to a different one. When the corpus was
# rebuilt for step 13500, an A/B on the *identical* 710 frames — same logs,
# same binary, only the weights swapped — showed the **step-10000** net also
# exceeding the bin pin, so the gate was already red before the new export:
#
#   identical 710 frames     step 10000   step 13500    ratio
#   rel logit  median         8.505e-7     1.017e-6      1.20x
#   rel logit  p99            2.239e-6     2.740e-6      1.22x
#   rel logit  max            2.514e-6     3.944e-6      1.57x
#   rel bin    median         9.432e-7     1.151e-6      1.22x
#   rel bin    p99            2.965e-6     4.891e-6      1.65x
#   rel bin    max            4.993e-6     7.940e-6      1.59x   <- pin was 3.0e-6
#
# Read the body, not the max: a uniform ~1.2x shift at unchanged activation
# scale (logit scale median 17.12 -> 16.91, bin 10.03 -> 9.34) is ordinary
# float behaviour, while the max wanders ~1.6x because it is a max over 710
# frames. Absolute error stays well inside the nominal contract above:
# max |dlogit| 4.387e-5 (step 10000) and 7.057e-5 (step 13500) against 1e-4.
#
# So the pins below carry ~2x headroom over the worst measured, sized to the
# observed per-checkpoint drift of the max rather than to a tighter number
# that would only have to be raised again. That is a real loosening, so it is
# only defensible because `mutation_check` re-confirms at these values that
# the gate still kills all 9 planted bugs — including `qk-proj-swap`, the
# leaf-misalignment bug this tier exists to catch. Re-measure and re-run
# `mutation_check` after every export; exceeding these is still a finding.
#
# 2026-08-15: the forward pass moved off candle onto the in-house kernel
# (src/gemm.rs — port-plan §9 R1, taken for intake, not latency). Same
# weights, same step-13500 corpus, different GEMM summation order and an FMA
# per term, so the achieved numbers moved and were re-measured over the full
# 710 frames: rel logit max 4.726e-6 (was 3.944e-6), rel bin max 8.483e-6
# (was 7.940e-6) — both inside the pins below, which keep their sizing.
LOGIT_REL_ACHIEVED = 8.0e-6   # worst measured 4.726e-6 (step 13500, gemm.rs)
BIN_REL_ACHIEVED = 1.6e-5     # worst measured 8.483e-6 (step 13500, gemm.rs)
# |value| <= 1 by construction (bin_centers span [-1, 1]), so this one is
# already scale-free and stays absolute. It tracks bin sharpness rather than
# bin magnitude: 9.537e-7 at step 5000, 2.205e-6 at step 6000, and on the
# identical-frame A/B 1.520e-6 (step 10000) / 1.669e-6 (step 13500) under
# candle. The gemm.rs kernel measured 2.682e-6 on the same corpus — over the
# old 2.5e-6 pin by 7%, which is the kernel's different rounding, not drift —
# so the pin is re-sized with the same ~2x headroom the two pins above carry.
# Defensible only because `mutation_check` re-confirms all 9 kills at this
# value; exceeding it is still a finding.
VALUE_TOL_ACHIEVED = 5.0e-6   # worst measured 2.682e-6 (step 13500, gemm.rs)

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
