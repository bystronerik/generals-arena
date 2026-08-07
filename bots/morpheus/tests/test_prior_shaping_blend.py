"""Bounded prior-shaping blend (Part 17): parity, knob semantics, guards."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.morpheus

from action import PASS_INDEX
from shaping_boards import golden_cases
from tactics import (
    LEGACY_SHAPING_FLOOR_ABS,
    LEGACY_SHAPING_FLOOR_FRAC,
    LEGACY_SHAPING_LAMBDA,
    LEGACY_SHAPING_LOG_CLIP,
    apply_pre_contact_prior,
    blend_prior,
    heuristic_action_scores,
    mandatory_action_indices,
    play_mask,
    policy_ordered_candidates,
)

PARITY_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "shaping-parity.json"


def _legacy_shaped(prior, obs, mem, mask):
    return apply_pre_contact_prior(
        prior,
        obs,
        mem,
        mask=mask,
        lam=LEGACY_SHAPING_LAMBDA,
        log_clip=LEGACY_SHAPING_LOG_CLIP,
        floor_frac=LEGACY_SHAPING_FLOOR_FRAC,
        floor_abs=LEGACY_SHAPING_FLOOR_ABS,
    )


def test_refactor_reproduces_pre_part17_shaped_priors():
    """A1 parity: the split must not move a single shaped prior.

    The fixture was captured from the pre-refactor ``apply_pre_contact_prior``.
    The refactor normalizes in log space instead of multiplying, so equality is
    to floating-point tolerance, not bit-exact.
    """
    golden = json.loads(PARITY_FIXTURE.read_text(encoding="utf-8"))
    seen = set()
    for name, obs, mem, mask, prior in golden_cases():
        assert name in golden, f"missing golden case {name}"
        seen.add(name)
        case = golden[name]
        assert int(mask.sum()) == int(case["n_legal"])
        shaped = _legacy_shaped(prior, obs, mem, mask)
        expected = np.zeros(shaped.shape, dtype=np.float64)
        expected[np.asarray(case["indices"], dtype=np.int64)] = np.asarray(
            case["shaped"], dtype=np.float64
        )
        np.testing.assert_allclose(shaped, expected, rtol=1e-9, atol=1e-12)
    assert seen == set(golden), "golden fixture and board set disagree"


def test_scores_are_network_independent():
    """``heuristic_action_scores`` must not read the prior at all."""
    for _name, obs, mem, mask, _prior in golden_cases():
        a = heuristic_action_scores(obs, mem, mask)
        b = heuristic_action_scores(obs, mem, mask)
        np.testing.assert_array_equal(a, b)
        assert np.all(a[~mask] == 0.0)


def test_lambda_zero_is_the_renormalized_network_prior():
    for _name, obs, mem, mask, prior in golden_cases():
        shaped = apply_pre_contact_prior(prior, obs, mem, mask=mask, lam=0.0)
        expected = np.where(mask, prior, 0.0)
        expected = expected / float(expected.sum())
        np.testing.assert_allclose(shaped, expected, rtol=1e-12, atol=0.0)


def test_clip_bounds_every_action_shift():
    """No action may move more than ``exp(lam * log_clip)`` either way.

    Measured on the odds ratio against the network prior, which is what the
    softmax actually reweights; the normalizer cancels in the ratio of ratios.
    """
    log_clip = math.log(10.0)
    for _name, obs, mem, mask, prior in golden_cases():
        shaped = apply_pre_contact_prior(
            prior,
            obs,
            mem,
            mask=mask,
            lam=1.0,
            log_clip=log_clip,
            floor_frac=1e-3,
            floor_abs=0.0,
        )
        base = np.where(mask, prior, 0.0)
        floor = 1e-3 * float(base.max())
        base = np.where(mask, np.maximum(base, floor), 0.0)
        base = base / float(base.sum())
        ratio = shaped[mask] / base[mask]
        spread = float(ratio.max() / ratio.min())
        # Worst case is one action clipped to +log_clip and another to
        # -log_clip; the shared normalizer cannot widen that.
        assert spread <= math.exp(2.0 * log_clip) * (1.0 + 1e-9)


def test_zero_prior_action_cannot_outrank_network_top_beyond_clip():
    """A network zero stays below the network top unless the clip allows it."""
    log_clip = math.log(10.0)
    floor_frac = 1e-3
    n = PASS_INDEX + 1
    mask = np.zeros(n, dtype=bool)
    mask[:4] = True
    prior = np.zeros(n, dtype=np.float64)
    prior[0] = 0.7  # network top
    prior[1] = 0.3
    prior[2] = 0.0  # network zeroed
    prior[3] = 0.0
    scores = np.zeros(n, dtype=np.float64)
    scores[0] = 1.0
    scores[1] = 1.0
    scores[2] = 1e6  # heuristic screaming for a zeroed action
    scores[3] = 1.0
    shaped = blend_prior(
        prior,
        scores,
        mask,
        lam=1.0,
        log_clip=log_clip,
        floor_frac=floor_frac,
    )
    # The zeroed action is lifted only to floor_frac * max and nudged by at
    # most exp(log_clip): it cannot pass a top that holds 0.7 of the mass.
    assert shaped[2] < shaped[0]
    # The floor is a fraction of the network top, so the odds ratio against
    # that top is at most floor_frac widened by the two-sided clip.
    bound = floor_frac * math.exp(2.0 * log_clip)
    assert shaped[2] / shaped[0] <= bound * (1.0 + 1e-9)
    # Unbounded legacy shaping is exactly the failure mode being fixed.
    legacy = blend_prior(
        prior,
        scores,
        mask,
        lam=1.0,
        log_clip=math.inf,
        floor_frac=0.0,
        floor_abs=1e-6,
    )
    assert legacy[2] > legacy[0]


def test_mandatory_actions_are_root_candidates_for_every_lambda():
    """A4 guard: steer by inclusion, not by score — at any trust level."""
    for _name, obs, mem, mask, prior in golden_cases():
        mandatory = mandatory_action_indices(obs, mem, mask=mask)
        assert mandatory
        for lam in (0.0, 0.25, 0.5, 1.0):
            shaped = apply_pre_contact_prior(
                prior,
                obs,
                mem,
                mask=mask,
                lam=lam,
                log_clip=math.log(10.0),
                floor_frac=1e-3,
                floor_abs=0.0,
            )
            candidates = policy_ordered_candidates(
                shaped, mask, mandatory=mandatory, limit=int(mask.sum())
            )
            missing = [i for i in mandatory if i not in candidates]
            assert not missing, f"lambda={lam} dropped mandatory {missing}"


def test_blend_falls_back_to_prior_when_no_action_is_scored():
    n = PASS_INDEX + 1
    mask = np.zeros(n, dtype=bool)
    mask[:3] = True
    prior = np.zeros(n, dtype=np.float64)
    prior[:3] = (0.5, 0.3, 0.2)
    shaped = blend_prior(
        prior,
        np.zeros(n, dtype=np.float64),
        mask,
        lam=1.0,
        log_clip=math.log(10.0),
        floor_frac=1e-3,
    )
    np.testing.assert_allclose(shaped, prior, rtol=1e-12)


def test_blend_falls_back_to_uniform_when_prior_is_empty():
    n = PASS_INDEX + 1
    mask = np.zeros(n, dtype=bool)
    mask[:4] = True
    shaped = blend_prior(
        np.zeros(n, dtype=np.float64),
        np.zeros(n, dtype=np.float64),
        mask,
        lam=1.0,
        log_clip=math.log(10.0),
        floor_frac=1e-3,
    )
    assert shaped[mask] == pytest.approx(0.25)
    assert float(shaped.sum()) == pytest.approx(1.0)


def test_shaped_prior_never_leaves_the_play_mask():
    for _name, obs, mem, mask, prior in golden_cases():
        for lam in (0.0, 0.5, 1.0):
            shaped = apply_pre_contact_prior(prior, obs, mem, mask=mask, lam=lam)
            assert np.all(shaped[~mask] == 0.0)
            assert float(shaped.sum()) == pytest.approx(1.0)
            assert np.all(shaped >= 0.0)


def test_play_mask_is_the_default_mask():
    for _name, obs, mem, mask, prior in golden_cases():
        np.testing.assert_array_equal(np.asarray(play_mask(obs, mem), dtype=bool), mask)


def test_probe_fields_compare_chosen_against_the_unshaped_network_top():
    """Part 17 B: agreement is measured against the raw prior, not the shaped one."""
    from action import decode_action, encode_action
    from runtime import _prior_probe_fields

    n = PASS_INDEX + 1
    mask = np.zeros(n, dtype=bool)
    mask[:5] = True
    nn = np.zeros(n, dtype=np.float64)
    nn[:5] = (0.05, 0.5, 0.2, 0.15, 0.10)  # network top is index 1
    shaped = np.zeros(n, dtype=np.float64)
    shaped[:5] = (0.6, 0.1, 0.1, 0.1, 0.1)  # shaping moved the top to index 0

    chosen = tuple(int(x) for x in decode_action(0))
    fields = _prior_probe_fields(
        shaped,
        mask,
        chosen=chosen,
        policy_fallback=None,
        has_root_result=True,
        unshaped_prior=nn,
        enemy_visible=True,
    )
    assert fields["root_top_action"] == 0  # shaped top
    assert fields["nn_top_action"] == 1  # network top
    assert fields["nn_top_prior_milli"] == 500
    assert fields["chosen_matches_nn_top"] == 0
    assert fields["chosen_in_nn_top3"] == 0  # nn top3 is {1, 2, 3}
    assert fields["enemy_visible"] == 1

    agreeing = tuple(int(x) for x in decode_action(2))
    fields = _prior_probe_fields(
        shaped,
        mask,
        chosen=agreeing,
        policy_fallback=None,
        has_root_result=True,
        unshaped_prior=nn,
        enemy_visible=False,
    )
    assert fields["chosen_matches_nn_top"] == 0
    assert fields["chosen_in_nn_top3"] == 1
    assert fields["enemy_visible"] == 0
    del encode_action


def test_probe_fields_are_neutral_without_an_unshaped_prior():
    from action import decode_action
    from runtime import _prior_probe_fields

    n = PASS_INDEX + 1
    mask = np.zeros(n, dtype=bool)
    mask[:3] = True
    shaped = np.zeros(n, dtype=np.float64)
    shaped[:3] = (0.5, 0.3, 0.2)
    fields = _prior_probe_fields(
        shaped,
        mask,
        chosen=tuple(int(x) for x in decode_action(0)),
        policy_fallback=None,
        has_root_result=True,
    )
    assert fields["nn_top_action"] == -1
    assert fields["chosen_matches_nn_top"] == 0
    assert fields["chosen_in_nn_top3"] == 0
