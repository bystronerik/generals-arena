"""Unit tests for smoothed train-loss metrics."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.morpheus

from training.morpheus.trainer.metrics import DEFAULT_EMA_BETA, LossSmoother


def test_window_mean_and_ema_closed_form():
    smoother = LossSmoother(ema_beta=0.5)
    values = [10.0, 0.0, 4.0]
    policies = [1.0, 3.0, 5.0]
    wdls = [2.0, 2.0, 8.0]

    snaps = []
    for loss, policy, wdl in zip(values, policies, wdls, strict=True):
        snaps.append(smoother.update(loss, {"policy": policy, "wdl": wdl}))

    # Window mean over all three steps (no reset).
    assert snaps[-1].window_count == 3
    assert snaps[-1].loss_avg == pytest.approx(sum(values) / 3)
    assert snaps[-1].policy_avg == pytest.approx(sum(policies) / 3)
    assert snaps[-1].wdl_avg == pytest.approx(sum(wdls) / 3)
    assert snaps[-1].loss_inst == pytest.approx(4.0)

    # EMA with beta=0.5: e0=v0; e_t = 0.5*e_{t-1} + 0.5*v_t
    ema_loss = values[0]
    for v in values[1:]:
        ema_loss = 0.5 * ema_loss + 0.5 * v
    ema_policy = policies[0]
    for v in policies[1:]:
        ema_policy = 0.5 * ema_policy + 0.5 * v
    ema_wdl = wdls[0]
    for v in wdls[1:]:
        ema_wdl = 0.5 * ema_wdl + 0.5 * v
    assert snaps[-1].loss_ema == pytest.approx(ema_loss)
    assert snaps[-1].policy_ema == pytest.approx(ema_policy)
    assert snaps[-1].wdl_ema == pytest.approx(ema_wdl)


def test_reset_window_keeps_ema():
    smoother = LossSmoother(ema_beta=DEFAULT_EMA_BETA)
    first = smoother.update(8.0, {"policy": 4.0, "wdl": 2.0})
    smoother.reset_window()
    assert smoother.snapshot().window_count == 0
    second = smoother.update(0.0, {"policy": 0.0, "wdl": 0.0})
    assert second.window_count == 1
    assert second.loss_avg == pytest.approx(0.0)
    # EMA continues from first observation, not reset.
    expected = DEFAULT_EMA_BETA * first.loss_ema + (1.0 - DEFAULT_EMA_BETA) * 0.0
    assert second.loss_ema == pytest.approx(expected)
    assert second.loss_inst == pytest.approx(0.0)


def test_log_line_shape():
    smoother = LossSmoother()
    snap = smoother.update(1.25, {"policy": 0.5, "wdl": 0.75})
    line = snap.log_line(step=100)
    assert line.startswith("step=100 ")
    assert "loss_avg=" in line
    assert "loss_ema=" in line
    assert "loss_inst=" in line
    assert "policy_avg=" in line
    assert "policy_ema=" in line
    assert "wdl_avg=" in line
    assert "wdl_ema=" in line
    # No noisy auxiliary heads on the main line.
    assert "enemy_general" not in line
