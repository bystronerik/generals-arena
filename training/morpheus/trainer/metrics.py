"""Smoothed train-loss metrics for readable logging.

Tracks an EMA and a resettable window mean for the total loss and the primary
heads (policy, wdl). Instantaneous values stay available for honesty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


DEFAULT_EMA_BETA = 0.99
PRIMARY_TERMS = ("policy", "wdl")


@dataclass(frozen=True)
class SmoothedSnapshot:
    """Point-in-time view of smoothed totals and primary terms."""

    loss_inst: float
    loss_avg: float
    loss_ema: float
    policy_inst: float
    policy_avg: float
    policy_ema: float
    wdl_inst: float
    wdl_avg: float
    wdl_ema: float
    window_count: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "loss_inst": self.loss_inst,
            "loss_avg": self.loss_avg,
            "loss_ema": self.loss_ema,
            "policy_inst": self.policy_inst,
            "policy_avg": self.policy_avg,
            "policy_ema": self.policy_ema,
            "wdl_inst": self.wdl_inst,
            "wdl_avg": self.wdl_avg,
            "wdl_ema": self.wdl_ema,
            "window_count": self.window_count,
        }

    def log_line(self, *, step: int) -> str:
        return (
            f"step={step} "
            f"loss_avg={self.loss_avg:.6g} loss_ema={self.loss_ema:.6g} "
            f"loss_inst={self.loss_inst:.6g} "
            f"policy_avg={self.policy_avg:.6g} policy_ema={self.policy_ema:.6g} "
            f"wdl_avg={self.wdl_avg:.6g} wdl_ema={self.wdl_ema:.6g}"
        )


class LossSmoother:
    """Accumulate per-step losses; expose window mean + EMA."""

    def __init__(self, *, ema_beta: float = DEFAULT_EMA_BETA) -> None:
        if not 0.0 <= float(ema_beta) < 1.0:
            raise ValueError(f"ema_beta must be in [0, 1), got {ema_beta}")
        self.ema_beta = float(ema_beta)
        self._ema: dict[str, float] = {}
        self._window_sum: dict[str, float] = {
            "loss": 0.0,
            "policy": 0.0,
            "wdl": 0.0,
        }
        self._window_count = 0
        self._last_inst: dict[str, float] = {
            "loss": 0.0,
            "policy": 0.0,
            "wdl": 0.0,
        }

    def update(self, loss: float, terms: Mapping[str, float]) -> SmoothedSnapshot:
        """Record one step; return the current smoothed snapshot."""
        policy = float(terms.get("policy", 0.0))
        wdl = float(terms.get("wdl", 0.0))
        values = {
            "loss": float(loss),
            "policy": policy,
            "wdl": wdl,
        }
        self._last_inst = dict(values)
        beta = self.ema_beta
        for key, value in values.items():
            if key not in self._ema:
                self._ema[key] = value
            else:
                self._ema[key] = beta * self._ema[key] + (1.0 - beta) * value
            self._window_sum[key] = float(self._window_sum.get(key, 0.0)) + value
        self._window_count += 1
        return self.snapshot()

    def snapshot(self) -> SmoothedSnapshot:
        count = max(1, self._window_count)
        avg = {k: self._window_sum[k] / count for k in ("loss", "policy", "wdl")}
        return SmoothedSnapshot(
            loss_inst=self._last_inst["loss"],
            loss_avg=avg["loss"],
            loss_ema=float(self._ema.get("loss", self._last_inst["loss"])),
            policy_inst=self._last_inst["policy"],
            policy_avg=avg["policy"],
            policy_ema=float(self._ema.get("policy", self._last_inst["policy"])),
            wdl_inst=self._last_inst["wdl"],
            wdl_avg=avg["wdl"],
            wdl_ema=float(self._ema.get("wdl", self._last_inst["wdl"])),
            window_count=int(self._window_count),
        )

    def reset_window(self) -> None:
        """Clear the window accumulator after a log line (EMA is kept)."""
        self._window_sum = {"loss": 0.0, "policy": 0.0, "wdl": 0.0}
        self._window_count = 0
