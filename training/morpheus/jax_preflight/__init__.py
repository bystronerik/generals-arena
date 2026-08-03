"""Modal JAX preflight: competition transition compile, throughput, CPU/GPU parity."""

from training.morpheus.jax_preflight.measure import run_preflight_on_device
from training.morpheus.jax_preflight.parity import compare_snapshots, snapshots_match
from training.morpheus.jax_preflight.report import (
    build_report,
    decide_pass,
    write_report,
)

__all__ = [
    "build_report",
    "compare_snapshots",
    "decide_pass",
    "run_preflight_on_device",
    "snapshots_match",
    "write_report",
]
