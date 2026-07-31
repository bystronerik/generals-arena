"""Tests for arena/tournaments/parallel.py job caps."""

from __future__ import annotations

import pytest

from arena.tournaments.parallel import cap_jobs, default_jobs, physical_cpu_count


def test_physical_cpu_count_positive():
    assert physical_cpu_count() >= 1


def test_default_jobs_matches_physical():
    assert default_jobs() == physical_cpu_count()


def test_cap_jobs_clamps_to_physical():
    n = physical_cpu_count()
    assert cap_jobs(1) == 1
    assert cap_jobs(n) == n
    assert cap_jobs(n + 100) == n


def test_cap_jobs_rejects_zero():
    with pytest.raises(ValueError, match="jobs must be >= 1"):
        cap_jobs(0)
