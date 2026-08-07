"""Part 01 protocol shell — legal reply, clean EOF, no stderr protocol data."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.morpheus

BOT_DIR = Path(__file__).resolve().parents[1]
RUN_SH = BOT_DIR / "run.sh"

H = W = 2


def _frames(turns: int) -> str:
    """Handshake plus `turns` observation frames, in wire order."""
    grid = "\n".join(" ".join("0" for _ in range(W)) for _ in range(H))
    out = [f"0 {H} {W}"]
    for turn in range(1, turns + 1):
        out.append(f"{turn} {turn} {turn * 2} {turn} {turn * 3}")
        out.extend([grid, grid, grid])
    return "\n".join(out) + "\n"


def _run_shell(stdin_text: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHON"] = sys.executable
    return subprocess.run(
        [str(RUN_SH)],
        input=stdin_text,
        capture_output=True,
        text=True,
        cwd=str(BOT_DIR),
        env=env,
        timeout=30,
        check=False,
    )


def _assert_legal_line(line: str) -> None:
    parts = line.split()
    assert len(parts) == 5
    assert all(p.lstrip("-").isdigit() for p in parts)


def test_agent_returns_legal_five_tuple():
    from agent import Agent
    from _common.wire import Observation

    agent = Agent(player_id=0, H=H, W=W)
    obs = Observation(
        H=H,
        W=W,
        turn=7,
        my_land=3,
        my_army=12,
        opp_land=2,
        opp_army=9,
        type_grid=[[1, 1], [1, 4]],
        owner_grid=[[1, 0], [0, 2]],
        army_grid=[[5, 0], [0, 4]],
    )
    action = agent.act(obs)
    assert isinstance(action, tuple) and len(action) == 5
    assert all(isinstance(x, int) for x in action)


def test_one_legal_line_per_observation():
    result = _run_shell(_frames(3))
    assert result.returncode == 0, result.stderr
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    assert len(lines) == 3
    for line in lines:
        _assert_legal_line(line)


def test_eof_exits_status_zero_with_empty_stderr():
    result = _run_shell(_frames(2))
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("\n") == 2
    # Torch may print a Quantizer deprecation warning on stderr; that is not
    # protocol data.
    for line in result.stderr.splitlines():
        assert "Quantizer" in line or "quantize" in line or not line.strip()


def test_no_handshake_is_a_clean_exit():
    result = _run_shell("")
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    for line in result.stderr.splitlines():
        assert "Quantizer" in line or "quantize" in line or not line.strip()
