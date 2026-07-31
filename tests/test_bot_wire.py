"""The shared stdio wire loop: pure protocol, and silent on stderr."""
from __future__ import annotations

import io

import pytest

from _common.wire import Observation, run_stdio

H = W = 2


class _EchoAgent:
    """Plays a move derived from the observation, so frames are traceable."""

    def __init__(self, player_id: int, H: int, W: int):
        self.player_id = player_id
        self.seen: list[Observation] = []

    def act(self, obs: Observation):
        self.seen.append(obs)
        return (0, 0, 0, obs.turn % 4, 0)


def _frames(turns: int) -> str:
    """Handshake plus `turns` observation frames, in wire order."""
    grid = "\n".join(" ".join("0" for _ in range(W)) for _ in range(H))
    out = [f"0 {H} {W}"]
    for turn in range(1, turns + 1):
        out.append(f"{turn} {turn} {turn * 2} {turn} {turn * 3}")
        out.extend([grid, grid, grid])
    return "\n".join(out) + "\n"


@pytest.fixture
def wire(monkeypatch):
    """Run the loop over canned stdin, returning (stdout, stderr)."""

    def run(stdin_text: str) -> tuple[str, str]:
        stdout, stderr = io.StringIO(), io.StringIO()
        monkeypatch.setattr("sys.stdin", io.StringIO(stdin_text))
        monkeypatch.setattr("sys.stdout", stdout)
        monkeypatch.setattr("sys.stderr", stderr)
        run_stdio(_EchoAgent)
        return stdout.getvalue(), stderr.getvalue()

    return run


def test_one_action_line_per_observation(wire):
    stdout, _ = wire(_frames(3))
    assert stdout == "0 0 0 1 0\n0 0 0 2 0\n0 0 0 3 0\n"


def test_the_loop_writes_nothing_to_stderr(wire):
    """
    The bundled bot is game logic only.

    Anything on stderr here would be instrumentation inside the closure, which
    is what the recorder work removed: introspection lives in
    `arena.instrument.runner` plus `bots/<name>/probe.py`, outside every hash
    and every bundle.
    """
    _, stderr = wire(_frames(5))
    assert stderr == ""


def test_eof_ends_the_loop_silently(wire):
    """EOF used to be when the telemetry line was emitted. Now it just returns."""
    stdout, stderr = wire(_frames(2))
    assert stdout.count("\n") == 2
    assert stderr == ""


def test_no_handshake_is_a_clean_exit(wire):
    assert wire("") == ("", "")


def test_observation_scalars_are_parsed_in_wire_order(monkeypatch):
    seen: list[Observation] = []

    class _Capture(_EchoAgent):
        def act(self, obs):
            seen.append(obs)
            return (1, 0, 0, 0, 0)

    monkeypatch.setattr("sys.stdin", io.StringIO(_frames(1)))
    monkeypatch.setattr("sys.stdout", io.StringIO())
    run_stdio(_Capture)

    obs = seen[0]
    assert (obs.turn, obs.my_land, obs.my_army, obs.opp_land, obs.opp_army) == (1, 1, 2, 1, 3)
    assert obs.H == H and obs.W == W
    assert obs.type_grid == obs.owner_grid == obs.army_grid == [[0, 0], [0, 0]]
