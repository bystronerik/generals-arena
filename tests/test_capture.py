"""
Rich per-frame capture: the codec, the module contract, and non-interference.

A capture is heavier than a probe and reaches further into a bot — it wraps the
agent's RNG. What makes that acceptable is that it observes and never steers,
so the tests that matter here are the ones that pin *equivalence*: the same
observations must produce the same action stream whether or not a capture is
running, and every recorded array must come back bit-identical.
"""
from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from arena.instrument import capture as cap

REPO_ROOT = Path(__file__).resolve().parent.parent


# --- array codec ------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype", ["int8", "uint8", "int32", "int64", "float32", "float64", "bool"]
)
def test_arrays_round_trip_bit_exactly(dtype):
    """
    Not "close" — identical. The consumer is a parity check against a Rust
    implementation, so a codec that rounded would hide exactly the class of
    bug the corpus exists to find.
    """
    array = (np.arange(24).reshape(2, 3, 4) % 7).astype(dtype)
    back = cap.decode_array(cap.encode_array(array))
    assert back.dtype == array.dtype
    assert back.shape == array.shape
    assert np.array_equal(back, array)


def test_float_payloads_survive_exactly():
    values = np.array([0.1, -1e-8, np.pi, 1e30, 0.0, -0.0], dtype=np.float64)
    back = cap.decode_array(cap.encode_array(values))
    assert back.tobytes() == values.tobytes()


def test_non_contiguous_arrays_encode_their_logical_values():
    """A transposed view must record what it *is*, not its parent's buffer."""
    view = np.arange(12, dtype=np.int32).reshape(3, 4).T
    back = cap.decode_array(cap.encode_array(view))
    assert np.array_equal(back, view)


def test_unsupported_payloads_raise_rather_than_stringify():
    with pytest.raises(TypeError):
        cap.write_frames([{"x": object()}], Path("/dev/null"))


# --- sink -------------------------------------------------------------------


def test_frames_round_trip_through_the_sink(tmp_path):
    out = tmp_path / "c.jsonl.gz"
    frames = [
        {"t": 1, "a": np.arange(3, dtype=np.float32), "s": 5},
        {"t": 2, "a": np.zeros((2, 2), dtype=bool), "s": 6},
    ]
    assert cap.write_frames(frames, out) == 2

    read = cap.read_frames(out)
    assert [f["t"] for f in read] == [1, 2]
    assert np.array_equal(read[0]["a"], frames[0]["a"])
    assert read[1]["a"].dtype == np.bool_


def test_reading_without_arrays_leaves_them_encoded(tmp_path):
    """The report path reads scalars from megabytes of tensors; it must not decode."""
    out = tmp_path / "c.jsonl.gz"
    cap.write_frames([{"t": 1, "a": np.arange(3, dtype=np.float32)}], out)
    raw = cap.read_frames(out, arrays=False)[0]
    assert isinstance(raw["a"], dict) and cap.ARRAY_TAG in raw["a"]


def test_a_sink_that_is_never_written_leaves_no_file(tmp_path):
    """
    An empty file would read as "this game produced no frames", which is a
    different claim from "no capture ran here".
    """
    sink = cap.CaptureSink(tmp_path / "unused.jsonl.gz")
    sink.close()
    assert not (tmp_path / "unused.jsonl.gz").exists()


def test_frames_are_on_disk_and_readable_before_the_sink_closes(tmp_path):
    """
    Written-so-far frames survive a process that never gets to close cleanly —
    the matchup harness allows three seconds before SIGKILL. Buffered gzip
    would leave a file that is not merely short but undecodable.
    """
    out = tmp_path / "c.jsonl.gz"
    sink = cap.CaptureSink(out)
    sink.write({"t": 1})
    sink.write({"t": 2})
    assert out.stat().st_size > 0
    assert [f["t"] for f in cap.read_frames(out, allow_truncated=True)] == [1, 2]
    sink.close()
    assert len(cap.read_frames(out)) == 2


def test_a_truncated_capture_raises_and_names_what_survived(tmp_path):
    """
    Truncation means the seat's process died mid-game. A report that quietly
    analysed the prefix would treat half a game as a whole one.
    """
    out = tmp_path / "c.jsonl.gz"
    sink = cap.CaptureSink(out)
    sink.write({"t": 1})
    sink.write({"t": 2})
    out.write_bytes(out.read_bytes()[:-4])  # a killed process, mid-block

    with pytest.raises(cap.CaptureError, match="truncated"):
        cap.read_frames(out)
    assert len(cap.read_frames(out, allow_truncated=True)) >= 1


# --- module contract --------------------------------------------------------


def _module(tmp_path: Path, body: str, name: str = "cap_mod.py") -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def test_a_missing_capture_module_raises(tmp_path):
    with pytest.raises(cap.CaptureError):
        cap.load_capture(tmp_path / "nope.py")


def test_a_module_without_frame_is_rejected_at_load(tmp_path):
    module = _module(tmp_path, "def install(agent):\n    pass\n")
    with pytest.raises(cap.CaptureError):
        cap.load_capture(module)


def test_a_frame_that_is_not_a_mapping_raises(tmp_path):
    """
    Not swallowed. A capture that silently drops frames yields a corpus that
    looks complete and is not, and every parity claim built on it is wrong.
    """
    module = cap.load_capture(
        _module(tmp_path, "def frame(agent, obs, action, turn):\n    return [1, 2]\n")
    )
    with pytest.raises(cap.CaptureError):
        cap.capture_frame(module, None, None, None, 1)


def test_a_frame_of_none_means_skip_this_turn(tmp_path):
    module = cap.load_capture(
        _module(tmp_path, "def frame(agent, obs, action, turn):\n    return None\n")
    )
    assert cap.capture_frame(module, None, None, None, 1) is None


def test_install_and_close_are_optional(tmp_path):
    module = cap.load_capture(
        _module(tmp_path, "def frame(agent, obs, action, turn):\n    return {}\n")
    )
    cap.capture_install(module, object())
    cap.capture_close(module)


def test_a_capture_module_is_not_importable_by_name(tmp_path):
    """
    Same rule as probes: unhashed code must stay unreachable from the hashed
    program. A capture loaded under an importable name could be picked up by a
    bot's own `import`, which would put it inside the closure it must stay out of.
    """
    module = cap.load_capture(
        _module(tmp_path, "def frame(agent, obs, action, turn):\n    return {}\n")
    )
    assert module.__name__.startswith("_arena_capture_")
    assert "cap_mod" not in sys.modules


def test_capture_is_armed_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.delenv(cap.CAPTURE_MODULE_ENV, raising=False)
    assert cap.capture_module_from_env() is None
    monkeypatch.setenv(cap.CAPTURE_MODULE_ENV, str(tmp_path / "m.py"))
    assert cap.capture_module_from_env() == tmp_path / "m.py"


# --- non-interference -------------------------------------------------------

STUB_AGENT = """
class Agent:
    def __init__(self, player_id, H, W):
        self.player_id, self.H, self.W = player_id, H, W
        self.calls = 0

    def act(self, obs):
        self.calls += 1
        return (0, obs.turn % self.H, 0, obs.turn % 4, 0)
"""

STUB_CAPTURE = """
def install(agent):
    agent.installed = True

def frame(agent, obs, action, turn):
    return {"t": turn, "turn": obs.turn, "action": list(action),
            "installed": getattr(agent, "installed", False)}
"""


def _stub_bot(tmp_path: Path) -> Path:
    bot = tmp_path / "stub"
    bot.mkdir()
    (bot / "agent.py").write_text(STUB_AGENT, encoding="utf-8")
    return bot


def _frames_stdin(turns: int, H: int, W: int) -> str:
    """A scripted match: scalars line plus three H-row grids, per turn."""
    grid = "\n".join(" ".join("0" for _ in range(W)) for _ in range(H))
    return "".join(f"{t} 1 1 1 1\n{grid}\n{grid}\n{grid}\n" for t in range(turns))


def _run_seat(bot: Path, stdin: str, cwd: Path, *args: str) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "arena.instrument.runner", str(bot), *args],
        input=f"0 4 4\n{stdin}",
        capture_output=True,
        text=True,
        cwd=str(cwd),
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.mark.slow
def test_capture_does_not_change_the_action_stream(tmp_path):
    """
    The equivalence the whole harness rests on: same observations in, same
    bytes out, capture or no capture. A capture that perturbed play would
    produce a corpus describing a bot that never played.
    """
    bot = _stub_bot(tmp_path)
    module = _module(tmp_path, STUB_CAPTURE)
    stdin = _frames_stdin(6, 4, 4)

    plain = _run_seat(bot, stdin, REPO_ROOT, "--trace", str(tmp_path / "t1.jsonl"))
    captured = _run_seat(
        bot,
        stdin,
        REPO_ROOT,
        "--trace",
        str(tmp_path / "t2.jsonl"),
        "--capture",
        str(module),
        "--capture-out",
        str(tmp_path / "c.jsonl.gz"),
    )
    assert plain == captured
    assert plain.strip().splitlines()

    frames = cap.read_frames(tmp_path / "c.jsonl.gz")
    assert [f["t"] for f in frames] == [1, 2, 3, 4, 5, 6]
    # `install` ran before the first move, not after it: an RNG proxy attached
    # a turn late would record a stream that starts in the wrong place.
    assert all(f["installed"] for f in frames)
    assert [f["action"] for f in frames] == [
        [int(x) for x in line.split()] for line in plain.strip().splitlines()
    ]


@pytest.mark.slow
def test_capture_out_defaults_beside_the_trace(tmp_path):
    bot = _stub_bot(tmp_path)
    module = _module(tmp_path, STUB_CAPTURE)
    trace = tmp_path / "seat.jsonl"
    _run_seat(
        bot,
        _frames_stdin(2, 4, 4),
        REPO_ROOT,
        "--trace",
        str(trace),
        "--capture",
        str(module),
    )
    assert (tmp_path / f"seat.jsonl{cap.CAPTURE_SUFFIX}").is_file()


def test_a_capture_without_a_trace_has_nowhere_to_write(tmp_path):
    """
    Env-armed capture derives its path from the trace path, so an untraced
    seat is simply not captured — rather than guessing a filename.
    """
    from arena.instrument.runner import _capture_target

    assert _capture_target(tmp_path / "m.py", None, None) is None
    assert _capture_target(None, tmp_path / "t.jsonl", None) is None
    module, out = _capture_target(tmp_path / "m.py", tmp_path / "t.jsonl", None)
    assert out.name.endswith(cap.CAPTURE_SUFFIX)


# --- harness wiring ---------------------------------------------------------


def test_the_spawn_command_carries_the_capture_destination(tmp_path):
    from arena.matches import loop
    from arena.records.fingerprint import BOTS_DIR

    run_sh = BOTS_DIR / "metro" / "run.sh"
    command, _ = loop.agent_command(run_sh, tmp_path / "a.jsonl", tmp_path / "c.gz")
    assert command[-2:] == ["--capture-out", str(tmp_path / "c.gz")]


def test_capture_destinations_are_per_seat(tmp_path):
    from arena.records.trajectories import capture_path

    a = capture_path("g1", "a", tmp_path)
    b = capture_path("g1", "b", tmp_path)
    assert a != b
    assert a.name.endswith(".capture.a.jsonl.gz")
    with pytest.raises(ValueError):
        capture_path("g1", "c", tmp_path)


def test_a_capture_file_is_gzipped_json_lines(tmp_path):
    """The format Rust-side tooling reads; asserted rather than assumed."""
    out = tmp_path / "c.jsonl.gz"
    cap.write_frames([{"t": 1}, {"t": 2}], out)
    with gzip.open(out, "rt", encoding="utf-8") as handle:
        assert [json.loads(line)["t"] for line in handle] == [1, 2]
