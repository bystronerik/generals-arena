"""
Load and drive a rich per-frame capture module on a recorded match.

A *probe* (`arena.instrument.probes`) samples a handful of declared scalars per
turn into the telemetry schema. A *capture module* is the heavier sibling: it
writes whole arrays — observations, belief particles, network tensors, RNG draw
logs — for offline reuse. Nothing here enters `GameRecord.metrics`; a capture
is a side file, read by tooling, never by the rating fit.

The split exists because the two have opposite constraints. Probe extras are
typed, declared, and cheap enough to sample on every rated game. A capture is
untyped by design (its consumer is a specific analysis, not the schema),
megabytes per game, and only ever requested deliberately.

Like a probe, a capture module is loaded **by file path under a private module
name**, never by import: it must stay unreachable from inside a bot's source
closure so that unhashed code cannot influence a hashed program.

The module contract, all optional except `frame`:

    def install(agent) -> None   # once, after the agent is built
    def frame(agent, obs, action, turn) -> Mapping | None
    def close() -> None          # once, after the last turn

`install` is where a capture wraps mutable agent internals — the canonical case
being an RNG proxy that logs every draw, which is how the morpheus-rs parity
harness replays a Python decision in Rust (docs/bots/morpheus-rs/parity-corpus.md).

Frames are written **as they arrive**, not buffered like a probe trace. A trace
is a few hundred kilobytes and can be flushed at exit; a capture is megabytes of
tensors, and the matchup harness gives a closing agent three seconds before it
sends SIGKILL — long enough to close a stream, not to serialize a whole game.
The writes land after the move is already on the wire, so they sit outside every
clock the agent reports.

See docs/arena/trajectories.md for how captures ride the recorded-match path.
"""

from __future__ import annotations

import base64
import gzip
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping

import numpy as np

# Set to a capture module path to arm capture on every traced seat of a
# recorded match. Read by `arena.instrument.runner`, which derives each seat's
# output path from that seat's trace path, so the two seats never collide.
CAPTURE_MODULE_ENV = "ARENA_CAPTURE_MODULE"

# Appended to a seat's `--trace` path when capture is armed through the env.
CAPTURE_SUFFIX = ".capture.jsonl.gz"

# Marker key identifying an encoded ndarray inside a captured frame.
ARRAY_TAG = "__ndarray__"


class CaptureError(RuntimeError):
    """A capture module exists but could not be loaded or called."""


# --------------------------------------------------------------- array codec


def encode_array(array: np.ndarray) -> dict[str, Any]:
    """
    One ndarray as a JSON object: raw little-endian bytes, dtype, shape.

    Base64 of the buffer rather than nested lists. A 49x21x21 float32 root
    tensor is 86 KB of buffer against roughly 500 KB of JSON floats, and the
    buffer round-trips bit-exactly — which is the whole point, since the
    consumer is a parity check that compares against a Rust implementation.
    """
    arr = np.ascontiguousarray(array)
    if arr.dtype.byteorder == ">":
        arr = arr.astype(arr.dtype.newbyteorder("<"))
    return {
        ARRAY_TAG: base64.b64encode(arr.tobytes()).decode("ascii"),
        "dtype": arr.dtype.str,
        "shape": list(arr.shape),
    }


def decode_array(obj: Mapping[str, Any]) -> np.ndarray:
    """Inverse of :func:`encode_array`."""
    buf = base64.b64decode(obj[ARRAY_TAG])
    arr = np.frombuffer(buf, dtype=np.dtype(obj["dtype"]))
    return arr.reshape(tuple(int(n) for n in obj["shape"]))


def _default(obj: Any) -> Any:
    """JSON fallback: ndarrays encode, numpy scalars degrade to Python."""
    if isinstance(obj, np.ndarray):
        return encode_array(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"cannot capture {type(obj).__name__}")


def revive(obj: Any) -> Any:
    """Walk a decoded frame, turning encoded arrays back into ndarrays."""
    if isinstance(obj, dict):
        if ARRAY_TAG in obj:
            return decode_array(obj)
        return {k: revive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [revive(v) for v in obj]
    return obj


# ------------------------------------------------------------------- writing


class CaptureSink:
    """
    Streaming gzipped-JSON-lines writer for one seat's capture.

    Opened on the first frame, so a capture module that recognizes nothing
    (the other seat's bot, say) leaves no empty file behind to be mistaken for
    a game that produced no frames.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.written = 0
        self._handle = None

    def write(self, frame: Mapping[str, Any]) -> None:
        if self._handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = gzip.open(self.path, "wt", encoding="utf-8")
        self._handle.write(json.dumps(frame, separators=(",", ":"), default=_default))
        self._handle.write("\n")
        # Flushed per frame, which costs a sync point in the deflate stream and
        # buys the only thing that matters when the harness kills a closing
        # agent: everything written so far is on disk and decodable. Without
        # it a SIGKILL leaves a buffer's worth of frames lost and the file
        # unreadable from the last block onward — not a short capture, a
        # broken one.
        self._handle.flush()
        self.written += 1

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def write_frames(frames: Iterable[Mapping[str, Any]], out_path: Path) -> int:
    """Write frames as gzipped JSON lines in one shot. Returns the count."""
    sink = CaptureSink(out_path)
    try:
        for frame in frames:
            sink.write(frame)
    finally:
        sink.close()
    return sink.written


def read_frames(
    path: Path, *, arrays: bool = True, allow_truncated: bool = False
) -> list[dict[str, Any]]:
    """
    Read a capture file back.

    `arrays=False` leaves encoded arrays as their JSON objects, which is what
    a caller that only wants scalars should pass — decoding every tensor to
    count move times is the slow way to read a corpus.

    A truncated file raises by default, naming how many frames were readable.
    A capture ends when the seat's process ends, so truncation means the
    process died mid-game — which a corpus report has to say out loud rather
    than quietly analysing a short game as if it were a whole one.
    """
    frames: list[dict[str, Any]] = []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                frame = json.loads(line)
                frames.append(revive(frame) if arrays else frame)
    except (EOFError, gzip.BadGzipFile, json.JSONDecodeError) as exc:
        if not allow_truncated:
            raise CaptureError(
                f"{path} is truncated after {len(frames)} frame(s): {exc}"
            ) from exc
    return frames


# ------------------------------------------------------------------- loading


def load_capture(module_path: Path) -> ModuleType:
    """Import a capture module by file path, under a name no import can reach."""
    path = Path(module_path)
    if not path.is_file():
        raise CaptureError(f"no capture module at {path}")

    spec = importlib.util.spec_from_file_location(
        f"_arena_capture_{path.stem.replace('-', '_')}", path
    )
    if spec is None or spec.loader is None:
        raise CaptureError(f"cannot load capture module at {path}")
    module = importlib.util.module_from_spec(spec)
    # Registered before execution, same reason as probes: class bodies look
    # their defining module up in `sys.modules` while they run.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "frame"):
        raise CaptureError(f"{path} defines no frame(agent, obs, action, turn)")
    return module


def capture_module_from_env() -> Path | None:
    """The armed capture module, or None when capture is off."""
    value = os.environ.get(CAPTURE_MODULE_ENV, "").strip()
    return Path(value) if value else None


def capture_install(module: ModuleType | None, agent: Any) -> None:
    """Give the capture module its one shot at wrapping agent internals."""
    if module is None:
        return
    install = getattr(module, "install", None)
    if install is not None:
        install(agent)


def capture_frame(
    module: ModuleType | None, agent: Any, obs: Any, action: Any, turn: int
) -> dict[str, Any] | None:
    """
    Sample one frame. `None` means "this turn is not part of the corpus".

    Errors are **not** swallowed, for the same reason probe errors are not: a
    capture that silently drops frames after a refactor produces a corpus that
    looks complete and is not, and every parity claim built on it is wrong.
    """
    if module is None:
        return None
    captured = module.frame(agent, obs, action, turn)
    if captured is None:
        return None
    if not isinstance(captured, Mapping):
        raise CaptureError(
            f"{module.__file__} returned {type(captured).__name__}, expected Mapping"
        )
    return dict(captured)


def capture_close(module: ModuleType | None) -> None:
    """Let the capture module undo whatever `install` wrapped."""
    if module is None:
        return
    close = getattr(module, "close", None)
    if close is not None:
        close()
