"""Shared plumbing for the unclejoe parity drivers (port-plan §6).

The binary speaks a flat whitespace-separated integer stream on
stdin/stdout; floats travel as their f32 bit patterns. Every layout here is
positional and mirrored in `src/parity.rs`.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np

BOT_DIR = Path(__file__).resolve().parent.parent
# JOE_RS_BIN lets tools/mutation_check.py aim the drivers at the mutation
# profile's binary.
BINARY = Path(os.environ.get("JOE_RS_BIN", BOT_DIR / "target" / "release" / "unclejoe"))
CORPUS = BOT_DIR.parent.parent / "data" / "joe" / "unclejoe-parity" / "games"
SMOKE = Path(__file__).resolve().parent / "fixtures"

PAD = 21
CELLS = PAD * PAD
HISTORY = 7
N_CHANNELS = 39
TEMPORAL_WINDOW = 512
N_LOGITS = 10 * CELLS

# The AugmentedObsState field order — also the stream serialization order.
STATE_FIELDS = (
    "army_stack", "enemy_stack", "last_army", "last_enemy_army",
    "castles", "generals", "mountains", "seen", "enemy_seen",
    "last_enemy_army_seen_value", "last_enemy_army_seen_timestep",
    "opponent_army_history", "opponent_land_history", "temporal_step",
)
STATE_BOOL_FIELDS = {"castles", "generals", "mountains", "seen", "enemy_seen"}


def f32_bits(arr) -> np.ndarray:
    """f32 array -> uint32 bit patterns, flattened."""
    return np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32).ravel()


def bits_f32(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.uint32).view(np.float32)


def encode_state(state: dict) -> list[int]:
    """Serialize one obs state in the stream order."""
    out: list[int] = []
    for field in STATE_FIELDS:
        v = state[field]
        if field == "temporal_step":
            out.append(int(v))
        elif field in STATE_BOOL_FIELDS:
            out.extend(np.asarray(v, dtype=np.int64).ravel().tolist())
        else:
            out.extend(f32_bits(v).tolist())
    return out


def state_stream_len() -> int:
    return 2 * HISTORY * CELLS + 2 * CELLS + 5 * CELLS + 2 * CELLS + 2 * TEMPORAL_WINDOW + 1


def decode_state(values: np.ndarray) -> dict:
    """Inverse of `encode_state`."""
    out = {}
    pos = 0

    def take(n):
        nonlocal pos
        chunk = values[pos:pos + n]
        assert len(chunk) == n, "truncated state in output stream"
        pos += n
        return chunk

    for field in STATE_FIELDS:
        if field == "temporal_step":
            out[field] = int(take(1)[0])
        elif field in STATE_BOOL_FIELDS:
            out[field] = take(CELLS).astype(bool).reshape(PAD, PAD)
        elif field.endswith("_history"):
            out[field] = bits_f32(take(TEMPORAL_WINDOW))
        elif field.endswith("_stack"):
            out[field] = bits_f32(take(HISTORY * CELLS)).reshape(HISTORY, PAD, PAD)
        else:
            out[field] = bits_f32(take(CELLS)).reshape(PAD, PAD)
    assert pos == len(values)
    return out


def parse_in_log(path: Path):
    """Handshake + frames, the same parse as `tools/capture_fixtures.py`."""
    lines = path.read_text().splitlines()
    player_id, H, W = (int(x) for x in lines[0].split())
    frames = []
    pos = 1
    frame_len = 1 + 3 * H
    while pos < len(lines):
        chunk = lines[pos:pos + frame_len]
        if len(chunk) < frame_len:
            break
        scalars = [int(x) for x in chunk[0].split()]
        grids = np.array(
            [[int(x) for x in row.split()] for row in chunk[1:]],
            dtype=np.int64).reshape(3, H, W)
        frames.append((scalars, grids))
        pos += frame_len
    return player_id, H, W, frames


def encode_frame(scalars, grids) -> list[int]:
    return list(scalars) + grids.ravel().tolist()


def run_surface(surface: str, cases: list[list[int]]) -> np.ndarray:
    """Feed cases to `unclejoe parity <surface>`, return the output ints."""
    stream: list[str] = [str(len(cases))]
    for case in cases:
        stream.extend(str(v) for v in case)
    proc = subprocess.run(
        [str(BINARY), "parity", surface],
        input="\n".join(stream).encode(),
        capture_output=True,
        env={"JOE_RS_ARTIFACT": str(BOT_DIR / "artifact"), "PATH": "/usr/bin:/bin"},
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"unclejoe parity {surface} failed: {proc.stderr.decode()[-2000:]}")
    return np.array(proc.stdout.split(), dtype=np.int64)


def ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-element ULP distance between two f32 arrays."""
    ai = np.ascontiguousarray(a, dtype=np.float32).view(np.int32).astype(np.int64)
    bi = np.ascontiguousarray(b, dtype=np.float32).view(np.int32).astype(np.int64)
    # Map to a monotonic integer line (sign-magnitude -> offset).
    ai = np.where(ai < 0, np.int64(-0x80000000) - ai, ai)
    bi = np.where(bi < 0, np.int64(-0x80000000) - bi, bi)
    return np.abs(ai - bi)


def corpus_games() -> list[Path]:
    """Full corpus when present, else the committed smoke slice.
    `JOE_RS_PARITY_SMOKE=1` forces the smoke slice (the mutation checker's
    scope: one rebuild per mutation is the cost, not the corpus size)."""
    if os.environ.get("JOE_RS_PARITY_SMOKE") != "1" and CORPUS.is_dir():
        games = sorted(CORPUS.glob("*.npz"))
        if games:
            return games
    return sorted(SMOKE.glob("*.npz"))
