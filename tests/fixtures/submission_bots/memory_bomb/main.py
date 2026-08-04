"""Grow RSS past an injected harness memory cap while still replying."""
from __future__ import annotations

import sys
import threading
import time


def _read_grid(stdin, H: int) -> None:
    for _ in range(H):
        stdin.readline()


def main() -> None:
    handshake = sys.stdin.readline()
    if not handshake:
        return
    _, H, _W = (int(x) for x in handshake.split())
    chunks: list[bytearray] = []

    def grow() -> None:
        while True:
            blob = bytearray(32 * 1024 * 1024)
            for i in range(0, len(blob), 4096):
                blob[i] = 1
            chunks.append(blob)
            time.sleep(0.01)

    threading.Thread(target=grow, name="rss-grow", daemon=True).start()

    while True:
        line = sys.stdin.readline()
        if not line:
            return
        _read_grid(sys.stdin, H)
        _read_grid(sys.stdin, H)
        _read_grid(sys.stdin, H)
        sys.stdout.write("1 0 0 0 0\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
