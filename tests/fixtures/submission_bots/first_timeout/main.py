"""Sleep past the first-reply deadline before writing."""
from __future__ import annotations

import sys
import time


def _read_grid(stdin, H: int) -> None:
    for _ in range(H):
        stdin.readline()


def main() -> None:
    handshake = sys.stdin.readline()
    if not handshake:
        return
    _, H, _W = (int(x) for x in handshake.split())
    first = True
    while True:
        line = sys.stdin.readline()
        if not line:
            return
        _read_grid(sys.stdin, H)
        _read_grid(sys.stdin, H)
        _read_grid(sys.stdin, H)
        if first:
            time.sleep(0.5)
            first = False
        sys.stdout.write("1 0 0 0 0\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
