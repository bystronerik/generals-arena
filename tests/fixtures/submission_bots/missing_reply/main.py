"""Write an empty reply line (missing five integers)."""
from __future__ import annotations

import sys


def _read_grid(stdin, H: int) -> None:
    for _ in range(H):
        stdin.readline()


def main() -> None:
    handshake = sys.stdin.readline()
    if not handshake:
        return
    _, H, _W = (int(x) for x in handshake.split())
    while True:
        line = sys.stdin.readline()
        if not line:
            return
        _read_grid(sys.stdin, H)
        _read_grid(sys.stdin, H)
        _read_grid(sys.stdin, H)
        sys.stdout.write("\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
