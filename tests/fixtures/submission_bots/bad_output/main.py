"""Emit non-integer and wrong-arity lines."""
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
    replies = 0
    while True:
        line = sys.stdin.readline()
        if not line:
            return
        _read_grid(sys.stdin, H)
        _read_grid(sys.stdin, H)
        _read_grid(sys.stdin, H)
        if replies == 0:
            sys.stdout.write("not integers\n")
        else:
            sys.stdout.write("1 2 3\n")
        sys.stdout.flush()
        replies += 1


if __name__ == "__main__":
    main()
