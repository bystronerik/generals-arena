"""Exit nonzero after the handshake (immediate forfeit)."""
from __future__ import annotations

import sys


def main() -> None:
    handshake = sys.stdin.readline()
    if not handshake:
        return
    sys.exit(1)


if __name__ == "__main__":
    main()
