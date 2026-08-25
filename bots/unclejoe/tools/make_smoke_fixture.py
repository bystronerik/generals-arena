"""Cut the committed smoke slice from the full parity corpus (port-plan §6).

Takes the first `--turns` frames of one captured game's wire logs into
`tests/fixtures/`, then runs the same capture as the full corpus over the
truncated logs (a game prefix is a valid game: the state machine starts from
zeros). The pytest drivers pick fixtures up automatically when the full
corpus is absent, so a clean checkout still runs a real parity check.

Usage: .venv/bin/python bots/unclejoe/tools/make_smoke_fixture.py [--game NAME]
"""
import argparse
from pathlib import Path

from capture_fixtures import (
    CORPUS,
    capture_game,
    parse_in_log,
    record_selection_goldens,
)

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default="aegis-seed0")
    parser.add_argument("--turns", type=int, default=40)
    parser.add_argument("--stride", type=int, default=8,
                        help="sampled-frame target for the slice (<= 10 committed frames)")
    args = parser.parse_args()

    games_dir = CORPUS / "games"
    name = f"smoke-{args.game}"
    src_in = games_dir / f"{args.game}.in.log"
    src_out = games_dir / f"{args.game}.out.log"

    lines = src_in.read_text().splitlines()
    H = int(lines[0].split()[1])
    frame_len = 1 + 3 * H
    keep = lines[:1 + args.turns * frame_len]
    FIXTURES.mkdir(parents=True, exist_ok=True)
    (FIXTURES / f"{name}.in.log").write_text("\n".join(keep) + "\n")
    replies = src_out.read_text().splitlines()[:args.turns]
    (FIXTURES / f"{name}.out.log").write_text("\n".join(replies) + "\n")

    npz = FIXTURES / f"{name}.npz"
    npz.unlink(missing_ok=True)
    capture_game(name, FIXTURES, stride_target=args.stride)
    # The committed slice carries its own wire-replay golden too, so the
    # mutation checker's smoke scope grades the full played path on a clean
    # checkout. Needs the release binary (selection-plan S1).
    record_selection_goldens(FIXTURES)

    _, _, _, frames = parse_in_log(FIXTURES / f"{name}.in.log")
    assert len(frames) == args.turns
    size = sum(f.stat().st_size for f in FIXTURES.glob(f"{name}.*"))
    print(f"[smoke] {name}: {args.turns} turns, {size // 1024} KiB total")
    assert size < 1_000_000, "smoke slice must stay cheap to commit"


if __name__ == "__main__":
    main()
