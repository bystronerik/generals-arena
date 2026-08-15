"""Characterize joe's action cycles from a recorded diagnostic trace.

Reads `bots/joe/probe.py`'s per-turn keys back out of a `.trace.<seat>.jsonl.gz`
and answers, in order: when does a cycle start, how long is it, how much of the
game sits inside one, and does it break on its own. Then it separates the three
candidate faults — near-tie argmax, confident loop, stale observation state — by
the margin, the entropy, and the logit hash.

    python scripts/inspect_joe_oscillation.py <traj-dir> <game-id> <seat>
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from arena.records.trajectories import (
    read_jsonl_gz,
    read_trajectory,
    trace_path,
    trajectory_path,
)


def label(row: dict) -> str:
    """The chosen action: the first entry of the packed top-5, without a logit."""
    return row["joe_top5"].split("|")[0].rsplit(":", 1)[0]


def runs(rows: list[dict]) -> list[tuple[int, int, int]]:
    """Maximal `(start_turn, end_turn, period)` spans with a nonzero period."""
    out: list[tuple[int, int, int]] = []
    start = None
    for row in rows:
        p = row["joe_cycle_period"]
        if p and start is None:
            start = (row["t"], p)
        elif not p and start is not None:
            out.append((start[0], row["t"] - 1, start[1]))
            start = None
    if start is not None:
        out.append((start[0], rows[-1]["t"], start[1]))
    return out


def pct(part: int, whole: int) -> str:
    return f"{part} ({100.0 * part / whole:.1f}%)" if whole else "0"


def main(argv: list[str]) -> int:
    traj_dir, game_id, seat = Path(argv[1]), argv[2], argv[3]
    rows = list(read_jsonl_gz(trace_path(game_id, seat, traj_dir)))
    traj = read_trajectory(trajectory_path(game_id, traj_dir))
    n = len(rows)

    # A cycle_period is reported once three repetitions have completed, so the
    # loop actually began 3p-1 turns earlier. Report the true first turn.
    spans = runs(rows)
    inside = sum(e - s + 1 for s, e, _ in spans)

    print(f"{game_id} seat {seat}: {n} probed turns, "
          f"engine winner={traj.end.get('winner')} turns={len(traj.frames)}")
    print(f"turns inside a detected cycle: {pct(inside, n)}")
    print(f"cycle spans: {len(spans)}; periods seen: "
          f"{sorted(Counter(p for _, _, p in spans).items())}")

    print("\nlongest 12 cycle spans (start is the turn the loop was detected):")
    for s, e, p in sorted(spans, key=lambda x: x[1] - x[0], reverse=True)[:12]:
        window = [r for r in rows if s <= r["t"] <= e]
        acts = sorted({label(r) for r in window})
        margins = [r["joe_margin_milli"] for r in window]
        vals = [r["joe_value_milli"] for r in window]
        print(f"  t={s}-{e} len={e - s + 1:>4} p={p} "
              f"margin[min/med/max]={min(margins)}/"
              f"{sorted(margins)[len(margins) // 2]}/{max(margins)} "
              f"value[{min(vals)}..{max(vals)}] actions={acts[:6]}")

    # Fault separation. Near-tie vs confident loop is the margin; stale input is
    # the logit hash, which is exact.
    hashes = Counter(r["joe_logits_hash"] for r in rows)
    dup_turns = sum(c for c in hashes.values() if c > 1)
    in_cycle = [r for r in rows if r["joe_cycle_period"]]
    out_cycle = [r for r in rows if not r["joe_cycle_period"]]

    print("\n--- fault separation ---")
    for name, group in (("in-cycle", in_cycle), ("out-of-cycle", out_cycle)):
        if not group:
            continue
        m = sorted(r["joe_margin_milli"] for r in group)
        h = sorted(r["joe_entropy_milli"] for r in group)
        v = sorted(r["joe_value_milli"] for r in group)
        print(f"{name:<13} n={len(group):<4} "
              f"margin med={m[len(m) // 2]:<7} p10={m[len(m) // 10]:<7} "
              f"entropy med={h[len(h) // 2]:<6} value med={v[len(v) // 2]}")
    ties = sum(1 for r in rows if r["joe_margin_milli"] < 100)
    print(f"turns with margin < 0.1 logit: {pct(ties, n)}")
    print(f"turns whose logit vector repeats another turn's: {pct(dup_turns, n)}"
          f"  (distinct vectors: {len(hashes)})")

    masks = sorted(r["joe_move_cells"] for r in rows)
    print(f"legal move entries: min={masks[0]} med={masks[len(masks) // 2]} "
          f"max={masks[-1]}")
    rev = sorted(r["joe_cell_revisits"] for r in rows)
    print(f"cell revisits in last 11 turns: med={rev[len(rev) // 2]} "
          f"max={rev[-1]}")

    # Engine truth alongside, so a cycle can be read against material.
    print("\nengine land/army every 50 turns (a=macaria, b=joe):")
    for f in traj.frames[::50]:
        print(f"  t={f.turn:<4} land={f.land} army={f.army}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
