#!/usr/bin/env python3
"""
Corpus census + deterministic 90/10 fit/holdout split for the Kubic replays.

Split rule: playable replays (not forfeits) sorted by numeric match id; every
10th (index % 10 == 9) goes to holdout, the rest to fit. Derived outcomes come
from the replay itself (Replay.outcome), never the folder.

Writes docs/research/measurements/fable-kubic-split.json with the manifest and
a census (outcome counts, board sizes, opponents, lengths).
"""

import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arena.instrument.replay.loader import iter_replay_paths, load_replay

PLAYER = "Kubic"
OUT = REPO_ROOT / "docs/research/measurements/fable-kubic-split.json"


def main() -> None:
    rows = []
    forfeits = []
    for folder, path in iter_replay_paths(PLAYER):
        replay = load_replay(path, queried_player=PLAYER, folder=folder)
        if replay.is_forfeit:
            forfeits.append(replay.match_id)
            continue
        us = replay.seat_of(PLAYER)
        rows.append(
            {
                "match_id": replay.match_id,
                "folder": folder,
                "outcome": replay.outcome,
                "seat": us,
                "opponent": replay.players[1 - us],
                "rows": replay.rows,
                "cols": replay.cols,
                "total_ticks": replay.total_ticks,
                "kubic_in_players": PLAYER in replay.players,
            }
        )

    rows.sort(key=lambda r: int(r["match_id"]))
    for index, row in enumerate(rows):
        row["set"] = "holdout" if index % 10 == 9 else "fit"

    fit = [r for r in rows if r["set"] == "fit"]
    holdout = [r for r in rows if r["set"] == "holdout"]
    census = {
        "playable": len(rows),
        "forfeits": len(forfeits),
        "forfeit_ids": forfeits,
        "outcomes": dict(Counter(r["outcome"] for r in rows)),
        "fit_outcomes": dict(Counter(r["outcome"] for r in fit)),
        "holdout_outcomes": dict(Counter(r["outcome"] for r in holdout)),
        "folder_vs_outcome_disagreements": sum(1 for r in rows if r["folder"] != r["outcome"]),
        "not_a_player": [r["match_id"] for r in rows if not r["kubic_in_players"]],
        "board_sizes": dict(Counter(f"{r['rows']}x{r['cols']}" for r in rows)),
        "opponents": dict(Counter(r["opponent"] for r in rows).most_common()),
        "tick_len_min": min(r["total_ticks"] for r in rows),
        "tick_len_max": max(r["total_ticks"] for r in rows),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"census": census, "replays": rows}, indent=2))
    print(json.dumps(census, indent=2)[:4000])
    print(f"fit={len(fit)} holdout={len(holdout)} -> {OUT}")


if __name__ == "__main__":
    main()
