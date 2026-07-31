#!/usr/bin/env python3
"""Resume missing round4 matches, rebuild Elo, write the measurement report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from arena.parallel import default_jobs, run_pool
from arena.ratings import rebuild_from_games
from arena.store import GAMES_DIR, git_commit_or_tag, load_all_games, round_games_dir
from arena.tournament_worker import run_one_worker
from scripts.measure_heuristics import game_entry_from_record, write_reports


def main() -> int:
    games_dir = round_games_dir("round4")
    manifest = json.loads((games_dir / "manifest.json").read_text(encoding="utf-8"))
    assignments = {(a["bot_a"], a["bot_b"], a["seed"]) for a in manifest["assignments"]}
    done = {(r.bot_a, r.bot_b, r.seed) for r in load_all_games(games_dir)}
    missing = sorted(assignments - done)
    jobs = default_jobs()
    commit = git_commit_or_tag()
    print(f"[resume] missing={len(missing)} jobs={jobs} dir={games_dir}")

    if missing:
        payloads = [
            {
                "bot_a_run": str((REPO / "bots" / bot_a / "run.sh").resolve()),
                "bot_b_run": str((REPO / "bots" / bot_b / "run.sh").resolve()),
                "seed": seed,
                "games_dir": str(games_dir),
                "mode": "competition",
                "timeout": None,
                "commit": commit,
            }
            for bot_a, bot_b, seed in missing
        ]

        def on_result(done_n: int, total: int, record) -> None:
            print(
                f"[resume] ({done_n}/{total}) {record.bot_a} vs {record.bot_b} "
                f"seed={record.seed} -> {record.winner} turns={record.turns}"
            )

        records_new = run_pool(
            payloads, run_one_worker, jobs=jobs, on_result=on_result
        )
        print(f"[resume] finished {len(records_new)} missing match(es)")

    all_records = load_all_games(games_dir)
    print(f"[resume] round4 total stored={len(all_records)}")
    book = rebuild_from_games(games_dir=GAMES_DIR)
    print(f"[resume] rebuilt ratings ({len(book.rated_game_ids)} game(s))")

    games = [game_entry_from_record(r, tag="pair") for r in all_records]
    grid_desc = {
        "rule": "C",
        "roster": manifest.get("bots"),
        "games_per_pair": manifest.get("games_per_pair"),
        "round_seed": manifest.get("round_seed"),
        "swap_sides": manifest.get("swap_sides"),
        "jobs": manifest.get("jobs"),
        "games_dir": str(games_dir),
        "resumed_missing": len(missing),
    }
    json_path, md_path = write_reports(
        games, round_name="round4", grid_desc=grid_desc
    )
    print(f"[resume] wrote {json_path}")
    print(f"[resume] wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
