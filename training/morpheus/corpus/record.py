"""Record the Morpheus bootstrap corpus through the competition tournament."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from arena.paths import REPO_ROOT
from arena.records.store import round_games_dir
from arena.records.trajectories import round_trajectory_dir
from arena.tournaments.competition import ALTERNATING_SEATS, run_tournament
from training.morpheus.corpus.panel import (
    PanelError,
    load_panel,
    panel_run_scripts,
    validate_panel_against_checkout,
)


def write_corpus_index(
    trajectories_dir: Path,
    *,
    panel: dict[str, Any],
    round_name: str,
    records: list[Any],
) -> Path:
    """Persist source labels and panel pins beside the trajectory files."""
    games: dict[str, Any] = {}
    for record in records:
        games[record.game_id] = {
            "source_label": panel["source_label"],
            "bot_a": record.bot_a,
            "bot_b": record.bot_b,
            "bot_a_content_hash": record.bot_a_content_hash,
            "bot_b_content_hash": record.bot_b_content_hash,
            "seed": record.seed,
            "mode": record.mode,
            "engine_version": record.engine_version,
            "winner": record.winner,
            "turns": record.turns,
        }
    payload = {
        "round": round_name,
        "panel_name": panel["name"],
        "source_label": panel["source_label"],
        "rating_era": panel["rating_era"],
        "selection_date": panel["selection_date"],
        "members": [
            {
                "bot_id": m["bot_id"],
                "content_hash": m["content_hash"],
                "role": m["role"],
            }
            for m in panel["members"]
        ],
        "game_count": len(games),
        "games": games,
    }
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    path = trajectories_dir / "corpus-index.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def record_corpus(
    *,
    panel_path: Path,
    round_name: str,
    round_seed: int | None = None,
    seat_policy: str = ALTERNATING_SEATS,
    games_per_pair: int | None = None,
    output: Path | None = None,
    jobs: int | None = None,
    update_ratings: bool = True,
    strict_versions: bool = False,
) -> dict[str, Any]:
    """
    Validate the panel, run competition matches with trajectories, write the index.

    Games store under `data/games/<round>/`. Trajectories store under
    `data/trajectories/<round>/` (or `--output` when it matches that path).
    """
    panel = load_panel(panel_path)
    problems = validate_panel_against_checkout(panel)
    if problems:
        raise PanelError(
            "panel disagrees with this checkout:\n  " + "\n  ".join(problems)
        )

    if seat_policy != ALTERNATING_SEATS:
        raise PanelError(
            f"corpus recording requires seat_policy='alternate' (got {seat_policy!r})"
        )

    seed = int(panel["round_seed"] if round_seed is None else round_seed)
    pair_games = int(
        panel["games_per_pair"] if games_per_pair is None else games_per_pair
    )
    scripts = panel_run_scripts(panel)

    expected_traj = round_trajectory_dir(round_name)
    if output is not None:
        resolved = output.resolve()
        if resolved != expected_traj.resolve():
            raise PanelError(
                f"--output {resolved} must equal the round trajectory dir {expected_traj}; "
                f"the tournament keys trajectories by --round"
            )

    records = run_tournament(
        scripts,
        round_name=round_name,
        games_per_pair=pair_games,
        round_seed=seed,
        seat_policy=seat_policy,
        record_trajectories=True,
        update_ratings=update_ratings,
        strict_versions=strict_versions,
        jobs=jobs,
    )

    traj_dir = expected_traj
    index_path = write_corpus_index(
        traj_dir,
        panel=panel,
        round_name=round_name,
        records=records,
    )
    games_dir = round_games_dir(round_name)
    return {
        "round": round_name,
        "games_dir": str(games_dir.relative_to(REPO_ROOT)),
        "trajectories_dir": str(traj_dir.relative_to(REPO_ROOT)),
        "corpus_index": str(index_path.relative_to(REPO_ROOT)),
        "match_count": len(records),
        "panel": panel["name"],
        "source_label": panel["source_label"],
    }
