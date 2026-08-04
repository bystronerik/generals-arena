"""Batch rebuild scraped replays into trajectories + corpus-index + report."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from arena.instrument.replay.loader import (
    REPLAYS_DIR,
    load_replay,
    iter_replay_paths,
)
from arena.paths import REPO_ROOT
from arena.records.store import engine_version as current_engine_version
from training.morpheus.scraped_rebuild.infer import infer_joint_actions
from training.morpheus.scraped_rebuild.source import source_label_for
from training.morpheus.scraped_rebuild.write_trajectory import (
    corpus_index_payload,
    write_trajectory_from_inference,
)


@dataclass
class RebuildReport:
    player: str
    source_label: str
    replays_root: str
    output: str
    engine_version: str
    scanned: int = 0
    kept: int = 0
    skipped_forfeit: int = 0
    skipped_unresolved: int = 0
    skipped_verify: int = 0
    skipped_other: int = 0
    ambiguous_tick_total: int = 0
    failures: list[dict[str, Any]] = field(default_factory=list)
    kept_game_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return path


def rebuild_player(
    *,
    player: str,
    replays: Path | None = None,
    output: Path,
    report_path: Path | None = None,
    max_games: int | None = None,
    force: bool = False,
    repo_root: Path | None = None,
) -> RebuildReport:
    """
    Reconstruct fully-resolved scraped games into arena trajectories.

    Games with any unresolved tick, missing seed, forfeit length, or failed
    engine verify are skipped. Always returns a report.
    """
    root = repo_root or REPO_ROOT
    replays_root = Path(replays) if replays is not None else REPLAYS_DIR / player
    # loader expects parent of win/lose/draw; allow either player dir or REPLAYS_DIR.
    if (replays_root / "win").is_dir() or (replays_root / "lose").is_dir():
        loader_root = replays_root.parent
        player_name = replays_root.name
    else:
        loader_root = replays_root
        player_name = player

    label = source_label_for(player_name)
    engine = current_engine_version()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    report = RebuildReport(
        player=player_name,
        source_label=label,
        replays_root=str(replays_root),
        output=str(output),
        engine_version=engine,
    )
    games_index: dict[str, dict[str, Any]] = {}

    paths = list(iter_replay_paths(player_name, folder="all", root=loader_root))
    if max_games is not None:
        paths = paths[: int(max_games)]

    for folder, path in paths:
        report.scanned += 1
        replay = load_replay(path, queried_player=player_name, folder=folder)
        if replay.is_forfeit:
            report.skipped_forfeit += 1
            continue

        inference = infer_joint_actions(replay)
        report.ambiguous_tick_total += len(inference.ambiguous)
        if not inference.ok:
            report.skipped_unresolved += 1
            report.failures.append(
                {
                    "match_id": replay.match_id,
                    "reason": "unresolved",
                    "unresolved": inference.unresolved[:20],
                }
            )
            continue

        result = write_trajectory_from_inference(
            inference,
            player=player_name,
            directory=output,
            engine=engine,
            force=force,
        )
        if not result.ok:
            if result.reason == "verify_failed":
                report.skipped_verify += 1
            else:
                report.skipped_other += 1
            report.failures.append(
                {
                    "match_id": replay.match_id,
                    "game_id": result.game_id,
                    "reason": result.reason,
                    "verify": result.verify_summary,
                }
            )
            continue

        report.kept += 1
        report.kept_game_ids.append(result.game_id)
        winner = "draw"
        if inference.winner == 0:
            winner = "a"
        elif inference.winner == 1:
            winner = "b"
        games_index[result.game_id] = {
            "source_label": label,
            "match_id": replay.match_id,
            "bot_a": inference.players[0],
            "bot_b": inference.players[1],
            "seed": inference.seed,
            "mode": "competition",
            "engine_version": engine,
            "winner": winner,
            "turns": result.turns,
            "ambiguous_ticks": result.ambiguous_ticks,
            "folder": folder,
        }

    index = corpus_index_payload(
        player=player_name,
        source_label=label,
        round_name=output.name,
        engine_version=engine,
        games=games_index,
    )
    (output / "corpus-index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if report_path is None:
        report_path = (
            root
            / "docs"
            / "research"
            / "measurements"
            / f"morpheus-{player_name.lower().replace('.', '-')}-rebuild.json"
        )
    report.write(report_path)
    return report
