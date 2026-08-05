"""Batch rebuild scraped replays into trajectories + corpus-index + report.

Each player-and-result pair becomes its own self-contained round under
``--output``::

    <output>/<player>-<win|lose|draw>-reconstructions/
        corpus-index.json
        <game_id>.traj.jsonl.gz

The result is the queried player's own, from ``Replay.outcome`` — derived from
the replay's ``winner`` and the player's seat — never the scraped folder name,
which records the result of whoever the list endpoint called side A.

One round per pair means rounds stay flat (what every trajectory consumer
already expects) and no two concurrent rebuilds ever write the same directory,
so players can be rebuilt fully in parallel.
"""

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
    round_directory,
    write_trajectory_from_inference,
)


VALID_OUTCOMES = frozenset({"win", "lose", "draw", "all"})
OUTCOMES = ("win", "lose", "draw")


@dataclass
class RebuildReport:
    player: str
    source_label: str
    replays_root: str
    output: str
    engine_version: str
    outcome_filter: str = "all"
    keep_target: int | None = None
    scanned: int = 0
    kept: int = 0
    skipped_outcome: int = 0
    skipped_forfeit: int = 0
    skipped_unresolved: int = 0
    skipped_verify: int = 0
    skipped_other: int = 0
    ambiguous_tick_total: int = 0
    # Kept trajectories per queried-player outcome, i.e. per round written.
    kept_by_outcome: dict[str, int] = field(
        default_factory=lambda: {"win": 0, "lose": 0, "draw": 0}
    )
    # outcome -> round directory, for the rounds this run actually wrote.
    rounds: dict[str, str] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)
    kept_game_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return path


def _write_index(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def rebuild_player(
    *,
    player: str,
    replays: Path | None = None,
    output: Path,
    report_path: Path | None = None,
    max_games: int | None = None,
    keep: int | None = None,
    outcome: str = "all",
    force: bool = False,
    repo_root: Path | None = None,
) -> RebuildReport:
    """
    Reconstruct fully-resolved scraped games into arena trajectories.

    Games with any unresolved tick, missing seed, forfeit length, or failed
    engine verify are skipped. ``outcome`` filters by ``Replay.outcome`` for the
    queried player (never by folder name). ``keep`` stops after N verify-ok
    trajectories; ``max_games`` still caps how many files are scanned.

    ``output`` is the parent the rounds are created under, not a round itself:
    each result gets its own flat, self-contained round directory
    ``<output>/<player>-<outcome>-reconstructions/`` with its own
    ``corpus-index.json``. Only rounds that actually kept a game are written.
    """
    outcome_filter = str(outcome).strip().lower()
    if outcome_filter not in VALID_OUTCOMES:
        raise ValueError(
            f"outcome must be one of {sorted(VALID_OUTCOMES)}, got {outcome!r}"
        )
    if keep is not None and int(keep) < 1:
        raise ValueError(f"keep must be >= 1, got {keep}")

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
        outcome_filter=outcome_filter,
        keep_target=int(keep) if keep is not None else None,
    )
    # One index per round, keyed by the queried player's outcome.
    games_index: dict[str, dict[str, dict[str, Any]]] = {o: {} for o in OUTCOMES}

    paths = list(iter_replay_paths(player_name, folder="all", root=loader_root))
    # max_games caps scanned files; keep continues until N kept unless capped.
    if max_games is not None and keep is None:
        paths = paths[: int(max_games)]

    for folder, path in paths:
        if keep is not None and report.kept >= int(keep):
            break
        if max_games is not None and report.scanned >= int(max_games):
            break

        report.scanned += 1
        replay = load_replay(path, queried_player=player_name, folder=folder)
        if outcome_filter != "all" and replay.outcome != outcome_filter:
            report.skipped_outcome += 1
            continue
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

        # Route by the player's own derived result, not `folder`. `replay.folder`
        # is provenance only; see arena.instrument.replay.loader.
        game_outcome = replay.outcome
        round_dir = round_directory(output, player_name, game_outcome)
        result = write_trajectory_from_inference(
            inference,
            player=player_name,
            directory=round_dir,
            engine=engine,
            force=force,
            round_name=round_dir.name,
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
        report.kept_by_outcome[game_outcome] += 1
        report.kept_game_ids.append(result.game_id)
        winner = "draw"
        if inference.winner == 0:
            winner = "a"
        elif inference.winner == 1:
            winner = "b"
        games_index[game_outcome][result.game_id] = {
            "source_label": label,
            "match_id": replay.match_id,
            "bot_a": inference.players[0],
            "bot_b": inference.players[1],
            "queried_player": player_name,
            "sample_seat": int(replay.seat_of(player_name)),
            "queried_outcome": replay.outcome,
            "seed": inference.seed,
            "mode": "competition",
            "engine_version": engine,
            "winner": winner,
            "turns": result.turns,
            "ambiguous_ticks": result.ambiguous_ticks,
            "folder": folder,
        }

    # Only write rounds this run produced, so an outcome-filtered run does not
    # leave empty `<player>-lose-reconstructions/` directories behind.
    for round_outcome, games in games_index.items():
        if not games:
            continue
        round_dir = round_directory(output, player_name, round_outcome)
        _write_index(
            round_dir / "corpus-index.json",
            corpus_index_payload(
                player=player_name,
                source_label=label,
                round_name=round_dir.name,
                engine_version=engine,
                games=games,
                outcome=round_outcome,
            ),
        )
        report.rounds[round_outcome] = str(round_dir)

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
