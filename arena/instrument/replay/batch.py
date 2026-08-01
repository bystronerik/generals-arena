"""
One cheap line per replay, over a whole folder.

Deliberately not the full pipeline: a player's window is ~300 files of ~0.6 MB,
so this loads one replay at a time, runs the single metrics pass plus fog and
the largest-stack path, and drops the second grid-diff pass that castle and
big-capture detection need. What survives is exactly the flaw signature the
per-game reports are read for — did we make contact, did we ever gather, did
the biggest stack go anywhere useful — aggregated across the folder.

Forfeits (`total_ticks <= 1`) are skipped and counted separately: they are
scoring artefacts, not games.

Every outcome here — the per-game label, the `--outcome` filter, the aggregate
buckets — is the queried player's own result derived from the replay, never the
directory the scraper filed it in. Half of one player's window is side B, where
the two disagree.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from arena.instrument.replay.events import GATHER_MIN_TICKS, find_first_contact, find_gather_waves
from arena.instrument.replay.fog import compute_vision
from arena.instrument.replay.loader import (
    REPLAYS_DIR,
    Replay,
    iter_replay_paths,
    load_replay,
)
from arena.instrument.replay.metrics import compute_metrics, summarize_seat
from arena.instrument.replay.path import compute_path, summarize_path


@dataclass(frozen=True)
class GameLine:
    """One game's flaw signature. `outcome` is the queried player's own result
    and `folder` is only where the scraper filed it; `final_tiles` is
    `(us, them)` at the last frame where both sides still held land — see
    `_last_contested_frame`."""

    match_id: str
    outcome: str
    folder: str
    ticks: int
    seat: int
    opponent: str
    self_match: bool
    first_contact: int | None
    gather_waves: int
    toward_fraction: float | None
    moves: int
    peak_stack: int
    saw_general: int | None
    closest_approach: int | None
    final_tiles: tuple[int, int]

    def as_json(self) -> dict[str, Any]:
        return {
            "match_id": self.match_id,
            "outcome": self.outcome,
            "folder": self.folder,
            "ticks": self.ticks,
            "seat": self.seat,
            "opponent": self.opponent,
            "self_match": self.self_match,
            "first_contact": self.first_contact,
            "gather_waves": self.gather_waves,
            "toward_fraction": self.toward_fraction,
            "moves": self.moves,
            "peak_stack": self.peak_stack,
            "saw_general": self.saw_general,
            "closest_approach": self.closest_approach,
            "final_tiles": list(self.final_tiles),
        }

    def render(self) -> str:
        toward = "  n/a" if self.toward_fraction is None else f"{self.toward_fraction:>5.0%}"
        contact = "none" if self.first_contact is None else f"t{self.first_contact}"
        general = "unseen" if self.saw_general is None else f"t{self.saw_general}"
        flag = " [self]" if self.self_match else ""
        return (
            f"{self.match_id:>7} {self.outcome:<5} {self.ticks:>5}t  "
            f"vs {self.opponent[:16]:<16} contact {contact:<6} waves {self.gather_waves:>2}  "
            f"toward {toward}  stack {self.peak_stack:>4}  gen {general:<7} "
            f"tiles {self.final_tiles[0]:>3}/{self.final_tiles[1]:<3}{flag}"
        )


def _last_contested_frame(metrics):
    """
    The last frame where both seats still held land.

    Capturing a general hands over every tile the loser owned, so the final
    frame of a decided game always reads 0 against everything — useless as a
    measure of how close the game was. This is the frame before that.
    """
    for entry in reversed(metrics):
        if entry.seats[0].tiles and entry.seats[1].tiles:
            return entry.seats
    return metrics[-1].seats


def scan_game(replay: Replay) -> GameLine:
    """The cheap subset: metrics, fog, our own stack's path, contact, gathers."""
    us = replay.seat_of(replay.queried_player)
    metrics = compute_metrics(replay)
    vision = compute_vision(replay)
    steps = compute_path(replay, metrics, vision, us)
    path = summarize_path(steps)
    summary = summarize_seat(metrics, us)
    final = _last_contested_frame(metrics)
    return GameLine(
        match_id=replay.match_id,
        outcome=replay.outcome,
        folder=replay.folder,
        ticks=replay.total_ticks,
        seat=us,
        opponent=replay.name(1 - us),
        self_match=replay.is_self_match,
        first_contact=find_first_contact(replay),
        gather_waves=len(find_gather_waves(steps, us, GATHER_MIN_TICKS)),
        toward_fraction=path.toward_fraction,
        moves=path.moves,
        peak_stack=summary.peak_stack,
        saw_general=vision.first_general_sight[us],
        closest_approach=summary.closest_approach,
        final_tiles=(final[us].tiles, final[1 - us].tiles),
    )


def iter_games(
    player: str, outcome: str = "all", root: Path = REPLAYS_DIR
) -> Iterator[tuple[GameLine | None, str, Path]]:
    """
    Yield `(line, folder, path)` per replay, loading one file at a time.

    `outcome` selects on the queried player's *derived* result, not on the
    directory — the directory holds side A's result and contradicts side B.
    That means every folder is globbed whatever the filter, and the file has to
    be opened before it can be excluded.

    `line` is None for a forfeit, so the caller can count it and move on.
    Forfeits are surfaced only when nothing is filtered: a game that ended on
    tick 1 has a scored winner but no play behind it, so letting it answer
    "show me the losses" would put an artefact in a sample of games.
    """
    for folder, path in iter_replay_paths(player, "all", root):
        replay = load_replay(path, player, folder)
        if replay.is_forfeit:
            if outcome == "all":
                yield None, folder, path
            continue
        if outcome != "all" and replay.outcome != outcome:
            continue
        yield scan_game(replay), folder, path


def aggregate(lines: list[GameLine], forfeits: int) -> dict[str, Any]:
    """Flaw signatures, whole-folder and per outcome."""
    out: dict[str, Any] = {
        "games": len(lines),
        "forfeits_skipped": forfeits,
        "self_matches": sum(1 for line in lines if line.self_match),
        "by_outcome": {},
    }
    groups: dict[str, list[GameLine]] = {}
    for line in lines:
        groups.setdefault(line.outcome, []).append(line)
    if len(groups) > 1:
        groups = {"all": lines, **groups}
    for name, group in groups.items():
        if not group:
            continue
        towards = [line.toward_fraction for line in group if line.toward_fraction is not None]
        out["by_outcome"][name] = {
            "games": len(group),
            "no_gather_wave": sum(1 for line in group if line.gather_waves == 0),
            "never_saw_general": sum(1 for line in group if line.saw_general is None),
            "no_contact": sum(1 for line in group if line.first_contact is None),
            "median_gather_waves": median([line.gather_waves for line in group]),
            "median_toward_fraction": median(towards) if towards else None,
            "median_peak_stack": median([line.peak_stack for line in group]),
            "median_ticks": median([line.ticks for line in group]),
            "median_closest_approach": median(
                [line.closest_approach for line in group if line.closest_approach is not None]
            )
            if any(line.closest_approach is not None for line in group)
            else None,
        }
    return out


def render_aggregate(summary: dict[str, Any], player: str) -> str:
    lines = [
        f"SUMMARY for {player}: {summary['games']} played games "
        f"({summary['forfeits_skipped']} forfeits skipped, "
        f"{summary['self_matches']} self-matches)"
    ]
    for name, stats in summary["by_outcome"].items():
        total = stats["games"]
        toward = stats["median_toward_fraction"]
        closest = stats["median_closest_approach"]
        lines.append(
            f"  {name:<5} n={total:<4} "
            f"no gather wave in {stats['no_gather_wave']}/{total}, "
            f"enemy general never seen in {stats['never_saw_general']}/{total}, "
            f"no contact in {stats['no_contact']}/{total}"
        )
        lines.append(
            f"        median: {stats['median_ticks']:.0f} ticks, "
            f"{stats['median_gather_waves']:.0f} gather waves, "
            f"peak stack {stats['median_peak_stack']:.0f}, "
            f"toward {'n/a' if toward is None else f'{toward:.0%}'}, "
            f"closest approach {'n/a' if closest is None else f'{closest:.0f}'}"
        )
    return "\n".join(lines)
