"""Aggregate remote game logs for human-block reporting."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from arena.remote_block import counts_as_human_block_game

# Opponent star bands for win-rate splits (human-95-plan §3.5).
STAR_BANDS: tuple[tuple[str, int | None, int | None], ...] = (
    ("unrated", None, None),
    ("0-49", 0, 49),
    ("50-99", 50, 99),
    ("100-149", 100, 149),
    ("150+", 150, None),
)


def star_band(stars: int | None) -> str:
    """Map opponent star count to a report band label."""
    if stars is None:
        return "unrated"
    for label, low, high in STAR_BANDS:
        if low is None and high is None:
            continue
        if low is not None and stars < low:
            continue
        if high is not None and stars > high:
            continue
        return label
    return "150+"


def wilson_lower_bound(successes: int, trials: int, *, z: float = 1.96) -> float:
    """Wilson score interval lower bound at the given z (default 95%)."""
    if trials <= 0:
        return 0.0
    p_hat = successes / trials
    n = trials
    z2 = z * z
    denominator = 1.0 + z2 / n
    centre = p_hat + z2 / (2.0 * n)
    margin = z * math.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * n)) / n)
    return max(0.0, (centre - margin) / denominator)


def load_remote_records(log_dir: Path) -> list[dict[str, Any]]:
    """Load every game JSON under log_dir except session_error files."""
    if not log_dir.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(log_dir.glob("*.json")):
        if path.name.startswith("session_error_"):
            continue
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return records


@dataclass
class StarBandStats:
    label: str
    games: int = 0
    wins: int = 0
    losses: int = 0

    @property
    def win_rate(self) -> float | None:
        if self.games == 0:
            return None
        return self.wins / self.games


@dataclass
class RemoteReportStats:
    total_files: int = 0
    counted_human_games: int = 0
    wins: int = 0
    losses: int = 0
    lobby_games: int = 0
    queue_games: int = 0
    bot_opponent_games: int = 0
    discarded_games: int = 0
    star_bands: dict[str, StarBandStats] = field(default_factory=dict)
    loss_replays: list[tuple[str | None, str | None]] = field(default_factory=list)
    bot_ids: set[str] = field(default_factory=set)

    @property
    def win_rate(self) -> float | None:
        decided = self.wins + self.losses
        if decided == 0:
            return None
        return self.wins / decided

    @property
    def wilson_lb(self) -> float:
        decided = self.wins + self.losses
        return wilson_lower_bound(self.wins, decided)


def aggregate_remote_games(records: list[dict[str, Any]]) -> RemoteReportStats:
    """Build report stats from loaded remote game records."""
    stats = RemoteReportStats(total_files=len(records))

    for record in records:
        bot_id = record.get("bot_id")
        if isinstance(bot_id, str):
            stats.bot_ids.add(bot_id)

        if record.get("opponent_is_bot") is True:
            stats.bot_opponent_games += 1

        if not counts_as_human_block_game(record):
            if record.get("counts_toward_block"):
                stats.discarded_games += 1
            continue

        stats.counted_human_games += 1
        room = record.get("room_mode")
        if room == "lobby":
            stats.lobby_games += 1
        elif room == "1v1":
            stats.queue_games += 1

        result = record.get("result")
        if result == "win":
            stats.wins += 1
        elif result == "loss":
            stats.losses += 1
            stats.loss_replays.append(
                (record.get("replay_id"), record.get("opponent_username"))
            )

        band_label = star_band(record.get("opponent_stars"))
        band = stats.star_bands.setdefault(band_label, StarBandStats(label=band_label))
        band.games += 1
        if result == "win":
            band.wins += 1
        elif result == "loss":
            band.losses += 1

    return stats


def format_markdown_report(
    stats: RemoteReportStats,
    *,
    log_dir: Path,
    generated_at: datetime | None = None,
) -> str:
    """Render a human-block markdown report."""
    ts = generated_at or datetime.now(timezone.utc)
    decided = stats.wins + stats.losses
    win_rate_pct = f"{100.0 * stats.win_rate:.1f}%" if stats.win_rate is not None else "n/a"
    wilson_pct = f"{100.0 * stats.wilson_lb:.1f}%"

    lines = [
        "# Remote human-block report",
        "",
        f"Generated: {ts.strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"Source: `{log_dir}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| JSON files scanned | {stats.total_files} |",
        f"| Counted human games | {stats.counted_human_games} |",
        f"| Wins / losses | {stats.wins} / {stats.losses} |",
        f"| Win rate (decided) | {win_rate_pct} |",
        f"| Wilson 95% lower bound | {wilson_pct} |",
        f"| Lobby share | {stats.lobby_games} |",
        f"| 1v1 queue share | {stats.queue_games} |",
        f"| Bot-opponent (excluded) | {stats.bot_opponent_games} |",
        f"| Other discarded | {stats.discarded_games} |",
    ]
    if stats.bot_ids:
        lines.append(f"| Bots seen | {', '.join(sorted(stats.bot_ids))} |")

    lines.extend(["", "## Win rate by opponent stars", ""])
    lines.extend(["| Band | Games | W | L | Win rate |", "| --- | ---: | ---: | ---: | --- |"])

    for label, _, _ in STAR_BANDS:
        band = stats.star_bands.get(label)
        if band is None or band.games == 0:
            lines.append(f"| {label} | 0 | 0 | 0 | n/a |")
            continue
        rate = f"{100.0 * band.win_rate:.1f}%" if band.win_rate is not None else "n/a"
        lines.append(f"| {label} | {band.games} | {band.wins} | {band.losses} | {rate} |")

    if stats.loss_replays:
        lines.extend(["", "## Loss replays", ""])
        for replay_id, opponent in stats.loss_replays:
            rid = replay_id or "unknown"
            opp = opponent or "unknown"
            lines.append(f"- `{rid}` vs {opp}")

    lines.append("")
    return "\n".join(lines)
