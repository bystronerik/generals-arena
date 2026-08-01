"""
Offline analysis of scraped generals.bot leaderboard replays.

Read-only over `competition-replays/`. Nothing here writes anywhere, and
nothing here may feed `data/games/`, `data/ratings/`, or `data/remote_games/`:
these are observational games played by other people's agents under rules we
did not run, so they are evidence about play, never rated matches. See
docs/engine/leaderboard-replays.md.

Layering, cheapest first — each stage takes the one before it:

    loader   -> Replay            file discovery, typed frames
    metrics  -> TickMetrics       per-tick economy, one pass over the grids
    fog      -> Vision            who could see the enemy general, and when
    path     -> StackStep         where the biggest stack went, and toward what
    events   -> Event             timestamped incidents
    phases   -> Phase             coarse segmentation
    analysis -> Analysis          the bundle
    report   -> Report            text and JSON sections
    batch    -> GameLine          one cheap line per replay, for whole folders

Driven by `scripts/replay.py`.
"""

from __future__ import annotations

from arena.instrument.replay.analysis import Analysis, analyze
from arena.instrument.replay.loader import (
    OUTCOMES,
    REPLAYS_DIR,
    Replay,
    ReplayNotFound,
    find_replay,
    iter_replay_paths,
    load_replay,
)
from arena.instrument.replay.report import Report, build_report

__all__ = [
    "Analysis",
    "OUTCOMES",
    "REPLAYS_DIR",
    "Replay",
    "ReplayNotFound",
    "Report",
    "analyze",
    "build_report",
    "find_replay",
    "iter_replay_paths",
    "load_replay",
]
