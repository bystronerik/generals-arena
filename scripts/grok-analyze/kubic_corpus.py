#!/usr/bin/env python3
"""Shared Kubic replay corpus: true outcomes, fit/holdout split, load helpers.

Deterministic split (wins only for the primary evidence set):
  sorted match_id ascending; every 10th index (0-based i % 10 == 9) -> holdout;
  the rest -> fit.

Folder labels are NOT outcomes — see docs/engine/leaderboard-replays.md.
This module never writes to data/games/, data/ratings/, or data/remote_games/.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arena.instrument.replay.analysis import ForfeitReplay, analyze, Analysis
from arena.instrument.replay.loader import REPLAYS_DIR, Replay, iter_replay_paths, load_replay

PLAYER = "Kubic"
HOLDOUT_EVERY = 10  # i % 10 == 9 -> holdout
MEASUREMENTS = REPO_ROOT / "docs" / "research" / "measurements"


@dataclass(frozen=True)
class CorpusMeta:
    """Inventory after seat-resolved outcomes (forfeits excluded from played)."""

    folder_files: dict[str, int]
    played: dict[str, int]
    forfeits: int
    fit_win_ids: tuple[str, ...]
    holdout_win_ids: tuple[str, ...]
    loss_ids: tuple[str, ...]
    draw_ids: tuple[str, ...]

    def as_json(self) -> dict:
        return {
            "player": PLAYER,
            "note": (
                "Folder file counts include one .json per match. "
                "played.* uses Replay.outcome (seat-resolved), not folder."
            ),
            "folder_replay_counts": self.folder_files,
            "played": self.played,
            "forfeits": self.forfeits,
            "split_rule": (
                f"wins sorted by match_id; index i with i%{HOLDOUT_EVERY}==9 -> holdout; "
                "else fit. Losses/draws are held out of fit for rule derivation; "
                "skimmed separately for failure modes."
            ),
            "fit_win_ids": list(self.fit_win_ids),
            "holdout_win_ids": list(self.holdout_win_ids),
            "loss_ids": list(self.loss_ids),
            "draw_ids": list(self.draw_ids),
            "n_fit_wins": len(self.fit_win_ids),
            "n_holdout_wins": len(self.holdout_win_ids),
        }


def _inventory() -> CorpusMeta:
    folder_files = {"win": 0, "lose": 0, "draw": 0}
    wins: list[str] = []
    losses: list[str] = []
    draws: list[str] = []
    forfeits = 0
    for folder, path in iter_replay_paths(PLAYER, root=REPLAYS_DIR):
        folder_files[folder] = folder_files.get(folder, 0) + 1
        replay = load_replay(path, queried_player=PLAYER, folder=folder)
        if replay.is_forfeit:
            forfeits += 1
            continue
        mid = str(replay.match_id)
        if replay.outcome == "win":
            wins.append(mid)
        elif replay.outcome == "lose":
            losses.append(mid)
        else:
            draws.append(mid)
    wins_sorted = tuple(sorted(wins, key=lambda x: int(x) if x.isdigit() else x))
    fit = tuple(w for i, w in enumerate(wins_sorted) if i % HOLDOUT_EVERY != 9)
    holdout = tuple(w for i, w in enumerate(wins_sorted) if i % HOLDOUT_EVERY == 9)
    return CorpusMeta(
        folder_files=folder_files,
        played={"win": len(wins), "lose": len(losses), "draw": len(draws)},
        forfeits=forfeits,
        fit_win_ids=fit,
        holdout_win_ids=holdout,
        loss_ids=tuple(sorted(losses, key=lambda x: int(x) if x.isdigit() else x)),
        draw_ids=tuple(sorted(draws, key=lambda x: int(x) if x.isdigit() else x)),
    )


_META: CorpusMeta | None = None


def corpus_meta() -> CorpusMeta:
    global _META
    if _META is None:
        _META = _inventory()
    return _META


def write_corpus_meta(path: Path | None = None) -> Path:
    path = path or (MEASUREMENTS / "grok-kubic-corpus-split.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(corpus_meta().as_json(), indent=2) + "\n")
    return path


def load_by_id(match_id: str) -> Replay:
    for folder, path in iter_replay_paths(PLAYER, root=REPLAYS_DIR):
        if path.stem == str(match_id):
            return load_replay(path, queried_player=PLAYER, folder=folder)
    raise FileNotFoundError(f"Kubic replay {match_id} not found")


def iter_set(
    which: str,
    *,
    include_losses: bool = False,
    include_draws: bool = False,
) -> Iterator[Replay]:
    """
    Yield replays for `fit` | `holdout` | `losses` | `draws` | `all_wins`.

    `fit` / `holdout` are win-only by design. Pass include_* only for skim passes.
    """
    meta = corpus_meta()
    if which == "fit":
        ids = list(meta.fit_win_ids)
        if include_losses:
            ids.extend(meta.loss_ids)
        if include_draws:
            ids.extend(meta.draw_ids)
    elif which == "holdout":
        ids = list(meta.holdout_win_ids)
    elif which == "losses":
        ids = list(meta.loss_ids)
    elif which == "draws":
        ids = list(meta.draw_ids)
    elif which == "all_wins":
        ids = list(meta.fit_win_ids) + list(meta.holdout_win_ids)
    else:
        raise ValueError(f"unknown set {which!r}")
    for mid in ids:
        yield load_by_id(mid)


def iter_analyzed(
    which: str,
    *,
    include_losses: bool = False,
    include_draws: bool = False,
) -> Iterator[tuple[Replay, Analysis]]:
    for replay in iter_set(
        which, include_losses=include_losses, include_draws=include_draws
    ):
        try:
            yield replay, analyze(replay)
        except ForfeitReplay:
            continue


def percentile(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return float(s[f])
    return float(s[f] + (s[c] - s[f]) * (k - f))


def dist_summary(vals: list[float] | list[int]) -> dict:
    if not vals:
        return {"n": 0}
    fvals = [float(v) for v in vals]
    fvals.sort()
    return {
        "n": len(fvals),
        "min": fvals[0],
        "p10": percentile(fvals, 10),
        "p25": percentile(fvals, 25),
        "median": percentile(fvals, 50),
        "p75": percentile(fvals, 75),
        "p90": percentile(fvals, 90),
        "max": fvals[-1],
        "mean": sum(fvals) / len(fvals),
    }


def dump_json(name: str, payload: dict) -> Path:
    path = MEASUREMENTS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


if __name__ == "__main__":
    out = write_corpus_meta()
    m = corpus_meta()
    print(json.dumps(m.as_json(), indent=2))
    print(f"wrote {out}")
