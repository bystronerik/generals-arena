"""Coverage classification for Morpheus bootstrap trajectories."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from arena.records.trajectories import Trajectory, read_trajectory, replay_states
from training.morpheus.corpus.panel import BOARD_SIDE_MAX, BOARD_SIDE_MIN

# Turn bands for Part 03 army-normalization and curriculum timing.
TURN_BANDS: tuple[tuple[str, int, int], ...] = (
    ("1-200", 1, 200),
    ("201-400", 201, 400),
    ("401-600", 401, 600),
    ("601-800", 601, 800),
    ("801-1000", 801, 1000),
    ("1001-1200", 1001, 1200),
)

DEATHTOUCH_TURN = 800
SOURCE_LABEL_KEY = "source_label"


@dataclass
class TrajectoryCoverage:
    """One trajectory's coverage flags. Missing is None/False, never invented zero."""

    game_id: str
    mode: str
    engine_version: str
    bot_a: str
    bot_b: str
    source_label: str
    H: int
    W: int
    board_size: str
    turns: int
    turn_band: str | None
    outcome: str
    decisive: bool
    contact: bool
    first_contact_turn: int | None
    sight: bool
    first_sight_turn: int | None
    castle: bool
    first_castle_turn: int | None
    deathtouch: bool
    forced_mismatch_eligible: bool


@dataclass
class CoverageReport:
    """Aggregated coverage for a trajectory directory."""

    trajectories: list[TrajectoryCoverage] = field(default_factory=list)
    engine_versions: Counter = field(default_factory=Counter)
    modes: Counter = field(default_factory=Counter)
    source_labels: Counter = field(default_factory=Counter)
    board_sizes: Counter = field(default_factory=Counter)
    turn_bands: Counter = field(default_factory=Counter)
    outcomes: Counter = field(default_factory=Counter)
    contact: int = 0
    sight: int = 0
    castle: int = 0
    deathtouch: int = 0
    decisive: int = 0
    forced_mismatch_eligible: int = 0
    verify_failures: list[str] = field(default_factory=list)

    def expected_board_sizes(self) -> list[str]:
        return [
            f"{h}x{w}"
            for h in range(BOARD_SIDE_MIN, BOARD_SIDE_MAX + 1)
            for w in range(BOARD_SIDE_MIN, BOARD_SIDE_MAX + 1)
        ]

    def expected_turn_bands(self) -> list[str]:
        return [name for name, _, _ in TURN_BANDS]

    def expected_outcomes(self) -> list[str]:
        return ["a", "b", "draw"]

    def missing_classes(self) -> dict[str, list[str]]:
        """Name every required class or event with zero observations."""
        missing: dict[str, list[str]] = {}
        board_missing = [s for s in self.expected_board_sizes() if self.board_sizes[s] == 0]
        if board_missing:
            missing["board_size"] = board_missing
        band_missing = [b for b in self.expected_turn_bands() if self.turn_bands[b] == 0]
        if band_missing:
            missing["turn_band"] = band_missing
        outcome_missing = [o for o in self.expected_outcomes() if self.outcomes[o] == 0]
        if outcome_missing:
            missing["outcome"] = outcome_missing
        events = {
            "contact": self.contact,
            "sight": self.sight,
            "castle": self.castle,
            "deathtouch": self.deathtouch,
            "decisive": self.decisive,
            "forced_mismatch_eligible": self.forced_mismatch_eligible,
        }
        event_missing = [name for name, count in events.items() if count == 0]
        if event_missing:
            missing["events"] = event_missing
        return missing


def turn_band_for(turns: int) -> str | None:
    for name, lo, hi in TURN_BANDS:
        if lo <= turns <= hi:
            return name
    return None


def board_size_key(H: int, W: int) -> str:
    return f"{int(H)}x{int(W)}"


def _ownership_contact(ownership: np.ndarray) -> bool:
    """True when seats 0 and 1 own orthogonally adjacent cells."""
    own0 = ownership == 0
    own1 = ownership == 1
    if not own0.any() or not own1.any():
        return False
    # Shift seat-0 mask and test overlap with seat-1.
    up = np.zeros_like(own0)
    up[1:, :] = own0[:-1, :]
    down = np.zeros_like(own0)
    down[:-1, :] = own0[1:, :]
    left = np.zeros_like(own0)
    left[:, 1:] = own0[:, :-1]
    right = np.zeros_like(own0)
    right[:, :-1] = own0[:, 1:]
    touch = (up | down | left | right) & own1
    return bool(touch.any())


def _enemy_general_visible(obs) -> bool:
    generals = np.asarray(obs.generals, dtype=bool)
    opponent = np.asarray(obs.opponent_cells, dtype=bool)
    return bool((generals & opponent).any())


def classify_trajectory(
    traj: Trajectory,
    *,
    source_label: str,
    scan_events: bool = True,
) -> TrajectoryCoverage:
    """
    Classify one trajectory.

    Header fields never invent coverage. Event flags require a same-era replay
    when `scan_events` is True.
    """
    end = traj.end
    turns = int(end.get("turns", len(traj.frames)))
    winner = str(end.get("winner", "draw"))
    outcome = winner if winner in ("a", "b") else "draw"
    H = int(traj.header.get("H") or 0)
    W = int(traj.header.get("W") or 0)

    contact = False
    first_contact: int | None = None
    sight = False
    first_sight: int | None = None
    castle = False
    first_castle: int | None = None

    if scan_events:
        from generals.core.game import get_observation

        for turn, state, _info in replay_states(traj):
            ownership = np.asarray(state.ownership)
            if first_contact is None and _ownership_contact(ownership):
                first_contact = turn
                contact = True
            castles = np.asarray(state.castles, dtype=bool)
            if first_castle is None and castles.any():
                first_castle = turn
                castle = True
            if first_sight is None:
                for player in (0, 1):
                    obs = get_observation(state, player)
                    if _enemy_general_visible(obs):
                        first_sight = turn
                        sight = True
                        break
            # Once every event has fired, stop early.
            if contact and sight and castle and turn >= DEATHTOUCH_TURN:
                break

    deathtouch = turns >= DEATHTOUCH_TURN
    decisive = outcome in ("a", "b")
    # Part 05 forced-mismatch measurement needs contact so an enemy action
    # proposal can disagree with the recorded move.
    forced_mismatch_eligible = contact

    return TrajectoryCoverage(
        game_id=traj.game_id,
        mode=str(traj.header.get("mode", "")),
        engine_version=traj.engine_version,
        bot_a=str(traj.header.get("bot_a", "")),
        bot_b=str(traj.header.get("bot_b", "")),
        source_label=source_label,
        H=H,
        W=W,
        board_size=board_size_key(H, W) if H and W else "unknown",
        turns=turns,
        turn_band=turn_band_for(turns),
        outcome=outcome,
        decisive=decisive,
        contact=contact,
        first_contact_turn=first_contact,
        sight=sight,
        first_sight_turn=first_sight,
        castle=castle,
        first_castle_turn=first_castle,
        deathtouch=deathtouch,
        forced_mismatch_eligible=forced_mismatch_eligible,
    )


def iter_trajectory_paths(directory: Path) -> Iterator[Path]:
    yield from sorted(directory.glob("*.traj.jsonl.gz"))


def load_source_index(directory: Path) -> dict[str, str]:
    """Optional per-game source labels written beside trajectories."""
    path = directory / "corpus-index.json"
    if not path.is_file():
        return {}
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    games = data.get("games") or {}
    return {str(gid): str(meta.get(SOURCE_LABEL_KEY, "")) for gid, meta in games.items()}


def build_coverage(
    directory: Path,
    *,
    default_source_label: str,
    scan_events: bool = True,
) -> CoverageReport:
    """Scan every trajectory under `directory` and aggregate coverage."""
    report = CoverageReport()
    index = load_source_index(directory)
    for path in iter_trajectory_paths(directory):
        traj = read_trajectory(path)
        label = index.get(traj.game_id) or default_source_label
        if not label:
            label = "unlabeled"
        cov = classify_trajectory(traj, source_label=label, scan_events=scan_events)
        report.trajectories.append(cov)
        report.engine_versions[cov.engine_version] += 1
        report.modes[cov.mode] += 1
        report.source_labels[cov.source_label] += 1
        report.board_sizes[cov.board_size] += 1
        if cov.turn_band is not None:
            report.turn_bands[cov.turn_band] += 1
        report.outcomes[cov.outcome] += 1
        report.contact += int(cov.contact)
        report.sight += int(cov.sight)
        report.castle += int(cov.castle)
        report.deathtouch += int(cov.deathtouch)
        report.decisive += int(cov.decisive)
        report.forced_mismatch_eligible += int(cov.forced_mismatch_eligible)
    return report


def coverage_to_dict(report: CoverageReport) -> dict[str, Any]:
    return {
        "trajectory_count": len(report.trajectories),
        "engine_versions": dict(report.engine_versions),
        "modes": dict(report.modes),
        "source_labels": dict(report.source_labels),
        "board_sizes": dict(report.board_sizes),
        "turn_bands": dict(report.turn_bands),
        "outcomes": dict(report.outcomes),
        "events": {
            "contact": report.contact,
            "sight": report.sight,
            "castle": report.castle,
            "deathtouch": report.deathtouch,
            "decisive": report.decisive,
            "forced_mismatch_eligible": report.forced_mismatch_eligible,
        },
        "missing_classes": report.missing_classes(),
        "games": [
            {
                "game_id": g.game_id,
                "source_label": g.source_label,
                "mode": g.mode,
                "engine_version": g.engine_version,
                "bot_a": g.bot_a,
                "bot_b": g.bot_b,
                "board_size": g.board_size,
                "turns": g.turns,
                "turn_band": g.turn_band,
                "outcome": g.outcome,
                "decisive": g.decisive,
                "contact": g.contact,
                "first_contact_turn": g.first_contact_turn,
                "sight": g.sight,
                "first_sight_turn": g.first_sight_turn,
                "castle": g.castle,
                "first_castle_turn": g.first_castle_turn,
                "deathtouch": g.deathtouch,
                "forced_mismatch_eligible": g.forced_mismatch_eligible,
            }
            for g in report.trajectories
        ],
    }
