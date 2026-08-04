"""Opponent mixture sampler: checkpoint league vs fixed bot panel (Part 11).

Default: 70% league, 30% fixed panel. Replace only from held-out exploitability,
cycling, opponent coverage, decisive rate, and pairwise arena contrast.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from training.morpheus.self_play.league import League, LeagueError, Snapshot

DEFAULT_LEAGUE_WEIGHT = 0.7
DEFAULT_PANEL_WEIGHT = 0.3
SOURCE_LEAGUE = "league"
SOURCE_FIXED_PANEL = "fixed_panel"
EXCLUDED_PANEL_TOKENS = frozenset(
    {"classic_duel", "yankee", "resbot", "remote", "classic"}
)


class SamplerError(ValueError):
    """Opponent mixture config or sample is invalid."""


@dataclass(frozen=True)
class PanelMember:
    bot_id: str
    content_hash: str | None = None
    role: str | None = None
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "bot_id": self.bot_id,
            "content_hash": self.content_hash,
            "role": self.role,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PanelMember:
        return cls(
            bot_id=str(data["bot_id"]),
            content_hash=(
                str(data["content_hash"]) if data.get("content_hash") is not None else None
            ),
            role=(str(data["role"]) if data.get("role") is not None else None),
            weight=float(data.get("weight", 1.0)),
        )


@dataclass(frozen=True)
class SeatSpec:
    """One seat identity for a sampled matchup."""

    kind: str  # "checkpoint" | "stub" | "panel"
    seat: int
    snapshot_id: str | None = None
    role: str | None = None
    content_digest: str | None = None
    path: str | None = None
    bot_id: str | None = None
    content_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "seat": self.seat,
            "snapshot_id": self.snapshot_id,
            "role": self.role,
            "content_digest": self.content_digest,
            "path": self.path,
            "bot_id": self.bot_id,
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True)
class Matchup:
    source: str
    learner_seat: int
    seats: tuple[SeatSpec, SeatSpec]
    map_seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "learner_seat": self.learner_seat,
            "map_seed": self.map_seed,
            "seats": [s.to_dict() for s in self.seats],
        }


@dataclass(frozen=True)
class MixtureConfig:
    league_weight: float = DEFAULT_LEAGUE_WEIGHT
    panel_weight: float = DEFAULT_PANEL_WEIGHT
    panel_path: str | None = None
    panel_members: tuple[PanelMember, ...] = ()

    def __post_init__(self) -> None:
        lw = float(self.league_weight)
        pw = float(self.panel_weight)
        if lw < 0.0 or pw < 0.0:
            raise SamplerError("mixture weights must be >= 0")
        if lw + pw <= 0.0:
            raise SamplerError("mixture weights must sum to a positive value")

    @property
    def league_prob(self) -> float:
        total = float(self.league_weight) + float(self.panel_weight)
        return float(self.league_weight) / total

    def to_dict(self) -> dict[str, Any]:
        return {
            "league_weight": self.league_weight,
            "panel_weight": self.panel_weight,
            "panel_path": self.panel_path,
            "panel_members": [m.to_dict() for m in self.panel_members],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MixtureConfig:
        members = tuple(
            PanelMember.from_dict(row) for row in (data.get("panel_members") or [])
        )
        return cls(
            league_weight=float(data.get("league_weight", DEFAULT_LEAGUE_WEIGHT)),
            panel_weight=float(data.get("panel_weight", DEFAULT_PANEL_WEIGHT)),
            panel_path=(
                str(data["panel_path"]) if data.get("panel_path") is not None else None
            ),
            panel_members=members,
        )


def _is_excluded_bot(bot_id: str) -> bool:
    lower = bot_id.lower()
    return any(token in lower for token in EXCLUDED_PANEL_TOKENS)


def load_panel_members(path: Path) -> list[PanelMember]:
    """Load panel members from bootstrap-panel.json (or a smoke subset)."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    members: list[PanelMember] = []
    for row in data.get("members") or []:
        bot_id = str(row["bot_id"])
        if _is_excluded_bot(bot_id):
            continue
        members.append(
            PanelMember(
                bot_id=bot_id,
                content_hash=(
                    str(row["content_hash"]) if "content_hash" in row else None
                ),
                role=(str(row["role"]) if "role" in row else None),
                weight=float(row.get("weight", 1.0)),
            )
        )
    if not members:
        raise SamplerError(f"panel {path} has no usable members")
    return members


def resolve_mixture(config: MixtureConfig, *, repo_root: Path | None = None) -> MixtureConfig:
    """Fill panel_members from panel_path when the list is empty."""
    if config.panel_members:
        return config
    if not config.panel_path:
        raise SamplerError("mixture needs panel_members or panel_path")
    path = Path(config.panel_path)
    if not path.is_file() and repo_root is not None:
        path = Path(repo_root) / config.panel_path
    members = tuple(load_panel_members(path))
    return MixtureConfig(
        league_weight=config.league_weight,
        panel_weight=config.panel_weight,
        panel_path=str(path),
        panel_members=members,
    )


def _seat_from_snapshot(snapshot: Snapshot, seat: int) -> SeatSpec:
    kind = "stub" if snapshot.kind == "stub" else "checkpoint"
    return SeatSpec(
        kind=kind,
        seat=seat,
        snapshot_id=snapshot.snapshot_id,
        role=snapshot.role,
        content_digest=snapshot.content_digest,
        path=snapshot.path,
    )


def _seat_from_panel(member: PanelMember, seat: int) -> SeatSpec:
    return SeatSpec(
        kind="panel",
        seat=seat,
        bot_id=member.bot_id,
        content_hash=member.content_hash,
        role=member.role,
    )


def _sample_weighted(rng, items: Sequence[Any], weights: Sequence[float]) -> Any:
    total = float(sum(max(float(w), 0.0) for w in weights))
    if total <= 0.0:
        index = int(rng.integers(0, len(items)))
        return items[index]
    probs = [max(float(w), 0.0) / total for w in weights]
    return items[int(rng.choice(len(items), p=probs))]


def sample_matchup(
    rng,
    league: League,
    mixture: MixtureConfig,
    *,
    map_seed: int | None = None,
) -> Matchup:
    """Sample one matchup and random seat / map assignment."""
    if not league.frozen:
        raise LeagueError("sample_matchup requires a frozen league epoch")
    mixture = resolve_mixture(mixture)
    learner = league.learner()
    seed = int(map_seed) if map_seed is not None else int(rng.integers(0, 2**31 - 1))
    learner_seat = int(rng.integers(0, 2))
    opponent_seat = 1 - learner_seat

    if float(rng.random()) < mixture.league_prob:
        opponent = league.sample_opponent(rng, exclude_ids=())
        # Prefer non-learner opponents when available.
        if len(league.snapshots()) > 1:
            opponent = league.sample_opponent(
                rng, exclude_ids={learner.snapshot_id}
            )
        seats = [None, None]
        seats[learner_seat] = _seat_from_snapshot(learner, learner_seat)
        seats[opponent_seat] = _seat_from_snapshot(opponent, opponent_seat)
        return Matchup(
            source=SOURCE_LEAGUE,
            learner_seat=learner_seat,
            seats=(seats[0], seats[1]),  # type: ignore[arg-type]
            map_seed=seed,
        )

    member = _sample_weighted(
        rng,
        mixture.panel_members,
        [m.weight for m in mixture.panel_members],
    )
    seats = [None, None]
    seats[learner_seat] = _seat_from_snapshot(learner, learner_seat)
    seats[opponent_seat] = _seat_from_panel(member, opponent_seat)
    return Matchup(
        source=SOURCE_FIXED_PANEL,
        learner_seat=learner_seat,
        seats=(seats[0], seats[1]),  # type: ignore[arg-type]
        map_seed=seed,
    )


def mixture_proportions(matchups: Sequence[Matchup]) -> dict[str, float]:
    """Empirical source proportions for a batch of matchups."""
    n = len(matchups)
    if n == 0:
        return {SOURCE_LEAGUE: 0.0, SOURCE_FIXED_PANEL: 0.0}
    league_n = sum(1 for m in matchups if m.source == SOURCE_LEAGUE)
    panel_n = n - league_n
    return {
        SOURCE_LEAGUE: league_n / n,
        SOURCE_FIXED_PANEL: panel_n / n,
    }
