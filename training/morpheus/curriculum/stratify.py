"""Global win/lose stratification for scraped reconstruction curricula."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


class StratifyError(ValueError):
    """Stratify pools cannot fill the requested quotas."""


@dataclass(frozen=True)
class GameCandidate:
    """One reconstruction game eligible for curriculum build."""

    path: Path
    game_id: str
    player: str
    outcome: str  # queried-player win|lose|draw
    source_label: str


@dataclass
class StratifyResult:
    """Kept candidates plus a machine-readable keep report."""

    kept: list[GameCandidate]
    n_pairs: int
    n_wins: int
    n_losses: int
    top_win_quota: int
    top_win_players: tuple[str, ...]
    top_win_fraction: float
    counts_by_player_outcome: dict[str, dict[str, int]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_pairs": self.n_pairs,
            "n_wins": self.n_wins,
            "n_losses": self.n_losses,
            "top_win_quota": self.top_win_quota,
            "top_win_players": list(self.top_win_players),
            "top_win_fraction": self.top_win_fraction,
            "counts_by_player_outcome": {
                player: dict(outcomes)
                for player, outcomes in sorted(self.counts_by_player_outcome.items())
            },
            "kept_game_ids": [c.game_id for c in self.kept],
            "notes": list(self.notes),
        }


def _take_sorted(pool: Sequence[GameCandidate], count: int) -> list[GameCandidate]:
    ordered = sorted(pool, key=lambda c: c.game_id)
    return list(ordered[:count])


def split_equal_thirds(total: int, players: Sequence[str]) -> dict[str, int]:
    """Split ``total`` across players in equal thirds; remainder by sorted name."""
    names = sorted(str(p) for p in players)
    if not names:
        raise StratifyError("top_win_players must be non-empty")
    if total < 0:
        raise StratifyError(f"top win quota must be >= 0, got {total}")
    base = total // len(names)
    rem = total % len(names)
    quotas = {name: base for name in names}
    for name in names[:rem]:
        quotas[name] += 1
    return quotas


def stratify_global_wdl(
    candidates: Sequence[GameCandidate],
    *,
    top_win_players: Sequence[str],
    top_win_fraction: float = 0.5,
    max_pairs: int | None = None,
) -> StratifyResult:
    """
    Keep equal wins and losses globally.

    ``n = n_lose`` (optionally capped by ``max_pairs``). Exactly
    ``floor(n * top_win_fraction)`` wins come from ``top_win_players``, split in
    equal thirds (remainder by sorted name). Remaining wins come from other
    players. Fail closed if any pool is too small.
    """
    top = tuple(sorted(str(p) for p in top_win_players))
    if not top:
        raise StratifyError("top_win_players must be non-empty")
    if not (0.0 < float(top_win_fraction) <= 1.0):
        raise StratifyError(
            f"top_win_fraction must be in (0, 1], got {top_win_fraction}"
        )

    losses = [c for c in candidates if c.outcome == "lose"]
    wins = [c for c in candidates if c.outcome == "win"]
    n = len(losses)
    if max_pairs is not None:
        n = min(n, int(max_pairs))
    if n < 1:
        raise StratifyError("need at least one decisive lose game")

    top_set = set(top)
    top_wins = [c for c in wins if c.player in top_set]
    other_wins = [c for c in wins if c.player not in top_set]

    top_quota = int(n * float(top_win_fraction) // 1)  # floor(n * fraction)
    # Plan locks floor(n/2) when fraction is 0.5; keep generic floor(n * f).
    other_quota = n - top_quota
    per_player = split_equal_thirds(top_quota, top)

    kept_wins: list[GameCandidate] = []
    for player, quota in per_player.items():
        pool = [c for c in top_wins if c.player == player]
        if len(pool) < quota:
            raise StratifyError(
                f"top player {player!r} has {len(pool)} wins; need {quota}"
            )
        kept_wins.extend(_take_sorted(pool, quota))

    if len(other_wins) < other_quota:
        raise StratifyError(
            f"non-top players have {len(other_wins)} wins; need {other_quota}"
        )
    kept_wins.extend(_take_sorted(other_wins, other_quota))

    kept_losses = _take_sorted(losses, n)
    if len(kept_losses) < n:
        raise StratifyError(f"only {len(kept_losses)} losses; need {n}")

    kept = sorted(kept_wins + kept_losses, key=lambda c: c.game_id)
    counts: dict[str, dict[str, int]] = {}
    for c in kept:
        bucket = counts.setdefault(c.player, {"win": 0, "lose": 0, "draw": 0})
        bucket[c.outcome] = int(bucket.get(c.outcome, 0)) + 1

    notes = [
        f"global 1:1 WDL with n_pairs={n}",
        f"top_win_fraction={top_win_fraction} top_quota={top_quota}",
        f"top_win_players={list(top)} equal thirds={per_player}",
    ]
    return StratifyResult(
        kept=kept,
        n_pairs=n,
        n_wins=len(kept_wins),
        n_losses=len(kept_losses),
        top_win_quota=top_quota,
        top_win_players=top,
        top_win_fraction=float(top_win_fraction),
        counts_by_player_outcome=counts,
        notes=notes,
    )


def candidate_from_meta(
    *,
    path: Path,
    game_id: str,
    meta: Mapping[str, Any],
    round_player: str | None = None,
    round_outcome: str | None = None,
) -> GameCandidate | None:
    """Build a candidate from corpus-index fields; skip draws."""
    outcome = str(meta.get("queried_outcome") or round_outcome or "").strip().lower()
    if outcome not in ("win", "lose"):
        return None
    player = str(meta.get("queried_player") or round_player or "").strip()
    if not player:
        return None
    label = str(meta.get("source_label") or f"{player}_reconstructions")
    return GameCandidate(
        path=Path(path),
        game_id=str(game_id),
        player=player,
        outcome=outcome,
        source_label=label,
    )
