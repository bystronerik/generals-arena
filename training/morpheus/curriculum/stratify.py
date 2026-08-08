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


def _max_feasible_pairs(
    *,
    n_lose: int,
    top_win_counts: Mapping[str, int],
    other_win_count: int,
    top_win_players: Sequence[str],
    top_win_fraction: float,
) -> int:
    """Largest n <= n_lose that top/other win pools can fill at the locked split."""
    if n_lose < 1:
        return 0
    lo = 1
    hi = int(n_lose)
    best = 0
    names = tuple(sorted(str(p) for p in top_win_players))
    while lo <= hi:
        mid = (lo + hi) // 2
        top_quota = int(mid * float(top_win_fraction) // 1)
        other_quota = mid - top_quota
        per_player = split_equal_thirds(top_quota, names)
        ok = other_win_count >= other_quota and all(
            int(top_win_counts.get(name, 0)) >= int(quota)
            for name, quota in per_player.items()
        )
        if ok:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def stratify_global_wdl(
    candidates: Sequence[GameCandidate],
    *,
    top_win_players: Sequence[str],
    top_win_fraction: float = 0.5,
    max_pairs: int | None = None,
) -> StratifyResult:
    """
    Keep equal wins and losses globally.

    ``n = n_lose`` (optionally capped by ``max_pairs``), then further capped to
    the largest n the top/other win pools can fill. Exactly
    ``floor(n * top_win_fraction)`` wins come from ``top_win_players``, split
    equally (remainder by sorted name). Remaining wins come from other players.
    Fail closed if no positive feasible n exists.
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
    n_lose = len(losses)
    n = n_lose
    if max_pairs is not None:
        n = min(n, int(max_pairs))
    if n < 1:
        raise StratifyError("need at least one decisive lose game")

    top_set = set(top)
    top_wins = [c for c in wins if c.player in top_set]
    other_wins = [c for c in wins if c.player not in top_set]
    top_win_counts = {
        name: sum(1 for c in top_wins if c.player == name) for name in top
    }
    feasible = _max_feasible_pairs(
        n_lose=n,
        top_win_counts=top_win_counts,
        other_win_count=len(other_wins),
        top_win_players=top,
        top_win_fraction=float(top_win_fraction),
    )
    if feasible < 1:
        raise StratifyError(
            "no feasible n: top/other win pools cannot fill floor(n * "
            f"{top_win_fraction}) equal split against n_lose={n_lose}"
        )
    capped_from = n
    n = feasible

    top_quota = int(n * float(top_win_fraction) // 1)  # floor(n * fraction)
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
        f"top_win_players={list(top)} equal_split={per_player}",
    ]
    if n < capped_from:
        notes.append(
            f"capped n from {capped_from} to {n} so top/other win pools can fill"
        )
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
