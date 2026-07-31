"""
Reading and writing `data/ratings/`.

**No timestamp goes into `fit.json`.** Identical games must produce a
byte-identical file, so that two rebuilds can be diffed and a stale fit can be
spotted. The only clock that appears anywhere is in the Markdown leaderboard's
header, which is presentation, not state.

`counts_digest` lets a caller decide whether a refit is needed without doing
one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from arena.paths import REPO_ROOT
from arena.records.ratings.fit import Estimate, LoadedFit, RatingFit
from arena.records.ratings.policy import Policy, Prior
from arena.records.reporting import leaderboard_table_lines
from arena.records.store import utc_now_iso

RATINGS_DIR = REPO_ROOT / "data" / "ratings"
FIT_JSON = "fit.json"
LEADERBOARD_JSON = "leaderboard.json"
LEADERBOARD_MD = "leaderboard.md"

FIT_FORMAT_VERSION = 1

# Ratings are published to 2 dp, which makes the published numbers literally
# identical across runs even though the fitted floats agree only to ~1e-9.
PUBLISHED_DECIMALS = 2


@dataclass(frozen=True)
class LeaderboardRow:
    """One rendered row. Presentation only — decisions use contrasts."""

    rank: int
    entity: str
    bot_id: str
    content_hash: str
    rating: float
    se: float
    ci_low: float
    ci_high: float
    games: int
    wins: int
    losses: int
    draws: int
    provisional: bool
    # Connectivity group. Rows in different groups are not comparable: no chain
    # of games links them, so their difference is prior, not evidence.
    component: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "entity": self.entity,
            "bot_id": self.bot_id,
            "content_hash": self.content_hash,
            "rating": round(self.rating, PUBLISHED_DECIMALS),
            "se": round(self.se, PUBLISHED_DECIMALS),
            "ci_low": round(self.ci_low, PUBLISHED_DECIMALS),
            "ci_high": round(self.ci_high, PUBLISHED_DECIMALS),
            "games": self.games,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "provisional": self.provisional,
            "component": self.component,
        }


def split_entity(entity: str) -> tuple[str, str]:
    bot_id, _, content_hash = entity.partition("@")
    return bot_id, content_hash


def leaderboard_rows(fit: RatingFit) -> list[LeaderboardRow]:
    """
    Ranked entities first, provisional ones after — never interleaved.

    Provisional entities keep rank 0: they are in the fit, so they moved every
    other rating, but a rank next to a 20-game record would invite exactly the
    comparison the minimum-games gate exists to prevent.
    """
    rows: list[LeaderboardRow] = []
    for rank, entity in enumerate(fit.ranked(), start=1):
        rows.append(_row(fit, entity, rank))
    for entity in fit.provisional_entities():
        rows.append(_row(fit, entity, 0))
    return rows


def _row(fit: RatingFit, entity: str, rank: int) -> LeaderboardRow:
    bot_id, content_hash = split_entity(entity)
    low, high = fit.interval(entity)
    wins, losses, draws = fit.record(entity)
    return LeaderboardRow(
        rank=rank,
        entity=entity,
        bot_id=bot_id,
        content_hash=content_hash,
        rating=fit.rating(entity),
        se=fit.se(entity),
        ci_low=low,
        ci_high=high,
        games=wins + losses + draws,
        wins=wins,
        losses=losses,
        draws=draws,
        provisional=fit.provisional(entity),
        component=fit.component_of(entity),
    )


def fit_payload(fit: RatingFit) -> dict[str, Any]:
    """The `fit.json` body. Contains no clock reading, by design."""
    order = list(fit.entities)
    lower: list[float] = []
    for i in range(len(order)):
        for j in range(i + 1):
            lower.append(float(fit._covariance[i, j]))

    entities = []
    for entity in order:
        bot_id, content_hash = split_entity(entity)
        wins, losses, draws = fit.record(entity)
        entities.append(
            {
                "entity": entity,
                "bot_id": bot_id,
                "content_hash": content_hash,
                "rating": round(fit.rating(entity), PUBLISHED_DECIMALS),
                "se": round(fit.se(entity), PUBLISHED_DECIMALS),
                "games": wins + losses + draws,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "provisional": fit.provisional(entity),
            }
        )

    return {
        "version": FIT_FORMAT_VERSION,
        "anchor": fit.anchor,
        "anchor_rating": fit.spec.anchor_rating,
        "prior": fit.prior.to_dict(),
        "policy": fit.policy.to_dict(),
        "counts_digest": fit.counts.digest,
        "solver": {
            "iterations": fit.solver.iterations,
            "max_abs_grad": fit.solver.max_abs_grad,
            "converged": fit.solver.converged,
        },
        "excluded": dict(fit.counts.excluded),
        "rated_games": fit.counts.games,
        "seat_advantage": {
            "value": round(fit.seat_advantage.value, 4),
            "se": round(fit.seat_advantage.se, 4),
        },
        "draw_log_nu": {
            "value": round(fit.draw_log_nu.value, 4),
            "se": round(fit.draw_log_nu.se, 4),
        },
        "entities": entities,
        "covariance": {"order": order, "lower_triangle": lower},
    }


def fit_from_payload(data: dict[str, Any]) -> LoadedFit:
    order = list(data["covariance"]["order"])
    lower = data["covariance"]["lower_triangle"]
    size = len(order)
    covariance = np.zeros((size, size))
    k = 0
    for i in range(size):
        for j in range(i + 1):
            covariance[i, j] = covariance[j, i] = lower[k]
            k += 1

    ratings, errors, records = {}, {}, {}
    for row in data["entities"]:
        entity = row["entity"]
        ratings[entity] = float(row["rating"])
        errors[entity] = float(row["se"])
        records[entity] = (int(row["wins"]), int(row["losses"]), int(row["draws"]))

    policy_data = data.get("policy", {})
    return LoadedFit(
        anchor=data["anchor"],
        entities=order,
        ratings=ratings,
        standard_errors=errors,
        records=records,
        covariance=covariance,
        seat_advantage=Estimate(**data["seat_advantage"]),
        draw_log_nu=Estimate(**data["draw_log_nu"]),
        counts_digest=data["counts_digest"],
        prior=Prior(**data.get("prior", {})),
        policy=Policy(
            modes=tuple(policy_data.get("modes", ("competition",))),
            engine_version=policy_data.get("engine_version"),
            require_registered=policy_data.get("require_registered", True),
            include_self_play=policy_data.get("include_self_play", True),
            min_games_display=policy_data.get("min_games_display", 30),
        ),
        excluded=dict(data.get("excluded", {})),
    )


def _write_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_fit(fit: RatingFit, ratings_dir: Path | None = None) -> Path:
    return _write_json(fit_payload(fit), (ratings_dir or RATINGS_DIR) / FIT_JSON)


def load_fit(ratings_dir: Path | None = None) -> LoadedFit | None:
    path = (ratings_dir or RATINGS_DIR) / FIT_JSON
    if not path.exists():
        return None
    return fit_from_payload(json.loads(path.read_text(encoding="utf-8")))


def stored_counts_digest(ratings_dir: Path | None = None) -> str | None:
    """Whether a refit is needed, without doing one."""
    path = (ratings_dir or RATINGS_DIR) / FIT_JSON
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("counts_digest")


def _connectivity_lines(fit: RatingFit) -> list[str]:
    """The warning block, or nothing at all when the pool is connected."""
    if fit.connected:
        return []
    sizes = ", ".join(str(len(group)) for group in fit.components)
    return [
        f"> **This pool is not connected: {len(fit.components)} groups "
        f"({sizes} entities).**",
        ">",
        "> Groups share no games, so nothing links their scales — the offset",
        "> between them comes from the prior, not from evidence. Ratings and",
        "> ranks are meaningful **only within one group**. A contrast across",
        "> groups reports an infinite interval and `P(better) = 0.50`.",
        ">",
        "> Fix it by playing games between the groups, not by comparing anyway.",
        "",
    ]


def leaderboard_markdown(fit: RatingFit, *, updated_at: str | None = None) -> str:
    rows = leaderboard_rows(fit)
    ranked = [r for r in rows if not r.provisional]
    provisional = [r for r in rows if r.provisional]
    seat = fit.seat_advantage
    lines = [
        "# Arena leaderboard",
        "",
        f"Updated: {updated_at or utc_now_iso()}",
        f"Rated games: {fit.counts.games}",
        f"Anchor: `{fit.anchor}` pinned at {fit.spec.anchor_rating:.1f}",
        f"Seat-A advantage: {seat.value:+.1f} ± {seat.se:.1f} Elo",
        f"Draw log-nu: {fit.draw_log_nu.value:.3f} ± {fit.draw_log_nu.se:.3f}",
        "",
        "Ratings are Bradley-Terry + Davidson draws + a shared seat term, fitted",
        "jointly over every eligible game. Decide from the pairwise contrast, never",
        "from rank: see [decision-rule.md](../../docs/arena/decision-rule.md).",
        "",
        *_connectivity_lines(fit),
        *leaderboard_table_lines(ranked, split=not fit.connected),
    ]
    if provisional:
        lines += [
            "",
            f"## Provisional (< {fit.policy.min_games_display} games)",
            "",
            "In the fit, but not ranked and not eligible as a decision baseline.",
            "",
            *leaderboard_table_lines(provisional, split=not fit.connected),
        ]
    lines.append("")
    return "\n".join(lines)


def write_leaderboard(
    fit: RatingFit, ratings_dir: Path | None = None, *, updated_at: str | None = None
) -> tuple[Path, Path]:
    directory = ratings_dir or RATINGS_DIR
    rows = leaderboard_rows(fit)
    payload = {
        "anchor": fit.anchor,
        "rated_games": fit.counts.games,
        "counts_digest": fit.counts.digest,
        "min_games_display": fit.policy.min_games_display,
        "seat_advantage": fit.seat_advantage.to_dict(),
        "connected": fit.connected,
        "components": [list(group) for group in fit.components],
        "entities": [row.to_dict() for row in rows],
    }
    json_path = _write_json(payload, directory / LEADERBOARD_JSON)
    md_path = directory / LEADERBOARD_MD
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(leaderboard_markdown(fit, updated_at=updated_at), encoding="utf-8")
    return json_path, md_path


def write_all(
    fit: RatingFit, ratings_dir: Path | None = None
) -> tuple[Path, Path, Path]:
    directory = ratings_dir or RATINGS_DIR
    fit_path = write_fit(fit, directory)
    json_path, md_path = write_leaderboard(fit, directory)
    return fit_path, json_path, md_path


__all__ = [
    "FIT_JSON",
    "LEADERBOARD_JSON",
    "LEADERBOARD_MD",
    "LeaderboardRow",
    "RATINGS_DIR",
    "leaderboard_markdown",
    "leaderboard_rows",
    "load_fit",
    "split_entity",
    "stored_counts_digest",
    "write_all",
    "write_fit",
    "write_leaderboard",
]
