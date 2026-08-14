"""
Reading and writing `data/ratings/`.

**No timestamp goes into a fit file.** Identical games must produce a
byte-identical file, so that two rebuilds can be diffed and a stale fit can be
spotted. The only clock that appears anywhere is in the Markdown leaderboard's
header, which is presentation, not state.

Fits are **one file per round**, under `data/ratings/fits/<round>.json`. That is
what keeps the byte-identical property worth having: a fit file carries the
covariance matrix, one new game in one round would otherwise rewrite every
round's covariance, and `load_fit` would parse 16 matrices to answer one
contrast. With the split, a new round changes exactly one file.

`stored_counts_digest(round=...)` says whether one round needs refitting;
`stored_rounds_digest()` answers the same question for the whole store.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from arena.paths import REPO_ROOT
from arena.records.ratings.cache import round_file_stem
from arena.records.ratings.fit import Estimate, LoadedFit, RatingFit
from arena.records.ratings.policy import Policy, Prior
from arena.records.ratings.rounds import RoundFits, RoundResult
from arena.records.reporting import round_section_lines
from arena.records.store import utc_now_iso

RATINGS_DIR = REPO_ROOT / "data" / "ratings"
FITS_DIRNAME = "fits"
LEADERBOARD_JSON = "leaderboard.json"
LEADERBOARD_MD = "leaderboard.md"

# The pooled fit's filename. Nothing writes it any more; the writer deletes it,
# because a file called `fit.json` holding one table over every round is the
# claim this refactor exists to withdraw.
LEGACY_FIT_JSON = "fit.json"

# v2: one file per round, with the round, its anchor tier and its scale token in
# the payload. There is no v1 reader — the directory is derived and gitignored,
# so the migration is a refit.
FIT_FORMAT_VERSION = 2
LEADERBOARD_FORMAT_VERSION = 2
LEADERBOARD_FORMAT = "per_round"

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
    """One fit's body. Contains no clock reading, by design."""
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
        "round": fit.round,
        "anchor": fit.anchor,
        "anchor_kind": fit.anchor_kind,
        "scale_id": fit.scale_id,
        "anchor_rating": fit.spec.anchor_rating,
        "prior": fit.prior.to_dict(),
        "policy": fit.policy.to_dict(),
        "counts_digest": fit.counts.digest,
        "connected": fit.connected,
        "components": [list(group) for group in fit.components],
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

    # Absent in files written before the connectivity guard. `LoadedFit` then
    # refuses cross-entity contrasts instead of assuming a connected pool.
    raw_components = data.get("components")
    components = (
        None
        if raw_components is None
        else tuple(tuple(group) for group in raw_components)
    )

    policy_data = data.get("policy", {})
    return LoadedFit(
        round=data.get("round"),
        anchor_kind=data.get("anchor_kind"),
        scale_id=data.get("scale_id"),
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
        components=components,
    )


def round_fit_payload(result: RoundResult) -> dict[str, Any]:
    """
    One rated round's fit file body.

    The fit's own payload plus the round-level provenance that is not a property
    of the solve: how many games were stored versus rated, and which engine eras
    the stored games came from.
    """
    if result.fit is None:
        raise ValueError(
            f"round {result.round!r} is unrated ({result.reason}); an unrated round "
            f"gets no fit file, only a status in {LEADERBOARD_JSON}"
        )
    payload = fit_payload(result.fit)
    payload.update(
        {
            "status": result.status,
            "round": result.round,
            "anchor_kind": result.anchor_kind,
            "scale_id": result.scale_id,
            "stored_games": result.stored_games,
            "engine_versions": list(result.engine_versions),
            "era_split": result.era_split,
        }
    )
    return payload


def round_leaderboard_entry(result: RoundResult) -> dict[str, Any]:
    """One element of `leaderboard.json`'s `rounds` array."""
    entry: dict[str, Any] = {
        "round": result.round,
        "status": result.status,
        "stored_games": result.stored_games,
        "rated_games": result.rated_games,
        "excluded": dict(result.excluded),
        "engine_versions": list(result.engine_versions),
        "era_split": result.era_split,
        "counts_digest": result.counts_digest,
    }
    if result.fit is None:
        entry["reason"] = result.reason
        return entry

    fit = result.fit
    entry.update(
        {
            "scale_id": result.scale_id,
            "anchor": result.anchor,
            "anchor_kind": result.anchor_kind,
            "anchor_rating": result.anchor_rating,
            "anchor_component": result.anchor_component,
            "connected": fit.connected,
            "components": [list(group) for group in fit.components],
            "seat_advantage": fit.seat_advantage.to_dict(),
            "draw_log_nu": fit.draw_log_nu.to_dict(),
            "solver": {
                "iterations": fit.solver.iterations,
                "max_abs_grad": fit.solver.max_abs_grad,
                "converged": fit.solver.converged,
            },
            "entities": [row.to_dict() for row in leaderboard_rows(fit)],
        }
    )
    return entry


def leaderboard_payload(fits: RoundFits) -> dict[str, Any]:
    """
    `leaderboard.json` at format version 2: per-round, with no pooled snapshot.

    `rounds` is an array rather than an object keyed by round name: `sort_keys`
    would re-sort an object anyway, and JSON object order is not a contract to
    any consumer. An array is ordered by construction — round name ascending —
    and each element self-identifies.
    """
    return {
        "version": LEADERBOARD_FORMAT_VERSION,
        "format": LEADERBOARD_FORMAT,
        "engine_era": fits.policy.engine_version,
        "policy": fits.policy.to_dict(),
        "prior": fits.prior.to_dict(),
        "rounds_digest": fits.rounds_digest,
        "rounds": [round_leaderboard_entry(result) for result in fits],
    }


def _write_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def fits_dir(ratings_dir: Path | None = None) -> Path:
    return (ratings_dir or RATINGS_DIR) / FITS_DIRNAME


def round_fit_path(round_name: str, ratings_dir: Path | None = None) -> Path:
    return fits_dir(ratings_dir) / f"{round_file_stem(round_name)}.json"


def available_rounds(ratings_dir: Path | None = None) -> list[str]:
    """Rounds that have a published fit file, from the files themselves."""
    directory = fits_dir(ratings_dir)
    if not directory.exists():
        return []
    names = []
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        names.append(data.get("round") or path.stem)
    return sorted(names)


def write_round_fits(
    fits: RoundFits, ratings_dir: Path | None = None
) -> tuple[list[Path], list[Path]]:
    """
    Write one fit file per rated round, and delete the files that no longer apply.

    `(written, pruned)`. Pruning matters more here than one directory over: a
    stale count cache is harmless, but a stale *published fit* is a table for a
    round that no longer exists, and nothing on it says so.
    """
    directory = fits_dir(ratings_dir)
    written = [
        _write_json(
            round_fit_payload(result), round_fit_path(result.round, ratings_dir)
        )
        for result in fits.rated
    ]
    keep = {path.name for path in written}
    pruned = []
    if directory.exists():
        for path in sorted(directory.glob("*.json")):
            if path.name not in keep:
                path.unlink()
                pruned.append(path)
    legacy = (ratings_dir or RATINGS_DIR) / LEGACY_FIT_JSON
    if legacy.exists():
        legacy.unlink()
        pruned.append(legacy)
    return written, pruned


def load_fit(ratings_dir: Path | None = None, *, round: str | None = None) -> LoadedFit | None:
    """
    Read one round's published fit.

    `round` is required: there is no pooled fit to fall back to, and returning
    "the first one" would hand back a table on a scale the caller did not ask
    for. Returns None when that round has no published fit.
    """
    if round is None:
        raise ValueError(
            "load_fit needs a round: fits are per-round and are not comparable "
            f"across rounds. Available: {', '.join(available_rounds(ratings_dir)) or 'none'}"
        )
    path = round_fit_path(round, ratings_dir)
    if not path.exists():
        return None
    return fit_from_payload(json.loads(path.read_text(encoding="utf-8")))


def stored_counts_digest(
    ratings_dir: Path | None = None, *, round: str | None = None
) -> str | None:
    """Whether one round needs a refit, without doing one."""
    if round is None:
        raise ValueError(
            "stored_counts_digest needs a round; use stored_rounds_digest() for "
            "the whole-store question"
        )
    path = round_fit_path(round, ratings_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("counts_digest")


def stored_rounds_digest(ratings_dir: Path | None = None) -> str | None:
    """Whether *anything* needs a refit: one read over the combined output."""
    path = (ratings_dir or RATINGS_DIR) / LEADERBOARD_JSON
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("rounds_digest")


BANNER = (
    "> **Ratings in two different tables are on two different scales. Do not",
    "> compare them.** Each table is anchored inside its own round and carries its",
    "> own `scale` token; two tables share a scale only if the tokens match, and no",
    "> two rounds ever do. A shared anchor fixes the additive constant, not the",
    "> conditions the games were played under: two byte-identical programs measured",
    "> in different rounds fitted **46 Elo apart** in this repo. Rank is per-round",
    "> and is never a result — decide from a pairwise contrast **inside one round**.",
    "> See [decision-rule.md](../../docs/arena/decision-rule.md).",
)

ROUNDS_INDEX_HEADER = (
    "| Round | Status | Rated / stored | Entities | Ranked | Groups | Anchor | Scale |",
    "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
)


def _anchor_link(name: str) -> str:
    """A GitHub-style heading anchor for a round name."""
    return name.lower().replace(" ", "-")


def rounds_index_lines(fits: RoundFits) -> list[str]:
    """
    The index: provenance and counts, never a rating and never a rank.

    A cross-round comparison has to be a deliberate act — open two sections and
    do the arithmetic — rather than reading two rows of one table.
    """
    lines = list(ROUNDS_INDEX_HEADER)
    for result in fits:
        link = f"[{result.round}](#{_anchor_link(result.round)})"
        if not result.rated:
            lines.append(
                f"| {link} | **unrated** | {result.rated_games:,} / "
                f"{result.stored_games:,} | — | — | — | — | — |"
            )
            continue
        ranked = result.ranked_count
        lines.append(
            f"| {link} | rated | {result.rated_games:,} / {result.stored_games:,} "
            f"| {len(result.entities)} | {'**0**' if ranked == 0 else ranked} "
            f"| {len(result.components)} | `{result.anchor}` ({result.anchor_kind}) "
            f"| `{result.scale_token}` |"
        )
    return lines


def leaderboard_markdown(
    fits: RoundFits,
    *,
    updated_at: str | None = None,
    only: Sequence[str] | None = None,
) -> str:
    """
    The whole document: banner, index, then one section per round in name order.

    There is no pooled ranked table, by design. `only` filters which sections are
    rendered — a reporting convenience; it never affects what is written.
    """
    selected = [r for r in fits if only is None or r.round in set(only)]
    era = fits.policy.engine_version
    lines = [
        "# Arena leaderboard",
        "",
        f"Updated: {updated_at or utc_now_iso()}",
        f"Engine era: `{era[:12]}`" if era else "Engine era: every era (`--all-eras`)",
        f"Rounds: {len(fits.rated)} rated, {len(fits.unrated)} unrated "
        f"· {fits.rated_games:,} rated games",
        "",
        "Each round below is an **independent** Bradley-Terry + Davidson + seat fit",
        "over that round's games only. No number here is fitted across rounds.",
        "",
        *BANNER,
        "",
        "## Rounds",
        "",
        *rounds_index_lines(fits),
    ]
    for result in selected:
        rows = leaderboard_rows(result.fit) if result.fit is not None else []
        lines += [
            "",
            "---",
            "",
            *round_section_lines(
                result,
                ranked=[r for r in rows if not r.provisional],
                provisional=[r for r in rows if r.provisional],
            ),
        ]
    lines.append("")
    return "\n".join(lines)


def write_leaderboard(
    fits: RoundFits, ratings_dir: Path | None = None, *, updated_at: str | None = None
) -> tuple[Path, Path]:
    directory = ratings_dir or RATINGS_DIR
    json_path = _write_json(leaderboard_payload(fits), directory / LEADERBOARD_JSON)
    md_path = directory / LEADERBOARD_MD
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(leaderboard_markdown(fits, updated_at=updated_at), encoding="utf-8")
    return json_path, md_path


def write_all(
    fits: RoundFits, ratings_dir: Path | None = None
) -> tuple[list[Path], Path, Path]:
    directory = ratings_dir or RATINGS_DIR
    fit_paths, _ = write_round_fits(fits, directory)
    json_path, md_path = write_leaderboard(fits, directory)
    return fit_paths, json_path, md_path


__all__ = [
    "FITS_DIRNAME",
    "LEADERBOARD_JSON",
    "LEADERBOARD_MD",
    "LeaderboardRow",
    "RATINGS_DIR",
    "available_rounds",
    "fits_dir",
    "leaderboard_markdown",
    "leaderboard_payload",
    "leaderboard_rows",
    "load_fit",
    "round_fit_path",
    "round_fit_payload",
    "rounds_index_lines",
    "split_entity",
    "stored_counts_digest",
    "stored_rounds_digest",
    "write_all",
    "write_leaderboard",
    "write_round_fits",
]
