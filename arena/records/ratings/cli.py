"""`python -m arena.records.ratings` — refit from stored games and report."""

from __future__ import annotations

import argparse
from pathlib import Path

from arena.records.ratings import io
from arena.records.ratings.cache import CACHE_DIRNAME, CacheStats, cached_count_table
from arena.records.ratings.counts import CountTable
from arena.records.ratings.fit import RatingFit, fit_ratings
from arena.records.ratings.lineage import lineage_table_lines
from arena.records.ratings.policy import Policy, Prior, entity_key
from arena.records.registry import Registry
from arena.records.store import GAMES_DIR, engine_version

# `bots/cm_expander/` wraps the same upstream ExpanderAgent the competition
# module ships, lives in the repo, hashes cleanly, and carries a "do not
# retune" comment. Pinning it is what turns free-floating strengths into
# statements anyone can reproduce.
DEFAULT_ANCHOR_BOT = "cm_expander"


class MissingAnchor(RuntimeError):
    """No anchor means no scale: `theta` would be free up to a constant."""


def resolve_anchor(registry: Registry, bot_id: str = DEFAULT_ANCHOR_BOT) -> str:
    """
    The anchor entity: the anchor bot's most recent registered version.

    Editing the anchor bot re-bases the whole scale, which is why
    `bots/cm_expander/agent.py` carries a "do not retune" comment.
    """
    entry = registry.load(bot_id)
    if entry is None or not entry.steps:
        raise MissingAnchor(
            f"anchor bot {bot_id!r} is not registered, so there is nothing to "
            f"pin the rating scale to; run "
            f"`python -m arena.records.registry --register {bot_id}` first"
        )
    return entity_key(bot_id, entry.steps[-1].content_hash)


def build_counts(
    *,
    games_dir: Path | None = None,
    ratings_dir: Path | None = None,
    policy: Policy,
    registry: Registry,
    anchor: str,
    use_cache: bool = True,
) -> tuple[CountTable, CacheStats]:
    """Aggregate every eligible game, reusing per-round caches where valid."""
    return cached_count_table(
        games_dir=games_dir,
        cache_dir=(ratings_dir or io.RATINGS_DIR) / CACHE_DIRNAME,
        policy=policy,
        registry=registry,
        extra_entities=[anchor],
        use_cache=use_cache,
    )


def refit(
    *,
    games_dir: Path | None = None,
    ratings_dir: Path | None = None,
    policy: Policy | None = None,
    prior: Prior | None = None,
    registry: Registry | None = None,
    anchor_bot: str = DEFAULT_ANCHOR_BOT,
    persist: bool = True,
    use_cache: bool = True,
) -> RatingFit:
    """
    Rebuild the whole fit from `data/games/`. There is no incremental path.

    Sequential Elo's incremental update and its rebuild silently disagreed,
    because a path-dependent estimator has no single right answer. A batch fit
    has one, and it is cheap enough to pay for on every write.
    """
    registry = registry or Registry()
    policy = policy or Policy(engine_version=engine_version())
    anchor = resolve_anchor(registry, anchor_bot)
    counts, stats = build_counts(
        games_dir=games_dir,
        ratings_dir=ratings_dir,
        policy=policy,
        registry=registry,
        anchor=anchor,
        use_cache=use_cache,
    )
    fit = fit_ratings(counts, prior=prior, anchor=anchor, policy=policy)
    fit.cache_stats = stats
    if persist:
        io.write_all(fit, ratings_dir)
    return fit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Refit arena ratings from stored data/games/ records."
    )
    parser.add_argument(
        "--games-dir",
        type=Path,
        default=GAMES_DIR,
        help=f"directory of game JSON files (default: {GAMES_DIR})",
    )
    parser.add_argument(
        "--ratings-dir",
        type=Path,
        default=io.RATINGS_DIR,
        help=f"output directory (default: {io.RATINGS_DIR})",
    )
    parser.add_argument(
        "--anchor",
        default=DEFAULT_ANCHOR_BOT,
        help=f"bot id pinned at the prior mean (default: {DEFAULT_ANCHOR_BOT})",
    )
    parser.add_argument(
        "--era",
        default=None,
        help="engine_version to rate (default: the current submodule HEAD)",
    )
    parser.add_argument(
        "--all-eras",
        action="store_true",
        help="do not filter on engine_version (unsound across an engine bump)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=Prior.sigma,
        help=f"prior standard deviation in Elo (default: {Prior.sigma})",
    )
    parser.add_argument(
        "--min-games",
        type=int,
        default=Policy.min_games_display,
        help=f"below this an entity is provisional (default: {Policy.min_games_display})",
    )
    parser.add_argument(
        "--lineage",
        metavar="BOT",
        default=None,
        help="also print one bot's improvement history",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        dest="print_table",
        help="print the Markdown leaderboard to stdout",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="fit and report without writing data/ratings/",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="re-read every game instead of reusing per-round count caches",
    )
    args = parser.parse_args(argv)

    registry = Registry()
    era = None if args.all_eras else (args.era or engine_version())
    policy = Policy(engine_version=era, min_games_display=args.min_games)
    try:
        fit = refit(
            games_dir=args.games_dir,
            ratings_dir=args.ratings_dir,
            policy=policy,
            prior=Prior(sigma=args.sigma),
            registry=registry,
            anchor_bot=args.anchor,
            persist=not args.dry_run,
            use_cache=not args.no_cache,
        )
    except MissingAnchor as exc:
        print(f"[ratings] {exc}")
        return 1

    print(
        f"[ratings] fitted {len(fit.entities)} entit(ies) over {fit.counts.games} game(s) "
        f"in {fit.solver.iterations} Newton step(s) (max|grad| {fit.solver.max_abs_grad:.2e})"
    )
    stats = fit.cache_stats
    if stats is not None and stats.rounds:
        print(
            f"[ratings] rounds: {stats.hits} cached, {stats.misses} re-aggregated"
        )
    print(f"[ratings] anchor {fit.anchor} = {fit.spec.anchor_rating:.1f}")
    print(
        f"[ratings] seat advantage {fit.seat_advantage.value:+.2f} "
        f"± {fit.seat_advantage.se:.2f} Elo"
    )
    for reason, count in sorted(fit.counts.excluded.items()):
        print(f"[ratings] excluded {count} game(s): {reason}")
    if not fit.solver.converged:
        print("[ratings] WARNING: the solver did not converge")
    if args.dry_run:
        print("[ratings] dry run: nothing written")
    else:
        print(f"[ratings] wrote {args.ratings_dir / io.FIT_JSON}")
        print(f"[ratings] wrote {args.ratings_dir / io.LEADERBOARD_JSON}")
        print(f"[ratings] wrote {args.ratings_dir / io.LEADERBOARD_MD}")

    if args.print_table:
        print()
        print(io.leaderboard_markdown(fit))
    if args.lineage:
        print()
        print("\n".join(lineage_table_lines(args.lineage, fit, registry)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
