"""`python -m arena.records.ratings` — refit every round and report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from arena.records.ratings import io
from arena.records.ratings.policy import Policy, Prior, entity_key
from arena.records.ratings.rounds import RoundFits, fit_rounds
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
    The global anchor entity: the anchor bot's most recent registered version.

    Rounds that played it are anchored on it (tier 1), which makes their numbers
    reproducible statements about one pinned program. Rounds that did not are
    anchored locally instead — the anchor is never forced into a round it did not
    play. See `rounds.resolve_round_anchor`.

    Editing the anchor bot re-bases every scale, which is why
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


def refit(
    *,
    games_dir: Path | None = None,
    ratings_dir: Path | None = None,
    policy: Policy | None = None,
    prior: Prior | None = None,
    registry: Registry | None = None,
    anchor_bot: str = DEFAULT_ANCHOR_BOT,
    persist: bool = True,
) -> RoundFits:
    """
    Rebuild one independent fit per round from `data/games/`.

    There is no incremental path and no pooled fit. Sequential Elo's incremental
    update and its rebuild silently disagreed, because a path-dependent estimator
    has no single right answer; and a pooled batch fit has one answer that is not
    the question — it presents rounds that drift by more than the decision
    thresholds as one comparable column.

    Every write refits **every** round. Reading the store costs about 1.3 s and
    13 Newton solves cost 0.077 s, so there is nothing to buy by refitting a
    subset.
    """
    registry = registry or Registry()
    policy = policy or Policy(engine_version=engine_version())
    anchor = resolve_anchor(registry, anchor_bot)
    fits = fit_rounds(
        games_dir=games_dir,
        policy=policy,
        prior=prior,
        registry=registry,
        global_anchor=anchor,
    )
    if persist:
        io.write_all(fits, ratings_dir)
    return fits


def list_rounds_lines(fits: RoundFits) -> list[str]:
    """The round set as plain text: `--list-rounds`."""
    lines = [
        f"{'round':40} {'status':8} {'rated/stored':>14} {'ents':>5} {'grps':>5}  anchor"
    ]
    for result in fits:
        if not result.rated:
            anchor = f"— ({result.reason})"
        else:
            anchor = f"{result.anchor} ({result.anchor_kind})"
        counts = f"{result.rated_games}/{result.stored_games}"
        lines.append(
            f"{result.round:40} {result.status:8} {counts:>14} "
            f"{len(result.entities):>5} {len(result.components):>5}  {anchor}"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Refit arena ratings from stored data/games/ records: one "
            "independent fit per round."
        )
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
        "--sigma",
        type=float,
        default=Prior.sigma,
        help=f"prior standard deviation in Elo (default: {Prior.sigma})",
    )
    parser.add_argument(
        "--min-games",
        type=int,
        default=Policy.min_games_display,
        help=(
            f"below this an entity is provisional, counted per round "
            f"(default: {Policy.min_games_display})"
        ),
    )
    parser.add_argument(
        "--round",
        action="append",
        dest="rounds",
        metavar="NAME",
        default=None,
        help=(
            "report filter, repeatable: restricts --print and the console "
            "summary. Every write still refits every round"
        ),
    )
    parser.add_argument(
        "--list-rounds",
        action="store_true",
        help="print the round set and exit without writing",
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
    args = parser.parse_args(argv)

    registry = Registry()
    # Always exactly one era. `--era` names a past one; there is deliberately no
    # flag that disables the filter, because pooling across an engine bump moves
    # win probabilities and is a silent correctness bug.
    era = args.era or engine_version()
    policy = Policy(engine_version=era, min_games_display=args.min_games)
    try:
        fits = refit(
            games_dir=args.games_dir,
            ratings_dir=args.ratings_dir,
            policy=policy,
            prior=Prior(sigma=args.sigma),
            registry=registry,
            anchor_bot=args.anchor,
            persist=not (args.dry_run or args.list_rounds),
        )
    except MissingAnchor as exc:
        print(f"[ratings] {exc}")
        return 1

    if args.list_rounds:
        print("\n".join(list_rounds_lines(fits)))
        return 0

    unknown = sorted(set(args.rounds or ()) - set(fits.names))
    if unknown:
        print(
            f"[ratings] no such round(s): {', '.join(unknown)}; rounds are: "
            f"{', '.join(fits.names) or 'none'}",
            file=sys.stderr,
        )
        return 1
    shown = [r for r in fits if args.rounds is None or r.round in set(args.rounds)]

    print(
        f"[ratings] fitted {len(fits.rated)} round(s) over {fits.rated_games} "
        f"rated game(s); {len(fits.unrated)} round(s) unrated"
    )

    for result in shown:
        if not result.rated:
            print(
                f"[ratings] {result.round}: unrated ({result.reason}) — "
                f"{result.stored_games} stored game(s)",
                file=sys.stderr,
            )
            for reason, count in sorted(result.excluded.items()):
                # Same stream as the line above, so an unrated round's block stays
                # together instead of interleaving with the next round's summary.
                print(f"[ratings]   excluded {count} game(s): {reason}", file=sys.stderr)
            continue
        print(
            f"[ratings] {result.round}: {len(result.entities)} entit(ies), "
            f"{result.rated_games} of {result.stored_games} game(s), anchor "
            f"{result.anchor} ({result.anchor_kind}), scale {result.scale_token}"
        )
        if not result.connected:
            sizes = " + ".join(str(len(group)) for group in result.components)
            print(
                f"[ratings] WARNING: {result.round} is not connected — "
                f"{len(result.components)} groups ({sizes} entities) that share no "
                f"games. The anchor is in group {result.anchor_component}; across "
                f"groups the offset is prior, not evidence.",
                file=sys.stderr,
            )
        if result.era_split:
            print(
                f"[ratings] WARNING: {result.round} spans "
                f"{len(result.engine_versions)} engine eras; it is rated on "
                f"{era} only. Re-run the round.",
                file=sys.stderr,
            )
        if result.solver is not None and not result.solver.converged:
            print(
                f"[ratings] WARNING: the solver did not converge for "
                f"{result.round} — {result.solver.iterations} iterations, "
                f"max|grad| {result.solver.max_abs_grad:.2e}",
                file=sys.stderr,
            )

    # One unrated round never aborts the others — a refit is a pure function of
    # the stored games and can be re-run at any time. Everything unrated is a
    # roster or era problem the report names.
    if fits.names and not fits.rated:
        print(
            "[ratings] every round is unrated; nothing was rated",
            file=sys.stderr,
        )
        return 1

    if args.dry_run:
        print("[ratings] dry run: nothing written")
    else:
        print(
            f"[ratings] wrote {len(fits.rated)} fit file(s) under "
            f"{io.fits_dir(args.ratings_dir)}"
        )
        print(f"[ratings] wrote {args.ratings_dir / io.LEADERBOARD_JSON}")
        print(f"[ratings] wrote {args.ratings_dir / io.LEADERBOARD_MD}")

    if args.print_table:
        print()
        print(io.leaderboard_markdown(fits, only=args.rounds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
