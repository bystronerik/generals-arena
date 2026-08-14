"""
Order invariance, count-table canonicality, pooling policy, and artifacts.

The property this suite exists to protect is R1: the fit is a pure function of
the *set* of stored games. The old suite asserted idempotence by `game_id`, a
much weaker property that sequential Elo passed while being badly broken.

Since ratings became per-round, R1 has a scope: each round's fit is a pure
function of *that round's* games, and nothing else in the store can move it.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from arena.records.ratings import io
from arena.records.ratings.counts import count_table, merge
from arena.records.ratings.fit import fit_ratings
from arena.records.ratings.policy import (
    ENGINE_MISMATCH,
    MODE_NOT_ELIGIBLE,
    SELF_PLAY_EXCLUDED,
    UNREGISTERED_HASH,
    Policy,
    entity_key,
)
from arena.records.ratings.rounds import (
    ANCHOR_GLOBAL,
    ANCHOR_ROUND_LOCAL,
    NO_ELIGIBLE_GAMES,
    RoundFits,
    fit_rounds,
    resolve_round_anchor,
)
from arena.records.store import GameRecord, save_game

ENGINE = "9e3b9d13cca51caa1bb07db48bb85c9e90ce0462"
OTHER_ENGINE = "0" * 40
ANCHOR = entity_key("smoke", "aaaaaaaaaaaa")

HASHES = {
    "smoke": "aaaaaaaaaaaa",
    "blitz": "bbbbbbbbbbbb",
    "aegis": "cccccccccccc",
    "metro": "dddddddddddd",
}


class FakeRegistry:
    """Just the surface `policy` and `cache` use, so these tests need no git."""

    def __init__(self, known: dict[str, str] | None = None) -> None:
        self.known = dict(HASHES if known is None else known)

    def is_registered(self, bot_id: str, content_hash: str) -> bool:
        return self.known.get(bot_id) == content_hash

    def bot_ids(self) -> list[str]:
        return sorted(self.known)

    def load(self, bot_id: str):
        if bot_id not in self.known:
            return None
        return SimpleNamespace(
            bot_id=bot_id,
            versions=[SimpleNamespace(content_hash=self.known[bot_id])],
            steps=[],
        )


def record(
    bot_a: str,
    bot_b: str,
    winner: str,
    *,
    index: int = 0,
    mode: str = "competition",
    engine: str = ENGINE,
    hash_a: str | None = None,
    hash_b: str | None = None,
    truncated: bool = False,
) -> GameRecord:
    return GameRecord(
        game_id=f"g{index}",
        seed=index,
        mode=mode,
        round="roundT",
        bot_a=bot_a,
        bot_b=bot_b,
        bot_a_content_hash=hash_a or HASHES[bot_a],
        bot_b_content_hash=hash_b or HASHES[bot_b],
        engine_version=engine,
        winner=winner,  # type: ignore[arg-type]
        turns=1200 if truncated else 400,
        truncated=truncated,
    )


def synthetic_games(count: int = 200) -> list[GameRecord]:
    """A deterministic, reasonably connected set of records."""
    rng = random.Random(4)
    bots = list(HASHES)
    games: list[GameRecord] = []
    for i in range(count):
        a, b = rng.sample(bots, 2)
        winner = rng.choices(["a", "b", "draw"], weights=[0.4, 0.35, 0.25])[0]
        games.append(record(a, b, winner, index=i, truncated=winner == "draw"))
    return games


# --- per-round fixtures -----------------------------------------------------
#
# Deliberately tiny: two or three rounds of a handful of records each. The whole
# ratings suite runs in ~1.3 s and the budget in AGENTS.md has about a second of
# headroom, so nothing here reads `data/games/` or fits a realistic round.


def pair_games(a: str, b: str, *, wins_a: int, wins_b: int, start: int, **kwargs):
    """`wins_a` + `wins_b` records for one ordered pair."""
    return [
        record(a, b, "a" if i < wins_a else "b", index=start + i, **kwargs)
        for i in range(wins_a + wins_b)
    ]


def games_tree(root: Path, rounds: dict[str, list[GameRecord]]) -> Path:
    """Write `{round: records}` out as `data/games/`-shaped directories."""
    for name, games in rounds.items():
        for game in games:
            save_game(game, root / name)
    return root


def fit_tree(root: Path, cache: Path, *, anchor: str = ANCHOR, **policy_kwargs):
    return fit_rounds(
        games_dir=root,
        cache_dir=cache,
        policy=Policy(engine_version=ENGINE, **policy_kwargs),
        registry=FakeRegistry(),
        global_anchor=anchor,
        use_cache=False,
    )


TWO_ROUNDS = {
    "alpha": lambda: pair_games("smoke", "blitz", wins_a=4, wins_b=2, start=0),
    "beta": lambda: pair_games("aegis", "metro", wins_a=3, wins_b=3, start=100),
}


def two_round_tree(root: Path) -> Path:
    return games_tree(root, {name: make() for name, make in TWO_ROUNDS.items()})


# --- T1: order invariance ---------------------------------------------------


def test_count_table_digest_is_identical_under_any_ordering():
    games = synthetic_games()
    policy = Policy()
    registry = FakeRegistry()
    baseline = count_table(games, policy=policy, registry=registry)

    for seed in range(20):
        shuffled = list(games)
        random.Random(seed).shuffle(shuffled)
        table = count_table(shuffled, policy=policy, registry=registry)
        assert table.digest == baseline.digest, seed
        assert table.cells == baseline.cells
        assert table.entities == baseline.entities


def test_every_fitted_quantity_is_identical_under_any_ordering():
    games = synthetic_games()
    policy = Policy()
    registry = FakeRegistry()
    anchor = entity_key("smoke", HASHES["smoke"])
    baseline = fit_ratings(
        count_table(games, policy=policy, registry=registry), anchor=anchor
    )

    for seed in range(20):
        shuffled = list(games)
        random.Random(seed).shuffle(shuffled)
        other = fit_ratings(
            count_table(shuffled, policy=policy, registry=registry), anchor=anchor
        )
        for entity in baseline.entities:
            assert abs(baseline.rating(entity) - other.rating(entity)) < 1e-9
            assert abs(baseline.se(entity) - other.se(entity)) < 1e-9
        assert abs(baseline.seat_advantage.value - other.seat_advantage.value) < 1e-9
        assert abs(baseline.draw_log_nu.value - other.draw_log_nu.value) < 1e-9
        for i in range(len(baseline.entities)):
            for j in range(len(baseline.entities)):
                assert abs(baseline._covariance[i, j] - other._covariance[i, j]) < 1e-9


def test_no_incremental_rating_api_exists_to_diverge_from_the_rebuild():
    """
    The old incremental path and the rebuild silently disagreed. Nothing to
    disagree with now: every write refits.
    """
    import arena.records.ratings as ratings

    for gone in ("RatingBook", "rate_stored_game", "rebuild_from_games", "apply_game"):
        assert not hasattr(ratings, gone), gone


# --- T2: count-table canonicality ------------------------------------------


def test_cells_are_emitted_in_sorted_order_with_integer_counts():
    table = count_table(
        synthetic_games(), policy=Policy(), registry=FakeRegistry()
    )
    assert list(table.entities) == sorted(table.entities)
    keys = [(c.seat_a, c.seat_b) for c in table.cells]
    assert keys == sorted(keys)
    for cell in table.cells:
        assert all(isinstance(v, int) for v in (cell.wins_a, cell.wins_b, cell.draws))


def test_merging_per_round_tables_in_any_order_matches_one_shot():
    """Count tables are exactly additive, which is what makes the cache safe."""
    games = synthetic_games(240)
    policy, registry = Policy(), FakeRegistry()
    one_shot = count_table(games, policy=policy, registry=registry)

    chunks = [games[i::4] for i in range(4)]
    tables = [count_table(chunk, policy=policy, registry=registry) for chunk in chunks]
    for seed in range(5):
        order = list(tables)
        random.Random(seed).shuffle(order)
        assert merge(order).digest == one_shot.digest


def test_digest_changes_when_the_games_change():
    games = synthetic_games()
    policy, registry = Policy(), FakeRegistry()
    before = count_table(games, policy=policy, registry=registry)
    after = count_table(
        games + [record("smoke", "blitz", "a", index=999)],
        policy=policy,
        registry=registry,
    )
    assert before.digest != after.digest


# --- T9: pooling policy -----------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        ({"mode": "classic"}, MODE_NOT_ELIGIBLE),
        ({"engine": "0" * 40}, ENGINE_MISMATCH),
        ({"hash_a": "ffffffffffff"}, UNREGISTERED_HASH),
    ],
)
def test_ineligible_games_are_excluded_and_counted(kwargs, reason):
    games = [record("smoke", "blitz", "a", index=0, **kwargs)]
    table = count_table(
        games, policy=Policy(engine_version=ENGINE), registry=FakeRegistry()
    )
    assert table.cells == ()
    assert dict(table.excluded) == {reason: 1}


def test_an_unknown_hash_is_rejected_before_it_reaches_the_fit():
    """`GameRecord` refuses to build one, so it cannot reach the count table."""
    with pytest.raises(ValueError, match="unknown"):
        record("smoke", "blitz", "a", hash_a="unknown")


def test_truncated_draws_are_included():
    """A 1200-turn truncation *is* a draw; excluding them would delete 35%."""
    games = [record("smoke", "blitz", "draw", index=i, truncated=True) for i in range(10)]
    table = count_table(games, policy=Policy(), registry=FakeRegistry())
    assert table.excluded == {}
    assert sum(c.draws for c in table.cells) == 10


def test_self_play_is_included_and_moves_only_the_nuisance_parameters():
    """Strength cancels in a self-play cell — it is a clean beta/kappa estimator."""
    base = synthetic_games()
    self_games = [
        record("smoke", "smoke", ["a", "b", "draw"][i % 3], index=1000 + i)
        for i in range(60)
    ]
    policy, registry = Policy(), FakeRegistry()
    anchor = entity_key("blitz", HASHES["blitz"])

    without = fit_ratings(count_table(base, policy=policy, registry=registry), anchor=anchor)
    with_self = fit_ratings(
        count_table(base + self_games, policy=policy, registry=registry), anchor=anchor
    )

    assert any(c.seat_a == c.seat_b for c in with_self.counts.cells)
    assert with_self.draw_log_nu.se < without.draw_log_nu.se
    for entity in without.entities:
        assert abs(without.rating(entity) - with_self.rating(entity)) < 1.0


def test_self_play_can_be_excluded_and_is_counted_when_it_is():
    games = [record("smoke", "smoke", "a", index=i) for i in range(5)]
    table = count_table(
        games, policy=Policy(include_self_play=False), registry=FakeRegistry()
    )
    assert dict(table.excluded) == {SELF_PLAY_EXCLUDED: 5}


def test_no_rating_code_path_reaches_classic_or_remote_games():
    """AGENTS.md's exclusion, enforced by code rather than by convention."""
    import arena.records.ratings.cli as cli
    import arena.records.ratings.counts as counts_module
    import arena.records.ratings.io as io_module
    import arena.records.ratings.policy as policy_module

    for module in (cli, counts_module, io_module, policy_module):
        source = open(module.__file__, encoding="utf-8").read()
        assert "classic_games" not in source, module.__name__
        assert "remote_games" not in source, module.__name__

    # And the mode filter would reject them even if a path did.
    assert Policy().modes == ("competition",)


# --- T13: reproducible artifacts -------------------------------------------


def test_two_fits_over_the_same_games_write_a_byte_identical_fit_json(tmp_path):
    """
    The A17 regression guard: no timestamp in any payload.

    Now per round, which is what the split bought — after a new round, exactly one
    fit file changes and a diff of the others is meaningful.
    """
    root = two_round_tree(tmp_path / "games")
    first, second = tmp_path / "one", tmp_path / "two"
    for directory in (first, second):
        io.write_all(fit_tree(root, tmp_path / f"cache-{directory.name}"), directory)

    for name in ("alpha", "beta"):
        assert (
            io.round_fit_path(name, first).read_bytes()
            == io.round_fit_path(name, second).read_bytes()
        ), name
    assert (first / io.LEADERBOARD_JSON).read_bytes() == (
        second / io.LEADERBOARD_JSON
    ).read_bytes()

    payload = json.loads(io.round_fit_path("alpha", first).read_text(encoding="utf-8"))
    assert "updated_at" not in payload
    assert "generated_at" not in payload


def test_each_round_is_fitted_only_from_its_own_games(tmp_path):
    """
    The core invariant. A game played in one round cannot move another's numbers,
    because no fit ever sees more than one round's count table.
    """
    root = two_round_tree(tmp_path / "games")
    before = tmp_path / "before"
    io.write_all(fit_tree(root, tmp_path / "c1"), before)
    alpha_before = io.round_fit_path("alpha", before).read_bytes()

    save_game(record("aegis", "metro", "a", index=900), root / "beta")
    after = tmp_path / "after"
    io.write_all(fit_tree(root, tmp_path / "c2"), after)

    assert io.round_fit_path("alpha", after).read_bytes() == alpha_before
    assert io.round_fit_path("beta", after).read_bytes() != io.round_fit_path(
        "beta", before
    ).read_bytes()


def test_root_games_influence_no_published_number(tmp_path):
    """
    Loose `data/games/*.json` are not a round and enter nothing.

    They carry a `round` field no directory backs — 49 distinct values over 102
    files — so a fit over them would be a fit over an invented design.
    """
    root = two_round_tree(tmp_path / "games")
    clean = fit_tree(root, tmp_path / "c1")

    ghost = record("smoke", "blitz", "a", index=700)
    save_game(ghost, root)
    with_root = fit_tree(root, tmp_path / "c2")

    assert with_root.names == clean.names == ("alpha", "beta")
    assert with_root.rounds_digest == clean.rounds_digest
    assert with_root.rated_games == clean.rated_games
    markdown = io.leaderboard_markdown(with_root, updated_at="2026-01-01T00:00:00Z")
    assert "_root" not in markdown


def test_round_order_is_name_ascending_and_stable_under_shuffled_discovery(tmp_path):
    root = games_tree(
        tmp_path / "games",
        {
            "zulu": pair_games("smoke", "blitz", wins_a=2, wins_b=1, start=0),
            "alpha": pair_games("aegis", "metro", wins_a=2, wins_b=1, start=50),
            "mike": pair_games("smoke", "metro", wins_a=2, wins_b=1, start=90),
        },
    )
    fits = fit_tree(root, tmp_path / "cache")
    assert fits.names == ("alpha", "mike", "zulu")

    # And the ordering is enforced by the container, not by discovery order.
    shuffled = list(fits.results)
    random.Random(0).shuffle(shuffled)
    assert RoundFits(shuffled, policy=fits.policy, prior=fits.prior).names == fits.names


# --- the anchor is resolved per round ---------------------------------------


def test_a_round_without_the_global_anchor_uses_its_most_played_entity(tmp_path):
    root = games_tree(
        tmp_path / "games",
        {
            "with_anchor": pair_games("smoke", "blitz", wins_a=3, wins_b=1, start=0),
            # No `smoke`, and `metro` plays twice as many games as anyone else.
            "no_anchor": (
                pair_games("aegis", "metro", wins_a=2, wins_b=1, start=50)
                + pair_games("blitz", "metro", wins_a=1, wins_b=2, start=60)
            ),
        },
    )
    fits = fit_tree(root, tmp_path / "cache")

    with_anchor = fits.result("with_anchor")
    assert (with_anchor.anchor, with_anchor.anchor_kind) == (ANCHOR, ANCHOR_GLOBAL)

    local = fits.result("no_anchor")
    assert local.anchor == entity_key("metro", HASHES["metro"])
    assert local.anchor_kind == ANCHOR_ROUND_LOCAL
    assert local.rated
    assert local.fit.rating(local.anchor) == pytest.approx(1500.0)


def test_a_tie_on_appearances_breaks_by_entity_key_ascending():
    """Deterministic, and the same rule the round report snippet used."""
    from arena.records.ratings.counts import build

    table = build({("b@2", "a@1"): (1, 1, 0), ("c@3", "d@4"): (1, 1, 0)})
    assert resolve_round_anchor(table, "ghost@0") == ("a@1", ANCHOR_ROUND_LOCAL)


def test_the_global_anchor_is_never_forced_into_a_round_that_did_not_play_it(tmp_path):
    """
    Forcing it in costs 68 Elo of contrast and inflates the SE.

    A zero-game anchor is its own connectivity component, so the round's real
    entities land in a second one whose location comes from the prior — and the
    prior then does the work an anchor should be doing.
    """
    root = games_tree(
        tmp_path / "games",
        {"no_anchor": pair_games("aegis", "metro", wins_a=5, wins_b=1, start=0)},
    )
    result = fit_tree(root, tmp_path / "cache").result("no_anchor")

    assert ANCHOR not in result.entities
    assert len(result.components) == 1
    assert result.connected
    assert result.fit.delta(
        entity_key("aegis", HASHES["aegis"]), entity_key("metro", HASHES["metro"])
    ).comparable


# --- unrated rounds ---------------------------------------------------------


def test_a_round_with_no_eligible_games_is_reported_unrated_with_its_exclusions(tmp_path):
    root = games_tree(
        tmp_path / "games",
        {
            "good": pair_games("smoke", "blitz", wins_a=2, wins_b=1, start=0),
            "unregistered": pair_games(
                "aegis", "metro", wins_a=2, wins_b=1, start=50, hash_a="ffffffffffff"
            ),
        },
    )
    fits = fit_tree(root, tmp_path / "cache")

    bad = fits.result("unregistered")
    assert not bad.rated
    assert bad.reason == NO_ELIGIBLE_GAMES
    assert bad.stored_games == 3 and bad.rated_games == 0
    assert dict(bad.excluded) == {UNREGISTERED_HASH: 3}

    # Never silently dropped: the round is in the document with its counts.
    markdown = io.leaderboard_markdown(fits, updated_at="2026-01-01T00:00:00Z")
    assert "## unregistered — unrated" in markdown
    assert "`unregistered_hash` 3" in markdown
    # And no fit file, because there is nothing to describe.
    written, _ = io.write_round_fits(fits, tmp_path / "ratings")
    assert [p.stem for p in written] == ["good"]
    with pytest.raises(KeyError, match="unrated"):
        fits["unregistered"]


def test_a_round_where_nobody_reaches_the_gate_ranks_nobody_and_still_renders(tmp_path):
    """A 6-game probe *is* a probe and should read like one."""
    root = two_round_tree(tmp_path / "games")
    fits = fit_tree(root, tmp_path / "cache")
    alpha = fits.result("alpha")

    assert alpha.rated and alpha.ranked_count == 0
    markdown = io.leaderboard_markdown(fits, updated_at="2026-01-01T00:00:00Z")
    assert "ranks nobody" in markdown
    assert "### Provisional in this round (< 30 games)" in markdown
    # The index says 0 ranked without hiding the round.
    assert "| [alpha](#alpha) | rated | 6 / 6 | 2 | **0** | 1 |" in markdown


def test_a_round_split_by_the_era_filter_rates_one_era_and_reports_era_split(tmp_path):
    root = games_tree(
        tmp_path / "games",
        {
            "spanning": (
                pair_games("smoke", "blitz", wins_a=2, wins_b=1, start=0)
                + pair_games(
                    "smoke", "blitz", wins_a=1, wins_b=1, start=50, engine=OTHER_ENGINE
                )
            )
        },
    )
    result = fit_tree(root, tmp_path / "cache").result("spanning")

    assert result.rated
    assert result.era_split
    assert result.engine_versions == (OTHER_ENGINE, ENGINE)
    assert result.stored_games == 5 and result.rated_games == 3
    assert dict(result.excluded) == {ENGINE_MISMATCH: 2}


# --- the anti-comparison invariant ------------------------------------------


def test_two_rounds_with_identical_games_get_different_scale_ids(tmp_path):
    same = lambda start: pair_games("smoke", "blitz", wins_a=4, wins_b=2, start=start)
    root = games_tree(tmp_path / "games", {"first": same(0), "second": same(0)})
    fits = fit_tree(root, tmp_path / "cache")

    first, second = fits.result("first"), fits.result("second")
    assert first.counts_digest == second.counts_digest  # byte-identical games
    assert first.scale_id != second.scale_id  # and still not one scale
    assert first.anchor == second.anchor  # a shared anchor does not join them


def test_the_markdown_has_no_pooled_ranked_table(tmp_path):
    """Guards the removal: a single ranked list asserts what is not true."""
    root = two_round_tree(tmp_path / "games")
    markdown = io.leaderboard_markdown(
        fit_tree(root, tmp_path / "cache"), updated_at="2026-01-01T00:00:00Z"
    )

    assert "Do not\n> compare them" in markdown
    assert "## Rounds" in markdown
    preamble, _, rest = markdown.partition("## Rounds")
    # Nothing ranked above the index, and the index itself carries provenance
    # and counts, never a rating.
    assert "| Rank |" not in preamble
    index = rest.split("---", 1)[0]
    assert "Rating" not in index and "95% CI" not in index
    # Every ranked or provisional table lives under a round heading.
    for chunk in markdown.split("\n## ")[1:]:
        assert "| Rank |" not in chunk or chunk.split("\n", 1)[0] in {
            "alpha",
            "beta",
        }


def test_a_leftover_pooled_fit_json_is_deleted(tmp_path):
    """A file named `fit.json` holding one table over every round is the claim
    this refactor withdraws. Leaving it published would keep making it."""
    root = two_round_tree(tmp_path / "games")
    ratings = tmp_path / "ratings"
    ratings.mkdir()
    legacy = ratings / io.LEGACY_FIT_JSON
    legacy.write_text("{}", encoding="utf-8")

    _, pruned = io.write_round_fits(fit_tree(root, tmp_path / "cache"), ratings)
    assert legacy in pruned
    assert not legacy.exists()


def test_leaderboard_json_has_no_top_level_entities_key(tmp_path):
    """A partial revert to the v1 flat snapshot must fail loudly."""
    root = two_round_tree(tmp_path / "games")
    io.write_leaderboard(fit_tree(root, tmp_path / "cache"), tmp_path / "ratings")
    payload = json.loads(
        (tmp_path / "ratings" / io.LEADERBOARD_JSON).read_text(encoding="utf-8")
    )

    assert payload["version"] == 2
    assert payload["format"] == "per_round"
    for gone in (
        "entities",
        "anchor",
        "rated_games",
        "counts_digest",
        "connected",
        "components",
        "seat_advantage",
    ):
        assert gone not in payload, gone
    assert [r["round"] for r in payload["rounds"]] == ["alpha", "beta"]
    assert payload["rounds"][0]["scale_id"].startswith("scale:")


def test_no_per_game_refit_path_exists():
    """
    A removed path that nothing asserts is removed grows back.

    Per-round fitting made the per-game refit strictly worse: one ad-hoc game
    would solve every round in the store to publish a game that lands in no
    round's table.
    """
    import inspect

    from arena.matches import run_match

    assert "update_ratings" not in inspect.signature(run_match.run_and_store).parameters
    with pytest.raises(SystemExit):
        run_match.main(["a/run.sh", "b/run.sh", "--update-ratings"])


# --- reading one round back -------------------------------------------------


def test_fit_round_trips_through_its_own_payload(tmp_path):
    root = two_round_tree(tmp_path / "games")
    fits = fit_tree(root, tmp_path / "cache")
    io.write_round_fits(fits, tmp_path / "ratings")

    fit = fits["alpha"]
    loaded = io.load_fit(tmp_path / "ratings", round="alpha")
    assert loaded.round == "alpha"
    assert loaded.anchor_kind == ANCHOR_GLOBAL
    assert loaded.scale_id == fits.result("alpha").scale_id
    assert loaded.anchor == fit.anchor
    assert loaded.counts_digest == fit.counts.digest
    for entity in fit.entities:
        assert loaded.rating(entity) == pytest.approx(fit.rating(entity), abs=0.01)
    original = fit.delta(fit.entities[0], fit.entities[1])
    restored = loaded.delta(fit.entities[0], fit.entities[1])
    assert restored.value == pytest.approx(original.value, abs=0.01)
    assert restored.se == pytest.approx(original.se, abs=0.01)


def test_load_fit_without_a_round_refuses_and_names_the_rounds(tmp_path):
    """There is no pooled fit, so "the first one" is never the right answer."""
    root = two_round_tree(tmp_path / "games")
    io.write_round_fits(fit_tree(root, tmp_path / "cache"), tmp_path / "ratings")

    with pytest.raises(ValueError, match="alpha, beta"):
        io.load_fit(tmp_path / "ratings")
    assert io.load_fit(tmp_path / "ratings", round="never-ran") is None


def test_stored_digest_answers_whether_a_refit_is_needed(tmp_path):
    root = two_round_tree(tmp_path / "games")
    ratings = tmp_path / "ratings"
    fits = fit_tree(root, tmp_path / "c1")
    io.write_all(fits, ratings)

    assert io.stored_counts_digest(ratings, round="alpha") == fits.result(
        "alpha"
    ).counts_digest
    assert io.stored_rounds_digest(ratings) == fits.rounds_digest
    with pytest.raises(ValueError, match="needs a round"):
        io.stored_counts_digest(ratings)

    save_game(record("smoke", "blitz", "a", index=999), root / "alpha")
    grown = fit_tree(root, tmp_path / "c2")
    assert io.stored_counts_digest(ratings, round="alpha") != grown.result(
        "alpha"
    ).counts_digest
    assert io.stored_rounds_digest(ratings) != grown.rounds_digest
    # One round changed, so the other round's digest still matches.
    assert io.stored_counts_digest(ratings, round="beta") == grown.result(
        "beta"
    ).counts_digest


# --- leaderboard rendering --------------------------------------------------


def test_leaderboard_lists_provisional_entities_below_the_ranked_block(tmp_path):
    """Per round: one entity can be ranked in one round and provisional in another."""
    root = games_tree(
        tmp_path / "games",
        {
            "big": (
                pair_games("smoke", "blitz", wins_a=12, wins_b=8, start=0)
                + pair_games("blitz", "smoke", wins_a=8, wins_b=12, start=100)
                # `metro` plays 3 games here and stays under the gate.
                + pair_games("smoke", "metro", wins_a=2, wins_b=1, start=200)
            )
        },
    )
    fits = fit_tree(root, tmp_path / "cache")
    rows = io.leaderboard_rows(fits["big"])

    ranks = [r.provisional for r in rows]
    assert ranks == sorted(ranks)  # every ranked row precedes every provisional one
    metro = next(r for r in rows if r.bot_id == "metro")
    assert metro.provisional is True and metro.rank == 0

    markdown = io.leaderboard_markdown(fits, updated_at="2026-01-01T00:00:00Z")
    assert "95% CI" in markdown
    assert "### Provisional in this round (< 30 games)" in markdown
    assert f"Anchor `{ANCHOR}` (global) pinned at 1500.0" in markdown


# --- the anchor is what pins the scale ------------------------------------


def test_a_missing_anchor_is_a_clear_error_not_a_crash(tmp_path):
    """No anchor means no scale — theta would be free up to a constant."""
    from arena.records.ratings.cli import MissingAnchor, resolve_anchor
    from arena.records.registry import Registry

    with pytest.raises(MissingAnchor, match="not registered"):
        resolve_anchor(Registry(tmp_path), "cm_expander")


def test_fit_rejects_an_anchor_that_is_not_in_the_table():
    table = count_table(synthetic_games(), policy=Policy(), registry=FakeRegistry())
    with pytest.raises(ValueError, match="not in the count table"):
        fit_ratings(table, anchor="ghost@000000000000")
