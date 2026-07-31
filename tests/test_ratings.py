"""
Order invariance, count-table canonicality, pooling policy, and artifacts.

The property this suite exists to protect is R1: the fit is a pure function of
the *set* of stored games. The old suite asserted idempotence by `game_id`, a
much weaker property that sequential Elo passed while being badly broken.
"""
from __future__ import annotations

import json
import random
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
from arena.records.store import GameRecord

ENGINE = "9e3b9d13cca51caa1bb07db48bb85c9e90ce0462"

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
        terminated=not truncated,
        truncated=truncated,
        started_at="2026-01-01T00:00:00Z",
        finished_at="2026-01-01T00:01:00Z",
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
    """The A17 regression guard: no timestamp in the payload."""
    games = synthetic_games()
    policy, registry = Policy(), FakeRegistry()
    anchor = entity_key("smoke", HASHES["smoke"])

    first = tmp_path / "one"
    second = tmp_path / "two"
    for directory, seed in ((first, 0), (second, 1)):
        shuffled = list(games)
        random.Random(seed).shuffle(shuffled)
        fit = fit_ratings(
            count_table(shuffled, policy=policy, registry=registry), anchor=anchor
        )
        io.write_fit(fit, directory)

    assert (first / io.FIT_JSON).read_bytes() == (second / io.FIT_JSON).read_bytes()
    payload = json.loads((first / io.FIT_JSON).read_text(encoding="utf-8"))
    assert "updated_at" not in payload
    assert "generated_at" not in payload


def test_fit_round_trips_through_its_own_payload(tmp_path):
    games = synthetic_games()
    anchor = entity_key("smoke", HASHES["smoke"])
    fit = fit_ratings(
        count_table(games, policy=Policy(), registry=FakeRegistry()), anchor=anchor
    )
    io.write_fit(fit, tmp_path)
    loaded = io.load_fit(tmp_path)

    assert loaded.anchor == fit.anchor
    assert loaded.counts_digest == fit.counts.digest
    for entity in fit.entities:
        assert loaded.rating(entity) == pytest.approx(fit.rating(entity), abs=0.01)
    original = fit.delta(fit.entities[0], fit.entities[1])
    restored = loaded.delta(fit.entities[0], fit.entities[1])
    assert restored.value == pytest.approx(original.value, abs=0.01)
    assert restored.se == pytest.approx(original.se, abs=0.01)


def test_stored_digest_answers_whether_a_refit_is_needed(tmp_path):
    games = synthetic_games()
    anchor = entity_key("smoke", HASHES["smoke"])
    table = count_table(games, policy=Policy(), registry=FakeRegistry())
    io.write_fit(fit_ratings(table, anchor=anchor), tmp_path)

    assert io.stored_counts_digest(tmp_path) == table.digest
    grown = count_table(
        games + [record("smoke", "blitz", "a", index=999)],
        policy=Policy(),
        registry=FakeRegistry(),
    )
    assert io.stored_counts_digest(tmp_path) != grown.digest


# --- leaderboard rendering --------------------------------------------------


def test_leaderboard_lists_provisional_entities_below_the_ranked_block():
    games = synthetic_games()
    extra = [record("smoke", "metro", "a", index=2000 + i) for i in range(3)]
    table = count_table(
        [g for g in games if "metro" not in (g.bot_a, g.bot_b)] + extra,
        policy=Policy(),
        registry=FakeRegistry(),
    )
    fit = fit_ratings(table, anchor=entity_key("smoke", HASHES["smoke"]))
    rows = io.leaderboard_rows(fit)

    ranks = [r.provisional for r in rows]
    assert ranks == sorted(ranks)  # every ranked row precedes every provisional one
    metro = next(r for r in rows if r.bot_id == "metro")
    assert metro.provisional is True
    assert metro.rank == 0

    markdown = io.leaderboard_markdown(fit, updated_at="2026-01-01T00:00:00Z")
    assert "95% CI" in markdown
    assert "## Provisional" in markdown
    assert f"`{fit.anchor}` pinned at 1500.0" in markdown


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
