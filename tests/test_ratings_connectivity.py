"""
The connectivity guard: no contrast without a chain of games behind it.

The likelihood only ever sees differences `theta_i - theta_j`, and only for
pairs that played. It is therefore flat along "add a constant to everybody",
once per connected component. With one component the anchor pins that
constant; with two, the second is pinned by the prior alone — and because the
prior also makes the Hessian invertible, the fit converges, the covariance
inverts, and a cross-group contrast comes back with a small finite SE that
describes prior curvature rather than evidence.

That is not hypothetical. Deleting a stderr write that ran after the last move
forked every bot's content hash; the next round played only the new hashes, so
the two generations shared no games, and the lineage table reported +474 Elo
with `P(better) = 1.00` for an edit that cannot change a single action.
"""
from __future__ import annotations

import math

import pytest

from arena.records.ratings import io
from arena.records.ratings.counts import (
    build,
    component_index,
    connected_components,
)
from arena.records.ratings.fit import fit_ratings
from arena.records.ratings.policy import Policy


def table(pairs, *, entities=()):
    """`{(a, b): (wins_a, wins_b, draws)}` -> CountTable."""
    return build(pairs, entities=entities)


CONNECTED = {("a", "b"): (6, 4, 0), ("b", "c"): (5, 5, 0)}
SPLIT = {("a", "b"): (6, 4, 0), ("c", "d"): (5, 5, 0)}


# --- the component computation ----------------------------------------------


def test_a_pool_everyone_is_linked_through_is_one_component():
    """a-b and b-c, so a reaches c through b: one group, not two."""
    assert connected_components(table(CONNECTED)) == (("a", "b", "c"),)


def test_two_groups_that_never_met_are_two_components():
    assert connected_components(table(SPLIT)) == (("a", "b"), ("c", "d"))


def test_an_entity_with_no_games_is_its_own_component():
    """The anchor is forced into an empty pool; nobody played it."""
    components = connected_components(table(CONNECTED, entities=["lonely"]))
    assert components == (("a", "b", "c"), ("lonely",))


def test_components_are_ordered_largest_first_then_by_name():
    got = connected_components(table({("y", "z"): (1, 0, 0), ("a", "b"): (1, 0, 0),
                                      ("b", "c"): (1, 0, 0)}))
    assert got == (("a", "b", "c"), ("y", "z"))


def test_component_order_does_not_depend_on_input_order():
    """Same canonicality guarantee the count table itself carries."""
    forward = connected_components(table(SPLIT))
    backward = connected_components(table(dict(reversed(list(SPLIT.items())))))
    assert forward == backward


def test_component_index_numbers_every_entity():
    assert component_index(table(SPLIT)) == {"a": 0, "b": 0, "c": 1, "d": 1}


def test_a_draw_still_links_two_entities():
    """Any game is a link; the guard is about identifiability, not decisiveness."""
    assert connected_components(table({("a", "b"): (0, 0, 1)})) == (("a", "b"),)


# --- what the fit does with it ----------------------------------------------


def fit_for(pairs, anchor="a", entities=()):
    return fit_ratings(table(pairs, entities=entities), anchor=anchor, policy=Policy())


def test_a_connected_pool_is_unaffected():
    fit = fit_for(CONNECTED)
    assert fit.connected
    assert len(fit.components) == 1
    delta = fit.delta("a", "c")
    assert delta.comparable
    assert math.isfinite(delta.se)


def test_a_split_pool_is_reported_as_split():
    fit = fit_for(SPLIT)
    assert not fit.connected
    assert [len(g) for g in fit.components] == [2, 2]
    assert fit.comparable("a", "b")
    assert not fit.comparable("a", "c")


def test_a_cross_group_contrast_carries_no_evidence():
    """
    The whole point: an infinite interval and P = 0.50, not a confident number.

    Without the guard this contrast returns a small finite SE, because the
    prior's curvature is indistinguishable from data's once the Hessian is
    inverted.
    """
    delta = fit_for(SPLIT).delta("a", "c")
    assert not delta.comparable
    assert delta.se == math.inf
    assert delta.p_stronger == 0.5
    low, high = delta.ci
    assert low == -math.inf and high == math.inf


def test_within_group_contrasts_still_work_in_a_split_pool():
    """The guard removes the false claims, not the true ones."""
    delta = fit_for(SPLIT).delta("a", "b")
    assert delta.comparable
    assert math.isfinite(delta.se) and delta.se > 0


def test_the_decision_rule_lands_on_unproven_across_groups():
    """`improvement` needs P >= 0.95 and CI.low > +10. Neither can hold now."""
    delta = fit_for(SPLIT).delta("a", "c")
    assert not (delta.p_stronger >= 0.95 and delta.ci[0] > 10)
    assert not (delta.p_stronger <= 0.05 and delta.ci[1] < -10)


def test_an_isolated_anchor_is_flagged_rather_than_silently_compared():
    fit = fit_for(CONNECTED, entities=["lonely"])
    assert not fit.connected
    assert not fit.comparable("a", "lonely")


def test_component_of_rejects_an_unknown_entity():
    with pytest.raises(KeyError):
        fit_for(CONNECTED).component_of("never-played")


# --- what a reader sees ------------------------------------------------------


def test_the_leaderboard_warns_and_groups_when_the_pool_is_split():
    markdown = io.leaderboard_markdown(fit_for(SPLIT), updated_at="2026-01-01T00:00:00Z")
    assert "not connected: 2 groups" in markdown
    assert "only within one group" in markdown
    # A rank must not read as a comparison between entities that never met.
    assert "| Rank | Group | Entity |" in markdown


def test_a_connected_leaderboard_is_unchanged():
    markdown = io.leaderboard_markdown(fit_for(CONNECTED), updated_at="2026-01-01T00:00:00Z")
    assert "not connected" not in markdown
    assert "| Rank | Entity |" in markdown
    assert "Group" not in markdown


def test_the_stored_payload_records_the_grouping(tmp_path):
    import json

    io.write_leaderboard(fit_for(SPLIT), tmp_path)
    payload = json.loads((tmp_path / "leaderboard.json").read_text(encoding="utf-8"))

    assert payload["connected"] is False
    assert payload["components"] == [["a", "b"], ["c", "d"]]
    assert {row["entity"]: row["component"] for row in payload["entities"]} == {
        "a": 0, "b": 0, "c": 1, "d": 1
    }


def test_provisional_rows_carry_the_group_too():
    """
    Ranked and provisional tables are rendered separately.

    Either one on its own can look connected while the pool is not, so the
    split has to be decided from the fit and passed down — not inferred from
    whichever rows a given table happens to hold.
    """
    fit = fit_for({("a", "b"): (20, 20, 0), ("c", "d"): (5, 5, 0)})
    markdown = io.leaderboard_markdown(fit, updated_at="2026-01-01T00:00:00Z")

    assert "## Provisional" in markdown  # c and d are under the games floor
    assert markdown.count("| Rank | Group | Entity |") == 2


def test_games_to_resolve_answers_the_question_the_guard_raises():
    """
    "Not comparable" has a remediation, and the existing machinery states it.

    An infinite SE means zero precision in hand, so the count comes out as the
    games needed from scratch — no special case required.
    """
    fit = fit_for(SPLIT)
    needed = fit.games_to_resolve("a", "c", target_se=12.75)
    assert needed > 0


# --- the stored fit ----------------------------------------------------------


def roundtrip(fit):
    """`fit.json` and back, through actual JSON so types survive nothing."""
    import json

    return io.fit_from_payload(json.loads(json.dumps(io.fit_payload(fit))))


def test_fit_json_records_the_grouping():
    payload = io.fit_payload(fit_for(SPLIT))
    assert payload["connected"] is False
    assert payload["components"] == [["a", "b"], ["c", "d"]]


def test_a_reloaded_split_fit_refuses_the_same_contrast():
    """
    The guard must survive `fit.json`.

    Without this, the stored covariance hands back a small finite SE for the
    cross-group pair — the exact confident number the guard exists to refuse,
    resurfacing one file-read later.
    """
    loaded = roundtrip(fit_for(SPLIT))
    assert not loaded.connected
    assert loaded.components == (("a", "b"), ("c", "d"))
    delta = loaded.delta("a", "c")
    assert not delta.comparable
    assert delta.se == math.inf
    assert delta.p_stronger == 0.5


def test_a_reloaded_connected_fit_still_answers():
    loaded = roundtrip(fit_for(CONNECTED))
    assert loaded.connected
    delta = loaded.delta("a", "c")
    assert delta.comparable
    assert math.isfinite(delta.se) and delta.se > 0


def test_a_reloaded_fit_keeps_within_group_contrasts():
    delta = roundtrip(fit_for(SPLIT)).delta("a", "b")
    assert delta.comparable
    assert math.isfinite(delta.se)


def test_a_legacy_payload_without_the_grouping_refuses_rather_than_guesses():
    """
    A file written before the guard cannot say which entities share games,
    and the games are not in the payload to recompute it. Assuming
    "connected" is exactly the bug; the honest answer is a refusal, and a
    refit rewrites the file with the grouping in it.
    """
    payload = io.fit_payload(fit_for(SPLIT))
    del payload["components"]
    del payload["connected"]
    loaded = io.fit_from_payload(payload)

    assert loaded.components is None
    assert not loaded.connected
    delta = loaded.delta("a", "b")  # linked by games, but the file lost that
    assert not delta.comparable
    assert delta.se == math.inf


def test_a_reloaded_fit_rejects_an_unknown_entity():
    with pytest.raises(KeyError):
        roundtrip(fit_for(SPLIT)).comparable("a", "never-played")
