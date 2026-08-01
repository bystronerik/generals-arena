"""
Loading, metrics, fog, path and event detection over synthetic replays.

Every expectation here is counted by hand off the fixtures in
`tests/fixtures/replay_game.py` — nothing reads `competition-replays/`.
"""
from __future__ import annotations

import pytest
from fixtures.replay_game import (
    NEUTRAL,
    GameBuilder,
    attack_game,
    blind_game,
    forfeit_game,
    meta_row,
    write_game,
)

from arena.instrument.replay.analysis import ForfeitReplay, analyze
from arena.instrument.replay.events import (
    detect_events,
    find_loss_streaks,
    find_stalls,
)
from arena.instrument.replay.fog import compute_vision, sees_cell
from arena.instrument.replay.loader import (
    ReplayNotFound,
    find_replay,
    initial_structures,
    iter_replay_paths,
    load_replay,
    open_replay,
)
from arena.instrument.replay.metrics import PlayerTick, TickMetrics, compute_metrics
from arena.instrument.replay.path import (
    TARGET_GENERAL,
    TARGET_NONE,
    TARGET_VISIBLE_TILE,
    compute_path,
    summarize_path,
)
from arena.instrument.replay.phases import collapse_start


def board(rows: int = 6, cols: int = 6) -> GameBuilder:
    """The default two-corner board: seat 0 at (5,0), seat 1 at (0,5)."""
    return GameBuilder(rows, cols, generals=[(rows - 1, 0), (0, cols - 1)])


def analyzed(tmp_path, raw, player: str = "us_bot", outcome: str = "win", match_id: str = "1"):
    write_game(tmp_path, player, outcome, match_id, raw)
    return analyze(open_replay(player, match_id, tmp_path))


# --------------------------------------------------------------------------- loader


def test_open_replay_round_trips_the_scraper_layout(tmp_path):
    write_game(tmp_path, "us_bot", "lose", "77", attack_game(), meta_row("77", winner="B"))
    replay = open_replay("us_bot", "77", tmp_path)

    # Filed under lose/ — that is side A's result. us_bot holds seat 0 and seat
    # 0 wins the replay, so its own outcome is the opposite of the folder.
    assert (replay.folder, replay.outcome, replay.folder_disagrees) == ("lose", "win", True)
    assert replay.path == tmp_path / "us_bot" / "lose" / "77.json"
    assert (replay.rows, replay.cols) == (6, 6)
    assert replay.players == ("us_bot", "them_bot")
    assert replay.generals == ((5, 0), (0, 5))
    assert replay.seed == 7
    assert replay.total_ticks == 18
    assert len(replay.ticks) == 19
    assert replay.winner == 0
    assert replay.meta is not None and replay.meta.winner == "B"
    assert replay.enemy_general(0) == (0, 5)
    assert not replay.is_forfeit


def test_missing_replay_and_missing_player_both_raise(tmp_path):
    write_game(tmp_path, "us_bot", "win", "77", attack_game())

    with pytest.raises(ReplayNotFound, match="no replay '78'"):
        find_replay("us_bot", "78", tmp_path)
    with pytest.raises(ReplayNotFound, match="no replays for 'nobody'"):
        find_replay("nobody", "77", tmp_path)


def test_forfeit_is_flagged_and_refused_by_analyze(tmp_path):
    write_game(tmp_path, "us_bot", "draw", "79", forfeit_game())
    replay = open_replay("us_bot", "79", tmp_path)

    assert replay.total_ticks == 1
    assert replay.is_forfeit
    with pytest.raises(ForfeitReplay, match="a forfeit"):
        analyze(replay)
    assert analyze(replay, allow_forfeit=True).us == 0


def test_seat_resolves_by_name(tmp_path):
    write_game(tmp_path, "us_bot", "lose", "80", attack_game(players=("them_bot", "us_bot")))
    replay = open_replay("us_bot", "80", tmp_path)

    assert not replay.is_self_match
    assert replay.seat_of("us_bot") == 1
    assert replay.seat_of("them_bot") == 0


def test_self_match_falls_back_to_meta_a_side(tmp_path):
    raw = attack_game(players=("us_bot", "us_bot"))
    write_game(tmp_path, "us_bot", "win", "81", raw, meta_row("81", b_name="us_bot", a_side=1))
    write_game(tmp_path, "us_bot", "win", "82", raw)

    with_meta = open_replay("us_bot", "81", tmp_path)
    without_meta = open_replay("us_bot", "82", tmp_path)

    assert with_meta.is_self_match and without_meta.is_self_match
    assert with_meta.seat_of("us_bot") == 1
    assert without_meta.meta is None
    assert without_meta.seat_of("us_bot") == 0


def test_iter_replay_paths_orders_ids_numerically_and_skips_sidecars(tmp_path):
    for outcome, match_id in (("win", "77"), ("win", "9"), ("lose", "10")):
        write_game(tmp_path, "us_bot", outcome, match_id, blind_game(), meta_row(match_id))

    found = list(iter_replay_paths("us_bot", root=tmp_path))
    assert [(outcome, path.stem) for outcome, path in found] == [
        ("win", "9"),
        ("win", "77"),
        ("lose", "10"),
    ]
    assert list(iter_replay_paths("us_bot", "lose", root=tmp_path)) == [found[2]]


def test_initial_structures_reports_neutral_army_at_tick_zero(tmp_path):
    plain = board()
    plain.commit(3)
    stocked = board()
    stocked.set((2, 2), NEUTRAL, 40).commit(3)

    write_game(tmp_path, "us_bot", "win", "1", plain.raw())
    write_game(tmp_path, "us_bot", "win", "2", stocked.raw())

    assert initial_structures(open_replay("us_bot", "1", tmp_path)) == []
    assert initial_structures(open_replay("us_bot", "2", tmp_path)) == [(2, 2)]


# --------------------------------------------------------------------------- metrics


def test_metrics_track_army_tiles_and_ownership_transfers(tmp_path):
    g = GameBuilder(4, 4, generals=[(3, 0), (0, 3)])
    g.commit()  # t0
    g.set((3, 0), 0, 5).commit()  # t1
    g.march((3, 0), (3, 1)).commit()  # t2: 4 army walks onto neutral land
    g.set((3, 1), 1, 2).commit()  # t3: seat 1 takes it back
    write_game(tmp_path, "us_bot", "win", "1", g.raw())
    metrics = compute_metrics(open_replay("us_bot", "1", tmp_path))

    first = metrics[1].seats[0]
    assert (first.army, first.tiles, first.max_stack, first.max_stack_pos) == (5, 1, 5, (3, 0))
    assert (first.tiles_gained, first.tiles_lost) == (0, 0)

    walked = metrics[2].seats[0]
    assert (walked.army, walked.tiles) == (5, 2)
    assert (walked.max_stack, walked.max_stack_pos) == (4, (3, 1))
    assert (walked.tiles_gained, walked.tiles_lost) == (1, 0)

    lost, took = metrics[3].seats
    assert (lost.tiles, lost.tiles_gained, lost.tiles_lost) == (1, 0, 1)
    assert (took.tiles, took.army, took.tiles_gained) == (2, 3, 1)


def test_largest_stack_stays_on_the_held_cell_through_a_tie(tmp_path):
    g = GameBuilder(4, 4, generals=[(3, 0), (0, 3)])
    g.commit()  # t0
    g.set((3, 1), 0, 5).commit()  # t1: the pile is unambiguously at (3,1)
    g.set((3, 0), 0, 5).commit()  # t2: (3,0) ties it and sorts first
    write_game(tmp_path, "us_bot", "win", "1", g.raw())
    metrics = compute_metrics(open_replay("us_bot", "1", tmp_path))

    assert metrics[1].seats[0].max_stack_pos == (3, 1)
    assert metrics[2].seats[0].max_stack == 5
    assert metrics[2].seats[0].max_stack_pos == (3, 1)


def test_seat_summary_extremes_of_the_attack_game(tmp_path):
    analysis = analyzed(tmp_path, attack_game())
    ours = analysis.summaries[0]

    assert (ours.peak_stack, ours.peak_stack_tick) == (14, 13)
    assert (ours.peak_tiles, ours.peak_tiles_tick) == (15, 18)
    assert (ours.closest_approach, ours.closest_approach_tick) == (0, 18)
    assert (ours.final_tiles, analysis.summaries[1].final_tiles) == (15, 0)


# --------------------------------------------------------------------------- fog


def test_general_becomes_visible_the_tick_a_tile_enters_its_neighbourhood(tmp_path):
    g = board()
    g.commit()  # t0
    g.set((2, 2), 0, 1).commit()  # t1: nowhere near (0,5)
    g.set((1, 4), 0, 1).commit()  # t2: Chebyshev 1 from (0,5)
    write_game(tmp_path, "us_bot", "win", "1", g.raw())
    replay = open_replay("us_bot", "1", tmp_path)

    assert compute_vision(replay).first_general_sight == (2, None)
    assert not sees_cell(replay.ticks[1], 0, (0, 5), 6, 6)
    assert sees_cell(replay.ticks[2], 0, (0, 5), 6, 6)


def test_general_stays_knowable_after_the_watching_tile_is_lost(tmp_path):
    g = board()
    g.commit()  # t0
    g.set((1, 4), 0, 1).commit()  # t1: sighted
    g.set((1, 4), NEUTRAL, 0).commit()  # t2: the watcher is gone
    write_game(tmp_path, "us_bot", "win", "1", g.raw())
    replay = open_replay("us_bot", "1", tmp_path)
    vision = compute_vision(replay)

    assert vision.first_general_sight[0] == 1
    assert not sees_cell(replay.ticks[2], 0, (0, 5), 6, 6)
    assert not vision.knows_general(0, 0)
    assert vision.knows_general(0, 1)
    assert vision.knows_general(0, 2)


# --------------------------------------------------------------------------- path


def path_game(tmp_path):
    """
    Blind move, then a visible-tile target, then the general, then away, hold,
    jump — one of each, in that order.
    """
    g = board()
    g.commit()  # t0
    g.set((4, 0), 0, 5).commit()  # t1: move, nothing in sight
    g.set((3, 1), 1, 1).march((4, 0), (3, 0)).commit()  # t2: an enemy tile appears, in vision
    g.set((1, 4), 0, 1).march((3, 0), (2, 0)).commit()  # t3: the general is sighted
    g.march((2, 0), (3, 0)).commit()  # t4: back away from it
    g.commit()  # t5: hold
    g.set((0, 0), 0, 9).commit()  # t6: a different pile becomes the largest
    write_game(tmp_path, "us_bot", "win", "1", g.raw())
    replay = open_replay("us_bot", "1", tmp_path)
    metrics = compute_metrics(replay)
    return compute_path(replay, metrics, compute_vision(replay), 0)


def test_target_switches_from_visible_tile_to_general(tmp_path):
    steps = path_game(tmp_path)

    assert [step.tick for step in steps] == [1, 2, 3, 4, 5, 6]
    assert (steps[0].target_kind, steps[0].target) == (TARGET_NONE, None)
    assert (steps[1].target_kind, steps[1].target) == (TARGET_VISIBLE_TILE, (3, 1))
    assert (steps[2].target_kind, steps[2].target) == (TARGET_GENERAL, (0, 5))
    assert steps[5].target_kind == TARGET_GENERAL


def test_step_kinds_and_toward_away_classification(tmp_path):
    steps = path_game(tmp_path)

    assert [step.kind for step in steps] == ["move", "move", "move", "move", "hold", "jump"]
    assert steps[0].delta is None  # no target: neither toward nor away
    assert (steps[1].toward, steps[1].away) == (True, False)
    assert (steps[2].toward, steps[2].delta) == (True, -1)
    assert (steps[3].away, steps[3].delta) == (True, 1)
    assert not steps[4].moved and not steps[5].moved


def test_summarize_path_excludes_holds_jumps_and_blind_moves(tmp_path):
    summary = summarize_path(path_game(tmp_path))

    assert (summary.moves, summary.toward, summary.away) == (4, 2, 1)
    assert summary.blind_moves == 1
    assert summary.directed == 3
    assert summary.toward_fraction == pytest.approx(2 / 3)
    assert summary.away_fraction == pytest.approx(1 / 3)


# --------------------------------------------------------------------------- events


def climb_game(marches: int):
    """Seat 0's stack walks up column 0 over its own tiles, growing by 1 each tick."""
    g = board()
    for row in range(5):
        g.set((row, 0), 0, 2)
    g.set((5, 0), 0, 5)
    g.commit()  # t0
    g.grow((5, 0)).commit()  # t1: the stack holds at 6 — the run can only start at t2
    column = [(5, 0), (4, 0), (3, 0), (2, 0), (1, 0), (0, 0)]
    for index in range(marches):
        g.march(column[index], column[index + 1]).commit()
    return g.raw()


def test_gather_wave_fires_at_exactly_the_threshold(tmp_path):
    analysis = analyzed(tmp_path, climb_game(5))
    waves = analysis.events.of_kind("gather_wave", 0)

    assert analysis.events.gather_waves == (1, 0)
    assert len(waves) == 1
    assert waves[0].tick == 2
    assert waves[0].data["start"] == 2
    assert waves[0].data["end"] == 6
    assert (waves[0].data["from_army"], waves[0].data["to_army"]) == (6, 11)


def test_gather_wave_one_tick_short_does_not_fire(tmp_path):
    analysis = analyzed(tmp_path, climb_game(4))
    steps = [s for s in analysis.steps[0] if s.moved]

    assert len(steps) == 4
    assert analysis.events.of_kind("gather_wave", 0) == []
    assert analysis.events.gather_waves == (0, 0)


def capture_game(defender_army: int):
    g = GameBuilder(5, 5, generals=[(4, 0), (0, 4)])
    g.set((2, 1), 0, defender_army + 2).set((2, 2), 1, defender_army)
    g.commit(2)  # t0, t1
    g.march((2, 1), (2, 2)).commit()  # t2
    return g.raw()


def test_big_capture_fires_at_the_army_threshold(tmp_path):
    analysis = analyzed(tmp_path, capture_game(10))
    events = analysis.events.of_kind("big_capture", 0)

    assert len(events) == 1
    assert (events[0].tick, events[0].cell, events[0].data["army"]) == (2, (2, 2), 10)
    assert analysis.events.first_capture == (2, None)


def test_capture_below_the_threshold_is_not_a_big_capture(tmp_path):
    analysis = analyzed(tmp_path, capture_game(9))

    assert analysis.events.of_kind("big_capture") == []
    assert analysis.events.first_capture == (2, None)


def test_general_capture_and_game_end_close_the_log(tmp_path):
    analysis = analyzed(tmp_path, attack_game())
    (killed,) = analysis.events.of_kind("general_captured")
    (ended,) = analysis.events.of_kind("game_end")

    assert (killed.tick, killed.player, killed.cell) == (18, 0, (0, 5))
    assert killed.data == {"loser": 1}
    assert ended.tick == 18
    assert "us_bot" in ended.detail
    assert analysis.events.first_contact == 6
    assert analysis.vision.first_general_sight == (6, None)


def castle_game(production_ticks: tuple[int, ...], length: int, cells=((2, 2),)):
    """Owned cells that gain +1 on the given ticks and nothing else."""
    g = board()
    for cell in cells:
        g.set(cell, 0, 1)
    g.commit()  # t0
    for tick in range(1, length + 1):
        if tick in production_ticks:
            for cell in cells:
                g.grow(cell)
        g.commit()
    return g.raw()


def test_castle_is_detected_from_repeated_off_growth_production(tmp_path):
    analysis = analyzed(tmp_path, castle_game((2, 4), length=6))
    (built,) = analysis.events.of_kind("castle_built")

    assert analysis.events.castles == {(2, 2): 2}
    assert (built.tick, built.player, built.cell) == (2, 0, (2, 2))


def test_one_production_tick_is_not_enough_to_confirm_a_castle(tmp_path):
    analysis = analyzed(tmp_path, castle_game((2,), length=6))

    assert analysis.events.castles == {}
    assert analysis.events.of_kind("castle_built") == []


def test_bulk_growth_tick_does_not_count_as_a_confirmation(tmp_path):
    """+1 on tick 50 is the every-50 land growth, not a structure producing."""
    bulk = analyzed(tmp_path, castle_game((50, 52), length=54), match_id="1")
    off_growth = analyzed(tmp_path, castle_game((52, 54), length=56), match_id="2")

    assert bulk.events.castles == {}
    assert off_growth.events.castles == {(2, 2): 52}


def test_a_whole_block_of_cells_producing_at_once_is_not_castles(tmp_path):
    block = tuple((r, c) for r in (1, 2, 3) for c in (1, 2, 3))
    analysis = analyzed(tmp_path, castle_game((2, 4), length=6, cells=block))

    assert len(block) == 9
    assert analysis.events.castles == {}


def tiles_metrics(ours: list[int], theirs: list[int]) -> list[TickMetrics]:
    """TickMetrics carrying nothing but a tile count per seat."""

    def seat(tiles: int) -> PlayerTick:
        return PlayerTick(
            army=tiles,
            tiles=tiles,
            max_stack=0,
            max_stack_pos=None,
            max_stack_dist=None,
            general_army=0,
            nearest_tile_dist=None,
            nearest_tile_pos=None,
            tiles_gained=0,
            tiles_lost=0,
        )

    return [
        TickMetrics(tick=tick, seats=(seat(a), seat(b)))
        for tick, (a, b) in enumerate(zip(ours, theirs, strict=True))
    ]


def test_expansion_stall_needs_more_than_the_window(tmp_path):
    long_enough = tiles_metrics([5] * 52, list(range(52)))
    one_short = tiles_metrics([5] * 51, list(range(51)))

    (stall,) = find_stalls(long_enough, 0, 50, 10)
    assert (stall.tick, stall.data["end"], stall.data["tiles"]) == (0, 51, 5)
    assert stall.data["opponent_gain"] == 51
    assert find_stalls(one_short, 0, 50, 10) == []


def test_expansion_stall_needs_the_opponent_to_actually_gain(tmp_path):
    flat_opponent = tiles_metrics([5] * 52, [3] * 46 + list(range(3, 9)))

    assert find_stalls(flat_opponent, 0, 50, 10) == []


def test_tile_loss_streak_needs_both_length_and_depth(tmp_path):
    bleeding = tiles_metrics(list(range(20, -1, -1)), [1] * 21)
    short = tiles_metrics(list(range(20, 0, -1)), [1] * 20)
    shallow = tiles_metrics([20] * 22, [1] * 22)

    (streak,) = find_loss_streaks(bleeding, 0, 20, 5)
    assert (streak.tick, streak.data["end"], streak.data["tiles_lost"]) == (0, 20, 20)
    assert find_loss_streaks(short, 0, 20, 5) == []
    assert find_loss_streaks(shallow, 0, 20, 5) == []


# --------------------------------------------------------------------------- phases


def test_phases_cover_every_tick_of_the_attack_game(tmp_path):
    analysis = analyzed(tmp_path, attack_game())

    assert [(p.name, p.start, p.end) for p in analysis.phases] == [
        ("expansion", 0, 5),
        ("contest", 6, 18),
    ]
    covered = {tick for phase in analysis.phases for tick in range(phase.start, phase.end + 1)}
    assert covered == set(range(19))


def test_a_game_without_contact_is_one_long_expansion(tmp_path):
    analysis = analyzed(tmp_path, blind_game())

    assert analysis.events.first_contact is None
    assert [(p.name, p.start, p.end) for p in analysis.phases] == [("expansion", 0, 7)]


def test_collapse_start_ignores_dips_that_recover(tmp_path):
    sustained = tiles_metrics([20] * 30 + [10] * 10, [1] * 40)
    recovering = tiles_metrics([20] * 30 + [10] * 5 + [20] * 5, [1] * 40)

    assert collapse_start(sustained, 0, after=5) == 30
    assert collapse_start(recovering, 0, after=5) is None


def test_detect_events_is_a_pure_read_of_the_pipeline(tmp_path):
    """`detect_events` re-run on the same inputs yields the same log."""
    analysis = analyzed(tmp_path, attack_game())
    again = detect_events(
        analysis.replay, analysis.metrics, analysis.vision, analysis.steps
    )

    assert [event.as_json() for event in again.events] == [
        event.as_json() for event in analysis.events.events
    ]
