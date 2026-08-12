"""
Report sections, batch aggregation, and the `scripts/replay.py` CLI.

The CLI is driven through `main(argv)` rather than a subprocess: the suite
budget is 15 s (AGENTS.md tester §4) and an interpreter launch buys nothing
here.
"""
from __future__ import annotations

import json

from fixtures.replay_game import (
    attack_game,
    blind_game,
    forfeit_game,
    meta_row,
    write_game,
)

from arena.instrument.replay.analysis import analyze
from arena.instrument.replay.batch import aggregate, iter_games, scan_game
from arena.instrument.replay.loader import open_replay
from arena.instrument.replay.report import build_report, render
from scripts.replay import main


def folder(tmp_path):
    """A player folder holding one win, one loss, and one forfeit."""
    write_game(tmp_path, "us_bot", "win", "77", attack_game(), meta_row("77"))
    write_game(tmp_path, "us_bot", "lose", "78", blind_game(), meta_row("78", winner="B", turns=7))
    write_game(tmp_path, "us_bot", "draw", "79", forfeit_game(), meta_row("79", winner="D", turns=1))
    return tmp_path


def report_of(tmp_path, raw, every: int = 16, match_id: str = "77"):
    write_game(tmp_path, "us_bot", "win", match_id, raw, meta_row(match_id))
    return build_report(analyze(open_replay("us_bot", match_id, tmp_path)), every=every)


# --------------------------------------------------------------------------- report


def test_header_names_the_seats_and_the_outcome(tmp_path):
    header = report_of(tmp_path, attack_game()).header

    assert header["match_id"] == "77"
    assert header["queried_player"] == "us_bot"
    assert (header["us"], header["them"]) == (0, 1)
    assert header["players"] == ["us_bot", "them_bot"]
    assert header["board"] == {"rows": 6, "cols": 6}
    assert (header["total_ticks"], header["winner"], header["winner_name"]) == (18, 0, "us_bot")
    assert header["generals"] == [[5, 0], [0, 5]]
    assert header["created_at"] == "2026-08-01T00:00:00Z"
    assert not header["self_match"]


def test_timeline_samples_every_n_ticks_and_always_keeps_the_last(tmp_path):
    report = report_of(tmp_path, attack_game(), every=8)

    assert [row["tick"] for row in report.timeline] == [0, 8, 16, 18]
    assert [row["phase"] for row in report.timeline] == [
        "expansion",
        "contest",
        "contest",
        "contest",
    ]
    assert report.timeline[1]["us"]["max_stack"] == 9
    assert report.timeline[1]["us"]["max_stack_pos"] == [5, 0]


def test_flow_pairs_what_was_knowable_with_what_was_done(tmp_path):
    seats = report_of(tmp_path, attack_game()).flow["seats"]

    ours = seats["us"]
    assert (ours["first_general_sight"], ours["knowable_from"]) == (6, 6)
    assert ours["ticks_knowing"] == 13
    assert ours["moves_toward_known_general"] == 10
    assert ours["moves_away_from_known_general"] == 0
    assert ours["ever_moved_at_known_general"]
    assert (ours["gather_waves"], ours["gather_waves_at_general"]) == (1, 1)
    assert ours["biggest_gather_wave"] == 14
    assert ours["path"]["toward_fraction"] == 1.0
    assert [phase["label"] for phase in ours["path_by_phase"]] == ["expansion", "contest"]

    theirs = seats["them"]
    assert theirs["first_general_sight"] is None
    assert theirs["ticks_knowing"] == 0
    assert theirs["gather_waves"] == 0
    assert theirs["path"]["moves"] == 0
    assert theirs["path"]["toward_fraction"] is None


def test_verdict_calls_out_a_general_that_was_never_seen(tmp_path):
    verdict = report_of(tmp_path, blind_game(), match_id="78").verdict

    assert "never made contact" in verdict
    assert "never saw them_bot's general at [0, 5]" in verdict
    assert "played blind" in verdict
    assert "never once gathered" in verdict


def test_verdict_reports_the_gather_wave_it_found(tmp_path):
    verdict = report_of(tmp_path, attack_game()).verdict

    assert "touched at tick 6" in verdict
    assert "saw them_bot's general at tick 6" in verdict
    assert "gathered 1 time(s), the biggest carrying 14 army" in verdict
    assert "us_bot won at tick 18" in verdict


def test_render_emits_one_block_per_requested_section(tmp_path):
    report = report_of(tmp_path, attack_game())
    text = render(report, ("timeline", "events", "flow", "verdict"))

    assert text.startswith("match 77 — us_bot (seat 0) vs them_bot (seat 1)")
    for heading in ("TIMELINE", "PHASES", "EVENTS", "INFORMATION vs ACTION", "VERDICT"):
        assert heading in text
    assert "gather_wave" in text
    assert "general_captured" in text
    assert "INFORMATION vs ACTION" not in render(report, ("verdict",))


def test_json_sections_follow_the_requested_subset(tmp_path):
    report = report_of(tmp_path, attack_game())

    assert set(report.as_json(("timeline", "verdict"))) == {"header", "timeline", "verdict"}
    assert set(report.as_json(("events",))) == {"header", "phases", "events"}
    assert set(report.as_json(("flow",))) == {"header", "flow"}


# --------------------------------------------------------------------------- CLI


def test_cli_full_json_carries_every_section(tmp_path, capsys):
    folder(tmp_path)

    assert main(["--root", str(tmp_path), "full", "us_bot", "77", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert set(payload) == {"header", "timeline", "phases", "events", "flow", "verdict"}
    assert payload["header"]["outcome"] == "win"
    kinds = {event["kind"] for event in payload["events"]}
    assert {"first_contact", "first_general_sight", "gather_wave", "general_captured"} <= kinds
    assert all("phase" in event for event in payload["events"])


def test_cli_text_run_prints_the_named_sections(tmp_path, capsys):
    folder(tmp_path)

    assert main(["--root", str(tmp_path), "summarize", "us_bot", "77", "--every", "8"]) == 0
    out = capsys.readouterr().out

    assert "TIMELINE" in out and "VERDICT" in out
    assert "EVENTS" not in out


def test_cli_reports_a_missing_replay_and_a_forfeit_separately(tmp_path, capsys):
    folder(tmp_path)

    assert main(["--root", str(tmp_path), "full", "us_bot", "999"]) == 2
    assert "no replay '999'" in capsys.readouterr().err
    assert main(["--root", str(tmp_path), "full", "us_bot", "79"]) == 3
    assert "a forfeit" in capsys.readouterr().err


# --------------------------------------------------------------------------- batch


def test_scan_game_reduces_a_replay_to_its_flaw_signature(tmp_path):
    folder(tmp_path)
    won = scan_game(open_replay("us_bot", "77", tmp_path))
    lost = scan_game(open_replay("us_bot", "78", tmp_path))

    assert (won.outcome, won.ticks, won.seat, won.opponent) == ("win", 18, 0, "them_bot")
    assert (won.first_contact, won.gather_waves, won.moves) == (6, 1, 10)
    assert (won.toward_fraction, won.peak_stack, won.saw_general) == (1.0, 14, 6)
    assert won.final_tiles == (11, 4)  # the frame before the general fell, not after

    assert (lost.first_contact, lost.gather_waves, lost.moves) == (None, 0, 0)
    assert (lost.toward_fraction, lost.saw_general) == (None, None)
    assert (lost.peak_stack, lost.closest_approach, lost.final_tiles) == (4, 8, (3, 5))


def test_iter_games_yields_none_for_a_forfeit(tmp_path):
    folder(tmp_path)
    seen = [(line, outcome, path.stem) for line, outcome, path in iter_games("us_bot", root=tmp_path)]

    assert [entry[1:] for entry in seen] == [("win", "77"), ("lose", "78"), ("draw", "79")]
    assert seen[2][0] is None
    assert seen[0][0] is not None and seen[1][0] is not None


def test_aggregate_counts_flaws_overall_and_per_outcome(tmp_path):
    folder(tmp_path)
    lines = [line for line, _, _ in iter_games("us_bot", root=tmp_path) if line is not None]
    summary = aggregate(lines, forfeits=1)

    assert (summary["games"], summary["forfeits_skipped"], summary["self_matches"]) == (2, 1, 0)
    assert list(summary["by_outcome"]) == ["all", "win", "lose"]

    everything = summary["by_outcome"]["all"]
    assert everything["no_gather_wave"] == 1
    assert everything["never_saw_general"] == 1
    assert everything["no_contact"] == 1
    assert everything["median_gather_waves"] == 0.5
    assert everything["median_toward_fraction"] == 1.0  # the blind game has no directed moves
    assert everything["median_ticks"] == 12.5

    assert summary["by_outcome"]["win"]["never_saw_general"] == 0
    assert summary["by_outcome"]["lose"]["never_saw_general"] == 1


def test_cli_batch_json_lists_games_forfeits_and_the_summary(tmp_path, capsys):
    folder(tmp_path)

    assert main(["--root", str(tmp_path), "batch", "us_bot", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["player"] == "us_bot"
    assert payload["outcome_filter"] == "all"
    assert payload["forfeits"] == ["79"]
    assert [game["match_id"] for game in payload["games"]] == ["77", "78"]
    assert payload["summary"]["games"] == 2


def test_cli_batch_streams_a_line_per_game_then_the_summary(tmp_path, capsys):
    folder(tmp_path)

    assert main(["--root", str(tmp_path), "batch", "us_bot"]) == 0
    out = capsys.readouterr().out

    assert "forfeit (<= 1 tick), skipped" in out
    assert "SUMMARY for us_bot: 2 played games (1 forfeits skipped, 0 self-matches)" in out
    assert "no gather wave in 1/2" in out


def test_cli_batch_can_restrict_to_one_outcome(tmp_path, capsys):
    folder(tmp_path)

    assert main(["--root", str(tmp_path), "batch", "us_bot", "--outcome", "lose", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert [game["match_id"] for game in payload["games"]] == ["78"]
    assert payload["forfeits"] == []
    assert list(payload["summary"]["by_outcome"]) == ["lose"]


def test_batch_selects_on_the_real_outcome_not_the_folder(tmp_path, capsys):
    """
    A side-B game is filed under side A's result: `win/` for a game us_bot lost.

    Roughly half of every scraped window is side B, so a folder-driven filter
    silently samples the wrong games — the defect this guards.
    """
    write_game(
        tmp_path,
        "us_bot",
        "win",
        "90",
        attack_game(players=("them_bot", "us_bot")),
        meta_row("90", a_name="them_bot", b_name="us_bot"),
    )
    line = scan_game(open_replay("us_bot", "90", tmp_path))
    assert (line.seat, line.folder, line.outcome) == (1, "win", "lose")

    assert main(["--root", str(tmp_path), "batch", "us_bot", "--outcome", "lose", "--json"]) == 0
    assert [g["match_id"] for g in json.loads(capsys.readouterr().out)["games"]] == ["90"]
    assert main(["--root", str(tmp_path), "batch", "us_bot", "--outcome", "win"]) == 2
    assert "no played replays" in capsys.readouterr().err


def test_cli_batch_with_nothing_playable_fails_loudly(tmp_path, capsys):
    write_game(tmp_path, "us_bot", "draw", "79", forfeit_game())

    assert main(["--root", str(tmp_path), "batch", "us_bot"]) == 2
    assert "no played replays for us_bot" in capsys.readouterr().err
