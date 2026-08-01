"""
Compose an analysis into sections an agent can read, or parse.

Four sections, each with a text and a JSON form:

- `timeline` — sampled economy for both seats;
- `events` — the chronological log, phase-tagged;
- `flow` — information against action: when the enemy general became knowable,
  and whether the biggest stack ever went at it;
- `verdict` — a few sentences of plain prose, generated from the same numbers.

"Us" is the seat of the player whose folder the replay came from; "them" is the
other one. Nothing here re-derives anything: it only formats `Analysis`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from arena.instrument.replay.analysis import Analysis
from arena.instrument.replay.events import GATHER_MIN_TICKS
from arena.instrument.replay.path import TARGET_GENERAL
from arena.instrument.replay.phases import phase_at

DEFAULT_EVERY = 16


def _cell(pos) -> str:
    return f"({pos[0]},{pos[1]})" if pos else "-"


def _fraction(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


@dataclass(frozen=True)
class Report:
    header: dict[str, Any]
    timeline: list[dict[str, Any]]
    events: list[dict[str, Any]]
    phases: list[dict[str, Any]]
    flow: dict[str, Any]
    verdict: str

    def as_json(self, sections: tuple[str, ...]) -> dict[str, Any]:
        out: dict[str, Any] = {"header": self.header}
        if "timeline" in sections:
            out["timeline"] = self.timeline
        if "events" in sections:
            out["phases"] = self.phases
            out["events"] = self.events
        if "flow" in sections:
            out["flow"] = self.flow
        if "verdict" in sections:
            out["verdict"] = self.verdict
        return out


def build_report(analysis: Analysis, every: int = DEFAULT_EVERY) -> Report:
    return Report(
        header=_header(analysis),
        timeline=_timeline(analysis, every),
        events=[
            {**event.as_json(), "phase": phase_at(analysis.phases, event.tick)}
            for event in analysis.events.events
        ],
        phases=[phase.as_json() for phase in analysis.phases],
        flow=_flow(analysis),
        verdict=verdict(analysis),
    )


def _header(analysis: Analysis) -> dict[str, Any]:
    replay = analysis.replay
    return {
        "match_id": replay.match_id,
        "path": str(replay.path),
        "outcome": replay.outcome,
        "folder": replay.folder,
        "folder_disagrees": replay.folder_disagrees,
        "queried_player": replay.queried_player,
        "us": analysis.us,
        "them": analysis.them,
        "players": list(replay.players),
        "self_match": replay.is_self_match,
        "board": {"rows": replay.rows, "cols": replay.cols},
        "mountains": len(replay.mountains),
        "seed": replay.seed,
        "total_ticks": replay.total_ticks,
        "winner": replay.winner,
        "winner_name": replay.name(replay.winner) if replay.winner >= 0 else None,
        "generals": [list(g) for g in replay.generals],
        "created_at": replay.meta.created_at if replay.meta else None,
    }


def _timeline(analysis: Analysis, every: int) -> list[dict[str, Any]]:
    every = max(1, every)
    last = analysis.metrics[-1].tick
    rows = []
    for entry in analysis.metrics:
        if entry.tick % every and entry.tick != last:
            continue
        rows.append(
            {
                "tick": entry.tick,
                "phase": phase_at(analysis.phases, entry.tick),
                "us": entry.seats[analysis.us].as_json(),
                "them": entry.seats[analysis.them].as_json(),
            }
        )
    return rows


def _flow(analysis: Analysis) -> dict[str, Any]:
    out: dict[str, Any] = {"seats": {}}
    for label, player in (("us", analysis.us), ("them", analysis.them)):
        seen = analysis.vision.first_general_sight[player]
        steps = analysis.steps[player]
        after = [s for s in steps if s.target_kind == TARGET_GENERAL]
        toward_after = sum(1 for s in after if s.toward)
        away_after = sum(1 for s in after if s.away)
        waves = analysis.events.of_kind("gather_wave", player)
        summary = analysis.summaries[player]
        overall = analysis.path_summary(player)
        out["seats"][label] = {
            "seat": player,
            "name": analysis.seat_name(player),
            "enemy_general": list(analysis.replay.enemy_general(player)),
            "first_general_sight": seen,
            "knowable_from": seen,
            "ticks_knowing": (analysis.metrics[-1].tick - seen + 1) if seen is not None else 0,
            "moves_toward_known_general": toward_after,
            "moves_away_from_known_general": away_after,
            "ever_moved_at_known_general": toward_after > 0,
            "closest_approach": summary.closest_approach,
            "closest_approach_tick": summary.closest_approach_tick,
            "peak_stack": summary.peak_stack,
            "peak_stack_tick": summary.peak_stack_tick,
            "gather_waves": len(waves),
            "gather_waves_at_general": sum(
                1 for w in waves if w.data["target_kind"] == TARGET_GENERAL
            ),
            "biggest_gather_wave": max((w.data["to_army"] for w in waves), default=0),
            "gather_wave_detail": [w.as_json() for w in waves],
            "path": overall.as_json(),
            "path_by_phase": [s.as_json() for s in analysis.phase_summaries(player)],
        }
    return out


def verdict(analysis: Analysis) -> str:
    """Three to six sentences, all of them restatements of measured numbers."""
    replay = analysis.replay
    us, them = analysis.us, analysis.them
    ours, theirs = analysis.summaries[us], analysis.summaries[them]
    our_name, their_name = analysis.seat_name(us), analysis.seat_name(them)
    contact = analysis.events.first_contact
    seen = analysis.vision.first_general_sight[us]
    waves = analysis.events.of_kind("gather_wave", us)
    their_waves = analysis.events.of_kind("gather_wave", them)
    stalls = analysis.events.of_kind("expansion_stall", us)
    path = analysis.path_summary(us)

    lines = []
    opening = (
        f"{our_name} (seat {us}) and {their_name} (seat {them}) split {replay.rows}x{replay.cols} "
        f"land until they touched at tick {contact}"
        if contact is not None
        else f"{our_name} (seat {us}) and {their_name} (seat {them}) never made contact"
    )
    lines.append(
        f"{opening}; land peaked at {ours.peak_tiles} tiles for {our_name} "
        f"(tick {ours.peak_tiles_tick}) against {theirs.peak_tiles} for {their_name}."
    )
    lines.append(
        f"By the end the split was {ours.final_tiles} tiles / {ours.final_army} army to "
        f"{theirs.final_tiles} / {theirs.final_army}."
    )
    if seen is None:
        lines.append(
            f"{our_name} never saw {their_name}'s general at {list(replay.enemy_general(us))} — "
            f"the closest any of its tiles came was {ours.closest_approach} steps "
            f"at tick {ours.closest_approach_tick}, so every move it made was played blind."
        )
    else:
        lines.append(
            f"{our_name} saw {their_name}'s general at tick {seen} and had "
            f"{analysis.metrics[-1].tick - seen + 1} ticks of knowing where it was, moving "
            f"toward it on {_fraction(path.toward_fraction)} of its largest-stack moves."
        )
    if waves:
        at_general = sum(1 for w in waves if w.data["target_kind"] == TARGET_GENERAL)
        biggest = max(w.data["to_army"] for w in waves)
        aim = (
            f"{at_general} of them aimed at a general it could place"
            if at_general
            else "none of them aimed at a general it could place"
        )
        lines.append(
            f"It gathered {len(waves)} time(s), the biggest carrying {biggest} army, {aim}."
        )
    else:
        lines.append(
            f"It never once gathered: no run of {GATHER_MIN_TICKS} moving ticks with a growing "
            f"stack, and the largest pile it ever held was {ours.peak_stack} army at tick "
            f"{ours.peak_stack_tick} — it expanded and defended, and never assembled an attack."
        )
    if stalls:
        worst = max(stalls, key=lambda e: e.data["end"] - e.data["start"])
        lines.append(
            f"Its expansion stalled at ticks {worst.data['start']}-{worst.data['end']} "
            f"(<= {worst.data['tiles']} tiles) while {their_name} gained "
            f"{worst.data['opponent_gain']}."
        )
    ending = (
        f"{replay.name(replay.winner)} won at tick {replay.total_ticks}"
        if replay.winner >= 0
        else f"the game was drawn after {replay.total_ticks} ticks"
    )
    lines.append(
        f"{ending}; {their_name} ran {len(their_waves)} gather wave(s) to "
        f"{our_name}'s {len(waves)}."
    )
    return " ".join(lines)


def render_header(report: Report) -> str:
    header = report.header
    board = header["board"]
    winner = header["winner_name"] or "draw"
    tag = " [self-match]" if header["self_match"] else ""
    filed = (
        f" [filed under {header['folder']}/, side A's result]"
        if header["folder_disagrees"]
        else ""
    )
    return (
        f"match {header['match_id']} — {header['players'][0]} (seat 0) vs "
        f"{header['players'][1]} (seat 1){tag}\n"
        f"  {board['rows']}x{board['cols']} board, {header['mountains']} mountains, "
        f"seed {header['seed']}, {header['total_ticks']} ticks\n"
        f"  queried {header['queried_player']} = seat {header['us']} ({header['outcome']}){filed}, "
        f"winner: {winner}\n"
        f"  generals: seat 0 {tuple(header['generals'][0])}, seat 1 {tuple(header['generals'][1])}\n"
        f"  {header['path']}"
    )


def render_timeline(report: Report) -> str:
    us = report.header["players"][report.header["us"]]
    them = report.header["players"][report.header["them"]]
    lines = [
        "TIMELINE  (dgen = Manhattan distance to the enemy general; "
        "near = closest owned tile to it)",
        f"{'':>6} {'':<9} | {us[:36]:^36} | {them[:36]:^36}",
        f"{'tick':>6} {'phase':<9} | {'army':>5} {'tiles':>5} {'stack':>6} {'at':>8} "
        f"{'dgen':>3} {'near':>4} | {'army':>5} {'tiles':>5} {'stack':>6} {'at':>8} "
        f"{'dgen':>3} {'near':>4}",
    ]
    for row in report.timeline:
        cells = []
        for side in ("us", "them"):
            seat = row[side]
            cells.append(
                f"{seat['army']:>5} {seat['tiles']:>5} {seat['max_stack']:>6} "
                f"{_cell(seat['max_stack_pos']):>8} "
                f"{seat['max_stack_dist'] if seat['max_stack_dist'] is not None else '-':>3} "
                f"{seat['nearest_tile_dist'] if seat['nearest_tile_dist'] is not None else '-':>4}"
            )
        lines.append(f"{row['tick']:>6} {row['phase']:<9} | {cells[0]} | {cells[1]}")
    return "\n".join(lines)


def render_events(report: Report) -> str:
    names = report.header["players"]
    lines = ["PHASES"]
    for phase in report.phases:
        lines.append(f"  {phase['name']:<10} {phase['start']:>5}-{phase['end']:<5} {phase['note']}")
    lines.append("")
    lines.append("EVENTS")
    if not report.events:
        lines.append("  (none)")
    for event in report.events:
        who = "" if event["player"] is None else f"{names[event['player']]}: "
        where = f" at {tuple(event['cell'])}" if event["cell"] else ""
        lines.append(
            f"  t{event['tick']:<5} [{event['phase']:<9}] {event['kind']:<20} "
            f"{who}{event['detail']}{where}"
        )
    return "\n".join(lines)


def render_flow(report: Report) -> str:
    lines = ["INFORMATION vs ACTION"]
    for label in ("us", "them"):
        seat = report.flow["seats"][label]
        lines.append(f"  {label.upper()}  {seat['name']} (seat {seat['seat']})")
        if seat["first_general_sight"] is None:
            lines.append(
                f"    enemy general {tuple(seat['enemy_general'])}: NEVER seen — "
                f"closest owned tile {seat['closest_approach']} steps away "
                f"at tick {seat['closest_approach_tick']}"
            )
        else:
            lines.append(
                f"    enemy general {tuple(seat['enemy_general'])}: first seen tick "
                f"{seat['first_general_sight']}, knowable for {seat['ticks_knowing']} ticks; "
                f"largest stack moved at it {seat['moves_toward_known_general']} times, "
                f"away {seat['moves_away_from_known_general']} "
                f"({'acted on it' if seat['ever_moved_at_known_general'] else 'NEVER acted on it'})"
            )
        path = seat["path"]
        lines.append(
            f"    largest stack: peak {seat['peak_stack']} army (tick {seat['peak_stack_tick']}), "
            f"{path['moves']} moves — toward {_fraction(path['toward_fraction'])}, "
            f"away {_fraction(path['away_fraction'])}, {path['blind_moves']} with no target"
        )
        for phase in seat["path_by_phase"]:
            lines.append(
                f"      {phase['label']:<10} {phase['start']:>5}-{phase['end']:<5} "
                f"{phase['moves']:>4} moves, toward {_fraction(phase['toward_fraction'])}, "
                f"away {_fraction(phase['away_fraction'])}"
            )
        if seat["gather_waves"]:
            lines.append(
                f"    gather waves: {seat['gather_waves']} "
                f"({seat['gather_waves_at_general']} aimed at a known general, "
                f"biggest carried {seat['biggest_gather_wave']} army)"
            )
            for wave in seat["gather_wave_detail"]:
                data = wave["data"]
                lines.append(
                    f"      t{data['start']}-{data['end']}: {data['from_army']}->"
                    f"{data['to_army']} army, ended {data['end_dist']} from its "
                    f"{data['target_kind']} target"
                )
        else:
            lines.append("    gather waves: NONE — never carried a growing stack anywhere")
    return "\n".join(lines)


def render_verdict(report: Report) -> str:
    return f"VERDICT\n  {report.verdict}"


RENDERERS = {
    "timeline": render_timeline,
    "events": render_events,
    "flow": render_flow,
    "verdict": render_verdict,
}


def render(report: Report, sections: tuple[str, ...]) -> str:
    blocks = [render_header(report)]
    blocks.extend(RENDERERS[name](report) for name in sections if name in RENDERERS)
    return "\n\n".join(blocks)
