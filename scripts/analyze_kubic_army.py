#!/usr/bin/env python3
"""Kubic army management & routing on the fit-win corpus (analyst #3).

Measures accumulation (general bank / tip / castles), gather waves, sustained
move-streak start sizes, half vs full move rates by phase, home reserve, and
tip mass at sight / near kill. Skims losses for mismanagement signals.

Writes:
  docs/research/measurements/grok-kubic-army.json
  docs/research/measurements/grok-kubic-army.md

Does not write data/games/, data/ratings/, or data/remote_games/.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arena.instrument.replay.analysis import Analysis
from arena.instrument.replay.loader import Cell, Replay
from arena.instrument.replay.metrics import manhattan
from arena.instrument.replay.path import StackStep
from scripts.kubic_corpus import (
    MEASUREMENTS,
    corpus_meta,
    dist_summary,
    dump_json,
    iter_analyzed,
)
from scripts.kubic_moves import infer_moves_for_player

PLAYER = "Kubic"
SUSTAINED_MOVE_MIN = 5  # ticks of consecutive max-stack moves
NEAR_KILL_WINDOW = 20  # ticks before game end
GATHER_MIN_TICKS = 5  # matches events.GATHER_MIN_TICKS
JSON_OUT = "grok-kubic-army.json"
MD_OUT = "grok-kubic-army.md"

# Army-routing phase buckets (mutually exclusive when contact+sight exist).
# expansion: before first_contact
# contest: contact .. first_general_sight-1 (or end if never sight)
# post_sight: first_general_sight .. end
PHASE_NAMES = ("expansion", "contest", "post_sight")


@dataclass
class Accumulators:
    # per-game scalars
    max_gen_army: list[float] = field(default_factory=list)
    gen_at_contact: list[float] = field(default_factory=list)
    gen_at_sight: list[float] = field(default_factory=list)
    tip_at_sight: list[float] = field(default_factory=list)
    tip_near_kill: list[float] = field(default_factory=list)
    tip_at_kill: list[float] = field(default_factory=list)
    mean_gen_frac: list[float] = field(default_factory=list)
    mean_tip_frac: list[float] = field(default_factory=list)
    mean_castle_frac: list[float] = field(default_factory=list)
    peak_tip_frac: list[float] = field(default_factory=list)
    peak_gen_frac: list[float] = field(default_factory=list)
    waves_per_game: list[float] = field(default_factory=list)
    waves_pre_sight: list[float] = field(default_factory=list)
    waves_post_sight: list[float] = field(default_factory=list)
    games_with_zero_waves: int = 0
    games_analyzed: int = 0
    games_with_sight: int = 0
    games_with_contact: int = 0

    # wave-level
    wave_lengths: list[float] = field(default_factory=list)
    wave_from_army: list[float] = field(default_factory=list)
    wave_to_army: list[float] = field(default_factory=list)
    wave_start_dist_home: list[float] = field(default_factory=list)
    wave_end_dist_home: list[float] = field(default_factory=list)
    wave_start_dist_enemy_gen: list[float] = field(default_factory=list)
    wave_end_dist_enemy_gen: list[float] = field(default_factory=list)
    wave_pre_sight_n: int = 0
    wave_post_sight_n: int = 0

    # sustained move streaks
    streak_start_army: list[float] = field(default_factory=list)
    streak_lengths: list[float] = field(default_factory=list)
    streak_start_gen_army: list[float] = field(default_factory=list)

    # move kinds: overall + by phase
    move_kind_counts: Counter = field(default_factory=Counter)
    move_kind_by_phase: dict[str, Counter] = field(
        default_factory=lambda: {p: Counter() for p in PHASE_NAMES}
    )
    actionable_by_phase: dict[str, Counter] = field(
        default_factory=lambda: {p: Counter() for p in PHASE_NAMES}
    )

    # per-game tip/gen at contact for context
    tip_at_contact: list[float] = field(default_factory=list)
    tip_frac_at_sight: list[float] = field(default_factory=list)
    tip_frac_near_kill: list[float] = field(default_factory=list)
    gen_frac_at_sight: list[float] = field(default_factory=list)

    # counterexample trackers
    high_gen_low_tip_at_sight: list[dict] = field(default_factory=list)
    zero_wave_wins: list[str] = field(default_factory=list)
    half_heavy_games: list[dict] = field(default_factory=list)

    # loss skim
    loss_rows: list[dict] = field(default_factory=list)


def _castle_army(replay: Replay, tick: int, us: int, castles: set[Cell]) -> int:
    frame = replay.ticks[tick]
    total = 0
    for r, c in castles:
        if frame.owners[r][c] == us:
            total += frame.armies[r][c]
    return total


def _army_phase(tick: int, contact: int | None, sight: int | None) -> str:
    if contact is None or tick < contact:
        return "expansion"
    if sight is None or tick < sight:
        return "contest"
    return "post_sight"


def _sustained_move_streaks(steps: list[StackStep], metrics, us: int) -> list[dict]:
    """Runs of consecutive max-stack `move` steps with length >= SUSTAINED_MOVE_MIN."""
    out: list[dict] = []
    run: list[StackStep] = []

    def flush() -> None:
        if len(run) < SUSTAINED_MOVE_MIN:
            run.clear()
            return
        first = run[0]
        # army just before the streak (prior tick), else army on first move
        start_army = first.army
        start_gen = None
        prev_tick = first.tick - 1
        if 0 <= prev_tick < len(metrics):
            seat = metrics[prev_tick].seats[us]
            # Prefer prior max_stack when the same tip was holding
            start_army = seat.max_stack
            start_gen = seat.general_army
        out.append(
            {
                "start_tick": first.tick,
                "end_tick": run[-1].tick,
                "length": len(run),
                "start_army": start_army,
                "end_army": run[-1].army,
                "start_gen_army": start_gen,
                "start_pos": list(first.pos) if first.pos else None,
                "end_pos": list(run[-1].pos) if run[-1].pos else None,
            }
        )
        run.clear()

    for step in steps:
        if step.kind == "move":
            run.append(step)
        else:
            flush()
    flush()
    return out


def _wave_start_pos(steps: list[StackStep], start_tick: int) -> Cell | None:
    """Position of max stack at the tick before the gather wave, else at start."""
    prev = None
    for step in steps:
        if step.tick == start_tick - 1 and step.pos is not None:
            return step.pos
        if step.tick == start_tick:
            prev = step.pos
            break
    return prev


def analyze_one(replay: Replay, an: Analysis, acc: Accumulators, *, is_loss: bool) -> dict:
    us = an.us
    home = replay.generals[us]
    enemy_gen = replay.enemy_general(us)
    contact = an.events.first_contact
    sight = an.vision.first_general_sight[us]
    # events.castles maps all confirmed castles; ownership checked per tick
    our_castles = set(an.events.castles.keys())

    steps = an.steps[us]
    metrics = an.metrics
    last_tick = metrics[-1].tick

    # --- accumulation time series ---
    gen_fracs: list[float] = []
    tip_fracs: list[float] = []
    castle_fracs: list[float] = []
    for entry in metrics:
        seat = entry.seats[us]
        total = seat.army
        if total <= 0:
            continue
        gen_fracs.append(seat.general_army / total)
        tip_fracs.append(seat.max_stack / total)
        castle_fracs.append(_castle_army(replay, entry.tick, us, our_castles) / total)

    max_gen = max((e.seats[us].general_army for e in metrics), default=0)
    row: dict[str, Any] = {
        "match_id": str(replay.match_id),
        "outcome": replay.outcome,
        "opponent": replay.players[1 - us],
        "ticks": last_tick + 1,
        "seat": us,
        "first_contact": contact,
        "first_sight": sight,
        "max_gen_army": max_gen,
        "mean_gen_frac": sum(gen_fracs) / len(gen_fracs) if gen_fracs else None,
        "mean_tip_frac": sum(tip_fracs) / len(tip_fracs) if tip_fracs else None,
        "mean_castle_frac": sum(castle_fracs) / len(castle_fracs) if castle_fracs else None,
        "peak_tip_frac": max(tip_fracs) if tip_fracs else None,
        "peak_gen_frac": max(gen_fracs) if gen_fracs else None,
    }

    if contact is not None and contact < len(metrics):
        seat = metrics[contact].seats[us]
        row["gen_at_contact"] = seat.general_army
        row["tip_at_contact"] = seat.max_stack
    else:
        row["gen_at_contact"] = None
        row["tip_at_contact"] = None

    if sight is not None and sight < len(metrics):
        seat = metrics[sight].seats[us]
        row["gen_at_sight"] = seat.general_army
        row["tip_at_sight"] = seat.max_stack
        row["tip_frac_at_sight"] = seat.max_stack / seat.army if seat.army else None
        row["gen_frac_at_sight"] = seat.general_army / seat.army if seat.army else None
    else:
        row["gen_at_sight"] = None
        row["tip_at_sight"] = None
        row["tip_frac_at_sight"] = None
        row["gen_frac_at_sight"] = None

    # tip near kill / at end
    near_start = max(0, last_tick - NEAR_KILL_WINDOW)
    near_peak = 0
    near_frac_peak = 0.0
    for entry in metrics:
        if entry.tick < near_start:
            continue
        seat = entry.seats[us]
        near_peak = max(near_peak, seat.max_stack)
        if seat.army > 0:
            near_frac_peak = max(near_frac_peak, seat.max_stack / seat.army)
    end_seat = metrics[-1].seats[us]
    row["tip_near_kill"] = near_peak
    row["tip_at_kill"] = end_seat.max_stack
    row["tip_frac_near_kill"] = near_frac_peak

    # --- gather waves ---
    waves = an.events.of_kind("gather_wave", player=us)
    pre = post = 0
    wave_details = []
    for w in waves:
        data = w.data or {}
        start = int(data.get("start", w.tick))
        end = int(data.get("end", w.tick))
        length = end - start + 1
        from_army = int(data.get("from_army", 0))
        to_army = int(data.get("to_army", 0))
        start_pos = _wave_start_pos(steps, start)
        end_pos = tuple(w.cell) if w.cell else None
        is_post = sight is not None and start >= sight
        if is_post:
            post += 1
        else:
            pre += 1
        detail = {
            "start": start,
            "end": end,
            "length": length,
            "from_army": from_army,
            "to_army": to_army,
            "start_pos": list(start_pos) if start_pos else None,
            "end_pos": list(end_pos) if end_pos else None,
            "pre_or_post_sight": "post" if is_post else "pre",
            "start_dist_home": manhattan(start_pos, home) if start_pos else None,
            "end_dist_home": manhattan(end_pos, home) if end_pos else None,
            "start_dist_enemy_gen": manhattan(start_pos, enemy_gen) if start_pos else None,
            "end_dist_enemy_gen": manhattan(end_pos, enemy_gen) if end_pos else None,
        }
        wave_details.append(detail)
        if not is_loss:
            acc.wave_lengths.append(float(length))
            acc.wave_from_army.append(float(from_army))
            acc.wave_to_army.append(float(to_army))
            if detail["start_dist_home"] is not None:
                acc.wave_start_dist_home.append(float(detail["start_dist_home"]))
            if detail["end_dist_home"] is not None:
                acc.wave_end_dist_home.append(float(detail["end_dist_home"]))
            if detail["start_dist_enemy_gen"] is not None:
                acc.wave_start_dist_enemy_gen.append(float(detail["start_dist_enemy_gen"]))
            if detail["end_dist_enemy_gen"] is not None:
                acc.wave_end_dist_enemy_gen.append(float(detail["end_dist_enemy_gen"]))
            if is_post:
                acc.wave_post_sight_n += 1
            else:
                acc.wave_pre_sight_n += 1

    row["gather_waves_total"] = len(waves)
    row["gather_waves_pre_sight"] = pre
    row["gather_waves_post_sight"] = post
    row["waves"] = wave_details

    # --- sustained move streaks ---
    streaks = _sustained_move_streaks(steps, metrics, us)
    row["sustained_move_streaks"] = len(streaks)
    row["streak_start_armies"] = [s["start_army"] for s in streaks]
    if not is_loss:
        for s in streaks:
            acc.streak_start_army.append(float(s["start_army"]))
            acc.streak_lengths.append(float(s["length"]))
            if s["start_gen_army"] is not None:
                acc.streak_start_gen_army.append(float(s["start_gen_army"]))

    # --- half vs full moves ---
    castle_cells = set(an.events.castles.keys())
    moves = infer_moves_for_player(replay, us, castle_cells)
    kind_counts: Counter = Counter(m.kind for m in moves)
    phase_counts: dict[str, Counter] = {p: Counter() for p in PHASE_NAMES}
    for m in moves:
        phase = _army_phase(m.tick, contact, sight)
        phase_counts[phase][m.kind] += 1
    row["move_kinds"] = {k: v for k, v in sorted(kind_counts.items())}
    row["move_kinds_by_phase"] = {
        p: {k: v for k, v in sorted(c.items())} for p, c in phase_counts.items()
    }

    # actionable = full + half (ignore pass/build/unknown/multi for ratio)
    full = kind_counts.get("full", 0)
    half = kind_counts.get("half", 0)
    amb = kind_counts.get("unknown", 0) + kind_counts.get("multi", 0)
    total_moves = sum(kind_counts.values()) or 1
    row["full_rate"] = full / total_moves
    row["half_rate"] = half / total_moves
    row["ambiguity_rate"] = amb / total_moves
    actionable = full + half
    row["half_of_actionable"] = (half / actionable) if actionable else None

    if is_loss:
        acc.loss_rows.append(row)
        return row

    # --- fit accumulators ---
    acc.games_analyzed += 1
    acc.max_gen_army.append(float(max_gen))
    if row["mean_gen_frac"] is not None:
        acc.mean_gen_frac.append(row["mean_gen_frac"])
    if row["mean_tip_frac"] is not None:
        acc.mean_tip_frac.append(row["mean_tip_frac"])
    if row["mean_castle_frac"] is not None:
        acc.mean_castle_frac.append(row["mean_castle_frac"])
    if row["peak_tip_frac"] is not None:
        acc.peak_tip_frac.append(row["peak_tip_frac"])
    if row["peak_gen_frac"] is not None:
        acc.peak_gen_frac.append(row["peak_gen_frac"])

    if row["gen_at_contact"] is not None:
        acc.games_with_contact += 1
        acc.gen_at_contact.append(float(row["gen_at_contact"]))
        acc.tip_at_contact.append(float(row["tip_at_contact"]))
    if row["gen_at_sight"] is not None:
        acc.games_with_sight += 1
        acc.gen_at_sight.append(float(row["gen_at_sight"]))
        acc.tip_at_sight.append(float(row["tip_at_sight"]))
        if row["tip_frac_at_sight"] is not None:
            acc.tip_frac_at_sight.append(row["tip_frac_at_sight"])
        if row["gen_frac_at_sight"] is not None:
            acc.gen_frac_at_sight.append(row["gen_frac_at_sight"])
    acc.tip_near_kill.append(float(row["tip_near_kill"]))
    acc.tip_at_kill.append(float(row["tip_at_kill"]))
    if row["tip_frac_near_kill"] is not None:
        acc.tip_frac_near_kill.append(row["tip_frac_near_kill"])

    acc.waves_per_game.append(float(len(waves)))
    acc.waves_pre_sight.append(float(pre))
    acc.waves_post_sight.append(float(post))
    if len(waves) == 0:
        acc.games_with_zero_waves += 1
        acc.zero_wave_wins.append(str(replay.match_id))

    acc.move_kind_counts.update(kind_counts)
    for p, c in phase_counts.items():
        acc.move_kind_by_phase[p].update(c)
        for k in ("full", "half"):
            if c.get(k):
                acc.actionable_by_phase[p][k] += c[k]

    # counterexamples: high gen bank + low tip at sight
    if (
        row["gen_at_sight"] is not None
        and row["tip_at_sight"] is not None
        and row["gen_at_sight"] >= 40
        and row["tip_at_sight"] < row["gen_at_sight"]
    ):
        acc.high_gen_low_tip_at_sight.append(
            {
                "match_id": str(replay.match_id),
                "gen_at_sight": row["gen_at_sight"],
                "tip_at_sight": row["tip_at_sight"],
            }
        )
    if row["half_of_actionable"] is not None and row["half_of_actionable"] >= 0.25:
        acc.half_heavy_games.append(
            {
                "match_id": str(replay.match_id),
                "half_of_actionable": row["half_of_actionable"],
                "half": half,
                "full": full,
            }
        )

    return row


def _rates_from_counter(c: Counter) -> dict:
    n = sum(c.values()) or 1
    return {k: {"count": int(v), "rate": v / n} for k, v in sorted(c.items())}


def _phase_move_summary(acc: Accumulators) -> dict:
    out = {}
    for phase in PHASE_NAMES:
        rates = _rates_from_counter(acc.move_kind_by_phase[phase])
        act = acc.actionable_by_phase[phase]
        full = act.get("full", 0)
        half = act.get("half", 0)
        actionable = full + half
        amb = (
            acc.move_kind_by_phase[phase].get("unknown", 0)
            + acc.move_kind_by_phase[phase].get("multi", 0)
        )
        total = sum(acc.move_kind_by_phase[phase].values()) or 1
        out[phase] = {
            "kinds": rates,
            "full_of_actionable": (full / actionable) if actionable else None,
            "half_of_actionable": (half / actionable) if actionable else None,
            "ambiguity_rate": amb / total,
            "n_ticks": total,
        }
    return out


def build_report(acc: Accumulators, per_game: list[dict]) -> dict:
    meta = corpus_meta()
    total_kinds = sum(acc.move_kind_counts.values()) or 1
    full = acc.move_kind_counts.get("full", 0)
    half = acc.move_kind_counts.get("half", 0)
    amb = acc.move_kind_counts.get("unknown", 0) + acc.move_kind_counts.get("multi", 0)
    actionable = full + half

    # Loss mismanagement signals
    loss_signals = []
    for lr in acc.loss_rows:
        signals = []
        if lr["gather_waves_total"] == 0:
            signals.append("zero_gather_waves")
        if lr.get("tip_at_sight") is not None and lr.get("mean_tip_frac") is not None:
            if lr["tip_at_sight"] < 15:
                signals.append("tiny_tip_at_sight")
        if lr.get("max_gen_army", 0) >= 50 and (lr.get("tip_near_kill") or 0) < 30:
            signals.append("army_parked_at_general")
        if lr.get("first_sight") is None:
            signals.append("never_saw_enemy_general")
        if (lr.get("half_of_actionable") or 0) >= 0.3:
            signals.append("half_move_heavy")
        loss_signals.append(
            {
                "match_id": lr["match_id"],
                "opponent": lr["opponent"],
                "ticks": lr["ticks"],
                "max_gen_army": lr["max_gen_army"],
                "gen_at_contact": lr.get("gen_at_contact"),
                "gen_at_sight": lr.get("gen_at_sight"),
                "tip_at_sight": lr.get("tip_at_sight"),
                "tip_near_kill": lr.get("tip_near_kill"),
                "gather_waves_total": lr["gather_waves_total"],
                "half_of_actionable": lr.get("half_of_actionable"),
                "signals": signals,
            }
        )

    rules = [
        {
            "id": "R1_tip_not_general_bank",
            "tag": "MEASURED",
            "claim": (
                "Tip stack holds more army than the general bank across the game "
                "(median mean_tip_frac ~2x mean_gen_frac). Most army still sits on "
                "ordinary land; castles hold a small extra bank."
            ),
            "thresholds": {
                "mean_tip_frac_median": None,  # filled below
                "mean_gen_frac_median": None,
            },
        },
        {
            "id": "R2_gather_before_sight",
            "tag": "MEASURED",
            "claim": (
                f"Every fit win has >=1 gather_wave (max-stack grows while moving "
                f">= {GATHER_MIN_TICKS} ticks). Most waves start before first_general_sight; "
                "waves begin near home and end closer to the enemy general."
            ),
            "thresholds": {
                "gather_min_ticks": GATHER_MIN_TICKS,
                "unit": "ticks of consecutive growing moves",
            },
        },
        {
            "id": "R3_sustained_move_start_size",
            "tag": "MEASURED",
            "claim": (
                f"Sustained tip marches (>= {SUSTAINED_MOVE_MIN} consecutive max-stack moves) "
                "typically start with tip army in the mid-teens (p25~9, p75~25)."
            ),
            "thresholds": {
                "sustained_move_min_ticks": SUSTAINED_MOVE_MIN,
                "unit": "consecutive max-stack move ticks",
            },
        },
        {
            "id": "R4_full_over_half",
            "tag": "INFERRED",
            "claim": (
                "Among classifiable sends, leave-1 (full) dominates (~98.5%); half is ~1.5%. "
                "Half is slightly more common in expansion than contest/post_sight. "
                "Ambiguity (unknown+multi) is ~32% of all ticks — treat half/full rates "
                "as conditional on successful classification."
            ),
            "thresholds": {
                "note": "Rates conditioned on inferred full|half; ambiguity (unknown+multi) is high.",
            },
        },
        {
            "id": "R5_home_reserve_modest",
            "tag": "MEASURED",
            "claim": (
                "General peak bank median ~27 army. At contact median gen~8; at sight "
                "median gen~14 — home is not the strike reservoir."
            ),
            "thresholds": {"units": "army on general cell"},
        },
        {
            "id": "R6_tip_mass_at_sight_and_kill",
            "tag": "MEASURED",
            "claim": (
                "Tip at first_general_sight median ~23 army; peak tip in the last "
                f"{NEAR_KILL_WINDOW} ticks median ~53 (strike consolidation). "
                "Tip at the final tick is lower (median ~24) after the kill spend."
            ),
            "thresholds": {"near_kill_window_ticks": NEAR_KILL_WINDOW},
        },
    ]

    tip_dist = dist_summary(acc.mean_tip_frac)
    gen_dist = dist_summary(acc.mean_gen_frac)
    rules[0]["thresholds"]["mean_tip_frac_median"] = tip_dist.get("median")
    rules[0]["thresholds"]["mean_gen_frac_median"] = gen_dist.get("median")
    rules[0]["thresholds"]["mean_castle_frac_median"] = dist_summary(
        acc.mean_castle_frac
    ).get("median")

    streak_dist = dist_summary(acc.streak_start_army)
    rules[2]["thresholds"]["streak_start_army_median"] = streak_dist.get("median")
    rules[2]["thresholds"]["streak_start_army_p25"] = streak_dist.get("p25")
    rules[2]["thresholds"]["streak_start_army_p75"] = streak_dist.get("p75")

    rules[1]["thresholds"]["pre_sight_wave_share"] = (
        acc.wave_pre_sight_n / max(acc.wave_pre_sight_n + acc.wave_post_sight_n, 1)
    )
    rules[1]["thresholds"]["waves_per_game_median"] = dist_summary(
        acc.waves_per_game
    ).get("median")
    rules[1]["thresholds"]["zero_wave_wins"] = acc.games_with_zero_waves
    rules[1]["thresholds"]["wave_start_dist_home_median"] = dist_summary(
        acc.wave_start_dist_home
    ).get("median")
    rules[1]["thresholds"]["wave_end_dist_enemy_gen_median"] = dist_summary(
        acc.wave_end_dist_enemy_gen
    ).get("median")

    rules[3]["thresholds"]["full_of_actionable"] = (
        full / actionable if actionable else None
    )
    rules[3]["thresholds"]["half_of_actionable"] = (
        half / actionable if actionable else None
    )
    rules[3]["thresholds"]["ambiguity_rate"] = amb / total_kinds

    rules[4]["thresholds"]["max_gen_army"] = dist_summary(acc.max_gen_army)
    rules[4]["thresholds"]["gen_at_contact"] = dist_summary(acc.gen_at_contact)
    rules[4]["thresholds"]["gen_at_sight"] = dist_summary(acc.gen_at_sight)

    rules[5]["thresholds"]["tip_at_sight"] = dist_summary(acc.tip_at_sight)
    rules[5]["thresholds"]["tip_near_kill"] = dist_summary(acc.tip_near_kill)
    rules[5]["thresholds"]["tip_at_kill"] = dist_summary(acc.tip_at_kill)

    payload = {
        "meta": {
            "player": PLAYER,
            "analyst": 3,
            "topic": "army_management_and_routing",
            "corpus": "fit wins only for rules; losses skimmed",
            "n_fit_wins": len(meta.fit_win_ids),
            "n_fit_analyzed": acc.games_analyzed,
            "n_holdout_wins_excluded": len(meta.holdout_win_ids),
            "n_losses_skimmed": len(acc.loss_rows),
            "split": "docs/research/measurements/grok-kubic-corpus-split.json",
            "thresholds": {
                "sustained_move_min_ticks": SUSTAINED_MOVE_MIN,
                "near_kill_window_ticks": NEAR_KILL_WINDOW,
                "gather_min_ticks": GATHER_MIN_TICKS,
                "phase_buckets": list(PHASE_NAMES),
                "phase_definition": (
                    "expansion: tick < first_contact; "
                    "contest: first_contact <= tick < first_general_sight; "
                    "post_sight: tick >= first_general_sight "
                    "(if no contact, all expansion; if no sight, contest to end)"
                ),
            },
        },
        "rules": rules,
        "distributions": {
            "accumulation": {
                "mean_gen_frac_dist": gen_dist,
                "mean_tip_frac": tip_dist,
                "mean_castle_frac": dist_summary(acc.mean_castle_frac),
                "peak_tip_frac": dist_summary(acc.peak_tip_frac),
                "peak_gen_frac": dist_summary(acc.peak_gen_frac),
                "tip_frac_at_sight": dist_summary(acc.tip_frac_at_sight),
                "gen_frac_at_sight": dist_summary(acc.gen_frac_at_sight),
                "tip_frac_near_kill": dist_summary(acc.tip_frac_near_kill),
            },
            "home_reserve": {
                "max_gen_army": dist_summary(acc.max_gen_army),
                "gen_at_contact": dist_summary(acc.gen_at_contact),
                "gen_at_sight": dist_summary(acc.gen_at_sight),
                "tip_at_contact": dist_summary(acc.tip_at_contact),
            },
            "tip_mass": {
                "tip_at_sight": dist_summary(acc.tip_at_sight),
                "tip_near_kill": dist_summary(acc.tip_near_kill),
                "tip_at_kill": dist_summary(acc.tip_at_kill),
            },
            "gather": {
                "waves_per_game": dist_summary(acc.waves_per_game),
                "waves_pre_sight_per_game": dist_summary(acc.waves_pre_sight),
                "waves_post_sight_per_game": dist_summary(acc.waves_post_sight),
                "games_with_zero_waves": acc.games_with_zero_waves,
                "zero_wave_rate": acc.games_with_zero_waves / max(acc.games_analyzed, 1),
                "wave_lengths_ticks": dist_summary(acc.wave_lengths),
                "wave_from_army": dist_summary(acc.wave_from_army),
                "wave_to_army": dist_summary(acc.wave_to_army),
                "wave_start_dist_home": dist_summary(acc.wave_start_dist_home),
                "wave_end_dist_home": dist_summary(acc.wave_end_dist_home),
                "wave_start_dist_enemy_gen": dist_summary(acc.wave_start_dist_enemy_gen),
                "wave_end_dist_enemy_gen": dist_summary(acc.wave_end_dist_enemy_gen),
                "wave_count_pre_sight": acc.wave_pre_sight_n,
                "wave_count_post_sight": acc.wave_post_sight_n,
                "pre_sight_share": (
                    acc.wave_pre_sight_n / max(acc.wave_pre_sight_n + acc.wave_post_sight_n, 1)
                ),
            },
            "sustained_move_streaks": {
                "start_army": streak_dist,
                "length_ticks": dist_summary(acc.streak_lengths),
                "start_gen_army": dist_summary(acc.streak_start_gen_army),
                "n_streaks": len(acc.streak_start_army),
            },
            "move_modes": {
                "overall_kinds": _rates_from_counter(acc.move_kind_counts),
                "full_of_actionable": full / actionable if actionable else None,
                "half_of_actionable": half / actionable if actionable else None,
                "ambiguity_rate_unknown_plus_multi": amb / total_kinds,
                "by_phase": _phase_move_summary(acc),
            },
        },
        "counterexamples": {
            "high_gen_low_tip_at_sight": acc.high_gen_low_tip_at_sight[:25],
            "high_gen_low_tip_at_sight_n": len(acc.high_gen_low_tip_at_sight),
            "zero_wave_wins_sample": acc.zero_wave_wins[:25],
            "zero_wave_wins_n": len(acc.zero_wave_wins),
            "half_heavy_games": acc.half_heavy_games[:25],
            "half_heavy_games_n": len(acc.half_heavy_games),
        },
        "unknowns": [
            {
                "id": "U1_multi_source_ticks",
                "tag": "UNKNOWN",
                "text": (
                    "Move inference marks many ticks as multi/unknown (simultaneous "
                    "sources or combat). Half/full rates on those ticks are unreliable."
                ),
            },
            {
                "id": "U2_castle_ownership_timing",
                "tag": "UNKNOWN",
                "text": (
                    "Castle army fraction uses all confirmed castle cells; enemy-captured "
                    "or late-built castles blur the owned-castle bank estimate."
                ),
            },
            {
                "id": "U3_gather_wave_definition",
                "tag": "UNKNOWN",
                "text": (
                    "gather_wave requires the largest stack to grow while moving. "
                    "Secondary gathering into a non-max tip is invisible."
                ),
            },
            {
                "id": "U4_routing_policy",
                "tag": "UNKNOWN",
                "text": (
                    "This report measures mass location and send mode, not path choice "
                    "(toward/away is path analysis, not army-routing thresholds)."
                ),
            },
        ],
        "loss_skim": loss_signals,
        "sample_games": per_game[:5],  # tiny sample; full stats in distributions
    }
    return payload


def write_markdown(payload: dict) -> str:
    d = payload["distributions"]
    meta = payload["meta"]
    lines: list[str] = []
    lines.append("# Kubic army management & routing (fit wins)")
    lines.append("")
    lines.append(
        f"Analyst #3. Corpus: **{meta['n_fit_analyzed']}** fit wins "
        f"(of {meta['n_fit_wins']} listed); holdout excluded; "
        f"{meta['n_losses_skimmed']} losses skimmed."
    )
    lines.append("")
    lines.append(
        "Phase buckets: `expansion` / `contest` / `post_sight` "
        "(see JSON `meta.thresholds.phase_definition`)."
    )
    lines.append("")
    lines.append("## Top rules")
    lines.append("")
    for r in payload["rules"]:
        lines.append(f"### {r['id']} [{r['tag']}]")
        lines.append("")
        lines.append(r["claim"])
        lines.append("")
        thr = r.get("thresholds") or {}
        if thr:
            lines.append("Thresholds / numbers:")
            lines.append("")
            lines.append("```")
            lines.append(json.dumps(thr, indent=2, default=str))
            lines.append("```")
            lines.append("")

    lines.append("## Key distributions")
    lines.append("")
    lines.append("### Where army sits (fraction of total army)")
    lines.append("")
    lines.append("| Metric | n | median | p25 | p75 | mean |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")

    def row(name: str, dist: dict) -> None:
        if not dist or dist.get("n", 0) == 0:
            lines.append(f"| {name} | 0 | — | — | — | — |")
            return
        lines.append(
            f"| {name} | {dist['n']} | {dist['median']:.3f} | {dist['p25']:.3f} | "
            f"{dist['p75']:.3f} | {dist['mean']:.3f} |"
        )

    accu = d["accumulation"]
    row("mean_tip_frac", accu["mean_tip_frac"])
    row("mean_gen_frac", accu["mean_gen_frac_dist"])
    row("mean_castle_frac", accu["mean_castle_frac"])
    row("tip_frac_at_sight", accu["tip_frac_at_sight"])
    row("gen_frac_at_sight", accu["gen_frac_at_sight"])
    row("tip_frac_near_kill", accu["tip_frac_near_kill"])
    lines.append("")

    lines.append("### Home reserve (army units on general)")
    lines.append("")
    lines.append("| Metric | n | median | p25 | p75 | mean | max |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")

    def row_abs(name: str, dist: dict) -> None:
        if not dist or dist.get("n", 0) == 0:
            lines.append(f"| {name} | 0 | — | — | — | — | — |")
            return
        lines.append(
            f"| {name} | {dist['n']} | {dist['median']:.1f} | {dist['p25']:.1f} | "
            f"{dist['p75']:.1f} | {dist['mean']:.1f} | {dist['max']:.1f} |"
        )

    hr = d["home_reserve"]
    row_abs("max_gen_army", hr["max_gen_army"])
    row_abs("gen_at_contact", hr["gen_at_contact"])
    row_abs("gen_at_sight", hr["gen_at_sight"])
    row_abs("tip_at_contact", hr["tip_at_contact"])
    lines.append("")

    lines.append("### Tip mass at sight / near kill (army units)")
    lines.append("")
    lines.append("| Metric | n | median | p25 | p75 | mean |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    tm = d["tip_mass"]
    row_abs2 = row  # fractions style with .3f — use abs
    for name, key in (
        ("tip_at_sight", "tip_at_sight"),
        ("tip_near_kill", "tip_near_kill"),
        ("tip_at_kill", "tip_at_kill"),
    ):
        dist = tm[key]
        if not dist or dist.get("n", 0) == 0:
            lines.append(f"| {name} | 0 | — | — | — | — |")
        else:
            lines.append(
                f"| {name} | {dist['n']} | {dist['median']:.1f} | {dist['p25']:.1f} | "
                f"{dist['p75']:.1f} | {dist['mean']:.1f} |"
            )
    lines.append("")

    g = d["gather"]
    lines.append("### Gather waves")
    lines.append("")
    lines.append(
        f"- Waves per game: median **{g['waves_per_game'].get('median')}** "
        f"(mean {g['waves_per_game'].get('mean'):.2f}), "
        f"zero-wave wins: **{g['games_with_zero_waves']}** "
        f"({100 * g['zero_wave_rate']:.1f}%)."
    )
    lines.append(
        f"- Pre-sight wave count share: **{100 * g['pre_sight_share']:.1f}%** "
        f"({g['wave_count_pre_sight']} pre / {g['wave_count_post_sight']} post)."
    )
    wl = g["wave_lengths_ticks"]
    if wl.get("n"):
        lines.append(
            f"- Wave length: median **{wl['median']:.0f}** ticks "
            f"(p25={wl['p25']:.0f}, p75={wl['p75']:.0f})."
        )
    fa = g["wave_from_army"]
    ta = g["wave_to_army"]
    if fa.get("n"):
        lines.append(
            f"- Wave army: from median **{fa['median']:.0f}** → to median **{ta['median']:.0f}**."
        )
    lines.append(
        f"- Start dist to home (Manhattan): median "
        f"**{g['wave_start_dist_home'].get('median')}**; "
        f"end dist to enemy general: median "
        f"**{g['wave_end_dist_enemy_gen'].get('median')}**."
    )
    lines.append("")

    sm = d["sustained_move_streaks"]
    lines.append(
        f"### Sustained tip marches (>= {meta['thresholds']['sustained_move_min_ticks']} moves)"
    )
    lines.append("")
    sa = sm["start_army"]
    if sa.get("n"):
        lines.append(
            f"- n={sa['n']}; start army median **{sa['median']:.0f}** "
            f"(p25={sa['p25']:.0f}, p75={sa['p75']:.0f}, mean={sa['mean']:.1f})."
        )
        lines.append(
            f"- Streak length median **{sm['length_ticks'].get('median'):.0f}** ticks."
        )
    lines.append("")

    mm = d["move_modes"]
    lines.append("### Half vs full sends")
    lines.append("")
    lines.append(
        f"- Of classifiable sends (full|half): "
        f"**{100 * (mm['full_of_actionable'] or 0):.1f}%** full, "
        f"**{100 * (mm['half_of_actionable'] or 0):.1f}%** half."
    )
    lines.append(
        f"- Ambiguity rate (unknown+multi over all ticks): "
        f"**{100 * mm['ambiguity_rate_unknown_plus_multi']:.1f}%**."
    )
    lines.append("")
    lines.append("| Phase | n_ticks | full/(full+half) | half/(full+half) | ambiguity |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for phase, block in mm["by_phase"].items():
        fa_ = block["full_of_actionable"]
        ha_ = block["half_of_actionable"]
        lines.append(
            f"| {phase} | {block['n_ticks']} | "
            f"{'—' if fa_ is None else f'{100 * fa_:.1f}%'} | "
            f"{'—' if ha_ is None else f'{100 * ha_:.1f}%'} | "
            f"{100 * block['ambiguity_rate']:.1f}% |"
        )
    lines.append("")
    lines.append("Overall kind rates:")
    lines.append("")
    lines.append("```")
    lines.append(json.dumps(mm["overall_kinds"], indent=2))
    lines.append("```")
    lines.append("")

    lines.append("## Counterexamples")
    lines.append("")
    cx = payload["counterexamples"]
    lines.append(
        f"- High gen (>=40) with tip < gen at sight: **{cx['high_gen_low_tip_at_sight_n']}** "
        f"games (sample ids: "
        f"{', '.join(x['match_id'] for x in cx['high_gen_low_tip_at_sight'][:8]) or 'none'})."
    )
    lines.append(
        f"- Zero gather-wave wins: **{cx['zero_wave_wins_n']}** "
        f"(sample: {', '.join(cx['zero_wave_wins_sample'][:8]) or 'none'})."
    )
    lines.append(
        f"- Half-heavy (>=25% of actionable): **{cx['half_heavy_games_n']}** games."
    )
    lines.append("")

    lines.append("## Loss skim (army mismanagement signals)")
    lines.append("")
    lines.append("| match | opp | ticks | max_gen | tip@sight | tip@near_kill | waves | signals |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for lr in payload["loss_skim"]:
        lines.append(
            f"| {lr['match_id']} | {lr['opponent']} | {lr['ticks']} | "
            f"{lr['max_gen_army']} | {lr['tip_at_sight']} | {lr['tip_near_kill']} | "
            f"{lr['gather_waves_total']} | {', '.join(lr['signals']) or '—'} |"
        )
    lines.append("")

    lines.append("## Unknowns")
    lines.append("")
    for u in payload["unknowns"]:
        lines.append(f"- **{u['id']}** [{u['tag']}]: {u['text']}")
    lines.append("")

    lines.append("## Paths")
    lines.append("")
    lines.append(f"- Script: `scripts/analyze_kubic_army.py`")
    lines.append(f"- JSON: `docs/research/measurements/{JSON_OUT}`")
    lines.append(f"- Markdown: `docs/research/measurements/{MD_OUT}`")
    lines.append(f"- Corpus split: `docs/research/measurements/grok-kubic-corpus-split.json`")
    lines.append(f"- Helpers: `scripts/kubic_corpus.py`, `scripts/kubic_moves.py`")
    lines.append(
        "- Replay APIs: `arena/instrument/replay/` "
        "(metrics.max_stack/general_army, path.StackStep, events.gather_wave)"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    acc = Accumulators()
    per_game: list[dict] = []
    print("Analyzing fit wins...", flush=True)
    for i, (replay, an) in enumerate(iter_analyzed("fit"), 1):
        row = analyze_one(replay, an, acc, is_loss=False)
        if i <= 20:
            per_game.append(
                {
                    k: row[k]
                    for k in (
                        "match_id",
                        "ticks",
                        "max_gen_army",
                        "gen_at_contact",
                        "gen_at_sight",
                        "tip_at_sight",
                        "tip_near_kill",
                        "gather_waves_total",
                        "gather_waves_pre_sight",
                        "gather_waves_post_sight",
                        "sustained_move_streaks",
                        "full_rate",
                        "half_rate",
                        "ambiguity_rate",
                        "mean_tip_frac",
                        "mean_gen_frac",
                    )
                    if k in row
                }
            )
        if i % 50 == 0:
            print(f"  fit {i}/{acc.games_analyzed}...", flush=True)

    print("Skimming losses...", flush=True)
    for replay, an in iter_analyzed("losses"):
        analyze_one(replay, an, acc, is_loss=True)

    payload = build_report(acc, per_game)
    json_path = dump_json(JSON_OUT, payload)
    md = write_markdown(payload)
    md_path = MEASUREMENTS / MD_OUT
    md_path.write_text(md)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(
        json.dumps(
            {
                "n_fit": acc.games_analyzed,
                "n_loss": len(acc.loss_rows),
                "mean_tip_frac_median": payload["distributions"]["accumulation"][
                    "mean_tip_frac"
                ].get("median"),
                "half_of_actionable": payload["distributions"]["move_modes"][
                    "half_of_actionable"
                ],
                "ambiguity": payload["distributions"]["move_modes"][
                    "ambiguity_rate_unknown_plus_multi"
                ],
                "waves_median": payload["distributions"]["gather"]["waves_per_game"].get(
                    "median"
                ),
                "streak_start_median": payload["distributions"]["sustained_move_streaks"][
                    "start_army"
                ].get("median"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
