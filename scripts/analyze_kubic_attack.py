#!/usr/bin/env python3
"""Measure Kubic ATTACKING behavior on the fit win set (plus loss skim).

Writes:
  docs/research/measurements/grok-kubic-attack.json
  docs/research/measurements/grok-kubic-attack.md

Uses scripts/kubic_corpus.py. Observational only — never writes data/games/.
"""

from __future__ import annotations

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
from arena.instrument.replay.path import summarize_path
from scripts.kubic_corpus import (
    MEASUREMENTS,
    corpus_meta,
    dist_summary,
    dump_json,
    iter_analyzed,
    percentile,
)
from scripts.kubic_moves import infer_moves_for_player

ORTH = ((-1, 0), (1, 0), (0, -1), (0, 1))
FRONT_WINDOW = 30  # ticks after first_contact for front behavior
NO_ATTACK_CAPTURE_GAP = 40  # contact without our capture for this long
COMMIT_SAMPLE_CAP = 40  # capture commit ratios per game (avoid domination games)


@dataclass
class GameAttack:
    match_id: str
    outcome: str
    ticks: int
    us: int
    them: int
    # engagement
    first_contact: int | None = None
    first_capture_us: int | None = None
    contact_to_capture: int | None = None
    army_ratio_contact: float | None = None
    tile_ratio_contact: float | None = None
    stack_at_contact: int | None = None
    gen_army_at_contact: int | None = None
    army_ratio_capture: float | None = None
    tile_ratio_capture: float | None = None
    stack_at_capture: int | None = None
    # commit ratios (own sent / enemy held) for successful enemy-tile captures
    commit_ratios: list[float] = field(default_factory=list)
    commit_stack_vs_held: list[float] = field(default_factory=list)
    commit_held: list[int] = field(default_factory=list)
    commit_surplus: list[int] = field(default_factory=list)  # sent - held
    first_capture_sent_over_held: float | None = None
    first_capture_held: int | None = None
    n_enemy_captures: int = 0
    n_big_captures: int = 0
    # castles
    own_castles_built: int = 0
    first_own_castle: int | None = None
    enemy_castles_captured: int = 0
    first_enemy_castle_capture: int | None = None
    own_castles_before_enemy_capture: int | None = None
    # front behavior class
    front_class: str | None = None  # push | trade | gather_then_push | passive | no_contact
    front_net_tiles: int | None = None
    front_our_gains: int | None = None
    front_our_losses: int | None = None
    gather_before_push: bool | None = None
    # sight / kill
    first_sight: int | None = None
    kill_tick: int | None = None
    sight_to_kill: int | None = None
    stack_at_sight: int | None = None
    army_ratio_sight: float | None = None
    toward_after_sight: float | None = None
    toward_moves_after_sight: int | None = None
    # gather vs attack
    gather_waves: int = 0
    gather_completed_between_contact_and_capture: bool | None = None
    gather_open_at_contact: bool | None = None
    capture_after_open_gather_ends: bool | None = None
    gather_open_at_sight: bool | None = None
    strike_after_open_gather_ends: bool | None = None
    ticks_gather_end_to_first_capture: int | None = None
    # no-attack-despite-contact
    contact_no_capture_long: bool = False
    delayed_capture: bool = False
    # loss skim helpers
    notes: list[str] = field(default_factory=list)

    def as_json(self) -> dict:
        skip = {"commit_ratios", "commit_stack_vs_held", "commit_held", "commit_surplus", "notes"}
        d = {k: v for k, v in self.__dict__.items() if k not in skip}
        d["commit_ratios_n"] = len(self.commit_ratios)
        d["commit_ratio_median"] = percentile(self.commit_ratios, 50) if self.commit_ratios else None
        d["commit_stack_vs_held_median"] = (
            percentile(self.commit_stack_vs_held, 50) if self.commit_stack_vs_held else None
        )
        d["notes"] = self.notes
        return d


def _ratio(a: float, b: float) -> float | None:
    if b <= 0:
        return None
    return a / b


def _seat_at(an: Analysis, tick: int, seat: int):
    if tick is None or tick < 0 or tick >= len(an.metrics):
        return None
    return an.metrics[tick].seats[seat]


def _engagement(an: Analysis, tick: int | None) -> tuple[float | None, float | None, int | None, int | None]:
    if tick is None:
        return None, None, None, None
    us_s = _seat_at(an, tick, an.us)
    them_s = _seat_at(an, tick, an.them)
    if us_s is None or them_s is None:
        return None, None, None, None
    return (
        _ratio(us_s.army, them_s.army),
        _ratio(us_s.tiles, them_s.tiles),
        us_s.max_stack,
        us_s.general_army,
    )


def _capture_commits(
    replay: Replay, an: Analysis, castles: set[Cell]
) -> tuple[list[float], list[float], list[int], list[int], float | None, int | None, int]:
    """For each successful capture of an enemy cell by us, estimate sent/held and stack/held."""
    us, them = an.us, an.them
    moves = infer_moves_for_player(replay, us, castles=castles)
    by_tick = {m.tick: m for m in moves}
    ratios: list[float] = []
    stack_ratios: list[float] = []
    helds: list[int] = []
    surplus: list[int] = []
    first_ratio: float | None = None
    first_held: int | None = None
    n = 0
    for t in range(1, len(replay.ticks)):
        before, after = replay.ticks[t - 1], replay.ticks[t]
        for r in range(replay.rows):
            for c in range(replay.cols):
                if before.owners[r][c] != them or after.owners[r][c] != us:
                    continue
                n += 1
                held = before.armies[r][c]
                mv = by_tick.get(t - 1)
                sent = None
                if mv is not None and mv.dst == (r, c) and mv.sent is not None and mv.sent > 0:
                    sent = mv.sent
                else:
                    best = 0
                    for dr, dc in ORTH:
                        rr, cc = r + dr, c + dc
                        if not (0 <= rr < replay.rows and 0 <= cc < replay.cols):
                            continue
                        if before.owners[rr][cc] != us:
                            continue
                        drop = before.armies[rr][cc] - (
                            after.armies[rr][cc] if after.owners[rr][cc] == us else 0
                        )
                        if drop > best:
                            best = drop
                    sent = best if best > 0 else None
                stack = an.metrics[t - 1].seats[us].max_stack
                if held > 0 and sent is not None and sent > 0:
                    ratio = sent / held
                    if first_ratio is None:
                        first_ratio = ratio
                        first_held = held
                    if len(ratios) < COMMIT_SAMPLE_CAP:
                        ratios.append(ratio)
                        helds.append(held)
                        surplus.append(sent - held)
                if held > 0 and stack > 0 and len(stack_ratios) < COMMIT_SAMPLE_CAP:
                    stack_ratios.append(stack / held)
    return ratios, stack_ratios, helds, surplus, first_ratio, first_held, n


def _front_behavior(an: Analysis, contact: int | None) -> tuple[str, int, int, int, bool | None]:
    """Classify early-contact window: push / trade / gather_then_push / passive / no_contact."""
    if contact is None:
        return "no_contact", 0, 0, 0, None
    end = min(contact + FRONT_WINDOW, an.metrics[-1].tick)
    us = an.us
    gains = losses = 0
    for t in range(contact + 1, end + 1):
        seat = an.metrics[t].seats[us]
        gains += max(0, seat.tiles_gained)
        losses += max(0, seat.tiles_lost)
    # tiles_gained/lost are per-tick deltas already in metrics
    # Recompute from consecutive tile counts for robustness
    start_tiles = an.metrics[contact].seats[us].tiles
    end_tiles = an.metrics[end].seats[us].tiles
    net = end_tiles - start_tiles
    # gather then push: a gather_wave ends in [contact-10, contact+FRONT_WINDOW]
    # and net tiles rise in the second half of the window more than first half
    waves = an.events.of_kind("gather_wave", player=us)
    gather_end_in_window = False
    for w in waves:
        w_end = int(w.data.get("end", w.tick))
        if contact - 10 <= w_end <= end:
            gather_end_in_window = True
            break
    mid = contact + FRONT_WINDOW // 2
    mid = min(mid, end)
    first_half_net = an.metrics[mid].seats[us].tiles - start_tiles
    second_half_net = end_tiles - an.metrics[mid].seats[us].tiles
    gather_then = False
    if gather_end_in_window and second_half_net >= 3 and second_half_net > first_half_net + 1:
        gather_then = True

    # Recalc gains/losses from ownership flips in window
    g = l = 0
    for t in range(contact + 1, end + 1):
        g += an.metrics[t].seats[us].tiles_gained
        l += an.metrics[t].seats[us].tiles_lost

    if gather_then:
        cls = "gather_then_push"
    elif g >= 5 and g > l * 1.5:
        cls = "push"
    elif g >= 3 and l >= 3 and abs(g - l) <= max(2, 0.35 * max(g, l)):
        cls = "trade"
    elif g <= 1 and l <= 1 and abs(net) <= 1:
        cls = "passive"
    elif net > 0 and g > l:
        cls = "push"
    elif l > g:
        cls = "trade"
    else:
        cls = "passive"
    return cls, net, g, l, gather_then if gather_end_in_window else False


def _wave_spans(an: Analysis) -> list[tuple[int, int]]:
    us = an.us
    out: list[tuple[int, int]] = []
    for w in an.events.of_kind("gather_wave", player=us):
        start = int(w.data.get("start", w.tick))
        end = int(w.data.get("end", w.tick))
        out.append((start, end))
    return out


def _gather_attack_link(
    an: Analysis,
    contact: int | None,
    capture: int | None,
    sight: int | None,
) -> dict[str, Any]:
    """Non-tautological gather vs attack ordering.

    Questions that can fail:
    - Was a gather_wave open at first_contact, and did first_capture wait until it ended?
    - Did any gather complete between contact and first_capture?
    - Was a gather open at first_general_sight, and did the first toward-general
      move wait until that wave ended?
    """
    waves = _wave_spans(an)
    result: dict[str, Any] = {
        "gather_completed_between_contact_and_capture": None,
        "gather_open_at_contact": None,
        "capture_after_open_gather_ends": None,
        "gather_open_at_sight": None,
        "strike_after_open_gather_ends": None,
        "ticks_gather_end_to_first_capture": None,
    }
    if contact is not None and capture is not None and capture >= contact:
        result["gather_completed_between_contact_and_capture"] = any(
            contact <= end <= capture for _, end in waves
        )
        open_at = [(s, e) for s, e in waves if s <= contact <= e]
        result["gather_open_at_contact"] = bool(open_at)
        if open_at:
            end = max(e for _, e in open_at)
            result["capture_after_open_gather_ends"] = capture >= end
            result["ticks_gather_end_to_first_capture"] = capture - end

    if sight is not None:
        open_sight = [(s, e) for s, e in waves if s <= sight <= e]
        result["gather_open_at_sight"] = bool(open_sight)
        if open_sight:
            end = max(e for _, e in open_sight)
            first_toward = None
            for step in an.steps[an.us]:
                if step.tick < sight:
                    continue
                if step.toward and step.target_kind == "enemy_general":
                    first_toward = step.tick
                    break
            if first_toward is not None:
                result["strike_after_open_gather_ends"] = first_toward >= end
    return result


def analyze_game(replay: Replay, an: Analysis) -> GameAttack:
    us, them = an.us, an.them
    g = GameAttack(
        match_id=str(replay.match_id),
        outcome=replay.outcome,
        ticks=replay.total_ticks,
        us=us,
        them=them,
    )
    contact = an.events.first_contact
    g.first_contact = contact
    g.first_capture_us = an.events.first_capture[us]
    if contact is not None and g.first_capture_us is not None:
        g.contact_to_capture = g.first_capture_us - contact
        if g.contact_to_capture >= NO_ATTACK_CAPTURE_GAP:
            g.delayed_capture = True
        if g.contact_to_capture < 0:
            # capture before recorded contact — should be rare
            g.notes.append("capture_before_contact")
    if contact is not None and g.first_capture_us is None:
        g.contact_no_capture_long = True
        g.notes.append("contact_never_captured")
    elif contact is not None and g.first_capture_us is not None:
        if g.contact_to_capture is not None and g.contact_to_capture >= NO_ATTACK_CAPTURE_GAP:
            g.contact_no_capture_long = True

    ar, tr, st, ga = _engagement(an, contact)
    g.army_ratio_contact, g.tile_ratio_contact = ar, tr
    g.stack_at_contact, g.gen_army_at_contact = st, ga
    ar, tr, st, _ = _engagement(an, g.first_capture_us)
    g.army_ratio_capture, g.tile_ratio_capture = ar, tr
    g.stack_at_capture = st

    castles = set(an.events.castles.keys())
    ratios, stack_ratios, helds, surplus, first_ratio, first_held, n_cap = _capture_commits(
        replay, an, castles
    )
    g.commit_ratios = ratios
    g.commit_stack_vs_held = stack_ratios
    g.commit_held = helds
    g.commit_surplus = surplus
    g.first_capture_sent_over_held = first_ratio
    g.first_capture_held = first_held
    g.n_enemy_captures = n_cap
    g.n_big_captures = len(an.events.of_kind("big_capture", player=us))

    own_builds = an.events.of_kind("castle_built", player=us)
    g.own_castles_built = len(own_builds)
    g.first_own_castle = own_builds[0].tick if own_builds else None
    enemy_caps = [
        e
        for e in an.events.of_kind("castle_captured", player=us)
        if e.data.get("from") == them
    ]
    g.enemy_castles_captured = len(enemy_caps)
    g.first_enemy_castle_capture = enemy_caps[0].tick if enemy_caps else None
    if g.first_enemy_castle_capture is not None:
        g.own_castles_before_enemy_capture = sum(
            1 for e in own_builds if e.tick < g.first_enemy_castle_capture
        )

    cls, net, gains, losses, gtp = _front_behavior(an, contact)
    g.front_class = cls
    g.front_net_tiles = net
    g.front_our_gains = gains
    g.front_our_losses = losses
    g.gather_before_push = gtp

    sight = an.vision.first_general_sight[us]
    g.first_sight = sight
    kill_events = [
        e for e in an.events.of_kind("general_captured", player=us) if e.data.get("loser") == them
    ]
    g.kill_tick = kill_events[0].tick if kill_events else (replay.total_ticks if replay.outcome == "win" else None)
    if sight is not None and g.kill_tick is not None:
        g.sight_to_kill = g.kill_tick - sight
    if sight is not None:
        seat = _seat_at(an, sight, us)
        them_s = _seat_at(an, sight, them)
        if seat is not None:
            g.stack_at_sight = seat.max_stack
        if seat is not None and them_s is not None:
            g.army_ratio_sight = _ratio(seat.army, them_s.army)
        path = summarize_path(an.steps[us], "post_sight", sight, an.metrics[-1].tick)
        g.toward_after_sight = path.toward_fraction
        g.toward_moves_after_sight = path.toward

    g.gather_waves = len(an.events.of_kind("gather_wave", player=us))
    link = _gather_attack_link(an, contact, g.first_capture_us, sight)
    g.gather_completed_between_contact_and_capture = link[
        "gather_completed_between_contact_and_capture"
    ]
    g.gather_open_at_contact = link["gather_open_at_contact"]
    g.capture_after_open_gather_ends = link["capture_after_open_gather_ends"]
    g.gather_open_at_sight = link["gather_open_at_sight"]
    g.strike_after_open_gather_ends = link["strike_after_open_gather_ends"]
    g.ticks_gather_end_to_first_capture = link["ticks_gather_end_to_first_capture"]

    return g


def _claim(
    tag: str,
    text: str,
    *,
    n: int | None = None,
    detail: Any = None,
) -> dict:
    out: dict[str, Any] = {"tag": tag, "claim": text}
    if n is not None:
        out["n"] = n
    if detail is not None:
        out["detail"] = detail
    return out


def _rate(num: int, den: int) -> float | None:
    return num / den if den else None


def build_report(games: list[GameAttack], losses: list[GameAttack]) -> dict:
    meta = corpus_meta()
    n = len(games)

    army_c = [g.army_ratio_contact for g in games if g.army_ratio_contact is not None]
    tile_c = [g.tile_ratio_contact for g in games if g.tile_ratio_contact is not None]
    stack_c = [g.stack_at_contact for g in games if g.stack_at_contact is not None]
    army_cap = [g.army_ratio_capture for g in games if g.army_ratio_capture is not None]
    tile_cap = [g.tile_ratio_capture for g in games if g.tile_ratio_capture is not None]
    stack_cap = [g.stack_at_capture for g in games if g.stack_at_capture is not None]
    c2c = [g.contact_to_capture for g in games if g.contact_to_capture is not None]

    all_commit = [r for g in games for r in g.commit_ratios]
    all_stack_held = [r for g in games for r in g.commit_stack_vs_held]
    all_held = [h for g in games for h in g.commit_held]
    all_surplus = [s for g in games for s in g.commit_surplus]
    # Stratify commit by held army on the captured cell
    commit_held1 = [r for g in games for r, h in zip(g.commit_ratios, g.commit_held) if h == 1]
    commit_held_2_9 = [
        r for g in games for r, h in zip(g.commit_ratios, g.commit_held) if 2 <= h <= 9
    ]
    commit_held_ge10 = [
        r for g in games for r, h in zip(g.commit_ratios, g.commit_held) if h >= 10
    ]
    surplus_held_ge10 = [
        s for g in games for s, h in zip(g.commit_surplus, g.commit_held) if h >= 10
    ]
    first_cap_ratio = [
        g.first_capture_sent_over_held
        for g in games
        if g.first_capture_sent_over_held is not None
    ]
    first_cap_held = [g.first_capture_held for g in games if g.first_capture_held is not None]

    front_counts = Counter(g.front_class for g in games)
    sight_to_kill = [g.sight_to_kill for g in games if g.sight_to_kill is not None]
    stack_sight = [g.stack_at_sight for g in games if g.stack_at_sight is not None]
    toward = [g.toward_after_sight for g in games if g.toward_after_sight is not None]
    army_sight = [g.army_ratio_sight for g in games if g.army_ratio_sight is not None]

    n_sight = sum(1 for g in games if g.first_sight is not None)
    n_contact = sum(1 for g in games if g.first_contact is not None)
    n_capture = sum(1 for g in games if g.first_capture_us is not None)

    # Gather vs attack (non-tautological)
    open_contact = [g for g in games if g.gather_open_at_contact is True]
    wait_cap_true = sum(1 for g in open_contact if g.capture_after_open_gather_ends is True)
    wait_cap_false = sum(1 for g in open_contact if g.capture_after_open_gather_ends is False)
    gather_between = sum(
        1 for g in games if g.gather_completed_between_contact_and_capture is True
    )
    open_sight = [g for g in games if g.gather_open_at_sight is True]
    wait_strike_true = sum(1 for g in open_sight if g.strike_after_open_gather_ends is True)
    wait_strike_false = sum(1 for g in open_sight if g.strike_after_open_gather_ends is False)
    wait_strike_n = wait_strike_true + wait_strike_false
    wait_cap_n = wait_cap_true + wait_cap_false

    own_before = [
        g.own_castles_before_enemy_capture
        for g in games
        if g.own_castles_before_enemy_capture is not None
    ]
    first_own = [g.first_own_castle for g in games if g.first_own_castle is not None]
    first_ecap = [
        g.first_enemy_castle_capture for g in games if g.first_enemy_castle_capture is not None
    ]
    n_enemy_castle = sum(1 for g in games if g.enemy_castles_captured > 0)

    # Engagement thresholds: fraction with army_ratio >= 1.0 / 1.2 / 1.5 at contact
    def frac_ge(vals: list[float], thr: float) -> float | None:
        if not vals:
            return None
        return sum(1 for v in vals if v >= thr) / len(vals)

    # No-attack despite contact
    no_attack = [g for g in games if g.contact_no_capture_long]
    delayed = [g for g in games if g.delayed_capture]
    no_attack_army = [g.army_ratio_contact for g in no_attack if g.army_ratio_contact is not None]
    attack_quick = [
        g
        for g in games
        if g.contact_to_capture is not None and g.contact_to_capture <= 10
    ]
    quick_army = [g.army_ratio_contact for g in attack_quick if g.army_ratio_contact is not None]

    # Counterexamples: push despite army ratio < 0.8 at contact
    push_behind = [
        g.match_id
        for g in games
        if g.front_class in ("push", "gather_then_push")
        and g.army_ratio_contact is not None
        and g.army_ratio_contact < 0.8
    ]
    # Sight→kill very long (>150) despite toward high
    long_strike = [
        {"match_id": g.match_id, "sight_to_kill": g.sight_to_kill, "toward": g.toward_after_sight}
        for g in games
        if g.sight_to_kill is not None and g.sight_to_kill > 150
    ][:15]

    # Commit ratio counterexamples: capture with sent/held < 1.0
    undercommit_games = []
    for g in games:
        unders = [r for r in g.commit_ratios if r < 1.0]
        if unders:
            undercommit_games.append(
                {"match_id": g.match_id, "n_under": len(unders), "min_ratio": min(unders)}
            )
    undercommit_games.sort(key=lambda x: x["n_under"], reverse=True)

    # Capture while gather still open (counterexample to wait rule)
    capture_during_gather = [
        g.match_id
        for g in open_contact
        if g.capture_after_open_gather_ends is False
    ]
    strike_during_gather = [
        g.match_id for g in open_sight if g.strike_after_open_gather_ends is False
    ]

    claims = []
    claims.append(
        _claim(
            "MEASURED",
            "At first_contact, Kubic's army/opponent army median is "
            f"{percentile(army_c, 50):.3f} (p25={percentile(army_c, 25):.3f}, "
            f"p75={percentile(army_c, 75):.3f}).",
            n=len(army_c),
            detail=dist_summary(army_c),
        )
    )
    claims.append(
        _claim(
            "MEASURED",
            "At first_contact, tile ratio median is "
            f"{percentile(tile_c, 50):.3f}; max_stack median is "
            f"{percentile(stack_c, 50):.0f}.",
            n=len(tile_c),
            detail={
                "tile_ratio": dist_summary(tile_c),
                "stack_at_contact": dist_summary(stack_c),
                "frac_army_ratio_ge_1.0": frac_ge(army_c, 1.0),
                "frac_army_ratio_ge_1.2": frac_ge(army_c, 1.2),
                "frac_army_ratio_ge_1.5": frac_ge(army_c, 1.5),
            },
        )
    )
    claims.append(
        _claim(
            "MEASURED",
            "Contact→first_capture latency median is "
            f"{percentile(c2c, 50):.0f} ticks "
            f"(p10={percentile(c2c, 10):.0f}, p90={percentile(c2c, 90):.0f}).",
            n=len(c2c),
            detail=dist_summary(c2c),
        )
    )
    claims.append(
        _claim(
            "MEASURED",
            "At first_capture, army ratio median is "
            f"{percentile(army_cap, 50):.3f}; stack median "
            f"{percentile(stack_cap, 50):.0f}.",
            n=len(army_cap),
            detail={
                "army_ratio": dist_summary(army_cap),
                "tile_ratio": dist_summary(tile_cap),
                "stack": dist_summary(stack_cap),
            },
        )
    )
    claims.append(
        _claim(
            "MEASURED",
            "Capture commit sent/held (all sampled): median "
            f"{percentile(all_commit, 50):.3f}. Stratified — held==1 median "
            f"{percentile(commit_held1, 50) if commit_held1 else float('nan'):.3f} (n={len(commit_held1)}); "
            f"held 2–9 median {percentile(commit_held_2_9, 50) if commit_held_2_9 else float('nan'):.3f} "
            f"(n={len(commit_held_2_9)}); held>=10 median "
            f"{percentile(commit_held_ge10, 50) if commit_held_ge10 else float('nan'):.3f} "
            f"(n={len(commit_held_ge10)}); surplus(sent-held) at held>=10 median "
            f"{percentile(surplus_held_ge10, 50) if surplus_held_ge10 else float('nan'):.0f}. "
            f"First-capture sent/held median "
            f"{percentile(first_cap_ratio, 50) if first_cap_ratio else float('nan'):.3f}.",
            n=len(all_commit),
            detail={
                "sent_over_held_all": dist_summary(all_commit),
                "sent_over_held_held1": dist_summary(commit_held1),
                "sent_over_held_held_2_9": dist_summary(commit_held_2_9),
                "sent_over_held_held_ge10": dist_summary(commit_held_ge10),
                "surplus_held_ge10": dist_summary(surplus_held_ge10),
                "stack_over_held": dist_summary(all_stack_held),
                "held_dist": dist_summary(all_held),
                "first_capture_sent_over_held": dist_summary(first_cap_ratio),
                "first_capture_held": dist_summary(first_cap_held),
                "frac_sent_ge_held": frac_ge(all_commit, 1.0),
                "frac_sent_ge_held_ge10": frac_ge(commit_held_ge10, 1.0),
                "frac_sent_ge_1.5x_held_ge10": frac_ge(commit_held_ge10, 1.5),
            },
        )
    )
    claims.append(
        _claim(
            "INFERRED",
            "Candidate commit rule on defended cells (held>=10): send at least held army "
            f"({100 * (frac_ge(commit_held_ge10, 1.0) or 0):.1f}% of samples) and typically a large "
            f"surplus (median surplus={percentile(surplus_held_ge10, 50) if surplus_held_ge10 else None}). "
            "On held==1 frontier cells the ratio is huge because a full stack rolls light tiles — "
            "do not treat the all-sample median as a threshold.",
            n=len(commit_held_ge10),
            detail={
                "supporting_frac_sent_ge_1_held_ge10": frac_ge(commit_held_ge10, 1.0),
                "counterexample_games_top": undercommit_games[:10],
            },
        )
    )
    claims.append(
        _claim(
            "MEASURED",
            f"Front-window ({FRONT_WINDOW} ticks post-contact) classes: "
            + ", ".join(f"{k}={v} ({100 * v / n:.1f}%)" for k, v in sorted(front_counts.items())),
            n=n,
            detail=dict(front_counts),
        )
    )
    claims.append(
        _claim(
            "MEASURED",
            f"Sight→kill latency median {percentile(sight_to_kill, 50):.0f} ticks "
            f"(p25={percentile(sight_to_kill, 25):.0f}, p75={percentile(sight_to_kill, 75):.0f}, "
            f"p90={percentile(sight_to_kill, 90):.0f}); "
            f"stack_at_sight median {percentile(stack_sight, 50):.0f}; "
            f"toward_fraction after sight median {percentile(toward, 50):.3f}.",
            n=len(sight_to_kill),
            detail={
                "sight_to_kill": dist_summary(sight_to_kill),
                "stack_at_sight": dist_summary(stack_sight),
                "toward_after_sight": dist_summary(toward),
                "army_ratio_sight": dist_summary(army_sight),
                "n_with_sight": n_sight,
                "frac_with_sight": _rate(n_sight, n),
            },
        )
    )
    claims.append(
        _claim(
            "INFERRED",
            "After first_general_sight, Kubic's largest stack moves toward the enemy "
            f"general on a median {percentile(toward, 50):.0%} of directed moves "
            f"(p10={percentile(toward, 10):.0%}) — strike is the default post-sight mode.",
            n=len(toward),
            detail={"toward_after_sight": dist_summary(toward)},
        )
    )

    claims.append(
        _claim(
            "MEASURED",
            f"Gather open at first_contact in {len(open_contact)}/{n} games; "
            f"among those, first_capture waited for gather end in "
            f"{wait_cap_true}/{wait_cap_n if wait_cap_n else 0}"
            + (f" ({100 * wait_cap_true / wait_cap_n:.1f}%)" if wait_cap_n else "")
            + f". Gather completed between contact and capture in {gather_between}/{n}. "
            f"Gather open at sight in {len(open_sight)}/{n}; among measurable strike waits "
            f"{wait_strike_true}/{wait_strike_n if wait_strike_n else 0}"
            + (f" ({100 * wait_strike_true / wait_strike_n:.1f}%)" if wait_strike_n else "")
            + ".",
            n=n,
            detail={
                "gather_open_at_contact": len(open_contact),
                "capture_waited": wait_cap_true,
                "capture_did_not_wait": wait_cap_false,
                "gather_between_contact_capture": gather_between,
                "gather_open_at_sight": len(open_sight),
                "strike_waited": wait_strike_true,
                "strike_did_not_wait": wait_strike_false,
                "capture_during_gather_ids": capture_during_gather[:20],
                "strike_during_gather_ids": strike_during_gather[:20],
            },
        )
    )
    if wait_cap_n == 0 and wait_strike_n == 0:
        claims.append(
            _claim(
                "UNKNOWN",
                "Rare open gather at contact/sight — cannot firmly decide a wait-for-gather gate.",
            )
        )
    elif wait_cap_n and wait_cap_true / wait_cap_n >= 0.7:
        claims.append(
            _claim(
                "INFERRED",
                "When a gather_wave is open at contact, first enemy capture usually waits "
                f"for that wave to end ({wait_cap_true}/{wait_cap_n}).",
                n=wait_cap_n,
            )
        )
    else:
        claims.append(
            _claim(
                "INFERRED",
                "Open gather at contact does not reliably gate first capture "
                f"(waited {wait_cap_true}/{wait_cap_n}).",
                n=wait_cap_n,
            )
        )

    claims.append(
        _claim(
            "MEASURED",
            f"Enemy castle captures occur in {n_enemy_castle}/{n} fit wins "
            f"({100 * n_enemy_castle / n:.1f}%). When they occur, own castles already built "
            f"before that capture: median {percentile(own_before, 50) if own_before else float('nan'):.0f}. "
            f"First own castle tick median {percentile(first_own, 50) if first_own else float('nan'):.0f}; "
            f"first enemy-castle-capture tick median "
            f"{percentile(first_ecap, 50) if first_ecap else float('nan'):.0f}.",
            n=n,
            detail={
                "n_enemy_castle_capture_games": n_enemy_castle,
                "own_castles_before_enemy_capture": dist_summary(own_before),
                "first_own_castle_tick": dist_summary(first_own),
                "first_enemy_castle_capture_tick": dist_summary(first_ecap),
                "own_castles_built": dist_summary([g.own_castles_built for g in games]),
            },
        )
    )

    # No-attack conditions
    claims.append(
        _claim(
            "MEASURED",
            f"Contact without our capture for >={NO_ATTACK_CAPTURE_GAP} ticks (or never): "
            f"{len(no_attack)}/{n} ({100 * len(no_attack) / n:.1f}%). "
            f"Quick capture (<=10 ticks): {len(attack_quick)}/{n}. "
            f"Median army_ratio at contact when delayed={percentile(no_attack_army, 50) if no_attack_army else None}; "
            f"when quick={percentile(quick_army, 50) if quick_army else None}.",
            n=n,
            detail={
                "delayed_or_never": len(no_attack),
                "delayed_army_ratio": dist_summary(no_attack_army),
                "quick_army_ratio": dist_summary(quick_army),
                "delayed_match_ids": [g.match_id for g in no_attack[:20]],
            },
        )
    )
    if no_attack_army and quick_army:
        claims.append(
            _claim(
                "INFERRED",
                "Delayed capture after contact is not explained by being army-behind alone: "
                f"delayed median army_ratio={percentile(no_attack_army, 50):.3f} vs "
                f"quick={percentile(quick_army, 50):.3f}. "
                "Likely causes include map chokepoints, unfinished gather, or castle-build priority "
                "(see UNKNOWN).",
                n=len(no_attack),
            )
        )

    claims.append(
        _claim(
            "MEASURED",
            f"Counterexample set: push/gather_then_push while army_ratio_contact < 0.8 — "
            f"{len(push_behind)} games.",
            n=len(push_behind),
            detail={"match_ids": push_behind[:20]},
        )
    )

    # Loss skim
    loss_claims = []
    loss_rows = []
    for g in losses:
        row = {
            "match_id": g.match_id,
            "ticks": g.ticks,
            "first_contact": g.first_contact,
            "first_capture_us": g.first_capture_us,
            "army_ratio_contact": g.army_ratio_contact,
            "first_sight": g.first_sight,
            "sight_to_kill": g.sight_to_kill,
            "stack_at_sight": g.stack_at_sight,
            "toward_after_sight": g.toward_after_sight,
            "front_class": g.front_class,
            "kill_tick": g.kill_tick,
        }
        fails = []
        if g.first_sight is None:
            fails.append("never_saw_general")
        if g.first_capture_us is None and g.first_contact is not None:
            fails.append("contact_no_capture")
        if g.army_ratio_contact is not None and g.army_ratio_contact < 0.9:
            fails.append("army_behind_at_contact")
        if g.toward_after_sight is not None and g.toward_after_sight < 0.5:
            fails.append("low_toward_after_sight")
        if g.kill_tick is None:
            fails.append("never_killed")
        if g.stack_at_sight is not None and g.stack_at_sight < 20:
            fails.append("small_stack_at_sight")
        row["failed_attack_conditions"] = fails
        loss_rows.append(row)
    fail_counter = Counter(f for r in loss_rows for f in r["failed_attack_conditions"])
    loss_claims.append(
        _claim(
            "MEASURED",
            "Loss skim failure tags: "
            + ", ".join(f"{k}={v}" for k, v in fail_counter.most_common()),
            n=len(losses),
            detail={"rows": loss_rows, "tag_counts": dict(fail_counter)},
        )
    )

    cannot = [
        "Intentional fog memory vs re-sight each tick (engine fades fog; bot may remember).",
        "Exact priority between castle-build spend and front push on the same tick.",
        "Whether half-moves are used specifically for probing vs economy (move inference ambiguous).",
        "True action when multi-stack combat occurs (largest-stack path only).",
        "Whether 'no attack despite contact' is a deliberate hold rule or a pathing/map artifact.",
        "Opponent-visible army estimation error under fog (we use true replay armies when adjacent).",
        "All-sample sent/held median is dominated by held==1 tiles; use held>=10 stratum for thresholds.",
    ]

    gather_rule_stmt = (
        f"When a gather_wave is open at contact (n={wait_cap_n}), first_capture waited for end in "
        f"{wait_cap_true}/{wait_cap_n}."
        if wait_cap_n
        else "Open gather at contact is rare; wait-for-gather gate not firmly measurable."
    )
    gather_rule_stmt += (
        f" Gather completed between contact and first_capture in {gather_between}/{n} games. "
        f"Gather open at sight: {len(open_sight)}/{n} (typically gather finishes before sight)."
    )
    if wait_strike_n:
        gather_rule_stmt += (
            f" When gather open at sight, strike waited in {wait_strike_true}/{wait_strike_n}."
        )

    ge10_med = percentile(commit_held_ge10, 50)
    ge10_surplus = percentile(surplus_held_ge10, 50)
    first_med = percentile(first_cap_ratio, 50)

    top_rules = [
        {
            "rule": "ENGAGE_RATIO",
            "statement": "First contact typically occurs near parity or slight army lead; "
            f"median army_ratio={percentile(army_c, 50):.2f}, "
            f"{100 * (frac_ge(army_c, 1.0) or 0):.0f}% of fit wins have army_ratio>=1.0; "
            f"median stack_at_contact={percentile(stack_c, 50):.0f}.",
            "units": "own_army / enemy_army at first_contact; stack army",
            "confidence": "MEASURED",
            "n": len(army_c),
        },
        {
            "rule": "COMMIT_SENT_GE_HELD",
            "statement": (
                f"On held>=10 cells, sent/held median={ge10_med:.2f}, "
                f"frac(sent>=held)={frac_ge(commit_held_ge10, 1.0):.2f}; "
                f"median surplus={ge10_surplus:.1f}. "
                f"First-capture sent/held median={first_med:.2f} "
                f"(first held median={percentile(first_cap_held, 50):.0f}, usually a 1-army frontier tile)."
            ),
            "units": "sent / enemy_cell_army; surplus army",
            "confidence": "INFERRED",
            "n": len(commit_held_ge10),
        },
        {
            "rule": "POST_SIGHT_STRIKE",
            "statement": "After first_general_sight, drive largest stack toward enemy general "
            f"(median toward_fraction={percentile(toward, 50):.2f}; "
            f"median sight→kill={percentile(sight_to_kill, 50):.0f} ticks; "
            f"median stack_at_sight={percentile(stack_sight, 50):.0f}).",
            "units": "toward_fraction of directed moves; ticks; army",
            "confidence": "MEASURED",
            "n": len(toward),
        },
        {
            "rule": "FRONT_PUSH_DEFAULT",
            "statement": f"In the {FRONT_WINDOW}-tick window after contact, dominant class is "
            f"'{front_counts.most_common(1)[0][0] if front_counts else '?'}' "
            f"({front_counts.most_common(1)[0][1] if front_counts else 0}/{n}). "
            f"Trade={front_counts.get('trade', 0)}, gather_then_push={front_counts.get('gather_then_push', 0)}, "
            f"passive={front_counts.get('passive', 0)}.",
            "units": "front_class counts",
            "confidence": "MEASURED",
            "n": n,
        },
        {
            "rule": "GATHER_THEN_STRIKE",
            "statement": gather_rule_stmt,
            "units": "boolean wait given open gather; counts",
            "confidence": "MEASURED" if (wait_cap_n or wait_strike_n) else "INFERRED",
            "n": wait_cap_n or gather_between,
        },
        {
            "rule": "CASTLE_THEN_RAID",
            "statement": (
                "Own castle production precedes enemy-castle capture when both occur "
                f"(median own castles already built="
                f"{percentile(own_before, 50):.0f}). "
                if own_before
                else "Own-vs-enemy castle ordering not measured (no overlapping cases). "
            )
            + (
                f"Enemy castle captures in {n_enemy_castle}/{n} wins — not required to win. "
                f"Median first enemy-castle-capture tick={percentile(first_ecap, 50):.0f}."
                if first_ecap
                else f"Enemy castle captures in {n_enemy_castle}/{n} wins — not required to win."
            ),
            "units": "castle counts / ticks",
            "confidence": "MEASURED",
            "n": n_enemy_castle,
        },
    ]

    return {
        "player": "Kubic",
        "dimension": "attack",
        "set": "fit_wins",
        "n_fit": n,
        "n_holdout_excluded": len(meta.holdout_win_ids),
        "n_losses_skimmed": len(losses),
        "split_ref": "docs/research/measurements/grok-kubic-corpus-split.json",
        "parameters": {
            "front_window_ticks": FRONT_WINDOW,
            "no_attack_capture_gap": NO_ATTACK_CAPTURE_GAP,
            "commit_sample_cap_per_game": COMMIT_SAMPLE_CAP,
        },
        "distributions": {
            "army_ratio_contact": dist_summary(army_c),
            "tile_ratio_contact": dist_summary(tile_c),
            "stack_at_contact": dist_summary(stack_c),
            "army_ratio_capture": dist_summary(army_cap),
            "tile_ratio_capture": dist_summary(tile_cap),
            "stack_at_capture": dist_summary(stack_cap),
            "contact_to_capture": dist_summary(c2c),
            "commit_sent_over_held": dist_summary(all_commit),
            "commit_sent_over_held_held1": dist_summary(commit_held1),
            "commit_sent_over_held_2_9": dist_summary(commit_held_2_9),
            "commit_sent_over_held_ge10": dist_summary(commit_held_ge10),
            "commit_surplus_held_ge10": dist_summary(surplus_held_ge10),
            "commit_stack_over_held": dist_summary(all_stack_held),
            "first_capture_sent_over_held": dist_summary(first_cap_ratio),
            "first_capture_held": dist_summary(first_cap_held),
            "sight_to_kill": dist_summary(sight_to_kill),
            "stack_at_sight": dist_summary(stack_sight),
            "toward_after_sight": dist_summary(toward),
            "army_ratio_sight": dist_summary(army_sight),
            "first_own_castle_tick": dist_summary(first_own),
            "first_enemy_castle_capture_tick": dist_summary(first_ecap),
            "own_castles_before_enemy_capture": dist_summary(own_before),
        },
        "rates": {
            "n_contact": n_contact,
            "n_capture": n_capture,
            "n_sight": n_sight,
            "frac_contact": _rate(n_contact, n),
            "frac_capture": _rate(n_capture, n),
            "frac_sight": _rate(n_sight, n),
            "front_class": dict(front_counts),
            "frac_army_ge_1_at_contact": frac_ge(army_c, 1.0),
            "frac_sent_ge_held": frac_ge(all_commit, 1.0),
            "frac_sent_ge_held_ge10": frac_ge(commit_held_ge10, 1.0),
            "gather_open_at_contact": len(open_contact),
            "capture_wait_for_open_gather_rate": _rate(wait_cap_true, wait_cap_n)
            if wait_cap_n
            else None,
            "gather_between_contact_capture": gather_between,
            "gather_open_at_sight": len(open_sight),
            "strike_wait_for_open_gather_rate": _rate(wait_strike_true, wait_strike_n)
            if wait_strike_n
            else None,
        },
        "top_rules": top_rules,
        "claims": claims,
        "counterexamples": {
            "push_while_army_ratio_lt_0.8": push_behind[:30],
            "long_sight_to_kill_gt_150": long_strike,
            "undercommit_captures": undercommit_games[:20],
            "delayed_capture_after_contact": [g.match_id for g in delayed[:30]],
            "capture_during_open_gather": capture_during_gather[:30],
            "strike_during_open_gather": strike_during_gather[:30],
        },
        "cannot_determine": cannot,
        "loss_skim": {
            "claims": loss_claims,
            "failure_tag_counts": dict(fail_counter),
            "rows": loss_rows,
        },
        "paths": {
            "script": "scripts/analyze_kubic_attack.py",
            "json": "docs/research/measurements/grok-kubic-attack.json",
            "md": "docs/research/measurements/grok-kubic-attack.md",
            "corpus": "scripts/kubic_corpus.py",
            "moves": "scripts/kubic_moves.py",
            "split": "docs/research/measurements/grok-kubic-corpus-split.json",
        },
        "games_compact": [g.as_json() for g in games],
    }


def render_md(report: dict) -> str:
    lines: list[str] = []
    lines.append("# Kubic attack behavior (fit set)")
    lines.append("")
    lines.append(
        f"Fit wins n={report['n_fit']}; holdout excluded n={report['n_holdout_excluded']}; "
        f"losses skimmed n={report['n_losses_skimmed']}."
    )
    lines.append("")
    lines.append("Reproduce: `python scripts/analyze_kubic_attack.py`")
    lines.append("")
    lines.append("## Top rules")
    lines.append("")
    for r in report["top_rules"]:
        lines.append(
            f"- **{r['rule']}** [{r['confidence']}, n={r['n']}]: {r['statement']}"
        )
    lines.append("")
    lines.append("## Key distributions")
    lines.append("")
    d = report["distributions"]
    for key in (
        "army_ratio_contact",
        "tile_ratio_contact",
        "stack_at_contact",
        "contact_to_capture",
        "commit_sent_over_held",
        "commit_sent_over_held_held1",
        "commit_sent_over_held_2_9",
        "commit_sent_over_held_ge10",
        "commit_surplus_held_ge10",
        "first_capture_sent_over_held",
        "sight_to_kill",
        "stack_at_sight",
        "toward_after_sight",
    ):
        s = d.get(key) or {}
        if not s or s.get("n", 0) == 0:
            lines.append(f"- `{key}`: n=0")
            continue
        lines.append(
            f"- `{key}`: n={s['n']}, median={s['median']:.3f}, "
            f"p25={s['p25']:.3f}, p75={s['p75']:.3f}, mean={s['mean']:.3f}"
        )
    lines.append("")
    lines.append("## Front behavior (post-contact window)")
    lines.append("")
    for k, v in sorted(report["rates"]["front_class"].items(), key=lambda x: -x[1]):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Tagged claims")
    lines.append("")
    for c in report["claims"]:
        extra = f" (n={c['n']})" if "n" in c else ""
        lines.append(f"- [{c['tag']}]{extra} {c['claim']}")
    lines.append("")
    lines.append("## Counterexamples")
    lines.append("")
    cex = report["counterexamples"]
    lines.append(
        f"- Push while army_ratio_contact < 0.8: {len(cex['push_while_army_ratio_lt_0.8'])} "
        f"(e.g. {', '.join(cex['push_while_army_ratio_lt_0.8'][:8]) or 'none'})"
    )
    lines.append(
        f"- Sight→kill > 150 ticks: {len(cex['long_sight_to_kill_gt_150'])} sample rows in JSON"
    )
    lines.append(
        f"- Undercommit captures (sent/held < 1): {len(cex['undercommit_captures'])} games"
    )
    lines.append(
        f"- Delayed capture after contact: {len(cex['delayed_capture_after_contact'])} listed"
    )
    lines.append("")
    lines.append("## Loss skim — attack conditions that failed")
    lines.append("")
    for k, v in sorted(report["loss_skim"]["failure_tag_counts"].items(), key=lambda x: -x[1]):
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    for row in report["loss_skim"]["rows"]:
        lines.append(
            f"- match {row['match_id']}: contact={row['first_contact']}, "
            f"army_ratio={row['army_ratio_contact']}, sight={row['first_sight']}, "
            f"toward={row['toward_after_sight']}, fails={row['failed_attack_conditions']}"
        )
    lines.append("")
    lines.append("## Cannot determine")
    lines.append("")
    for item in report["cannot_determine"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Paths")
    lines.append("")
    for k, v in report["paths"].items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    print("Analyzing fit wins…")
    games: list[GameAttack] = []
    for i, (replay, an) in enumerate(iter_analyzed("fit"), 1):
        games.append(analyze_game(replay, an))
        if i % 25 == 0:
            print(f"  fit {i}/{len(corpus_meta().fit_win_ids)}")
    print(f"Fit done: {len(games)}")

    print("Skimming losses…")
    losses: list[GameAttack] = []
    for replay, an in iter_analyzed("losses"):
        losses.append(analyze_game(replay, an))
    print(f"Losses: {len(losses)}")

    report = build_report(games, losses)
    # Drop bulky games_compact from markdown path but keep in JSON
    json_path = dump_json("grok-kubic-attack.json", report)
    md = render_md(report)
    md_path = MEASUREMENTS / "grok-kubic-attack.md"
    md_path.write_text(md)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")

    # stdout summary for parent agent
    print("\n=== TOP RULES ===")
    for r in report["top_rules"]:
        print(f"[{r['confidence']}] {r['rule']}: {r['statement']}")


if __name__ == "__main__":
    main()
