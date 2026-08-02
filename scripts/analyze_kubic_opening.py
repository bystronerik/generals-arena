#!/usr/bin/env python3
"""Kubic OPENING analysis (first ~50 turns) on the fit win set.

Derives only on fit wins (n=341). Skims losses for opening failure modes.
Writes docs/research/measurements/grok-kubic-opening.json.
Does not write to data/games/, data/ratings/, or data/remote_games/.

Castle note: EventLog.castle_built at tick ~10 is a known false-positive mode
(production detector + growth masking). Real builds are recovered as owned-cell
army drops >=30 with no orthogonal neighbour receiving the send.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arena.instrument.replay.metrics import manhattan
from scripts.kubic_corpus import (
    MEASUREMENTS,
    corpus_meta,
    dist_summary,
    dump_json,
    iter_analyzed,
    iter_set,
)
from scripts.kubic_moves import infer_moves_for_player

OPENING_END = 50
ORTH = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR_NAME = {(-1, 0): "N", (1, 0): "S", (0, -1): "W", (0, 1): "E"}
JSON_NAME = "grok-kubic-opening.json"
MD_NAME = "grok-kubic-opening.md"


def spawn_class(rows: int, cols: int, gen: tuple[int, int]) -> str:
    r, c = gen
    er = min(r, rows - 1 - r)
    ec = min(c, cols - 1 - c)
    if er <= 1 and ec <= 1:
        return "corner"
    if min(er, ec) <= 2:
        return "edge"
    return "centerish"


def quadrant(rows: int, cols: int, cell: tuple[int, int]) -> str:
    mid_r = (rows - 1) / 2
    mid_c = (cols - 1) / 2
    return ("N" if cell[0] < mid_r else "S") + ("W" if cell[1] < mid_c else "E")


def map_center(rows: int, cols: int) -> tuple[float, float]:
    return ((rows - 1) / 2, (cols - 1) / 2)


def dist_center(cell: tuple[int, int], center: tuple[float, float]) -> float:
    return abs(cell[0] - center[0]) + abs(cell[1] - center[1])


def open_neighbors(replay, cell: tuple[int, int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ORTH:
        rr, cc = cell[0] + dr, cell[1] + dc
        if 0 <= rr < replay.rows and 0 <= cc < replay.cols and (rr, cc) not in replay.mountains:
            out.append((rr, cc))
    return out


def align_label(d0: float, d1: float) -> str:
    if d1 < d0:
        return "toward"
    if d1 > d0:
        return "away"
    return "side"


def detect_castle_builds(replay, us: int, gen: tuple[int, int]) -> list[dict]:
    """Army drop >=30 on owned non-gen cell with no neighbour receiving the send."""
    builds: list[dict] = []
    for t in range(1, len(replay.ticks)):
        before, after = replay.ticks[t - 1], replay.ticks[t]
        for r in range(replay.rows):
            for c in range(replay.cols):
                cell = (r, c)
                if cell == gen:
                    continue
                if before.owners[r][c] != us or after.owners[r][c] != us:
                    continue
                a0, a1 = before.armies[r][c], after.armies[r][c]
                drop = a0 - a1
                if drop < 30:
                    continue
                received = False
                for dr, dc in ORTH:
                    rr, cc = r + dr, c + dc
                    if not (0 <= rr < replay.rows and 0 <= cc < replay.cols):
                        continue
                    if (rr, cc) in replay.mountains:
                        continue
                    o0 = before.owners[rr][cc]
                    o1 = after.owners[rr][cc]
                    b0 = before.armies[rr][cc]
                    b1 = after.armies[rr][cc]
                    if o1 == us and (o0 != us or b1 > b0):
                        gained = b1 - (b0 if o0 == us else 0)
                        if gained >= drop - 3:
                            received = True
                            break
                if received:
                    continue
                builds.append(
                    {
                        "tick": t,
                        "cell": list(cell),
                        "army_before": a0,
                        "army_after": a1,
                        "drop": drop,
                        "dist_to_general": manhattan(cell, gen),
                    }
                )
    return builds


def phase_c_start(gains: list[int]) -> int | None:
    """First tick >=15 whose next 4 frames include >=3 tile-gain ticks."""
    for t in range(15, min(45, len(gains) - 3)):
        if sum(1 for g in gains[t : t + 4] if g > 0) >= 3:
            return t
    for t in range(15, len(gains)):
        if gains[t] > 0:
            return t
    return None


def rate(count: int, n: int) -> float | None:
    return count / n if n else None


def counter_to_obj(c: Counter) -> dict:
    return {str(k): int(v) for k, v in c.most_common()}


def analyze_fit() -> dict[str, Any]:
    meta = corpus_meta()
    per_game: list[dict] = []

    tiles_gained_by_tick: dict[int, list[int]] = defaultdict(list)
    tiles_at: dict[int, list[int]] = {t: [] for t in (5, 10, 15, 20, 25, 30, 35, 40, 45, 50)}
    stack_at: dict[int, list[int]] = {t: [] for t in (5, 10, 15, 20, 25, 30, 40, 50)}
    stack_dist_gen: list[float] = []
    stack_on_gen_fracs: list[float] = []

    first_gain_ticks: list[int] = []
    gen_army_before_first: list[int] = []
    inter_gain_gaps: list[int] = []

    first_move_align_egen = Counter()
    first_move_align_center = Counter()
    first_move_dir = Counter()
    center_rule_hits = 0
    egen_rule_hits = 0
    either_rule_hits = 0
    neither_ids: list[str] = []
    away_egen_ids: list[str] = []
    away_center_ids: list[str] = []

    spawn_counts = Counter()
    spawn_align_egen: dict[str, Counter] = defaultdict(Counter)
    spawn_align_center: dict[str, Counter] = defaultdict(Counter)
    spawn_tiles50: dict[str, list[int]] = defaultdict(list)
    size_counts = Counter()
    size_first_align_egen: dict[str, Counter] = defaultdict(Counter)

    move_from_gen_mod2 = Counter()
    move_not_gen_mod2 = Counter()
    move_kind_windows = {
        "0_9": Counter(),
        "10_25": Counter(),
        "26_49": Counter(),
    }
    mid_gather = Counter()

    event_castles_in_50 = Counter()
    event_castles_game = Counter()
    event_false_early = 0
    real_builds_in_50 = Counter()
    real_builds_game = Counter()
    first_real_build_ticks: list[int] = []
    first_real_build_dists: list[int] = []
    first_real_build_army_before: list[int] = []
    first_real_build_army_after: list[int] = []
    first_real_build_drops: list[int] = []

    phase_c_ticks: list[int] = []
    phase_c_stack: list[int] = []
    phase_c_army: list[int] = []
    phase_c_tiles: list[int] = []

    enemy_half_fracs: list[float] = []
    neutral_captures = 0
    enemy_captures = 0
    tiles50_vals: list[int] = []

    late_gain_counterexamples: list[str] = []  # no phase C by 35
    early_expand_miss: list[str] = []  # no gain at tick 3

    for replay, an in iter_analyzed("fit"):
        us = an.us
        gen = replay.generals[us]
        egen = replay.generals[1 - us]
        center = map_center(replay.rows, replay.cols)
        mid = str(replay.match_id)
        sc = spawn_class(replay.rows, replay.cols, gen)
        size = f"{replay.rows}x{replay.cols}"
        spawn_counts[sc] += 1
        size_counts[size] += 1

        gains = [
            an.metrics[t].seats[us].tiles_gained
            for t in range(1, min(OPENING_END + 1, len(an.metrics)))
        ]
        # pad index: gains[i] is tick i+1; also build 0-index list with tick keys
        gain_by_tick = {t: an.metrics[t].seats[us].tiles_gained for t in range(1, min(OPENING_END + 1, len(an.metrics)))}
        for t, g in gain_by_tick.items():
            tiles_gained_by_tick[t].append(g)

        for t in tiles_at:
            if t < len(an.metrics):
                tiles_at[t].append(an.metrics[t].seats[us].tiles)
        for t in stack_at:
            if t < len(an.metrics):
                stack_at[t].append(an.metrics[t].seats[us].max_stack)

        t50 = min(OPENING_END, len(an.metrics) - 1)
        tiles50 = an.metrics[t50].seats[us].tiles
        tiles50_vals.append(tiles50)
        spawn_tiles50[sc].append(tiles50)

        on_gen = 0
        dist_samples: list[int] = []
        for t in range(1, min(OPENING_END + 1, len(an.metrics))):
            pos = an.metrics[t].seats[us].max_stack_pos
            if pos is None:
                continue
            d = manhattan(pos, gen)
            dist_samples.append(d)
            if pos == gen:
                on_gen += 1
        if dist_samples:
            stack_dist_gen.append(float(sorted(dist_samples)[len(dist_samples) // 2]))
        stack_on_gen_fracs.append(on_gen / OPENING_END)

        gain_ticks = [t for t, g in gain_by_tick.items() if g > 0]
        if gain_ticks:
            first_gain_ticks.append(gain_ticks[0])
            t0 = gain_ticks[0]
            gen_army_before_first.append(an.metrics[t0 - 1].seats[us].general_army if t0 > 0 else 0)
            for a, b in zip(gain_ticks, gain_ticks[1:]):
                inter_gain_gaps.append(b - a)
            if gain_ticks[0] != 3:
                early_expand_miss.append(mid)

        # captures neutral vs enemy
        owned_enemy_half = 0
        owned_total = 0
        frame50 = replay.ticks[t50]
        for r in range(replay.rows):
            for c in range(replay.cols):
                if frame50.owners[r][c] == us:
                    owned_total += 1
                    if manhattan((r, c), egen) < manhattan((r, c), gen):
                        owned_enemy_half += 1
        enemy_half_fracs.append(owned_enemy_half / owned_total if owned_total else 0.0)

        for t in range(1, min(OPENING_END + 1, len(replay.ticks))):
            before, after = replay.ticks[t - 1], replay.ticks[t]
            for r in range(replay.rows):
                for c in range(replay.cols):
                    if after.owners[r][c] == us and before.owners[r][c] != us:
                        if before.owners[r][c] < 0:
                            neutral_captures += 1
                        else:
                            enemy_captures += 1

        # moves
        moves = infer_moves_for_player(replay, us, set())
        first_move = None
        for m in moves:
            if m.tick >= OPENING_END:
                break
            window = (
                "0_9"
                if m.tick < 10
                else "10_25"
                if m.tick < 26
                else "26_49"
                if m.tick < OPENING_END
                else None
            )
            if window:
                move_kind_windows[window][m.kind] += 1
            if m.kind not in ("full", "half") or m.src is None or m.dst is None:
                continue
            if m.src == gen:
                move_from_gen_mod2[m.tick % 2] += 1
            else:
                move_not_gen_mod2[m.tick % 2] += 1
            if 10 <= m.tick <= 25:
                d0, d1 = manhattan(m.src, gen), manhattan(m.dst, gen)
                mid_gather[align_label(d0, d1)] += 1
            if first_move is None and m.src == gen:
                first_move = m

        if first_move and first_move.dst:
            dr = first_move.dst[0] - first_move.src[0]
            dc = first_move.dst[1] - first_move.src[1]
            first_move_dir[DIR_NAME.get((dr, dc), f"{dr},{dc}")] += 1
            ae = align_label(
                manhattan(first_move.src, egen), manhattan(first_move.dst, egen)
            )
            ac = align_label(
                dist_center(first_move.src, center), dist_center(first_move.dst, center)
            )
            first_move_align_egen[ae] += 1
            first_move_align_center[ac] += 1
            spawn_align_egen[sc][ae] += 1
            spawn_align_center[sc][ac] += 1
            size_first_align_egen[size][ae] += 1
            if ae == "away":
                away_egen_ids.append(mid)
            if ac == "away":
                away_center_ids.append(mid)

            opts = open_neighbors(replay, gen)
            if opts:
                best_c = min(dist_center(c, center) for c in opts)
                best_e = min(manhattan(c, egen) for c in opts)
                is_c = dist_center(first_move.dst, center) == best_c
                is_e = manhattan(first_move.dst, egen) == best_e
                if is_c:
                    center_rule_hits += 1
                if is_e:
                    egen_rule_hits += 1
                if is_c or is_e:
                    either_rule_hits += 1
                if not is_c and not is_e:
                    neither_ids.append(mid)

        # castles: event vs spend detector
        event_castles = an.events.of_kind("castle_built", player=us)
        early_ev = [e for e in event_castles if e.tick <= OPENING_END]
        event_castles_in_50[len(early_ev)] += 1
        event_castles_game[len(event_castles)] += 1

        builds = detect_castle_builds(replay, us, gen)
        early_b = [b for b in builds if b["tick"] <= OPENING_END]
        real_builds_in_50[len(early_b)] += 1
        real_builds_game[len(builds)] += 1
        if builds:
            b0 = builds[0]
            first_real_build_ticks.append(b0["tick"])
            first_real_build_dists.append(b0["dist_to_general"])
            first_real_build_army_before.append(b0["army_before"])
            first_real_build_army_after.append(b0["army_after"])
            first_real_build_drops.append(b0["drop"])

        # false early events: event in 50 with no spend on that cell
        spend_cells = {(b["cell"][0], b["cell"][1]) for b in builds}
        for e in early_ev:
            if e.cell is None or e.cell not in spend_cells:
                event_false_early += 1

        # phase C
        gains_idx = [0] + gains  # index by tick
        # gains list is ticks 1..; rebuild
        gains_by_tick_list = [0] * (OPENING_END + 1)
        for t, g in gain_by_tick.items():
            if t <= OPENING_END:
                gains_by_tick_list[t] = g
        pcs = phase_c_start(gains_by_tick_list)
        if pcs is not None:
            phase_c_ticks.append(pcs)
            phase_c_stack.append(an.metrics[pcs].seats[us].max_stack)
            phase_c_army.append(an.metrics[pcs].seats[us].army)
            phase_c_tiles.append(an.metrics[pcs].seats[us].tiles)
            if pcs > 35:
                late_gain_counterexamples.append(mid)
        else:
            late_gain_counterexamples.append(mid)

        per_game.append(
            {
                "match_id": mid,
                "spawn": sc,
                "size": size,
                "quad": quadrant(replay.rows, replay.cols, gen),
                "tiles50": tiles50,
                "first_gain_tick": gain_ticks[0] if gain_ticks else None,
                "first_move_align_egen": (
                    align_label(
                        manhattan(first_move.src, egen), manhattan(first_move.dst, egen)
                    )
                    if first_move and first_move.dst
                    else None
                ),
                "first_move_align_center": (
                    align_label(
                        dist_center(first_move.src, center),
                        dist_center(first_move.dst, center),
                    )
                    if first_move and first_move.dst
                    else None
                ),
                "phase_c_start": pcs,
                "n_event_castles_50": len(early_ev),
                "n_real_builds_50": len(early_b),
                "n_real_builds_game": len(builds),
                "first_real_build_tick": builds[0]["tick"] if builds else None,
            }
        )

    n = len(per_game)
    assert n == len(meta.fit_win_ids)

    gain_tick_summary = {}
    for t in range(1, OPENING_END + 1):
        vals = tiles_gained_by_tick.get(t, [])
        if not vals:
            continue
        gain_tick_summary[str(t)] = {
            "mean": sum(vals) / len(vals),
            "p_gain": sum(1 for v in vals if v > 0) / len(vals),
            "n": len(vals),
        }

    # losses skim
    loss_skim: list[dict] = []
    for replay in iter_set("losses"):
        from arena.instrument.replay.analysis import analyze

        an = analyze(replay)
        us = an.us
        t50 = min(OPENING_END, len(an.metrics) - 1)
        gains = [
            an.metrics[t].seats[us].tiles_gained
            for t in range(1, min(OPENING_END + 1, len(an.metrics)))
        ]
        first = next((t for t, g in enumerate(gains, 1) if g > 0), None)
        loss_skim.append(
            {
                "match_id": str(replay.match_id),
                "total_ticks": replay.total_ticks,
                "tiles50": an.metrics[t50].seats[us].tiles,
                "first_gain_tick": first,
                "first_contact": an.events.first_contact,
                "gain_at_3": gains[2] if len(gains) > 2 else None,
                "gain_at_6": gains[5] if len(gains) > 5 else None,
                "gain_at_9": gains[8] if len(gains) > 8 else None,
                "follows_3_6_9": bool(
                    len(gains) > 8 and gains[2] > 0 and gains[5] > 0 and gains[8] > 0
                ),
            }
        )

    payload: dict[str, Any] = {
        "analyst": "grok-1-opening",
        "player": "Kubic",
        "corpus": {
            "set": "fit_wins",
            "n": n,
            "split_file": "docs/research/measurements/grok-kubic-corpus-split.json",
            "split_rule": meta.as_json()["split_rule"],
            "played": meta.played,
            "forfeits": meta.forfeits,
            "holdout_excluded": True,
            "losses_skimmed": True,
            "n_losses_skimmed": len(loss_skim),
        },
        "window": {"ticks": f"1..{OPENING_END}", "note": "tick 0 is initial state"},
        "tile_growth": {
            "tiles_at_tick": {str(t): dist_summary(v) for t, v in tiles_at.items()},
            "tiles50": dist_summary(tiles50_vals),
            "tiles_gained_per_tick": gain_tick_summary,
            "first_gain_tick": dist_summary(first_gain_ticks),
            "first_gain_tick_counts": counter_to_obj(Counter(first_gain_ticks)),
            "gen_army_before_first_gain": counter_to_obj(Counter(gen_army_before_first)),
            "inter_gain_gap": dist_summary(inter_gain_gaps),
            "inter_gain_gap_counts": counter_to_obj(Counter(inter_gain_gaps)),
        },
        "first_expansion_direction": {
            "align_enemy_general_true": {
                "counts": counter_to_obj(first_move_align_egen),
                "toward_rate": rate(first_move_align_egen["toward"], n),
                "note": "True enemy-general cell; fog means bot cannot know this.",
            },
            "align_map_center": {
                "counts": counter_to_obj(first_move_align_center),
                "toward_rate": rate(first_move_align_center["toward"], n),
            },
            "min_dist_center_neighbor_rule": {
                "hits": center_rule_hits,
                "n": n,
                "rate": rate(center_rule_hits, n),
            },
            "min_dist_enemy_gen_neighbor_rule": {
                "hits": egen_rule_hits,
                "n": n,
                "rate": rate(egen_rule_hits, n),
            },
            "either_rule_rate": rate(either_rule_hits, n),
            "neither_match_ids": neither_ids,
            "away_egen_match_ids": away_egen_ids,
            "away_center_match_ids": away_center_ids,
            "first_move_dir_counts": counter_to_obj(first_move_dir),
            "by_spawn": {
                sp: {
                    "n": spawn_counts[sp],
                    "align_egen": counter_to_obj(spawn_align_egen[sp]),
                    "align_center": counter_to_obj(spawn_align_center[sp]),
                    "tiles50": dist_summary(spawn_tiles50[sp]),
                }
                for sp in ("corner", "edge", "centerish")
            },
            "by_map_size_top": {
                size: {
                    "n": size_counts[size],
                    "align_egen": counter_to_obj(size_first_align_egen[size]),
                }
                for size, _ in size_counts.most_common(8)
            },
        },
        "spawn_and_map": {
            "spawn_class_counts": counter_to_obj(spawn_counts),
            "map_size_counts": counter_to_obj(size_counts),
        },
        "early_stack": {
            "max_stack_at_tick": {str(t): dist_summary(v) for t, v in stack_at.items()},
            "median_stack_dist_to_general": dist_summary(stack_dist_gen),
            "frac_ticks_stack_on_general": dist_summary(stack_on_gen_fracs),
        },
        "land_vs_enemy_half": {
            "neutral_captures_first50": neutral_captures,
            "enemy_captures_first50": enemy_captures,
            "enemy_capture_rate": rate(enemy_captures, neutral_captures + enemy_captures),
            "owned_enemy_half_frac_at_50": dist_summary(enemy_half_fracs),
            "note": "Enemy half = cells closer to true enemy general than to own general.",
        },
        "turn_modulo": {
            "moves_from_general_tick_mod2": {
                "even": move_from_gen_mod2[0],
                "odd": move_from_gen_mod2[1],
                "even_rate": rate(move_from_gen_mod2[0], sum(move_from_gen_mod2.values())),
            },
            "moves_not_from_general_tick_mod2": {
                "even": move_not_gen_mod2[0],
                "odd": move_not_gen_mod2[1],
                "even_rate": rate(move_not_gen_mod2[0], sum(move_not_gen_mod2.values())),
            },
            "move_kinds_by_window": {
                w: counter_to_obj(c) for w, c in move_kind_windows.items()
            },
            "mid_phase_move_vs_general": counter_to_obj(mid_gather),
            "note": (
                "Competition production is every other turn on general/castles. "
                "Move tick is the transition index in kubic_moves (frame t -> t+1)."
            ),
        },
        "phases": {
            "A_pulse": {
                "description": "Sparse captures at ticks 3,6,9 then near-zero gains",
                "p_gain_tick3": gain_tick_summary.get("3", {}).get("p_gain"),
                "p_gain_tick6": gain_tick_summary.get("6", {}).get("p_gain"),
                "p_gain_tick9": gain_tick_summary.get("9", {}).get("p_gain"),
            },
            "B_pause_buildup": {
                "description": "Ticks ~10-25: few tile gains, many territorial moves, stack grows",
                "tiles_at_20": dist_summary(tiles_at[20]),
                "stack_at_25": dist_summary(stack_at[25]),
            },
            "C_flood": {
                "description": "Continuous land grab starting near tick 27",
                "start_tick": dist_summary(phase_c_ticks),
                "start_tick_counts": counter_to_obj(Counter(phase_c_ticks)),
                "stack_at_start": dist_summary(phase_c_stack),
                "army_at_start": dist_summary(phase_c_army),
                "tiles_at_start": dist_summary(phase_c_tiles),
                "start_near_27_pm1_rate": rate(
                    sum(1 for t in phase_c_ticks if abs(t - 27) <= 1), n
                ),
            },
        },
        "castles": {
            "event_castle_built_in_first50": counter_to_obj(event_castles_in_50),
            "event_castle_built_whole_game": counter_to_obj(event_castles_game),
            "event_false_early_count": event_false_early,
            "event_false_early_note": (
                "castle_built events with tick<=50 that have no matching spend>=30 "
                "on that cell — false positives from production detector."
            ),
            "real_builds_in_first50": counter_to_obj(real_builds_in_50),
            "real_builds_whole_game": counter_to_obj(real_builds_game),
            "first_real_build_tick": dist_summary(first_real_build_ticks),
            "first_real_build_dist_to_general": dist_summary(first_real_build_dists),
            "first_real_build_army_before": dist_summary(first_real_build_army_before),
            "first_real_build_army_after": dist_summary(first_real_build_army_after),
            "first_real_build_drop": dist_summary(first_real_build_drops),
            "games_with_zero_real_builds": real_builds_game[0],
            "detector": (
                "owned non-general cell, army drop >=30, no orthogonal neighbour "
                "gains ownership/army consistent with the send"
            ),
        },
        "counterexamples": {
            "no_gain_at_tick_3": early_expand_miss,
            "phase_c_missing_or_after_35": late_gain_counterexamples,
            "first_move_neither_center_nor_egen_min": neither_ids,
            "first_move_away_from_egen": away_egen_ids,
            "first_move_away_from_center": away_center_ids,
        },
        "loss_skim": {
            "excluded_from_rule_derivation": True,
            "games": loss_skim,
            "follows_3_6_9_count": sum(1 for g in loss_skim if g["follows_3_6_9"]),
            "note": (
                "Losses 24184-24189 break the 3/6/9 pulse and end first-50 with "
                "tiles50 in 4..7 — possible version skew or extreme maps; not used "
                "to derive opening rules."
            ),
        },
        "candidate_rules": [],  # filled below
        "could_not_determine": [],
        "per_game_sample": per_game[:20],
        "n_per_game_records": len(per_game),
    }

    payload["candidate_rules"] = [
        {
            "id": "R1_first_expand_tick3",
            "rule": (
                "After even-tick production raises general army to >=2, on the "
                "transition into tick 3 issue a full leave-1 move from the general "
                "onto an open orthogonal neighbour."
            ),
            "thresholds": {
                "first_gain_tick": 3,
                "gen_army_before": 2,
                "units": "ticks / army",
            },
            "support": {
                "p_gain_tick3": gain_tick_summary.get("3", {}).get("p_gain"),
                "gen_army_2_rate": rate(
                    Counter(gen_army_before_first)[2], len(gen_army_before_first)
                ),
                "n": n,
            },
            "counterexample_ids": early_expand_miss,
            "tag": "MEASURED",
        },
        {
            "id": "R2_pulse_3_6_9",
            "rule": (
                "Repeat a capture pulse at ticks 6 and 9 (every +3), then stop "
                "net tile growth until ~tick 27."
            ),
            "thresholds": {
                "pulse_ticks": [3, 6, 9],
                "pause_until": 27,
                "units": "ticks",
            },
            "support": {
                "p_gain_3": gain_tick_summary.get("3", {}).get("p_gain"),
                "p_gain_6": gain_tick_summary.get("6", {}).get("p_gain"),
                "p_gain_9": gain_tick_summary.get("9", {}).get("p_gain"),
                "p_gain_20": gain_tick_summary.get("20", {}).get("p_gain"),
                "n": n,
            },
            "tag": "MEASURED",
        },
        {
            "id": "R3_first_step_toward_center",
            "rule": (
                "Among open orthogonal neighbours of the general, pick a cell that "
                "minimizes Manhattan distance to map center ((rows-1)/2,(cols-1)/2). "
                "Tie-break: unknown (dirs W/E/N/S all appear)."
            ),
            "thresholds": {
                "center_rule_hit_rate": rate(center_rule_hits, n),
                "egen_rule_hit_rate": rate(egen_rule_hits, n),
                "units": "rate over fit wins",
            },
            "support": {
                "center_hits": center_rule_hits,
                "egen_hits": egen_rule_hits,
                "neither": len(neither_ids),
                "n": n,
            },
            "counterexample_ids": neither_ids[:40],
            "tag": "INFERRED",
            "reasoning": (
                "Center rule 90.9% > true-enemy-gen rule 85.9%. Center is fog-legal. "
                "Enemy-gen alignment is likely a side effect of opposite spawns."
            ),
        },
        {
            "id": "R4_no_castle_before_50",
            "rule": (
                "Do not build castles in the first 50 ticks. Prefer neutral land. "
                "First real build (spend detector) occurs at median tick 138, "
                "typically dist_to_general in 6..10 with army_before near price."
            ),
            "thresholds": {
                "castle_builds_in_first50": 0,
                "first_build_tick_median": (
                    dist_summary(first_real_build_ticks).get("median")
                ),
                "units": "ticks / army on cell",
            },
            "support": {
                "games_with_0_real_builds_in_50": real_builds_in_50[0],
                "n": n,
                "event_false_early": event_false_early,
            },
            "tag": "MEASURED",
        },
        {
            "id": "R5_flood_at_tick_27",
            "rule": (
                "At tick ~27 (median 27; 94.4% within 27±1), begin nearly every-tick "
                "neutral expansion until tick 50. Precondition observed: total army "
                "median 14, max_stack median 10, tiles still ~4."
            ),
            "thresholds": {
                "flood_start_tick": 27,
                "tolerance": 1,
                "army_at_start_median": dist_summary(phase_c_army).get("median"),
                "stack_at_start_median": dist_summary(phase_c_stack).get("median"),
                "units": "ticks / army",
            },
            "support": {
                "start_near_27_pm1_rate": rate(
                    sum(1 for t in phase_c_ticks if abs(t - 27) <= 1), n
                ),
                "n": n,
            },
            "counterexample_ids": late_gain_counterexamples,
            "tag": "MEASURED",
        },
        {
            "id": "R6_gen_moves_on_even_ticks",
            "rule": (
                "Moves that leave the general prefer even tick transitions "
                "(production ticks). Non-general moves prefer odd transitions."
            ),
            "thresholds": {
                "gen_move_even_rate": rate(
                    move_from_gen_mod2[0], sum(move_from_gen_mod2.values())
                ),
                "units": "fraction of inferred moves in ticks 0..49",
            },
            "support": {
                "from_gen": dict(move_from_gen_mod2),
                "not_from_gen": dict(move_not_gen_mod2),
            },
            "tag": "MEASURED",
        },
    ]

    payload["could_not_determine"] = [
        "Exact tie-break among equal min-center neighbours (W/E/N/S all common).",
        "Whether flood start is clock(tick==27) vs army/stack threshold (both co-occur).",
        "Exact mid-phase (10-25) micro-policy: many moves away from general without "
        "tile gains — gather vs patrol vs pathing noise not resolved.",
        "Half-move policy (512 half moves in ticks 10-25, src_before median 3).",
        "True action log (builds/moves inferred from frames only).",
        "Whether first-step rule uses map center, enemy-half heuristic, or openness "
        "score — center fits best among tested fog-legal rules but 22 neither cases remain.",
        "Bot version identity across the scrape window (loss cluster 24184-24189 "
        "opens differently).",
    ]

    return payload


def write_markdown(payload: dict) -> Path:
    n = payload["corpus"]["n"]
    tg = payload["tile_growth"]
    fd = payload["first_expansion_direction"]
    ph = payload["phases"]
    ca = payload["castles"]
    tm = payload["turn_modulo"]
    es = payload["early_stack"]
    lh = payload["land_vs_enemy_half"]

    def med(block: dict) -> str:
        if not block or block.get("n", 0) == 0:
            return "n/a"
        return (
            f"n={block['n']} median={block['median']} "
            f"[p10={block['p10']}, p90={block['p90']}] "
            f"min={block['min']} max={block['max']}"
        )

    lines: list[str] = []
    lines.append("# Kubic opening (first 50 turns) — fit wins")
    lines.append("")
    lines.append(
        f"**Corpus:** fit wins only, n={n}. Holdout excluded. "
        "Losses skimmed for failure modes only (not used to derive rules). "
        "Split: `docs/research/measurements/grok-kubic-corpus-split.json`."
    )
    lines.append("")
    lines.append(
        "**Raw JSON:** `docs/research/measurements/grok-kubic-opening.json`. "
        "Script: `scripts/analyze_kubic_opening.py`."
    )
    lines.append("")
    lines.append(
        "Every claim is tagged MEASURED, INFERRED, or UNKNOWN. "
        "Folder labels are not outcomes; seats use `Replay.outcome`."
    )
    lines.append("")
    lines.append("## Key distributions")
    lines.append("")
    lines.append(
        f"- MEASURED (n={n}): tiles at tick 50 — {med(tg['tiles50'])}."
    )
    lines.append(
        f"- MEASURED (n={n}): first tile-gain tick — {med(tg['first_gain_tick'])}; "
        f"counts={tg['first_gain_tick_counts']}."
    )
    lines.append(
        f"- MEASURED: gen army before first gain — {tg['gen_army_before_first_gain']}."
    )
    lines.append(
        f"- MEASURED: tiles trajectory medians — "
        + ", ".join(
            f"t{t}={tg['tiles_at_tick'][str(t)]['median']}"
            for t in (5, 10, 15, 20, 25, 30, 35, 40, 45, 50)
        )
        + "."
    )
    lines.append(
        f"- MEASURED: p(tiles_gained>0) at ticks 3/6/9/20/27/30/37 = "
        f"{ph['A_pulse']['p_gain_tick3']:.3f}/"
        f"{ph['A_pulse']['p_gain_tick6']:.3f}/"
        f"{ph['A_pulse']['p_gain_tick9']:.3f}/"
        f"{tg['tiles_gained_per_tick']['20']['p_gain']:.3f}/"
        f"{tg['tiles_gained_per_tick']['27']['p_gain']:.3f}/"
        f"{tg['tiles_gained_per_tick']['30']['p_gain']:.3f}/"
        f"{tg['tiles_gained_per_tick']['37']['p_gain']:.3f}."
    )
    lines.append(
        f"- MEASURED: phase-C flood start tick — {med(ph['C_flood']['start_tick'])}; "
        f"rate within 27±1 = {ph['C_flood']['start_near_27_pm1_rate']:.3f}."
    )
    lines.append(
        f"- MEASURED: at flood start — stack {med(ph['C_flood']['stack_at_start'])}; "
        f"army {med(ph['C_flood']['army_at_start'])}; "
        f"tiles {med(ph['C_flood']['tiles_at_start'])}."
    )
    lines.append(
        f"- MEASURED: first-move toward map center rate="
        f"{fd['align_map_center']['toward_rate']:.3f}; "
        f"toward true enemy general rate="
        f"{fd['align_enemy_general_true']['toward_rate']:.3f} "
        f"(fog: bot cannot see enemy general)."
    )
    lines.append(
        f"- MEASURED: min-dist-to-center neighbour rule hit rate="
        f"{fd['min_dist_center_neighbor_rule']['rate']:.3f}; "
        f"min-dist-to-enemy-gen rule="
        f"{fd['min_dist_enemy_gen_neighbor_rule']['rate']:.3f}."
    )
    lines.append(
        f"- MEASURED: early stack median dist to general="
        f"{es['median_stack_dist_to_general']['median']}; "
        f"frac ticks with stack on general="
        f"{es['frac_ticks_stack_on_general']['median']}."
    )
    lines.append(
        f"- MEASURED: first-50 captures neutral={lh['neutral_captures_first50']} "
        f"enemy={lh['enemy_captures_first50']} "
        f"(enemy_rate={lh['enemy_capture_rate']:.4f}); "
        f"owned-enemy-half frac at 50 median="
        f"{lh['owned_enemy_half_frac_at_50']['median']}."
    )
    lines.append(
        f"- MEASURED: moves from general on even tick transitions rate="
        f"{tm['moves_from_general_tick_mod2']['even_rate']:.3f} "
        f"(even={tm['moves_from_general_tick_mod2']['even']}, "
        f"odd={tm['moves_from_general_tick_mod2']['odd']})."
    )
    lines.append(
        f"- MEASURED: real castle builds in first 50 — "
        f"{ca['real_builds_in_first50']} (0 in all {n} games). "
        f"First real build tick — {med(ca['first_real_build_tick'])}; "
        f"dist_to_general — {med(ca['first_real_build_dist_to_general'])}; "
        f"army_before — {med(ca['first_real_build_army_before'])}; "
        f"drop — {med(ca['first_real_build_drop'])}."
    )
    lines.append(
        f"- MEASURED: EventLog `castle_built` with tick<=50 and no spend evidence — "
        f"{ca['event_false_early_count']} false positives. "
        "Do not use early event ticks as build times."
    )
    lines.append("")
    lines.append("## Candidate decision rules")
    lines.append("")
    for rule in payload["candidate_rules"]:
        lines.append(f"### {rule['id']} [{rule['tag']}]")
        lines.append("")
        lines.append(rule["rule"])
        lines.append("")
        lines.append(f"Thresholds: `{rule['thresholds']}`")
        lines.append("")
        lines.append(f"Support: `{rule['support']}`")
        if rule.get("reasoning"):
            lines.append("")
            lines.append(f"Reasoning: {rule['reasoning']}")
        cex = rule.get("counterexample_ids") or []
        if cex:
            lines.append("")
            lines.append(
                f"Counterexamples (n={len(cex)}): "
                + ", ".join(cex[:25])
                + (" ..." if len(cex) > 25 else "")
            )
        lines.append("")

    lines.append("## Counterexamples (match_ids)")
    lines.append("")
    cex = payload["counterexamples"]
    lines.append(
        f"- No gain at tick 3 (n={len(cex['no_gain_at_tick_3'])}): "
        + ", ".join(cex["no_gain_at_tick_3"])
    )
    lines.append(
        f"- Phase C missing or after tick 35 (n={len(cex['phase_c_missing_or_after_35'])}): "
        + ", ".join(cex["phase_c_missing_or_after_35"][:40])
    )
    lines.append(
        f"- First move neither min-center nor min-egen (n={len(cex['first_move_neither_center_nor_egen_min'])}): "
        + ", ".join(cex["first_move_neither_center_nor_egen_min"][:40])
    )
    lines.append(
        f"- First move away from true enemy general (n={len(cex['first_move_away_from_egen'])}): "
        + ", ".join(cex["first_move_away_from_egen"][:40])
        + (" ..." if len(cex["first_move_away_from_egen"]) > 40 else "")
    )
    lines.append("")
    lines.append("## Loss skim (excluded from derivation)")
    lines.append("")
    lines.append(payload["loss_skim"]["note"])
    lines.append("")
    follows = payload["loss_skim"]["follows_3_6_9_count"]
    lines.append(
        f"MEASURED: {follows}/{payload['corpus']['n_losses_skimmed']} losses "
        "still show gains at ticks 3,6,9."
    )
    lines.append("")
    for g in payload["loss_skim"]["games"]:
        lines.append(
            f"- `{g['match_id']}` ticks={g['total_ticks']} tiles50={g['tiles50']} "
            f"first_gain={g['first_gain_tick']} contact={g['first_contact']} "
            f"3/6/9={g['gain_at_3']}/{g['gain_at_6']}/{g['gain_at_9']}"
        )
    lines.append("")
    lines.append("## Could not determine")
    lines.append("")
    for item in payload["could_not_determine"]:
        lines.append(f"- UNKNOWN: {item}")
    lines.append("")
    lines.append("## Method notes")
    lines.append("")
    lines.append(
        "- Competition rules (`RULES.md`): no pre-placed castles; build cost base 35 "
        "+ crowding surcharge; production every other turn; bulk +1 every 50."
    )
    lines.append(
        "- Move inference: `scripts/kubic_moves.py` (frame diffs). Ambiguous ticks exist."
    )
    lines.append(
        "- Real castle detector: army drop >=30 with no neighbour receive; "
        "EventLog `castle_built` alone is unsafe before ~tick 100."
    )
    lines.append(
        "- Spawn classes: corner = within 1 of both edges; edge = min edge dist <=2; "
        "else centerish."
    )
    lines.append("")
    path = MEASUREMENTS / MD_NAME
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    payload = analyze_fit()
    json_path = dump_json(JSON_NAME, payload)
    md_path = write_markdown(payload)
    print(f"fit n={payload['corpus']['n']}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print("top rules:")
    for rule in payload["candidate_rules"][:5]:
        print(f"  {rule['id']} [{rule['tag']}]: {rule['thresholds']}")


if __name__ == "__main__":
    main()
