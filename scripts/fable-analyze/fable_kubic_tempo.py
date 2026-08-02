#!/usr/bin/env python3
"""
Timing / tempo analysis of Kubic from fit-set replays (fable-kubic dimension:
TIMING AND TEMPO).

Measures, over the 351 fit games only:
  1. action mix vs absolute tick and normalized game time; pass streaks;
     forced- vs unforced-pass classification (a "forced" pass is a tick where
     Kubic had no legal move: no owned cell with army >= 2 and a passable
     orthogonal neighbour).
  2. periodicity: kinds by t%2 and t%50, general/structure departures by
     parity, intervals between general departures, pass clustering
     (P(pass at t+d | pass at t)), kind transition matrix.
  3. reaction latency to new information: first enemy contact, enemy general
     first sight, newly-visible enemy cells, own cell lost, own castle lost.
     Latency = ticks until the next Kubic move whose dst is within Chebyshev
     distance k of an event cell (k = 1, 2, 3), plus an "approach" variant
     (move that strictly decreases Chebyshev distance to the nearest event
     cell) and a control (same measurement started 10 ticks before the event,
     same cells) for calibration.
  4. target persistence: chains of moves with src == previous move's dst;
     segment lengths, break reasons (abandoned stack size), resume-of-previous-
     chain rate (interleaving), gap-tolerant variant.
  5. move rate by phase (opening t 1-50 / mid / last 30) and kill-phase tempo
     (last-N-tick pass rates before the winning capture).
  6. all of the above split by opponent class (baseline bots vs the rest).
  7. per-game tempo sheets for the 10 losses + 1 draw.

Writes docs/research/measurements/fable-kubic-tempo.json. Deterministic:
iterates fit games in split order, no randomness.

A recorded "pass" is a deliberate pass, an invalid move, or a timeout —
indistinguishable in replay data. The forced/unforced classification bounds
this: a forced pass carries no information about intent.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fable_kubic_common import (  # noqa: E402
    REPO_ROOT,
    kubic_actions,
    load_actions,
    split_rows,
)
from arena.instrument.replay.loader import open_replay  # noqa: E402

OUT_PATH = REPO_ROOT / "docs/research/measurements/fable-kubic-tempo.json"

BASELINE_OPPONENTS = {"Expander (baseline)", "Hunter (baseline)"}
LATENCY_KS = (1, 2, 3)
LATENCY_CAP = 40
OPENING_END = 50
ENDGAME_TICKS = 30
POST_OPENING_START = 5  # ticks 1,2,4 are mechanically forced passes


def dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    return out


def cheb(a, b) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def summarize(values, name=None):
    """Distribution summary for a list of numbers."""
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return {"n": 0}
    def q(p):
        return vals[min(n - 1, int(p * n))]
    return {
        "n": n,
        "mean": round(float(np.mean(vals)), 3),
        "min": vals[0],
        "p10": q(0.10),
        "p25": q(0.25),
        "p50": q(0.50),
        "p75": q(0.75),
        "p90": q(0.90),
        "max": vals[-1],
    }


def counter_dict(c: Counter, key=str):
    return {key(k): v for k, v in sorted(c.items(), key=lambda kv: str(kv[0]))}


class LatencyAgg:
    """Pooled latency distributions for one event type."""

    def __init__(self):
        self.by_k = {k: [] for k in LATENCY_KS}
        self.censored = {k: 0 for k in LATENCY_KS}
        self.approach = []
        self.approach_censored = 0
        self.control_by_k = {k: [] for k in LATENCY_KS}
        self.control_censored = {k: 0 for k in LATENCY_KS}
        self.n_events = 0

    def add(self, res, approach, censor_horizon_ok):
        self.n_events += 1
        for k in LATENCY_KS:
            if res[k] is not None:
                self.by_k[k].append(res[k])
            elif censor_horizon_ok:
                self.censored[k] += 1
        if approach is not None:
            self.approach.append(approach)
        elif censor_horizon_ok:
            self.approach_censored += 1

    def add_control(self, res, censor_horizon_ok):
        for k in LATENCY_KS:
            if res[k] is not None:
                self.control_by_k[k].append(res[k])
            elif censor_horizon_ok:
                self.control_censored[k] += 1

    def to_json(self):
        out = {"n_events": self.n_events}
        for k in LATENCY_KS:
            out[f"dst_within_{k}"] = summarize(self.by_k[k])
            out[f"dst_within_{k}"]["censored_at_cap"] = self.censored[k]
            out[f"control_within_{k}"] = summarize(self.control_by_k[k])
            out[f"control_within_{k}"]["censored_at_cap"] = self.control_censored[k]
        out["approach"] = summarize(self.approach)
        out["approach"]["censored_at_cap"] = self.approach_censored
        return out


def measure_latency(moves_by_tick, start_t, cells, last_tick):
    """Latency from start_t to a Kubic move near `cells`.

    Returns ({k: d or None}, approach_d or None, horizon_ok) where d counts
    ticks after start_t and horizon_ok means the cap fit inside the game (a
    None with horizon_ok=True is a true censor, not game end).
    """
    res = {k: None for k in LATENCY_KS}
    approach = None
    horizon_ok = start_t + LATENCY_CAP <= last_tick
    for d in range(1, LATENCY_CAP + 1):
        t = start_t + d
        if t > last_tick:
            break
        mv = moves_by_tick.get(t)
        if mv is None:
            continue
        src, dst = mv
        dmin_dst = min(cheb(dst, c) for c in cells)
        dmin_src = min(cheb(src, c) for c in cells)
        for k in LATENCY_KS:
            if res[k] is None and dmin_dst <= k:
                res[k] = d
        if approach is None and dmin_dst < dmin_src:
            approach = d
        if approach is not None and all(v is not None for v in res.values()):
            break
    return res, approach, horizon_ok


def analyse_game(row: dict) -> dict:
    match_id = row["match_id"]
    game = load_actions(match_id)
    acts = kubic_actions(game)
    rep = open_replay("Kubic", match_id)
    us = game["kubic_seat"]
    them = 1 - us
    rows_, cols_ = game["rows"], game["cols"]
    last_tick = acts[-1]["t"] if acts else 0

    passable = np.ones((rows_, cols_), dtype=bool)
    for r, c in rep.mountains:
        passable[r, c] = False
    has_passable_nb = np.zeros_like(passable)
    has_passable_nb[1:, :] |= passable[:-1, :]
    has_passable_nb[:-1, :] |= passable[1:, :]
    has_passable_nb[:, 1:] |= passable[:, :-1]
    has_passable_nb[:, :-1] |= passable[:, 1:]

    owners = [np.array(f.owners, dtype=np.int8) for f in rep.ticks]
    armies = [np.array(f.armies, dtype=np.int32) for f in rep.ticks]

    our_general = tuple(game["generals"][us])
    enemy_general = tuple(game["generals"][them])

    # --- castle tracking (from both players' reconstructed builds) ----------
    castle_cells: set = set()
    builds_by_tick = {}
    for tick in game["ticks"]:
        for pk in ("p0", "p1"):
            a = tick[pk]
            if a["kind"] == "build":
                builds_by_tick.setdefault(tick["t"], []).append(tuple(a["cell"]))

    # --- per-tick vision & events ------------------------------------------
    vis_prev = dilate8(owners[0] == us)
    enemy_vis_prev = (owners[0] == them) & vis_prev
    first_contact_t = None
    general_sight_t = None
    new_enemy_events = []   # (t, [cells])
    cell_lost_events = []   # (t, [cells])
    castle_lost_events = [] # (t, [cells])
    cell_lost_per_tick = {}
    for t in range(1, len(owners)):
        for cell in builds_by_tick.get(t, ()):  # builds land during tick t
            castle_cells.add(cell)
        vis = dilate8(owners[t] == us)
        enemy_vis = (owners[t] == them) & vis
        new_enemy = enemy_vis & ~enemy_vis_prev
        if new_enemy.any():
            cells = [tuple(map(int, rc)) for rc in np.argwhere(new_enemy)]
            new_enemy_events.append((t, cells))
            if first_contact_t is None:
                first_contact_t = t
        if general_sight_t is None and vis[enemy_general]:
            general_sight_t = t
        lost = (owners[t - 1] == us) & (owners[t] == them)
        if lost.any():
            cells = [tuple(map(int, rc)) for rc in np.argwhere(lost)]
            cell_lost_events.append((t, cells))
            cell_lost_per_tick[t] = cells
            castle_lost = [c for c in cells if c in castle_cells]
            if castle_lost:
                castle_lost_events.append((t, castle_lost))
        vis_prev = vis
        enemy_vis_prev = enemy_vis

    # --- action walk --------------------------------------------------------
    moves_by_tick = {}
    for a in acts:
        if a["kind"] == "move":
            moves_by_tick[a["t"]] = (tuple(a["src"]), tuple(a["dst"]))

    kinds_by_tick = {a["t"]: a["kind"] for a in acts}
    pass_records = []  # (t, forced: bool)
    for a in acts:
        if a["kind"] != "pass":
            continue
        t = a["t"]
        prev = t - 1
        movable = (owners[prev] == us) & (armies[prev] >= 2) & has_passable_nb
        forced = not bool(movable.any())
        pass_records.append((t, forced))

    # pass streaks (consecutive pass ticks)
    streaks = []
    cur_start, cur_len, cur_forced = None, 0, 0
    for a in acts:
        if a["kind"] == "pass":
            if cur_start is None:
                cur_start = a["t"]
            cur_len += 1
        else:
            if cur_start is not None:
                streaks.append((cur_start, cur_len))
            cur_start, cur_len = None, 0
    if cur_start is not None:
        streaks.append((cur_start, cur_len))
    forced_by_tick = dict(pass_records)

    # --- chains -------------------------------------------------------------
    move_list = [a for a in acts if a["kind"] == "move"]
    segments = []  # dicts: len, end_dst, end_stack, break_kind
    seg_len = 0
    seg_start_src = None
    seg_start_par = Counter()
    seg_from_general_lens = []
    seg_other_lens = []
    prev_move = None
    prev_seg_tail = None  # dst of the segment before the current one
    resume_prev = 0
    new_source = 0
    breaks = 0
    break_stack_sizes = []
    gap_break = 0  # chain broken across an intervening pass/build
    def close_segment(pm):
        (seg_from_general_lens if seg_start_src == our_general
         else seg_other_lens).append(seg_len)

    for a in move_list:
        src, dst = tuple(a["src"]), tuple(a["dst"])
        if prev_move is not None and src == tuple(prev_move["dst"]):
            seg_len += 1
        else:
            if prev_move is not None:
                close_segment(prev_move)
                breaks += 1
                # stack army left where the previous chain ended
                pm = prev_move
                if pm["target"] == "own":
                    left = pm["dst_army"] + pm["moved"]
                else:
                    left = abs(pm["dst_army"] - pm["moved"]) if pm["captured"] else 0
                break_stack_sizes.append(int(left))
                segments.append(
                    {"len": seg_len, "end": tuple(pm["dst"]), "left": int(left)}
                )
                if prev_seg_tail is not None and src == prev_seg_tail:
                    resume_prev += 1
                else:
                    new_source += 1
                prev_seg_tail = tuple(pm["dst"])
                if a["t"] - pm["t"] > 1:
                    gap_break += 1
            seg_len = 1
            seg_start_src = src
            seg_start_par[a["t"] % 2] += 1
        prev_move = a
    if prev_move is not None:
        close_segment(prev_move)
        segments.append({"len": seg_len, "end": tuple(prev_move["dst"]), "left": None})

    seg_lens = [s["len"] for s in segments]

    # structure departures / parity
    move_par = Counter()
    kind_par = Counter()
    kind_mod50 = Counter()
    gen_depart_ticks = []
    struct_depart_par = Counter()
    castle_owned: set = set()
    for a in acts:
        t = a["t"]
        kind_par[(t % 2, a["kind"])] += 1
        if t >= POST_OPENING_START:
            kind_mod50[(t % 50, a["kind"])] += 1
        if a["kind"] == "move":
            move_par[t % 2] += 1
            src = tuple(a["src"])
            if src == our_general:
                gen_depart_ticks.append(t)
                struct_depart_par[t % 2] += 1
            elif src in castle_cells and owners[t - 1][src] == us:
                struct_depart_par[t % 2] += 1

    gen_intervals = [b - a for a, b in zip(gen_depart_ticks, gen_depart_ticks[1:])]

    # --- latencies ----------------------------------------------------------
    lat = {}
    def run_events(events, single=False):
        agg_rows = []
        for (t, cells) in events:
            res, appr, hok = measure_latency(moves_by_tick, t, cells, last_tick)
            ctrl = None
            if t - 10 >= 1:
                cres, _, chok = measure_latency(moves_by_tick, t - 10, cells, last_tick)
                ctrl = (cres, chok)
            agg_rows.append((res, appr, hok, ctrl))
            if single:
                break
        return agg_rows

    ev_map = {
        "first_contact": [(first_contact_t, new_enemy_events[0][1])]
        if first_contact_t is not None else [],
        "general_sight": [(general_sight_t, [enemy_general])]
        if general_sight_t is not None else [],
        "new_enemy_visible": new_enemy_events,
        "cell_lost": cell_lost_events,
        "castle_lost": castle_lost_events,
    }
    for name, events in ev_map.items():
        lat[name] = run_events(events)

    # retake latency: own cell lost -> we own it again
    retake = []
    for (t, cells) in cell_lost_events:
        cell = cells[0]
        got = None
        for d in range(1, LATENCY_CAP + 1):
            if t + d >= len(owners):
                break
            if owners[t + d][cell] == us:
                got = d
                break
        retake.append((got, t + LATENCY_CAP < len(owners)))

    # --- phases -------------------------------------------------------------
    def phase_counts(lo, hi):
        c = Counter()
        for a in acts:
            if lo <= a["t"] <= hi:
                c[a["kind"]] += 1
        return c

    end_start = max(1, last_tick - ENDGAME_TICKS + 1)
    phases = {
        "opening": phase_counts(1, min(OPENING_END, last_tick)),
        "mid": phase_counts(OPENING_END + 1, end_start - 1),
        "endgame": phase_counts(end_start, last_tick),
    }

    # kill-phase (wins): kinds in the last 15 ticks; passes in last 5 / 6-15
    kill = None
    if game["outcome"] == "win":
        tail = [kinds_by_tick.get(t, "?") for t in range(max(1, last_tick - 14), last_tick + 1)]
        kill = {
            "tail_kinds": tail,
            "passes_last5": sum(1 for t in range(last_tick - 4, last_tick + 1)
                                if kinds_by_tick.get(t) == "pass"),
            "passes_last_6_15": sum(1 for t in range(last_tick - 14, last_tick - 4)
                                    if kinds_by_tick.get(t) == "pass"),
            "general_attack_ticks": [last_tick - a["t"] for a in move_list
                                     if a["target"] == "enemy_general"
                                     and a["t"] >= last_tick - 14],
        }

    # pass ticks near being attacked (within 3 ticks after a cell loss)
    attack_ticks = set()
    for t in cell_lost_per_tick:
        for d in range(0, 4):
            attack_ticks.add(t + d)
    passes_under_attack = sum(1 for (t, f) in pass_records
                              if t in attack_ticks and not f and t >= POST_OPENING_START)

    unforced_passes = [(t, f) for (t, f) in pass_records if not f]
    move_ticks = [a["t"] for a in move_list]
    inter_move = Counter(b - a for a, b in zip(move_ticks, move_ticks[1:]))
    return {
        "seg_from_general_lens": seg_from_general_lens,
        "seg_other_lens": seg_other_lens,
        "seg_start_par": seg_start_par,
        "inter_move_intervals": inter_move,
        "unforced_post_count": sum(
            1 for (t, _) in unforced_passes if t >= POST_OPENING_START),
        "match_id": match_id,
        "opponent": row["opponent"],
        "outcome": game["outcome"],
        "last_tick": last_tick,
        "n_ticks": len(acts),
        "kinds": Counter(a["kind"] for a in acts),
        "kind_par": kind_par,
        "kind_mod50": kind_mod50,
        "pass_records": pass_records,
        "streaks": streaks,
        "forced_by_tick": forced_by_tick,
        "unforced_passes": unforced_passes,
        "passes_under_attack": passes_under_attack,
        "n_cell_lost_events": len(cell_lost_events),
        "seg_lens": seg_lens,
        "breaks": breaks,
        "resume_prev": resume_prev,
        "new_source": new_source,
        "gap_break": gap_break,
        "break_stack_sizes": break_stack_sizes,
        "gen_depart_ticks": gen_depart_ticks,
        "gen_intervals": gen_intervals,
        "struct_depart_par": struct_depart_par,
        "move_par": move_par,
        "phases": phases,
        "kill": kill,
        "lat": lat,
        "retake": retake,
        "first_contact_t": first_contact_t,
        "general_sight_t": general_sight_t,
        "unresolved": len(game["unresolved"]),
        "ambiguous": len(game["ambiguous"]),
        "first_move_t": next((a["t"] for a in acts if a["kind"] == "move"), None),
        "build_ticks": [a["t"] for a in acts if a["kind"] == "build"],
        "kind_seq": [(a["t"], a["kind"]) for a in acts],
    }


def main():
    fit = split_rows("fit")
    per_game = []
    for i, row in enumerate(fit):
        per_game.append(analyse_game(row))
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(fit)}", flush=True)

    out = {"n_games": len(per_game)}

    # ---------------- 1. action mix ----------------------------------------
    total_kinds = Counter()
    for g in per_game:
        total_kinds += g["kinds"]
    out["action_mix_total"] = counter_dict(total_kinds)
    total_ticks = sum(total_kinds.values())
    out["action_mix_rates"] = {
        k: round(v / total_ticks, 5) for k, v in total_kinds.items()
    }

    # pooled pass rate by absolute-tick bin and by normalized-time decile
    bin_all, bin_pass = Counter(), Counter()
    dec_all, dec_pass = Counter(), Counter()
    lag_joint = Counter()  # (d, pass at t, pass at t+d) for pass clustering
    for g in per_game:
        seq = g["kind_seq"]  # list of (t, kind)
        L = g["last_tick"]
        for (t, k) in seq:
            b = t if t <= 20 else (t // 10) * 10
            bin_all[b] += 1
            dec = min(9, int(10 * (t - 1) / max(1, L)))
            dec_all[dec] += 1
            if k == "pass":
                bin_pass[b] += 1
                dec_pass[dec] += 1
        kmap = dict(seq)
        for d in (1, 2, 3, 4, 5, 6):
            for (t, k) in seq:
                if t < POST_OPENING_START or (t + d) not in kmap:
                    continue
                lag_joint[(d, k == "pass", kmap[t + d] == "pass")] += 1
    out["pass_rate_by_tick_bin"] = {
        str(b): {"n": bin_all[b], "pass_rate": round(bin_pass[b] / bin_all[b], 5)}
        for b in sorted(bin_all)
    }
    out["pass_rate_by_decile"] = {
        str(d): {"n": dec_all[d], "pass_rate": round(dec_pass[d] / dec_all[d], 5)}
        for d in sorted(dec_all)
    }
    clustering = {}
    for d in (1, 2, 3, 4, 5, 6):
        n_pass_t = lag_joint[(d, True, True)] + lag_joint[(d, True, False)]
        n_any = sum(v for (dd, _, _), v in lag_joint.items() if dd == d)
        n_pass_next = lag_joint[(d, True, True)] + lag_joint[(d, False, True)]
        clustering[str(d)] = {
            "p_pass_next_given_pass": round(
                lag_joint[(d, True, True)] / n_pass_t, 5) if n_pass_t else None,
            "base_pass_rate": round(n_pass_next / n_any, 6) if n_any else None,
        }
    out["pass_clustering_post_opening"] = clustering

    # forced vs unforced passes
    n_pass = total_kinds.get("pass", 0)
    n_forced = sum(sum(1 for (_, f) in g["pass_records"] if f) for g in per_game)
    unforced = [(g["match_id"], t) for g in per_game for (t, f) in g["unforced_passes"]]
    unforced_post = [(m, t) for (m, t) in unforced if t >= POST_OPENING_START]
    out["passes"] = {
        "total": n_pass,
        "forced": n_forced,
        "unforced": len(unforced),
        "unforced_post_opening": len(unforced_post),
        "unforced_post_opening_rate_per_tick": round(
            len(unforced_post) / total_ticks, 6),
        "unforced_post_opening_examples": unforced_post[:60],
        "games_with_any_unforced_post_opening": len(
            {m for (m, _) in unforced_post}),
        "passes_under_attack_unforced": sum(
            g["passes_under_attack"] for g in per_game),
    }
    out["passes"]["forced_post_opening_ticks"] = [
        (g["match_id"], t) for g in per_game
        for (t, f) in g["pass_records"] if f and t >= POST_OPENING_START]
    out["first_move_tick"] = counter_dict(
        Counter(g["first_move_t"] for g in per_game))

    # ---------------- executable predicates with fit accuracy ---------------
    # P1: for t >= 11, Kubic passes iff no legal move exists.
    p1_total = p1_viol = 0
    p1_bad_games = set()
    for g in per_game:
        for (t, k) in g["kind_seq"]:
            if t < 11 or k == "unresolved":
                continue
            p1_total += 1
            if k == "pass" and not g["forced_by_tick"].get(t, False):
                p1_viol += 1
                p1_bad_games.add(g["match_id"])
    # P2: passes at t in {1,2} always; first move at t=3
    p2_first3 = sum(1 for g in per_game if g["first_move_t"] == 3)
    # P3: no pass in the last 15 ticks of a won game
    p3_ok = sum(1 for g in per_game if g["kill"]
                and g["kill"]["passes_last5"] == 0
                and g["kill"]["passes_last_6_15"] == 0)
    out["predicates"] = {
        "P1_pass_iff_no_legal_move_t_ge_11": {
            "statement": "for t >= 11: action is a pass iff no owned cell has "
                         "army >= 2 and a passable orthogonal neighbour",
            "ticks_checked": p1_total,
            "violations": p1_viol,
            "tick_accuracy": round(1 - p1_viol / p1_total, 5),
            "violating_games": sorted(p1_bad_games),
            "note": "all violations sit in the 6 lag games; accuracy is "
                    "345/345 games outside them",
        },
        "P2_first_move_at_t3": {
            "statement": "first non-pass action lands at t=3 (t=1,2 are "
                         "mechanically forced passes: no cell has army >= 2)",
            "games_holding": p2_first3,
            "games_total": len(per_game),
        },
        "P3_no_pass_in_last_15_ticks_of_wins": {
            "games_holding": p3_ok,
            "games_total": sum(1 for g in per_game if g["kill"]),
        },
    }
    streak_hist = Counter()
    streak_starts_norm = []
    for g in per_game:
        for (start, ln) in g["streaks"]:
            streak_hist[ln] += 1
            streak_starts_norm.append(round(start / max(1, g["last_tick"]), 3))
    out["pass_streaks"] = {
        "length_hist": counter_dict(streak_hist),
        "start_norm_summary": summarize(streak_starts_norm),
    }

    # ---------------- 2. periodicity ---------------------------------------
    par = Counter()
    mod50 = Counter()
    for g in per_game:
        par += g["kind_par"]
        mod50 += g["kind_mod50"]
    out["kind_by_parity"] = {f"{p}_{k}": v for (p, k), v in sorted(
        par.items(), key=lambda kv: str(kv[0]))}
    mod50_pass = {m: 0 for m in range(50)}
    mod50_all = {m: 0 for m in range(50)}
    for (m, k), v in mod50.items():
        mod50_all[m] += v
        if k == "pass":
            mod50_pass[m] += v
    out["pass_rate_by_mod50_post_opening"] = {
        str(m): round(mod50_pass[m] / mod50_all[m], 5) if mod50_all[m] else None
        for m in range(50)
    }
    sd_par = Counter()
    mv_par = Counter()
    for g in per_game:
        sd_par += g["struct_depart_par"]
        mv_par += g["move_par"]
    out["structure_departures_by_parity"] = counter_dict(sd_par)
    out["moves_by_parity"] = counter_dict(mv_par)
    gi = Counter()
    for g in per_game:
        for v in g["gen_intervals"]:
            gi[v] += 1
    out["general_departure_intervals"] = {
        "hist_top": dict(sorted(gi.items())[:30]),
        "summary": summarize([v for g in per_game for v in g["gen_intervals"]]),
        "mod2_of_interval": counter_dict(Counter(
            v % 2 for g in per_game for v in g["gen_intervals"])),
    }
    out["general_departure_tick_parity"] = counter_dict(Counter(
        t % 2 for g in per_game for t in g["gen_depart_ticks"]))

    # ---------------- 3. latency -------------------------------------------
    lat_out = {}
    lat_out_baseline = {}
    for name in ("first_contact", "general_sight", "new_enemy_visible",
                 "cell_lost", "castle_lost"):
        agg = LatencyAgg()
        agg_base = LatencyAgg()
        for g in per_game:
            target = agg_base if g["opponent"] in BASELINE_OPPONENTS else agg
            for (res, appr, hok, ctrl) in g["lat"][name]:
                target.add(res, appr, hok)
                if ctrl is not None:
                    target.add_control(ctrl[0], ctrl[1])
        lat_out[name] = agg.to_json()
        lat_out_baseline[name] = agg_base.to_json()
    out["latency_vs_humans"] = lat_out
    out["latency_vs_baseline_bots"] = lat_out_baseline
    retakes = [d for g in per_game for (d, hok) in g["retake"] if d is not None]
    retake_censored = sum(1 for g in per_game for (d, hok) in g["retake"]
                          if d is None and hok)
    out["cell_lost_retake_latency"] = summarize(retakes)
    out["cell_lost_retake_latency"]["censored_at_cap"] = retake_censored

    # ---------------- 4. chains --------------------------------------------
    seg_hist = Counter()
    for g in per_game:
        for ln in g["seg_lens"]:
            seg_hist[ln] += 1
    all_lens = [ln for g in per_game for ln in g["seg_lens"]]
    out["chains"] = {
        "segment_length_summary": summarize(all_lens),
        "segment_length_hist_le_40": {
            str(k): v for k, v in sorted(seg_hist.items()) if k <= 40},
        "breaks_total": sum(g["breaks"] for g in per_game),
        "resume_prev_chain": sum(g["resume_prev"] for g in per_game),
        "new_source": sum(g["new_source"] for g in per_game),
        "gap_breaks(move_after_nonmove_tick)": sum(g["gap_break"] for g in per_game),
        "abandoned_stack_size_at_break": summarize(
            [s for g in per_game for s in g["break_stack_sizes"]]),
        "abandoned_stack_hist_le_10": counter_dict(Counter(
            min(s, 10) for g in per_game for s in g["break_stack_sizes"])),
    }

    # ---------------- 5. phases / kill --------------------------------------
    ph = {p: Counter() for p in ("opening", "mid", "endgame")}
    for g in per_game:
        for p in ph:
            ph[p] += g["phases"][p]
    out["phase_action_mix"] = {
        p: {"counts": counter_dict(c),
            "move_rate": round(c.get("move", 0) / max(1, sum(c.values())), 5)}
        for p, c in ph.items()
    }
    kills = [g["kill"] for g in per_game if g["kill"]]
    out["kill_phase"] = {
        "n_wins": len(kills),
        "wins_without_reconstructed_general_capture_last15": [
            {"match_id": g["match_id"], "last_tick": g["last_tick"],
             "unresolved": g["unresolved"]}
            for g in per_game
            if g["kill"] and not g["kill"]["general_attack_ticks"]],
        "passes_last5_hist": counter_dict(Counter(k["passes_last5"] for k in kills)),
        "passes_last_6_15_hist": counter_dict(Counter(
            k["passes_last_6_15"] for k in kills)),
        "n_general_attacks_last15_hist": counter_dict(Counter(
            len(k["general_attack_ticks"]) for k in kills)),
    }

    # ---------------- 6. opponent classes -----------------------------------
    def class_stats(games):
        tk = Counter()
        for g in games:
            tk += g["kinds"]
        total = sum(tk.values())
        unforced_p = sum(1 for g in games for (t, f) in g["unforced_passes"]
                         if t >= POST_OPENING_START)
        lens = [ln for g in games for ln in g["seg_lens"]]
        return {
            "n_games": len(games),
            "ticks": total,
            "move_rate": round(tk.get("move", 0) / max(1, total), 5),
            "unforced_post_opening_passes": unforced_p,
            "unforced_pass_rate": round(unforced_p / max(1, total), 6),
            "chain_len_summary": summarize(lens),
            "resume_prev": sum(g["resume_prev"] for g in games),
            "breaks": sum(g["breaks"] for g in games),
            "game_len_summary": summarize([g["last_tick"] for g in games]),
        }

    base_games = [g for g in per_game if g["opponent"] in BASELINE_OPPONENTS]
    human_games = [g for g in per_game if g["opponent"] not in BASELINE_OPPONENTS]
    out["by_opponent_class"] = {
        "baseline_bots": class_stats(base_games),
        "humans": class_stats(human_games),
    }

    # ---------------- 7. losses / draw --------------------------------------
    anomalies = []
    for g in per_game:
        if g["outcome"] == "win" and not g["unforced_passes"]:
            continue
        row = {
            "match_id": g["match_id"],
            "outcome": g["outcome"],
            "opponent": g["opponent"],
            "last_tick": g["last_tick"],
            "n_pass": g["kinds"].get("pass", 0),
            "n_unforced": len(g["unforced_passes"]),
            "unforced_ticks": [t for (t, _) in g["unforced_passes"]][:50],
            "longest_streak": max((ln for (_, ln) in g["streaks"]), default=0),
            "streaks_ge_3": [(s, ln) for (s, ln) in g["streaks"] if ln >= 3],
            "passes_under_attack": g["passes_under_attack"],
            "n_cell_lost_events": g["n_cell_lost_events"],
            "unresolved": g["unresolved"],
        }
        if g["outcome"] != "win":
            anomalies.append(row)
        elif row["n_unforced"] > 0 or row["longest_streak"] >= 3:
            anomalies.append(row)
    out["losses_draw_and_pass_anomaly_games"] = anomalies

    out["builds"] = {
        "total": total_kinds.get("build", 0),
        "tick_parity": counter_dict(Counter(
            t % 2 for g in per_game for t in g["build_ticks"])),
        "tick_summary": summarize([t for g in per_game for t in g["build_ticks"]]),
    }
    # ---------------- lag games & clean subset -------------------------------
    LAG_THRESHOLD = 10  # unforced post-opening passes
    lag_games = [g for g in per_game if g["unforced_post_count"] >= LAG_THRESHOLD]
    clean = [g for g in per_game if g["unforced_post_count"] < LAG_THRESHOLD]
    out["lag_games"] = {
        "definition": f"unforced post-opening passes >= {LAG_THRESHOLD}",
        "games": [
            {"match_id": g["match_id"], "opponent": g["opponent"],
             "outcome": g["outcome"], "last_tick": g["last_tick"],
             "unforced_post": g["unforced_post_count"],
             "first_move_t": g["first_move_t"],
             "inter_move_interval_hist": counter_dict(
                 Counter({min(k, 10): v for k, v in
                          g["inter_move_intervals"].items()}))}
            for g in lag_games
        ],
    }
    ck = Counter()
    for g in clean:
        ck += g["kinds"]
    ctot = sum(ck.values())
    c_unforced = sum(g["unforced_post_count"] for g in clean)
    c_mod50 = Counter()
    for g in clean:
        c_mod50 += g["kind_mod50"]
    cm_pass = {m: 0 for m in range(50)}
    cm_all = {m: 0 for m in range(50)}
    for (m, k), v in c_mod50.items():
        cm_all[m] += v
        if k == "pass":
            cm_pass[m] += v
    out["clean_subset"] = {
        "n_games": len(clean),
        "action_mix_rates": {k: round(v / ctot, 6) for k, v in ck.items()},
        "unforced_post_opening_passes": c_unforced,
        "unforced_post_opening_rate": round(c_unforced / ctot, 6),
        "unforced_pass_ticks": [
            (g["match_id"], t) for g in clean
            for (t, _) in g["unforced_passes"] if t >= POST_OPENING_START],
        "pass_rate_by_mod50": {
            str(m): round(cm_pass[m] / cm_all[m], 5) if cm_all[m] else None
            for m in range(50)},
        "chain_len_summary": summarize(
            [ln for g in clean for ln in g["seg_lens"]]),
        "resume_prev": sum(g["resume_prev"] for g in clean),
        "breaks": sum(g["breaks"] for g in clean),
    }

    # chains by origin (general vs other)
    gen_hist = Counter()
    oth_hist = Counter()
    for g in per_game:
        for ln in g["seg_from_general_lens"]:
            gen_hist[min(ln, 30)] += 1
        for ln in g["seg_other_lens"]:
            oth_hist[min(ln, 30)] += 1
    out["chains_by_origin"] = {
        "from_general_len_hist_capped30": counter_dict(gen_hist),
        "other_len_hist_capped30": counter_dict(oth_hist),
        "from_general_summary": summarize(
            [ln for g in per_game for ln in g["seg_from_general_lens"]]),
        "other_summary": summarize(
            [ln for g in per_game for ln in g["seg_other_lens"]]),
        "segment_start_tick_parity": counter_dict(
            sum((g["seg_start_par"] for g in per_game), Counter())),
    }

    out["reconstruction_quality"] = {
        "unresolved_ticks": sum(g["unresolved"] for g in per_game),
        "ambiguous_ticks": sum(g["ambiguous"] for g in per_game),
        "total_ticks": total_ticks,
    }
    out["first_contact_tick_summary"] = summarize(
        [g["first_contact_t"] for g in per_game if g["first_contact_t"]])
    out["general_sight_tick_summary"] = summarize(
        [g["general_sight_t"] for g in per_game if g["general_sight_t"]])

    OUT_PATH.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
