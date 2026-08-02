#!/usr/bin/env python3
"""
Kubic OPENING analysis (ticks 1..50) over the fable fit set.

Measures, over every fit-set game (docs/research/measurements/fable-kubic-split.json):
- first non-pass action (tick, kind, source, general army),
- pass structure (forced = no owned cell with army >= 2, vs voluntary),
- move mix, split usage, wave/segment structure, land curve, builds <= 150,
- source-selection and destination-selection rule agreement rates,
- a composite full-policy predictor with exact-action accuracy,
- win vs loss/draw and lag-game subsets.

Fog discipline: every candidate rule conditions only on Kubic-visible
information: the cumulative "seen" mask (3x3 neighbourhoods of owned cells,
i.e. sight + perfect memory) or the instantaneous visibility mask. Mountains
and ownership outside the seen mask are treated as unknown (assumed open /
not-own). The one deliberately fog-illegal probe (toward enemy general) is
labelled as such in the output.

Reproduce:
    .venv/bin/python scripts/fable_kubic_opening.py

Writes docs/research/measurements/fable-kubic-opening.json. Deterministic:
iterates games in split order, no RNG.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from fable_kubic_common import DIRECTIONS, kubic_actions, load_actions, split_rows
from arena.instrument.replay.loader import open_replay

OPEN_END = 50
BUILD_SCAN_END = 150
OUT_PATH = REPO_ROOT / "docs/research/measurements/fable-kubic-opening.json"
DIR_NAMES = ("U", "D", "L", "R")


# ----------------------------------------------------------------- helpers

def seen_of(own: np.ndarray, R: int, C: int) -> np.ndarray:
    """3x3 dilation of the ownership mask = instantaneous visibility."""
    p = np.zeros((R + 2, C + 2), bool)
    p[1:-1, 1:-1] = own
    m = np.zeros((R, C), bool)
    for dr in (0, 1, 2):
        for dc in (0, 1, 2):
            m |= p[dr:dr + R, dc:dc + C]
    return m


def bfs_firsts(src, passable, target, R, C):
    """Directions that start a shortest path from src to any target cell."""
    best = None
    firsts = set()
    dq = deque()
    seen = {src}
    for di, (dr, dc) in enumerate(DIRECTIONS):
        nr, nc = src[0] + dr, src[1] + dc
        if 0 <= nr < R and 0 <= nc < C and passable[nr, nc]:
            if target[nr, nc]:
                firsts.add(di)
                best = 1
            dq.append(((nr, nc), di, 1))
            seen.add((nr, nc))
    if best:
        return firsts
    while dq:
        cell, di, d = dq.popleft()
        if best is not None and d > best:
            break
        if target[cell]:
            firsts.add(di)
            best = d
            continue
        for dr, dc in DIRECTIONS:
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < R and 0 <= nc < C and passable[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                dq.append(((nr, nc), di, d + 1))
    return firsts


def spawn_class(R, C, gen):
    r, c = gen
    er = r <= 2 or r >= R - 3
    ec = c <= 2 or c >= C - 3
    if er and ec:
        return "corner"
    if er or ec:
        return "edge"
    return "interior"


def dist_stats(values):
    if not values:
        return {"n": 0}
    a = np.array(sorted(values), dtype=float)
    return {
        "n": len(values),
        "mean": round(float(a.mean()), 3),
        "min": float(a[0]),
        "p25": float(np.percentile(a, 25)),
        "median": float(np.percentile(a, 50)),
        "p75": float(np.percentile(a, 75)),
        "max": float(a[-1]),
    }


def counter_json(c: Counter):
    return {str(k): v for k, v in sorted(c.items(), key=lambda kv: str(kv[0]))}


# ------------------------------------------------------------- per-game pass

def analyze_game(rm):
    mid = rm["match_id"]
    g = load_actions(mid)
    rep = open_replay("Kubic", mid)
    us = g["kubic_seat"]
    gen = tuple(g["generals"][us])
    egen = tuple(g["generals"][1 - us])
    R, C = rep.rows, rep.cols
    mts = np.zeros((R, C), bool)
    for r, c in rep.mountains:
        mts[r, c] = True
    center = ((R - 1) / 2.0, (C - 1) / 2.0)
    acts = kubic_actions(g)

    out = {
        "match_id": mid,
        "outcome": rm["outcome"],
        "rows": R,
        "cols": C,
        "spawn_class": spawn_class(R, C, gen),
        "first_action": None,
        "passes_forced": 0,
        "passes_voluntary": 0,
        "n_ticks": 0,
        "n_moves": 0,
        "n_builds_le50": 0,
        "splits": 0,
        "target_mix": Counter(),
        "gen_moves": 0,
        "gen_move_army": Counter(),
        "gen_dst_repeat": [0, 0],       # [repeats, opportunities]
        "gen_move_given_movable": [0, 0],
        "src_rules": Counter(),          # rule -> hits (denominator n_moves)
        "relay": [0, 0],                 # onto-own: [last_out repeats, n]
        "relay_with_neutral_alt": 0,
        "choice": Counter(),             # frontier-rule hits (denominator n_choice)
        "n_choice": 0,
        "choice_rand_base": 0.0,
        "straight_tie": [0, 0],          # straight taken / straight available in cands
        "segments": [],                  # [length, launch_from_gen, launch_army]
        "land": {},
        "first_build": None,
        "composite": [0, 0],             # exact-action hits / ticks
        "composite_moves": [0, 0],
        "capture_ticks": [],
        "gen_move_tick_army": [],
        "land_curve": [],
        "first_move_dirs": None,
        "ambiguous_le50": sum(1 for t in g.get("ambiguous", []) if t <= OPEN_END),
    }

    seen_mask = seen_of(np.array(rep.ticks[0].owners) == us, R, C)
    prev_dst = None
    prev_dir = None
    last_out = {}
    seg_len = 0
    seg_launch = None

    for a in acts:
        t = a["t"]
        if t > BUILD_SCAN_END or t >= len(rep.ticks):
            break
        fr = rep.ticks[t - 1]
        if a["kind"] == "build" and out["first_build"] is None:
            cell = tuple(a["cell"])
            out["first_build"] = {
                "t": t,
                "cost": a["cost"],
                "manhattan_to_gen": abs(cell[0] - gen[0]) + abs(cell[1] - gen[1]),
                "army_before": fr.armies[cell[0]][cell[1]],
            }
        if t > OPEN_END:
            continue

        out["n_ticks"] += 1
        owners = np.array(fr.owners)
        armies = np.array(fr.armies)
        movable = (owners == us) & (armies >= 2)
        any_movable = bool(movable.any())

        # ---- composite prediction (before seeing the action) -------------
        pred = {"kind": "pass"}
        if any_movable:
            # src cascade: prev_dst if movable, else general, else max army.
            if prev_dst is not None and movable[prev_dst]:
                psrc = prev_dst
            elif movable[gen]:
                psrc = gen
            else:
                cand = [tuple(x) for x in np.argwhere(movable)]
                psrc = max(
                    ((int(armies[r, c]), -r, -c, (r, c)) for r, c in cand)
                )[3]
            # dst cascade
            pdst = None
            lo = last_out.get(psrc)
            if lo is not None:
                nr, nc = psrc[0] + DIRECTIONS[lo][0], psrc[1] + DIRECTIONS[lo][1]
                if 0 <= nr < R and 0 <= nc < C and owners[nr, nc] == us:
                    pdst = (nr, nc)
            if pdst is None:
                nv = []
                for di, (dr, dc) in enumerate(DIRECTIONS):
                    nr, nc = psrc[0] + dr, psrc[1] + dc
                    if not (0 <= nr < R and 0 <= nc < C):
                        continue
                    if mts[nr, nc] and seen_mask[nr, nc]:
                        continue
                    if not (seen_mask[nr, nc] and owners[nr, nc] == us):
                        nv.append(di)
                if nv:
                    passable = ~(mts & seen_mask)
                    firsts = bfs_firsts(psrc, passable, ~seen_mask, R, C)
                    cands = sorted(firsts & set(nv)) or nv
                    if prev_dst == psrc and prev_dir in cands:
                        pdi = prev_dir
                    else:
                        def centd(di):
                            dr, dc = DIRECTIONS[di]
                            return (
                                (psrc[0] + dr - center[0]) ** 2
                                + (psrc[1] + dc - center[1]) ** 2,
                                di,
                            )
                        pdi = min(cands, key=centd)
                    pdst = (psrc[0] + DIRECTIONS[pdi][0], psrc[1] + DIRECTIONS[pdi][1])
            if pdst is not None:
                pred = {"kind": "move", "src": psrc, "dst": pdst, "split": 0}

        actual = (
            {"kind": "pass"}
            if a["kind"] != "move"
            else {
                "kind": "move",
                "src": tuple(a["src"]),
                "dst": tuple(a["dst"]),
                "split": a["split"],
            }
        )
        if a["kind"] != "build":
            out["composite"][1] += 1
            if pred == actual:
                out["composite"][0] += 1
            if a["kind"] == "move":
                out["composite_moves"][1] += 1
                if pred == actual:
                    out["composite_moves"][0] += 1

        # ---- measurements -------------------------------------------------
        if a["kind"] == "pass":
            if any_movable:
                out["passes_voluntary"] += 1
            else:
                out["passes_forced"] += 1
            seg_len = 0
            seg_launch = None
            prev_dst = None
            prev_dir = None
            seen_mask |= seen_of(np.array(rep.ticks[t].owners) == us, R, C)
            continue
        if a["kind"] == "build":
            out["n_builds_le50"] += 1
            seen_mask |= seen_of(np.array(rep.ticks[t].owners) == us, R, C)
            continue

        src = tuple(a["src"])
        dst = tuple(a["dst"])
        adir = DIRECTIONS.index((dst[0] - src[0], dst[1] - src[1]))
        out["n_moves"] += 1
        out["target_mix"][a["target"]] += 1
        if a["split"] == 1:
            out["splits"] += 1
        if out["first_action"] is None:
            out["first_action"] = {
                "t": t,
                "src_is_gen": src == gen,
                "src_army": a["src_army"],
                "split": a["split"],
                "target": a["target"],
            }
            # candidate first directions from the general
            nv0 = []
            for di, (dr, dc) in enumerate(DIRECTIONS):
                nr, nc = src[0] + dr, src[1] + dc
                if 0 <= nr < R and 0 <= nc < C and not (mts[nr, nc] and seen_mask[nr, nc]):
                    nv0.append(di)
            def toward(tgt, di):
                dr, dc = DIRECTIONS[di]
                return (abs(src[0] + dr - tgt[0]) + abs(src[1] + dc - tgt[1])) < (
                    abs(src[0] - tgt[0]) + abs(src[1] - tgt[1])
                )
            out["first_move_dirs"] = {
                "dir": DIR_NAMES[adir],
                "n_options": len(nv0),
                "toward_center": toward(center, adir),
                "toward_enemy_gen_FOG_ILLEGAL": toward(egen, adir),
            }

        if movable.any():
            gen_can = bool(movable[gen])
            if gen_can:
                out["gen_move_given_movable"][1] += 1
                if src == gen:
                    out["gen_move_given_movable"][0] += 1
        if src == gen:
            out["gen_moves"] += 1
            out["gen_move_army"][a["src_army"]] += 1
            lo = last_out.get(gen)
            if lo is not None:
                out["gen_dst_repeat"][1] += 1
                if lo == adir:
                    out["gen_dst_repeat"][0] += 1

        # src rules
        s1 = prev_dst if (prev_dst is not None and movable[prev_dst]) else (
            gen if movable[gen] else None
        )
        if s1 == src:
            out["src_rules"]["S1_prevdst_else_gen"] += 1
        s3 = gen if movable[gen] else (
            prev_dst if (prev_dst is not None and movable[prev_dst]) else None
        )
        if s3 == src:
            out["src_rules"]["S3_gen_else_prevdst"] += 1
        if prev_dst == src:
            out["src_rules"]["chain_continuation"] += 1

        # relay rule (moves onto own cells)
        if a["target"] == "own":
            out["relay"][1] += 1
            if last_out.get(src) == adir:
                out["relay"][0] += 1
            has_neutral_alt = False
            for di, (dr, dc) in enumerate(DIRECTIONS):
                nr, nc = src[0] + dr, src[1] + dc
                if (
                    0 <= nr < R and 0 <= nc < C
                    and not (mts[nr, nc] and seen_mask[nr, nc])
                    and not (seen_mask[nr, nc] and owners[nr, nc] == us)
                ):
                    has_neutral_alt = True
            if has_neutral_alt:
                out["relay_with_neutral_alt"] += 1

        # frontier choice rules (moves onto neutral, >=2 known-neutral options)
        nv = []
        for di, (dr, dc) in enumerate(DIRECTIONS):
            nr, nc = src[0] + dr, src[1] + dc
            if not (0 <= nr < R and 0 <= nc < C):
                continue
            if mts[nr, nc] and seen_mask[nr, nc]:
                continue
            if not (seen_mask[nr, nc] and owners[nr, nc] == us):
                nv.append(di)
        if a["target"] == "neutral" and len(nv) >= 2:
            out["n_choice"] += 1
            out["choice_rand_base"] += 1.0 / len(nv)
            is_chain = prev_dst == src
            passable = ~(mts & seen_mask)
            firsts = bfs_firsts(src, passable, ~seen_mask, R, C)
            cands = sorted(firsts & set(nv)) or nv
            if adir in cands:
                out["choice"]["in_bfs_firsts"] += 1
            vis_now = seen_of(owners == us, R, C)
            nv_now = []
            for di, (dr, dc) in enumerate(DIRECTIONS):
                nr, nc = src[0] + dr, src[1] + dc
                if not (0 <= nr < R and 0 <= nc < C):
                    continue
                if mts[nr, nc] and vis_now[nr, nc]:
                    continue
                if not (vis_now[nr, nc] and owners[nr, nc] == us):
                    nv_now.append(di)
            firsts_now = bfs_firsts(src, ~(mts & vis_now), ~vis_now, R, C)
            cands_now = sorted(firsts_now & set(nv_now)) or nv_now
            if adir in cands_now:
                out["choice"]["in_bfs_firsts_memoryless"] += 1
            if is_chain and prev_dir in cands and len(cands) >= 2:
                out["straight_tie"][1] += 1
                if adir == prev_dir:
                    out["straight_tie"][0] += 1

            def centd(di):
                dr, dc = DIRECTIONS[di]
                return (
                    (src[0] + dr - center[0]) ** 2 + (src[1] + dc - center[1]) ** 2,
                    di,
                )
            pred_d = prev_dir if (is_chain and prev_dir in cands) else min(cands, key=centd)
            if adir == pred_d:
                out["choice"]["straight_else_center"] += 1
            if adir == (prev_dir if (is_chain and prev_dir in cands) else min(cands)):
                out["choice"]["straight_else_dirorder"] += 1
            if adir == min(nv, key=centd):
                out["choice"]["center_only"] += 1
            def egd(di):
                dr, dc = DIRECTIONS[di]
                return (abs(src[0] + dr - egen[0]) + abs(src[1] + dc - egen[1]), di)
            if adir == min(nv, key=egd):
                out["choice"]["toward_enemy_gen_FOG_ILLEGAL"] += 1

        # per-tick phase structure
        if a["target"] == "neutral" and a.get("captured"):
            out["capture_ticks"].append(t)
        if src == gen:
            out["gen_move_tick_army"].append([t, a["src_army"]])

        # segments
        if prev_dst == src:
            seg_len += 1
        else:
            if seg_len:
                out["segments"].append([seg_len, seg_launch[0], seg_launch[1]])
            seg_len = 1
            seg_launch = (src == gen, a["src_army"])

        last_out[src] = adir
        prev_dst = dst
        prev_dir = adir
        seen_mask |= seen_of(np.array(rep.ticks[t].owners) == us, R, C)
        continue

    if seg_len:
        out["segments"].append([seg_len, seg_launch[0], seg_launch[1]])

    # land curve (every tick 0..50)
    for tt in range(0, OPEN_END + 1):
        if tt < len(rep.ticks):
            out["land_curve"].append(sum(
                1
                for r in range(R)
                for c in range(C)
                if rep.ticks[tt].owners[r][c] == us
            ))
    for tt in (10, 25, 50):
        if tt < len(rep.ticks):
            own = sum(
                1
                for r in range(R)
                for c in range(C)
                if rep.ticks[tt].owners[r][c] == us
            )
            out["land"][str(tt)] = own
    out["board_cells"] = R * C
    return out


# ------------------------------------------------------------------- main

def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "fit"
    rows = split_rows(which)
    per_game = []
    for i, rm in enumerate(rows):
        per_game.append(analyze_game(rm))
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(rows)}", file=sys.stderr, flush=True)

    def agg(games):
        A = {
            "n_games": len(games),
            "first_action_tick": counter_json(Counter(
                g["first_action"]["t"] for g in games if g["first_action"]
            )),
            "first_action_from_general": sum(
                1 for g in games if g["first_action"] and g["first_action"]["src_is_gen"]
            ),
            "first_action_src_army": counter_json(Counter(
                g["first_action"]["src_army"] for g in games if g["first_action"]
            )),
            "first_action_split": counter_json(Counter(
                g["first_action"]["split"] for g in games if g["first_action"]
            )),
            "first_move_n_options": counter_json(Counter(
                g["first_move_dirs"]["n_options"] for g in games if g["first_move_dirs"]
            )),
            "first_move_toward_center": sum(
                1 for g in games if g["first_move_dirs"] and g["first_move_dirs"]["toward_center"]
            ),
            "first_move_toward_enemy_FOG_ILLEGAL": sum(
                1 for g in games
                if g["first_move_dirs"] and g["first_move_dirs"]["toward_enemy_gen_FOG_ILLEGAL"]
            ),
        }
        tot = lambda k: sum(g[k] for g in games)  # noqa: E731
        A["ticks"] = tot("n_ticks")
        A["moves"] = tot("n_moves")
        A["passes_forced"] = tot("passes_forced")
        A["passes_voluntary"] = tot("passes_voluntary")
        A["builds_le50"] = tot("n_builds_le50")
        A["splits"] = tot("splits")
        A["split_rate"] = round(A["splits"] / max(1, A["moves"]), 4)
        tm = Counter()
        for g in games:
            tm.update(g["target_mix"])
        A["target_mix"] = counter_json(tm)
        gma = Counter()
        for g in games:
            gma.update(g["gen_move_army"])
        A["gen_move_out_army"] = counter_json(gma)
        A["gen_move_given_movable"] = [
            sum(g["gen_move_given_movable"][0] for g in games),
            sum(g["gen_move_given_movable"][1] for g in games),
        ]
        A["gen_dst_repeat"] = [
            sum(g["gen_dst_repeat"][0] for g in games),
            sum(g["gen_dst_repeat"][1] for g in games),
        ]
        sr = Counter()
        for g in games:
            sr.update(g["src_rules"])
        A["src_rule_hits"] = counter_json(sr)
        A["src_rule_rates"] = {
            k: round(v / max(1, A["moves"]), 4) for k, v in sr.items()
        }
        A["relay"] = [sum(g["relay"][0] for g in games), sum(g["relay"][1] for g in games)]
        A["relay_repeat_rate"] = round(A["relay"][0] / max(1, A["relay"][1]), 4)
        A["relay_with_neutral_alt"] = tot("relay_with_neutral_alt")
        A["n_choice"] = tot("n_choice")
        A["choice_random_baseline"] = round(
            sum(g["choice_rand_base"] for g in games) / max(1, A["n_choice"]), 4
        )
        ch = Counter()
        for g in games:
            ch.update(g["choice"])
        A["choice_rule_hits"] = counter_json(ch)
        A["choice_rule_rates"] = {
            k: round(v / max(1, A["n_choice"]), 4) for k, v in ch.items()
        }
        A["straight_tie"] = [
            sum(g["straight_tie"][0] for g in games),
            sum(g["straight_tie"][1] for g in games),
        ]
        seglens = [s[0] for g in games for s in g["segments"]]
        A["segment_length"] = dist_stats(seglens)
        A["segment_length_hist"] = counter_json(Counter(seglens))
        launch_army = [s[2] for g in games for s in g["segments"] if s[1]]
        A["gen_launch_army"] = dist_stats(launch_army)
        A["segments_from_gen"] = sum(1 for g in games for s in g["segments"] if s[1])
        A["segments_total"] = len(seglens)
        for tt in ("10", "25", "50"):
            A[f"land_t{tt}"] = dist_stats([
                g["land"][tt] for g in games if tt in g["land"]
            ])
        A["land_frac_t50"] = dist_stats([
            g["land"]["50"] / g["board_cells"] for g in games if "50" in g["land"]
        ])
        fb = [g["first_build"] for g in games if g["first_build"]]
        A["games_with_build_le150"] = len(fb)
        A["first_build_tick"] = dist_stats([b["t"] for b in fb])
        A["first_build_tick_hist"] = counter_json(Counter(b["t"] for b in fb))
        A["first_build_cost"] = counter_json(Counter(b["cost"] for b in fb))
        A["first_build_manhattan_to_gen"] = counter_json(
            Counter(b["manhattan_to_gen"] for b in fb)
        )
        A["first_build_army_before"] = dist_stats([b["army_before"] for b in fb])
        A["composite_exact"] = [
            sum(g["composite"][0] for g in games),
            sum(g["composite"][1] for g in games),
        ]
        A["composite_exact_rate"] = round(
            A["composite_exact"][0] / max(1, A["composite_exact"][1]), 4
        )
        A["composite_moves_exact"] = [
            sum(g["composite_moves"][0] for g in games),
            sum(g["composite_moves"][1] for g in games),
        ]
        A["composite_moves_rate"] = round(
            A["composite_moves_exact"][0] / max(1, A["composite_moves_exact"][1]), 4
        )
        A["ambiguous_ticks_le50"] = tot("ambiguous_le50")
        # per-tick phase curves
        land_mean = []
        for tt in range(0, OPEN_END + 1):
            vals = [g["land_curve"][tt] for g in games if tt < len(g["land_curve"])]
            land_mean.append(round(float(np.mean(vals)), 2) if vals else None)
        A["land_curve_mean_by_tick"] = land_mean
        cap = Counter(t for g in games for t in g["capture_ticks"])
        A["capture_rate_by_tick"] = {
            str(t): round(cap.get(t, 0) / max(1, len(games)), 3)
            for t in range(1, OPEN_END + 1)
        }
        gla = Counter()
        for g in games:
            for t, army in g["gen_move_tick_army"]:
                bucket = f"{(t - 1) // 10 * 10 + 1}-{(t - 1) // 10 * 10 + 10}"
                gla[(bucket, army)] += 1
        A["gen_move_army_by_tick_bucket"] = counter_json(gla)
        # spawn dependence
        A["spawn_class_counts"] = counter_json(Counter(g["spawn_class"] for g in games))
        A["land_t50_by_spawn"] = {
            sc: dist_stats([
                g["land"]["50"] for g in games
                if g["spawn_class"] == sc and "50" in g["land"]
            ])
            for sc in ("corner", "edge", "interior")
        }
        return A

    wins = [g for g in per_game if g["outcome"] == "win"]
    losses = [g for g in per_game if g["outcome"] == "lose"]
    draws = [g for g in per_game if g["outcome"] == "draw"]
    laggy = [
        g["match_id"] for g in per_game
        if g["passes_voluntary"] >= 4
        or (g["first_action"] and g["first_action"]["t"] > 3)
    ]
    clean = [g for g in per_game if g["match_id"] not in laggy]

    result = {
        "set": which,
        "n_games": len(per_game),
        "open_end": OPEN_END,
        "build_scan_end": BUILD_SCAN_END,
        "laggy_games": sorted(laggy),
        "all": agg(per_game),
        "clean": agg(clean),
        "wins": agg(wins),
        "losses": agg(losses),
        "draws": agg(draws),
        "per_game_brief": [
            {
                "match_id": g["match_id"],
                "outcome": g["outcome"],
                "spawn_class": g["spawn_class"],
                "first_tick": g["first_action"]["t"] if g["first_action"] else None,
                "passes_voluntary": g["passes_voluntary"],
                "land50": g["land"].get("50"),
                "first_build_t": g["first_build"]["t"] if g["first_build"] else None,
                "composite": g["composite"],
            }
            for g in per_game
        ],
    }
    OUT_PATH.write_text(json.dumps(result, indent=1))
    print(f"wrote {OUT_PATH}")
    for name in ("all", "clean", "wins", "losses"):
        A = result[name]
        print(
            f"{name}: n={A['n_games']} composite={A['composite_exact_rate']}"
            f" moves={A['composite_moves_rate']}"
            f" relay={A['relay_repeat_rate']}"
            f" choice_bfs={A['choice_rule_rates'].get('in_bfs_firsts')}"
        )


if __name__ == "__main__":
    main()
