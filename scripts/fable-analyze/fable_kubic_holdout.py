#!/usr/bin/env python3
"""
Holdout verification for the Kubic behavior spec (fable-kubic).

Turns the fit-set rules from the six dimension analyses (opening, expansion,
army, attack, defense, tempo) into falsifiable predictions and evaluates them
on the 39 holdout games only. Rules and tolerances were frozen from the
fit-set reports BEFORE this script was run; nothing here feeds back into the
spec except verdicts.

Writes docs/research/measurements/fable-kubic-holdout-verification.json.

Definitions shared with the fit analyses:
- action at tick t reads state t-1;
- vision(player) = 8-neighborhood of owned cells;
- BFS distance over non-mountain cells;
- "lag game" = >50 unforced passes at t>=11 (fit: the five bist losses).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from statistics import median

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from arena.instrument.replay.loader import open_replay
from fable_kubic_common import DIRECTIONS, load_actions, split_rows

OUT = REPO_ROOT / "docs/research/measurements/fable-kubic-holdout-verification.json"


def vision_mask(owned: np.ndarray) -> np.ndarray:
    """8-neighborhood dilation of a boolean mask."""
    v = owned.copy()
    v[:-1, :] |= owned[1:, :]
    v[1:, :] |= owned[:-1, :]
    v[:, :-1] |= owned[:, 1:]
    v[:, 1:] |= owned[:, :-1]
    v[:-1, :-1] |= owned[1:, 1:]
    v[:-1, 1:] |= owned[1:, :-1]
    v[1:, :-1] |= owned[:-1, 1:]
    v[1:, 1:] |= owned[:-1, :-1]
    return v


def bfs_dist(rows: int, cols: int, mountains: frozenset, start: tuple) -> np.ndarray:
    dist = np.full((rows, cols), 10**6, dtype=np.int64)
    if start in mountains:
        return dist
    from collections import deque

    dist[start] = 0
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in mountains:
                if dist[nr, nc] > dist[r, c] + 1:
                    dist[nr, nc] = dist[r, c] + 1
                    q.append((nr, nc))
    return dist


def analyze_game(row: dict) -> dict:
    mid = row["match_id"]
    game = load_actions(mid)
    rep = open_replay("Kubic", mid)
    us = game["kubic_seat"]
    them = 1 - us
    rows_n, cols_n = rep.rows, rep.cols
    gen = tuple(rep.generals[us])
    egen = tuple(rep.generals[them])
    egen_dist = bfs_dist(rows_n, cols_n, rep.mountains, egen)

    armies = [np.array(f.armies, dtype=np.int64) for f in rep.ticks]
    owners = [np.array(f.owners, dtype=np.int64) for f in rep.ticks]

    acts = []
    for tick in game["ticks"]:
        a = dict(tick[f"p{us}"])
        a["t"] = tick["t"]
        acts.append(a)

    # per-tick masks derived lazily
    def state(t):  # decision state of the action at tick t
        return armies[t - 1], owners[t - 1]

    # first sight of enemy general (state ticks)
    first_sight = None
    for t in range(len(rep.ticks)):
        if vision_mask(owners[t] == us)[egen]:
            first_sight = t
            break

    out = {"match_id": mid, "outcome": row["outcome"], "total_ticks": game["total_ticks"]}

    # ---- R1 pass legality (t>=11): pass iff no owned cell army>=2 w/ passable nbr
    passable = np.ones((rows_n, cols_n), dtype=bool)
    for r, c in rep.mountains:
        passable[r, c] = False
    has_nbr = np.zeros((rows_n, cols_n), dtype=bool)
    has_nbr[:-1, :] |= passable[1:, :]
    has_nbr[1:, :] |= passable[:-1, :]
    has_nbr[:, :-1] |= passable[:, 1:]
    has_nbr[:, 1:] |= passable[:, :-1]

    r1_n = r1_bad = unforced_passes = 0
    for a in acts:
        t = a["t"]
        if t < 11:
            continue
        ar, ow = state(t)
        can_move = bool((((ow == us) & (ar >= 2)) & has_nbr & passable).any())
        is_pass = a["kind"] == "pass"
        r1_n += 1
        if is_pass == can_move:
            r1_bad += 1
        if is_pass and can_move:
            unforced_passes += 1
    out["r1"] = {"n": r1_n, "violations": r1_bad}
    out["lag_game"] = unforced_passes > 50
    out["unforced_passes"] = unforced_passes

    # ---- R2 first non-pass
    first = next((a for a in acts if a["kind"] != "pass"), None)
    out["r2"] = {
        "tick": first["t"] if first else None,
        "canonical": bool(
            first
            and first["t"] == 3
            and first["kind"] == "move"
            and tuple(first["src"]) == gen
            and first["target"] == "neutral"
            and first["moved"] == 1
        ),
    }

    # ---- R3 split usage; R9/R10 attack sends; R4/R5 general pulls; R6 snake
    moves = [a for a in acts if a["kind"] == "move"]
    r3 = [a for a in moves if a["src_army"] >= 3]
    out["r3"] = {"n": len(r3), "full": sum(1 for a in r3 if a["split"] == 0)}

    pulls = [a for a in moves if tuple(a["src"]) == gen]
    out["r4"] = {"n": len(pulls), "leave1": sum(1 for a in pulls if a["split"] == 0)}
    out["r5"] = {"n": len(pulls), "odd": sum(1 for a in pulls if a["t"] % 2 == 1)}

    snake_n = snake_hit = 0
    prev = None
    for a in acts:
        if a["kind"] == "move":
            if prev is not None and prev["t"] == a["t"] - 1:
                snake_n += 1
                if a["src"] == prev["dst"]:
                    snake_hit += 1
            prev = a
        elif a["kind"] != "pass":
            prev = None
    out["r6"] = {"n": snake_n, "hit": snake_hit}

    # ---- R7 relay direction-repeat (own-target moves, t<=50)
    last_dir: dict[tuple, tuple] = {}
    r7_n = r7_hit = 0
    for a in moves:
        src, dst = tuple(a["src"]), tuple(a["dst"])
        d = (dst[0] - src[0], dst[1] - src[1])
        if a["t"] <= 50 and a["target"] == "own" and src in last_dir:
            r7_n += 1
            if last_dir[src] == d:
                r7_hit += 1
        last_dir[src] = d
    out["r7"] = {"n": r7_n, "hit": r7_hit}

    # ---- attacks
    attacks = [a for a in moves if a["target"] in ("enemy", "enemy_general")]
    out["r9"] = {"n": len(attacks), "full": sum(1 for a in attacks if a["split"] == 0)}
    out["r10"] = {
        "n": len(attacks),
        "winning": sum(1 for a in attacks if a["moved"] > a["dst_army"]),
    }

    # R8 src = largest stack adjacent to visible enemy (>=2 candidates)
    r8_n = r8_hit = 0
    r11_n = r11_hit = 0
    r18_events = []
    for a in attacks:
        t = a["t"]
        ar, ow = state(t)
        vis = vision_mask(ow == us)
        vis_enemy = (ow == them) & vis
        if not vis_enemy.any():
            continue
        adj_enemy = np.zeros_like(vis_enemy)
        adj_enemy[:-1, :] |= vis_enemy[1:, :]
        adj_enemy[1:, :] |= vis_enemy[:-1, :]
        adj_enemy[:, :-1] |= vis_enemy[:, 1:]
        adj_enemy[:, 1:] |= vis_enemy[:, :-1]
        cand = (ow == us) & (ar >= 2) & adj_enemy
        n_cand = int(cand.sum())
        src = tuple(a["src"])
        if n_cand >= 2:
            r8_n += 1
            if ar[src] >= ar[cand].max():
                r8_hit += 1
        # R11: dst tie-break post-sight
        nbrs = []
        for dr, dc in DIRECTIONS:
            nr, nc = src[0] + dr, src[1] + dc
            if 0 <= nr < rows_n and 0 <= nc < cols_n and vis_enemy[nr, nc]:
                nbrs.append((nr, nc))
        if first_sight is not None and t - 1 >= first_sight and len(nbrs) >= 2:
            r11_n += 1
            best = min(int(egen_dist[c]) for c in nbrs)
            if int(egen_dist[tuple(a["dst"])]) == best:
                r11_hit += 1
        # R18: pre-sight belief march
        if (first_sight is None or t - 1 < first_sight) and a["captured"] and nbrs:
            reduce_opts = [c for c in nbrs if egen_dist[c] < egen_dist[src]]
            r18_events.append(
                (
                    1 if int(egen_dist[tuple(a["dst"])]) < int(egen_dist[src]) else 0,
                    len(reduce_opts) / len(nbrs),
                )
            )
    out["r8"] = {"n": r8_n, "hit": r8_hit}
    out["r11"] = {"n": r11_n, "hit": r11_hit}
    out["r18"] = {
        "n": len(r18_events),
        "chosen_reduce": sum(e[0] for e in r18_events),
        "baseline_sum": sum(e[1] for e in r18_events),
    }

    # ---- R12 mod-50 wave: neutral-capture rate wave vs gather residues
    caps = [a["t"] for a in moves if a["target"] == "neutral" and a["captured"]]
    r12 = {"wave_caps": 0, "wave_ticks": 0, "gather_caps": 0, "gather_ticks": 0}
    if caps:
        lo, hi = min(caps), max(caps)
        capset = set(caps)
        for t in range(lo, hi + 1):
            m = t % 50
            wave = m >= 28 or m <= 9
            key = "wave" if wave else "gather"
            r12[f"{key}_ticks"] += 1
            if t in capset:
                r12[f"{key}_caps"] += 1
    out["r12"] = r12

    # ---- R13 castles
    builds = [a for a in acts if a["kind"] == "build"]
    structures = [gen]
    build_rows = []
    for b in builds:
        cell = tuple(b["cell"])
        sd = min(abs(cell[0] - s[0]) + abs(cell[1] - s[1]) for s in structures)
        build_rows.append(
            {"t": b["t"], "cost": b["cost"], "struct_dist": sd, "mod50": b["t"] % 50}
        )
        structures.append(cell)
    out["r13"] = {"builds": build_rows}

    # ---- R14 build trigger: army ratio at state 120
    if len(armies) > 120:
        a120, o120 = armies[120], owners[120]
        my = int(a120[o120 == us].sum())
        opp = int(a120[o120 == them].sum())
        out["r14"] = {
            "ratio": my / max(1, opp),
            "built_after": any(b["t"] > 120 for b in builds) or bool(builds),
        }
    else:
        out["r14"] = None

    # ---- R15 general-drain threat independence (Manhattan<=6 proxy for BFS<=6)
    drain_t = drain_c = elig_t = elig_c = 0
    rr, cc = np.meshgrid(np.arange(rows_n), np.arange(cols_n), indexing="ij")
    near_gen = (np.abs(rr - gen[0]) + np.abs(cc - gen[1])) <= 6
    drains = {a["t"] for a in pulls if a["split"] == 0 and a["src_army"] >= 8}
    for a in acts:
        t = a["t"]
        ar, ow = state(t)
        if ow[gen] != us or ar[gen] < 8:
            continue
        vis = vision_mask(ow == us)
        threat = bool(((ow == them) & vis & near_gen).any())
        if threat:
            elig_t += 1
            drain_t += 1 if t in drains else 0
        else:
            elig_c += 1
            drain_c += 1 if t in drains else 0
    out["r15"] = {
        "threat_ticks": elig_t,
        "threat_drains": drain_t,
        "clear_ticks": elig_c,
        "clear_drains": drain_c,
    }

    # ---- R16 kill phase (wins only)
    if row["outcome"] == "win":
        last15 = [a for a in acts if a["t"] > game["total_ticks"] - 15]
        out["r16"] = {
            "passes_last15": sum(1 for a in last15 if a["kind"] == "pass"),
            "sight_tick": first_sight,
            "sight_before_end": first_sight is not None and first_sight < game["total_ticks"],
            "unresolved_final": game["total_ticks"] in set(game["unresolved"]),
        }
    else:
        out["r16"] = None

    # ---- R17 land at t50
    if len(owners) > 50:
        out["r17"] = {"land50": int((owners[50] == us).sum())}
    else:
        out["r17"] = None

    # ---- R19 opening source rule S1: src = prev dst if army>=2 else general
    s1_n = s1_hit = 0
    prev_dst = None
    for a in acts:
        if a["t"] > 50:
            break
        if a["kind"] != "move":
            continue
        t = a["t"]
        ar, ow = state(t)
        pred = None
        if prev_dst is not None and ow[prev_dst] == us and ar[prev_dst] >= 2:
            pred = prev_dst
        elif ow[gen] == us and ar[gen] >= 2:
            pred = gen
        if pred is not None:
            s1_n += 1
            if tuple(a["src"]) == pred:
                s1_hit += 1
        prev_dst = tuple(a["dst"])
    out["r19"] = {"n": s1_n, "hit": s1_hit}

    return out


def main() -> None:
    rows = split_rows("holdout")
    games = [analyze_game(r) for r in rows]
    clean = [g for g in games if not g["lag_game"]]
    wins = [g for g in games if g["outcome"] == "win"]

    def rate(gs, key, num, den):
        n = sum(g[key][den] for g in gs if g.get(key))
        h = sum(g[key][num] for g in gs if g.get(key))
        return {"n": n, "hits": h, "rate": h / n if n else None}

    r12w = sum(g["r12"]["wave_caps"] for g in games)
    r12wt = sum(g["r12"]["wave_ticks"] for g in games)
    r12g = sum(g["r12"]["gather_caps"] for g in games)
    r12gt = sum(g["r12"]["gather_ticks"] for g in games)

    builds = [b for g in games for b in g["r13"]["builds"]]
    r14 = [g["r14"] for g in games if g["r14"]]
    r14_pred = [(g["ratio"] <= 1.10, g["built_after"]) for g in r14]
    r15t = sum(g["r15"]["threat_ticks"] for g in games)
    r15td = sum(g["r15"]["threat_drains"] for g in games)
    r15c = sum(g["r15"]["clear_ticks"] for g in games)
    r15cd = sum(g["r15"]["clear_drains"] for g in games)
    r18n = sum(g["r18"]["n"] for g in games)

    summary = {
        "games": len(games),
        "outcomes": dict(Counter(g["outcome"] for g in games)),
        "lag_games": [g["match_id"] for g in games if g["lag_game"]],
        "rules": {
            "R1_pass_legality": {
                "fit": 0.9833,
                "clean_fit": 1.0,
                "holdout": rate(games, "r1", "violations", "n") | {"note": "violations counted"},
                "holdout_clean": rate(clean, "r1", "violations", "n"),
            },
            "R2_first_move_t3_canonical": {
                "fit": 0.972,
                "holdout_ticks": [g["r2"]["tick"] for g in games],
                "canonical": sum(1 for g in games if g["r2"]["canonical"]),
                "of": len(games),
            },
            "R3_full_send_army3plus": {"fit": 0.9881, "holdout": rate(games, "r3", "full", "n")},
            "R4_general_pull_leave1": {"fit": 0.9535, "holdout": rate(games, "r4", "leave1", "n")},
            "R5_general_pull_odd_tick": {"fit": 0.7551, "holdout": rate(games, "r5", "odd", "n")},
            "R6_snake_src_eq_prev_dst": {"fit": 0.7778, "holdout": rate(games, "r6", "hit", "n")},
            "R7_relay_direction_repeat_t50": {"fit": 0.961, "holdout": rate(games, "r7", "hit", "n")},
            "R8_attack_src_largest_adjacent": {"fit": 0.945, "holdout": rate(games, "r8", "hit", "n")},
            "R9_attack_full_send": {"fit": 0.998, "holdout": rate(games, "r9", "full", "n")},
            "R10_attack_winning_margin": {"fit": 0.948, "holdout": rate(games, "r10", "winning", "n")},
            "R11_attack_dst_min_bfs_to_general_post_sight": {
                "fit": 0.931,
                "holdout": rate(games, "r11", "hit", "n"),
            },
            "R12_mod50_wave_capture_ratio": {
                "fit_ratio": 0.450 / 0.171,
                "holdout": {
                    "wave_rate": r12w / r12wt if r12wt else None,
                    "gather_rate": r12g / r12gt if r12gt else None,
                    "ratio": (r12w / r12wt) / (r12g / r12gt) if r12wt and r12gt and r12g else None,
                },
            },
            "R13_castles": {
                "fit": {"cost35": 0.817, "first_build_min": 116, "mod50_gather_band": "p50 18"},
                "holdout": {
                    "n_builds": len(builds),
                    "cost_hist": dict(Counter(b["cost"] for b in builds)),
                    "struct_dist_hist": dict(Counter(b["struct_dist"] for b in builds)),
                    "first_build_ticks": sorted(
                        min(b["t"] for b in g["r13"]["builds"])
                        for g in games
                        if g["r13"]["builds"]
                    ),
                    "mod50_hist": sorted(b["mod50"] for b in builds),
                    "builds_per_game_max": max(
                        (len(g["r13"]["builds"]) for g in games), default=0
                    ),
                },
            },
            "R14_build_trigger_ratio110": {
                "fit_acc": 0.776,
                "holdout": {
                    "n": len(r14_pred),
                    "acc": sum(1 for p, b in r14_pred if p == b) / len(r14_pred)
                    if r14_pred
                    else None,
                    "behind_no_build": sum(
                        1 for g in r14 if g["ratio"] < 0.95 and not g["built_after"]
                    ),
                },
            },
            "R15_drain_threat_independent": {
                "fit_gap_pp": 0.4,
                "holdout": {
                    "threat_rate": r15td / r15t if r15t else None,
                    "clear_rate": r15cd / r15c if r15c else None,
                    "threat_n": r15t,
                    "clear_n": r15c,
                },
            },
            "R16_kill_phase": {
                "fit": {"no_pass_last15": 1.0, "sight_before_kill": 1.0},
                "holdout": {
                    "wins": len(wins),
                    "zero_pass_last15": sum(
                        1 for g in wins if g["r16"]["passes_last15"] == 0
                    ),
                    "sight_before_end": sum(1 for g in wins if g["r16"]["sight_before_end"]),
                    "unresolved_final": sum(1 for g in wins if g["r16"]["unresolved_final"]),
                },
            },
            "R17_land_at_t50": {
                "fit": {"median": 24, "iqr": [23, 25]},
                "holdout_values": sorted(
                    g["r17"]["land50"] for g in games if g["r17"]
                ),
            },
            "R18_presight_belief_march": {
                "fit": {"chosen": 0.92, "baseline": 0.64},
                "holdout": {
                    "n": r18n,
                    "chosen_rate": sum(g["r18"]["chosen_reduce"] for g in games) / r18n
                    if r18n
                    else None,
                    "baseline_rate": sum(g["r18"]["baseline_sum"] for g in games) / r18n
                    if r18n
                    else None,
                },
            },
            "R19_opening_src_prevdst_else_general": {
                "fit": 0.862,
                "holdout": rate(games, "r19", "hit", "n"),
            },
        },
        "per_game": games,
    }
    OUT.write_text(json.dumps(summary, indent=2))
    slim = {k: v for k, v in summary.items() if k != "per_game"}
    print(json.dumps(slim, indent=2))


if __name__ == "__main__":
    main()
