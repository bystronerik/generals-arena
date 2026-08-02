#!/usr/bin/env python3
"""
Kubic army management & routing from exact reconstructed actions (fable analyst).

Reads ONLY the fit set of docs/research/measurements/fable-kubic-split.json
(340 wins, 10 losses, 1 draw) via fable_kubic_common.load_actions (bit-exact
per-tick action reconstruction) plus raw board grids for concentration and
fog-legal context.

Measures:
  1. army concentration (largest stack / general / castles / 1-2 army cells)
     over normalized and absolute time,
  2. carry chains (src of next move == dst of previous), fan-in merges,
  3. big-stack departure triggers (stationary time, army, visibility, parity),
  4. split (50%) vs full (all-but-one) usage by context,
  5. general/castle milking (intervals, army at pull, remainder),
  6. routing efficiency of chains vs BFS shortest path,
  7. wave cadence from the general (inter-departure intervals, mod-2/mod-50),
  8. the same core stats for the 10 fit losses + 1 draw.

Writes docs/research/measurements/fable-kubic-army.json. Deterministic.
Never touches holdout replays or data/games|ratings|remote_games.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, deque
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fable_kubic_common import load_actions, split_rows  # noqa: E402
from arena.instrument.replay.loader import open_replay  # noqa: E402

OUT_PATH = REPO_ROOT / "docs/research/measurements/fable-kubic-army.json"

CHAIN_MAX_GAP = 50          # ticks of idling allowed inside one carry chain
BIG_STACK_MIN = 15          # "big stack" departure threshold (army units)
STATIONARY_MIN = 5          # ticks a cell must sit untouched to count as parked
ROUTE_MIN_MOVES = 6         # chain length required for routing analysis
ROUTE_MIN_DISP = 3          # net manhattan displacement required for routing
TICK_BUCKETS = ((0, 25), (25, 50), (50, 100), (100, 200), (200, 400), (400, 10**9))


# ------------------------------------------------------------------ helpers

def dist_summary(values, percentiles=(5, 10, 25, 50, 75, 90, 95)) -> dict:
    vals = np.asarray(sorted(values), dtype=float)
    if vals.size == 0:
        return {"n": 0}
    out = {
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "min": float(vals[0]),
        "max": float(vals[-1]),
    }
    for p in percentiles:
        out[f"p{p}"] = float(np.percentile(vals, p))
    out["median"] = out["p50"]
    return out


def hist(values, cap: int | None = None) -> dict:
    c = Counter()
    for v in values:
        v = int(v)
        if cap is not None and v > cap:
            v = cap
        c[v] += 1
    return {str(k): int(v) for k, v in sorted(c.items())}


def tick_bucket(t: int) -> str:
    for lo, hi in TICK_BUCKETS:
        if lo <= t < hi:
            return f"{lo}+" if hi >= 10**9 else f"{lo}-{hi - 1}"
    return "?"


def manhattan(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def bfs_dist(passable: np.ndarray, src, dst) -> int | None:
    """Shortest orthogonal path length over passable cells (omniscient map)."""
    if src == dst:
        return 0
    rows, cols = passable.shape
    seen = np.zeros_like(passable, dtype=bool)
    seen[src] = True
    q = deque([(src, 0)])
    while q:
        (r, c), d = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not seen[nr, nc] and passable[nr, nc]:
                if (nr, nc) == dst:
                    return d + 1
                seen[nr, nc] = True
                q.append(((nr, nc), d + 1))
    return None


def vision_mask(owned: np.ndarray) -> np.ndarray:
    """3x3 dilation of the owned mask = fog-of-war visibility."""
    vis = owned.copy()
    vis[:-1, :] |= owned[1:, :]
    vis[1:, :] |= owned[:-1, :]
    vis[:, :-1] |= owned[:, 1:]
    vis[:, 1:] |= owned[:, :-1]
    vis[:-1, :-1] |= owned[1:, 1:]
    vis[:-1, 1:] |= owned[1:, :-1]
    vis[1:, :-1] |= owned[:-1, 1:]
    vis[1:, 1:] |= owned[:-1, :-1]
    return vis


# ------------------------------------------------------------ per-game pass

def analyze_game(match_id: str) -> dict:
    game = load_actions(match_id)
    rep = open_replay("Kubic", match_id)
    us = game["kubic_seat"]
    them = 1 - us
    home = tuple(game["generals"][us])
    egen = tuple(game["generals"][them])
    rows, cols = game["rows"], game["cols"]
    passable = np.ones((rows, cols), dtype=bool)
    for r, c in rep.mountains:
        passable[r, c] = False
    n_frames = len(rep.ticks)

    armies = [np.array(f.armies, dtype=np.int64) for f in rep.ticks]
    owners = [np.array(f.owners, dtype=np.int64) for f in rep.ticks]

    # --- castles: build tick per cell, from both players' actions + recovery
    castle_born: dict[tuple[int, int], int] = {}
    for tick in game["ticks"]:
        for key in ("p0", "p1"):
            a = tick[key]
            if a.get("kind") == "build":
                castle_born.setdefault(tuple(a["cell"]), tick["t"])
        for cell in tick.get("castle_recovered", []):
            castle_born.setdefault(tuple(cell), tick["t"])

    # --- fog-legal context per tick: enemy visible, enemy general visible
    first_contact = None
    first_gen_sight = None
    enemy_visible = np.zeros(n_frames, dtype=bool)
    for t in range(n_frames):
        own = owners[t] == us
        vis = vision_mask(own)
        if (vis & (owners[t] == them)).any():
            enemy_visible[t] = True
            if first_contact is None:
                first_contact = t
        if first_gen_sight is None and vis[egen]:
            first_gen_sight = t

    # --- concentration time series
    conc = []  # (t, total, max_stack, gen, castle, small12)
    for t in range(n_frames):
        own = owners[t] == us
        a = armies[t]
        total = int(a[own].sum())
        if total <= 0:
            conc.append((t, 0, 0, 0, 0, 0))
            continue
        mx = int(a[own].max())
        gen = int(a[home]) if own[home] else 0
        cas = sum(
            int(a[cell])
            for cell, born in castle_born.items()
            if born <= t and owners[t][cell] == us
        )
        small = int(a[own & (a <= 2)].sum())
        conc.append((t, total, mx, gen, cas, small))

    # --- kubic actions
    acts = []
    key = f"p{us}"
    for tick in game["ticks"]:
        a = dict(tick[key])
        a["t"] = tick["t"]
        a["ambiguous"] = "ambiguous" in tick
        acts.append(a)
    moves = [a for a in acts if a["kind"] == "move"]

    # forced-pass detection: no owned cell with army>=2 that has a passable
    # orthogonal neighbour (state at t-1)
    passes = []
    for a in acts:
        if a["kind"] != "pass":
            continue
        t = a["t"]
        prev = t - 1
        own = owners[prev] == us
        movable = own & (armies[prev] >= 2)
        can_move = False
        for r, c in zip(*np.nonzero(movable)):
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and passable[nr, nc]:
                    can_move = True
                    break
            if can_move:
                break
        passes.append({"t": t, "forced": not can_move})

    # --- per-move context: is src the current max stack (state t-1)?
    move_rows = []
    for m in moves:
        t = m["t"]
        prev = t - 1
        src = tuple(m["src"])
        own = owners[prev] == us
        a_prev = armies[prev]
        mx = int(a_prev[own].max()) if own.any() else 0
        move_rows.append(
            {
                **m,
                "src_t": src,
                "dst_t": tuple(m["dst"]),
                "is_max_stack": int(a_prev[src]) == mx and mx >= 2,
                "src_is_general": src == home,
                "src_is_castle": src in castle_born and castle_born[src] < t,
                "dst_is_general": tuple(m["dst"]) == home,
                "enemy_vis": bool(enemy_visible[prev]),
            }
        )

    # --- carry chains
    chains = []
    open_heads: dict[tuple[int, int], int] = {}
    last_touch: dict[tuple[int, int], int] = {}
    fan_in_links = 0
    for m in move_rows:
        src, dst, t = m["src_t"], m["dst_t"], m["t"]
        ci = open_heads.pop(src, None)
        if ci is not None and t - chains[ci]["last_t"] <= CHAIN_MAX_GAP:
            ch = chains[ci]
            ch["gaps"].append(t - ch["last_t"] - 1)
            ch["moves"].append(m)
            ch["last_t"] = t
        else:
            stationary = t - last_touch.get(src, 0)
            ci = len(chains)
            chains.append(
                {
                    "moves": [m],
                    "gaps": [],
                    "last_t": t,
                    "start_stationary": stationary,
                }
            )
        if m["target"] == "own" and dst in open_heads:
            fan_in_links += 1  # merged into another active flow
        open_heads[dst] = ci
        last_touch[src] = t
        last_touch[dst] = t

    chain_rows = []
    for ch in chains:
        ms = ch["moves"]
        first, last = ms[0], ms[-1]
        start, end = first["src_t"], last["dst_t"]
        pickup = sum(m["dst_army"] for m in ms if m["target"] == "own")
        row = {
            "n_moves": len(ms),
            "start_t": first["t"],
            "end_t": last["t"],
            "gaps_sum": sum(ch["gaps"]),
            "max_gap": max(ch["gaps"], default=0),
            "start_cell": start,
            "end_cell": end,
            "start_moved": first["moved"],
            "end_moved": last["moved"],
            "max_moved": max(m["moved"] for m in ms),
            "pickup_own": pickup,
            "start_is_general": first["src_is_general"],
            "start_is_castle": first["src_is_castle"],
            "start_dist_home": manhattan(start, home),
            "end_dist_home": manhattan(end, home),
            "start_dist_egen": manhattan(start, egen),
            "end_dist_egen": manhattan(end, egen),
            "end_target": last["target"],
            "end_captured": last["captured"],
            "captures_enemy": sum(1 for m in ms if m["target"] in ("enemy", "enemy_general") and m["captured"]),
            "captures_neutral": sum(1 for m in ms if m["target"] == "neutral" and m["captured"]),
            "own_share": sum(1 for m in ms if m["target"] == "own") / len(ms),
            "start_stationary": ch["start_stationary"],
            "net_disp": manhattan(start, end),
        }
        if row["n_moves"] >= ROUTE_MIN_MOVES and row["net_disp"] >= ROUTE_MIN_DISP:
            bd = bfs_dist(passable, start, end)
            row["bfs_dist"] = bd
            row["detour_ratio"] = (row["n_moves"] / bd) if bd else None
            cells = [m["dst_t"] for m in ms]
            row["revisit_moves"] = len(cells) - len(set(cells))
        chain_rows.append(row)

    # --- general pulls
    gen_pulls = [m for m in move_rows if m["src_is_general"]]
    castle_pulls = [m for m in move_rows if m["src_is_castle"]]

    # --- big-stack departures (parked >= STATIONARY_MIN then moved)
    departures = [
        {
            "t": ch["moves"][0]["t"],
            "army": ch["moves"][0]["src_army"],
            "stationary": ch["start_stationary"],
            "enemy_vis": ch["moves"][0]["enemy_vis"],
            "from_general": ch["moves"][0]["src_is_general"],
            "mod50": ch["moves"][0]["t"] % 50,
            "mod2": ch["moves"][0]["t"] % 2,
        }
        for ch in chains
        if ch["start_stationary"] >= STATIONARY_MIN
        and ch["moves"][0]["src_army"] >= BIG_STACK_MIN
    ]

    return {
        "match_id": match_id,
        "outcome": rep.outcome,
        "total_ticks": game["total_ticks"],
        "n_frames": n_frames,
        "first_contact": first_contact,
        "first_gen_sight": first_gen_sight,
        "home": home,
        "egen": egen,
        "conc": conc,
        "acts": acts,
        "move_rows": move_rows,
        "passes": passes,
        "chains": chain_rows,
        "fan_in_links": fan_in_links,
        "gen_pulls": gen_pulls,
        "castle_pulls": castle_pulls,
        "departures": departures,
        "castles_built_by_us": sorted(
            c for c, born in castle_born.items()
            if any(t[f"p{us}"].get("kind") == "build" and tuple(t[f"p{us}"]["cell"]) == c for t in game["ticks"])
        ),
        "unresolved": len(game["unresolved"]),
        "ambiguous": len(game["ambiguous"]),
        "cells_t25": int((owners[25] == us).sum()) if n_frames > 25 else None,
        "cells_t50": int((owners[50] == us).sum()) if n_frames > 50 else None,
    }


# ------------------------------------------------------------- aggregation

def aggregate(games: list[dict]) -> dict:
    agg: dict = {}

    # ---- 1. concentration profiles
    deciles = {d: {"max": [], "gen": [], "castle": [], "small": []} for d in range(10)}
    buckets = {}
    final_frac = {"max": [], "gen": [], "castle": [], "small": []}
    for g in games:
        n = len(g["conc"])
        for t, total, mx, gen, cas, small in g["conc"]:
            if total <= 0:
                continue
            d = min(9, int(10 * t / max(1, n - 1)))
            deciles[d]["max"].append(mx / total)
            deciles[d]["gen"].append(gen / total)
            deciles[d]["castle"].append(cas / total)
            deciles[d]["small"].append(small / total)
            b = tick_bucket(t)
            bk = buckets.setdefault(b, {"max": [], "gen": [], "castle": [], "small": []})
            bk["max"].append(mx / total)
            bk["gen"].append(gen / total)
            bk["castle"].append(cas / total)
            bk["small"].append(small / total)
        t, total, mx, gen, cas, small = g["conc"][-1]
        if total > 0:
            for k, v in (("max", mx), ("gen", gen), ("castle", cas), ("small", small)):
                final_frac[k].append(v / total)
    agg["concentration"] = {
        "by_decile": {
            str(d): {k: dist_summary(v) for k, v in dd.items()} for d, dd in deciles.items()
        },
        "by_tick_bucket": {
            b: {k: dist_summary(v) for k, v in bb.items()} for b, bb in sorted(buckets.items())
        },
        "final_tick": {k: dist_summary(v) for k, v in final_frac.items()},
    }

    # ---- action mix / passes
    kinds = Counter()
    forced_pass = vol_pass = 0
    vol_pass_examples = []
    for g in games:
        kinds.update(a["kind"] for a in g["acts"])
        for p in g["passes"]:
            if p["forced"]:
                forced_pass += 1
            else:
                vol_pass += 1
                if len(vol_pass_examples) < 30:
                    vol_pass_examples.append({"match_id": g["match_id"], "t": p["t"]})
    agg["action_mix"] = {
        "kinds": dict(kinds),
        "passes_forced": forced_pass,
        "passes_voluntary": vol_pass,
        "voluntary_pass_examples": vol_pass_examples,
    }

    # ---- 4. split vs full
    sp = {
        "n_moves": 0,
        "n_ambiguous_ticks": 0,
        "src_army_2": 0,  # encoding-ambiguous
        "n_ge3": 0,
        "n_split_ge3": 0,
        "split_by_target": Counter(),
        "ge3_by_target": Counter(),
        "split_by_bucket": Counter(),
        "ge3_by_bucket": Counter(),
        "split_from_general": 0,
        "ge3_from_general": 0,
        "split_src_army": [],
        "split_mod50": Counter(),
        "split_dst_own_army": [],
        "split_next_src_is_dst": 0,
        "split_next_src_is_src": 0,
        "examples": [],
    }
    for g in games:
        mr = g["move_rows"]
        for i, m in enumerate(mr):
            sp["n_moves"] += 1
            if m["ambiguous"]:
                sp["n_ambiguous_ticks"] += 1
                continue
            if m["src_army"] == 2:
                sp["src_army_2"] += 1
                continue
            sp["n_ge3"] += 1
            b = tick_bucket(m["t"])
            sp["ge3_by_target"][m["target"]] += 1
            sp["ge3_by_bucket"][b] += 1
            if m["src_is_general"]:
                sp["ge3_from_general"] += 1
            if m["split"] == 1:
                sp["n_split_ge3"] += 1
                sp["split_by_target"][m["target"]] += 1
                sp["split_by_bucket"][b] += 1
                if m["src_is_general"]:
                    sp["split_from_general"] += 1
                sp["split_src_army"].append(m["src_army"])
                sp["split_mod50"][m["t"] % 50] += 1
                if m["target"] == "own":
                    sp["split_dst_own_army"].append(m["dst_army"])
                if i + 1 < len(mr):
                    nxt = mr[i + 1]
                    if nxt["src_t"] == m["dst_t"]:
                        sp["split_next_src_is_dst"] += 1
                    elif nxt["src_t"] == m["src_t"]:
                        sp["split_next_src_is_src"] += 1
                if len(sp["examples"]) < 40:
                    sp["examples"].append(
                        {
                            "match_id": g["match_id"],
                            "t": m["t"],
                            "src": list(m["src_t"]),
                            "dst": list(m["dst_t"]),
                            "src_army": m["src_army"],
                            "moved": m["moved"],
                            "target": m["target"],
                            "dst_army": m["dst_army"],
                        }
                    )
    sp["split_rate_ge3"] = sp["n_split_ge3"] / max(1, sp["n_ge3"])
    sp["split_by_target"] = dict(sp["split_by_target"])
    sp["ge3_by_target"] = dict(sp["ge3_by_target"])
    sp["split_by_bucket"] = dict(sp["split_by_bucket"])
    sp["ge3_by_bucket"] = dict(sp["ge3_by_bucket"])
    sp["split_mod50"] = {str(k): v for k, v in sorted(sp["split_mod50"].items())}
    sp["split_src_army_hist"] = hist(sp.pop("split_src_army"), cap=60)
    sp["split_dst_own_army_hist"] = hist(sp.pop("split_dst_own_army"), cap=60)
    agg["split_vs_full"] = sp

    # ---- 2+6. chains
    ch_all = [c for g in games for c in g["chains"]]
    long_ch = [c for c in ch_all if c["n_moves"] >= 5]
    routed = [c for c in ch_all if "bfs_dist" in c and c["bfs_dist"]]
    agg["chains"] = {
        "n_chains": len(ch_all),
        "n_moves_total": sum(c["n_moves"] for c in ch_all),
        "chain_len": dist_summary([c["n_moves"] for c in ch_all]),
        "chain_len_hist": hist([c["n_moves"] for c in ch_all], cap=100),
        "share_moves_in_chains_ge5": sum(c["n_moves"] for c in long_ch)
        / max(1, sum(c["n_moves"] for c in ch_all)),
        "gaps_sum": dist_summary([c["gaps_sum"] for c in ch_all]),
        "max_gap_hist": hist([c["max_gap"] for c in ch_all], cap=50),
        "start_moved": dist_summary([c["start_moved"] for c in ch_all]),
        "end_moved_ge5": dist_summary([c["end_moved"] for c in long_ch]),
        "growth_ge5": dist_summary([c["end_moved"] - c["start_moved"] for c in long_ch]),
        "pickup_own_ge5": dist_summary([c["pickup_own"] for c in long_ch]),
        "start_is_general_rate": sum(c["start_is_general"] for c in ch_all) / max(1, len(ch_all)),
        "start_is_general_rate_ge5": sum(c["start_is_general"] for c in long_ch) / max(1, len(long_ch)),
        "start_dist_home_ge5": dist_summary([c["start_dist_home"] for c in long_ch]),
        "end_dist_egen_ge5": dist_summary([c["end_dist_egen"] for c in long_ch]),
        "delta_dist_egen_ge5": dist_summary(
            [c["end_dist_egen"] - c["start_dist_egen"] for c in long_ch]
        ),
        "end_target_ge5": dict(Counter(c["end_target"] for c in long_ch)),
        "end_captured_rate_ge5": sum(c["end_captured"] for c in long_ch) / max(1, len(long_ch)),
        "own_share_ge5": dist_summary([c["own_share"] for c in long_ch]),
        "fan_in_links": sum(g["fan_in_links"] for g in games),
        "routing": {
            "n_routed": len(routed),
            "detour_ratio": dist_summary([c["detour_ratio"] for c in routed if c["detour_ratio"]]),
            "detour_le_1_05": sum(1 for c in routed if c["detour_ratio"] and c["detour_ratio"] <= 1.05)
            / max(1, len(routed)),
            "detour_le_1_25": sum(1 for c in routed if c["detour_ratio"] and c["detour_ratio"] <= 1.25)
            / max(1, len(routed)),
            "revisit_moves": dist_summary([c.get("revisit_moves", 0) for c in routed]),
            "own_share": dist_summary([c["own_share"] for c in routed]),
        },
    }

    # snake rate: consecutive-tick continuation
    snake_num = snake_den = 0
    snake_after = {"own": [0, 0], "neutral": [0, 0], "enemy": [0, 0], "enemy_general": [0, 0]}
    max_stack_moves = 0
    max_stack_by_bucket = {}
    for g in games:
        mr = g["move_rows"]
        for i in range(1, len(mr)):
            if mr[i]["t"] == mr[i - 1]["t"] + 1:
                snake_den += 1
                prev = mr[i - 1]
                cont = mr[i]["src_t"] == prev["dst_t"]
                snake_num += cont
                k = prev["target"]
                ok = prev["target"] == "own" or prev["captured"]
                if k in snake_after and ok:
                    snake_after[k][0] += cont
                    snake_after[k][1] += 1
        for m in mr:
            max_stack_moves += m["is_max_stack"]
            b = tick_bucket(m["t"])
            e = max_stack_by_bucket.setdefault(b, [0, 0])
            e[0] += m["is_max_stack"]
            e[1] += 1
    agg["snake"] = {
        "consecutive_pairs": snake_den,
        "continuation_rate": snake_num / max(1, snake_den),
        "continuation_after_target": {
            k: {"rate": a / max(1, b), "n": b} for k, (a, b) in snake_after.items()
        },
        "max_stack_move_rate": max_stack_moves / max(1, sum(len(g["move_rows"]) for g in games)),
        "max_stack_move_rate_by_bucket": {
            b: {"rate": a / max(1, n), "n": n} for b, (a, n) in sorted(max_stack_by_bucket.items())
        },
    }

    # ---- 5. general / castle milking
    for name, key in (("general_pulls", "gen_pulls"), ("castle_pulls", "castle_pulls")):
        pulls = [p for g in games for p in g[key]]
        intervals = []
        for g in games:
            ts = [p["t"] for p in g[key]]
            intervals += [b - a for a, b in zip(ts, ts[1:])]
        agg[name] = {
            "n": len(pulls),
            "per_game": dist_summary([len(g[key]) for g in games]),
            "src_army_at_pull": dist_summary([p["src_army"] for p in pulls]),
            "src_army_hist": hist([p["src_army"] for p in pulls], cap=80),
            "moved": dist_summary([p["moved"] for p in pulls]),
            "left_after_pull_hist": hist([p["src_army"] - p["moved"] for p in pulls], cap=20),
            "split_rate": sum(p["split"] for p in pulls) / max(1, len(pulls)),
            "interval": dist_summary(intervals),
            "interval_hist": hist(intervals, cap=100),
            "t_mod2": hist([p["t"] % 2 for p in pulls]),
            "t_mod50": hist([p["t"] % 50 for p in pulls]),
            "bucket": dict(Counter(tick_bucket(p["t"]) for p in pulls)),
        }
    # general army level profile (reserve) by decile
    gen_profile = {d: [] for d in range(10)}
    for g in games:
        n = len(g["conc"])
        for t, total, mx, gen, cas, small in g["conc"]:
            d = min(9, int(10 * t / max(1, n - 1)))
            gen_profile[d].append(gen)
    agg["general_reserve_by_decile"] = {str(d): dist_summary(v) for d, v in gen_profile.items()}

    # ---- 3+7. departures & cadence
    deps = [d for g in games for d in g["departures"]]
    agg["big_stack_departures"] = {
        "n": len(deps),
        "army": dist_summary([d["army"] for d in deps]),
        "stationary_ticks": dist_summary([d["stationary"] for d in deps]),
        "stationary_hist": hist([d["stationary"] for d in deps], cap=100),
        "enemy_visible_rate": sum(d["enemy_vis"] for d in deps) / max(1, len(deps)),
        "from_general_rate": sum(d["from_general"] for d in deps) / max(1, len(deps)),
        "mod2": hist([d["mod2"] for d in deps]),
        "mod50": hist([d["mod50"] for d in deps]),
    }
    # wave cadence: chain starts at/near general
    wave_int = []
    wave_mod50 = []
    for g in games:
        starts = sorted(
            c["start_t"] for c in g["chains"] if c["start_is_general"] or c["start_dist_home"] <= 1
        )
        wave_int += [b - a for a, b in zip(starts, starts[1:])]
        wave_mod50 += [s % 50 for s in starts]
    agg["wave_cadence"] = {
        "inter_start_interval": dist_summary(wave_int),
        "inter_start_hist": hist(wave_int, cap=100),
        "start_mod50": hist(wave_mod50),
    }

    # ---- chain typology: expansion snakes vs mass carries
    def chain_type(c) -> str:
        if c["max_moved"] >= 10:
            return "mass_carry"
        if c["max_moved"] <= 5:
            return "expansion"
        return "mid"

    typo = {}
    for c in ch_all:
        if c["n_moves"] < 5:
            continue
        t = typo.setdefault(
            chain_type(c),
            {
                "n": 0,
                "len": [],
                "own_share": [],
                "end_target": Counter(),
                "delta_egen": [],
                "start_is_general": 0,
                "end_captured": 0,
                "detour": [],
            },
        )
        t["n"] += 1
        t["len"].append(c["n_moves"])
        t["own_share"].append(c["own_share"])
        t["end_target"][c["end_target"]] += 1
        t["delta_egen"].append(c["end_dist_egen"] - c["start_dist_egen"])
        t["start_is_general"] += c["start_is_general"]
        t["end_captured"] += c["end_captured"]
        if c.get("detour_ratio"):
            t["detour"].append(c["detour_ratio"])
    agg["chain_typology_ge5"] = {
        name: {
            "n": t["n"],
            "len": dist_summary(t["len"]),
            "own_share": dist_summary(t["own_share"]),
            "end_target": dict(t["end_target"]),
            "delta_dist_egen": dist_summary(t["delta_egen"]),
            "start_is_general_rate": t["start_is_general"] / max(1, t["n"]),
            "end_captured_rate": t["end_captured"] / max(1, t["n"]),
            "detour_ratio": dist_summary(t["detour"]),
        }
        for name, t in sorted(typo.items())
    }

    # ---- move target mix (all moves, by bucket)
    tgt_all = Counter()
    tgt_bucket = {}
    for g in games:
        for m in g["move_rows"]:
            tgt_all[m["target"]] += 1
            tgt_bucket.setdefault(tick_bucket(m["t"]), Counter())[m["target"]] += 1
    agg["move_target_mix"] = {
        "overall": dict(tgt_all),
        "by_bucket": {b: dict(c) for b, c in sorted(tgt_bucket.items())},
    }

    # ---- first move tick, opening vs late pull intervals
    first_move = Counter()
    iv_open, iv_late = [], []
    pull_army_late = []
    for g in games:
        mr = g["move_rows"]
        if mr:
            first_move[mr[0]["t"]] += 1
        ts = [p["t"] for p in g["gen_pulls"]]
        for a, b in zip(ts, ts[1:]):
            (iv_open if b < 25 else iv_late).append(b - a)
        pull_army_late += [p["src_army"] for p in g["gen_pulls"] if p["t"] >= 50]
    agg["opening"] = {
        "owned_cells_t25": dist_summary([g["cells_t25"] for g in games if g["cells_t25"] is not None]),
        "owned_cells_t50": dist_summary([g["cells_t50"] for g in games if g["cells_t50"] is not None]),
        "first_move_tick_hist": {str(k): v for k, v in sorted(first_move.items())},
        "gen_pull_interval_opening_hist": hist(iv_open, cap=30),
        "gen_pull_interval_late_hist": hist(iv_late, cap=60),
        "gen_pull_interval_late": dist_summary(iv_late),
        "gen_pull_army_after_t50": dist_summary(pull_army_late),
    }

    # ---- predicate accuracies (fit set)
    n_move = sum(len(g["move_rows"]) for g in games)
    n_build = sum(1 for g in games for a in g["acts"] if a["kind"] == "build")
    n_vol_pass = sum(1 for g in games for p in g["passes"] if not p["forced"])
    ge3 = [
        m
        for g in games
        for m in g["move_rows"]
        if not m["ambiguous"] and m["src_army"] >= 3
    ]
    snake_pairs = agg["snake"]["consecutive_pairs"]
    pulls = [p for g in games for p in g["gen_pulls"]]
    agg["predicates"] = {
        "P1_always_act": {
            "rule": "if any owned cell with army>=2 has a passable neighbour, action != pass",
            "n": n_move + n_build + n_vol_pass,
            "accuracy": (n_move + n_build) / max(1, n_move + n_build + n_vol_pass),
        },
        "P2_full_send": {
            "rule": "every move uses split=0 (all-but-one) when src_army>=3",
            "n": len(ge3),
            "accuracy": sum(1 for m in ge3 if m["split"] == 0) / max(1, len(ge3)),
        },
        "P3_snake": {
            "rule": "on consecutive move ticks, src(t) == dst(t-1)",
            "n": snake_pairs,
            "accuracy": agg["snake"]["continuation_rate"],
        },
        "P3b_snake_after_own_or_capture": {
            "rule": "src(t) == dst(t-1) given prev move targeted own or captured",
            "detail": agg["snake"]["continuation_after_target"],
        },
        "P4_move_max_stack": {
            "rule": "the moved src holds the current largest stack (state t-1)",
            "n": n_move,
            "accuracy": agg["snake"]["max_stack_move_rate"],
        },
        "P5_pull_leaves_1": {
            "rule": "a move off the general leaves exactly 1 army",
            "n": len(pulls),
            "accuracy": sum(1 for p in pulls if p["src_army"] - p["moved"] == 1)
            / max(1, len(pulls)),
        },
        "P6_pull_odd_tick": {
            "rule": "moves off the general happen on odd ticks (right after even-tick production)",
            "n": len(pulls),
            "accuracy": sum(1 for p in pulls if p["t"] % 2 == 1) / max(1, len(pulls)),
        },
        "P7_first_move_t3": {
            "rule": "the first move of the game is issued at tick 3",
            "n": len(first_move),
            "accuracy": first_move.get(3, 0) / max(1, sum(first_move.values())),
        },
    }

    # ---- split context detail (opening vs late) and follow-up behaviour
    open_split = late_split = 0
    open_split_gen = late_split_gen = 0
    late_split_mod50 = Counter()
    open_split_t = Counter()
    followup = Counter()
    for g in games:
        mr = g["move_rows"]
        for i, m in enumerate(mr):
            if m["ambiguous"] or m["src_army"] < 3 or m["split"] != 1:
                continue
            if m["t"] < 25:
                open_split += 1
                open_split_gen += m["src_is_general"]
                open_split_t[m["t"]] += 1
            else:
                late_split += 1
                late_split_gen += m["src_is_general"]
                late_split_mod50[m["t"] % 50] += 1
            nxt = None
            for j in range(i + 1, min(i + 6, len(mr))):
                if mr[j]["src_t"] == m["src_t"] and mr[j]["t"] <= m["t"] + 5:
                    nxt = mr[j]
                    break
            if nxt is None:
                followup["src_idle_5t"] += 1
            elif nxt["dst_t"] == m["dst_t"]:
                followup["src_resends_same_dst"] += 1
            else:
                followup["src_moves_other_dir"] += 1
    agg["split_vs_full"]["opening_vs_late"] = {
        "open_lt25": {"n": open_split, "from_general": open_split_gen,
                      "t_hist": {str(k): v for k, v in sorted(open_split_t.items())}},
        "late_ge25": {"n": late_split, "from_general": late_split_gen,
                      "mod50_hist": {str(k): v for k, v in sorted(late_split_mod50.items())}},
        "src_followup_within_5t": dict(followup),
    }

    # ---- expansion snake exhaustion: consecutive neutral-capture runs
    run_final_moved = Counter()
    run_lengths = []
    for g in games:
        mr = [m for m in g["move_rows"]]
        run: list = []

        def flush(run):
            if len(run) >= 3:
                run_final_moved[min(run[-1]["moved"], 10)] += 1
                run_lengths.append(len(run))

        for m in mr:
            cont = run and m["t"] == run[-1]["t"] + 1 and m["src_t"] == run[-1]["dst_t"]
            if m["target"] == "neutral" and m["captured"] and (not run or cont):
                run.append(m)
            else:
                flush(run)
                run = [m] if (m["target"] == "neutral" and m["captured"]) else []
        flush(run)
    agg["expansion_snakes"] = {
        "n_runs_ge3": len(run_lengths),
        "run_length": dist_summary(run_lengths),
        "final_moved_hist_cap10": {str(k): v for k, v in sorted(run_final_moved.items())},
        "exhausted_rate_final_moved_1": run_final_moved.get(1, 0) / max(1, len(run_lengths)),
    }

    # data quality
    agg["quality"] = {
        "n_games": len(games),
        "unresolved_ticks": sum(g["unresolved"] for g in games),
        "ambiguous_ticks": sum(g["ambiguous"] for g in games),
        "total_ticks": sum(len(g["acts"]) for g in games),
    }
    return agg


def loss_summary(games: list[dict]) -> list[dict]:
    out = []
    for g in games:
        mr = g["move_rows"]
        conc_max = [mx / total for _, total, mx, *_ in g["conc"] if total > 0]
        # stranded: final-tick army-weighted distance to home
        t, total, mx, gen, cas, small = g["conc"][-1]
        out.append(
            {
                "match_id": g["match_id"],
                "outcome": g["outcome"],
                "ticks": g["total_ticks"],
                "first_contact": g["first_contact"],
                "first_gen_sight": g["first_gen_sight"],
                "n_moves": len(mr),
                "voluntary_passes": sum(1 for p in g["passes"] if not p["forced"]),
                "split_ge3": sum(1 for m in mr if m["src_army"] >= 3 and m["split"] == 1),
                "mean_max_stack_frac": float(np.mean(conc_max)) if conc_max else None,
                "final_gen_army": gen,
                "final_max_stack": mx,
                "final_total": total,
                "n_chains_ge5": sum(1 for c in g["chains"] if c["n_moves"] >= 5),
                "gen_pulls": len(g["gen_pulls"]),
                "castles_built": len(g["castles_built_by_us"]),
                "snake_rate": (
                    sum(
                        1
                        for i in range(1, len(mr))
                        if mr[i]["t"] == mr[i - 1]["t"] + 1 and mr[i]["src_t"] == mr[i - 1]["dst_t"]
                    )
                    / max(1, sum(1 for i in range(1, len(mr)) if mr[i]["t"] == mr[i - 1]["t"] + 1))
                ),
            }
        )
    return out


def main() -> None:
    fit = split_rows("fit")
    win_ids = sorted(r["match_id"] for r in fit if r["outcome"] == "win")
    other_ids = sorted(r["match_id"] for r in fit if r["outcome"] != "win")

    win_games = []
    for i, mid in enumerate(win_ids):
        win_games.append(analyze_game(mid))
        if (i + 1) % 50 == 0:
            print(f"wins {i + 1}/{len(win_ids)}", flush=True)
    other_games = [analyze_game(mid) for mid in other_ids]

    payload = {
        "meta": {
            "player": "Kubic",
            "dimension": "army_management_and_routing",
            "source": "fable_kubic_common.load_actions (exact reconstruction)",
            "split": "docs/research/measurements/fable-kubic-split.json (fit only)",
            "n_fit_wins": len(win_games),
            "n_fit_losses_draws": len(other_games),
            "params": {
                "chain_max_gap": CHAIN_MAX_GAP,
                "big_stack_min": BIG_STACK_MIN,
                "stationary_min": STATIONARY_MIN,
                "route_min_moves": ROUTE_MIN_MOVES,
                "route_min_disp": ROUTE_MIN_DISP,
                "tick_buckets": [list(b) for b in TICK_BUCKETS[:-1]] + [[400, None]],
            },
            "caveats": [
                "BFS routing uses the omniscient map (all mountains); Kubic only sees "
                "mountains inside its fog history, so detour vs its own belief may differ.",
                "split flag for src_army==2 is encoding-ambiguous (both flags move 1); "
                "those moves are excluded from split rates.",
                "dist to enemy general is an analysis-frame quantity; before "
                "first_gen_sight Kubic cannot know it (fog).",
            ],
        },
        "fit_wins": aggregate(win_games),
        "fit_losses_draws": {
            "aggregate": aggregate(other_games) if other_games else {},
            "per_game": loss_summary(other_games),
        },
    }
    OUT_PATH.write_text(json.dumps(payload, indent=1))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
