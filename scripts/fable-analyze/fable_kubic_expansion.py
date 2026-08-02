#!/usr/bin/env python3
"""
Kubic expansion / land-grab analysis (fable-kubic reverse-engineering, dim:
expansion after the opening).

Reads ONLY the fit set from docs/research/measurements/fable-kubic-split.json,
uses the derived per-tick actions cache plus raw replay frames, and writes
aggregates to docs/research/measurements/fable-kubic-expansion.json.

Deterministic: games processed in sorted match-id order, no randomness.

Tick semantics: ticks[t] is the state AFTER turn t; the action tagged t
transforms state t-1 into state t. All "state at decision time" reads use
state index t-1.

Fog: Kubic sees the 3x3 neighbourhood of every owned cell. Rules scored here
are fog-legal unless explicitly tagged "_ungated" (those exist to *test*
whether Kubic uses information it should not have).
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT))

from fable_kubic_common import load_actions, split_rows  # noqa: E402
from arena.instrument.replay.loader import open_replay  # noqa: E402

OUT_PATH = REPO_ROOT / "docs/research/measurements/fable-kubic-expansion.json"
BUCKET = 25
DIRS = {(-1, 0): "up", (1, 0): "down", (0, -1): "left", (0, 1): "right"}


# ------------------------------------------------------------------ helpers

def dilate3(mask: np.ndarray) -> np.ndarray:
    """3x3 (Chebyshev-1) dilation without scipy."""
    out = mask.copy()
    r, c = mask.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            src = mask[max(0, -dr):r - max(0, dr), max(0, -dc):c - max(0, dc)]
            out[max(0, dr):r - max(0, -dr), max(0, dc):c - max(0, -dc)] |= src
    return out


def manhattan(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def bfs_field(sources: np.ndarray, passable: np.ndarray) -> np.ndarray:
    """Multi-source BFS distance over passable cells; unreachable = big."""
    rows, cols = passable.shape
    dist = np.full((rows, cols), 10 ** 6, dtype=np.int64)
    dq = deque()
    for r, c in zip(*np.nonzero(sources)):
        dist[r, c] = 0
        dq.append((int(r), int(c)))
    while dq:
        r, c = dq.popleft()
        d = dist[r, c] + 1
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and passable[nr, nc] and dist[nr, nc] > d:
                dist[nr, nc] = d
                dq.append((nr, nc))
    return dist


def quantiles(values, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    if not values:
        return None
    arr = np.array(sorted(values), dtype=float)
    return {f"p{int(q * 100)}": round(float(np.quantile(arr, q)), 2) for q in qs}


def rate(num, den):
    return round(num / den, 4) if den else None


# ------------------------------------------------------------ per-game pass

RULES = (
    "reveal_now",        # maximize cells newly visible (vs current vision)
    "reveal_memory",     # maximize cells never seen before (memory vision)
    "fog_bfs",           # minimize BFS distance to nearest never-seen cell
    "enemy_general_gated",    # min manhattan to enemy general, only once seen
    "enemy_general_ungated",  # same WITHOUT the fog gate (legality probe)
    "board_center",      # min manhattan to board center
    "away_own_general",  # max manhattan to own general
    "straight",          # continue the stack's previous direction
    # composites: primary rule, tie broken by secondary (then still-tied counts)
    "comp_straight_else_reveal",   # straight when defined+available, else reveal_now
    "comp_reveal_tb_straight",     # reveal_now, ties -> straight
    "comp_reveal_tb_away",         # reveal_now, ties -> away_own_general
    "comp_fogbfs_tb_straight",     # fog_bfs, ties -> straight
)


def analyze_game(match_id: str, outcome: str) -> dict:
    game = load_actions(match_id)
    rep = open_replay("Kubic", match_id)
    s = game["kubic_seat"]
    opp = 1 - s
    rows, cols = rep.rows, rep.cols
    T = len(rep.ticks) - 1  # last state index
    g_own = tuple(rep.generals[s])
    g_enemy = tuple(rep.generals[opp])
    center = ((rows - 1) / 2.0, (cols - 1) / 2.0)
    passable = np.ones((rows, cols), dtype=bool)
    for r, c in rep.mountains:
        passable[r, c] = False
    playable = int(passable.sum())

    O = np.stack([np.array(f.owners, dtype=np.int8) for f in rep.ticks])
    A = np.stack([np.array(f.armies, dtype=np.int64) for f in rep.ticks])

    own = O == s
    visible = np.stack([dilate3(own[t]) for t in range(T + 1)])
    memory = np.zeros_like(visible)
    acc = np.zeros((rows, cols), dtype=bool)
    for t in range(T + 1):
        acc |= visible[t]
        memory[t] = acc

    enemy_vis_count = np.array(
        [int(((O[t] == opp) & visible[t]).sum()) for t in range(T + 1)]
    )
    land = own.sum(axis=(1, 2))
    army = np.array([int(A[t][own[t]].sum()) for t in range(T + 1)])

    # first-sight / contact state indices (exclude final transfer tick when the
    # game ended by general capture: ownership flips en masse there)
    last_state = T - 1 if rep.winner >= 0 else T
    first_enemy_seen = next(
        (t for t in range(last_state + 1) if enemy_vis_count[t] > 0), None
    )
    first_general_sight = next(
        (t for t in range(last_state + 1) if visible[t][g_enemy]), None
    )
    # orthogonal adjacency of the two territories
    first_contact = None
    for t in range(last_state + 1):
        o = O[t]
        h = (o[:, :-1] == s) & (o[:, 1:] == opp) | (o[:, :-1] == opp) & (o[:, 1:] == s)
        v = (o[:-1, :] == s) & (o[1:, :] == opp) | (o[:-1, :] == opp) & (o[1:, :] == s)
        if h.any() or v.any():
            first_contact = t
            break

    acts = [a for a in (dict(x) for x in _kubic_acts(game)) if a["kind"] != "unresolved"]

    # ---- structures (both players' builds; ownership from O)
    build_events = []  # (t, cell, player)
    for tick in game["ticks"]:
        for pkey, p in (("p0", 0), ("p1", 1)):
            a = tick[pkey]
            if a.get("kind") == "build":
                build_events.append((tick["t"], tuple(a["cell"]), p))
        for cell in tick.get("castle_recovered", []):
            build_events.append((tick["t"], tuple(cell), -1))

    def kubic_structures(state_t: int) -> list[tuple[int, int]]:
        cells = [g_own] if O[state_t][g_own] == s else []
        for bt, cell, _ in build_events:
            if bt <= state_t and O[state_t][cell] == s:
                cells.append(cell)
        return cells

    # ---- per-move walk
    tgt_bucket = defaultdict(Counter)     # bucket -> target counter
    tgt_delta = defaultdict(Counter)      # (t - first_enemy_seen) bucket
    cap_ticks = []                        # neutral capture ticks
    cap_mod50 = Counter()
    cap_moved = Counter()
    cap_srcarmy_min = 0                   # captures with src_army == moved+1
    dist_src_gown = defaultdict(list)     # bucket -> dists
    dist_dst_genemy = defaultdict(list)
    own_from_general = Counter()          # bucket -> own-moves from general
    own_moves_bucket = Counter()
    feed_next = 0                         # own-move dst == next move src
    own_moves_total = 0

    # chains
    heads: dict[tuple, int] = {}
    chains: dict[int, dict] = {}
    next_chain = 0

    rule_stats = {r: Counter() for r in RULES}  # agree / n / ties, split by phase
    tiebreak_dir = Counter()
    choice_sizes = Counter()

    prev_move_idx_by_head: dict[tuple, tuple] = {}  # head -> (t, dir)

    mix_mod50 = defaultdict(Counter)  # residue -> action-kind/target counter
    for a in acts:
        res = a["t"] % 50
        if a["kind"] == "move":
            mix_mod50[res][a["target"]] += 1
        else:
            mix_mod50[res][a["kind"]] += 1

    for i, a in enumerate(acts):
        t = a["t"]
        st = t - 1
        b = st // BUCKET
        if a["kind"] != "move":
            continue
        src, dst = tuple(a["src"]), tuple(a["dst"])
        d = (dst[0] - src[0], dst[1] - src[1])
        tgt_bucket[b][a["target"]] += 1
        if first_enemy_seen is not None:
            db = (t - first_enemy_seen) // BUCKET
            tgt_delta[db][a["target"]] += 1
        dist_src_gown[b].append(manhattan(src, g_own))
        if a["target"] == "own":
            own_moves_bucket[b] += 1
            own_moves_total += 1
            if src == g_own:
                own_from_general[b] += 1
            nxt = next((x for x in acts[i + 1:] if x["kind"] == "move"), None)
            if nxt and tuple(nxt["src"]) == dst:
                feed_next += 1

        # chain bookkeeping
        if src in heads:
            cid = heads.pop(src)
        else:
            cid = next_chain
            next_chain += 1
            chains[cid] = {"len": 0, "caps": 0, "start": t, "src0": src,
                           "gaps": [], "last_t": None}
        ch = chains[cid]
        if ch["last_t"] is not None:
            ch["gaps"].append(t - ch["last_t"])
        ch["last_t"] = t
        ch["len"] += 1
        if a.get("captured"):
            ch["caps"] += 1
        prev_dir = prev_move_idx_by_head.pop(src, None)
        heads[dst] = cid
        prev_move_idx_by_head[dst] = d

        if a["target"] == "neutral" and a.get("captured"):
            cap_ticks.append(t)
            cap_mod50[t % 50] += 1
            cap_moved[min(a["moved"], 30)] += 1
            if a["moved"] == 1:
                cap_srcarmy_min += 1  # minimum-force capture (1 unit onto 0)
            if first_general_sight is not None and st >= first_general_sight:
                dist_dst_genemy[b].append(manhattan(dst, g_enemy))

            # ---- direction-choice rule scoring
            alts = []
            for dd in DIRS:
                nr, nc = src[0] + dd[0], src[1] + dd[1]
                if 0 <= nr < rows and 0 <= nc < cols and passable[nr, nc] \
                        and O[st][nr, nc] == -1:
                    alts.append((nr, nc))
            if len(alts) >= 2:
                choice_sizes[len(alts)] += 1
                pre = first_enemy_seen is None or st < first_enemy_seen
                phase = "pre" if pre else "post"
                vis, mem = visible[st], memory[st]

                def reveal(cell, ref):
                    n = 0
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            rr, cc = cell[0] + dr, cell[1] + dc
                            if 0 <= rr < rows and 0 <= cc < cols and not ref[rr, cc]:
                                n += 1
                    return n

                fog_dist = None
                unseen = (~mem) & passable
                if unseen.any():
                    fog_dist = bfs_field(unseen, passable)

                scores = {
                    "reveal_now": {c: reveal(c, vis) for c in alts},
                    "reveal_memory": {c: reveal(c, mem) for c in alts},
                    "fog_bfs": (
                        {c: -int(fog_dist[c]) for c in alts} if fog_dist is not None else None
                    ),
                    "enemy_general_gated": (
                        {c: -manhattan(c, g_enemy) for c in alts}
                        if first_general_sight is not None and st >= first_general_sight
                        else None
                    ),
                    "enemy_general_ungated": {c: -manhattan(c, g_enemy) for c in alts},
                    "board_center": {
                        c: -(abs(c[0] - center[0]) + abs(c[1] - center[1])) for c in alts
                    },
                    "away_own_general": {c: manhattan(c, g_own) for c in alts},
                    "straight": (
                        {c: 1 if (c[0] - src[0], c[1] - src[1]) == prev_dir else 0
                         for c in alts} if prev_dir is not None else None
                    ),
                }
                # composite scores: primary + tie-break
                def with_tb(primary, tb):
                    if primary is None:
                        return None
                    if tb is None:
                        return primary
                    return {c: (primary[c], tb[c]) for c in primary}

                straight_sc = scores["straight"]
                straight_available = (
                    straight_sc is not None and max(straight_sc.values()) == 1
                )
                scores["comp_straight_else_reveal"] = (
                    straight_sc if straight_available else scores["reveal_now"]
                )
                scores["comp_reveal_tb_straight"] = with_tb(
                    scores["reveal_now"], straight_sc)
                scores["comp_reveal_tb_away"] = with_tb(
                    scores["reveal_now"], scores["away_own_general"])
                scores["comp_fogbfs_tb_straight"] = with_tb(
                    scores["fog_bfs"], straight_sc)

                for rname, sc in scores.items():
                    if sc is None:
                        continue
                    best = max(sc.values())
                    argmax = [c for c, v in sc.items() if v == best]
                    st_c = rule_stats[rname]
                    st_c["n"] += 1
                    st_c[f"n_{phase}"] += 1
                    if dst in argmax:
                        st_c["agree"] += 1
                        st_c[f"agree_{phase}"] += 1
                        if len(argmax) > 1:
                            st_c["agree_tied"] += 1
                    if len(argmax) > 1:
                        st_c["tied"] += 1
                # tie-break: when reveal_memory ties and chosen among ties,
                # record chosen direction
                sc = scores["reveal_memory"]
                best = max(sc.values())
                argmax = [c for c, v in sc.items() if v == best]
                if len(argmax) > 1 and dst in argmax:
                    tiebreak_dir[DIRS[d]] += 1

    # ---- castles
    castle_recs = []
    kubic_builds = [(a["t"], tuple(a["cell"]), a["cost"]) for a in acts if a["kind"] == "build"]
    for bt, cell, cost in kubic_builds:
        st = bt - 1
        structs = kubic_structures(st)
        d_near = min((manhattan(cell, x) for x in structs), default=None)
        # army walked in during the previous 15 ticks
        walked = 0
        walk_moves = 0
        for a in acts:
            if a["kind"] == "move" and bt - 15 <= a["t"] < bt and tuple(a["dst"]) == cell:
                walked += a["moved"]
                walk_moves += 1
        # nearest non-own passable cell (frontier proximity)
        not_own = passable & (O[st] != s)
        front = bfs_field(not_own, passable) if not_own.any() else None
        castle_recs.append({
            "t": bt, "cost": cost,
            "dist_nearest_structure": d_near,
            "dist_own_general": manhattan(cell, g_own),
            "dist_enemy_general": manhattan(cell, g_enemy),
            "generals_dist": manhattan(g_own, g_enemy),
            "army_on_cell_before": int(A[st][cell]),
            "remainder": int(A[st][cell]) - cost,
            "walked_in_15": walked, "walk_moves_15": walk_moves,
            "dist_frontier": int(front[cell]) if front is not None else None,
            "land": int(land[st]), "army_total": int(army[st]),
            "enemy_seen": bool(first_enemy_seen is not None and st >= first_enemy_seen),
            "mod50": bt % 50, "mod2": bt % 2,
        })

    # ---- stopping / re-expansion
    last_cap = cap_ticks[-1] if cap_ticks else None
    cap95 = None
    if cap_ticks:
        cap95 = cap_ticks[max(0, int(np.ceil(len(cap_ticks) * 0.95)) - 1)]
    pre_ticks = post_ticks = pre_caps = post_caps = 0
    if first_enemy_seen is not None:
        pre_ticks = max(0, first_enemy_seen - 1)
        post_ticks = max(0, last_state - first_enemy_seen)
        pre_caps = sum(1 for t in cap_ticks if t - 1 < first_enemy_seen)
        post_caps = len(cap_ticks) - pre_caps
    else:
        pre_ticks = last_state
        pre_caps = len(cap_ticks)

    # after first enemy sighting: capture rate when enemy visible vs not
    vis0_ticks = vis0_caps = vispos_ticks = vispos_caps = 0
    lull_runs = lull_runs_with_cap = 0
    if first_enemy_seen is not None:
        cap_set = set(cap_ticks)
        run = 0
        run_caps = 0
        for t in range(first_enemy_seen + 1, last_state + 1):
            visnow = enemy_vis_count[t - 1] > 0
            iscap = t in cap_set
            if visnow:
                vispos_ticks += 1
                vispos_caps += iscap
                if run >= 10:
                    lull_runs += 1
                    lull_runs_with_cap += run_caps > 0
                run = 0
                run_caps = 0
            else:
                vis0_ticks += 1
                vis0_caps += iscap
                run += 1
                run_caps += iscap
        if run >= 10:
            lull_runs += 1
            lull_runs_with_cap += run_caps > 0

    # land checkpoints
    land_ck = {}
    for ck in (25, 50, 75, 100, 150, 200, 300, 400):
        if ck <= last_state:
            land_ck[ck] = round(float(land[ck]) / playable, 4)

    # build/no-build discriminator snapshot at state 120 (if alive)
    snap120 = None
    if last_state >= 120:
        snap120 = {
            "land": int(land[120]), "army": int(army[120]),
            "enemy_vis": int(enemy_vis_count[120]),
            "generals_dist": manhattan(g_own, g_enemy),
        }

    return {
        "match_id": match_id, "outcome": outcome, "T": T, "playable": playable,
        "mix_mod50": {str(k): dict(v) for k, v in mix_mod50.items()},
        "snap120": snap120,
        "built": bool(kubic_builds),
        "first_build_t": kubic_builds[0][0] if kubic_builds else None,
        "first_enemy_seen": first_enemy_seen,
        "first_general_sight": first_general_sight,
        "first_contact": first_contact,
        "land_frac_ck": land_ck,
        "land_final_frac": round(float(land[last_state]) / playable, 4),
        "tgt_bucket": {str(k): dict(v) for k, v in tgt_bucket.items()},
        "tgt_delta": {str(k): dict(v) for k, v in tgt_delta.items()},
        "cap_ticks_n": len(cap_ticks),
        "cap_mod50": dict(cap_mod50),
        "cap_moved": dict(cap_moved),
        "cap_min_stack": cap_srcarmy_min,
        "cap_exposure": (
            [cap_ticks[0], cap_ticks[-1]] if cap_ticks else None
        ),
        "last_cap": last_cap, "cap95": cap95,
        "pre": {"ticks": pre_ticks, "caps": pre_caps},
        "post": {"ticks": post_ticks, "caps": post_caps},
        "vis0": {"ticks": vis0_ticks, "caps": vis0_caps},
        "vispos": {"ticks": vispos_ticks, "caps": vispos_caps},
        "lulls": {"runs": lull_runs, "with_cap": lull_runs_with_cap},
        "chains": [
            {"len": c["len"], "caps": c["caps"],
             "mean_gap": round(float(np.mean(c["gaps"])), 2) if c["gaps"] else None}
            for c in chains.values()
        ],
        "rules": {r: dict(v) for r, v in rule_stats.items()},
        "tiebreak_dir": dict(tiebreak_dir),
        "choice_sizes": dict(choice_sizes),
        "dist_src_gown": {str(k): quantiles(v) for k, v in dist_src_gown.items()},
        "dist_dst_genemy": {str(k): quantiles(v) for k, v in dist_dst_genemy.items()},
        "own_from_general": dict(own_from_general),
        "own_moves_bucket": dict(own_moves_bucket),
        "feed_next": feed_next, "own_moves_total": own_moves_total,
        "castles": castle_recs,
    }


def _kubic_acts(game: dict):
    key = f"p{game['kubic_seat']}"
    for tick in game["ticks"]:
        a = dict(tick[key])
        a["t"] = tick["t"]
        yield a


# ---------------------------------------------------------------- aggregate

def aggregate(per_game: list[dict]) -> dict:
    wins = [g for g in per_game if g["outcome"] == "win"]
    nonwins = [g for g in per_game if g["outcome"] != "win"]

    def agg_subset(games: list[dict]) -> dict:
        # target mix per bucket
        mix = defaultdict(Counter)
        for g in games:
            for b, c in g["tgt_bucket"].items():
                mix[int(b)].update(c)
        mix_out = {}
        for b in sorted(mix):
            c = mix[b]
            tot = sum(c.values())
            mix_out[str(b * BUCKET)] = {
                "moves": tot,
                **{k: rate(c.get(k, 0), tot) for k in
                   ("neutral", "own", "enemy", "enemy_general")},
            }
        # delta-vs-first-enemy-seen mix
        dmix = defaultdict(Counter)
        for g in games:
            for b, c in g["tgt_delta"].items():
                dmix[int(b)].update(c)
        dmix_out = {}
        for b in sorted(dmix):
            c = dmix[b]
            tot = sum(c.values())
            if tot < 50:
                continue
            dmix_out[str(b * BUCKET)] = {
                "moves": tot,
                **{k: rate(c.get(k, 0), tot) for k in
                   ("neutral", "own", "enemy", "enemy_general")},
            }
        # land curve
        land_ck = defaultdict(list)
        for g in games:
            for ck, v in g["land_frac_ck"].items():
                land_ck[int(ck)].append(v)
        # rules
        rules_out = {}
        for r in RULES:
            tot = Counter()
            for g in games:
                tot.update(g["rules"].get(r, {}))
            rules_out[r] = {
                "n": tot["n"], "agree_rate": rate(tot["agree"], tot["n"]),
                "tie_rate": rate(tot["tied"], tot["n"]),
                "agree_untied_rate": rate(
                    tot["agree"] - tot["agree_tied"], tot["n"] - tot["tied"]
                ) if tot["n"] > tot["tied"] else None,
                "pre": {"n": tot["n_pre"], "agree_rate": rate(tot["agree_pre"], tot["n_pre"])},
                "post": {"n": tot["n_post"], "agree_rate": rate(tot["agree_post"], tot["n_post"])},
            }
        tb = Counter()
        cs = Counter()
        for g in games:
            tb.update(g["tiebreak_dir"])
            cs.update({int(k): v for k, v in g["choice_sizes"].items()})
        # mod 50
        mod = Counter()
        exposure = Counter()
        for g in games:
            mod.update({int(k): v for k, v in g["cap_mod50"].items()})
            if g["cap_exposure"]:
                a, b = g["cap_exposure"]
                for t in range(a, b + 1):
                    exposure[t % 50] += 1
        mod_rate = {}
        for band in ((0, 9), (10, 19), (20, 29), (30, 39), (40, 49)):
            caps = sum(mod[x] for x in range(band[0], band[1] + 1))
            exp = sum(exposure[x] for x in range(band[0], band[1] + 1))
            mod_rate[f"{band[0]}-{band[1]}"] = {
                "caps": caps, "exposure": exp, "rate": rate(caps, exp)
            }
        moved = Counter()
        minstack = 0
        capn = 0
        for g in games:
            moved.update({int(k): v for k, v in g["cap_moved"].items()})
            minstack += g["cap_min_stack"]
            capn += g["cap_ticks_n"]
        # chains
        chain_lens = Counter()
        chain_gaps = []
        long_chain_caps = []
        for g in games:
            for c in g["chains"]:
                chain_lens[min(c["len"], 40)] += 1
                if c["mean_gap"] is not None:
                    chain_gaps.append(c["mean_gap"])
                if c["len"] >= 5:
                    long_chain_caps.append(c["caps"] / c["len"])
        # stopping
        pre_t = sum(g["pre"]["ticks"] for g in games)
        pre_c = sum(g["pre"]["caps"] for g in games)
        post_t = sum(g["post"]["ticks"] for g in games)
        post_c = sum(g["post"]["caps"] for g in games)
        v0t = sum(g["vis0"]["ticks"] for g in games)
        v0c = sum(g["vis0"]["caps"] for g in games)
        vpt = sum(g["vispos"]["ticks"] for g in games)
        vpc = sum(g["vispos"]["caps"] for g in games)
        lull_r = sum(g["lulls"]["runs"] for g in games)
        lull_w = sum(g["lulls"]["with_cap"] for g in games)
        last_cap_rel = [g["last_cap"] / g["T"] for g in games if g["last_cap"]]
        last_minus_seen = [
            g["last_cap"] - g["first_enemy_seen"]
            for g in games if g["last_cap"] and g["first_enemy_seen"] is not None
        ]
        # castles
        all_castles = [c for g in games for c in g["castles"]]
        per_game_counts = Counter(len(g["castles"]) for g in games)
        first_builds = [g["castles"][0] for g in games if g["castles"]]
        intervals = []
        for g in games:
            ts = [c["t"] for c in g["castles"]]
            intervals += [b - a for a, b in zip(ts, ts[1:])]
        nobuild_long = [g for g in games if not g["castles"] and g["T"] >= 130]
        build_games = [g for g in games if g["castles"]]
        castles_out = {
            "per_game_counts": {str(k): v for k, v in sorted(per_game_counts.items())},
            "n_builds": len(all_castles),
            "cost_hist": dict(Counter(c["cost"] for c in all_castles)),
            "dist_nearest_structure_hist": dict(Counter(
                c["dist_nearest_structure"] for c in all_castles)),
            "first_build_tick": quantiles([c["t"] for c in first_builds]),
            "first_build_tick_hist": dict(Counter(c["t"] for c in first_builds)),
            "build_tick_all": quantiles([c["t"] for c in all_castles]),
            "intervals": quantiles(intervals),
            "intervals_hist": dict(Counter(intervals)),
            "dist_own_general": quantiles([c["dist_own_general"] for c in all_castles]),
            "dist_enemy_general_norm": quantiles([
                round(c["dist_enemy_general"] / c["generals_dist"], 3)
                for c in all_castles if c["generals_dist"]]),
            "dist_frontier": dict(Counter(c["dist_frontier"] for c in all_castles)),
            "army_on_cell_before": quantiles([c["army_on_cell_before"] for c in all_castles]),
            "remainder": quantiles([c["remainder"] for c in all_castles]),
            "remainder_hist_capped": dict(Counter(
                min(c["remainder"], 20) for c in all_castles)),
            "walked_in_15": quantiles([c["walked_in_15"] for c in all_castles]),
            "walk_moves_15": dict(Counter(c["walk_moves_15"] for c in all_castles)),
            "enemy_seen_at_build": rate(
                sum(c["enemy_seen"] for c in all_castles), len(all_castles)),
            "mod2": dict(Counter(c["mod2"] for c in all_castles)),
            "mod50": quantiles([c["mod50"] for c in all_castles]),
            "land_at_first_build": quantiles([c["land"] for c in first_builds]),
            "army_at_first_build": quantiles([c["army_total"] for c in first_builds]),
            "games_with_builds": len(build_games),
            "games_T>=130_no_build": len(nobuild_long),
            "T_hist_no_build_long": quantiles([g["T"] for g in nobuild_long]),
            "first_enemy_seen_no_build_long": quantiles([
                g["first_enemy_seen"] for g in nobuild_long
                if g["first_enemy_seen"] is not None]),
            "first_enemy_seen_build_games": quantiles([
                g["first_enemy_seen"] for g in build_games
                if g["first_enemy_seen"] is not None]),
        }
        # action mix by t % 50 residue
        m50 = defaultdict(Counter)
        for g in games:
            for res, c in g["mix_mod50"].items():
                m50[int(res)].update(c)
        mix_mod50_out = {}
        for res in sorted(m50):
            c = m50[res]
            tot = sum(c.values())
            mix_mod50_out[str(res)] = {
                "n": tot,
                **{k: rate(c.get(k, 0), tot) for k in
                   ("neutral", "own", "enemy", "enemy_general", "pass", "build")},
            }
        # build vs no-build discriminator among games alive at state 120
        alive = [g for g in games if g["snap120"] is not None]
        disc = {}
        for label, sub in (("built", [g for g in alive if g["built"]]),
                           ("no_build", [g for g in alive if not g["built"]])):
            disc[label] = {
                "games": len(sub),
                "T": quantiles([g["T"] for g in sub]),
                "land120": quantiles([g["snap120"]["land"] for g in sub]),
                "army120": quantiles([g["snap120"]["army"] for g in sub]),
                "enemy_vis120": quantiles([g["snap120"]["enemy_vis"] for g in sub]),
                "generals_dist": quantiles([g["snap120"]["generals_dist"] for g in sub]),
                "land_final_frac": quantiles([g["land_final_frac"] for g in sub]),
            }
        castles_out["build_discriminator_alive_at_120"] = disc
        castles_out["build_mod50_hist"] = dict(Counter(
            c["mod50"] for c in all_castles))
        # feed / conveyor
        feed = sum(g["feed_next"] for g in games)
        ownm = sum(g["own_moves_total"] for g in games)
        ofg = Counter()
        omb = Counter()
        for g in games:
            ofg.update({int(k): v for k, v in g["own_from_general"].items()})
            omb.update({int(k): v for k, v in g["own_moves_bucket"].items()})
        from_general = {
            str(b * BUCKET): rate(ofg.get(b, 0), omb[b])
            for b in sorted(omb) if omb[b] >= 100
        }
        # distances
        dsg = defaultdict(list)
        for g in games:
            for b, q in g["dist_src_gown"].items():
                if q:
                    dsg[int(b)].append(q["p50"])
        return {
            "games": len(games),
            "action_mix_by_mod50": mix_mod50_out,
            "target_mix_by_tick_bucket": mix_out,
            "target_mix_by_delta_first_enemy_seen": dmix_out,
            "land_frac_at_tick": {
                str(k): quantiles(v) for k, v in sorted(land_ck.items())
            },
            "land_final_frac": quantiles([g["land_final_frac"] for g in games]),
            "capture_rules": rules_out,
            "tiebreak_dir_when_reveal_memory_tied": dict(tb),
            "choice_set_sizes": {str(k): v for k, v in sorted(cs.items())},
            "cap_mod50_rate": mod_rate,
            "cap_mod50_hist": {str(k): mod[k] for k in sorted(mod)},
            "cap_moved_hist": {str(k): moved[k] for k in sorted(moved)},
            "cap_min_stack_rate": rate(minstack, capn),
            "captures_total": capn,
            "chain_len_hist": {str(k): v for k, v in sorted(chain_lens.items())},
            "chain_mean_gap": quantiles(chain_gaps),
            "long_chain_capture_frac": quantiles(long_chain_caps),
            "stopping": {
                "rate_before_first_enemy_seen": rate(pre_c, pre_t),
                "rate_after_first_enemy_seen": rate(post_c, post_t),
                "rate_enemy_visible_now": rate(vpc, vpt),
                "rate_no_enemy_visible_now": rate(v0c, v0t),
                "counts": {"pre": [pre_c, pre_t], "post": [post_c, post_t],
                           "vis0": [v0c, v0t], "vispos": [vpc, vpt]},
                "lull_runs_ge10": lull_r,
                "lull_runs_with_capture": lull_w,
                "last_capture_frac_of_game": quantiles(last_cap_rel),
                "last_capture_minus_first_enemy_seen": quantiles(last_minus_seen),
                "first_enemy_seen": quantiles([
                    g["first_enemy_seen"] for g in games
                    if g["first_enemy_seen"] is not None]),
                "first_general_sight": quantiles([
                    g["first_general_sight"] for g in games
                    if g["first_general_sight"] is not None]),
                "first_contact": quantiles([
                    g["first_contact"] for g in games
                    if g["first_contact"] is not None]),
            },
            "castles": castles_out,
            "conveyor": {
                "own_move_feeds_next_move_src_rate": rate(feed, ownm),
                "own_moves_from_general_by_bucket": from_general,
            },
            "dist_src_to_own_general_median_by_bucket": {
                str(k * BUCKET): quantiles(v) for k, v in sorted(dsg.items())
            },
        }

    return {
        "n_games": len(per_game),
        "wins": agg_subset(wins),
        "nonwins": agg_subset(nonwins),
        "nonwin_ids": [g["match_id"] for g in nonwins],
    }


def main() -> None:
    rows = sorted(split_rows("fit"), key=lambda r: r["match_id"])
    per_game = []
    for i, r in enumerate(rows):
        per_game.append(analyze_game(r["match_id"], r["outcome"]))
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(rows)}", flush=True)
    out = aggregate(per_game)
    out["per_game_compact"] = [
        {k: g[k] for k in (
            "match_id", "outcome", "T", "first_enemy_seen", "first_general_sight",
            "first_contact", "last_cap", "cap95", "cap_ticks_n", "land_final_frac")}
        for g in per_game
    ]

    def sanitize(o):
        if isinstance(o, dict):
            return {str(k): sanitize(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [sanitize(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return o

    OUT_PATH.write_text(json.dumps(sanitize(out), indent=1))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
