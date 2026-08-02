#!/usr/bin/env python3
"""
Kubic defense/reaction analysis (fable round, analyst: DEFENSE AND REACTION).

Measures, from exact reconstructed per-tick actions (fable_kubic_common) plus
board state (arena.instrument.replay), how Kubic responds to incursions:
threat episodes and response latency/class, recall triggers, abandonment of
expansion under pressure, chase-rule usage, far-territory defense, castle
defense, passes under pressure, and tick-by-tick accounts of every fit loss.

Fit set only (docs/research/measurements/fable-kubic-split.json, set=="fit").
Writes docs/research/measurements/fable-kubic-defense.json. Deterministic.

Conventions:
- frames are indexed 0..T (rep.ticks); the action at tick t transforms frame
  t-1 into frame t, so the decision state for action t is frame t-1.
- distances are BFS over non-mountain cells ("bfs" units); manhattan is also
  reported where a threshold is being located.
- fog-legality: threat features use only enemy cells inside Kubic's 3x3
  vision at the decision frame (min_vis_dist); the omniscient distance
  (true_min_dist) is computed alongside to test whether vision gates behavior.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fable_kubic_common import REPO_ROOT, load_actions, split_rows

from arena.instrument.replay.loader import open_replay

OUT_PATH = REPO_ROOT / "docs/research/measurements/fable-kubic-defense.json"

INF = 10_000
D_REF = 6            # episode-defining visible-threat radius (bfs)
EP_GAP = 10          # frames beyond D_REF that close an episode
RESP_WINDOW = 25     # ticks after onset scanned for the first defensive move
CONTEST_WINDOW = 15  # ticks after a tile capture scanned for a contest
BIG_MOVE = 8         # moved-army floor for a "recall"-grade move
CHAIN_MIN_DROP = 3   # net bfs-dist drop for a homeward chain to count as recall


# ------------------------------------------------------------- small helpers

def dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def bfs_dist(passable: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    rows, cols = passable.shape
    dist = np.full((rows, cols), INF, dtype=np.int64)
    if not passable[start]:
        return dist
    dist[start] = 0
    q = deque([start])
    while q:
        r, c = q.popleft()
        d = dist[r, c] + 1
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < rows and 0 <= nc < cols and passable[nr, nc] and dist[nr, nc] > d:
                dist[nr, nc] = d
                q.append((nr, nc))
    return dist


def quantiles(xs, ps=(0.1, 0.25, 0.5, 0.75, 0.9)):
    if not xs:
        return None
    arr = np.array(sorted(xs), dtype=float)
    return {f"p{int(p * 100)}": float(np.quantile(arr, p)) for p in ps} | {
        "n": len(xs), "mean": float(arr.mean()), "min": float(arr.min()), "max": float(arr.max()),
    }


def rate(num, den):
    return {"num": num, "den": den, "rate": (num / den) if den else None}


# ------------------------------------------------------------ per-game state

class GameState:
    """Frame grids plus per-frame derived threat features for one game."""

    def __init__(self, row: dict):
        mid = row["match_id"]
        self.row = row
        self.game = load_actions(mid)
        self.rep = open_replay("Kubic", mid)
        self.me = self.game["kubic_seat"]
        self.opp = 1 - self.me
        self.T = len(self.rep.ticks) - 1
        self.rows, self.cols = self.rep.rows, self.rep.cols
        self.G_me = tuple(self.game["generals"][self.me])
        self.G_opp = tuple(self.game["generals"][self.opp])
        self.passable = np.ones((self.rows, self.cols), dtype=bool)
        for r, c in self.rep.mountains:
            self.passable[r, c] = False
        self.dist_me = bfs_dist(self.passable, self.G_me)
        rr, cc = np.meshgrid(np.arange(self.rows), np.arange(self.cols), indexing="ij")
        self.manh_me = np.abs(rr - self.G_me[0]) + np.abs(cc - self.G_me[1])

        self.owners = [np.array(f.owners, dtype=np.int64) for f in self.rep.ticks]
        self.armies = [np.array(f.armies, dtype=np.int64) for f in self.rep.ticks]

        # actions by tick (1..T)
        self.k_act = {}
        self.o_act = {}
        kk, ok = f"p{self.me}", f"p{self.opp}"
        for tick in self.game["ticks"]:
            self.k_act[tick["t"]] = tick[kk]
            self.o_act[tick["t"]] = tick[ok]

        # per-frame threat features
        n = self.T + 1
        self.min_vis_dist = np.full(n, INF, dtype=np.int64)
        self.min_vis_manh = np.full(n, INF, dtype=np.int64)
        self.min_true_dist = np.full(n, INF, dtype=np.int64)
        self.threat_cell = [None] * n          # nearest visible enemy cell (max army tiebreak)
        self.threat_army = np.zeros(n, dtype=np.int64)
        self.near_enemy_army = np.zeros(n, dtype=np.int64)  # max visible enemy army with dist<=D_REF
        self.gen_army = np.zeros(n, dtype=np.int64)
        self.main_cell = [None] * n            # largest own stack off-general
        self.main_dist = np.full(n, INF, dtype=np.int64)
        self.main_army = np.zeros(n, dtype=np.int64)
        self.my_tiles = np.zeros(n, dtype=np.int64)
        self.my_total = np.zeros(n, dtype=np.int64)
        self.opp_total = np.zeros(n, dtype=np.int64)

        for t in range(n):
            O, A = self.owners[t], self.armies[t]
            own = O == self.me
            self.my_tiles[t] = int(own.sum())
            self.my_total[t] = int(A[own].sum())
            self.opp_total[t] = int(A[O == self.opp].sum())
            self.gen_army[t] = int(A[self.G_me]) if own[self.G_me] else 0
            vis = dilate8(own)
            enemy = O == self.opp
            vis_enemy = enemy & vis
            if enemy.any():
                self.min_true_dist[t] = int(self.dist_me[enemy].min())
            if vis_enemy.any():
                dmin = int(self.dist_me[vis_enemy].min())
                self.min_vis_dist[t] = dmin
                self.min_vis_manh[t] = int(self.manh_me[vis_enemy].min())
                cells = np.argwhere(vis_enemy & (self.dist_me == dmin))
                best = max(
                    (int(A[r, c]), -r, -c) for r, c in cells
                )
                self.threat_cell[t] = (-best[1], -best[2])
                self.threat_army[t] = best[0]
                near = vis_enemy & (self.dist_me <= D_REF)
                if near.any():
                    self.near_enemy_army[t] = int(A[near].max())
            own_off = own.copy()
            own_off[self.G_me] = False
            if own_off.any():
                a_off = np.where(own_off, A, -1)
                idx = np.unravel_index(int(a_off.argmax()), a_off.shape)
                self.main_cell[t] = (int(idx[0]), int(idx[1]))
                self.main_dist[t] = int(self.dist_me[idx])
                self.main_army[t] = int(A[idx])

    # -- move geometry helpers -------------------------------------------

    def homeward(self, act) -> bool:
        return (
            act["kind"] == "move"
            and self.dist_me[tuple(act["dst"])] < self.dist_me[tuple(act["src"])]
        )

    def is_chase(self, t: int) -> bool:
        """Kubic's move this tick targets the cell the opponent's move leaves."""
        k, o = self.k_act.get(t), self.o_act.get(t)
        return (
            k is not None and o is not None
            and k["kind"] == "move" and o["kind"] == "move"
            and k["dst"] == o["src"]
        )


# -------------------------------------------------------- episode extraction

def find_episodes(gs: GameState) -> list[dict]:
    """Maximal visible-threat intervals: min_vis_dist <= D_REF, gap-closed."""
    eps = []
    t = 1
    n = gs.T + 1
    while t < n:
        if gs.min_vis_dist[t] <= D_REF and (t == 1 or gs.min_vis_dist[t - 1] > D_REF):
            # check it's a fresh onset (no threat in the previous EP_GAP frames)
            lo = max(0, t - EP_GAP)
            if all(gs.min_vis_dist[u] > D_REF for u in range(lo, t)):
                # find end: EP_GAP consecutive frames beyond D_REF
                end = t
                u = t
                clear = 0
                while u < n - 1:
                    u += 1
                    if gs.min_vis_dist[u] <= D_REF:
                        end = u
                        clear = 0
                    else:
                        clear += 1
                        if clear >= EP_GAP:
                            break
                eps.append({"t0": int(t), "t1": int(end)})
                t = u
        t += 1
    return eps


def track_intruder(gs: GameState, t0: int, t_end: int) -> dict[int, tuple[int, int]]:
    """Follow the onset threat stack through opponent moves, frame-indexed."""
    pos = {t0: gs.threat_cell[t0]}
    cur = gs.threat_cell[t0]
    for t in range(t0 + 1, min(t_end, gs.T) + 1):
        o = gs.o_act.get(t)
        if o and o["kind"] == "move" and tuple(o["src"]) == cur and o["moved"] > 0:
            cur = tuple(o["dst"])
        pos[t] = cur
    return pos


def classify_response(gs: GameState, ep: dict) -> dict:
    """First defensive move after onset, its class, and what continued before it."""
    t0 = ep["t0"]
    t_hi = min(t0 + RESP_WINDOW, gs.T)
    intr = track_intruder(gs, t0, t_hi)
    onset_cell = gs.threat_cell[t0]
    d_from_intr = bfs_dist(gs.passable, onset_cell)
    d_intr0 = int(gs.dist_me[onset_cell])

    first_def = None
    classes_hit = {}
    pre_expand = 0
    pre_pass = 0
    pre_build = 0
    for t in range(t0 + 1, t_hi + 1):
        k = gs.k_act.get(t)
        if k is None:
            continue
        f = t - 1  # decision frame
        cls = None
        if k["kind"] == "move":
            src, dst = tuple(k["src"]), tuple(k["dst"])
            intr_f = intr.get(f, onset_cell)
            if dst == intr_f:
                o = gs.o_act.get(t)
                if o and o["kind"] == "move" and tuple(o["src"]) == intr_f:
                    cls = "chase_intruder"
                else:
                    cls = "attack_intruder"
            elif k["target"] in ("enemy", "enemy_general") and gs.dist_me[dst] <= D_REF + 2:
                cls = "attack_near_home"
            elif dst == gs.G_me:
                cls = "reinforce_general"
            elif (
                gs.homeward(k)
                and gs.dist_me[dst] + d_from_intr[dst] <= d_intr0 + 2
                and gs.dist_me[dst] <= d_intr0
            ):
                cls = "block_path"
            elif gs.homeward(k) and k["moved"] >= BIG_MOVE:
                cls = "recall_homeward"
        if cls is not None:
            classes_hit.setdefault(cls, t - (t0 + 1))
            if first_def is None:
                first_def = {"tick": t, "latency": t - (t0 + 1), "class": cls,
                             "moved": k.get("moved"), "dst": k.get("dst")}
        else:
            if first_def is None:
                if k["kind"] == "pass":
                    pre_pass += 1
                elif k["kind"] == "build":
                    pre_build += 1
                elif k["target"] == "neutral" or (
                    k["target"] in ("enemy", "enemy_general") and gs.dist_me[tuple(k["dst"])] > D_REF + 2
                ):
                    pre_expand += 1

    # episode outcome
    outcome = "game_end"
    t1 = ep["t1"]
    for t in range(t0, min(t1, gs.T) + 1):
        c = intr.get(t)
        if c is not None and gs.owners[t][c] != gs.opp:
            outcome = "intruder_lost_cell"
            break
    else:
        if t1 < gs.T - EP_GAP:
            outcome = "threat_left_radius"
    if gs.game["outcome"] == "lose" and t1 >= gs.T - 1:
        outcome = "general_captured"

    tiles_lost = 0
    for t in range(t0 + 1, min(t1, gs.T) + 1):
        lost = (gs.owners[t - 1] == gs.me) & (gs.owners[t] == gs.opp)
        tiles_lost += int(lost.sum())

    # moves leaving the general during the episode
    src_gen_moves = sum(
        1 for t in range(t0 + 1, min(t1, gs.T) + 1)
        if (k := gs.k_act.get(t)) and k["kind"] == "move" and tuple(k["src"]) == gs.G_me
    )

    return {
        "t0": t0, "t1": t1,
        "onset_dist": d_intr0,
        "onset_manh": int(gs.manh_me[onset_cell]),
        "onset_threat_army": int(gs.threat_army[t0]),
        "onset_gen_army": int(gs.gen_army[t0]),
        "onset_main_dist": None if gs.main_dist[t0] >= INF else int(gs.main_dist[t0]),
        "onset_main_army": int(gs.main_army[t0]),
        "min_dist_reached": int(min(gs.min_vis_dist[t0:min(t1, gs.T) + 1])),
        "first_def": first_def,
        "classes_hit": classes_hit,
        "pre_expand": pre_expand, "pre_pass": pre_pass, "pre_build": pre_build,
        "outcome": outcome,
        "tiles_lost": tiles_lost,
        "src_general_moves": src_gen_moves,
    }


# --------------------------------------------------------- recall chain scan

def find_homeward_chains(gs: GameState) -> list[dict]:
    """Carry chains (src == previous dst) of big moves with net homeward drop."""
    chains = []
    t = 1
    used = set()
    while t <= gs.T:
        k = gs.k_act.get(t)
        if (
            t not in used and k and k["kind"] == "move" and k["moved"] >= BIG_MOVE
            and gs.homeward(k)
        ):
            start_t = t
            src0 = tuple(k["src"])
            cur = tuple(k["dst"])
            ticks = [t]
            u = t
            gap = 0
            while u < gs.T and gap <= 2:
                u += 1
                ku = gs.k_act.get(u)
                if ku and ku["kind"] == "move" and tuple(ku["src"]) == cur:
                    cur = tuple(ku["dst"])
                    ticks.append(u)
                    used.add(u)
                    gap = 0
                else:
                    gap += 1
            d0, d1 = int(gs.dist_me[src0]), int(gs.dist_me[cur])
            if d0 - d1 >= CHAIN_MIN_DROP or d1 <= 1:
                f = start_t - 1
                chains.append({
                    "start": start_t, "end": ticks[-1], "len": len(ticks),
                    "d0": d0, "d1": d1,
                    "start_army": k["src_army"],
                    "reached_general": cur == gs.G_me,
                    "vis_dist": None if gs.min_vis_dist[f] >= INF else int(gs.min_vis_dist[f]),
                    "true_dist": None if gs.min_true_dist[f] >= INF else int(gs.min_true_dist[f]),
                    "threat_army": int(gs.threat_army[f]),
                    "gen_army": int(gs.gen_army[f]),
                })
            t = ticks[-1] + 1
        else:
            t += 1
    return chains


# ------------------------------------------------------------ tile captures

def capture_events(gs: GameState) -> list[dict]:
    """Enemy captures of Kubic-owned cells, with the contest scan."""
    events = []
    for t in range(1, gs.T + 1):
        lost = (gs.owners[t - 1] == gs.me) & (gs.owners[t] == gs.opp)
        if not lost.any():
            continue
        # skip the final ownership-transfer frame of a lost game
        if gs.game["outcome"] == "lose" and t == gs.T:
            continue
        for r, c in np.argwhere(lost):
            cell = (int(r), int(c))
            d = int(gs.dist_me[cell])
            contested = None
            retaken = False
            for u in range(t + 1, min(t + CONTEST_WINDOW, gs.T) + 1):
                k = gs.k_act.get(u)
                if k and k["kind"] == "move":
                    dst = tuple(k["dst"])
                    if dst == cell or (
                        k["target"] in ("enemy", "enemy_general")
                        and abs(dst[0] - cell[0]) + abs(dst[1] - cell[1]) <= 2
                    ):
                        contested = u - t
                        break
            for u in range(t + 1, min(t + CONTEST_WINDOW, gs.T) + 1):
                if gs.owners[u][cell] == gs.me:
                    retaken = True
                    break
            events.append({
                "t": t, "cell": cell, "dist": d,
                "enemy_army_after": int(gs.armies[t][cell]),
                "visible_before": bool(
                    dilate8(gs.owners[t - 1] == gs.me)[cell]
                ),
                "contest_latency": contested,
                "retaken": retaken,
            })
    return events


# ----------------------------------------------------------------- castles

def castle_report(gs: GameState) -> dict:
    """Kubic castle builds, garrison profile, threats and recaptures."""
    builds = []
    for t in range(1, gs.T + 1):
        k = gs.k_act.get(t)
        if k and k["kind"] == "build":
            builds.append({"t": t, "cell": tuple(k["cell"]), "cost": k["cost"],
                           "dist": int(gs.dist_me[tuple(k["cell"])])})
    garrison = []
    captures = []
    for b in builds:
        cell = b["cell"]
        for dt in (10, 25, 50):
            u = b["t"] + dt
            if u <= gs.T and gs.owners[u][cell] == gs.me:
                garrison.append({"dt": dt, "army": int(gs.armies[u][cell]), "dist": b["dist"]})
        for t in range(b["t"] + 1, gs.T + 1):
            if gs.owners[t][cell] == gs.opp and gs.owners[t - 1][cell] == gs.me:
                if gs.game["outcome"] == "lose" and t == gs.T:
                    continue
                attempt = None
                back = False
                for u in range(t + 1, min(t + RESP_WINDOW, gs.T) + 1):
                    k = gs.k_act.get(u)
                    if attempt is None and k and k["kind"] == "move" and tuple(k["dst"]) == cell:
                        attempt = u - t
                    if gs.owners[u][cell] == gs.me:
                        back = True
                        break
                captures.append({"t": t, "cell": cell, "dist": b["dist"],
                                 "attempt_latency": attempt, "recaptured": back})
                break
    return {"builds": builds, "garrison": garrison, "captures": captures}


# --------------------------------------------------------------- loss detail

def trace_killer(gs: GameState) -> dict | None:
    """Walk the winning stack backwards; find when Kubic could first see it."""
    if gs.game["outcome"] not in ("lose",):
        return None
    t_end = gs.T
    o = gs.o_act.get(t_end)
    if not (o and o["kind"] == "move" and o["target"] == "enemy_general"):
        # deathtouch or unresolved ending
        return {"end_kind": o["kind"] if o else "none", "trajectory": []}
    traj = [{"t": t_end, "cell": tuple(o["src"]), "army": o["src_army"]}]
    cur = tuple(o["src"])
    t = t_end
    misses = 0
    while t > 1 and misses <= 30 and len(traj) < 300:
        t -= 1
        ou = gs.o_act.get(t)
        if ou and ou["kind"] == "move" and tuple(ou["dst"]) == cur:
            cur = tuple(ou["src"])
            traj.append({"t": t, "cell": cur, "army": ou["src_army"]})
            misses = 0
        else:
            misses += 1
    traj.reverse()
    first_vis = None
    for p in traj:
        f = p["t"]  # frame after the stack arrived at p.cell is f
        if f <= gs.T and dilate8(gs.owners[f] == gs.me)[p["cell"]]:
            first_vis = {"t": f, "cell": p["cell"], "army": p["army"],
                         "dist": int(gs.dist_me[p["cell"]])}
            break
    return {
        "end_kind": "general_capture",
        "final_moved": o["moved"], "final_def_army": o["dst_army"],
        "traj_len": len(traj),
        "traj_start_dist": int(gs.dist_me[traj[0]["cell"]]) if traj else None,
        "first_visible": first_vis,
        "warning_ticks": (gs.T - first_vis["t"]) if first_vis else 0,
    }


def per_tick_log(gs: GameState, t_lo: int, t_hi: int) -> list[dict]:
    out = []
    for t in range(max(1, t_lo), min(t_hi, gs.T) + 1):
        f = t - 1
        k, o = gs.k_act.get(t, {"kind": "?"}), gs.o_act.get(t, {"kind": "?"})

        def brief(a, dist):
            if a["kind"] != "move":
                return a["kind"]
            s, d = tuple(a["src"]), tuple(a["dst"])
            return (f"{s}->{d} n={a['moved']} {a['target'][:4]}"
                    f" dG={int(dist[d]) if dist[d] < INF else '?'}"
                    + (" HOME" if dist[d] < dist[s] else ""))

        out.append({
            "t": t,
            "kubic": brief(k, gs.dist_me),
            "opp": brief(o, gs.dist_me),
            "vis_d": None if gs.min_vis_dist[f] >= INF else int(gs.min_vis_dist[f]),
            "true_d": None if gs.min_true_dist[f] >= INF else int(gs.min_true_dist[f]),
            "threat_army": int(gs.threat_army[f]),
            "gen_army": int(gs.gen_army[f]),
            "main_d": None if gs.main_dist[f] >= INF else int(gs.main_dist[f]),
            "main_army": int(gs.main_army[f]),
        })
    return out


# ------------------------------------------------- conditional policy per tick

VIS_BUCKETS = ((0, 2, "d0-2"), (3, 4, "d3-4"), (5, 6, "d5-6"), (7, 8, "d7-8"),
               (9, 12, "d9-12"), (13, INF, "d13+/none"))


def vis_bucket(v: int) -> str:
    for lo, hi, name in VIS_BUCKETS:
        if lo <= v <= hi:
            return name
    return "d13+/none"


def classify_tick(gs: GameState, t: int) -> str:
    """Coarse action class for the conditional-policy table."""
    k = gs.k_act.get(t)
    if k is None:
        return "missing"
    if k["kind"] != "move":
        return k["kind"]
    f = t - 1
    dst = tuple(k["dst"])
    parts = []
    if k["target"] in ("enemy", "enemy_general"):
        tc = gs.threat_cell[f]
        if tc is not None and dst == tc:
            return "attack_threat_cell"
        if gs.dist_me[dst] <= D_REF + 2:
            return "attack_near_home"
        return "attack_far"
    if k["target"] == "neutral":
        return "expand_neutral"
    # own-cell landing
    if dst == gs.G_me:
        return "onto_general"
    if gs.homeward(k) and k["moved"] >= BIG_MOVE:
        return "own_homeward_big"
    if gs.homeward(k):
        return "own_homeward_small"
    return "own_outward"


# --------------------------------------------------------------- aggregation

def main() -> None:
    rows = split_rows("fit")
    rows.sort(key=lambda r: r["match_id"])

    all_eps = []
    all_chains = []
    all_caps = []
    castle_builds = []
    castle_garrison = []
    castle_caps = []
    passes_pressure = []
    pass_total = 0
    act_total = 0
    chase_moves = 0
    enemy_target_moves = 0
    chase_defensive = 0
    enemy_target_defensive = 0
    win_near_death = []
    loss_blocks = []
    unresolved_total = 0
    tick_total = 0

    # per-tick recall-trigger dataset: buckets over min_vis_dist
    eligible = Counter()          # vis_dist bucket -> ticks eligible
    recall_started = Counter()    # vis_dist bucket -> chain starts
    eligible_ratio = Counter()    # (vis bucket, army>=gen) -> ticks
    recall_ratio = Counter()

    # predicate grid results: predicate onset -> chain start within 8 ticks
    pred_grid = {}
    grid_D = [1, 2, 3, 4, 5, 6, 8]
    grid_K = [0.0, 0.5, 1.0]

    prox_games_vis = Counter()
    prox_games_true = Counter()

    policy = {}                     # vis bucket -> Counter of action classes
    engage = Counter()              # (ratio bucket, dist bucket) -> ticks
    engage_hit = Counter()          # -> attacked threat cell within 5 ticks
    drain_elig = Counter()
    drain_did = Counter()
    pass_runs = {}
    stall_by_game = []
    osc_events = []
    chase_opp = 0
    chase_taken = 0
    chase_examples = []
    gen_src_moves = 0
    gen_src_under_threat = 0
    gen_src_army_left = []
    move_total = 0
    pass_tick_frac = []
    homeward_big_in_ep = [0, 0]     # ticks with such a move, ticks total (in-episode)
    homeward_big_out_ep = [0, 0]
    class_latencies = {}            # response class -> list of latencies (episodes)

    for i, row in enumerate(rows):
        gs = GameState(row)
        unresolved_total += len(gs.game["unresolved"])
        tick_total += len(gs.game["ticks"])
        is_win = gs.game["outcome"] == "win"
        # cap frames for min-dist stats: exclude the final transfer frame
        f_hi = gs.T - 1 if gs.game["outcome"] != "draw" else gs.T

        for D in grid_D:
            if any(gs.min_vis_dist[t] <= D for t in range(0, f_hi + 1)):
                prox_games_vis[D] += 1
            if any(gs.min_true_dist[t] <= D for t in range(0, f_hi + 1)):
                prox_games_true[D] += 1

        eps = [classify_response(gs, e) for e in find_episodes(gs)]
        for e in eps:
            e["match_id"] = row["match_id"]
            e["outcome_game"] = gs.game["outcome"]
            for cls, lat in e["classes_hit"].items():
                class_latencies.setdefault(cls, []).append(lat)
        all_eps.extend(eps)
        in_ep = np.zeros(gs.T + 1, dtype=bool)
        for e in eps:
            in_ep[e["t0"]: min(e["t1"], gs.T) + 1] = True

        # conditional policy table + homeward-big baseline
        for t in range(1, gs.T + 1):
            f = t - 1
            b = vis_bucket(int(gs.min_vis_dist[f]))
            cls = classify_tick(gs, t)
            policy.setdefault(b, Counter())[cls] += 1
            k = gs.k_act.get(t)
            hb = bool(k and k["kind"] == "move" and gs.homeward(k) and k["moved"] >= BIG_MOVE)
            tgt = homeward_big_in_ep if in_ep[t] else homeward_big_out_ep
            tgt[0] += int(hb)
            tgt[1] += 1
            if k and k["kind"] == "move":
                move_total += 1
                if tuple(k["src"]) == gs.G_me:
                    gen_src_moves += 1
                    if gs.min_vis_dist[f] <= D_REF:
                        gen_src_under_threat += 1
                    gen_src_army_left.append(k["src_army"] - k["moved"])

        # pass-run lengths (consecutive passes), split by pressure at run start
        t = 1
        long_runs = 0
        while t <= gs.T:
            k = gs.k_act.get(t)
            if k and k["kind"] == "pass":
                run = 1
                while t + run <= gs.T and (ku := gs.k_act.get(t + run)) and ku["kind"] == "pass":
                    run += 1
                key = "pressure" if gs.min_vis_dist[t - 1] <= 5 else "clear"
                pass_runs.setdefault(key, Counter())[min(run, 10)] += 1
                if run >= 3 and t > 30:  # ignore the opening wait
                    long_runs += 1
                t += run
            else:
                t += 1
        stall_by_game.append({"match_id": row["match_id"], "outcome": gs.game["outcome"],
                              "opponent": row["opponent"], "long_pass_runs": long_runs})

        # general-feed oscillation: repeated tiny moves off the general
        t = 1
        while t <= gs.T:
            k = gs.k_act.get(t)
            if (k and k["kind"] == "move" and tuple(k["src"]) == gs.G_me
                    and k["moved"] <= 2):
                dst = tuple(k["dst"])
                reps = 1
                u = t
                last = t
                while u < min(t + 24, gs.T):
                    u += 1
                    ku = gs.k_act.get(u)
                    if (ku and ku["kind"] == "move" and tuple(ku["src"]) == gs.G_me
                            and tuple(ku["dst"]) == dst and ku["moved"] <= 2):
                        reps += 1
                        last = u
                if reps >= 3:
                    osc_events.append({
                        "match_id": row["match_id"], "t0": t, "t1": last, "reps": reps,
                        "vis_d": None if gs.min_vis_dist[t - 1] >= INF else int(gs.min_vis_dist[t - 1]),
                    })
                    t = last + 1
                else:
                    t += 1
            else:
                t += 1

        # drain gating: does a visible threat suppress taking the general's army?
        for t in range(1, gs.T + 1):
            f = t - 1
            g = int(gs.gen_army[f])
            if g < 8:
                continue
            threat = gs.min_vis_dist[f] <= D_REF
            k = gs.k_act.get(t)
            drained = bool(k and k["kind"] == "move" and tuple(k["src"]) == gs.G_me
                           and k["moved"] >= g - 1)
            key = ("threat" if threat else "clear",
                   "g8-19" if g < 20 else ("g20-39" if g < 40 else "g40+"))
            drain_elig[key] += 1
            drain_did[key] += int(drained)

        # engagement: does Kubic attack the nearest visible threat cell?
        for f in range(0, gs.T):
            if gs.min_vis_dist[f] > D_REF:
                continue
            tc = gs.threat_cell[f]
            ta = int(gs.threat_army[f])
            # biggest own stack within manhattan 3 of the threat cell
            r0, c0 = tc
            best = 0
            for r in range(max(0, r0 - 3), min(gs.rows, r0 + 4)):
                for c in range(max(0, c0 - 3), min(gs.cols, c0 + 4)):
                    if abs(r - r0) + abs(c - c0) <= 3 and gs.owners[f][r, c] == gs.me:
                        best = max(best, int(gs.armies[f][r, c]))
            ratio = "no_stack" if best <= 1 else (
                "ge_2x" if best >= 2 * ta + 1 else
                "gt_1x" if best > ta + 1 else "le_1x"
            )
            db = "d0-2" if gs.dist_me[tc] <= 2 else ("d3-4" if gs.dist_me[tc] <= 4 else "d5-6")
            engage[(ratio, db)] += 1
            hit = False
            for u in range(f + 1, min(f + 5, gs.T) + 1):
                k = gs.k_act.get(u)
                if k and k["kind"] == "move" and tuple(k["dst"]) == tc:
                    hit = True
                    break
            engage_hit[(ratio, db)] += int(hit)

        # chase opportunities: enemy move from src we could capture first
        for t in range(1, gs.T + 1):
            o = gs.o_act.get(t)
            if not (o and o["kind"] == "move" and o["moved"] > 0):
                continue
            f = t - 1
            s = tuple(o["src"])
            if gs.dist_me[s] > 8 or gs.dist_me[tuple(o["dst"])] >= gs.dist_me[s]:
                continue  # only enemy moves advancing on our general from <=8
            if not dilate8(gs.owners[f] == gs.me)[s]:
                continue  # fog: must be visible
            src_army = int(gs.armies[f][s])
            can = any(
                0 <= s[0] + dr < gs.rows and 0 <= s[1] + dc < gs.cols
                and gs.owners[f][s[0] + dr, s[1] + dc] == gs.me
                and int(gs.armies[f][s[0] + dr, s[1] + dc]) - 1 > src_army
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            )
            if not can:
                continue
            chase_opp += 1
            k = gs.k_act.get(t)
            took = bool(k and k["kind"] == "move" and tuple(k["dst"]) == s)
            chase_taken += int(took)
            if len(chase_examples) < 20 and not took:
                chase_examples.append({"match_id": row["match_id"], "t": t,
                                       "src": s, "src_army": src_army})

        chains = find_homeward_chains(gs)
        for c in chains:
            c["match_id"] = row["match_id"]
        all_chains.extend(chains)
        chain_start_ticks = {c["start"] for c in chains}

        caps = capture_events(gs)
        for c in caps:
            c["match_id"] = row["match_id"]
        all_caps.extend(caps)

        cr = castle_report(gs)
        castle_builds.extend(cr["builds"])
        castle_garrison.extend(cr["garrison"])
        for c in cr["captures"]:
            c["match_id"] = row["match_id"]
        castle_caps.extend(cr["captures"])

        # chase and pass accounting
        for t in range(1, gs.T + 1):
            k = gs.k_act.get(t)
            if k is None:
                continue
            act_total += 1
            f = t - 1
            threatened = gs.min_vis_dist[f] <= 5
            if k["kind"] == "pass":
                pass_total += 1
                pass_tick_frac.append(t / max(1, gs.T))
                if threatened:
                    passes_pressure.append({
                        "match_id": row["match_id"], "t": t, "T": gs.T,
                        "vis_d": int(gs.min_vis_dist[f]),
                        "threat_army": int(gs.threat_army[f]),
                        "gen_army": int(gs.gen_army[f]),
                    })
            if k["kind"] == "move" and k["target"] in ("enemy", "enemy_general"):
                enemy_target_moves += 1
                ch = gs.is_chase(t)
                if ch:
                    chase_moves += 1
                if threatened:
                    enemy_target_defensive += 1
                    if ch:
                        chase_defensive += 1

        # recall trigger dataset
        for t in range(1, gs.T + 1):
            f = t - 1
            if gs.main_dist[f] < 4 or gs.main_dist[f] >= INF or gs.main_army[f] < 10:
                continue
            v = gs.min_vis_dist[f]
            b = str(int(v)) if v <= 10 else "none"
            eligible[b] += 1
            big = gs.threat_army[f] >= max(1, gs.gen_army[f])
            eligible_ratio[(b, big)] += 1
            if t in chain_start_ticks:
                recall_started[b] += 1
                recall_ratio[(b, big)] += 1

        # predicate grid, event level
        for D in grid_D:
            for K in grid_K:
                key = f"D{D}_K{K}"
                st = pred_grid.setdefault(key, {"onsets": 0, "onset_hits": 0,
                                                "chains": 0, "chain_covered": 0})
                fire = [
                    bool(gs.min_vis_dist[f] <= D
                         and gs.threat_army[f] >= K * max(1, gs.gen_army[f]))
                    for f in range(gs.T + 1)
                ]
                for f in range(1, gs.T + 1):
                    if fire[f] and not fire[f - 1]:
                        st["onsets"] += 1
                        if any(u in chain_start_ticks for u in range(f + 1, min(f + 9, gs.T) + 1)):
                            st["onset_hits"] += 1
                for c in chains:
                    st["chains"] += 1
                    lo = max(0, c["start"] - 9)
                    if any(fire[u] for u in range(lo, c["start"])):
                        st["chain_covered"] += 1

        # wins: near-death check
        if is_win:
            md = int(min(gs.min_vis_dist[: f_hi + 1]))
            mt = int(min(gs.min_true_dist[: f_hi + 1]))
            if md <= 2:
                t_close = int(np.argmin(gs.min_vis_dist[: f_hi + 1]))
                saved_by_chain = any(
                    abs(c["start"] - t_close) <= 15 or (c["start"] < t_close < c["end"] + 15)
                    for c in chains
                )
                win_near_death.append({
                    "match_id": row["match_id"], "min_vis_dist": md,
                    "min_true_dist": mt, "t_closest": t_close,
                    "gen_army_at_closest": int(gs.gen_army[t_close]),
                    "threat_army_at_closest": int(gs.threat_army[t_close]),
                    "recall_chain_near": saved_by_chain,
                })

        # losses and the draw: full detail
        if not is_win:
            killer = trace_killer(gs)
            first_vis5 = next(
                (t for t in range(gs.T + 1) if gs.min_vis_dist[t] <= 5), None
            )
            block = {
                "match_id": row["match_id"],
                "outcome": gs.game["outcome"],
                "opponent": row["opponent"],
                "total_ticks": gs.T,
                "killer": killer,
                "first_vis_dist5_frame": first_vis5,
                "episodes": [e for e in eps],
                "chains": chains,
                "final_log": per_tick_log(gs, gs.T - 29, gs.T),
                "gen_army_last30": [int(x) for x in gs.gen_army[max(0, gs.T - 30):]],
                "my_total_last30": [int(x) for x in gs.my_total[max(0, gs.T - 30):]],
                "opp_total_last30": [int(x) for x in gs.opp_total[max(0, gs.T - 30):]],
            }
            loss_blocks.append(block)

        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(rows)} games", flush=True)

    # ---------------- aggregate episodes
    def ep_agg(eps):
        n = len(eps)
        responded = [e for e in eps if e["first_def"] is not None]
        lat = [e["first_def"]["latency"] for e in responded]
        cls = Counter(e["first_def"]["class"] for e in responded)
        return {
            "n": n,
            "responded": rate(len(responded), n),
            "latency": quantiles(lat),
            "latency_hist": dict(sorted(Counter(lat).items())),
            "first_class": dict(cls.most_common()),
            "pre_expand": quantiles([e["pre_expand"] for e in eps]),
            "pre_expand_hist": dict(sorted(Counter(e["pre_expand"] for e in eps).items())),
            "tiles_lost": quantiles([e["tiles_lost"] for e in eps]),
            "outcomes": dict(Counter(e["outcome"] for e in eps).most_common()),
            "src_general_moves": rate(
                sum(1 for e in eps if e["src_general_moves"] > 0), n
            ),
        }

    by_onset = {}
    for lo, hi, name in [(0, 2, "d0-2"), (3, 4, "d3-4"), (5, 6, "d5-6")]:
        sub = [e for e in all_eps if lo <= e["onset_dist"] <= hi]
        by_onset[name] = ep_agg(sub)
    by_army = {}
    for name, pred in [
        ("threat_lt_gen", lambda e: e["onset_threat_army"] < max(1, e["onset_gen_army"])),
        ("threat_ge_gen", lambda e: e["onset_threat_army"] >= max(1, e["onset_gen_army"])),
        ("threat_ge_8", lambda e: e["onset_threat_army"] >= 8),
        ("threat_lt_8", lambda e: e["onset_threat_army"] < 8),
    ]:
        by_army[name] = ep_agg([e for e in all_eps if pred(e)])

    # recall trigger curve
    curve = {}
    for b in sorted(eligible, key=lambda x: (x == "none", x)):
        curve[b] = rate(recall_started[b], eligible[b])
    curve_ratio = {}
    for (b, big), den in sorted(eligible_ratio.items(), key=lambda kv: (kv[0][0] == "none", kv[0])):
        curve_ratio[f"{b}_{'big' if big else 'small'}"] = rate(recall_ratio[(b, big)], den)

    for key, st in pred_grid.items():
        st["onset_precision"] = st["onset_hits"] / st["onsets"] if st["onsets"] else None
        st["chain_coverage"] = st["chain_covered"] / st["chains"] if st["chains"] else None

    chains_with_no_threat = [
        c for c in all_chains if c["vis_dist"] is None or c["vis_dist"] > 8
    ]
    chains_reaching_general = [c for c in all_chains if c["reached_general"]]

    out = {
        "meta": {
            "games": len(rows),
            "outcomes": dict(Counter(r["outcome"] for r in rows)),
            "unresolved_ticks": unresolved_total,
            "total_ticks": tick_total,
            "D_REF": D_REF, "EP_GAP": EP_GAP, "RESP_WINDOW": RESP_WINDOW,
            "BIG_MOVE": BIG_MOVE, "CHAIN_MIN_DROP": CHAIN_MIN_DROP,
        },
        "proximity_games": {
            "visible": {str(d): prox_games_vis[d] for d in grid_D},
            "true": {str(d): prox_games_true[d] for d in grid_D},
        },
        "episodes": {
            "total": ep_agg(all_eps),
            "wins_only": ep_agg([e for e in all_eps if e["outcome_game"] == "win"]),
            "losses_only": ep_agg([e for e in all_eps if e["outcome_game"] == "lose"]),
            "by_onset_dist": by_onset,
            "by_onset_army": by_army,
            "per_game_count": quantiles(
                list(Counter(e["match_id"] for e in all_eps).values())
            ),
            "sample_no_response": [
                {k: e[k] for k in ("match_id", "t0", "onset_dist", "onset_threat_army",
                                   "onset_gen_army", "outcome")}
                for e in all_eps if e["first_def"] is None
            ][:40],
        },
        "response_class_latency": {
            cls: quantiles(lats) for cls, lats in sorted(class_latencies.items())
        },
        "conditional_policy": {
            b: {
                "n": sum(cnt.values()),
                "dist": {cls: rate(v, sum(cnt.values()))["rate"]
                         for cls, v in cnt.most_common()},
            }
            for b, cnt in sorted(policy.items())
        },
        "engagement": {
            f"{ratio}_{db}": rate(engage_hit[(ratio, db)], n)
            for (ratio, db), n in sorted(engage.items())
        },
        "chase": {
            "enemy_target_moves": enemy_target_moves,
            "chase_moves": rate(chase_moves, enemy_target_moves),
            "chase_under_threat": rate(chase_defensive, enemy_target_defensive),
            "opportunities": rate(chase_taken, chase_opp),
            "missed_examples": chase_examples,
        },
        "general_as_source": {
            "moves_from_general": rate(gen_src_moves, move_total),
            "under_visible_threat_le6": gen_src_under_threat,
            "army_left_behind": quantiles(gen_src_army_left),
            "drain_rate_by_threat": {
                f"{a}_{b}": rate(drain_did[(a, b)], n)
                for (a, b), n in sorted(drain_elig.items())
            },
        },
        "homeward_big_rate": {
            "in_episode": rate(homeward_big_in_ep[0], homeward_big_in_ep[1]),
            "out_episode": rate(homeward_big_out_ep[0], homeward_big_out_ep[1]),
        },
        "recall": {
            "chains_total": len(all_chains),
            "chain_len": quantiles([c["len"] for c in all_chains]),
            "chain_start_army": quantiles([c["start_army"] for c in all_chains]),
            "chain_d0": quantiles([c["d0"] for c in all_chains]),
            "reached_general": rate(len(chains_reaching_general), len(all_chains)),
            "no_visible_threat_at_start": rate(len(chains_with_no_threat), len(all_chains)),
            "start_rate_by_vis_dist": curve,
            "start_rate_by_vis_dist_and_army": curve_ratio,
            "predicate_grid": pred_grid,
        },
        "territory_captures": {
            "n": len(all_caps),
            "by_dist": {
                name: {
                    "n": len(sub),
                    "contested": rate(sum(1 for c in sub if c["contest_latency"] is not None), len(sub)),
                    "contest_latency": quantiles(
                        [c["contest_latency"] for c in sub if c["contest_latency"] is not None]
                    ),
                    "retaken": rate(sum(1 for c in sub if c["retaken"]), len(sub)),
                    "visible_before": rate(sum(1 for c in sub if c["visible_before"]), len(sub)),
                }
                for name, sub in [
                    (f"d{lo}-{hi}", [c for c in all_caps if lo <= c["dist"] <= hi])
                    for lo, hi in [(0, 2), (3, 5), (6, 9), (10, 99)]
                ]
            },
        },
        "castles": {
            "builds": len(castle_builds),
            "build_dist": quantiles([b["dist"] for b in castle_builds]),
            "garrison_by_dt": {
                str(dt): quantiles([g["army"] for g in castle_garrison if g["dt"] == dt])
                for dt in (10, 25, 50)
            },
            "captures": len(castle_caps),
            "recapture_attempted": rate(
                sum(1 for c in castle_caps if c["attempt_latency"] is not None), len(castle_caps)
            ),
            "recaptured": rate(sum(1 for c in castle_caps if c["recaptured"]), len(castle_caps)),
            "capture_details": castle_caps[:40],
        },
        "passes": {
            "total_actions": act_total,
            "total_passes": pass_total,
            "passes_under_pressure": len(passes_pressure),
            "rate_under_pressure": rate(len(passes_pressure), pass_total),
            "pass_tick_frac": quantiles(pass_tick_frac),
            "pressure_pass_tick_frac": quantiles(
                [p["t"] / max(1, p["T"]) for p in passes_pressure]
            ),
            "run_lengths": {k: dict(sorted(v.items())) for k, v in sorted(pass_runs.items())},
            "stall_games": sorted(
                [g for g in stall_by_game if g["long_pass_runs"] >= 3],
                key=lambda g: -g["long_pass_runs"],
            ),
            "stall_rate_by_outcome": {
                oc: rate(
                    sum(1 for g in stall_by_game if g["outcome"] == oc and g["long_pass_runs"] >= 3),
                    sum(1 for g in stall_by_game if g["outcome"] == oc),
                )
                for oc in ("win", "lose", "draw")
            },
            "examples": passes_pressure[:30],
        },
        "general_feed_oscillation": {
            "n": len(osc_events),
            "games": len({e["match_id"] for e in osc_events}),
            "under_visible_threat_le3": sum(
                1 for e in osc_events if e["vis_d"] is not None and e["vis_d"] <= 3
            ),
            "reps": quantiles([e["reps"] for e in osc_events]),
            "examples": osc_events[:30],
        },
        "wins_near_death": {
            "n": len(win_near_death),
            "recall_chain_near": rate(
                sum(1 for w in win_near_death if w["recall_chain_near"]), len(win_near_death)
            ),
            "details": win_near_death,
        },
        "losses": loss_blocks,
    }

    OUT_PATH.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
