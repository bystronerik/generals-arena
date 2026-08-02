#!/usr/bin/env python3
"""
Kubic ATTACK-dimension analysis over the fable fit set (351 games).

Measures, from exact reconstructed per-tick actions (fable_kubic_common) plus
replay frames and fog masks:
  A. first contact -> first attack latency and balances
  B. engagement predicate: adjacency situations, margin curve, attack margins
  C. target selection among candidate attack pairs
  D. raid chains / big-commit context and post-commit path
  E. enemy castle visibility -> capture latency, castle-vs-plain priority
  F. general strike: sight -> kill, path shape, fog memory, wave count
  G. pre-sight hunt: do enemy-territory captures march toward the true general
  H. deathtouch window (t >= 800)
  I. per-game diagnostics for the 10 fit losses and 1 draw

Deterministic. Writes docs/research/measurements/fable-kubic-attack.json.

Fog-legality notes are attached to each block in the JSON: anything using the
true enemy-general position before first sight, or true mountains outside
vision, is labelled belief-level (analysis-only), not a fog-legal input.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, deque
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT))

from fable_kubic_common import split_rows, load_actions, kubic_actions  # noqa: E402
from arena.instrument.replay.loader import open_replay  # noqa: E402

OUT_JSON = REPO_ROOT / "docs/research/measurements/fable-kubic-attack.json"

ORTH = ((-1, 0), (1, 0), (0, -1), (0, 1))
CHAIN_GAP = 30          # max ticks a stack chain may sit idle before we cut it
BIG_COMMIT = 20         # "big stack" entry threshold examined (natural break checked)
POST_COMMIT_MOVES = 25  # chain moves classified after a commit


# ----------------------------------------------------------------- utilities

def dist_summary(values, name):
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return {"name": name, "n": 0}
    arr = np.array(vals)
    return {
        "name": name,
        "n": len(vals),
        "mean": round(float(arr.mean()), 3),
        "min": float(arr[0]),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr[-1]),
    }


def dilate8(mask):
    """8-neighbourhood dilation of a (T,R,C) or (R,C) boolean mask."""
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            shifted = np.zeros_like(mask)
            rs = slice(max(dr, 0), mask.shape[-2] + min(dr, 0))
            rd = slice(max(-dr, 0), mask.shape[-2] + min(-dr, 0))
            cs = slice(max(dc, 0), mask.shape[-1] + min(dc, 0))
            cd = slice(max(-dc, 0), mask.shape[-1] + min(-dc, 0))
            shifted[..., rd, cd] = mask[..., rs, cs]
            out |= shifted
    return out


def bfs_dist(passable, start):
    """BFS step counts from `start` over passable cells; unreachable = big."""
    R, C = passable.shape
    dist = np.full((R, C), 10**6, dtype=np.int64)
    if not passable[start]:
        # a general cell is always passable in practice; guard anyway
        return dist
    dist[start] = 0
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in ORTH:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and passable[nr, nc] and dist[nr, nc] > dist[r, c] + 1:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


class Game:
    """Everything one game contributes, precomputed once."""

    def __init__(self, row):
        self.row = row
        self.mid = row["match_id"]
        self.g = load_actions(self.mid)
        self.rep = open_replay("Kubic", self.mid)
        self.s = self.g["kubic_seat"]
        self.e = 1 - self.s
        self.R, self.C = self.rep.rows, self.rep.cols
        self.T = len(self.rep.ticks) - 1  # states 0..T, actions 1..T
        self.owners = np.array([f.owners for f in self.rep.ticks], dtype=np.int16)
        self.armies = np.array([f.armies for f in self.rep.ticks], dtype=np.int64)
        self.passable = np.ones((self.R, self.C), dtype=bool)
        for r, c in self.rep.mountains:
            self.passable[r, c] = False
        self.gen_k = tuple(self.g["generals"][self.s])
        self.gen_e = tuple(self.g["generals"][self.e])
        self.vis = dilate8(self.owners == self.s) | (self.owners == self.s)
        self.enemy_vis = self.vis & (self.owners == self.e)
        self.dist_gen = bfs_dist(self.passable, self.gen_e)
        self.acts = kubic_actions(self.g)          # index i -> action at tick i+1
        self.eacts = [dict(t["p%d" % self.e], t=t["t"]) for t in self.g["ticks"]]
        # first contact: first STATE tick with a fog-visible enemy cell
        cv = self.enemy_vis.reshape(self.T + 1, -1).any(axis=1)
        self.contact = int(np.argmax(cv)) if cv.any() else None
        # first sight of enemy general
        gv = self.vis[:, self.gen_e[0], self.gen_e[1]]
        self.sight = int(np.argmax(gv)) if gv.any() else None
        self.won = self.g["winner"] == self.s
        # kill = Kubic action that captured the enemy general
        self.kill_t = None
        for a in self.acts:
            if a["kind"] == "move" and a.get("target") == "enemy_general" and a.get("captured"):
                self.kill_t = a["t"]
                break

    def act_at(self, t):
        return self.acts[t - 1] if 1 <= t <= len(self.acts) else {"kind": "pass"}


# ------------------------------------------------- B. engagement situations

def adjacency_pairs(game, st):
    """(src, dst, stack, enemy_army) for every Kubic stack>=2 orthogonally
    adjacent to a fog-visible enemy cell, in state `st`."""
    own = (game.owners[st] == game.s) & (game.armies[st] >= 2)
    ev = game.enemy_vis[st]
    pairs = []
    for r, c in zip(*np.nonzero(own)):
        for dr, dc in ORTH:
            nr, nc = r + dr, c + dc
            if 0 <= nr < game.R and 0 <= nc < game.C and ev[nr, nc]:
                pairs.append(((int(r), int(c)), (int(nr), int(nc)),
                              int(game.armies[st, r, c]), int(game.armies[st, nr, nc])))
    return pairs


def margin_bucket(m):
    if m <= -10:
        return "<=-10"
    if m < 0:
        return "-9..-1"
    if m == 0:
        return "0"
    if m <= 2:
        return "1..2"
    if m <= 5:
        return "3..5"
    if m <= 10:
        return "6..10"
    if m <= 20:
        return "11..20"
    return ">20"


BUCKETS = ["<=-10", "-9..-1", "0", "1..2", "3..5", "6..10", "11..20", ">20"]


# ------------------------------------------------------------- chain builder

def build_chains(game):
    """Link Kubic moves into stack chains: move at t joins the chain whose head
    cell equals its src, if that head was produced within CHAIN_GAP ticks.
    Returns a list of chains: each a list of action dicts (with t)."""
    chains = []
    heads = {}  # cell -> (chain_index, tick_of_head)
    for a in game.acts:
        if a["kind"] != "move":
            continue
        src, dst, t = tuple(a["src"]), tuple(a["dst"]), a["t"]
        entry = heads.get(src)
        if entry is not None and t - entry[1] <= CHAIN_GAP and a["split"] == 0:
            ci = entry[0]
            chains[ci].append(a)
            del heads[src]
        else:
            chains.append([a])
            ci = len(chains) - 1
        # a split leaves the source alive; the moving half is the new head
        heads[dst] = (ci, t)
    return chains


# ------------------------------------------------------------------ per game

def analyze_game(game, agg):
    mid = game.mid
    s, e = game.s, game.e

    for a in game.acts:
        agg["mix"][a["kind"] if a["kind"] != "move" else "move_" + a["target"]] += 1

    # ---- A. contact -> first attack -------------------------------------
    first_attack = None
    for a in game.acts:
        if a["kind"] == "move" and a.get("target") in ("enemy", "enemy_general"):
            first_attack = a
            break
    rec = {"match_id": mid, "contact": game.contact, "sight": game.sight,
           "won": game.won, "kill_t": game.kill_t, "T": game.T,
           "outcome": game.g["outcome"]}
    if game.contact is not None:
        st = game.contact
        my_army = int(game.armies[st][game.owners[st] == s].sum())
        en_army = int(game.armies[st][game.owners[st] == e].sum())
        vis_en_army = int(game.armies[st][game.enemy_vis[st]].sum())
        own_mask = game.owners[st] == s
        max_stack = int(game.armies[st][own_mask].max()) if own_mask.any() else 0
        rec.update(army_ratio_contact=round(my_army / max(1, en_army), 4),
                   vis_enemy_army_contact=vis_en_army, max_stack_contact=max_stack)
        agg["contact"]["army_ratio"].append(my_army / max(1, en_army))
        agg["contact"]["max_stack"].append(max_stack)
        agg["contact"]["vis_enemy_army"].append(vis_en_army)
        if first_attack is not None:
            lat = first_attack["t"] - game.contact
            rec["contact_to_first_attack"] = lat
            agg["contact"]["latency"].append(lat)
            agg["contact"]["first_attack_moved"].append(first_attack["moved"])
            agg["contact"]["first_attack_margin"].append(
                first_attack["moved"] - first_attack["dst_army"])
        else:
            agg["contact"]["no_attack_games"].append(mid)

    # ---- B. engagement situations ---------------------------------------
    onset = None  # first tick a strictly positive full-send margin exists
    for t in range(1, game.T + 1):
        st = t - 1
        if not game.enemy_vis[st].any():
            onset = None if onset is None else onset
        pairs = adjacency_pairs(game, st)
        if not pairs:
            continue
        best = max(p[2] - 1 - p[3] for p in pairs)
        a = game.act_at(t)
        attacked = a["kind"] == "move" and a.get("target") in ("enemy", "enemy_general")
        b = margin_bucket(best)
        agg["engage"]["situations"][b] += 1
        # refined predicate: the board-max stack itself sits at the front with
        # a strictly positive full-send margin
        own_mask = game.owners[st] == game.s
        board_max = int(game.armies[st][own_mask].max()) if own_mask.any() else 0
        max_pairs = [p for p in pairs if p[2] == board_max and p[2] - 1 - p[3] > 0]
        if max_pairs:
            agg["engage"]["maxstack_front_n"] += 1
            if attacked and any(tuple(a["src"]) == p[0] for p in max_pairs):
                agg["engage"]["maxstack_front_attacked"] += 1
            elif attacked:
                agg["engage"]["maxstack_front_attacked_other"] += 1
            else:
                k = a["kind"] if a["kind"] != "move" else "move_" + a["target"]
                agg["engage"]["maxstack_front_instead"][k] += 1
                # when the fronted max stack moves onto an own/neutral cell
                # instead: is that move itself from the max stack, and does it
                # close BFS distance to the enemy general (path-following)?
                if a["kind"] == "move" and any(tuple(a["src"]) == p[0] for p in max_pairs):
                    agg["engage"]["maxstack_decline_moves"] += 1
                    if game.dist_gen[tuple(a["dst"])] < game.dist_gen[tuple(a["src"])]:
                        agg["engage"]["maxstack_decline_toward_gen"] += 1
        if attacked:
            agg["engage"]["attacked"][b] += 1
            m = a["moved"] - a["dst_army"]
            agg["engage"]["attack_margin"].append(m)
            agg["engage"]["attack_split"][a["split"]] += 1
            if m <= 0:
                # is this a "shave" — the cell is captured within 3 ticks after?
                shaved = any(
                    aa.get("captured") and tuple(aa["dst"]) == tuple(a["dst"])
                    for aa in (game.act_at(tt) for tt in range(t + 1, min(t + 4, game.T) + 1))
                    if aa["kind"] == "move")
                agg["engage"]["nonwinning_attacks"].append(
                    {"match_id": mid, "t": t, "moved": a["moved"],
                     "dst_army": a["dst_army"], "target": a["target"],
                     "tie": a["moved"] == a["dst_army"],
                     "captured_within_3": bool(shaved),
                     "deathtouch": t - 1 >= 800 and a["target"] == "enemy_general"})
        else:
            kind = a["kind"]
            if kind == "move":
                kind = "move_" + a["target"]
            agg["engage"]["instead"][b][kind] += 1
        if best > 0 and onset is None:
            onset = t
        if attacked and onset is not None:
            agg["engage"]["onset_to_attack"].append(t - onset)
            onset = None
        if best <= 0:
            onset = None

    # ---- C. target selection --------------------------------------------
    for t in range(1, game.T + 1):
        a = game.act_at(t)
        if not (a["kind"] == "move" and a.get("target") in ("enemy", "enemy_general")):
            continue
        st = t - 1
        pairs = adjacency_pairs(game, st)
        if len({p[1] for p in pairs}) < 2:
            continue  # no real choice
        chosen = (tuple(a["src"]), tuple(a["dst"]))
        pre_sight = game.sight is None or st < game.sight
        sel = agg["target_sel"]["pre" if pre_sight else "post"]
        sel["n"] += 1
        # selector: destination minimizing BFS distance to enemy general
        dmin = min(game.dist_gen[p[1]] for p in pairs)
        if game.dist_gen[chosen[1]] == dmin:
            sel["min_dist_gen"] += 1
        # selector: destination with minimum defending army
        amin = min(p[3] for p in pairs)
        if int(game.armies[st][chosen[1]]) == amin:
            sel["min_dst_army"] += 1
        # selector: pair with maximum full-send margin
        mmax = max(p[2] - 1 - p[3] for p in pairs)
        ch_pair = next((p for p in pairs if (p[0], p[1]) == chosen), None)
        if ch_pair is not None and ch_pair[2] - 1 - ch_pair[3] == mmax:
            sel["max_margin"] += 1
        # selector: source with largest stack
        smax = max(p[2] for p in pairs)
        if ch_pair is not None and ch_pair[2] == smax:
            sel["max_src_stack"] += 1
        # within-src destination tie-break: given the chosen src, which enemy
        # neighbour did it pick?
        same_src = [p for p in pairs if p[0] == chosen[0]]
        if len({p[1] for p in same_src}) >= 2:
            sel["dst_choice_n"] += 1
            if game.dist_gen[chosen[1]] == min(game.dist_gen[p[1]] for p in same_src):
                sel["dst_min_dist_gen"] += 1
            if int(game.armies[st][chosen[1]]) == min(p[3] for p in same_src):
                sel["dst_min_army"] += 1
        # chain continuation: previous non-pass action fed this src
        prev = next((game.act_at(tt) for tt in range(t - 1, max(0, t - 4), -1)
                     if game.act_at(tt)["kind"] == "move"), None)
        sel["chain_n"] += 1
        if prev is not None and tuple(prev["dst"]) == chosen[0]:
            sel["chain_continuation"] += 1

    # ---- D. raids / commits ---------------------------------------------
    chains = build_chains(game)
    for ch in chains:
        first_enemy = next((i for i, a in enumerate(ch)
                            if a.get("target") in ("enemy", "enemy_general")), None)
        if first_enemy is None:
            continue
        a0 = ch[first_enemy]
        entry = a0["moved"]
        agg["commit"]["entry_moved"].append(entry)
        n_enemy = sum(1 for a in ch if a.get("target") in ("enemy", "enemy_general"))
        agg["commit"]["raid_len_enemy_moves"].append(n_enemy)
        if entry >= BIG_COMMIT:
            t0 = a0["t"]
            st = t0 - 1
            my_army = int(game.armies[st][game.owners[st] == s].sum())
            en_army = int(game.armies[st][game.owners[st] == e].sum())
            seen = game.sight is not None and st >= game.sight
            # what the 10 prior actions were
            prev = [game.act_at(tt) for tt in range(max(1, t0 - 10), t0)]
            gather_frac = sum(1 for p in prev
                              if p["kind"] == "move" and p.get("target") == "own") / max(1, len(prev))
            # post-commit path: chain moves after entry
            after = ch[first_enemy:][:POST_COMMIT_MOVES]
            toward = [game.dist_gen[tuple(a["dst"])] < game.dist_gen[tuple(a["src"])]
                      for a in after if a["kind"] == "move"]
            ended_general = any(a.get("target") == "enemy_general" and a.get("captured")
                                for a in ch[first_enemy:])
            agg["commit"]["big"].append({
                "match_id": mid, "t": t0, "entry": entry,
                "general_seen": bool(seen),
                "army_ratio": round(my_army / max(1, en_army), 3),
                "gather_frac_prev10": round(gather_frac, 2),
                "toward_frac": round(float(np.mean(toward)), 3) if toward else None,
                "ended_at_general": bool(ended_general),
                "enemy_moves": n_enemy,
            })

    # ---- E. enemy castles -------------------------------------------------
    enemy_castles = [(tuple(a["cell"]), a["t"]) for a in game.eacts if a["kind"] == "build"]
    for cell, bt in enemy_castles:
        vis_col = game.vis[:, cell[0], cell[1]]
        vis_after = vis_col[bt:]
        first_vis = int(np.argmax(vis_after)) + bt if vis_after.any() else None
        cap = next((a for a in game.acts
                    if a["kind"] == "move" and tuple(a["dst"]) == cell
                    and a.get("captured") and a.get("target") in ("enemy", "enemy_general")
                    and a["t"] > bt), None)
        row = {"match_id": mid, "cell": list(cell), "built_t": bt,
               "first_visible_t": first_vis,
               "captured_t": cap["t"] if cap else None,
               "garrison": cap["dst_army"] if cap else None,
               "moved": cap["moved"] if cap else None}
        agg["castles"]["enemy_castles"].append(row)
        if first_vis is not None:
            if cap:
                agg["castles"]["vis_to_capture"].append(cap["t"] - first_vis)
                agg["castles"]["capture_margin"].append(cap["moved"] - cap["dst_army"])
            else:
                agg["castles"]["visible_never_captured"] += 1
    # castle-vs-plain choice: attack whose src also had a visible enemy castle
    castle_cells = {c for c, bt in enemy_castles}
    if castle_cells:
        for t in range(1, game.T + 1):
            a = game.act_at(t)
            if not (a["kind"] == "move" and a.get("target") in ("enemy", "enemy_general")):
                continue
            st = t - 1
            src = tuple(a["src"])
            adj = []
            for dr, dc in ORTH:
                nr, nc = src[0] + dr, src[1] + dc
                if 0 <= nr < game.R and 0 <= nc < game.C and game.enemy_vis[st][nr, nc]:
                    adj.append((nr, nc))
            adj_castle = [c for c in adj if c in castle_cells
                          and game.owners[st][c] == e]
            adj_plain = [c for c in adj if c not in castle_cells]
            if adj_castle and adj_plain:
                agg["castles"]["choice_n"] += 1
                if tuple(a["dst"]) in adj_castle:
                    agg["castles"]["chose_castle"] += 1
                else:
                    agg["castles"]["chose_plain_examples"].append(
                        {"match_id": mid, "t": t, "castle": list(adj_castle[0]),
                         "castle_army": int(game.armies[st][adj_castle[0]]),
                         "dst_army": a["dst_army"], "moved": a["moved"]})

    # ---- F. general strike ------------------------------------------------
    if game.won and game.kill_t is not None and game.sight is not None:
        agg["strike"]["sight_to_kill"].append(game.kill_t - game.sight)
        kill_act = game.act_at(game.kill_t)
        agg["strike"]["kill_moved"].append(kill_act["moved"])
        agg["strike"]["kill_dst_army"].append(kill_act["dst_army"])
        first_hit = next((a for a in game.acts
                          if a["kind"] == "move" and a.get("target") == "enemy_general"),
                         None)
        if first_hit is not None:
            agg["strike"]["sight_to_first_general_hit"].append(
                first_hit["t"] - game.sight)
        st = game.sight
        own = game.owners[st] == s
        agg["strike"]["stack_at_sight"].append(
            int(game.armies[st][own].max()) if own.any() else 0)
        agg["strike"]["defender_at_sight"].append(
            int(game.armies[st][game.gen_e]))
        # locate the killing chain
        chains = build_chains(game)
        kill_chain = next((ch for ch in chains
                           if any(a["t"] == game.kill_t for a in ch)), None)
        if kill_chain:
            idx = next(i for i, a in enumerate(kill_chain) if a["t"] == game.kill_t)
            chain = kill_chain[:idx + 1]
            dep = chain[0]
            dep_cell, dep_t = tuple(dep["src"]), dep["t"]
            agg["strike"]["departure_army"].append(dep["src_army"])
            agg["strike"]["departure_after_sight"].append(dep_t - game.sight)
            bfs = int(game.dist_gen[dep_cell])
            agg["strike"]["path_len"].append(len(chain))
            agg["strike"]["bfs_len"].append(bfs)
            if bfs > 0:
                agg["strike"]["path_over_bfs"].append(len(chain) / bfs)
            # fog memory: chain moves after sight while the general is NOT
            # currently visible — do they still close BFS distance?
            for a in chain:
                stt = a["t"] - 1
                if stt < game.sight:
                    continue
                visible_now = bool(game.vis[stt, game.gen_e[0], game.gen_e[1]])
                toward = game.dist_gen[tuple(a["dst"])] < game.dist_gen[tuple(a["src"])]
                key = "visible" if visible_now else "fogged"
                agg["strike"]["memory"][key]["n"] += 1
                agg["strike"]["memory"][key]["toward"] += int(toward)
        # failed strikes before the kill
        fails = [a for a in game.acts
                 if a["kind"] == "move" and a.get("target") == "enemy_general"
                 and not a.get("captured") and a["t"] < game.kill_t]
        agg["strike"]["failed_general_hits"].append(len(fails))
        if fails:
            agg["strike"]["multi_wave_games"].append(
                {"match_id": mid, "fails": len(fails),
                 "first_fail_t": fails[0]["t"], "kill_t": game.kill_t})
    if game.won and game.kill_t is None:
        agg["strike"]["win_without_kill"].append(
            {"match_id": mid, "unresolved": game.g["unresolved"], "T": game.T})
    if game.won and game.kill_t is not None and game.sight is None:
        agg["strike"]["kill_without_prior_sight"].append(
            {"match_id": mid, "kill_t": game.kill_t})

    # ---- G. pre-sight hunt ------------------------------------------------
    if game.contact is not None:
        sight = game.sight if game.sight is not None else game.T + 1
        toward_n = avail_toward = avail_total = 0
        cap_n = 0
        for t in range(game.contact + 1, min(sight, game.T) + 1):
            a = game.act_at(t)
            if not (a["kind"] == "move" and a.get("target") == "enemy"
                    and a.get("captured")):
                continue
            st = t - 1
            cap_n += 1
            if game.dist_gen[tuple(a["dst"])] < game.dist_gen[tuple(a["src"])]:
                toward_n += 1
            pairs = adjacency_pairs(game, st)
            opts = {p[1] for p in pairs}
            avail_total += len(opts)
            avail_toward += sum(1 for d in opts
                                if game.dist_gen[d] < min(game.dist_gen[p[0]]
                                                          for p in pairs if p[1] == d))
        if cap_n:
            agg["hunt"]["games"].append({
                "match_id": mid, "captures": cap_n,
                "toward_rate": round(toward_n / cap_n, 3),
                "baseline": round(avail_toward / max(1, avail_total), 3)})
            agg["hunt"]["toward_rates"].append(toward_n / cap_n)
            agg["hunt"]["baselines"].append(avail_toward / max(1, avail_total))

    # ---- H. deathtouch ----------------------------------------------------
    if game.T >= 800:
        pokes = [a for a in game.acts
                 if a["kind"] == "move" and a.get("target") == "enemy_general"
                 and a["t"] - 1 >= 800]
        agg["deathtouch"]["games"].append({
            "match_id": mid, "T": game.T, "outcome": game.g["outcome"],
            "sight": game.sight, "kill_t": game.kill_t,
            "general_moves_after_800": [
                {"t": a["t"], "moved": a["moved"], "dst_army": a["dst_army"],
                 "captured": a.get("captured")} for a in pokes],
            "attacks_700_800": sum(1 for a in game.acts
                                   if a["kind"] == "move"
                                   and a.get("target") in ("enemy", "enemy_general")
                                   and 700 <= a["t"] < 800),
            "attacks_800_900": sum(1 for a in game.acts
                                   if a["kind"] == "move"
                                   and a.get("target") in ("enemy", "enemy_general")
                                   and 800 <= a["t"] < 900),
        })

    # ---- I. losses / draw -------------------------------------------------
    if game.g["outcome"] in ("lose", "draw"):
        enemy_kill = next((a for a in game.eacts
                           if a["kind"] == "move" and a.get("target") == "enemy_general"
                           and a.get("captured")), None)
        # where was Kubic's biggest stack at the end, relative to its general
        stf = game.T - 1
        own = game.owners[stf] == s
        big_cell = None
        if own.any():
            flat = np.where(own, game.armies[stf], -1)
            big_cell = np.unravel_index(int(flat.argmax()), flat.shape)
        dist_home = bfs_dist(game.passable, game.gen_k)
        last20 = [game.act_at(t) for t in range(max(1, game.T - 19), game.T + 1)]
        cls = Counter((a["kind"] if a["kind"] != "move" else "move_" + a["target"])
                      for a in last20)
        rec.update(loss_detail={
            "enemy_kill_t": enemy_kill["t"] if enemy_kill else None,
            "kubic_attacks_total": sum(1 for a in game.acts
                                       if a["kind"] == "move"
                                       and a.get("target") in ("enemy", "enemy_general")),
            "max_stack_end": int(game.armies[stf][own].max()) if own.any() else 0,
            "big_stack_dist_from_own_general":
                int(dist_home[big_cell]) if big_cell else None,
            "big_stack_dist_from_enemy_general":
                int(game.dist_gen[big_cell]) if big_cell else None,
            "last20_actions": dict(cls),
            "general_hits_by_kubic": sum(
                1 for a in game.acts if a["kind"] == "move"
                and a.get("target") == "enemy_general"),
        })
    agg["per_game"].append(rec)


# ----------------------------------------------------------------- assemble

def main():
    rows = sorted(split_rows("fit"), key=lambda r: r["match_id"])
    agg = {
        "contact": {"army_ratio": [], "max_stack": [], "vis_enemy_army": [],
                    "latency": [], "first_attack_moved": [],
                    "first_attack_margin": [], "no_attack_games": []},
        "engage": {"situations": Counter(), "attacked": Counter(),
                   "instead": {b: Counter() for b in BUCKETS},
                   "attack_margin": [], "attack_split": Counter(),
                   "nonwinning_attacks": [], "onset_to_attack": [],
                   "maxstack_front_n": 0, "maxstack_front_attacked": 0,
                   "maxstack_front_attacked_other": 0,
                   "maxstack_decline_moves": 0, "maxstack_decline_toward_gen": 0,
                   "maxstack_front_instead": Counter()},
        "target_sel": {"pre": Counter(), "post": Counter()},
        "commit": {"entry_moved": [], "raid_len_enemy_moves": [], "big": []},
        "castles": {"enemy_castles": [], "vis_to_capture": [],
                    "capture_margin": [], "visible_never_captured": 0,
                    "choice_n": 0, "chose_castle": 0, "chose_plain_examples": []},
        "strike": {"sight_to_kill": [], "stack_at_sight": [],
                   "kill_moved": [], "kill_dst_army": [],
                   "sight_to_first_general_hit": [],
                   "defender_at_sight": [], "departure_army": [],
                   "departure_after_sight": [], "path_len": [], "bfs_len": [],
                   "path_over_bfs": [],
                   "memory": {"visible": {"n": 0, "toward": 0},
                              "fogged": {"n": 0, "toward": 0}},
                   "failed_general_hits": [], "multi_wave_games": [],
                   "win_without_kill": [], "kill_without_prior_sight": []},
        "hunt": {"games": [], "toward_rates": [], "baselines": []},
        "deathtouch": {"games": []},
        "per_game": [],
        "mix": Counter(),
    }
    unresolved = 0
    for i, row in enumerate(rows):
        game = Game(row)
        unresolved += len(game.g["unresolved"])
        analyze_game(game, agg)
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(rows)}", flush=True)

    # ---------- summarize
    eng = agg["engage"]
    curve = {}
    for b in BUCKETS:
        n = eng["situations"][b]
        k = eng["attacked"][b]
        curve[b] = {"situations": n, "attacked": k,
                    "rate": round(k / n, 4) if n else None,
                    "instead_top": dict(eng["instead"][b].most_common(4))}
    margins = eng["attack_margin"]
    entry = agg["commit"]["entry_moved"]
    out = {
        "meta": {
            "fit_games": len(rows),
            "unresolved_ticks_total": unresolved,
            "definitions": {
                "first_contact": "first state tick with any enemy-owned cell inside Kubic's 8-neighbourhood vision",
                "first_sight": "first state tick with the enemy general cell inside vision",
                "situation": "state with a Kubic stack>=2 orthogonally adjacent to a visible enemy cell",
                "full_send_margin": "stack-1 - enemy_cell_army (split=0 send)",
                "attack_margin": "moved - dst_army at decision time",
                "raid_chain": "moves linked by src == previous dst within 30 ticks (split=0 links)",
                "tick_semantics": "action at tick t reads state t-1",
            },
            "fog_legality": {
                "engage/target min_dst_army/max_margin/castles": "fog-legal (adjacent cells are visible)",
                "dist_gen pre-sight (hunt)": "NOT fog-legal - belief-level analysis vs true general",
                "dist_gen post-sight": "fog-legal position memory; true-map BFS assumes known terrain (caveat)",
            },
        },
        "A_contact": {
            "army_ratio": dist_summary(agg["contact"]["army_ratio"], "kubic/enemy total army at contact"),
            "max_stack": dist_summary(agg["contact"]["max_stack"], "max Kubic stack at contact"),
            "vis_enemy_army": dist_summary(agg["contact"]["vis_enemy_army"], "visible enemy army at contact"),
            "contact_to_first_attack": dist_summary(agg["contact"]["latency"], "ticks contact->first enemy-target move"),
            "first_attack_moved": dist_summary(agg["contact"]["first_attack_moved"], "army sent on first attack"),
            "first_attack_margin": dist_summary(agg["contact"]["first_attack_margin"], "moved-dst_army on first attack"),
            "games_with_no_attack": agg["contact"]["no_attack_games"],
        },
        "B_engagement": {
            "curve_by_best_margin": curve,
            "attack_margin": dist_summary(margins, "moved-dst_army over all attacks in situations"),
            "attack_margin_le0_count": sum(1 for m in margins if m <= 0),
            "attack_margin_hist_low": dict(Counter(
                m for m in margins if -5 <= m <= 10)),
            "split_usage": dict(eng["attack_split"]),
            "onset_to_attack": dist_summary(eng["onset_to_attack"], "ticks from positive-margin onset to attack"),
            "action_mix": dict(agg["mix"]),
            "nonwinning_attacks_n": len(eng["nonwinning_attacks"]),
            "nonwinning_by_target": dict(Counter(
                x["target"] for x in eng["nonwinning_attacks"])),
            "nonwinning_moved1_n": sum(
                1 for x in eng["nonwinning_attacks"] if x["moved"] == 1),
            "nonwinning_ties_n": sum(1 for x in eng["nonwinning_attacks"] if x["tie"]),
            "nonwinning_captured_within_3": sum(
                1 for x in eng["nonwinning_attacks"] if x["captured_within_3"]),
            "nonwinning_attacks_sample": eng["nonwinning_attacks"][:40],
            "maxstack_front": {
                "n": eng["maxstack_front_n"],
                "attacked_from_maxstack": eng["maxstack_front_attacked"],
                "attacked_from_other": eng["maxstack_front_attacked_other"],
                "rate": round(eng["maxstack_front_attacked"] /
                              max(1, eng["maxstack_front_n"]), 4),
                "instead_top": dict(eng["maxstack_front_instead"].most_common(6)),
                "decline_moves_from_maxstack": eng["maxstack_decline_moves"],
                "decline_toward_gen": eng["maxstack_decline_toward_gen"],
                "decline_toward_gen_rate": round(
                    eng["maxstack_decline_toward_gen"] /
                    max(1, eng["maxstack_decline_moves"]), 4),
            },
        },
        "C_target_selection": {
            phase: {
                "n": d.get("n", 0),
                **{k: {"agree": d.get(k, 0),
                       "rate": round(d.get(k, 0) / max(1, d.get("n", 0)), 4)}
                   for k in ("min_dist_gen", "min_dst_army", "max_margin",
                             "max_src_stack")},
                "dst_choice": {
                    "n": d.get("dst_choice_n", 0),
                    "min_dist_gen": d.get("dst_min_dist_gen", 0),
                    "min_dist_gen_rate": round(
                        d.get("dst_min_dist_gen", 0) / max(1, d.get("dst_choice_n", 0)), 4),
                    "min_army": d.get("dst_min_army", 0),
                    "min_army_rate": round(
                        d.get("dst_min_army", 0) / max(1, d.get("dst_choice_n", 0)), 4),
                },
                "chain_continuation": {
                    "n": d.get("chain_n", 0),
                    "continued": d.get("chain_continuation", 0),
                    "rate": round(d.get("chain_continuation", 0) /
                                  max(1, d.get("chain_n", 0)), 4),
                },
            }
            for phase, d in ((p, dict(agg["target_sel"][p])) for p in ("pre", "post"))
        },
        "D_commit": {
            "entry_moved": dist_summary(entry, "army moved on first enemy-target move of a raid chain"),
            "entry_moved_hist": dict(Counter(min(v, 60) for v in entry)),
            "raid_len_enemy_moves": dist_summary(agg["commit"]["raid_len_enemy_moves"], "enemy-target moves per raid"),
            "big_commits_n": len(agg["commit"]["big"]),
            "big_commit_general_seen_rate": round(
                np.mean([b["general_seen"] for b in agg["commit"]["big"]]), 4)
                if agg["commit"]["big"] else None,
            "big_commit_army_ratio": dist_summary(
                [b["army_ratio"] for b in agg["commit"]["big"]], "army ratio at big commit"),
            "big_commit_gather_frac": dist_summary(
                [b["gather_frac_prev10"] for b in agg["commit"]["big"]], "own-target move frac in prev 10 actions"),
            "big_commit_toward_frac": dist_summary(
                [b["toward_frac"] for b in agg["commit"]["big"] if b["toward_frac"] is not None],
                "post-commit chain moves closing BFS dist to general"),
            "big_commit_ended_at_general_rate": round(
                np.mean([b["ended_at_general"] for b in agg["commit"]["big"]]), 4)
                if agg["commit"]["big"] else None,
            "big_commit_sample": agg["commit"]["big"][:50],
        },
        "E_castles": {
            "enemy_castles_built_n": len(agg["castles"]["enemy_castles"]),
            "ever_visible_n": sum(1 for r in agg["castles"]["enemy_castles"]
                                  if r["first_visible_t"] is not None),
            "captured_n": sum(1 for r in agg["castles"]["enemy_castles"]
                              if r["captured_t"] is not None),
            "visible_never_captured": agg["castles"]["visible_never_captured"],
            "vis_to_capture": dist_summary(agg["castles"]["vis_to_capture"], "ticks castle visible->captured"),
            "capture_margin": dist_summary(agg["castles"]["capture_margin"], "moved-garrison at castle capture"),
            "castle_vs_plain_choice": {
                "n": agg["castles"]["choice_n"],
                "chose_castle": agg["castles"]["chose_castle"],
                "rate": round(agg["castles"]["chose_castle"] / max(1, agg["castles"]["choice_n"]), 4),
                "chose_plain_examples": agg["castles"]["chose_plain_examples"][:20],
            },
            "castles_detail_sample": agg["castles"]["enemy_castles"][:80],
        },
        "F_strike": {
            "sight_to_kill": dist_summary(agg["strike"]["sight_to_kill"], "ticks first sight->kill"),
            "sight_to_first_general_hit": dist_summary(
                agg["strike"]["sight_to_first_general_hit"],
                "ticks sight->first move onto general tile (negative = poked before sight tick? n/a)"),
            "kill_moved": dist_summary(agg["strike"]["kill_moved"], "army moved on the killing move"),
            "kill_dst_army": dist_summary(agg["strike"]["kill_dst_army"], "defender army at the kill"),
            "stack_at_sight": dist_summary(agg["strike"]["stack_at_sight"], "max Kubic stack at sight"),
            "defender_at_sight": dist_summary(agg["strike"]["defender_at_sight"], "army on enemy general at sight"),
            "departure_army": dist_summary(agg["strike"]["departure_army"], "src army when kill chain departs"),
            "departure_after_sight": dist_summary(agg["strike"]["departure_after_sight"], "ticks sight->kill-chain departure (negative = already moving)"),
            "path_over_bfs": dist_summary(agg["strike"]["path_over_bfs"], "kill-chain moves / BFS dist from departure"),
            "memory": {k: {"n": v["n"], "toward": v["toward"],
                           "toward_rate": round(v["toward"] / max(1, v["n"]), 4)}
                       for k, v in agg["strike"]["memory"].items()},
            "failed_general_hits": dist_summary(agg["strike"]["failed_general_hits"], "failed hits on general before kill"),
            "first_wave_success_rate": round(
                np.mean([f == 0 for f in agg["strike"]["failed_general_hits"]]), 4)
                if agg["strike"]["failed_general_hits"] else None,
            "multi_wave_games": agg["strike"]["multi_wave_games"],
            "win_without_kill": agg["strike"]["win_without_kill"],
            "kill_without_prior_sight": agg["strike"]["kill_without_prior_sight"],
        },
        "G_hunt": {
            "toward_rate": dist_summary(agg["hunt"]["toward_rates"], "per-game rate of pre-sight captures closing dist to true general"),
            "baseline": dist_summary(agg["hunt"]["baselines"], "per-game rate available options would close dist"),
            "games_sample": agg["hunt"]["games"][:40],
        },
        "H_deathtouch": agg["deathtouch"],
        "I_losses": [r for r in agg["per_game"] if r["outcome"] in ("lose", "draw")],
        "per_game_wins_sample": [r for r in agg["per_game"] if r["outcome"] == "win"][:20],
    }
    OUT_JSON.write_text(json.dumps(out, indent=1, default=int))
    print("wrote", OUT_JSON)
    # console digest
    print(json.dumps({
        "contact_latency": out["A_contact"]["contact_to_first_attack"],
        "curve": {b: curve[b]["rate"] for b in BUCKETS},
        "attack_margin": out["B_engagement"]["attack_margin"],
        "nonwinning": out["B_engagement"]["nonwinning_attacks_n"],
        "target_sel": out["C_target_selection"],
        "entry_moved": out["D_commit"]["entry_moved"],
        "sight_to_kill": out["F_strike"]["sight_to_kill"],
        "memory": out["F_strike"]["memory"],
        "hunt": {"toward": out["G_hunt"]["toward_rate"], "base": out["G_hunt"]["baseline"]},
        "castle_choice": out["E_castles"]["castle_vs_plain_choice"]["rate"],
    }, indent=1, default=float))


if __name__ == "__main__":
    main()
