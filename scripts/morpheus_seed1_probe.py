#!/usr/bin/env python3
"""Turn-level probe for Morpheus-vs-smoke on one seed (heuristic path).

Classifies each Morpheus move and tracks king-stack progress toward the seek
target. Prints phase summaries and the top failure modes.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import Counter
from io import StringIO
from pathlib import Path

import jax.numpy as jnp
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--seat", type=int, default=0)
    p.add_argument("--turns", type=int, default=1200)
    p.add_argument("--sample-every", type=int, default=50)
    ns = p.parse_args()

    sys.path[:0] = [
        str(REPO),
        str(REPO / "competition-module"),
        str(REPO / "bots"),
        str(REPO / "bots/morpheus"),
    ]
    from arena.matches.loop import make_board, make_transition
    from competition.protocol import encode_observation
    from generals import GeneralsEnv
    from _common.wire import _read_observation
    from memory import OWNER_ENEMY, OWNER_NEUTRAL, empty_memory, update_memory
    from tactics import (
        enemy_is_visible,
        enemy_seek_target,
        move_progress,
        select_play_action,
        structure_idle_army,
    )
    from transition import DIRECTIONS

    spec = importlib.util.spec_from_file_location(
        "smoke_agent", REPO / "bots/smoke/agent.py"
    )
    smoke_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(smoke_mod)

    morph_seat = int(ns.seat)
    env = GeneralsEnv(mode="competition")
    st = make_board(env, int(ns.seed))
    H, W = (int(x) for x in st.armies.shape)
    step = make_transition(env)
    smoke = smoke_mod.Agent(1 - morph_seat, H, W)
    mem = empty_memory(H, W)

    def read_obs(state, seat):
        from generals.core import game as game_mod

        cm = game_mod.get_observation(state, seat)
        frame = encode_observation(cm)
        lines = frame.splitlines()
        return _read_observation(
            StringIO("\n".join(lines[1:]) + "\n"), H, W, lines[0]
        )

    counts: Counter[str] = Counter()
    phase_counts: dict[str, Counter[str]] = {
        "pre": Counter(),
        "post": Counter(),
    }
    last_m = (1, 0, 0, 0, 0)
    recent_m: list = []
    contact = None
    gen_seen = False
    samples: list[str] = []
    dist_hist: list[tuple[int, int, int, str]] = []

    for t in range(int(ns.turns)):
        obs_m = read_obs(st, morph_seat)
        obs_s = read_obs(st, 1 - morph_seat)
        mem = update_memory(mem, obs_m)
        seen = enemy_is_visible(obs_m, mem)
        if contact is None and seen:
            contact = t
        types = np.asarray(obs_m.type_grid)
        owners = np.asarray(obs_m.owner_grid)
        armies = np.asarray(obs_m.army_grid)
        if np.any(mem.known_enemy_general) or np.any((types == 4) & (owners == 2)):
            gen_seen = True

        a_m = select_play_action(
            obs_m, mem, prev_action=last_m, recent_actions=tuple(recent_m)
        )
        last_m = tuple(int(x) for x in a_m)  # type: ignore[assignment]
        if int(last_m[0]) == 0:
            recent_m.append(last_m)
            if len(recent_m) > 8:
                recent_m = recent_m[-8:]

        target = enemy_seek_target(obs_m, mem)
        phase = "post" if seen else "pre"
        kind = "pass"
        progress = 0.0
        src_army = 0
        if int(last_m[0]) == 1:
            kind = "pass"
        elif int(last_m[0]) == 2:
            kind = "castle"
        elif int(last_m[0]) == 0:
            sr, sc, d = int(last_m[1]), int(last_m[2]), int(last_m[3])
            tr = sr + int(DIRECTIONS[d, 0])
            tc = sc + int(DIRECTIONS[d, 1])
            src_army = int(armies[sr, sc]) if 0 <= sr < H and 0 <= sc < W else 0
            if not (0 <= tr < H and 0 <= tc < W):
                kind = "oob"
            else:
                dest_o = int(owners[tr, tc])
                progress = move_progress(sr, sc, tr, tc, target)
                if dest_o == OWNER_ENEMY:
                    kind = "attack"
                elif dest_o == OWNER_NEUTRAL:
                    kind = "expand"
                elif dest_o == 1:
                    if progress > 0:
                        kind = "own_toward"
                    elif progress < 0:
                        kind = "own_away"
                    else:
                        kind = "own_lateral"
                else:
                    kind = f"own_other_{dest_o}"

        counts[kind] += 1
        phase_counts[phase][kind] += 1

        own = owners == 1
        max_army = int(armies[own].max()) if np.any(own) else 0
        if max_army > 0:
            locs = np.argwhere(own & (armies == max_army))
            kr, kc = int(locs[0, 0]), int(locs[0, 1])
        else:
            kr = kc = -1
        dist = -1
        if target is not None and kr >= 0:
            dist = abs(kr - target[0]) + abs(kc - target[1])
        idle = structure_idle_army(obs_m, mem)
        land_m = int(own.sum())
        land_e = int((owners == OWNER_ENEMY).sum())
        enemy_tiles = land_e

        if t % int(ns.sample_every) == 0 or t == contact:
            samples.append(
                f"t={t:4d} phase={phase:4s} kind={kind:11s} prog={progress:+.0f} "
                f"srcA={src_army:3d} maxA={max_army:3d} dist={dist:3d} "
                f"land={land_m:3d}/{enemy_tiles:3d} idle={idle:3d} "
                f"tgt={target}"
            )
        if seen and t % 25 == 0:
            dist_hist.append((t, max_army, dist, kind))

        a_s = smoke.act(obs_s)
        actions = [None, None]
        actions[morph_seat] = list(a_m)
        actions[1 - morph_seat] = list(a_s)
        st, _ = step(st, jnp.array(actions, dtype=jnp.int32))
        w = int(getattr(st, "winner", -1))
        if w >= 0:
            winner = "morph" if w == morph_seat else "smoke"
            print(f"ENDED winner={winner} turns={t+1} gen_seen={gen_seen}")
            break
    else:
        winner = "draw"
        print(
            f"ENDED winner=draw turns={ns.turns} "
            f"land_m={int(st.ownership[morph_seat].sum())} "
            f"land_s={int(st.ownership[1-morph_seat].sum())} "
            f"contact={contact} gen_seen={gen_seen}"
        )

    print("\n== samples ==")
    for line in samples:
        print(line)

    print("\n== move class totals ==")
    for k, v in counts.most_common():
        print(f"  {k:12s} {v:5d}")

    print("\n== pre-contact ==")
    for k, v in phase_counts["pre"].most_common():
        print(f"  {k:12s} {v:5d}")
    print("== post-contact ==")
    for k, v in phase_counts["post"].most_common():
        print(f"  {k:12s} {v:5d}")

    if dist_hist:
        print("\n== post king-stack distance (every 25t) ==")
        for t, max_a, dist, kind in dist_hist:
            print(f"  t={t:4d} maxA={max_a:3d} dist={dist:3d} kind={kind}")

    post = phase_counts["post"]
    own_moves = post["own_toward"] + post["own_away"] + post["own_lateral"]
    combat = post["attack"] + post["expand"]
    print("\n== aggression ratios (post) ==")
    print(f"  own_moves={own_moves} combat_or_expand={combat} attacks={post['attack']}")
    if own_moves:
        print(
            f"  toward={post['own_toward']} away={post['own_away']} "
            f"lateral={post['own_lateral']}"
        )
    return 0 if winner == "morph" else 1


if __name__ == "__main__":
    raise SystemExit(main())
