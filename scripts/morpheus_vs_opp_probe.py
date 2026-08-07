#!/usr/bin/env python3
"""Probe Morpheus-vs-macaria on one seed (heuristic path).

Tracks army concentration: general share, idle 1-army tiles, gather vs attack.
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


def _load_agent(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Ensure bots/<name>/ imports resolve (macaria needs sibling packages).
    bot_dir = path.parent
    bots_root = bot_dir.parent
    for p in (str(bots_root), str(bot_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--seat", type=int, default=0)
    p.add_argument("--turns", type=int, default=1200)
    p.add_argument("--sample-every", type=int, default=40)
    p.add_argument(
        "--opp",
        type=str,
        default="macaria",
        help="opponent bot folder under bots/",
    )
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
        path_distance_field,
        path_progress,
        seek_goals,
        select_play_action,
        structure_idle_army,
    )
    from transition import DIRECTIONS

    opp_mod = _load_agent(REPO / "bots" / ns.opp / "agent.py", f"{ns.opp}_agent")

    morph_seat = int(ns.seat)
    env = GeneralsEnv(mode="competition")
    st = make_board(env, int(ns.seed))
    H, W = (int(x) for x in st.armies.shape)
    step = make_transition(env)
    opp = opp_mod.Agent(1 - morph_seat, H, W)
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
    phase_counts: dict[str, Counter[str]] = {"pre": Counter(), "post": Counter()}
    last_m = (1, 0, 0, 0, 0)
    recent_m: list = []
    contact = None
    gen_seen = False
    samples: list[str] = []

    for t in range(int(ns.turns)):
        obs_m = read_obs(st, morph_seat)
        obs_o = read_obs(st, 1 - morph_seat)
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
        goals = seek_goals(obs_m, mem)
        dist = path_distance_field(obs_m, goals) if goals else None
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
            if 0 <= tr < H and 0 <= tc < W:
                dest_o = int(owners[tr, tc])
                progress = path_progress(
                    sr, sc, tr, tc, dist, fallback_target=target
                )
                if dest_o == OWNER_ENEMY:
                    kind = "attack"
                elif dest_o == OWNER_NEUTRAL:
                    kind = "expand"
                elif dest_o == 1:
                    dest_a = int(armies[tr, tc])
                    if src_army <= 3 and dest_a >= src_army:
                        kind = "gather_in"
                    elif progress > 0:
                        kind = "own_toward"
                    elif progress < 0:
                        kind = "own_away"
                    else:
                        kind = "own_lateral"
                else:
                    kind = "other"

        counts[kind] += 1
        phase_counts[phase][kind] += 1

        own = owners == 1
        land = int(own.sum())
        total_a = int(armies[own].sum()) if land else 0
        ones = int(((own) & (armies == 1)).sum()) if land else 0
        max_a = int(armies[own].max()) if land else 0
        gen_mask = np.asarray(mem.own_general, dtype=bool)
        if not np.any(gen_mask):
            gen_mask = (types == 4) & own
        gen_a = int(armies[gen_mask].sum()) if np.any(gen_mask) else 0
        idle = structure_idle_army(obs_m, mem)
        gen_share = (gen_a / total_a) if total_a else 0.0
        max_share = (max_a / total_a) if total_a else 0.0
        ones_frac = (ones / land) if land else 0.0

        if t % int(ns.sample_every) == 0 or t == contact:
            samples.append(
                f"t={t:4d} {phase:4s} {kind:11s} prog={progress:+.0f} "
                f"srcA={src_army:3d} maxA={max_a:3d} genA={gen_a:3d} "
                f"totA={total_a:4d} gen%={gen_share:4.0%} max%={max_share:4.0%} "
                f"1s={ones:3d}/{land:3d}({ones_frac:3.0%}) idle={idle:3d}"
            )

        a_o = opp.act(obs_o)
        actions = [None, None]
        actions[morph_seat] = list(a_m)
        actions[1 - morph_seat] = list(a_o)
        st, _ = step(st, jnp.array(actions, dtype=jnp.int32))
        w = int(getattr(st, "winner", -1))
        if w >= 0:
            winner = "morph" if w == morph_seat else ns.opp
            print(
                f"ENDED winner={winner} turns={t+1} gen_seen={gen_seen} "
                f"land_m={int(st.ownership[morph_seat].sum())} "
                f"land_o={int(st.ownership[1-morph_seat].sum())}"
            )
            break
    else:
        winner = "draw"
        print(
            f"ENDED winner=draw turns={ns.turns} "
            f"land_m={int(st.ownership[morph_seat].sum())} "
            f"land_o={int(st.ownership[1-morph_seat].sum())} "
            f"contact={contact} gen_seen={gen_seen}"
        )

    print("\n== samples ==")
    for line in samples:
        print(line)
    print("\n== move classes ==")
    for k, v in counts.most_common():
        print(f"  {k:12s} {v:5d}")
    print("== post ==")
    for k, v in phase_counts["post"].most_common():
        print(f"  {k:12s} {v:5d}")
    return 0 if winner == "morph" else 1


if __name__ == "__main__":
    raise SystemExit(main())
