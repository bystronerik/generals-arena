#!/usr/bin/env python3
"""Parallel Morpheus-vs-smoke seed grid (fast heuristic path + optional full bot).

Default mode uses ``select_play_action`` + smoke with memory updates only — no
network load — so many seeds finish in seconds. Pass ``--full-bot`` to exercise
the real Agent emit path (slower).
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import StringIO
from pathlib import Path

import jax.numpy as jnp
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _run_one(args: tuple) -> dict:
    seed, morph_seat, turns, full_bot = args
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
    from memory import empty_memory, update_memory
    from tactics import enemy_is_visible, select_play_action

    spec = importlib.util.spec_from_file_location(
        "smoke_agent", REPO / "bots/smoke/agent.py"
    )
    smoke_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(smoke_mod)

    env = GeneralsEnv(mode="competition")
    st = make_board(env, seed)
    H, W = (int(x) for x in st.armies.shape)
    step = make_transition(env)
    smoke = smoke_mod.Agent(1 - morph_seat, H, W)
    morph = None
    mem = empty_memory(H, W)
    if full_bot:
        import agent as morph_mod

        morph = morph_mod.Agent(morph_seat, H, W)

    def read_obs(state, seat):
        from generals.core import game as game_mod

        cm = game_mod.get_observation(state, seat)
        frame = encode_observation(cm)
        lines = frame.splitlines()
        return _read_observation(
            StringIO("\n".join(lines[1:]) + "\n"), H, W, lines[0]
        )

    contact = None
    gen_seen = False
    last_m = (1, 0, 0, 0, 0)
    recent_m: list = []
    for t in range(turns):
        obs_m = read_obs(st, morph_seat)
        obs_s = read_obs(st, 1 - morph_seat)
        if full_bot:
            assert morph is not None
            a_m = morph.act(obs_m)
            mem = morph._controller.memory
        else:
            mem = update_memory(mem, obs_m)
            a_m = select_play_action(
                obs_m, mem, prev_action=last_m, recent_actions=tuple(recent_m)
            )
        last_m = tuple(int(x) for x in a_m)  # type: ignore[assignment]
        if int(last_m[0]) == 0:
            recent_m.append(last_m)
            if len(recent_m) > 8:
                recent_m = recent_m[-8:]
        a_s = smoke.act(obs_s)
        if contact is None and enemy_is_visible(obs_m, mem):
            contact = t
        types = np.asarray(obs_m.type_grid)
        owners = np.asarray(obs_m.owner_grid)
        if np.any(mem.known_enemy_general) or np.any((types == 4) & (owners == 2)):
            gen_seen = True
        actions = [None, None]
        actions[morph_seat] = list(a_m)
        actions[1 - morph_seat] = list(a_s)
        st, _ = step(st, jnp.array(actions, dtype=jnp.int32))
        w = int(getattr(st, "winner", -1))
        if w >= 0:
            return {
                "seed": seed,
                "seat": morph_seat,
                "winner": "morph" if w == morph_seat else "smoke",
                "turns": t + 1,
                "land_m": int(st.ownership[morph_seat].sum()),
                "land_s": int(st.ownership[1 - morph_seat].sum()),
                "contact": contact,
                "gen_seen": gen_seen,
            }
    return {
        "seed": seed,
        "seat": morph_seat,
        "winner": "draw",
        "turns": turns,
        "land_m": int(st.ownership[morph_seat].sum()),
        "land_s": int(st.ownership[1 - morph_seat].sum()),
        "contact": contact,
        "gen_seen": gen_seen,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--turns", type=int, default=1200)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--both-seats", action="store_true")
    p.add_argument("--full-bot", action="store_true")
    ns = p.parse_args()
    jobs = []
    seats = (0, 1) if ns.both_seats else (0,)
    for seed in range(ns.seeds):
        for seat in seats:
            jobs.append((seed, seat, ns.turns, ns.full_bot))
    wins = draws = losses = 0
    print("seed seat winner turns land_m/land_s contact gen_seen")
    with ProcessPoolExecutor(max_workers=ns.workers) as pool:
        futs = [pool.submit(_run_one, job) for job in jobs]
        for fut in as_completed(futs):
            r = fut.result()
            print(
                f"{r['seed']:4d} {r['seat']:4d} {r['winner']:5s} {r['turns']:4d} "
                f"{r['land_m']:3d}/{r['land_s']:3d} {str(r['contact']):>6} {r['gen_seen']}"
            )
            if r["winner"] == "morph":
                wins += 1
            elif r["winner"] == "draw":
                draws += 1
            else:
                losses += 1
    print(f"SUMMARY wins={wins} draws={draws} losses={losses} / {len(jobs)}")
    return 0 if losses == 0 and draws == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
