#!/usr/bin/env python3
"""Compare raw NN, prior-shaped NN, and constrained final vs an opponent.

Goal: separate network passivity from reward/prior/constrain steering.
Does not import other bots' strategy modules — only the opponent Agent entry.
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
import torch

REPO = Path(__file__).resolve().parents[1]


def _load_opp(folder: str):
    """Load bots/<folder>/agent.py without colliding with Morpheus modules."""
    path = REPO / "bots" / folder / "agent.py"
    bot_dir = path.parent
    bots_root = bot_dir.parent
    # Park Morpheus modules that share names with the opponent package.
    parked: dict[str, object] = {}
    for k in list(sys.modules):
        if k in (
            "search",
            "params",
            "blitz_core",
            "agent",
            "probe",
            "tactics",
            "memory",
            "action",
        ):
            parked[k] = sys.modules.pop(k)
    # Prefer the opponent bot dir for sibling imports.
    inserted = []
    for p in (str(bots_root), str(bot_dir)):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
        inserted.append(p)
    try:
        spec = importlib.util.spec_from_file_location(f"{folder}_agent", path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
    finally:
        for p in inserted:
            if p in sys.path:
                sys.path.remove(p)
        sys.modules.update(parked)
    return mod


def _classify(obs, action, *, target) -> str:
    from memory import OWNER_ENEMY, OWNER_NEUTRAL
    from transition import DIRECTIONS

    a = tuple(int(x) for x in action)
    if int(a[0]) == 1:
        return "pass"
    if int(a[0]) == 2:
        return "castle"
    if int(a[0]) != 0:
        return "other"
    sr, sc, d = int(a[1]), int(a[2]), int(a[3])
    tr = sr + int(DIRECTIONS[d, 0])
    tc = sc + int(DIRECTIONS[d, 1])
    H, W = int(obs.H), int(obs.W)
    owners = np.asarray(obs.owner_grid)
    armies = np.asarray(obs.army_grid)
    if not (0 <= tr < H and 0 <= tc < W):
        return "oob"
    dest_o = int(owners[tr, tc])
    src_a = int(armies[sr, sc]) if 0 <= sr < H and 0 <= sc < W else 0
    prog = 0.0
    if target is not None:
        prog = float(
            (abs(sr - target[0]) + abs(sc - target[1]))
            - (abs(tr - target[0]) + abs(tc - target[1]))
        )
    if dest_o == OWNER_ENEMY:
        return "attack"
    if dest_o == OWNER_NEUTRAL:
        return "expand"
    if dest_o == 1:
        if prog > 0:
            return "own_toward"
        if prog < 0:
            return "own_away"
        if src_a <= 3:
            return "gather_tip"
        return "own_lateral"
    return "other"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--seat", type=int, default=0)
    p.add_argument("--turns", type=int, default=1200)
    p.add_argument("--sample-every", type=int, default=40)
    p.add_argument("--opp", type=str, default="macaria")
    ns = p.parse_args()

    sys.path[:0] = [
        str(REPO),
        str(REPO / "competition-module"),
        str(REPO / "bots"),
        str(REPO / "bots/morpheus"),
    ]

    # Import Morpheus first, then park/restore around opponent load.
    from arena.matches.loop import make_board, make_transition
    from competition.protocol import encode_observation
    from generals import GeneralsEnv
    from _common.wire import _read_observation
    import agent as morph_mod
    from action import decode_action
    from network import legal_normalized_policy
    from tactics import (
        apply_pre_contact_prior,
        army_concentration,
        constrain_nn_action,
        enemy_is_visible,
        enemy_seek_target,
        play_mask,
        structure_idle_army,
    )

    opp_mod = _load_opp(ns.opp)
    morph_seat = int(ns.seat)
    env = GeneralsEnv(mode="competition")
    st = make_board(env, int(ns.seed))
    H, W = (int(x) for x in st.armies.shape)
    step = make_transition(env)
    morph = morph_mod.Agent(morph_seat, H, W)
    opp = opp_mod.Agent(1 - morph_seat, H, W)

    def read_obs(state, seat):
        from generals.core import game as game_mod

        cm = game_mod.get_observation(state, seat)
        frame = encode_observation(cm)
        lines = frame.splitlines()
        return _read_observation(
            StringIO("\n".join(lines[1:]) + "\n"), H, W, lines[0]
        )

    raw_c: Counter[str] = Counter()
    shaped_c: Counter[str] = Counter()
    final_c: Counter[str] = Counter()
    phase_raw: dict[str, Counter[str]] = {"pre": Counter(), "post": Counter()}
    phase_final: dict[str, Counter[str]] = {"pre": Counter(), "post": Counter()}
    overrides = 0
    raw_eq_shaped = 0
    shaped_eq_final = 0
    samples: list[str] = []
    contact = None
    gen_seen = False

    for t in range(int(ns.turns)):
        obs_m = read_obs(st, morph_seat)
        obs_o = read_obs(st, 1 - morph_seat)
        ctrl = morph._controller
        # act() updates memory/belief inside; capture pre-state via a dry raw NN
        # after act using the same controller memory.
        a_final = morph.act(obs_m)
        a_final_t = tuple(int(x) for x in a_final)
        mem = ctrl.memory
        belief = ctrl.belief
        assert mem is not None

        seen = enemy_is_visible(obs_m, mem)
        if contact is None and seen:
            contact = t
        types = np.asarray(obs_m.type_grid)
        owners = np.asarray(obs_m.owner_grid)
        if np.any(mem.known_enemy_general) or np.any((types == 4) & (owners == 2)):
            gen_seen = True

        # Raw NN prior (no Morpheus reshape) vs shaped prior.
        ev = ctrl.evaluator
        x = ev._tensor(obs_m, mem, belief)  # type: ignore[attr-defined]
        policy, pass_logit, _wdl = ev.session.forward_policy_wdl(x)
        mask_play = np.asarray(play_mask(obs_m, mem), dtype=bool)
        mask_t = torch.from_numpy(mask_play).unsqueeze(0)
        raw_prior = (
            legal_normalized_policy(policy, pass_logit, mask_t)
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        shaped_prior = apply_pre_contact_prior(
            raw_prior, obs_m, mem, mask=mask_play, belief=belief
        )
        raw_idx = int(np.argmax(raw_prior))
        shaped_idx = int(np.argmax(shaped_prior))
        a_raw = tuple(int(x) for x in decode_action(raw_idx))
        a_shaped = tuple(int(x) for x in decode_action(shaped_idx))

        # What constrain would do to shaped top (policy fallback path).
        a_constrained = constrain_nn_action(
            obs_m,
            mem,
            a_shaped,
            prev_action=getattr(ctrl, "_last_action", None),
            recent_actions=tuple(getattr(ctrl, "_recent_actions", ())),
            prior=shaped_prior,
        )
        a_constrained_t = tuple(int(x) for x in a_constrained)

        target = enemy_seek_target(obs_m, mem)
        k_raw = _classify(obs_m, a_raw, target=target)
        k_shaped = _classify(obs_m, a_shaped, target=target)
        k_final = _classify(obs_m, a_final_t, target=target)
        raw_c[k_raw] += 1
        shaped_c[k_shaped] += 1
        final_c[k_final] += 1
        phase = "post" if seen else "pre"
        phase_raw[phase][k_raw] += 1
        phase_final[phase][k_final] += 1

        if a_raw == a_shaped:
            raw_eq_shaped += 1
        if a_shaped == a_final_t:
            shaped_eq_final += 1
        if a_shaped != a_constrained_t:
            overrides += 1

        share, max_a, tot = army_concentration(obs_m)
        idle = structure_idle_army(obs_m, mem)
        land = int((owners == 1).sum())
        if t % int(ns.sample_every) == 0 or t == contact:
            samples.append(
                f"t={t:4d} {phase:4s} raw={k_raw:11s} shaped={k_shaped:11s} "
                f"final={k_final:11s} max%={max_a / max(tot, 1):4.0%} "
                f"tot={tot:4d} land={land:3d} idle={idle:3d} "
                f"ovr={int(a_shaped != a_constrained_t)}"
            )

        a_o = opp.act(obs_o)
        actions = [None, None]
        actions[morph_seat] = list(a_final_t)
        actions[1 - morph_seat] = list(a_o)
        st, _ = step(st, jnp.array(actions, dtype=jnp.int32))
        w = int(getattr(st, "winner", -1))
        if w >= 0:
            winner = "morph" if w == morph_seat else ns.opp
            print(
                f"ENDED winner={winner} turns={t + 1} gen_seen={gen_seen} "
                f"contact={contact} "
                f"land_m={int(st.ownership[morph_seat].sum())} "
                f"land_o={int(st.ownership[1 - morph_seat].sum())}"
            )
            break
    else:
        winner = "draw"
        print(
            f"ENDED winner=draw turns={ns.turns} gen_seen={gen_seen} "
            f"contact={contact}"
        )

    n = max(sum(raw_c.values()), 1)
    print(f"\n== agreement ==")
    print(f"  raw==shaped     {raw_eq_shaped}/{n} ({100.0 * raw_eq_shaped / n:.0f}%)")
    print(f"  shaped==final   {shaped_eq_final}/{n} ({100.0 * shaped_eq_final / n:.0f}%)")
    print(f"  constrain_ovr   {overrides}/{n} ({100.0 * overrides / n:.0f}%)")

    def dump(title: str, c: Counter[str]) -> None:
        print(f"\n== {title} ==")
        for k, v in c.most_common():
            print(f"  {k:12s} {v:5d} ({100.0 * v / n:5.1f}%)")

    dump("raw NN top", raw_c)
    dump("shaped prior top", shaped_c)
    dump("final act", final_c)
    print("\n== post raw ==")
    for k, v in phase_raw["post"].most_common():
        print(f"  {k:12s} {v:5d}")
    print("== post final ==")
    for k, v in phase_final["post"].most_common():
        print(f"  {k:12s} {v:5d}")

    print("\n== samples ==")
    for line in samples:
        print(line)

    # Aggression summary for the verdict.
    def rate(c: Counter[str], keys: tuple[str, ...]) -> float:
        return 100.0 * sum(c[k] for k in keys) / n

    print("\n== aggression rates (% of turns) ==")
    for name, c in (("raw", raw_c), ("shaped", shaped_c), ("final", final_c)):
        print(
            f"  {name:7s} attack={rate(c, ('attack',)):5.1f} "
            f"expand={rate(c, ('expand',)):5.1f} "
            f"toward={rate(c, ('own_toward',)):5.1f} "
            f"away/pass={rate(c, ('own_away', 'pass')):5.1f}"
        )
    return 0 if winner == "morph" else 1


if __name__ == "__main__":
    raise SystemExit(main())
