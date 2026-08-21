"""S3 Gate 1: does a fabricated successor frame carry a usable value signal?

Selection-plan S3 (docs/bots/joe-rs/selection-plan.md) proposes a depth-1
value re-rank: fabricate the successor observation for each candidate move,
forward it, and let the value head break near-ties. Its first gate is free
and offline: over the recorded corpus, fabricate the successor for the move
joe actually played at every turn and compare `value(fabricated)` against
`value(real next frame)`. If the fabrication error drowns the value
differences a re-rank would read, S3 dies here without costing a game.

The fabricator mirrors `competition-module/generals/core/game.py` for the
own-move arithmetic (split = floor(A/2), full = A-1; strict-greater combat
with |diff| remaining; build subtracts the live cost) and `global_update`
for growth (all owned cells at time % 50 == 0, structures at time % 2 == 0,
applied AFTER the time increment). Everything the plan holds static stays
static: the opponent's move, the opponent's growth, and every fog cell the
move does not enter. A move into fog is modeled as capturing an empty plain
— the most common mid-expansion case; the reveal around a captured cell is
not modeled. All of that is deliberate fabrication noise, and measuring it
is this gate's whole job.

Reported per corpus and per game:
  - corr(value_fab_taken, value_real_next), and |error| percentiles;
  - |value(t+1) - value(t)| percentiles — the move-to-move signal scale the
    plan names;
  - the top1/top2 fabricated value gap |v1 - v2|, overall and on near-tie
    turns (masked-logit margin < 0.1) — the differences a re-rank would
    actually read, where shared fabrication error partially cancels;
  - the fraction of near-tie turns (the plan claims ~5%).

Usage: .venv/bin/python bots/joe-rs/tools/s3_gate1.py [--games N] [--json OUT]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent.parent
JOE_DIR = REPO / "bots" / "joe"
GAMES = REPO / "data" / "joe" / "joe-rs-parity" / "games"
PAD = 21

sys.path.insert(0, str(JOE_DIR))
sys.path.insert(0, str(REPO / "bots"))

from capture_fixtures import parse_in_log  # noqa: E402  (sibling tool)

# game.py's DIRECTIONS: 0 up, 1 down, 2 left, 3 right.
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def fabricate(raw: np.ndarray, cost: np.ndarray, action) -> np.ndarray:
    """Successor raw tensor for one own action; opponent and fog held static.

    `raw` is the (14, H, W) frame tensor (channels per agent.py::frame_to_raw),
    `cost` the live build-cost grid, `action` the engine 5-tuple.
    """
    raw = raw.copy()
    p, r, c, d, s = (int(x) for x in action)
    h, w = raw.shape[1], raw.shape[2]
    t_next = int(raw[13, 0, 0]) + 1

    if p == 0:
        dr, dc = DIRS[d]
        tr, tc = r + dr, c + dc
        army = int(raw[0, r, c])
        moved = army // 2 if s else army - 1
        moved = max(0, min(moved, army - 1))
        if (
            0 <= tr < h and 0 <= tc < w and raw[5, r, c] > 0
            and moved > 0 and raw[3, tr, tc] == 0
        ):
            raw[0, r, c] -= moved
            if raw[5, tr, tc] > 0:  # reinforce
                raw[0, tr, tc] += moved
            else:
                # Attack. A fog cell is modeled as an empty plain: the engine
                # would fight whatever is really there, and the gap is
                # fabrication noise this gate exists to measure.
                in_fog = raw[7, tr, tc] > 0 or raw[8, tr, tc] > 0
                target_army = 0 if in_fog else int(raw[0, tr, tc])
                was_opp = raw[6, tr, tc] > 0
                if moved > target_army:
                    raw[0, tr, tc] = moved - target_army
                    raw[5, tr, tc] = 1.0
                    raw[4, tr, tc] = 0.0
                    raw[6, tr, tc] = 0.0
                    raw[7, tr, tc] = 0.0
                    raw[8, tr, tc] = 0.0
                    raw[9] += 1.0
                    raw[10] -= target_army
                    if was_opp:
                        raw[11] -= 1.0
                        raw[12] -= target_army
                else:
                    raw[0, tr, tc] = target_army - moved
                    raw[10] -= moved
    elif p == 2:
        price = int(cost[r, c])
        if raw[5, r, c] > 0 and raw[2, r, c] == 0 and raw[1, r, c] == 0:
            raw[0, r, c] -= price
            raw[2, r, c] = 1.0
            raw[10] -= price

    # Growth runs on the incremented time (game.py::step order).
    mine = raw[5] > 0
    if t_next % 50 == 0:
        raw[0][mine] += 1.0
        raw[10] += float(mine.sum())
    if t_next % 2 == 0:
        structures = mine & ((raw[1] > 0) | (raw[2] > 0))
        raw[0][structures] += 1.0
        raw[10] += float(structures.sum())
    raw[13] = float(t_next)
    return raw


def pct(a, qs=(50, 90, 99)):
    a = np.asarray(a, dtype=np.float64)
    return {f"p{q}": round(float(np.percentile(np.abs(a), q)), 4) for q in qs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=None, help="limit game count")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    import equinox as eqx
    import jax.numpy as jnp
    import jax.random as jrandom

    from _common.wire import Observation
    from agent import frame_to_raw
    from joe_net import HistoryTransformer
    from joe_obs import (
        augment_obs,
        build_cost_from_raw,
        compute_build_mask_from_raw,
        compute_valid_move_mask,
        decode_action,
        init_obs_state,
    )

    with open(JOE_DIR / "artifact" / "manifest.json") as f:
        manifest = json.load(f)
    arch = manifest["network"]
    pad_to = int(arch["pad_to"])
    template = HistoryTransformer(
        grid_size=pad_to, pad_to=pad_to,
        history_size=int(arch["history_size"]), patch_size=int(arch["patch_size"]),
        depth=int(arch["depth"]), embed_dim=int(arch["embed_dim"]),
        n_head=int(arch["n_head"]), ff_factor=int(arch["ff_factor"]),
        use_bf16=False, value_loss=arch["value_loss"], num_bins=int(arch["num_bins"]),
        v_min=float(arch["v_min"]), v_max=float(arch["v_max"]),
        key=jrandom.PRNGKey(0))
    net = eqx.tree_deserialise_leaves(
        str(JOE_DIR / "artifact" / manifest["weights"]), template)

    @eqx.filter_jit
    def step(net, raw, obs_state):
        """The bot's real per-turn path: masked logits, value, next state."""
        cost = build_cost_from_raw(raw)
        aug, new_state = augment_obs(raw, cost, obs_state)
        move = compute_valid_move_mask(raw[0], raw[5] > 0, raw[3] > 0)
        build = compute_build_mask_from_raw(raw, cost)
        temporal = jnp.stack(
            [new_state.opponent_army_history, new_state.opponent_land_history])
        logits, value, _ = net._forward(aug, move, build, temporal)
        return cost, logits, value, new_state

    @eqx.filter_jit
    def value_of(net, raw, obs_state):
        """Value of a frame under a scratch copy of the state (JAX is
        functional, so passing the state IS the scratch copy — the caller's
        state object is never mutated)."""
        cost = build_cost_from_raw(raw)
        aug, new_state = augment_obs(raw, cost, obs_state)
        move = compute_valid_move_mask(raw[0], raw[5] > 0, raw[3] > 0)
        build = compute_build_mask_from_raw(raw, cost)
        temporal = jnp.stack(
            [new_state.opponent_army_history, new_state.opponent_land_history])
        _, value, _ = net._forward(aug, move, build, temporal)
        return value

    logs = sorted(p for p in GAMES.glob("*.in.log")
                  if not p.name.startswith("synthetic"))
    if args.games:
        logs = logs[: args.games]

    fab_vals, real_next_vals, real_step_diffs = [], [], []
    top_gaps, top_gaps_neartie, margins = [], [], []
    top_gaps_neartie_distinct, neartie_distinct = [], []
    per_game = {}
    for log in logs:
        name = log.name.removesuffix(".in.log")
        player_id, H, W, frames = parse_in_log(log)
        state = init_obs_state(pad_to)
        prev = None  # (fab_value_for_taken, real_value_t)
        g_fab, g_real = [], []
        for t, (scalars, grids) in enumerate(frames):
            obs = Observation(
                H=H, W=W, turn=scalars[0], my_land=scalars[1], my_army=scalars[2],
                opp_land=scalars[3], opp_army=scalars[4],
                type_grid=grids[0].tolist(), owner_grid=grids[1].tolist(),
                army_grid=grids[2].tolist())
            raw = np.asarray(frame_to_raw(obs), dtype=np.float32)
            cost, logits, value, state = step(net, jnp.asarray(raw), state)
            value = float(value)
            if prev is not None:
                fab_vals.append(prev[0])
                real_next_vals.append(value)
                real_step_diffs.append(value - prev[1])
                g_fab.append(prev[0]); g_real.append(value)

            # Rank the masked logits; joe's played move IS the argmax.
            lg = np.asarray(logits, dtype=np.float32)
            order = np.argsort(lg)[::-1]
            i1, i2 = int(order[0]), int(order[1])
            margin = float(lg[i1] - lg[i2])
            margins.append(margin)
            a1 = np.asarray(decode_action(i1, pad_to), dtype=np.int32)
            a2 = np.asarray(decode_action(i2, pad_to), dtype=np.int32)
            cost_np = np.asarray(cost, dtype=np.int32)
            fab1 = fabricate(raw, cost_np, a1)
            fab2 = fabricate(raw, cost_np, a2)
            # A near-tie is often the half/full pair of one move; from a
            # 2-army cell both push 1 unit and the successors are identical,
            # so no evaluator could (or needs to) split them. Count them
            # apart: the re-rank's usable signal lives on distinct pairs.
            distinct = not np.array_equal(fab1, fab2)
            v1 = float(value_of(net, jnp.asarray(fab1), state))
            v2 = float(value_of(net, jnp.asarray(fab2), state))
            top_gaps.append(v1 - v2)
            if margin < 0.1:
                top_gaps_neartie.append(v1 - v2)
                neartie_distinct.append(distinct)
                if distinct:
                    top_gaps_neartie_distinct.append(v1 - v2)
            prev = (v1, value)  # taken action == top-1 for the argmax bot
        if g_fab:
            per_game[name] = {
                "turns": len(g_fab),
                "corr": round(float(np.corrcoef(g_fab, g_real)[0, 1]), 4),
                "err": pct(np.subtract(g_fab, g_real)),
            }
        print(f"[gate1] {name}: {per_game.get(name)}")

    err = np.subtract(fab_vals, real_next_vals)
    margins = np.asarray(margins)
    report = {
        "turns": len(fab_vals),
        "corr_fab_vs_real_next": round(float(np.corrcoef(fab_vals, real_next_vals)[0, 1]), 4),
        "fabrication_error": pct(err),
        "move_to_move_value_diff": pct(real_step_diffs),
        "top1_top2_fab_gap": pct(top_gaps),
        "top1_top2_fab_gap_neartie": pct(top_gaps_neartie) if top_gaps_neartie else None,
        "top1_top2_fab_gap_neartie_distinct": (
            pct(top_gaps_neartie_distinct) if top_gaps_neartie_distinct else None),
        "neartie_distinct_successor_fraction": (
            round(float(np.mean(neartie_distinct)), 4) if neartie_distinct else None),
        "neartie_fraction_margin_lt_0.1": round(float((margins < 0.1).mean()), 4),
        "per_game": per_game,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "per_game"}, indent=2))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"[gate1] wrote {args.json}")


if __name__ == "__main__":
    main()
