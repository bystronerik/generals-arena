#!/usr/bin/env python3
"""Stage B behavioural probes for clocked Sosipolis from recorded trajectories.

Metrics (Kubic soft bands):
  1. land @ t=50 in [20, 25]
  2. wave land-gain rate >> gather (direction of ~2.6x+)
  3. chain continue rate when prev dst is reused as src
  4. no real castle before turn 116
  5. pass rate after t=50 < 0.05
  6. post-sight toward_frac (moves that close Manhattan to latched gen) >= 0.70

Usage:
  python scripts/analyze_sosipolis_clocked_probes.py sosipolis-kubic-clocked-probes
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from arena.records.trajectories import (
    read_jsonl_gz,
    read_trajectory,
    trajectory_path,
    trace_path,
)

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GATHER_LO, GATHER_HI = 10, 27
LAND50_LO, LAND50_HI = 20, 25
CASTLE_EARLIEST = 116
PASS_RATE_MAX = 0.05
TOWARD_GATE = 0.70


def decode_move(action: tuple[int, ...]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if not action or action[0] != 0:
        return None
    _, r, c, d, _ = action
    if not (0 <= int(d) < 4):
        return None
    dr, dc = DIRECTIONS[int(d)]
    return (int(r), int(c)), (int(r) + dr, int(c) + dc)


def is_gather(turn: int) -> bool:
    r = turn % 50
    return GATHER_LO <= r <= GATHER_HI


def seat_index(bot_a: str, bot_b: str) -> tuple[str, int]:
    if bot_a == "sosipolis":
        return "a", 0
    if bot_b == "sosipolis":
        return "b", 1
    raise ValueError("sosipolis not in game")


def load_trace(game_id: str, seat: str, traj_dir: Path) -> dict[int, dict]:
    path = trace_path(game_id, seat, traj_dir)
    out: dict[int, dict] = {}
    if not path.exists():
        return out
    for line in read_jsonl_gz(path):
        t = line.get("t")
        if t is None:
            continue
        out[int(t)] = line
    return out


def analyze_game(game: dict, traj_dir: Path) -> dict:
    game_id = game["game_id"]
    seat, idx = seat_index(game["bot_a"], game["bot_b"])
    traj = read_trajectory(trajectory_path(game_id, traj_dir))
    trace = load_trace(game_id, seat, traj_dir)

    land50 = None
    land50_probe = None
    gather_gains = 0
    wave_gains = 0
    gather_ticks = 0
    wave_ticks = 0
    passes_after_50 = 0
    moves_after_50 = 0
    chain_continue = 0
    chain_opps = 0
    prev_dst = None
    first_castle = -1
    toward_close = 0
    toward_moves = 0
    sight_turn = None
    enemy_gen = None

    for frame in traj.frames:
        t = frame.turn
        land = frame.land[idx]
        action = frame.action_a if seat == "a" else frame.action_b
        tr = trace.get(t, {})

        if t == 50:
            land50 = land
        if land50_probe is None and tr.get("land_at_50", -1) not in (-1, None):
            land50_probe = int(tr["land_at_50"])

        fct = tr.get("first_castle_turn", -1)
        if first_castle < 0 and fct not in (-1, None) and int(fct) >= 0:
            first_castle = int(fct)

        if sight_turn is None and (
            tr.get("enemy_general_sighted") in (1, True)
            or tr.get("phase") == "strike"
        ):
            sight_turn = t
            eg = tr.get("enemy_gen", "none")
            if eg and eg != "none" and "," in str(eg):
                er, ec = str(eg).split(",", 1)
                enemy_gen = (int(er), int(ec))

        # Land gains by clock (skip t<=2 forced pass zone noise).
        if t >= 3 and t < len(traj.frames) + 1:
            # Compare to previous frame land.
            pass
        if t > 1:
            prev = next((f for f in traj.frames if f.turn == t - 1), None)
            if prev is not None:
                gain = land - prev.land[idx]
                if gain > 0:
                    if is_gather(t):
                        gather_gains += gain
                    else:
                        wave_gains += gain
                if is_gather(t):
                    gather_ticks += 1
                else:
                    wave_ticks += 1

        if t > 50:
            moves_after_50 += 1
            if action[0] == 1:
                passes_after_50 += 1

        decoded = decode_move(tuple(action))
        if decoded is not None:
            src, dst = decoded
            if prev_dst is not None:
                chain_opps += 1
                if src == prev_dst:
                    chain_continue += 1
            prev_dst = dst

            if (
                sight_turn is not None
                and t >= sight_turn
                and enemy_gen is not None
            ):
                toward_moves += 1
                before = abs(src[0] - enemy_gen[0]) + abs(src[1] - enemy_gen[1])
                after = abs(dst[0] - enemy_gen[0]) + abs(dst[1] - enemy_gen[1])
                if after < before:
                    toward_close += 1
        elif action[0] == 1:
            # Pass does not break chain head identity in Kubic; keep prev_dst.
            pass

    land50_final = land50_probe if land50_probe is not None else land50
    gather_rate = gather_gains / gather_ticks if gather_ticks else 0.0
    wave_rate = wave_gains / wave_ticks if wave_ticks else 0.0
    ratio = (wave_rate / gather_rate) if gather_rate > 1e-9 else None
    chain_rate = chain_continue / chain_opps if chain_opps else None
    pass_rate = passes_after_50 / moves_after_50 if moves_after_50 else None
    toward_frac = toward_close / toward_moves if toward_moves else None

    return {
        "game_id": game_id,
        "seed": game.get("seed"),
        "seat": seat,
        "winner": game.get("winner"),
        "turns": game.get("turns"),
        "land_at_50": land50_final,
        "land50_in_band": (
            land50_final is not None and LAND50_LO <= land50_final <= LAND50_HI
        ),
        "gather_gain_per_tick": round(gather_rate, 4),
        "wave_gain_per_tick": round(wave_rate, 4),
        "wave_over_gather": None if ratio is None else round(ratio, 3),
        "chain_continue_rate": None if chain_rate is None else round(chain_rate, 3),
        "chain_opps": chain_opps,
        "first_castle_turn": first_castle,
        "castle_ok": first_castle < 0 or first_castle >= CASTLE_EARLIEST,
        "pass_rate_after_50": None if pass_rate is None else round(pass_rate, 4),
        "pass_ok": pass_rate is not None and pass_rate < PASS_RATE_MAX,
        "sight_turn": sight_turn,
        "toward_frac": None if toward_frac is None else round(toward_frac, 3),
        "toward_moves": toward_moves,
        "toward_ok": toward_frac is not None and toward_frac >= TOWARD_GATE,
    }


def median(xs: list[float | int]) -> float | None:
    if not xs:
        return None
    return float(statistics.median(xs))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("round_name", help="round under data/games and data/trajectories")
    parser.add_argument(
        "--write",
        action="store_true",
        help="write docs/research/measurements/<round>-probes.{json,md}",
    )
    args = parser.parse_args(argv)

    games_dir = Path("data/games") / args.round_name
    traj_dir = Path("data/trajectories") / args.round_name
    game_files = sorted(
        p for p in games_dir.glob("*.json") if p.name != "manifest.json"
    )
    if not game_files:
        print(f"no games in {games_dir}", flush=True)
        return 1
    if not traj_dir.exists():
        print(f"no trajectories in {traj_dir}; re-run with --record", flush=True)
        return 1

    rows = []
    for gf in game_files:
        game = json.loads(gf.read_text())
        if "sosipolis" not in (game["bot_a"], game["bot_b"]):
            continue
        try:
            rows.append(analyze_game(game, traj_dir))
        except FileNotFoundError as exc:
            print(f"skip {game['game_id']}: {exc}", flush=True)

    if not rows:
        print("no analyzable sosipolis games", flush=True)
        return 1

    land50s = [r["land_at_50"] for r in rows if r["land_at_50"] is not None]
    ratios = [r["wave_over_gather"] for r in rows if r["wave_over_gather"] is not None]
    chains = [r["chain_continue_rate"] for r in rows if r["chain_continue_rate"] is not None]
    passes = [r["pass_rate_after_50"] for r in rows if r["pass_rate_after_50"] is not None]
    toward = [r["toward_frac"] for r in rows if r["toward_frac"] is not None]
    castles = [r["first_castle_turn"] for r in rows if r["first_castle_turn"] >= 0]

    summary = {
        "round": args.round_name,
        "games": len(rows),
        "land_at_50_median": median(land50s),
        "land50_in_band_frac": (
            sum(1 for r in rows if r["land50_in_band"]) / len(rows)
        ),
        "wave_over_gather_median": median(ratios),
        "chain_continue_median": median(chains),
        "castle_ok_frac": sum(1 for r in rows if r["castle_ok"]) / len(rows),
        "first_castle_median": median(castles),
        "pass_rate_median": median(passes),
        "pass_ok_frac": sum(1 for r in rows if r["pass_ok"]) / len(rows),
        "sight_rate": sum(1 for r in rows if r["sight_turn"] is not None) / len(rows),
        "toward_frac_median": median(toward),
        "toward_ok_frac": (
            sum(1 for r in rows if r["toward_ok"]) / max(1, sum(1 for r in rows if r["toward_frac"] is not None))
        ),
        "gates": {
            "land50_band": median(land50s) is not None
            and LAND50_LO <= median(land50s) <= LAND50_HI,
            "wave_gt_gather": median(ratios) is not None and median(ratios) > 1.0,
            "chain_high": median(chains) is not None and median(chains) >= 0.50,
            "castle_earliest": all(r["castle_ok"] for r in rows),
            "pass_low": median(passes) is not None and median(passes) < PASS_RATE_MAX,
            "toward_gate": median(toward) is None
            or (median(toward) is not None and median(toward) >= TOWARD_GATE),
        },
        "games_detail": rows,
    }

    # Soft chain target is ~0.78; report 0.50 as architecture floor.
    print(f"games={summary['games']}")
    print(
        f"land@50 median={summary['land_at_50_median']} "
        f"in_band={summary['land50_in_band_frac']:.0%} gate={summary['gates']['land50_band']}"
    )
    print(
        f"wave/gather median={summary['wave_over_gather_median']} "
        f"gate={summary['gates']['wave_gt_gather']}"
    )
    print(
        f"chain continue median={summary['chain_continue_median']} "
        f"gate>={0.50} → {summary['gates']['chain_high']}"
    )
    print(
        f"castle ok={summary['castle_ok_frac']:.0%} "
        f"first_median={summary['first_castle_median']} "
        f"gate={summary['gates']['castle_earliest']}"
    )
    print(
        f"pass@>50 median={summary['pass_rate_median']} "
        f"gate={summary['gates']['pass_low']}"
    )
    print(
        f"sight_rate={summary['sight_rate']:.0%} "
        f"toward_frac median={summary['toward_frac_median']} "
        f"gate={summary['gates']['toward_gate']}"
    )

    if args.write:
        out_dir = Path("docs/research/measurements")
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = args.round_name
        if not stem.endswith("-probes"):
            stem = f"{stem}-probes"
        json_path = out_dir / f"{stem}.json"
        md_path = out_dir / f"{stem}.md"
        json_path.write_text(json.dumps(summary, indent=2) + "\n")
        gates = summary["gates"]
        lines = [
            f"# Stage B probes — `{args.round_name}`",
            "",
            f"Games: {summary['games']} (sosipolis seats with trajectories).",
            "",
            "| Probe | Target | Result | Gate |",
            "| --- | --- | --- | --- |",
            f"| Land @ t=50 | median ∈ [{LAND50_LO},{LAND50_HI}] | "
            f"{summary['land_at_50_median']} "
            f"({summary['land50_in_band_frac']:.0%} in band) | "
            f"{'PASS' if gates['land50_band'] else 'FAIL'} |",
            f"| Wave / gather land-gain | ≫ 1 (Kubic ~2.6×) | "
            f"{summary['wave_over_gather_median']} | "
            f"{'PASS' if gates['wave_gt_gather'] else 'FAIL'} |",
            f"| Chain continue | high (~0.78; floor 0.50) | "
            f"{summary['chain_continue_median']} | "
            f"{'PASS' if gates['chain_high'] else 'FAIL'} |",
            f"| Castle earliest | none before {CASTLE_EARLIEST} | "
            f"ok={summary['castle_ok_frac']:.0%}, "
            f"first median={summary['first_castle_median']} | "
            f"{'PASS' if gates['castle_earliest'] else 'FAIL'} |",
            f"| Pass rate after t=50 | < {PASS_RATE_MAX} | "
            f"{summary['pass_rate_median']} | "
            f"{'PASS' if gates['pass_low'] else 'FAIL'} |",
            f"| Post-sight toward_frac | ≥ {TOWARD_GATE} | "
            f"{summary['toward_frac_median']} "
            f"(sight rate {summary['sight_rate']:.0%}) | "
            f"{'PASS' if gates['toward_gate'] else 'FAIL'} |",
            "",
            f"Machine-readable: [`{stem}.json`]({stem}.json)",
            "",
        ]
        md_path.write_text("\n".join(lines))
        print(f"wrote {json_path}")
        print(f"wrote {md_path}")

    failed = [k for k, v in summary["gates"].items() if not v]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
