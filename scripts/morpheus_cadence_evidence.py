#!/usr/bin/env python3
"""Part 13 cadence evidence CLI.

Examples:

  python scripts/morpheus_cadence_evidence.py class-wdl \\
    --artifact bots/morpheus_ckpt200/artifact \\
    --output docs/research/measurements/morpheus-cadence-class-wdl.json

  python scripts/morpheus_cadence_evidence.py belief \\
    --prior /tmp/morpheus-ckpt100 --later /tmp/morpheus-ckpt200 \\
    --output docs/research/measurements/morpheus-cadence-belief.json

  python scripts/morpheus_cadence_evidence.py pairwise \\
    --round morpheus-cadence-ckpt100-ckpt200 \\
    --prior-bot morpheus_ckpt100 --later-bot morpheus_ckpt200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from training.morpheus.compute.cadence_evidence import (
    decision_rule_verdict,
    held_out_class_items,
    load_manifest_items,
    measure_belief_calibration,
    measure_class_wdl,
    pairwise_star_specs,
)

DEFAULT_MANIFEST = REPO / "training/morpheus/manifests/scraped-classes13.json"
DEFAULT_PANEL = REPO / "scripts/configs/morpheus/bootstrap-panel.json"
MEAS = REPO / "docs/research/measurements"


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "output": str(path)}, indent=2))


def cmd_class_wdl(args: argparse.Namespace) -> int:
    class_id = int(args.class_id)
    items = held_out_class_items(
        load_manifest_items(Path(args.manifest)),
        class_id=class_id,
        holdout_mod=args.holdout_mod,
        holdout_residue=args.holdout_residue,
        one_per_game=True,
    )
    print(
        f"[class-wdl] class_id={class_id} held_out_games={len(items)} "
        f"artifact={args.artifact}"
    )
    result = measure_class_wdl(
        items,
        artifact_dir=Path(args.artifact),
        class_id=class_id,
        min_wins=args.min_wins,
        min_losses=args.min_losses,
        max_games=args.max_games,
        n_particles=args.n_particles,
    )
    payload = {
        "ok": not result.insufficient,
        **result.to_dict(),
        "manifest": str(args.manifest),
        "artifact": str(args.artifact),
        "min_wins": args.min_wins,
        "min_losses": args.min_losses,
    }
    _write(Path(args.output), payload)
    return 0 if not result.insufficient else 2


def cmd_belief(args: argparse.Namespace) -> int:
    class_id = int(args.class_id)
    items = held_out_class_items(
        load_manifest_items(Path(args.manifest)),
        class_id=class_id,
        holdout_mod=args.holdout_mod,
        holdout_residue=args.holdout_residue,
        one_per_game=True,
    )
    print(
        f"[belief] class_id={class_id} held_out_games={len(items)} "
        f"scoring max={args.max_items}"
    )
    report = measure_belief_calibration(
        items,
        prior_checkpoint=Path(args.prior),
        later_checkpoint=Path(args.later),
        objective_path=Path(args.objective) if args.objective else None,
        max_items=args.max_items,
        n_particles=args.n_particles,
    )
    _write(Path(args.output), report)
    return 0


def cmd_pairwise(args: argparse.Namespace) -> int:
    from arena.paths import REPO_ROOT
    from arena.records.registry import Registry
    from arena.records.store import (
        bot_id_from_run_sh,
        engine_version,
        round_games_dir,
    )
    from arena.tournaments.competition import (
        ALTERNATING_SEATS,
        write_round_manifest,
    )
    from arena.tournaments.parallel import cap_jobs, default_jobs, run_pool
    from arena.tournaments.worker import run_one_worker
    from arena.records.ratings.cli import refit

    panel_cfg = json.loads(Path(args.panel).read_text(encoding="utf-8"))
    panel_ids = [m["bot_id"] for m in panel_cfg["members"]]
    prior_run = REPO_ROOT / "bots" / args.prior_bot / "run.sh"
    later_run = REPO_ROOT / "bots" / args.later_bot / "run.sh"
    panel_runs = [REPO_ROOT / "bots" / bid / "run.sh" for bid in panel_ids]
    for path in [prior_run, later_run, *panel_runs]:
        if not path.is_file():
            raise SystemExit(f"missing run.sh: {path}")

    round_seed = int(args.round_seed if args.round_seed is not None else panel_cfg.get("round_seed", 7))
    games_per_pair = int(args.games_per_pair)
    round_name = str(args.round)
    specs = pairwise_star_specs(
        [prior_run, later_run],
        panel_runs,
        games_per_pair=games_per_pair,
        round_seed=round_seed,
        seat_policy=ALTERNATING_SEATS,
    )
    games_dir = round_games_dir(round_name)
    jobs = cap_jobs(default_jobs() if args.jobs is None else int(args.jobs))
    engine = engine_version()
    run_scripts = [prior_run, later_run, *panel_runs]
    content_hashes = Registry().register_run_scripts(run_scripts, strict=bool(args.strict_versions))
    write_round_manifest(
        games_dir,
        round_name=round_name,
        round_seed=round_seed,
        games_per_pair=games_per_pair,
        jobs=jobs,
        bots=[bot_id_from_run_sh(p) for p in run_scripts],
        specs=specs,
        seat_policy=ALTERNATING_SEATS,
        fixed_seeds=None,
        engine=engine,
        content_hashes=content_hashes,
    )
    print(
        f"[pairwise] round={round_name} matches={len(specs)} "
        f"jobs={jobs} games_dir={games_dir}"
    )
    payloads = [
        {
            "bot_a_run": str(a.resolve()),
            "bot_b_run": str(b.resolve()),
            "seed": seed,
            "games_dir": str(games_dir),
            "mode": "competition",
            "round": round_name,
            "timeout": args.timeout,
            "engine_version": engine,
            "bot_a_content_hash": content_hashes[bot_id_from_run_sh(a)],
            "bot_b_content_hash": content_hashes[bot_id_from_run_sh(b)],
            "trajectories_dir": None,
        }
        for a, b, seed in specs
    ]

    def _on_result(done: int, total: int, record) -> None:
        print(
            f"[pairwise] ({done}/{total}) {record.bot_a} vs {record.bot_b} "
            f"seed={record.seed} -> {record.winner} turns={record.turns}"
        )

    if not args.skip_play:
        run_pool(payloads, run_one_worker, jobs=jobs, on_result=_on_result)

    fit = refit()
    prior_entity = f"{args.prior_bot}@{content_hashes[args.prior_bot]}"
    later_entity = f"{args.later_bot}@{content_hashes[args.later_bot]}"
    delta = fit.delta(prior_entity, later_entity)
    verdict = decision_rule_verdict(delta)

    from arena.records.store import list_game_paths, load_game

    games = []
    for path in list_game_paths(games_dir):
        try:
            games.append(load_game(path).to_dict())
        except Exception:
            games.append(json.loads(path.read_text(encoding="utf-8")))

    def _arm_stats(bot: str) -> dict:
        rows = [
            g
            for g in games
            if g.get("bot_a") == bot or g.get("bot_b") == bot
        ]
        decisive = 0
        for g in rows:
            w = g.get("winner")
            if w in ("a", "b"):
                decisive += 1
        per_opp: dict[str, int] = {}
        for g in rows:
            other = g["bot_b"] if g.get("bot_a") == bot else g["bot_a"]
            per_opp[other] = per_opp.get(other, 0) + 1
        return {
            "games": len(rows),
            "decisive": decisive,
            "per_opponent": per_opp,
        }

    report = {
        "round": round_name,
        "round_seed": round_seed,
        "games_per_pair": games_per_pair,
        "seat_policy": "alternate",
        "prior_entity": prior_entity,
        "later_entity": later_entity,
        "delta": {
            "value": float(delta.value),
            "se": float(delta.se),
            "ci": [float(delta.ci[0]), float(delta.ci[1])],
            "p_stronger": float(delta.p_stronger),
            "comparable": bool(delta.comparable),
        },
        "pairwise_verdict": verdict,
        "arm_stats": {
            args.prior_bot: _arm_stats(args.prior_bot),
            args.later_bot: _arm_stats(args.later_bot),
        },
        "panel": panel_ids,
        "matches": len(specs),
        "content_hashes": content_hashes,
    }
    if verdict == "unproven":
        try:
            report["games_to_resolve"] = int(
                fit.games_to_resolve(prior_entity, later_entity, target_se=12.75)
            )
        except Exception as exc:  # noqa: BLE001
            report["games_to_resolve_error"] = str(exc)

    out = Path(args.output)
    _write(out, report)
    md = out.with_suffix(".md")
    md.write_text(
        "\n".join(
            [
                "# Morpheus cadence pairwise",
                "",
                f"Round: `{round_name}`",
                "",
                f"- prior: `{prior_entity}`",
                f"- later: `{later_entity}`",
                f"- Δ: {delta.value:.2f} ± {delta.se:.2f}",
                f"- CI₉₅: [{delta.ci[0]:.2f}, {delta.ci[1]:.2f}]",
                f"- P(later > prior): {delta.p_stronger:.4f}",
                f"- verdict: **{verdict}**",
                "",
                "Panel: " + ", ".join(panel_ids),
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[pairwise] wrote {md}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    wdl = sub.add_parser("class-wdl", help="Class-1 WDL vs smoke from held-out prefixes")
    wdl.add_argument("--class-id", type=int, default=1)
    wdl.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    wdl.add_argument("--artifact", type=Path, required=True)
    wdl.add_argument(
        "--output",
        type=Path,
        default=MEAS / "morpheus-cadence-class-wdl.json",
    )
    wdl.add_argument("--min-wins", type=int, default=32)
    wdl.add_argument("--min-losses", type=int, default=32)
    wdl.add_argument("--max-games", type=int, default=512)
    wdl.add_argument("--n-particles", type=int, default=4)
    wdl.add_argument("--holdout-mod", type=int, default=10)
    wdl.add_argument("--holdout-residue", type=int, default=0)
    wdl.set_defaults(func=cmd_class_wdl)

    bel = sub.add_parser("belief", help="Held-out belief-head compare prior vs later")
    bel.add_argument("--class-id", type=int, default=1)
    bel.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    bel.add_argument("--prior", type=Path, required=True)
    bel.add_argument("--later", type=Path, required=True)
    bel.add_argument("--objective", type=Path, default=None)
    bel.add_argument(
        "--output",
        type=Path,
        default=MEAS / "morpheus-cadence-belief.json",
    )
    bel.add_argument("--max-items", type=int, default=64)
    bel.add_argument("--n-particles", type=int, default=4)
    bel.add_argument("--holdout-mod", type=int, default=10)
    bel.add_argument("--holdout-residue", type=int, default=0)
    bel.set_defaults(func=cmd_belief)

    pair = sub.add_parser("pairwise", help="Full bootstrap panel both arms")
    pair.add_argument("--round", default="morpheus-cadence-ckpt100-ckpt200")
    pair.add_argument("--prior-bot", default="morpheus_ckpt100")
    pair.add_argument("--later-bot", default="morpheus_ckpt200")
    pair.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    pair.add_argument("--games-per-pair", type=int, default=50)
    pair.add_argument("--round-seed", type=int, default=None)
    pair.add_argument("--jobs", type=int, default=None)
    pair.add_argument("--timeout", type=float, default=None)
    pair.add_argument("--strict-versions", action="store_true")
    pair.add_argument("--skip-play", action="store_true", help="refit only from stored games")
    pair.add_argument(
        "--output",
        type=Path,
        default=MEAS / "morpheus-cadence-pairwise.json",
    )
    pair.set_defaults(func=cmd_pairwise)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
