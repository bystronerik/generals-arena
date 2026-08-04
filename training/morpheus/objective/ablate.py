"""Objective ablation runner and measurement report (Part 12).

A100 wall time from this path charges to Part 13. Selection for the main run
stays unset until belief, entropy, coverage, cycling, and held-out arena
evidence exist — this module never invents a silent default weight.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from training.morpheus.objective.config import (
    ExplorationConfig,
    ObjectiveConfig,
    ObjectiveConfigError,
    assert_rated_disables_exploration,
    load_ablation_candidates,
)
from training.morpheus.objective.losses import (
    compute_objective_losses,
    hand_policy_ce,
    hand_wdl_ce,
)
from training.morpheus.objective.reward import (
    FORBIDDEN_REWARD_TERMS,
    assert_no_shaping,
    reward_contract,
    terminal_reward,
    wdl_one_hot,
)
from training.morpheus.objective.targets import (
    army_bin_index,
    apply_root_noise,
    build_seat_targets,
    sample_action_from_strategy,
    sparse_policy_to_dense,
)
from training.morpheus.objective.augmentation import (
    AugmentableSample,
    all_named_symmetries,
    round_trip_ok,
)
from training.morpheus.self_play.schema import SparsePolicy

REPO = Path(__file__).resolve().parents[3]
DEFAULT_JSON = REPO / "docs/research/measurements/morpheus-objective-ablation.json"
DEFAULT_MD = REPO / "docs/research/measurements/morpheus-objective-ablation.md"


def _fixture_board() -> dict[str, Any]:
    """Tiny 4×4 engine-truth board for hand checks (padded by target builders)."""
    H = W = 4
    ownership = np.zeros((2, H, W), dtype=bool)
    ownership[0, 0, 0] = True
    ownership[0, 0, 1] = True
    ownership[1, 3, 3] = True
    ownership[1, 3, 2] = True
    armies = np.zeros((H, W), dtype=np.int32)
    armies[0, 0] = 5
    armies[0, 1] = 2
    armies[3, 3] = 8
    armies[3, 2] = 3
    generals = np.zeros((H, W), dtype=bool)
    generals[0, 0] = True
    generals[3, 3] = True
    castles = np.zeros((H, W), dtype=bool)
    castles[3, 2] = True
    return {
        "ownership": ownership,
        "armies": armies,
        "generals": generals,
        "castles": castles,
        "final_land": (2, 2),
        "final_army": (7, 11),
        "final_castles": (0, 1),
        "current_turn": 10,
        "terminal_turn": 40,
        "winner": "a",
    }


def _hand_fixture_checks() -> dict[str, Any]:
    board = _fixture_board()
    policy = SparsePolicy(indices=(0, 1, 3969), probs=(0.2, 0.3, 0.5))
    targets = build_seat_targets(
        winner=board["winner"],
        seat=0,
        policy=policy,
        ownership=board["ownership"],
        armies=board["armies"],
        generals=board["generals"],
        castles=board["castles"],
        final_land=board["final_land"],
        final_army=board["final_army"],
        final_castles=board["final_castles"],
        current_turn=board["current_turn"],
        terminal_turn=board["terminal_turn"],
    )
    # Hand WDL / value
    assert targets.value == 1.0
    assert tuple(targets.wdl.tolist()) == (1.0, 0.0, 0.0)
    assert terminal_reward("a", seat=1) == -1.0
    assert wdl_one_hot("draw", seat=0) == (0.0, 1.0, 0.0)

    # Hand policy CE: equal logits vs one-hot target → -log(1/n)
    n = targets.policy.shape[0]
    logits = np.zeros(n, dtype=np.float64)
    one_hot = np.zeros(n, dtype=np.float64)
    one_hot[3969] = 1.0
    expected_ce = float(-np.log(1.0 / n))
    got_ce = hand_policy_ce(logits, one_hot)
    policy_ok = abs(got_ce - expected_ce) < 1e-5

    wdl_logits = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    wdl_target = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    expected_wdl = float(-np.log(1.0 / 3.0))
    got_wdl = hand_wdl_ce(wdl_logits, wdl_target)
    wdl_ok = abs(got_wdl - expected_wdl) < 1e-5

    # Army bin uses Part 04 edges
    bin0 = army_bin_index(0.0)
    bin_mid = army_bin_index(8.0)
    bins_ok = 0 <= bin0 < 16 and 0 <= bin_mid < 16 and bin_mid >= bin0

    # Reward shaping rejected
    shaping_rejected = False
    try:
        assert_no_shaping({"win": 1.0, "land": 0.1})
    except ValueError:
        shaping_rejected = True

    # Symmetry round-trip on a padded synthetic sample
    pad = targets.board_mask.shape[0]
    tensor = np.zeros((49, pad, pad), dtype=np.float32)
    tensor[0] = targets.board_mask
    tensor[7, 0, 0] = 1.0  # own general plane mark
    from training.morpheus.objective.augmentation import apply_symmetry

    sample = apply_symmetry(
        AugmentableSample(
            tensor=tensor,
            legal_mask=(targets.policy > 0).astype(np.float32),
            targets=targets,
        ),
        "id",
    )
    sym_ok = all(round_trip_ok(sample, name) for name in all_named_symmetries())

    return {
        "policy_ce_matches_hand": policy_ok,
        "wdl_ce_matches_hand": wdl_ok,
        "army_bins_versioned": bins_ok,
        "shaping_rejected": shaping_rejected,
        "symmetry_round_trips": sym_ok,
        "reward_contract": reward_contract(),
        "forbidden_reward_terms": sorted(FORBIDDEN_REWARD_TERMS),
        "hand_policy_ce": got_ce,
        "expected_policy_ce": expected_ce,
        "hand_wdl_ce": got_wdl,
        "expected_wdl_ce": expected_wdl,
    }


def _evaluate_candidate(config: ObjectiveConfig) -> dict[str, Any]:
    """Run fail-closed config + tiny loss forward for one candidate."""
    board = _fixture_board()
    policy = SparsePolicy(indices=(10, 20, 3969), probs=(0.1, 0.2, 0.7))
    targets = build_seat_targets(
        winner=board["winner"],
        seat=0,
        policy=policy,
        ownership=board["ownership"],
        armies=board["armies"],
        generals=board["generals"],
        castles=board["castles"],
        final_land=board["final_land"],
        final_army=board["final_army"],
        final_castles=board["final_castles"],
        current_turn=board["current_turn"],
        terminal_turn=board["terminal_turn"],
    )
    norm = config.scalar_normalization
    scaled = targets.scaled_scalars(norm)
    n = targets.policy.shape[0]
    pad = targets.board_mask.shape[0]
    n_bins = 16

    # Synthetic network outputs near targets so losses stay finite.
    policy_logits = torch.zeros(1, n)
    policy_logits[0] = torch.as_tensor(np.log(np.maximum(targets.policy, 1e-8)))
    wdl_logits = torch.tensor([[5.0, 0.0, -5.0]])
    hidden_owner = torch.as_tensor(targets.hidden_owner).view(1, 1, pad, pad)
    # Convert bin indices to one-hot logits
    army_logits = torch.zeros(1, n_bins, pad, pad)
    for r in range(pad):
        for c in range(pad):
            b = int(targets.enemy_army_bin[r, c])
            if b >= 0:
                army_logits[0, b, r, c] = 5.0
    enemy_general = torch.as_tensor(targets.enemy_general).view(1, 1, pad, pad) * 5.0
    hidden_castle = torch.as_tensor(targets.hidden_castle).view(1, 1, pad, pad)
    land_p = torch.tensor([[scaled["land_margin"]]])
    army_p = torch.tensor([[scaled["army_margin"]]])
    castle_p = torch.tensor([[scaled["castle_margin"]]])
    turns_p = torch.tensor([[scaled["turns_to_termination"]]])

    total, terms = compute_objective_losses(
        config=config,
        policy_logits=policy_logits,
        wdl_logits=wdl_logits,
        hidden_owner_logits=hidden_owner,
        enemy_army_logits=army_logits,
        enemy_general_logits=enemy_general,
        hidden_castle_logits=hidden_castle,
        land_margin_pred=land_p,
        army_margin_pred=army_p,
        castle_margin_pred=castle_p,
        turns_pred=turns_p,
        policy_target=torch.as_tensor(targets.policy).view(1, -1),
        wdl_target=torch.as_tensor(targets.wdl).view(1, 3),
        hidden_owner_target=torch.as_tensor(targets.hidden_owner).view(1, 1, pad, pad),
        enemy_army_bin_target=torch.as_tensor(targets.enemy_army_bin).view(1, pad, pad),
        enemy_general_target=torch.as_tensor(targets.enemy_general).view(1, 1, pad, pad),
        hidden_castle_target=torch.as_tensor(targets.hidden_castle).view(1, 1, pad, pad),
        land_margin_target=torch.tensor([[scaled["land_margin"]]]),
        army_margin_target=torch.tensor([[scaled["army_margin"]]]),
        castle_margin_target=torch.tensor([[scaled["castle_margin"]]]),
        turns_target=torch.tensor([[scaled["turns_to_termination"]]]),
        board_mask=torch.as_tensor(targets.board_mask).view(1, 1, pad, pad),
    )

    rng = np.random.default_rng(0)
    prior = sparse_policy_to_dense(policy)
    noisy = apply_root_noise(
        prior,
        epsilon=config.exploration.root_noise_epsilon,
        alpha=config.exploration.root_noise_alpha,
        rng=rng,
    )
    # Entropy of noisy / temperature path for coverage diagnostics
    p = np.maximum(noisy, 1e-12)
    p = p / p.sum()
    entropy = float(-(p * np.log(p)).sum())
    action = sample_action_from_strategy(
        noisy,
        temperature=config.exploration.action_temperature,
        turn=0,
        deterministic_turn=config.exploration.deterministic_turn,
        rng=rng,
    )
    return {
        "name": config.name,
        "ok": bool(np.isfinite(terms.total)),
        "loss_total": terms.total,
        "loss_terms": terms.to_dict()["terms"],
        "weights": config.loss_weights.to_dict(),
        "exploration": config.exploration.to_dict(),
        "policy_entropy": entropy,
        "sampled_action": int(action),
        "exploration_enabled": config.exploration.exploration_enabled,
    }


def run_ablation(
    config_path: Path | str,
    *,
    json_path: Path | None = None,
    md_path: Path | None = None,
    a100_hours: float = 0.0,
) -> dict[str, Any]:
    """Evaluate every explicit candidate; write measurement JSON/MD."""
    config_path = Path(config_path)
    candidates = load_ablation_candidates(config_path)
    fixture = _hand_fixture_checks()

    # Fail-closed probes
    omit_ok = False
    try:
        ObjectiveConfig.from_dict({"name": "bad"})
    except ObjectiveConfigError:
        omit_ok = True

    rated = ObjectiveConfig(
        name="rated-play",
        mode="rated",
        loss_weights=candidates[0].loss_weights,
        exploration=ExplorationConfig.rated(),
        scalar_normalization=candidates[0].scalar_normalization,
    )
    rated_ok = False
    try:
        assert_rated_disables_exploration(rated)
        rated_ok = True
    except ObjectiveConfigError:
        rated_ok = False

    rated_reject = False
    try:
        ObjectiveConfig(
            name="bad-rated",
            mode="rated",
            loss_weights=candidates[0].loss_weights,
            exploration=ExplorationConfig(
                root_noise_epsilon=0.25,
                root_noise_alpha=0.3,
                action_temperature=1.0,
                deterministic_turn=0,
            ),
            scalar_normalization=candidates[0].scalar_normalization,
        )
    except ObjectiveConfigError:
        rated_reject = True

    rows = [_evaluate_candidate(c) for c in candidates]
    checks = {
        "hand_targets_and_losses": bool(
            fixture["policy_ce_matches_hand"]
            and fixture["wdl_ce_matches_hand"]
            and fixture["army_bins_versioned"]
        ),
        "symmetry_round_trips": bool(fixture["symmetry_round_trips"]),
        "reward_no_shaping": bool(fixture["shaping_rejected"]),
        "omitted_weights_fail_closed": omit_ok,
        "rated_exploration_disabled": rated_ok and rated_reject,
        "all_candidates_finite_loss": all(r["ok"] for r in rows),
    }
    # Selection requires held-out arena strength evidence (Part 13 A100).
    # Fixture-only ablation never promotes a candidate into the main run.
    selected = None
    selection_note = (
        "No candidate is selected. Fixture checks passed, but belief "
        "calibration, action coverage, cycling, and held-out arena strength "
        "require the Part 13 A100 ablation budget. Charged a100_hours are "
        "recorded for that gate."
    )
    passed = all(checks.values())
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "part": "12-training-objective",
        "config_path": str(config_path),
        "decision": {
            "pass": passed,
            "verdict": "yes" if passed else "no",
            "selected_candidate": selected,
            "selection_note": selection_note,
            "checks": checks,
            "note": (
                "Targets, losses, augmentation, and fail-closed configs match "
                "hand fixtures. Main-run weights stay unset until Part 13 "
                "records arena evidence."
                if passed
                else "One or more Part 12 exit checks failed."
            ),
        },
        "accounting": {
            "a100_hours": float(a100_hours),
            "charged_to_part": "13-modal-compute-gate",
            "budget_note": "Part 12 objective ablations charge A100 time to Part 13.",
        },
        "fixture_checks": fixture,
        "candidates": rows,
        "metrics_pending_a100": [
            "belief_calibration",
            "policy_entropy_vs_coverage",
            "action_coverage",
            "cycling",
            "held_out_arena_strength",
        ],
    }
    out_json = Path(json_path) if json_path else DEFAULT_JSON
    out_md = Path(md_path) if md_path else DEFAULT_MD
    write_report(report, out_json, out_md)
    return report


def render_markdown(report: dict[str, Any]) -> str:
    d = report["decision"]
    lines = [
        "# Morpheus objective ablation",
        "",
        f"> Verdict: **{d['verdict']}** — {d['note']}",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Selection",
        "",
        f"Selected candidate: `{d.get('selected_candidate')}`",
        "",
        d.get("selection_note", ""),
        "",
        "## Checks",
        "",
        "| Check | Result |",
        "| --- | --- |",
    ]
    for name, ok in d["checks"].items():
        lines.append(f"| `{name}` | {'yes' if ok else 'no'} |")
    lines += [
        "",
        "## Accounting",
        "",
        f"```json\n{json.dumps(report.get('accounting'), indent=2)}\n```",
        "",
        "## Candidates",
        "",
        f"```json\n{json.dumps(report.get('candidates'), indent=2)}\n```",
        "",
        "## Metrics pending A100",
        "",
    ]
    for m in report.get("metrics_pending_a100") or []:
        lines.append(f"- `{m}`")
    lines.append("")
    return "\n".join(lines)


def write_report(
    report: dict[str, Any],
    json_path: Path,
    md_path: Path,
) -> None:
    json_path = Path(json_path)
    md_path = Path(md_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    md_path.write_text(render_markdown(report))
