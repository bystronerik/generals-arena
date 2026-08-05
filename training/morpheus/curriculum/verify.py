"""Verify curriculum manifests against trajectory digests and fog observations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from arena.paths import REPO_ROOT
from arena.records.trajectories import (
    read_trajectory,
    require_same_era,
    verify_trajectory,
)
from training.morpheus.curriculum.definitions import is_banned_source
from training.morpheus.curriculum.reconstruct import (
    fog_observations_at_prefix,
    observation_fingerprint,
    replay_prefix_states,
)
from training.morpheus.curriculum.schema import CurriculumItem, CurriculumManifest


def _resolve_traj_path(item: CurriculumItem, root: Path) -> Path | None:
    if not item.trajectory_relpath:
        return None
    path = Path(item.trajectory_relpath)
    if path.is_file():
        return path
    cand = root / item.trajectory_relpath
    return cand if cand.is_file() else path


def verify_manifest(
    manifest_path: Path,
    *,
    repo_root: Path | None = None,
    max_items: int | None = None,
    check_belief: bool = False,
    n_particles: int = 4,
    require_sample_seat: bool = False,
) -> dict[str, Any]:
    """
    Reproduce trajectory digests and both-seat fog observations at each prefix.

    Returns a report with ``ok`` True only when every checked item passes and
    no ResBot source appears.
    """
    root = repo_root or REPO_ROOT
    manifest = CurriculumManifest.load(manifest_path)
    items = list(manifest.items)
    if max_items is not None:
        items = items[: int(max_items)]

    mismatches: list[str] = []
    digest_checked = 0
    fog_checked = 0
    belief_checked = 0
    traj_cache: dict[str, Any] = {}
    source_labels: set[str] = set()

    # Confidence rule must be present and executable.
    rule = manifest.confidence_rule or {}
    if rule.get("interval_method") != "wilson":
        mismatches.append("confidence_rule.interval_method must be 'wilson'")
    if "min_samples_per_active_class" not in rule:
        mismatches.append("confidence_rule missing min_samples_per_active_class")
    if not rule.get("selected_before_main_run"):
        mismatches.append("confidence_rule.selected_before_main_run must be true")

    for item in items:
        source_labels.add(item.source_label)
        if is_banned_source(item.source_label):
            mismatches.append(f"{item.item_id}: banned source {item.source_label}")
            continue
        if require_sample_seat and item.class_id != 5 and item.sample_seat is None:
            mismatches.append(f"{item.item_id}: missing sample_seat")
            continue
        if item.sample_seat is not None and item.sample_seat not in (0, 1):
            mismatches.append(f"{item.item_id}: invalid sample_seat {item.sample_seat}")
            continue

        traj = None
        if item.class_id != 5:
            path = _resolve_traj_path(item, root)
            if path is None or not path.is_file():
                mismatches.append(f"{item.item_id}: missing trajectory {item.trajectory_relpath}")
                continue
            key = str(path)
            if key not in traj_cache:
                traj = read_trajectory(path)
                require_same_era(traj, engine=item.engine_version)
                report = verify_trajectory(traj, engine=item.engine_version)
                digest_checked += 1
                if not report.ok:
                    mismatches.append(f"{item.item_id}: {report.summary()}")
                    traj_cache[key] = None
                    continue
                traj_cache[key] = traj
            else:
                traj = traj_cache[key]
                if traj is None:
                    continue

            if int(traj.seed) != int(item.map_seed):
                mismatches.append(
                    f"{item.item_id}: map_seed {item.map_seed} != traj seed {traj.seed}"
                )
            if traj.engine_version != item.engine_version:
                mismatches.append(
                    f"{item.item_id}: engine_version mismatch with trajectory"
                )

        # Fog observations at prefix (independent replay).
        try:
            fog_a = fog_observations_at_prefix(
                item, trajectories_root=root, traj=traj, engine=item.engine_version
            )
            # Second independent replay for equality.
            _turn, state = replay_prefix_states(
                traj=traj,
                map_seed=item.map_seed,
                prefix_len=item.prefix_len,
                engine=item.engine_version,
            )
            from training.morpheus.curriculum.reconstruct import engine_fog_observation

            fog_b = {
                0: engine_fog_observation(state, 0),
                1: engine_fog_observation(state, 1),
            }
            for seat in (0, 1):
                if observation_fingerprint(fog_a[seat]) != observation_fingerprint(
                    fog_b[seat]
                ):
                    mismatches.append(
                        f"{item.item_id}: seat {seat} fog observation mismatch"
                    )
            fog_checked += 1
        except Exception as exc:  # noqa: BLE001 — report and continue
            mismatches.append(f"{item.item_id}: fog replay failed: {exc}")

        if check_belief:
            try:
                from training.morpheus.curriculum.reconstruct import reconstruct_prefix

                recon = reconstruct_prefix(
                    item,
                    trajectories_root=root,
                    traj=traj,
                    n_particles=n_particles,
                    update_belief_flag=True,
                    engine=item.engine_version,
                )
                for seat in (0, 1):
                    seat_rec = recon.seats[seat]
                    if seat_rec.belief_seed != item.belief_seed(seat):
                        mismatches.append(
                            f"{item.item_id}: seat {seat} belief seed drift"
                        )
                    if len(seat_rec.action_history) != item.prefix_len:
                        mismatches.append(
                            f"{item.item_id}: seat {seat} action history "
                            f"len {len(seat_rec.action_history)} != {item.prefix_len}"
                        )
                belief_checked += 1
            except Exception as exc:  # noqa: BLE001
                mismatches.append(f"{item.item_id}: belief reconstruct failed: {exc}")

    ok = not mismatches
    return {
        "ok": ok,
        "item_count": len(items),
        "digest_checked": digest_checked,
        "fog_checked": fog_checked,
        "belief_checked": belief_checked,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "confidence_rule": rule.get("name"),
        "source_labels": sorted(source_labels),
        "class_counts": {
            str(cid): sum(1 for i in manifest.items if i.class_id == cid)
            for cid in sorted({i.class_id for i in manifest.items})
        },
    }
