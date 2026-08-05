"""Part 10 curriculum — classifiers, seeds, confidence rule, reconstruct."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.morpheus

from training.morpheus.corpus.coverage import _ownership_contact
from training.morpheus.curriculum.classify import (
    assign_class,
    capture_or_defense_threat,
    both_generals_alive,
)
from training.morpheus.curriculum.confidence import (
    DEFAULT_CONFIDENCE_RULE,
    class_wdl_is_non_degenerate,
    may_advance_toward_earlier_class,
    wilson_interval,
)
from training.morpheus.curriculum.definitions import (
    CLASS_AFTER_SIGHT,
    CLASS_CONTACT,
    CLASS_FULL_START,
    CLASS_PRE_CONTACT,
    CLASS_TACTICAL,
    TACTICAL_HORIZON,
    belief_rng_seed,
    is_banned_source,
)
from training.morpheus.curriculum.sample import class_weights, sample_item
from training.morpheus.curriculum.schema import CurriculumItem, CurriculumManifest

REPO = Path(__file__).resolve().parents[3]
TRAJ_DIR = REPO / "data" / "trajectories" / "morpheus-bootstrap"
PANEL = REPO / "scripts" / "configs" / "morpheus" / "bootstrap-panel.json"


def test_ownership_contact_uses_engine_stack():
    own = np.zeros((2, 4, 4), dtype=bool)
    own[0, 1, 1] = True
    own[1, 1, 2] = True
    assert _ownership_contact(own) is True
    own2 = np.zeros((2, 4, 4), dtype=bool)
    own2[0, 0, 0] = True
    own2[1, 3, 3] = True
    assert _ownership_contact(own2) is False


def test_banned_resbot_sources():
    assert is_banned_source("ResBot")
    assert is_banned_source("scraped_resbot")
    assert is_banned_source("fixed_panel") is False
    assert is_banned_source("ResBot_reconstructions") is False
    assert is_banned_source("erik.bystron_reconstructions") is False


def test_belief_seed_is_deterministic_and_seat_sensitive():
    kwargs = dict(
        engine_version="abc",
        map_seed=7,
        source_label="fixed_panel",
        game_id="g1",
    )
    s0 = belief_rng_seed(**kwargs, seat=0)
    s1 = belief_rng_seed(**kwargs, seat=1)
    assert s0 == belief_rng_seed(**kwargs, seat=0)
    assert s0 != s1
    # Shared across prefixes of the same game.
    assert CurriculumItem.build(
        class_id=CLASS_PRE_CONTACT,
        engine_version="abc",
        map_seed=7,
        source_label="fixed_panel",
        prefix_len=3,
        game_id="g1",
    ).belief_seed(0) == CurriculumItem.build(
        class_id=CLASS_PRE_CONTACT,
        engine_version="abc",
        map_seed=7,
        source_label="fixed_panel",
        prefix_len=12,
        game_id="g1",
    ).belief_seed(0)
    assert belief_rng_seed(**kwargs, seat=0) != belief_rng_seed(
        engine_version="abc",
        map_seed=7,
        source_label="fixed_panel",
        game_id="g2",
        seat=0,
    )


def test_assign_class_priority():
    assert (
        assign_class(
            turn=10,
            both_alive=True,
            contact_occurred=True,
            sight_occurred=True,
            capture_or_defense=True,
            decisive=True,
            terminal_turn=20,
        )
        == CLASS_TACTICAL
    )
    assert (
        assign_class(
            turn=10,
            both_alive=True,
            contact_occurred=True,
            sight_occurred=True,
            capture_or_defense=False,
            decisive=True,
            terminal_turn=20,
        )
        == CLASS_AFTER_SIGHT
    )
    assert (
        assign_class(
            turn=10,
            both_alive=True,
            contact_occurred=True,
            sight_occurred=False,
            capture_or_defense=False,
            decisive=True,
            terminal_turn=20,
        )
        == CLASS_CONTACT
    )
    assert (
        assign_class(
            turn=10,
            both_alive=True,
            contact_occurred=False,
            sight_occurred=False,
            capture_or_defense=False,
            decisive=True,
            terminal_turn=20,
        )
        == CLASS_PRE_CONTACT
    )
    # Terminal horizon without adjacent threat.
    assert (
        assign_class(
            turn=15,
            both_alive=True,
            contact_occurred=True,
            sight_occurred=True,
            capture_or_defense=False,
            decisive=True,
            terminal_turn=15 + TACTICAL_HORIZON,
        )
        == CLASS_TACTICAL
    )


def test_wilson_and_promotion_gate():
    lo, hi = wilson_interval(20, 40, confidence_level=0.95)
    assert 0.0 <= lo < 0.5 < hi <= 1.0
    degenerate = class_wdl_is_non_degenerate(40, 0, 0)
    assert degenerate["ok"] is False
    healthy = class_wdl_is_non_degenerate(20, 20, 0)
    # 40 samples >= 32 and both outcomes.
    assert healthy["ok"] is True
    gate = may_advance_toward_earlier_class(
        {
            1: {"wins": 20, "losses": 20},
            2: {"wins": 40, "losses": 0},
        },
        active_classes=[1, 2],
    )
    assert gate["ok"] is False


def test_manifest_rejects_resbot_item():
    with pytest.raises(ValueError, match="banned"):
        CurriculumItem.build(
            class_id=CLASS_FULL_START,
            engine_version="e",
            map_seed=1,
            source_label="ResBot",
            prefix_len=0,
        )


def test_sampler_blocks_earlier_bias_until_gate_passes():
    items = [
        CurriculumItem.build(
            class_id=CLASS_TACTICAL,
            engine_version="e",
            map_seed=1,
            source_label="fixed_panel",
            prefix_len=10,
            game_id="g",
        ),
        CurriculumItem.build(
            class_id=CLASS_FULL_START,
            engine_version="e",
            map_seed=2,
            source_label="full_start",
            prefix_len=0,
        ),
    ]
    manifest = CurriculumManifest(
        panel_name="t",
        panel_path="p",
        engine_version="e",
        source_label="fixed_panel",
        confidence_rule=dict(DEFAULT_CONFIDENCE_RULE),
        items=items,
    )
    w_blocked = class_weights(
        active_classes=[1, 5],
        class_wdl={1: {"wins": 1, "losses": 0}, 5: {"wins": 1, "losses": 0}},
        prefer_earlier=True,
    )
    assert abs(w_blocked[1] - w_blocked[5]) < 1e-9
    w_open = class_weights(
        active_classes=[1, 5],
        class_wdl={1: {"wins": 20, "losses": 20}, 5: {"wins": 20, "losses": 20}},
        prefer_earlier=True,
    )
    assert w_open[1] > w_open[5]
    rng = np.random.default_rng(0)
    picked = sample_item(manifest, rng=rng, active_classes=[1, 5])
    assert picked.class_id in (1, 5)


def test_manifest_round_trip(tmp_path: Path):
    item = CurriculumItem.build(
        class_id=CLASS_FULL_START,
        engine_version="era",
        map_seed=99,
        source_label="full_start",
        prefix_len=0,
        sample_seat=0,
    )
    man = CurriculumManifest(
        panel_name="p",
        panel_path="panel.json",
        engine_version="era",
        source_label="fixed_panel",
        items=[item],
    )
    path = tmp_path / "curriculum.json"
    man.write(path)
    loaded = CurriculumManifest.load(path)
    assert loaded.confidence_rule["interval_method"] == "wilson"
    assert loaded.items[0].belief_seed(0) == item.belief_seed(0)
    assert loaded.items[0].class_id == CLASS_FULL_START
    assert loaded.items[0].sample_seat == 0


def test_sample_seat_optional_on_old_manifests():
    item = CurriculumItem.build(
        class_id=CLASS_FULL_START,
        engine_version="era",
        map_seed=1,
        source_label="full_start",
        prefix_len=0,
    )
    assert item.sample_seat is None
    data = item.to_dict()
    assert data["sample_seat"] is None
    round_trip = CurriculumItem.from_dict(data)
    assert round_trip.sample_seat is None


def test_reconstruction_label_accepted_raw_resbot_rejected():
    ok = CurriculumItem.build(
        class_id=CLASS_TACTICAL,
        engine_version="e",
        map_seed=1,
        source_label="ResBot_reconstructions",
        prefix_len=10,
        sample_seat=0,
        decisive=True,
        outcome="a",
    )
    assert ok.source_label == "ResBot_reconstructions"
    with pytest.raises(ValueError, match="banned"):
        CurriculumItem.build(
            class_id=CLASS_TACTICAL,
            engine_version="e",
            map_seed=1,
            source_label="ResBot",
            prefix_len=10,
        )


def test_resolve_sample_seat_from_corpus_meta():
    from training.morpheus.curriculum.build import resolve_sample_seat

    seat = resolve_sample_seat(
        game_id="g",
        traj_header={"bot_a": "ResBot", "bot_b": "Other"},
        game_meta={"queried_player": "ResBot", "sample_seat": 0},
        require_seat=True,
    )
    assert seat == 0
    seat_b = resolve_sample_seat(
        game_id="g2",
        traj_header={"bot_a": "X", "bot_b": "erik.bystron"},
        game_meta={"queried_player": "erik.bystron"},
        require_seat=True,
    )
    assert seat_b == 1


def test_provenance_panel_loads():
    from training.morpheus.corpus.panel import load_panel

    panel = load_panel(REPO / "scripts/configs/morpheus/pilot-class1-scraped.json")
    assert panel["kind"] == "provenance"
    assert panel["source_label"] == "scraped_class1_pilot"
    assert is_banned_source(panel["source_label"]) is False


def test_scraped_classes13_panel_and_config_load():
    from training.morpheus.corpus.panel import load_panel
    from training.morpheus.trainer.config import load_train_run_config

    panel = load_panel(REPO / "scripts/configs/morpheus/scraped-classes13.json")
    assert panel["kind"] == "provenance"
    assert panel["top_win_players"] == ["ResBot", "Kubic", "thor"]
    assert float(panel["top_win_fraction"]) == 0.5
    cfg = load_train_run_config(
        REPO / "training/morpheus/configs/scraped-classes13-run.json"
    )
    assert cfg.promotable_main_run is False
    assert cfg.replay.class_balance == {"1": 1.0}
    assert cfg.replay.window_size >= 65536



@pytest.mark.skipif(not TRAJ_DIR.is_dir(), reason="bootstrap trajectories absent")
def test_build_and_verify_on_bootstrap(tmp_path: Path):
    from training.morpheus.curriculum.build import build_curriculum
    from training.morpheus.curriculum.verify import verify_manifest

    out = tmp_path / "curriculum.json"
    result = build_curriculum(
        panel_path=PANEL,
        trajectories_dir=TRAJ_DIR,
        output=out,
        full_start_count=2,
        max_games=2,
        skip_verify=False,
        repo_root=REPO,
    )
    assert result["ok"], result.get("verify_failures")
    assert result["item_count"] >= 3
    assert "5" in result["class_counts"]
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["confidence_rule"]["selected_before_main_run"] is True
    assert all("resbot" not in i["source_label"].lower() for i in data["items"])

    report = verify_manifest(out, repo_root=REPO, max_items=None, check_belief=False)
    assert report["ok"], report["mismatches"]
    assert report["fog_checked"] == result["item_count"]


@pytest.mark.skipif(not TRAJ_DIR.is_dir(), reason="bootstrap trajectories absent")
def test_reconstruct_short_prefix_both_seats():
    from arena.records.trajectories import read_trajectory
    from training.morpheus.curriculum.classify import classify_trajectory_prefixes
    from training.morpheus.curriculum.reconstruct import reconstruct_prefix

    path = sorted(TRAJ_DIR.glob("*.traj.jsonl.gz"))[0]
    traj = read_trajectory(path)
    classified = classify_trajectory_prefixes(traj)
    pre = [c for c in classified if c.class_id == CLASS_PRE_CONTACT]
    assert pre, "expected a pre-contact prefix"
    row = pre[min(20, len(pre) - 1)]
    item = CurriculumItem.build(
        class_id=row.class_id,
        engine_version=traj.engine_version,
        map_seed=traj.seed,
        source_label="fixed_panel",
        prefix_len=min(row.turn, 5),
        game_id=traj.game_id,
        trajectory_relpath=str(path.relative_to(REPO)),
    )
    recon = reconstruct_prefix(
        item, traj=traj, n_particles=4, update_belief_flag=True
    )
    assert recon.turn == item.prefix_len
    for seat in (0, 1):
        seat_rec = recon.seats[seat]
        assert seat_rec.memory.H > 0
        assert len(seat_rec.action_history) == item.prefix_len
        assert seat_rec.belief_seed == item.belief_seed(seat)
        assert seat_rec.belief.n > 0


def _belief_fingerprint(belief) -> tuple:
    """Stable compare for incremental vs single-item reconstruct."""
    parts = []
    for p in belief.particles:
        armies = tuple(np.asarray(p.state.armies, dtype=np.int32).reshape(-1).tolist())
        own = tuple(np.asarray(p.state.ownership, dtype=np.uint8).reshape(-1).tolist())
        parts.append((float(p.weight), armies, own))
    return (int(belief.seat), bool(belief.collapsed), tuple(parts))


@pytest.mark.skipif(not TRAJ_DIR.is_dir(), reason="bootstrap trajectories absent")
def test_incremental_reconstruct_matches_single_and_bounds_updates(monkeypatch):
    from arena.records.trajectories import read_trajectory
    import recovery as recovery_mod
    from training.morpheus.curriculum.reconstruct import (
        reconstruct_prefix,
        reconstruct_prefixes,
    )

    path = sorted(TRAJ_DIR.glob("*.traj.jsonl.gz"))[0]
    traj = read_trajectory(path)
    t1, t2 = 2, 5
    assert len(traj.frames) >= t2
    items = [
        CurriculumItem.build(
            class_id=CLASS_PRE_CONTACT,
            engine_version=traj.engine_version,
            map_seed=traj.seed,
            source_label="fixed_panel",
            prefix_len=t,
            game_id=traj.game_id,
            trajectory_relpath=str(path.relative_to(REPO)),
            sample_seat=0,
        )
        for t in (t1, t2)
    ]

    single = reconstruct_prefix(
        items[1], traj=traj, n_particles=4, update_belief_flag=True
    )
    multi = reconstruct_prefixes(
        items, traj=traj, n_particles=4, update_belief_flag=True
    )
    assert set(multi) == {items[0].item_id, items[1].item_id}
    incr = multi[items[1].item_id]
    assert incr.turn == single.turn == t2
    for seat in (0, 1):
        assert len(incr.seats[seat].action_history) == t2
        assert _belief_fingerprint(incr.seats[seat].belief) == _belief_fingerprint(
            single.seats[seat].belief
        )

    calls = {"n": 0}
    real_update = recovery_mod.update_belief

    def counting_update(*args, **kwargs):
        calls["n"] += 1
        return real_update(*args, **kwargs)

    monkeypatch.setattr(recovery_mod, "update_belief", counting_update)
    # reconstruct imports update_belief by name; patch the reconstruct binding too.
    import training.morpheus.curriculum.reconstruct as recon_mod

    monkeypatch.setattr(recon_mod, "update_belief", counting_update)
    reconstruct_prefixes(items, traj=traj, n_particles=4, update_belief_flag=True)
    # Two seats × (max_prefix) belief updates for turns 1..max, not sum of prefixes.
    assert calls["n"] <= 2 * t2
    assert calls["n"] < 2 * (t1 + t2)


def test_capture_or_defense_on_synthetic_state():
    import sys

    bot = REPO / "bots" / "morpheus"
    if str(bot) not in sys.path:
        sys.path.insert(0, str(bot))

    from state import GameState, create_initial_state

    # 5x5 open board; move seat-0 army next to enemy general.
    grid = np.zeros((5, 5), dtype=np.int32)
    grid[0, 0] = 1
    grid[4, 4] = 2
    state = create_initial_state(grid)
    armies = np.asarray(state.armies).copy()
    ownership = np.asarray(state.ownership).copy()
    # Place seat-0 stack adjacent to general at (4,4).
    ownership[0, 4, 3] = True
    ownership[0, 0, 0] = True
    armies[4, 3] = 5
    armies[0, 0] = 1
    threat = GameState(
        armies=armies,
        ownership=ownership,
        ownership_neutral=np.asarray(state.ownership_neutral),
        generals=np.asarray(state.generals),
        castles=np.asarray(state.castles),
        mountains=np.asarray(state.mountains),
        passable=np.asarray(state.passable),
        general_positions=np.asarray(state.general_positions),
        time=int(state.time),
        winner=int(state.winner),
        pool_idx=int(state.pool_idx),
    )
    assert both_generals_alive(threat)
    assert capture_or_defense_threat(threat)
