"""In-process seat players for self-play: Morpheus stack or fixed panel bot."""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np

from training.morpheus.self_play.sampler import SeatSpec
from training.morpheus.self_play.schema import SparsePolicy

REPO = Path(__file__).resolve().parents[3]
MORPHEUS_BOT = REPO / "bots" / "morpheus"


def _ensure_morpheus_path() -> None:
    for entry in (REPO, REPO / "bots", MORPHEUS_BOT):
        s = str(entry)
        if s not in sys.path:
            sys.path.insert(0, s)


Action5 = tuple[int, int, int, int, int]


class SeatPlayer(Protocol):
    def act(self, obs) -> tuple[Action5, SparsePolicy | None]: ...

    @property
    def uses_full_stack(self) -> bool: ...


def _array_digest(array) -> str:
    buf = np.ascontiguousarray(np.asarray(array))
    sha = hashlib.sha256()
    sha.update(str(buf.dtype).encode("ascii"))
    sha.update(str(buf.shape).encode("ascii"))
    sha.update(buf.tobytes())
    return f"sha256:{sha.hexdigest()}"


def truth_from_engine_state(state) -> dict[str, Any]:
    """Digestable engine-truth fields for shard targets."""
    from arena.records.trajectories import state_digest

    return {
        "ownership_digest": _array_digest(state.ownership),
        "armies_digest": _array_digest(state.armies),
        "generals_digest": _array_digest(state.generals),
        "castles_digest": _array_digest(state.castles),
        "state_digest": state_digest(state),
    }


def extract_root_policy(controller) -> SparsePolicy | None:
    """Normalized root average strategy S_A over candidate logit indices."""
    _ensure_morpheus_path()
    from matrix import normalize_average_strategy

    root = controller.search.tree.root
    if root is None or not root.actions:
        return None
    avg = np.asarray(root.avg_strategy, dtype=np.float64).reshape(-1)
    if avg.size != len(root.actions):
        # No completed backup yet — fall back to normalized prior.
        prior = np.asarray(root.prior, dtype=np.float64).reshape(-1)
        if prior.size != len(root.actions):
            return None
        probs = normalize_average_strategy(prior)
    else:
        probs = normalize_average_strategy(avg)
    return SparsePolicy(
        indices=tuple(int(x) for x in root.actions),
        probs=tuple(float(x) for x in probs),
    )


def training_runtime_config(
    *,
    n_particles: int = 4,
    target_simulations: int = 8,
    min_simulations: int = 4,
    search_depth: int = 4,
    pending_leaf_batch: int = 2,
    deadline_ms: float = 60_000.0,
):
    """Generous deadlines so training seats complete the configured search."""
    _ensure_morpheus_path()
    from runtime import DEFAULT_OFFLINE_P99_MS, RuntimeConfig

    offline = {name: 0.1 for name in DEFAULT_OFFLINE_P99_MS}
    return RuntimeConfig(
        normal_deadline_ms=float(deadline_ms),
        reserve_ms=1.0,
        first_move_limit_ms=float(deadline_ms),
        target_simulations=int(target_simulations),
        min_simulations=int(min_simulations),
        pending_leaf_batch=int(pending_leaf_batch),
        max_forward_equivalents=10_000,
        admission_guard_ms=0.0,
        max_tree_nodes=512,
        n_particles=int(n_particles),
        max_proposal_batch=16,
        search_depth=int(search_depth),
        offline_p99_ms=offline,
        p99_window=8,
        widen_freeze_below=0,
    )


@dataclass
class MorpheusSeatPlayer:
    """Full belief + simultaneous-search stack for one seat."""

    controller: Any
    seat: int

    @property
    def uses_full_stack(self) -> bool:
        return True

    def act(self, obs) -> tuple[Action5, SparsePolicy | None]:
        action = self.controller.decide(obs)
        policy = extract_root_policy(self.controller)
        return (
            (
                int(action[0]),
                int(action[1]),
                int(action[2]),
                int(action[3]),
                int(action[4]),
            ),
            policy,
        )


@dataclass
class PanelSeatPlayer:
    """Fixed existing-bot panel seat (no Morpheus search stack)."""

    agent: Any
    seat: int
    bot_id: str

    @property
    def uses_full_stack(self) -> bool:
        return False

    def act(self, obs) -> tuple[Action5, SparsePolicy | None]:
        action = self.agent.act(obs)
        return (
            (
                int(action[0]),
                int(action[1]),
                int(action[2]),
                int(action[3]),
                int(action[4]),
            ),
            None,
        )


def _build_stub_evaluator():
    _ensure_morpheus_path()
    from search import UniformEvaluator

    return UniformEvaluator(0.0)


def _build_checkpoint_evaluator(path: Path):
    _ensure_morpheus_path()
    from evaluator import NetworkEvaluator
    from export import export_from_checkpoint
    from inference import load_session

    path = Path(path)
    # Accept either an export artifact dir or a training checkpoint dir.
    manifest = path / "manifest.json"
    if manifest.is_file():
        session = load_session(path)
    else:
        # Tiny / training checkpoint → export into a sibling artifact cache.
        artifact = path / "_self_play_artifact"
        if not (artifact / "manifest.json").is_file():
            export_from_checkpoint(path, artifact)
        session = load_session(artifact)
    return NetworkEvaluator(session)


def build_seat_player(
    spec: SeatSpec,
    *,
    H: int,
    W: int,
    rng_seed: int,
    runtime_kwargs: Optional[dict[str, Any]] = None,
) -> SeatPlayer:
    """Construct an in-process seat from a Matchup SeatSpec."""
    _ensure_morpheus_path()
    from runtime import RuntimeController

    if spec.kind in ("stub", "checkpoint"):
        if spec.kind == "stub" or not spec.path:
            evaluator = _build_stub_evaluator()
        else:
            evaluator = _build_checkpoint_evaluator(Path(spec.path))
        cfg = training_runtime_config(**(runtime_kwargs or {}))
        ctl = RuntimeController(
            seat=int(spec.seat),
            H=int(H),
            W=int(W),
            evaluator=evaluator,
            config=cfg,
            rng=np.random.default_rng(int(rng_seed)),
        )
        return MorpheusSeatPlayer(controller=ctl, seat=int(spec.seat))

    if spec.kind == "panel":
        if not spec.bot_id:
            raise ValueError("panel seat requires bot_id")
        from arena.bot_api import load_strategy_class

        agent_cls = load_strategy_class(spec.bot_id)
        agent = agent_cls(player_id=int(spec.seat), H=int(H), W=int(W))
        return PanelSeatPlayer(agent=agent, seat=int(spec.seat), bot_id=spec.bot_id)

    raise ValueError(f"unknown seat kind {spec.kind!r}")
