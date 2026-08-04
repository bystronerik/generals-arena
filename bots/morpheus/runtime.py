"""Runtime controller: deadlines, admission, degradation, and turn order.

Part 07. Part 09 replaces coupled timing fields after measurement. Tests inject
a monotonic clock and fixed cost forecasts for every degradation level.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from action import decode_action, legal_mask
from belief import (
    BeliefConfig,
    BeliefState,
    Action5,
    ess,
    initialize_belief,
)
from memory import VisibleMemory, empty_memory, update_memory
from recovery import update_belief
from search import SearchConfig, SearchController, SearchEvaluator, UniformEvaluator

ClockFn = Callable[[], float]
Array = np.ndarray

# Named cost components. Admission uses a separate rolling p99 for each.
COST_COMPONENTS = (
    "belief_tensor",
    "belief_proposal",
    "particle_transitions",
    "hashing",
    "root_inference",
    "leaf_batch",
    "enemy_prior_batch",
    "backup",
    "reply",
    "selection",
)

# Initial guesses — Part 09 replaces the coupled set.
DEFAULT_NORMAL_DEADLINE_MS = 125.0
DEFAULT_RESERVE_MS = 25.0
DEFAULT_FIRST_MOVE_LIMIT_MS = 8500.0
DEFAULT_TARGET_SIMULATIONS = 32
DEFAULT_MIN_SIMULATIONS = 8
DEFAULT_PENDING_LEAF_BATCH = 4
DEFAULT_MAX_FORWARD_EQUIVALENTS = 113
DEFAULT_ADMISSION_GUARD_MS = 10.0
DEFAULT_MAX_TREE_NODES = 4096
DEFAULT_RESIDENT_MEMORY_TARGET_MB = 256.0
DEFAULT_P99_WINDOW = 64

# Offline qualification p99 placeholders (ms) until Part 09 measures them.
DEFAULT_OFFLINE_P99_MS: dict[str, float] = {
    "belief_tensor": 5.0,
    "belief_proposal": 40.0,
    "particle_transitions": 20.0,
    "hashing": 1.0,
    "root_inference": 8.0,
    "leaf_batch": 30.0,
    "enemy_prior_batch": 8.0,
    "backup": 2.0,
    "reply": 1.0,
    "selection": 2.0,
}

PASS: Action5 = (1, 0, 0, 0, 0)


class FallbackLevel(str, Enum):
    """Degradation band that produced the committed action."""

    PASS = "pass"  # no root result
    POLICY = "policy"  # 0 completed simulations
    VISIT = "visit"  # 1–7 completed simulations
    AVERAGE = "average"  # 8+ completed simulations


def nearest_rank_p99(samples: Sequence[float]) -> float:
    """Nearest-rank empirical 99th percentile (1-indexed rank ``ceil(0.99 n)``)."""
    n = len(samples)
    if n <= 0:
        raise ValueError("nearest-rank p99 needs at least one sample")
    ordered = sorted(float(x) for x in samples)
    rank = int(math.ceil(0.99 * n))
    return ordered[rank - 1]


@dataclass
class RuntimeConfig:
    """Explicit runtime fields. Part 09 overwrites guesses after measurement."""

    normal_deadline_ms: float = DEFAULT_NORMAL_DEADLINE_MS
    reserve_ms: float = DEFAULT_RESERVE_MS
    first_move_limit_ms: float = DEFAULT_FIRST_MOVE_LIMIT_MS
    target_simulations: int = DEFAULT_TARGET_SIMULATIONS
    min_simulations: int = DEFAULT_MIN_SIMULATIONS
    pending_leaf_batch: int = DEFAULT_PENDING_LEAF_BATCH
    max_forward_equivalents: int = DEFAULT_MAX_FORWARD_EQUIVALENTS
    admission_guard_ms: float = DEFAULT_ADMISSION_GUARD_MS
    max_tree_nodes: int = DEFAULT_MAX_TREE_NODES
    resident_memory_target_mb: float = DEFAULT_RESIDENT_MEMORY_TARGET_MB
    p99_window: int = DEFAULT_P99_WINDOW
    offline_p99_ms: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_OFFLINE_P99_MS)
    )
    n_particles: int = 64
    widen_freeze_below: int = 16  # stop widening when forecast sims < this


@dataclass
class NearestRankP99Estimator:
    """Rolling nearest-rank p99 with deterministic warm-up.

    Before ``W`` local samples exist, the forecast is the maximum of the offline
    qualification p99 and every observed local sample. After ``W`` samples, the
    forecast is the nearest-rank empirical p99 of the bounded window.
    """

    window: int
    offline_p99_ms: float
    _samples: deque[float] = field(default_factory=deque)

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError("p99 window must be >= 1")
        self._samples = deque(maxlen=int(self.window))

    @property
    def n_samples(self) -> int:
        return len(self._samples)

    @property
    def warmed_up(self) -> bool:
        return self.n_samples >= self.window

    def observe(self, ms: float) -> None:
        self._samples.append(float(ms))

    def forecast(self) -> float:
        if self.n_samples == 0:
            return float(self.offline_p99_ms)
        if not self.warmed_up:
            return float(max(self.offline_p99_ms, max(self._samples)))
        return nearest_rank_p99(self._samples)


@dataclass
class TurnMetrics:
    """Passive per-turn counters. Probe reads these; they never change play."""

    move_ms: int = 0
    completed_simulations: int = 0
    forward_equivalents: int = 0
    belief_ess: int = 0  # ess * 1000
    recovery: int = 0  # 0/1
    tree_size: int = 0
    fallback_level: str = FallbackLevel.PASS.value
    cost_belief_ms: int = 0
    cost_root_ms: int = 0
    cost_search_ms: int = 0
    cost_reply_ms: int = 0
    component_ms: dict[str, float] = field(default_factory=dict)


class RuntimeController:
    """Owns first-move setup, normal-turn work order, and admission control."""

    def __init__(
        self,
        seat: int,
        H: int,
        W: int,
        *,
        evaluator: Optional[SearchEvaluator] = None,
        config: Optional[RuntimeConfig] = None,
        clock: Optional[ClockFn] = None,
        fixed_forecasts_ms: Optional[Mapping[str, float]] = None,
        rng: Optional[np.random.Generator] = None,
        charge_fixed_forecasts: bool = False,
    ) -> None:
        self.seat = int(seat)
        self.H = int(H)
        self.W = int(W)
        self.config = config or RuntimeConfig()
        self.clock: ClockFn = clock or time.monotonic
        self.rng = rng or np.random.default_rng(0)
        self.fixed_forecasts_ms = (
            dict(fixed_forecasts_ms) if fixed_forecasts_ms is not None else None
        )
        # When True, each admitted component advances the injected clock by its
        # forecast. Production uses real elapsed time instead.
        self.charge_fixed_forecasts = bool(charge_fixed_forecasts)

        self._estimators: dict[str, NearestRankP99Estimator] = {
            name: NearestRankP99Estimator(
                window=self.config.p99_window,
                offline_p99_ms=float(
                    self.config.offline_p99_ms.get(
                        name, DEFAULT_OFFLINE_P99_MS.get(name, 1.0)
                    )
                ),
            )
            for name in COST_COMPONENTS
        }

        self.evaluator: SearchEvaluator = evaluator or UniformEvaluator(0.0)
        self.search = SearchController(
            seat=self.seat,
            evaluator=self.evaluator,
            config=SearchConfig(
                pending_batch=self.config.pending_leaf_batch,
                max_nodes=self.config.max_tree_nodes,
                n_particles=self.config.n_particles,
            ),
            rng=self.rng,
        )

        self.memory: Optional[VisibleMemory] = None
        self.belief: Optional[BeliefState] = None
        self._setup_done = False
        self._last_action: Action5 = PASS
        self._pending_recovery = False
        self.metrics = TurnMetrics()

        # Per-turn scratch (also mirrored onto Agent for probe.py).
        self.completed_simulations = 0
        self.forward_equivalents = 0
        self.belief_ess = 0
        self.recovery = 0
        self.tree_size = 0
        self.fallback_level = FallbackLevel.PASS.value
        self.move_ms = 0
        self.cost_belief_ms = 0
        self.cost_root_ms = 0
        self.cost_search_ms = 0
        self.cost_reply_ms = 0

    # ------------------------------------------------------------------ clock

    def _now(self) -> float:
        return float(self.clock())

    def _deadline_for_turn(self, *, first_move: bool) -> float:
        budget_ms = (
            self.config.first_move_limit_ms
            if first_move
            else self.config.normal_deadline_ms
        )
        return self._turn_start + budget_ms / 1000.0

    def forecast_ms(self, component: str) -> float:
        if self.fixed_forecasts_ms is not None and component in self.fixed_forecasts_ms:
            return float(self.fixed_forecasts_ms[component])
        return self._estimators[component].forecast()

    def can_admit(self, component: str, deadline: float) -> bool:
        """True when forecast p99 + guard fits before the internal deadline."""
        remaining_ms = (deadline - self._now()) * 1000.0
        need = self.forecast_ms(component) + self.config.admission_guard_ms
        return remaining_ms >= need

    def _observe(self, component: str, ms: float) -> None:
        if component in self._estimators:
            self._estimators[component].observe(ms)
        self.metrics.component_ms[component] = (
            self.metrics.component_ms.get(component, 0.0) + float(ms)
        )

    def _charge(self, component: str) -> None:
        """Advance an injected clock by the component forecast (test mode)."""
        if not self.charge_fixed_forecasts:
            return
        advance = self.forecast_ms(component) / 1000.0
        # Fake clocks expose ``advance(ms)``; real monotonic does not.
        adv = getattr(self.clock, "advance", None)
        if callable(adv):
            adv(self.forecast_ms(component))
        else:
            # No-op for production clocks; elapsed comes from wall time.
            pass

    def _run_admitted(
        self,
        component: str,
        deadline: float,
        fn: Callable[[], object],
    ) -> tuple[bool, object]:
        """Admit, run, record. Returns ``(admitted, result_or_None)``."""
        if not self.can_admit(component, deadline):
            return False, None
        t0 = self._now()
        self._charge(component)
        result = fn()
        elapsed_ms = (self._now() - t0) * 1000.0
        if self.charge_fixed_forecasts:
            elapsed_ms = self.forecast_ms(component)
        self._observe(component, elapsed_ms)
        return True, result

    # ----------------------------------------------------------- first move

    def first_move_setup(self, obs) -> None:
        """Load-time work that fits the first-frame grace window.

        Network load and warm belong to Agent.__init__. This allocates memory,
        belief, and bounded tree storage and classifies initial terrain via
        the memory update.
        """
        if self._setup_done:
            return
        self.memory = update_memory(empty_memory(self.H, self.W), obs)
        belief_cfg = BeliefConfig(n_particles=self.config.n_particles)
        try:
            self.belief = initialize_belief(
                obs, seat=self.seat, rng=self.rng, config=belief_cfg
            )
        except ValueError:
            # Tiny boards in protocol tests cannot place an enemy general.
            self.belief = BeliefState(
                seat=self.seat, particles=[], config=belief_cfg, collapsed=True
            )
        # Allocate bounded tree storage (empty until ensure_root).
        self.search.tree.clear()
        self._setup_done = True

    # -------------------------------------------------------------- decide

    def decide(self, obs) -> Action5:
        """Run one turn. Always stores pass before optional work."""
        self._turn_start = self._now()
        first_move = not self._setup_done
        deadline = self._deadline_for_turn(first_move=first_move)

        # 1. Protocol-safe fallback.
        action: Action5 = PASS
        self._policy_fallback: Optional[Action5] = None
        has_root_result = False
        self.search.tree.completed_simulations = 0
        self.forward_equivalents = 0
        self.metrics = TurnMetrics()
        recovery_flag = 0

        if first_move:
            admitted, _ = self._run_admitted(
                "belief_tensor", deadline, lambda: self.first_move_setup(obs)
            )
            if not admitted:
                self.first_move_setup(obs)  # must still allocate; first frame
            deadline = self._deadline_for_turn(first_move=True)
        else:
            assert self.memory is not None
            # 2. Update visible memory and particle belief.
            self.memory = update_memory(self.memory, obs)
            if self.belief is not None and self.belief.n > 0:
                if self.can_admit("belief_proposal", deadline) and self.can_admit(
                    "particle_transitions", deadline
                ):
                    prev = self.belief

                    def _upd():
                        return update_belief(
                            self.belief,
                            self._last_action,
                            obs,
                            self.memory,
                            self.rng,
                        )

                    admitted, nxt = self._run_admitted(
                        "belief_proposal", deadline, _upd
                    )
                    if admitted and nxt is not None:
                        self.belief = nxt  # type: ignore[assignment]
                        self._observe(
                            "particle_transitions",
                            self.forecast_ms("particle_transitions")
                            if self.charge_fixed_forecasts
                            else 0.0,
                        )
                        if self.charge_fixed_forecasts:
                            self._charge("particle_transitions")
                    else:
                        recovery_flag = 1
                        self._pending_recovery = True
                else:
                    # Keep last valid set; lower reported ESS; defer recovery.
                    recovery_flag = 1
                    self._pending_recovery = True
            elif self._pending_recovery and self.can_admit(
                "belief_proposal", deadline
            ):
                # Full recovery when admission permits (next turn path).
                self._pending_recovery = False

        assert self.memory is not None
        belief = self.belief or BeliefState(
            seat=self.seat,
            particles=[],
            config=BeliefConfig(n_particles=self.config.n_particles),
            collapsed=True,
        )

        # 3–5. Legal masks + root network evaluation + policy fallback.
        if self.can_admit("root_inference", deadline):
            def _root():
                if first_move or self.search.tree.root is None:
                    return self.search.ensure_root(obs, self.memory, belief)
                return self.search.reuse_or_reset(
                    self._last_action, obs, self.memory, belief
                )

            admitted, root = self._run_admitted(
                "root_inference", deadline, _root
            )
            if admitted and root is not None:
                has_root_result = True
                self.forward_equivalents += 1
                prior_full = self.search.last_root_prior
                if prior_full is None:
                    prior_full, _ = self.evaluator.evaluate(
                        obs, self.memory, belief, from_root=True
                    )
                    self.forward_equivalents += 1
                mask = legal_mask(obs, self.memory)
                self._policy_fallback = highest_prior_legal(prior_full, mask)
                action = self._policy_fallback
                self._observe("hashing", self.forecast_ms("hashing"))
                if self.charge_fixed_forecasts:
                    self._charge("hashing")

        # 6–7. Search in batches of up to pending_leaf_batch.
        search_t0 = self._now()
        if has_root_result and belief.n > 0:
            self._run_search(belief, deadline)
        self.cost_search_ms = int(round((self._now() - search_t0) * 1000.0))
        if self.charge_fixed_forecasts:
            self.cost_search_ms = int(
                round(self.metrics.component_ms.get("leaf_batch", 0.0))
            )

        # 8. Serialize best available action (degradation path).
        action, level = select_degraded_action(
            completed_simulations=self.search.tree.completed_simulations,
            has_root_result=has_root_result,
            policy_fallback=self._policy_fallback,
            search=self.search,
        )

        # Reply cost accounting (serialization is cheap; still recorded).
        if self.can_admit("reply", deadline):
            self._observe("reply", self.forecast_ms("reply"))
            if self.charge_fixed_forecasts:
                self._charge("reply")

        self._last_action = action
        self._publish_metrics(
            action_level=level,
            recovery_flag=recovery_flag,
        )
        return action

    def _run_search(self, belief: BeliefState, deadline: float) -> None:
        cfg = self.config
        target = cfg.target_simulations
        while self.search.tree.completed_simulations < target:
            remaining = target - self.search.tree.completed_simulations
            # Forecast how many sims we can still finish.
            per_sim = max(
                self.forecast_ms("selection")
                + self.forecast_ms("leaf_batch") / max(cfg.pending_leaf_batch, 1)
                + self.forecast_ms("backup"),
                1e-6,
            )
            remaining_ms = (deadline - self._now()) * 1000.0 - cfg.admission_guard_ms
            forecast_total = self.search.tree.completed_simulations + int(
                remaining_ms // per_sim
            )
            freeze_widening = forecast_total < cfg.widen_freeze_below

            if not self.can_admit("selection", deadline):
                break
            if not self.can_admit("leaf_batch", deadline):
                break
            if self.forward_equivalents >= cfg.max_forward_equivalents:
                break

            batch_n = min(cfg.pending_leaf_batch, remaining)
            paths = []
            for _ in range(batch_n):
                if not self.can_admit("selection", deadline):
                    break
                if self.search.tree.completed_simulations + len(paths) >= target:
                    break
                t0 = self._now()
                self._charge("selection")
                path = self.search.select_path(
                    belief, freeze_snapshot=True, freeze_widening=freeze_widening
                )
                elapsed = (self._now() - t0) * 1000.0
                if self.charge_fixed_forecasts:
                    elapsed = self.forecast_ms("selection")
                self._observe("selection", elapsed)
                paths.append(path)

            if not paths:
                break

            # Inference is not cancelable: check once before the whole batch.
            if not self.can_admit("leaf_batch", deadline):
                # Discard selected paths — no statistics change.
                break

            t0 = self._now()
            self._charge("leaf_batch")
            values = [self.search.evaluate_leaf(p, belief) for p in paths]
            elapsed = (self._now() - t0) * 1000.0
            if self.charge_fixed_forecasts:
                elapsed = self.forecast_ms("leaf_batch")
            self._observe("leaf_batch", elapsed)
            self.forward_equivalents += len(paths)

            for path, value in zip(paths, values):
                if not self.can_admit("backup", deadline):
                    # Remaining paths in the batch are discarded.
                    break
                t0 = self._now()
                self._charge("backup")
                self.search.backup_path(path, value)
                elapsed = (self._now() - t0) * 1000.0
                if self.charge_fixed_forecasts:
                    elapsed = self.forecast_ms("backup")
                self._observe("backup", elapsed)

    def _publish_metrics(
        self,
        *,
        action_level: FallbackLevel,
        recovery_flag: int,
    ) -> None:
        move_ms = int(round((self._now() - self._turn_start) * 1000.0))
        ess_milli = 0
        if self.belief is not None and self.belief.n > 0:
            weights = [p.weight for p in self.belief.particles]
            ess_milli = int(round(ess(weights) * 1000.0))
            if recovery_flag:
                ess_milli = min(ess_milli, int(round(0.25 * self.config.n_particles * 1000)))

        self.completed_simulations = int(self.search.tree.completed_simulations)
        self.tree_size = int(len(self.search.tree.nodes))
        self.fallback_level = action_level.value
        self.recovery = int(recovery_flag)
        self.belief_ess = ess_milli
        self.move_ms = move_ms
        self.cost_belief_ms = int(
            round(
                self.metrics.component_ms.get("belief_proposal", 0.0)
                + self.metrics.component_ms.get("belief_tensor", 0.0)
                + self.metrics.component_ms.get("particle_transitions", 0.0)
            )
        )
        self.cost_root_ms = int(
            round(self.metrics.component_ms.get("root_inference", 0.0))
        )
        self.cost_reply_ms = int(round(self.metrics.component_ms.get("reply", 0.0)))

        self.metrics.move_ms = self.move_ms
        self.metrics.completed_simulations = self.completed_simulations
        self.metrics.forward_equivalents = self.forward_equivalents
        self.metrics.belief_ess = self.belief_ess
        self.metrics.recovery = self.recovery
        self.metrics.tree_size = self.tree_size
        self.metrics.fallback_level = self.fallback_level
        self.metrics.cost_belief_ms = self.cost_belief_ms
        self.metrics.cost_root_ms = self.cost_root_ms
        self.metrics.cost_search_ms = self.cost_search_ms
        self.metrics.cost_reply_ms = self.cost_reply_ms


def highest_prior_legal(prior: Array, mask: Array) -> Action5:
    """Highest-prior legal action; pass wins ties when priors are equal."""
    prior = np.asarray(prior, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    scores = np.where(mask, prior, -1.0)
    idx = int(np.argmax(scores))
    return tuple(int(x) for x in decode_action(idx))  # type: ignore[return-value]


def select_degraded_action(
    *,
    completed_simulations: int,
    has_root_result: bool,
    policy_fallback: Optional[Action5],
    search: SearchController,
) -> tuple[Action5, FallbackLevel]:
    """Deterministic degradation path from runtime.md.

    ``has_root_result`` is False when root inference never completed (distinct
    from ``completed_simulations == 0``, which still has a policy fallback).
    """
    if not has_root_result:
        return PASS, FallbackLevel.PASS
    if completed_simulations <= 0:
        return (policy_fallback or PASS), FallbackLevel.POLICY
    if completed_simulations < 8:
        return search.best_action_by_visits(), FallbackLevel.VISIT
    return search.best_action(), FallbackLevel.AVERAGE


class FakeClock:
    """Injectable monotonic clock for deadline tests (seconds)."""

    def __init__(self, start: float = 0.0) -> None:
        self.t = float(start)

    def __call__(self) -> float:
        return self.t

    def advance(self, ms: float) -> None:
        self.t += float(ms) / 1000.0
