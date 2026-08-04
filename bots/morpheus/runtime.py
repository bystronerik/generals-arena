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
    filter_step,
    initialize_belief,
)
from memory import VisibleMemory, empty_memory, update_memory
from proposal import PolicyFn, propose_enemy_actions
from recovery import recover_belief
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
DEFAULT_MAX_PROPOSAL_BATCH = 64
DEFAULT_SEARCH_DEPTH = 16

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

FORWARD_CONSUMERS = (
    "belief_proposal",
    "root",
    "enemy_prior",
    "leaf_batch",
)


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
    max_proposal_batch: int = DEFAULT_MAX_PROPOSAL_BATCH
    search_depth: int = DEFAULT_SEARCH_DEPTH
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
    forward_by_consumer: dict[str, int] = field(default_factory=dict)
    belief_ess: int = 0  # ess * 1000
    recovery: int = 0  # 0/1
    tree_size: int = 0
    fallback_level: str = FallbackLevel.PASS.value
    cost_belief_ms: int = 0
    cost_root_ms: int = 0
    cost_search_ms: int = 0
    cost_reply_ms: int = 0
    belief_plus_root_ok: int = 0  # 1 when belief update and root both finished
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
        proposal_policy: Optional[PolicyFn] = None,
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
        self.proposal_policy = proposal_policy

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
                depth=self.config.search_depth,
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
        self.forward_by_consumer: dict[str, int] = {name: 0 for name in FORWARD_CONSUMERS}
        self.belief_ess = 0
        self.recovery = 0
        self.tree_size = 0
        self.fallback_level = FallbackLevel.PASS.value
        self.move_ms = 0
        self.cost_belief_ms = 0
        self.cost_root_ms = 0
        self.cost_search_ms = 0
        self.cost_reply_ms = 0
        self.belief_plus_root_ok = 0

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
        belief_update_ok = first_move  # first move allocates; later turns track update
        self.search.tree.completed_simulations = 0
        self.forward_equivalents = 0
        self.forward_by_consumer = {name: 0 for name in FORWARD_CONSUMERS}
        self.metrics = TurnMetrics()
        recovery_flag = 0

        if first_move:
            admitted, _ = self._run_admitted(
                "belief_tensor", deadline, lambda: self.first_move_setup(obs)
            )
            if not admitted:
                self.first_move_setup(obs)  # must still allocate; first frame
            belief_update_ok = True
            deadline = self._deadline_for_turn(first_move=True)
        else:
            assert self.memory is not None
            # 2. Update visible memory and particle belief.
            self.memory = update_memory(self.memory, obs)
            if self.belief is not None and self.belief.n > 0:
                remaining_ms = (deadline - self._now()) * 1000.0
                guard = self.config.admission_guard_ms
                admit_belief = self.can_admit(
                    "belief_proposal", deadline
                ) and self.can_admit("particle_transitions", deadline)
                if not admit_belief:
                    # Warm-up lockout: one high-diversity spike makes forecast
                    # = max(sample) until the window fills, so belief never
                    # remeasures. Probe again while offline p99 still fits.
                    prop_est = self._estimators["belief_proposal"]
                    tr_est = self._estimators["particle_transitions"]
                    offline_fit = (
                        not prop_est.warmed_up
                        and remaining_ms
                        >= float(prop_est.offline_p99_ms) + guard
                        and remaining_ms
                        >= float(tr_est.offline_p99_ms) + guard
                    )
                    admit_belief = offline_fit
                if admit_belief:
                    policy = self.proposal_policy
                    if policy is not None:
                        def _counting_policy(batch, _policy=policy):
                            n = int(np.asarray(batch).shape[0])
                            self.forward_by_consumer["belief_proposal"] += n
                            self.forward_equivalents += n
                            return _policy(batch)

                        policy = _counting_policy

                    t0 = self._now()
                    self._charge("belief_proposal")
                    enemy_actions = propose_enemy_actions(
                        self.belief,
                        self.rng,
                        policy=policy,
                        max_proposal_batch=self.config.max_proposal_batch,
                    )
                    propose_ms = (self._now() - t0) * 1000.0
                    if self.charge_fixed_forecasts:
                        propose_ms = self.forecast_ms("belief_proposal")
                    self._observe("belief_proposal", propose_ms)

                    t1 = self._now()
                    self._charge("particle_transitions")
                    nxt = filter_step(
                        self.belief,
                        self._last_action,
                        obs,
                        enemy_actions,
                        self.rng,
                    )
                    if not any(p.weight > 0.0 for p in nxt.particles):
                        nxt = recover_belief(
                            self.belief,
                            self._last_action,
                            obs,
                            self.memory,
                            self.rng,
                            policy=self.proposal_policy,
                        )
                    filter_ms = (self._now() - t1) * 1000.0
                    if self.charge_fixed_forecasts:
                        filter_ms = self.forecast_ms("particle_transitions")
                    self._observe("particle_transitions", filter_ms)
                    self.belief = nxt
                    belief_update_ok = True
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
                self.forward_by_consumer["root"] += 1
                prior_full = self.search.last_root_prior
                if prior_full is None:
                    prior_full, _ = self.evaluator.evaluate(
                        obs, self.memory, belief, from_root=True
                    )
                    self.forward_equivalents += 1
                    self.forward_by_consumer["root"] += 1
                mask = legal_mask(obs, self.memory)
                self._policy_fallback = highest_prior_legal(prior_full, mask)
                action = self._policy_fallback
                # Hashing already ran inside ensure_root / reuse_or_reset.
                # Record elapsed (not a forecast) so estimators stay finite.
                t_hash = self._now()
                self._charge("hashing")
                hash_ms = (self._now() - t_hash) * 1000.0
                if self.charge_fixed_forecasts:
                    hash_ms = self.forecast_ms("hashing")
                self._observe("hashing", hash_ms)

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

        # Reply cost accounting: wall elapsed (serialization is at the caller).
        if self.can_admit("reply", deadline):
            t_reply = self._now()
            self._charge("reply")
            reply_ms = (self._now() - t_reply) * 1000.0
            if self.charge_fixed_forecasts:
                reply_ms = self.forecast_ms("reply")
            self._observe("reply", reply_ms)

        self._last_action = action
        set_prev = getattr(self.evaluator, "set_previous_action", None)
        if callable(set_prev):
            set_prev(action)
        # Finite samples for components that may not run every turn.
        if "enemy_prior_batch" not in self.metrics.component_ms:
            self._observe("enemy_prior_batch", 0.0)
        self._publish_metrics(
            action_level=level,
            recovery_flag=recovery_flag,
            belief_plus_root_ok=int(belief_update_ok and has_root_result),
        )
        return action

    def _run_search(self, belief: BeliefState, deadline: float) -> None:
        from search import EnemyPriorRequest, PendingPath

        cfg = self.config
        target = cfg.target_simulations
        while self.search.tree.completed_simulations < target:
            remaining = target - self.search.tree.completed_simulations
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
            paths: list[PendingPath] = []
            to_resume: list[EnemyPriorRequest] = []

            def _select_once(
                resume: EnemyPriorRequest | None = None,
            ) -> PendingPath | EnemyPriorRequest | None:
                if not self.can_admit("selection", deadline):
                    return None
                if self.search.tree.completed_simulations + len(paths) >= target:
                    return None
                t0 = self._now()
                self._charge("selection")
                outcome = self.search.select_path(
                    belief,
                    freeze_snapshot=True,
                    freeze_widening=freeze_widening,
                    resume=resume,
                )
                elapsed = (self._now() - t0) * 1000.0
                selection_ms = (
                    self.forecast_ms("selection")
                    if self.charge_fixed_forecasts
                    else elapsed
                )
                self._observe("selection", selection_ms)
                return outcome

            def _materialize(reqs: list[EnemyPriorRequest]) -> bool:
                if not reqs:
                    return True
                if not self.can_admit("enemy_prior_batch", deadline):
                    return False
                if self.forward_equivalents >= cfg.max_forward_equivalents:
                    return False
                t0 = self._now()
                self._charge("enemy_prior_batch")
                prior_n = self.search.materialize_enemy_priors(reqs, belief)
                prior_ms = (self._now() - t0) * 1000.0
                if self.charge_fixed_forecasts:
                    prior_ms = (
                        self.forecast_ms("enemy_prior_batch") if prior_n > 0 else 0.0
                    )
                self._observe("enemy_prior_batch", prior_ms)
                if prior_n > 0:
                    self.forward_equivalents += prior_n
                    self.forward_by_consumer["enemy_prior"] += prior_n
                return True

            while len(paths) < batch_n:
                resume = to_resume.pop(0) if to_resume else None
                outcome = _select_once(resume=resume)
                if outcome is None:
                    break

                if isinstance(outcome, EnemyPriorRequest):
                    missing = [outcome]
                    # Collect more missing priors before one batched forward.
                    while (
                        len(paths) + len(to_resume) + len(missing) < batch_n
                    ):
                        nxt = _select_once()
                        if nxt is None:
                            break
                        if isinstance(nxt, EnemyPriorRequest):
                            missing.append(nxt)
                        else:
                            paths.append(nxt)
                            break
                    if not _materialize(missing):
                        # Discard partial selections — no statistics change.
                        for req in missing:
                            req.node.pending_pins.discard(req.info_hash)
                        for req in to_resume:
                            req.node.pending_pins.discard(req.info_hash)
                        to_resume.clear()
                        break
                    to_resume.extend(missing)
                    continue

                paths.append(outcome)

            # Finish any remaining resumed paths that fit the batch.
            while to_resume and len(paths) < batch_n:
                outcome = _select_once(resume=to_resume.pop(0))
                if outcome is None:
                    break
                if isinstance(outcome, EnemyPriorRequest):
                    if not _materialize([outcome]):
                        outcome.node.pending_pins.discard(outcome.info_hash)
                        for req in to_resume:
                            req.node.pending_pins.discard(req.info_hash)
                        to_resume.clear()
                        break
                    to_resume.insert(0, outcome)
                    continue
                paths.append(outcome)

            # Drop unfinished prior requests; keep pins only for paths we keep.
            for req in to_resume:
                req.node.pending_pins.discard(req.info_hash)
            to_resume.clear()

            if not paths:
                break

            # Inference is not cancelable: check once before the whole batch.
            if not self.can_admit("leaf_batch", deadline):
                # Discard selected paths — no statistics change.
                break

            t0 = self._now()
            self._charge("leaf_batch")
            values = self.search.evaluate_leaves(paths, belief)
            elapsed = (self._now() - t0) * 1000.0
            if self.charge_fixed_forecasts:
                elapsed = self.forecast_ms("leaf_batch")
            self._observe("leaf_batch", elapsed)
            self.forward_equivalents += len(paths)
            self.forward_by_consumer["leaf_batch"] += len(paths)

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
        belief_plus_root_ok: int = 0,
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
        self.belief_plus_root_ok = int(belief_plus_root_ok)
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
        self.metrics.forward_by_consumer = dict(self.forward_by_consumer)
        self.metrics.belief_ess = self.belief_ess
        self.metrics.recovery = self.recovery
        self.metrics.tree_size = self.tree_size
        self.metrics.fallback_level = self.fallback_level
        self.metrics.cost_belief_ms = self.cost_belief_ms
        self.metrics.cost_root_ms = self.cost_root_ms
        self.metrics.cost_search_ms = self.cost_search_ms
        self.metrics.cost_reply_ms = self.cost_reply_ms
        self.metrics.belief_plus_root_ok = self.belief_plus_root_ok


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
