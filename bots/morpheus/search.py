"""Root-sampled information-set MCTS with simultaneous matrices.

Uses Part 02 ``transition`` only. Network priors and values are injectable so
tests do not require a trained checkpoint. Part 07 wires the runtime deadline.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence

import numpy as np

from action import PASS_INDEX, decode_action, legal_mask
from belief import BeliefState, Particle, as_action5, pass_action
from hashing import (
    ZERO_DIGEST,
    child_edge_key,
    enemy_info_hash_prehashed,
    info_state_key_prehashed,
    memory_digest,
    roll_history_digest,
)
from matrix import (
    enemy_widening_limit,
    mixed_strategy,
    sample_index,
    self_widening_limit,
)
from memory import VisibleMemory, update_memory
from observe import emit_observation, observation_hash
from tactics import mandatory_action_indices, policy_ordered_candidates
from transition import PASS_ACTION, transition
from tree import InfoNode, SearchTree

Array = np.ndarray
Action5 = tuple[int, int, int, int, int]

SEARCH_DEPTH = 16
PENDING_LEAF_BATCH = 4


class SearchEvaluator(Protocol):
    """Injectable policy/value interface for search."""

    def evaluate(
        self,
        obs,
        memory: VisibleMemory,
        belief: BeliefState,
        *,
        from_root: bool,
        shape: bool = False,
    ) -> tuple[Array, float]:
        """Return ``(prior_3970, V)`` from the requested perspective.

        ``from_root`` is the *perspective* flag: ``V`` must already be in the
        root-player perspective (no sign flip inside backup). ``shape`` is the
        *root-prior shaping* flag: only the actual root evaluation passes True.
        Leaf evaluations are root-perspective but unshaped — conflating the two
        made every leaf pay the heuristic blend and clobbered the
        ``last_unshaped_prior`` probe.
        """
        ...


@dataclass
class SearchConfig:
    depth: int = SEARCH_DEPTH
    pending_batch: int = PENDING_LEAF_BATCH
    max_nodes: int = 4096
    max_enemy_tables: int = 8
    n_particles: int = 64
    # Cross-turn LRU of enemy priors keyed by enemy info hash. The tree is
    # rebuilt on most turns (obs-hash reuse rarely matches), so without this
    # every turn re-runs the same enemy-prior forwards (~36 ms/turn measured).
    # 512 float32 vectors ≈ 8 MB.
    enemy_prior_cache_size: int = 512


@dataclass
class PendingPath:
    """One selected simulation path awaiting leaf evaluation and backup."""

    nodes: list[InfoNode]
    edges: list[tuple[bytes, int, int, bytes]]  # (h, a_idx, b_idx, edge_key)
    particle: Particle
    leaf_state: object  # GameState
    leaf_node: Optional[InfoNode]
    needs_expand: bool
    terminal_value: Optional[float]


@dataclass
class EnemyPriorRequest:
    """Selection paused until an enemy-table prior is materialised."""

    nodes: list[InfoNode]
    edges: list[tuple[bytes, int, int, bytes]]
    particle: Particle
    state: object  # GameState
    depth: int
    freeze_widening: bool
    info_hash: bytes
    enemy_obs: object
    enemy_mem: VisibleMemory
    my_obs: object
    my_mem: VisibleMemory

    @property
    def node(self) -> InfoNode:
        return self.nodes[-1]


@dataclass
class SearchController:
    """Owns the tree and runs batched simulations."""

    seat: int
    evaluator: SearchEvaluator
    config: SearchConfig = field(default_factory=SearchConfig)
    tree: SearchTree = field(init=False)
    history_digest: bytes = ZERO_DIGEST
    memory: Optional[VisibleMemory] = None
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0)
    )
    last_root_prior: Optional[Array] = field(default=None, init=False)
    # Enemy-prior network time from materialize_enemy_priors (never select_path).
    last_select_enemy_prior_ms: float = field(default=0.0, init=False)
    last_select_enemy_prior_forwards: int = field(default=0, init=False)
    # Cross-turn enemy-prior LRU: enemy info hash -> full-length float32 prior.
    # Content-addressed (hash covers enemy obs + memory), so entries never go
    # stale — they only cost memory. Survives tree.clear() by design.
    enemy_prior_cache: "OrderedDict[bytes, Array]" = field(init=False)
    enemy_prior_cache_hits: int = field(default=0, init=False)
    enemy_prior_cache_misses: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.tree = SearchTree(
            max_nodes=self.config.max_nodes,
            max_enemy_tables=self.config.max_enemy_tables,
            seat=self.seat,
        )
        self.last_root_prior = None
        self.last_select_enemy_prior_ms = 0.0
        self.last_select_enemy_prior_forwards = 0
        self.enemy_prior_cache = OrderedDict()
        self.enemy_prior_cache_hits = 0
        self.enemy_prior_cache_misses = 0

    def _cached_enemy_prior(self, info_hash: bytes) -> Optional[Array]:
        prior = self.enemy_prior_cache.get(info_hash)
        if prior is None:
            return None
        self.enemy_prior_cache.move_to_end(info_hash)
        self.enemy_prior_cache_hits += 1
        return np.asarray(prior, dtype=np.float64)

    def _store_enemy_prior(self, info_hash: bytes, prior: Array) -> None:
        self.enemy_prior_cache[info_hash] = np.asarray(prior, dtype=np.float32)
        self.enemy_prior_cache.move_to_end(info_hash)
        while len(self.enemy_prior_cache) > self.config.enemy_prior_cache_size:
            self.enemy_prior_cache.popitem(last=False)

    def ensure_root(
        self,
        obs,
        memory: VisibleMemory,
        belief: BeliefState,
    ) -> InfoNode:
        self.memory = memory
        obs_h = observation_hash(obs)
        mem_d = memory_digest(memory)
        key = info_state_key_prehashed(
            int(obs.turn), mem_d, obs_h, self.history_digest
        )
        prior, value = self.evaluator.evaluate(
            obs, memory, belief, from_root=True, shape=True
        )
        self.last_root_prior = np.asarray(prior, dtype=np.float64)
        node = self.tree.make_node(
            key=key,
            turn=int(obs.turn),
            memory_digest=mem_d,
            obs_hash=obs_h,
            history_digest=self.history_digest,
            network_value=float(value),
            capacity=self.config.n_particles,
        )
        if not node.expanded:
            self._expand_self_candidates(node, obs, memory, prior)
            node.reservoir.replace_from_belief(belief)
            node.expanded = True
            node.network_value = float(value)
        self.tree.set_root(node)
        return node

    def reuse_or_reset(
        self,
        sent_action: Action5,
        obs,
        memory: VisibleMemory,
        belief: BeliefState,
    ) -> InfoNode:
        """Follow the child for ``sent_action`` + new obs, or start a new root."""
        self.history_digest = roll_history_digest(
            self.history_digest, sent_action, observation_hash(obs)
        )
        edge = child_edge_key(sent_action, observation_hash(obs))
        child: Optional[InfoNode] = None
        if self.tree.root is not None:
            child = self.tree.root.children.get(edge)
        mem_d = memory_digest(memory)
        obs_h = observation_hash(obs)
        if child is not None and self.tree.matches_info(
            child, turn=int(obs.turn), memory_digest=mem_d, obs_hash=obs_h
        ):
            child.reservoir.replace_from_belief(belief)
            prior_e, value = self.evaluator.evaluate(
                obs, memory, belief, from_root=True, shape=True
            )
            child.network_value = float(value)
            self.last_root_prior = np.asarray(prior_e, dtype=np.float64)
            # Always refresh masses + widen from the live network prior. Skipping
            # when child.actions is non-empty left pass-only priors stuck after
            # early turns where only pass was legal.
            self._expand_self_candidates(child, obs, memory, prior_e)
            self.memory = memory
            self.tree.set_root(child)
            return child
        # Exact match failed — new root (drop reuse).
        self.tree.clear()
        self.history_digest = ZERO_DIGEST
        # Rebuild history as single-step from this observation alone.
        self.history_digest = roll_history_digest(
            ZERO_DIGEST, sent_action, obs_h
        )
        return self.ensure_root(obs, memory, belief)

    def _expand_self_candidates(
        self,
        node: InfoNode,
        obs,
        memory: VisibleMemory,
        prior: Array,
    ) -> None:
        from action import live_build_cost
        from tactics import play_mask

        cost_grid = live_build_cost(obs, memory)
        mask = play_mask(obs, memory, cost_grid=cost_grid)
        limit = self_widening_limit(node.N)
        # Cache the full network prior so later widening does not rebuild mass
        # only from already-accepted candidates (which zeros new expands).
        node.cached_policy_prior = np.asarray(prior, dtype=np.float64).reshape(-1).copy()
        if node.actions:
            node.refresh_self_priors(node.cached_policy_prior)
        mandatory = mandatory_action_indices(
            obs, memory, mask=mask, cost_grid=cost_grid
        )
        candidates = policy_ordered_candidates(
            prior, mask, mandatory=mandatory, limit=limit
        )
        for idx in candidates:
            mass = float(prior[idx]) if idx < len(prior) else 0.0
            node.widen_self(idx, mass)
        # Re-normalize from the live prior after any new widen appends so early
        # pass-only mass cannot dominate newly added expands.
        if node.actions:
            node.refresh_self_priors(node.cached_policy_prior)

    def _enemy_view(self, particle: Particle) -> tuple[object, VisibleMemory, bytes]:
        enemy_seat = 1 - self.seat
        enemy_obs = emit_observation(particle.state, enemy_seat, as_arrays=True)
        enemy_mem = update_memory(particle.enemy_memory, enemy_obs)
        h = enemy_info_hash_prehashed(
            observation_hash(enemy_obs), memory_digest(enemy_mem)
        )
        return enemy_obs, enemy_mem, h

    def _install_enemy_table(
        self,
        node: InfoNode,
        info_hash: bytes,
        enemy_obs,
        enemy_mem: VisibleMemory,
        prior_e: Array,
    ) -> object:
        """Create an enemy table from an already-evaluated prior (no network)."""
        from action import live_build_cost

        cost_grid = live_build_cost(enemy_obs, enemy_mem)
        mask = legal_mask(enemy_obs, enemy_mem, cost_grid=cost_grid)
        limit = enemy_widening_limit(node.N)
        mandatory = mandatory_action_indices(
            enemy_obs, enemy_mem, mask=mask, cost_grid=cost_grid
        )
        candidates = policy_ordered_candidates(
            prior_e, mask, mandatory=mandatory, limit=limit
        )
        priors = np.asarray(
            [float(prior_e[i]) if i < len(prior_e) else 0.0 for i in candidates],
            dtype=np.float64,
        )
        return self.tree.get_or_create_enemy_table(node, info_hash, candidates, priors)

    def materialize_enemy_priors(
        self,
        requests: Sequence[EnemyPriorRequest],
        belief: BeliefState,
    ) -> int:
        """Batch-evaluate missing enemy priors and install tables.

        Returns the number of network forwards. Selection must not call this.
        """
        import time

        self.last_select_enemy_prior_ms = 0.0
        self.last_select_enemy_prior_forwards = 0
        if not requests:
            return 0

        unique: dict[tuple[int, bytes], EnemyPriorRequest] = {}
        for req in requests:
            node = req.node
            if req.info_hash in node.enemy_tables:
                continue
            key = (id(node), req.info_hash)
            if key not in unique:
                unique[key] = req
        if not unique:
            return 0

        enemy_seat = 1 - self.seat
        ordered = list(unique.values())
        prior_items: list[tuple[object, VisibleMemory, BeliefState]] = []
        for req in ordered:
            enemy_belief = BeliefState(
                seat=enemy_seat,
                particles=[req.particle],
                config=belief.config,
            )
            prior_items.append((req.enemy_obs, req.enemy_mem, enemy_belief))

        t0 = time.perf_counter()
        # Prefer policy-only priors: enemy tables discard value.
        prior_fn = getattr(self.evaluator, "policy_priors_many", None)
        if callable(prior_fn):
            priors = prior_fn(prior_items)
        else:
            batch_fn = getattr(self.evaluator, "evaluate_many", None)
            if callable(batch_fn):
                results = batch_fn(
                    [(obs, mem, blf, False, False) for obs, mem, blf in prior_items]
                )
            else:
                results = [
                    self.evaluator.evaluate(obs, mem, blf, from_root=False)
                    for obs, mem, blf in prior_items
                ]
            priors = [prior for prior, _value in results]
        self.last_select_enemy_prior_ms = (time.perf_counter() - t0) * 1000.0
        self.last_select_enemy_prior_forwards = len(priors)

        for req, prior in zip(ordered, priors):
            node = req.node
            self._store_enemy_prior(req.info_hash, np.asarray(prior))
            self.enemy_prior_cache_misses += 1
            if req.info_hash in node.enemy_tables:
                continue
            self._install_enemy_table(
                node,
                req.info_hash,
                req.enemy_obs,
                req.enemy_mem,
                np.asarray(prior, dtype=np.float64),
            )
        return int(self.last_select_enemy_prior_forwards)

    def _widen_if_needed(
        self,
        node: InfoNode,
        obs,
        memory: VisibleMemory,
        prior: Array,
        *,
        enemy: bool,
        enemy_obs=None,
        enemy_mem=None,
        enemy_prior: Optional[Array] = None,
        table=None,
    ) -> None:
        if enemy:
            assert table is not None and enemy_obs is not None and enemy_mem is not None
            limit = enemy_widening_limit(node.N)
            if len(table.actions) >= limit:
                return
            from action import live_build_cost

            cost_grid = live_build_cost(enemy_obs, enemy_mem)
            mask = legal_mask(enemy_obs, enemy_mem, cost_grid=cost_grid)
            mandatory = mandatory_action_indices(
                enemy_obs, enemy_mem, mask=mask, cost_grid=cost_grid
            )
            p = enemy_prior if enemy_prior is not None else table.prior
            # Rebuild a full-length prior vector for ordering.
            full = np.zeros(len(mask), dtype=np.float64)
            for i, act in enumerate(table.actions):
                if i < len(table.prior):
                    full[act] = table.prior[i]
            if enemy_prior is not None:
                full = enemy_prior
            candidates = policy_ordered_candidates(
                full, mask, mandatory=mandatory, limit=limit
            )
            for idx in candidates:
                if idx in table.actions:
                    continue
                mass = float(full[idx]) if idx < len(full) else 0.0
                table.widen_enemy(idx, mass)
                if len(table.actions) >= limit:
                    break
        else:
            limit = self_widening_limit(node.N)
            if len(node.actions) >= limit:
                return
            self._expand_self_candidates(node, obs, memory, prior)

    def select_path(
        self,
        belief: BeliefState,
        *,
        freeze_snapshot: bool = True,
        freeze_widening: bool = False,
        resume: Optional[EnemyPriorRequest] = None,
    ) -> PendingPath | EnemyPriorRequest:
        """Select one simulation path from a frozen statistics snapshot.

        Performs zero network calls. When an enemy table prior is missing,
        returns an ``EnemyPriorRequest`` so the caller can batch-materialise
        priors and resume. An unexpanded node stops for leaf evaluation.

        When ``freeze_widening`` is True, progressive widening is skipped so
        the matrix width stays fixed (runtime forecast below 16 simulations).
        """
        del freeze_snapshot  # snapshot freeze is enforced by the caller batching
        assert self.tree.root is not None and self.memory is not None

        if resume is not None:
            node = resume.node
            particle = resume.particle
            state = resume.state
            nodes = list(resume.nodes)
            edges = list(resume.edges)
            depth = int(resume.depth)
            freeze_widening = bool(resume.freeze_widening)
        else:
            node = self.tree.root
            particle = node.reservoir.sample(self.rng)
            state = particle.state
            nodes = [node]
            edges = []
            depth = 0

        while depth < self.config.depth:
            if state.winner >= 0:
                seat_win = 1.0 if state.winner == self.seat else -1.0
                return PendingPath(
                    nodes=nodes,
                    edges=edges,
                    particle=particle,
                    leaf_state=state,
                    leaf_node=node,
                    needs_expand=False,
                    terminal_value=seat_win,
                )

            # Unexpanded node: stop for leaf evaluation (no network here).
            if not node.actions:
                return PendingPath(
                    nodes=nodes,
                    edges=edges,
                    particle=particle,
                    leaf_state=state,
                    leaf_node=node,
                    needs_expand=True,
                    terminal_value=None,
                )

            my_obs = emit_observation(state, self.seat, as_arrays=True)
            my_mem = (
                self.memory
                if node is self.tree.root
                else update_memory(self.memory, my_obs)
            )

            enemy_obs, enemy_mem, h = self._enemy_view(particle)
            table = node.enemy_tables.get(h)
            if table is None:
                # Cross-turn cache first: an install from a cached prior needs
                # no network, so selection keeps its zero-forward invariant.
                cached = self._cached_enemy_prior(h)
                if cached is not None:
                    self._install_enemy_table(node, h, enemy_obs, enemy_mem, cached)
                    table = node.enemy_tables[h]
            if table is None:
                # Pin before materialisation so a later install cannot evict it.
                self.tree.pin_enemy(node, h)
                return EnemyPriorRequest(
                    nodes=nodes,
                    edges=edges,
                    particle=particle,
                    state=state,
                    depth=depth,
                    freeze_widening=freeze_widening,
                    info_hash=h,
                    enemy_obs=enemy_obs,
                    enemy_mem=enemy_mem,
                    my_obs=my_obs,
                    my_mem=my_mem,
                )
            table = self.tree.get_or_create_enemy_table(
                node, h, table.actions, table.prior
            )
            self.tree.pin_enemy(node, h)

            if not freeze_widening:
                # Prefer a full-length network prior. Rebuilding from node.prior
                # alone assigns mass 0 to newly legal expands (pass lock).
                if node is self.tree.root and self.last_root_prior is not None:
                    prior_for_self = np.asarray(self.last_root_prior, dtype=np.float64)
                elif node.cached_policy_prior is not None:
                    prior_for_self = np.asarray(
                        node.cached_policy_prior, dtype=np.float64
                    )
                else:
                    prior_for_self = np.zeros(3970, dtype=np.float64)
                    for i, act in enumerate(node.actions):
                        if i < len(node.prior):
                            prior_for_self[act] = node.prior[i]
                self._widen_if_needed(
                    node, my_obs, my_mem, prior_for_self, enemy=False
                )
                self._widen_if_needed(
                    node,
                    my_obs,
                    my_mem,
                    prior_for_self,
                    enemy=True,
                    enemy_obs=enemy_obs,
                    enemy_mem=enemy_mem,
                    table=table,
                )

            sigma_a = mixed_strategy(node.regret, node.prior, node.N)
            sigma_b = mixed_strategy(table.regret, table.prior, node.N)
            a_idx = sample_index(sigma_a, self.rng)
            b_idx = sample_index(sigma_b, self.rng)
            a = as_action5(decode_action(node.actions[a_idx]))
            b = as_action5(decode_action(table.actions[b_idx]))

            actions = np.stack([PASS_ACTION, PASS_ACTION]).astype(np.int32)
            actions[self.seat] = np.asarray(a, dtype=np.int32)
            actions[1 - self.seat] = np.asarray(b, dtype=np.int32)
            next_state, info = transition(state, actions)

            enemy_seat = 1 - self.seat
            # One observation pair per depth step; reuse for hash, memory, edges.
            child_obs = emit_observation(next_state, self.seat, as_arrays=True)
            child_obs_h = observation_hash(child_obs)
            enemy_next_obs = emit_observation(
                next_state, enemy_seat, as_arrays=True
            )
            enemy_next_mem = update_memory(particle.enemy_memory, enemy_next_obs)

            if info.is_done or next_state.winner >= 0:
                if next_state.winner == self.seat:
                    term = 1.0
                elif next_state.winner < 0:
                    term = 0.0
                else:
                    term = -1.0
                if info.is_done and next_state.winner < 0:
                    term = 0.0
                edge = child_edge_key(a, child_obs_h)
                edges.append((h, a_idx, b_idx, edge))
                return PendingPath(
                    nodes=nodes,
                    edges=edges,
                    particle=Particle(
                        state=next_state,
                        weight=particle.weight,
                        enemy_memory=enemy_next_mem,
                        enemy_prev_action=b,
                        history=particle.history,
                    ),
                    leaf_state=next_state,
                    leaf_node=None,
                    needs_expand=False,
                    terminal_value=term,
                )

            edge = child_edge_key(a, child_obs_h)
            edges.append((h, a_idx, b_idx, edge))

            child = node.children.get(edge)
            if child is None:
                child_mem = update_memory(my_mem, child_obs)
                child_mem_d = memory_digest(child_mem)
                child_hist = roll_history_digest(
                    node.history_digest, a, child_obs_h
                )
                child_key = info_state_key_prehashed(
                    int(child_obs.turn), child_mem_d, child_obs_h, child_hist
                )
                try:
                    child = self.tree.make_node(
                        key=child_key,
                        turn=int(child_obs.turn),
                        memory_digest=child_mem_d,
                        obs_hash=child_obs_h,
                        history_digest=child_hist,
                        network_value=0.0,
                        capacity=self.config.n_particles,
                    )
                except MemoryError:
                    return PendingPath(
                        nodes=nodes,
                        edges=edges,
                        particle=particle,
                        leaf_state=next_state,
                        leaf_node=node,
                        needs_expand=True,
                        terminal_value=None,
                    )
                node.children[edge] = child
                arriving = Particle(
                    state=next_state,
                    weight=1.0,
                    enemy_memory=enemy_next_mem,
                    enemy_prev_action=b,
                    history=particle.history,
                )
                child.reservoir.admit(arriving, self.rng)
                # New child is unexpanded: stop for leaf evaluation.
                return PendingPath(
                    nodes=nodes + [child],
                    edges=edges,
                    particle=arriving,
                    leaf_state=next_state,
                    leaf_node=child,
                    needs_expand=True,
                    terminal_value=None,
                )

            arriving = Particle(
                state=next_state,
                weight=particle.weight,
                enemy_memory=enemy_next_mem,
                enemy_prev_action=b,
                history=particle.history,
            )
            child.reservoir.admit(arriving, self.rng)
            node = child
            particle = arriving
            state = next_state
            nodes.append(node)
            depth += 1

        return PendingPath(
            nodes=nodes,
            edges=edges,
            particle=particle,
            leaf_state=state,
            leaf_node=node,
            needs_expand=False,
            terminal_value=None,
        )

    def complete_select_path(
        self,
        belief: BeliefState,
        *,
        freeze_snapshot: bool = True,
        freeze_widening: bool = False,
    ) -> PendingPath:
        """Select one path, materialising enemy priors as needed (test helper)."""
        outcome: PendingPath | EnemyPriorRequest = self.select_path(
            belief,
            freeze_snapshot=freeze_snapshot,
            freeze_widening=freeze_widening,
        )
        while isinstance(outcome, EnemyPriorRequest):
            self.materialize_enemy_priors([outcome], belief)
            outcome = self.select_path(
                belief,
                freeze_snapshot=freeze_snapshot,
                freeze_widening=freeze_widening,
                resume=outcome,
            )
        return outcome

    def evaluate_leaf(
        self,
        path: PendingPath,
        belief: BeliefState,
    ) -> float:
        return self.evaluate_leaves([path], belief)[0]

    def evaluate_leaves(
        self,
        paths: Sequence[PendingPath],
        belief: BeliefState,
    ) -> list[float]:
        """Evaluate many leaves; batches network forwards when the evaluator allows."""
        values: list[Optional[float]] = [None] * len(paths)
        pending_idx: list[int] = []
        pending_items: list[
            tuple[object, VisibleMemory, BeliefState, bool, bool]
        ] = []
        pending_paths: list[PendingPath] = []

        for i, path in enumerate(paths):
            if path.terminal_value is not None:
                values[i] = float(path.terminal_value)
                continue
            state = path.leaf_state
            if state.winner == self.seat:
                values[i] = 1.0
                continue
            if state.winner >= 0:
                values[i] = -1.0
                continue
            obs = emit_observation(state, self.seat, as_arrays=True)
            mem = update_memory(self.memory, obs) if self.memory is not None else self.memory
            assert mem is not None
            leaf_belief = BeliefState(
                seat=self.seat,
                particles=[path.particle],
                config=belief.config,
            )
            pending_idx.append(i)
            # Root perspective (no sign flip), but NOT shaped: leaves keep the
            # raw network prior so the tree interior stays faithful to the net.
            pending_items.append((obs, mem, leaf_belief, True, False))
            pending_paths.append(path)

        if pending_items:
            batch_fn = getattr(self.evaluator, "evaluate_many", None)
            if callable(batch_fn):
                results = batch_fn(pending_items)
            else:
                results = [
                    self.evaluator.evaluate(
                        obs, mem, blf, from_root=fr, shape=sh
                    )
                    for obs, mem, blf, fr, sh in pending_items
                ]
            for local_j, (i, path, (prior, value)) in enumerate(
                zip(pending_idx, pending_paths, results)
            ):
                values[i] = float(value)
                if path.leaf_node is not None and path.needs_expand:
                    path.leaf_node.network_value = float(value)
                    if not path.leaf_node.actions:
                        obs, mem, _leaf_belief, _, _ = pending_items[local_j]
                        self._expand_self_candidates(
                            path.leaf_node, obs, mem, np.asarray(prior, dtype=np.float64)
                        )
                        path.leaf_node.expanded = True

        return [float(v) for v in values]  # type: ignore[arg-type]

    def backup_path(self, path: PendingPath, leaf_value: float) -> None:
        """Backup from leaf to root. Only fully completed paths call this."""
        value = float(leaf_value)
        # edges[i] was taken at nodes[i]
        for i in range(len(path.edges) - 1, -1, -1):
            node = path.nodes[i]
            h, a_idx, b_idx, _edge = path.edges[i]
            self.tree.backup_node(
                node, h_star=h, a_idx=a_idx, b_idx=b_idx, leaf_value=value
            )
            self.tree.clear_pins(node)
        self.tree.completed_simulations += 1

    def run_batch(
        self,
        belief: BeliefState,
        *,
        n_sims: Optional[int] = None,
        freeze_widening: bool = False,
    ) -> int:
        """Select up to ``pending_batch`` paths, evaluate, backup in order."""
        if self.tree.root is None:
            raise ValueError("root not set")
        batch_n = n_sims if n_sims is not None else self.config.pending_batch
        batch_n = min(batch_n, self.config.pending_batch)
        paths: list[PendingPath] = []
        for _ in range(batch_n):
            paths.append(
                self.complete_select_path(
                    belief,
                    freeze_snapshot=True,
                    freeze_widening=freeze_widening,
                )
            )
        completed = 0
        for path in paths:
            value = self.evaluate_leaf(path, belief)
            self.backup_path(path, value)
            completed += 1
        return completed

    def root_marginal_visits(self) -> Array:
        assert self.tree.root is not None
        visits = np.zeros(len(self.tree.root.actions), dtype=np.float64)
        for table in self.tree.root.enemy_tables.values():
            if table.visits.shape[0] == len(self.tree.root.actions):
                visits = visits + table.visits.sum(axis=1)
        return visits

    def best_action(self) -> Action5:
        idx = self.tree.root_action_index()
        assert self.tree.root is not None
        return as_action5(decode_action(self.tree.root.actions[idx]))

    def best_action_by_visits(self) -> Action5:
        """Highest marginal visit; ties by root prior (1–7 simulation band)."""
        assert self.tree.root is not None and self.tree.root.actions
        visits = self.root_marginal_visits()
        prior = np.asarray(self.tree.root.prior, dtype=np.float64)
        order = np.lexsort((-prior, -visits))
        return as_action5(decode_action(self.tree.root.actions[int(order[0])]))

    def best_action_or_pass(self) -> Action5:
        if self.tree.root is None or not self.tree.root.actions:
            return pass_action()
        if self.tree.completed_simulations <= 0:
            # Highest prior among candidates.
            root = self.tree.root
            order = int(np.argmax(root.prior)) if len(root.prior) else 0
            return as_action5(decode_action(root.actions[order]))
        # One selector for any completed simulation (see select_degraded_action).
        return self.best_action()


@dataclass
class UniformEvaluator:
    """Deterministic stub: uniform legal prior, constant value."""

    value: float = 0.0

    def evaluate(
        self,
        obs,
        memory: VisibleMemory,
        belief: BeliefState,
        *,
        from_root: bool,
        shape: bool = False,
    ) -> tuple[Array, float]:
        mask = legal_mask(obs, memory)
        prior = np.zeros(mask.shape, dtype=np.float64)
        n = int(mask.sum())
        if n <= 0:
            prior[PASS_INDEX] = 1.0
        else:
            prior[mask] = 1.0 / float(n)
        return prior, float(self.value)


@dataclass
class ScriptedEvaluator:
    """Test helper: fixed prior vector and value."""

    prior: Array
    value: float = 0.0

    def evaluate(
        self,
        obs,
        memory: VisibleMemory,
        belief: BeliefState,
        *,
        from_root: bool,
        shape: bool = False,
    ) -> tuple[Array, float]:
        prior = np.asarray(self.prior, dtype=np.float64).reshape(-1).copy()
        mask = legal_mask(obs, memory)
        prior = np.where(mask, prior, 0.0)
        s = float(prior.sum())
        if s <= 0.0:
            prior = np.zeros_like(prior)
            prior[mask] = 1.0 / max(int(mask.sum()), 1)
        else:
            prior = prior / s
        return prior, float(self.value)
