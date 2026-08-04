"""Root-sampled information-set MCTS with simultaneous matrices.

Uses Part 02 ``transition`` only. Network priors and values are injectable so
tests do not require a trained checkpoint. Part 07 wires the runtime deadline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence

import numpy as np

from action import PASS_INDEX, decode_action, legal_mask
from belief import BeliefState, Particle, as_action5, pass_action
from hashing import (
    ZERO_DIGEST,
    child_edge_key,
    enemy_info_hash,
    info_state_key,
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
    ) -> tuple[Array, float]:
        """Return ``(prior_3970, V)`` from the requested perspective.

        ``V`` must already be in the root-player perspective (no sign flip
        inside backup).
        """
        ...


@dataclass
class SearchConfig:
    depth: int = SEARCH_DEPTH
    pending_batch: int = PENDING_LEAF_BATCH
    max_nodes: int = 4096
    max_enemy_tables: int = 8
    n_particles: int = 64


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

    def __post_init__(self) -> None:
        self.tree = SearchTree(
            max_nodes=self.config.max_nodes,
            max_enemy_tables=self.config.max_enemy_tables,
            seat=self.seat,
        )
        self.last_root_prior = None

    def ensure_root(
        self,
        obs,
        memory: VisibleMemory,
        belief: BeliefState,
    ) -> InfoNode:
        self.memory = memory
        obs_h = observation_hash(obs)
        mem_d = memory_digest(memory)
        key = info_state_key(int(obs.turn), memory, obs_h, self.history_digest)
        prior, value = self.evaluator.evaluate(
            obs, memory, belief, from_root=True
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
                obs, memory, belief, from_root=True
            )
            child.network_value = float(value)
            self.last_root_prior = np.asarray(prior_e, dtype=np.float64)
            if not child.actions:
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
        mask = legal_mask(obs, memory)
        limit = self_widening_limit(node.N)
        mandatory = mandatory_action_indices(obs, memory)
        candidates = policy_ordered_candidates(
            prior, mask, mandatory=mandatory, limit=limit
        )
        for idx in candidates:
            mass = float(prior[idx]) if idx < len(prior) else 0.0
            node.widen_self(idx, mass)

    def _ensure_enemy_table(
        self,
        node: InfoNode,
        particle: Particle,
        belief: BeliefState,
    ) -> tuple[bytes, object]:
        enemy_seat = 1 - self.seat
        enemy_obs = emit_observation(particle.state, enemy_seat)
        enemy_mem = update_memory(particle.enemy_memory, enemy_obs)
        h = enemy_info_hash(enemy_obs, enemy_mem)
        if h in node.enemy_tables:
            table = self.tree.get_or_create_enemy_table(
                node, h, node.enemy_tables[h].actions, node.enemy_tables[h].prior
            )
            return h, table
        # Build enemy candidates from enemy-perspective prior.
        enemy_belief = BeliefState(
            seat=enemy_seat,
            particles=[particle],
            config=belief.config,
        )
        prior_e, _ = self.evaluator.evaluate(
            enemy_obs, enemy_mem, enemy_belief, from_root=False
        )
        mask = legal_mask(enemy_obs, enemy_mem)
        limit = enemy_widening_limit(node.N)
        mandatory = mandatory_action_indices(enemy_obs, enemy_mem)
        candidates = policy_ordered_candidates(
            prior_e, mask, mandatory=mandatory, limit=limit
        )
        priors = np.asarray(
            [float(prior_e[i]) if i < len(prior_e) else 0.0 for i in candidates],
            dtype=np.float64,
        )
        table = self.tree.get_or_create_enemy_table(node, h, candidates, priors)
        return h, table

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
            mask = legal_mask(enemy_obs, enemy_mem)
            mandatory = mandatory_action_indices(enemy_obs, enemy_mem)
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
    ) -> PendingPath:
        """Select one simulation path from a frozen statistics snapshot.

        When ``freeze_widening`` is True, progressive widening is skipped so
        the matrix width stays fixed (runtime forecast below 16 simulations).
        """
        assert self.tree.root is not None and self.memory is not None
        node = self.tree.root
        particle = node.reservoir.sample(self.rng)
        state = particle.state
        nodes = [node]
        edges: list[tuple[bytes, int, int, bytes]] = []
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

            my_obs = emit_observation(state, self.seat)
            my_mem = (
                self.memory
                if node is self.tree.root
                else update_memory(
                    # Use root memory only at root; deeper nodes use updated
                    # memory from the simulated observation path via particle.
                    self.memory,
                    my_obs,
                )
            )
            # Prefer node-local expansion using current legal mask.
            prior_full = np.zeros(3970, dtype=np.float64)
            for i, act in enumerate(node.actions):
                if i < len(node.prior):
                    prior_full[act] = node.prior[i]
            if not node.actions:
                prior_eval, value = self.evaluator.evaluate(
                    my_obs,
                    my_mem,
                    belief,
                    from_root=True,
                )
                node.network_value = float(value)
                self._expand_self_candidates(node, my_obs, my_mem, prior_eval)
                prior_full = prior_eval

            h, table = self._ensure_enemy_table(node, particle, belief)
            self.tree.pin_enemy(node, h)

            # Optional progressive widening under frozen snapshot: only add
            # candidates when the node's visit count warrants it. When
            # freeze_snapshot is True we still allow widening based on N
            # observed at selection start for this node.
            enemy_seat = 1 - self.seat
            enemy_obs = emit_observation(state, enemy_seat)
            enemy_mem = update_memory(particle.enemy_memory, enemy_obs)
            if not freeze_widening:
                self._widen_if_needed(
                    node, my_obs, my_mem, prior_full, enemy=False
                )
                self._widen_if_needed(
                    node,
                    my_obs,
                    my_mem,
                    prior_full,
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

            if info.is_done or next_state.winner >= 0:
                if next_state.winner == self.seat:
                    term = 1.0
                elif next_state.winner < 0:
                    term = 0.0
                else:
                    term = -1.0
                # Draw when both lose? winner stays -1 with is_done from mutual.
                if info.is_done and next_state.winner < 0:
                    term = 0.0
                edge = child_edge_key(a, observation_hash(emit_observation(next_state, self.seat)))
                edges.append((h, a_idx, b_idx, edge))
                return PendingPath(
                    nodes=nodes,
                    edges=edges,
                    particle=Particle(
                        state=next_state,
                        weight=particle.weight,
                        enemy_memory=update_memory(
                            particle.enemy_memory,
                            emit_observation(next_state, enemy_seat),
                        ),
                        enemy_prev_action=b,
                        history=particle.history,
                    ),
                    leaf_state=next_state,
                    leaf_node=None,
                    needs_expand=False,
                    terminal_value=term,
                )

            child_obs = emit_observation(next_state, self.seat)
            child_obs_h = observation_hash(child_obs)
            edge = child_edge_key(a, child_obs_h)
            edges.append((h, a_idx, b_idx, edge))

            child = node.children.get(edge)
            if child is None:
                # Expand one new child or stop at depth.
                child_mem = update_memory(my_mem, child_obs)
                child_hist = roll_history_digest(
                    node.history_digest, a, child_obs_h
                )
                child_key = info_state_key(
                    int(child_obs.turn), child_mem, child_obs_h, child_hist
                )
                try:
                    child = self.tree.make_node(
                        key=child_key,
                        turn=int(child_obs.turn),
                        memory_digest=memory_digest(child_mem),
                        obs_hash=child_obs_h,
                        history_digest=child_hist,
                        network_value=0.0,
                        capacity=self.config.n_particles,
                    )
                except MemoryError:
                    # Bound hit: bootstrap here without a new node.
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
                    enemy_memory=update_memory(
                        particle.enemy_memory,
                        emit_observation(next_state, enemy_seat),
                    ),
                    enemy_prev_action=b,
                    history=particle.history,
                )
                child.reservoir.admit(arriving, self.rng)
                return PendingPath(
                    nodes=nodes + [child],
                    edges=edges,
                    particle=arriving,
                    leaf_state=next_state,
                    leaf_node=child,
                    needs_expand=True,
                    terminal_value=None,
                )

            # Follow existing child.
            arriving = Particle(
                state=next_state,
                weight=particle.weight,
                enemy_memory=update_memory(
                    particle.enemy_memory,
                    emit_observation(next_state, enemy_seat),
                ),
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
        pending_items: list[tuple[object, VisibleMemory, BeliefState, bool]] = []
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
            obs = emit_observation(state, self.seat)
            mem = update_memory(self.memory, obs) if self.memory is not None else self.memory
            assert mem is not None
            leaf_belief = BeliefState(
                seat=self.seat,
                particles=[path.particle],
                config=belief.config,
            )
            pending_idx.append(i)
            pending_items.append((obs, mem, leaf_belief, True))
            pending_paths.append(path)

        if pending_items:
            batch_fn = getattr(self.evaluator, "evaluate_many", None)
            if callable(batch_fn):
                results = batch_fn(pending_items)
            else:
                results = [
                    self.evaluator.evaluate(obs, mem, blf, from_root=fr)
                    for obs, mem, blf, fr in pending_items
                ]
            for local_j, (i, path, (prior, value)) in enumerate(
                zip(pending_idx, pending_paths, results)
            ):
                values[i] = float(value)
                if path.leaf_node is not None and path.needs_expand:
                    path.leaf_node.network_value = float(value)
                    if not path.leaf_node.actions:
                        obs, mem, _leaf_belief, _ = pending_items[local_j]
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
                self.select_path(
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
        if self.tree.completed_simulations < 8:
            return self.best_action_by_visits()
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
