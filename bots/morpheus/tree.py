"""Bounded information-set tree storage for Morpheus search.

Matrix math lives in ``matrix.py``. This module owns nodes, enemy tables,
joint statistics, children, and eviction accounting.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from hashing import digests_equal
from matrix import (
    accumulate_average_strategy,
    aggregate_self_utilities,
    apply_joint_backup,
    effective_q,
    matrix_utilities,
    mixed_strategy,
    regret_plus_update,
    select_root_action,
)
from reservoir import ParticleReservoir

Array = np.ndarray

MAX_ENEMY_TABLES = 8
MAX_TREE_NODES = 4096
LRU_TOUCH_WEIGHT = 0.25


@dataclass
class EnemyTable:
    """Per-enemy-information-hash action statistics."""

    info_hash: bytes
    actions: list[int]  # logit indices
    prior: Array
    regret: Array
    avg_strategy: Array
    visits: Array  # (n_self, n_enemy)
    value_sum: Array
    q: Array
    last_used: int = 0
    touch_count: int = 0

    @classmethod
    def create(
        cls,
        info_hash: bytes,
        actions: list[int],
        prior: Array,
        n_self: int,
    ) -> "EnemyTable":
        n_b = len(actions)
        prior = np.asarray(prior, dtype=np.float64).reshape(-1)
        if len(prior) != n_b:
            raise ValueError("enemy prior length mismatch")
        s = float(np.maximum(prior, 0.0).sum())
        if s > 0.0:
            prior = np.maximum(prior, 0.0) / s
        else:
            prior = np.full(n_b, 1.0 / max(n_b, 1), dtype=np.float64)
        return cls(
            info_hash=info_hash,
            actions=list(actions),
            prior=prior,
            regret=np.zeros(n_b, dtype=np.float64),
            avg_strategy=np.zeros(n_b, dtype=np.float64),
            visits=np.zeros((n_self, n_b), dtype=np.float64),
            value_sum=np.zeros((n_self, n_b), dtype=np.float64),
            q=np.zeros((n_self, n_b), dtype=np.float64),
        )

    def retention_score(self) -> float:
        return float(self.last_used) + LRU_TOUCH_WEIGHT * math.log1p(self.touch_count)

    def touch(self, node_visits: int) -> None:
        self.last_used = int(node_visits)
        self.touch_count += 1

    def ensure_self_rows(self, n_self: int) -> None:
        """Pad joint matrices when self candidates widen."""
        cur = int(self.visits.shape[0])
        if n_self <= cur:
            return
        pad = n_self - cur
        n_b = int(self.visits.shape[1])
        self.visits = np.vstack(
            [self.visits, np.zeros((pad, n_b), dtype=np.float64)]
        )
        self.value_sum = np.vstack(
            [self.value_sum, np.zeros((pad, n_b), dtype=np.float64)]
        )
        self.q = np.vstack([self.q, np.zeros((pad, n_b), dtype=np.float64)])

    def widen_enemy(self, action: int, prior_mass: float) -> int:
        """Append one enemy action; return its column index."""
        if action in self.actions:
            return self.actions.index(action)
        n_self = int(self.visits.shape[0])
        self.actions.append(int(action))
        self.prior = np.append(self.prior, float(max(prior_mass, 0.0)))
        # Renormalize prior over current columns.
        s = float(np.maximum(self.prior, 0.0).sum())
        if s > 0.0:
            self.prior = np.maximum(self.prior, 0.0) / s
        self.regret = np.append(self.regret, 0.0)
        self.avg_strategy = np.append(self.avg_strategy, 0.0)
        col = np.zeros((n_self, 1), dtype=np.float64)
        self.visits = np.hstack([self.visits, col.copy()])
        self.value_sum = np.hstack([self.value_sum, col.copy()])
        self.q = np.hstack([self.q, col.copy()])
        return len(self.actions) - 1


@dataclass
class InfoNode:
    """One information-set node from the root player's perspective."""

    key: bytes
    turn: int
    memory_digest: bytes
    obs_hash: bytes
    history_digest: bytes
    reservoir: ParticleReservoir = field(default_factory=ParticleReservoir)
    N: int = 0
    actions: list[int] = field(default_factory=list)
    prior: Array = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    regret: Array = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    avg_strategy: Array = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    enemy_tables: dict[bytes, EnemyTable] = field(default_factory=dict)
    children: dict[bytes, "InfoNode"] = field(default_factory=dict)
    network_value: float = 0.0
    pending_pins: set[bytes] = field(default_factory=set)
    expanded: bool = False
    # Cache: particle enemy-info hashes for the current reservoir version.
    _enemy_hash_version: int = -1
    _enemy_hash_cache: list[tuple[float, bytes]] = field(default_factory=list)

    def widen_self(self, action: int, prior_mass: float) -> int:
        if action in self.actions:
            return self.actions.index(action)
        self.actions.append(int(action))
        self.prior = np.append(self.prior, float(max(prior_mass, 0.0)))
        s = float(np.maximum(self.prior, 0.0).sum())
        if s > 0.0:
            self.prior = np.maximum(self.prior, 0.0) / s
        self.regret = np.append(self.regret, 0.0)
        self.avg_strategy = np.append(self.avg_strategy, 0.0)
        n_self = len(self.actions)
        for table in self.enemy_tables.values():
            table.ensure_self_rows(n_self)
        return n_self - 1

    def retention_evict(
        self,
        *,
        max_tables: int = MAX_ENEMY_TABLES,
        protect: Optional[set[bytes]] = None,
    ) -> Optional[EnemyTable]:
        """Evict the lowest-score unpinned unprotected table if over capacity."""
        protect = protect or set()
        while len(self.enemy_tables) > max_tables:
            candidates = [
                (h, t)
                for h, t in self.enemy_tables.items()
                if h not in self.pending_pins and h not in protect
            ]
            if not candidates:
                return None
            h_kill, table = min(candidates, key=lambda it: it[1].retention_score())
            del self.enemy_tables[h_kill]
            return table
        return None


@dataclass
class SearchTree:
    """Bounded forest with a single active root."""

    max_nodes: int = MAX_TREE_NODES
    max_enemy_tables: int = MAX_ENEMY_TABLES
    seat: int = 0
    root: Optional[InfoNode] = None
    nodes: dict[bytes, InfoNode] = field(default_factory=dict)
    eviction_loss: float = 0.0
    total_joint_visits: float = 0.0
    table_hits: int = 0
    table_misses: int = 0
    completed_simulations: int = 0

    def clear(self) -> None:
        self.root = None
        self.nodes.clear()
        self.eviction_loss = 0.0
        self.total_joint_visits = 0.0
        self.table_hits = 0
        self.table_misses = 0
        self.completed_simulations = 0

    def _register(self, node: InfoNode) -> InfoNode:
        if node.key in self.nodes:
            return self.nodes[node.key]
        if len(self.nodes) >= self.max_nodes and node.key not in self.nodes:
            # Bound: refuse new nodes beyond the cap (caller must reuse / reset).
            raise MemoryError(
                f"search tree node cap {self.max_nodes} reached"
            )
        self.nodes[node.key] = node
        return node

    def make_node(
        self,
        *,
        key: bytes,
        turn: int,
        memory_digest: bytes,
        obs_hash: bytes,
        history_digest: bytes,
        network_value: float = 0.0,
        capacity: int = 64,
    ) -> InfoNode:
        existing = self.nodes.get(key)
        if existing is not None:
            return existing
        node = InfoNode(
            key=key,
            turn=int(turn),
            memory_digest=memory_digest,
            obs_hash=obs_hash,
            history_digest=history_digest,
            reservoir=ParticleReservoir(capacity=capacity),
            network_value=float(network_value),
        )
        return self._register(node)

    def set_root(self, node: InfoNode) -> None:
        self.root = node
        self.nodes.setdefault(node.key, node)

    def matches_info(
        self,
        node: InfoNode,
        *,
        turn: int,
        memory_digest: bytes,
        obs_hash: bytes,
    ) -> bool:
        return (
            int(node.turn) == int(turn)
            and digests_equal(node.memory_digest, memory_digest)
            and digests_equal(node.obs_hash, obs_hash)
        )

    def enemy_weights(self, node: InfoNode) -> tuple[list[bytes], Array]:
        hashes = list(node.enemy_tables.keys())
        if not hashes:
            return [], np.zeros(0, dtype=np.float64)
        from hashing import enemy_info_hash
        from memory import update_memory
        from observe import emit_observation

        version = int(node.reservoir.version)
        if (
            node._enemy_hash_version != version
            or len(node._enemy_hash_cache) != node.reservoir.n
        ):
            cache: list[tuple[float, bytes]] = []
            enemy_seat = 1 - self.seat
            for particle in node.reservoir.particles:
                enemy_obs = emit_observation(
                    particle.state, enemy_seat, as_arrays=True
                )
                enemy_mem = update_memory(particle.enemy_memory, enemy_obs)
                h = enemy_info_hash(enemy_obs, enemy_mem)
                cache.append((max(float(particle.weight), 0.0), h))
            node._enemy_hash_cache = cache
            node._enemy_hash_version = version

        mass = {h: 0.0 for h in hashes}
        for weight, h in node._enemy_hash_cache:
            if h in mass:
                mass[h] += weight
        weights = np.asarray([mass[h] for h in hashes], dtype=np.float64)
        total = float(weights.sum())
        if total <= 0.0 and hashes:
            weights = np.full(len(hashes), 1.0 / len(hashes), dtype=np.float64)
        elif total > 0.0:
            weights = weights / total
        return hashes, weights

    def get_or_create_enemy_table(
        self,
        node: InfoNode,
        info_hash: bytes,
        actions: list[int],
        prior: Array,
    ) -> EnemyTable:
        table = node.enemy_tables.get(info_hash)
        if table is not None:
            self.table_hits += 1
            table.touch(node.N)
            return table
        self.table_misses += 1
        n_self = max(len(node.actions), 1)
        table = EnemyTable.create(info_hash, actions, prior, n_self=n_self)
        if len(node.actions) > 0:
            table.ensure_self_rows(len(node.actions))
        node.enemy_tables[info_hash] = table
        table.touch(node.N)
        while True:
            evicted = node.retention_evict(
                max_tables=self.max_enemy_tables,
                protect={info_hash},
            )
            if evicted is None:
                break
            self._account_eviction(evicted)
        return node.enemy_tables[info_hash]

    def _account_eviction(self, table: EnemyTable) -> None:
        loss = float(np.sum(table.visits * np.abs(table.q)))
        self.eviction_loss += loss

    def pin_enemy(self, node: InfoNode, info_hash: bytes) -> None:
        node.pending_pins.add(info_hash)

    def clear_pins(self, node: InfoNode) -> None:
        node.pending_pins.clear()

    def backup_node(
        self,
        node: InfoNode,
        *,
        h_star: bytes,
        a_idx: int,
        b_idx: int,
        leaf_value: float,
    ) -> None:
        """Fully backup one simulation step at ``node`` (root perspective)."""
        table = node.enemy_tables[h_star]
        table.ensure_self_rows(len(node.actions))
        table.visits, table.value_sum, table.q = apply_joint_backup(
            visits=table.visits,
            value_sum=table.value_sum,
            q=table.q,
            a_idx=a_idx,
            b_idx=b_idx,
            leaf_value=leaf_value,
        )
        self.total_joint_visits += 1.0
        table.touch(node.N)

        sigma_self = mixed_strategy(node.regret, node.prior, node.N)
        hashes, weights = self.enemy_weights(node)
        enemy_sigmas: list[Array] = []
        q_eff_list: list[Array] = []
        for h in hashes:
            t = node.enemy_tables[h]
            t.ensure_self_rows(len(node.actions))
            sigma_b = mixed_strategy(t.regret, t.prior, node.N)
            enemy_sigmas.append(sigma_b)
            q_eff_list.append(
                effective_q(t.visits, t.q, node.network_value)
            )
        u_self, v = aggregate_self_utilities(
            sigma_self, weights, enemy_sigmas, q_eff_list
        )
        # Enemy utilities on the sampled hash only.
        q_star = effective_q(table.visits, table.q, node.network_value)
        sigma_b_star = mixed_strategy(table.regret, table.prior, node.N)
        _, u_enemy, _ = matrix_utilities(sigma_self, sigma_b_star, q_star)
        node.regret = regret_plus_update(
            node.regret, u_self, v, maximizing=True
        )
        table.regret = regret_plus_update(
            table.regret, u_enemy, v, maximizing=False
        )
        node.avg_strategy = accumulate_average_strategy(
            node.avg_strategy, sigma_self
        )
        table.avg_strategy = accumulate_average_strategy(
            table.avg_strategy, sigma_b_star
        )
        node.N += 1

    def root_action_index(self) -> int:
        if self.root is None or not self.root.actions:
            raise ValueError("empty root")

        visits = np.zeros(len(self.root.actions), dtype=np.float64)
        for table in self.root.enemy_tables.values():
            if table.visits.shape[0] == len(self.root.actions):
                visits = visits + table.visits.sum(axis=1)
        return select_root_action(
            self.root.avg_strategy, visits, self.root.prior
        )

    def eviction_loss_rate(self) -> float:
        return float(self.eviction_loss / max(1.0, self.total_joint_visits))

    def table_hit_rate(self) -> float:
        total = self.table_hits + self.table_misses
        if total == 0:
            return 0.0
        return float(self.table_hits / total)
