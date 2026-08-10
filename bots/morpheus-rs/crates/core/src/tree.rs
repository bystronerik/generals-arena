//! Bounded information-set tree storage: nodes, enemy tables, eviction.
//!
//! Port of `bots/morpheus/tree.py`. Matrix math lives in [`crate::matrix`].
//!
//! **The Python's object graph becomes an arena.** `InfoNode` there holds
//! direct references to its children and a dict of enemy tables; here nodes
//! live in one `Vec` addressed by a `u32` handle and children are a digest map
//! to handles, which is rewrite-plan §6's layout and also the only shape Rust's
//! ownership rules accept without reference counting every node.
//!
//! Two orderings are load-bearing and are reproduced deliberately rather than
//! inherited from a container:
//!
//! * `node.enemy_tables` is a Python `dict`, so `enemy_weights` walks it in
//!   **insertion order** and eviction removes from the middle. The port uses a
//!   `Vec` with linear lookup — at most eight entries — because a `HashMap`
//!   would iterate in an order the oracle never had.
//! * `min(candidates, key=...)` and `max(...)` keep the *first* extreme in
//!   Python. Every selection here spells that out.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};


use crate::board::hashing::enemy_info_hash;
use crate::matrix::{
    accumulate_average_strategy, aggregate_self_utilities, apply_joint_backup, effective_q,
    matrix_utilities, mixed_strategy, regret_plus_update, select_root_action,
};
use crate::board::memory::update_memory;
use crate::board::observe::emit_observation;
use crate::reservoir::ParticleReservoir;
use crate::support::rng::npsum;

pub type Digest = [u8; 32];

pub const MAX_ENEMY_TABLES: usize = 8;
pub const MAX_TREE_NODES: usize = 4096;
pub const LRU_TOUCH_WEIGHT: f64 = 0.25;

/// Hasher over an already-uniform SHA-256 digest.
///
/// The keys are cryptographic digests, so eight of their bytes are as good a
/// hash as anything SipHash would compute from them, and rewrite-plan §6 asks
/// for exactly this. Non-digest keys would be hashed badly; nothing else is
/// ever put in these maps.
#[derive(Default)]
pub struct DigestHasher(u64);

impl Hasher for DigestHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        let mut buf = [0u8; 8];
        let take = bytes.len().min(8);
        buf[..take].copy_from_slice(&bytes[..take]);
        self.0 ^= u64::from_le_bytes(buf);
    }
}

pub type DigestMap<V> = HashMap<Digest, V, BuildHasherDefault<DigestHasher>>;

pub fn digest_map<V>() -> DigestMap<V> {
    DigestMap::default()
}

// -------------------------------------------------------------- enemy tables

/// Per-enemy-information-hash action statistics.
///
/// `visits`, `value_sum` and `q` are row-major `n_self × n_enemy`.
#[derive(Clone)]
pub struct EnemyTable {
    pub info_hash: Digest,
    /// Logit indices.
    pub actions: Vec<usize>,
    pub prior: Vec<f64>,
    pub regret: Vec<f64>,
    pub avg_strategy: Vec<f64>,
    pub visits: Vec<f64>,
    pub value_sum: Vec<f64>,
    pub q: Vec<f64>,
    pub n_self: usize,
    pub last_used: i64,
    pub touch_count: i64,
}

impl EnemyTable {
    pub fn create(info_hash: Digest, actions: &[usize], prior: &[f64], n_self: usize) -> Self {
        let n_b = actions.len();
        assert_eq!(prior.len(), n_b, "enemy prior length mismatch");
        let positive: Vec<f64> = prior.iter().map(|&v| v.max(0.0)).collect();
        let total = npsum(&positive);
        let prior = if total > 0.0 {
            positive.iter().map(|&v| v / total).collect()
        } else {
            vec![1.0 / n_b.max(1) as f64; n_b]
        };
        Self {
            info_hash,
            actions: actions.to_vec(),
            prior,
            regret: vec![0.0; n_b],
            avg_strategy: vec![0.0; n_b],
            visits: vec![0.0; n_self * n_b],
            value_sum: vec![0.0; n_self * n_b],
            q: vec![0.0; n_self * n_b],
            n_self,
            last_used: 0,
            touch_count: 0,
        }
    }

    pub fn n_enemy(&self) -> usize {
        self.actions.len()
    }

    pub fn retention_score(&self) -> f64 {
        self.last_used as f64 + LRU_TOUCH_WEIGHT * (self.touch_count as f64).ln_1p()
    }

    pub fn touch(&mut self, node_visits: i64) {
        self.last_used = node_visits;
        self.touch_count += 1;
    }

    /// Pad the joint matrices when the self candidate set widens.
    pub fn ensure_self_rows(&mut self, n_self: usize) {
        if n_self <= self.n_self {
            return;
        }
        let n_b = self.n_enemy();
        let extra = (n_self - self.n_self) * n_b;
        self.visits.extend(std::iter::repeat(0.0).take(extra));
        self.value_sum.extend(std::iter::repeat(0.0).take(extra));
        self.q.extend(std::iter::repeat(0.0).take(extra));
        self.n_self = n_self;
    }

    /// Append one enemy action; return its column index.
    pub fn widen_enemy(&mut self, action: usize, prior_mass: f64) -> usize {
        if let Some(at) = self.actions.iter().position(|&a| a == action) {
            return at;
        }
        let old_b = self.n_enemy();
        self.actions.push(action);
        self.prior.push(prior_mass.max(0.0));
        // Renormalize the prior over the widened column set.
        let positive: Vec<f64> = self.prior.iter().map(|&v| v.max(0.0)).collect();
        let total = npsum(&positive);
        if total > 0.0 {
            self.prior = positive.iter().map(|&v| v / total).collect();
        }
        self.regret.push(0.0);
        self.avg_strategy.push(0.0);
        let new_b = old_b + 1;
        for plane in [&mut self.visits, &mut self.value_sum, &mut self.q] {
            let mut grown = vec![0.0f64; self.n_self * new_b];
            for row in 0..self.n_self {
                grown[row * new_b..row * new_b + old_b]
                    .copy_from_slice(&plane[row * old_b..(row + 1) * old_b]);
            }
            *plane = grown;
        }
        self.actions.len() - 1
    }

    fn eviction_loss(&self) -> f64 {
        let terms: Vec<f64> = self
            .visits
            .iter()
            .zip(&self.q)
            .map(|(&n, &value)| n * value.abs())
            .collect();
        npsum(&terms)
    }
}

// --------------------------------------------------------------------- nodes

/// One information-set node from the root player's perspective.
pub struct InfoNode {
    pub key: Digest,
    pub turn: i32,
    pub memory_digest: Digest,
    /// The observation's raw payload bytes, not a digest — `observation_hash`
    /// in the Python returns the payload and the reuse test compares it
    /// directly, so hashing it here would be a different equality.
    pub obs_payload: Vec<u8>,
    pub history_digest: Digest,
    pub reservoir: ParticleReservoir,
    pub n: i64,
    /// Logit indices of the self candidates.
    pub actions: Vec<usize>,
    pub prior: Vec<f64>,
    pub regret: Vec<f64>,
    pub avg_strategy: Vec<f64>,
    /// Insertion-ordered, mirroring the Python dict.
    pub enemy_tables: Vec<EnemyTable>,
    pub children: DigestMap<u32>,
    pub network_value: f64,
    pub pending_pins: Vec<Digest>,
    pub expanded: bool,
    /// Full-length network prior from the last evaluate/expand at this node.
    pub cached_policy_prior: Option<Vec<f64>>,
    enemy_hash_version: i64,
    enemy_hash_cache: Vec<(f64, Digest)>,
}

impl InfoNode {
    pub fn new(
        key: Digest,
        turn: i32,
        memory_digest: Digest,
        obs_payload: Vec<u8>,
        history_digest: Digest,
        network_value: f64,
        capacity: usize,
    ) -> Self {
        Self {
            key,
            turn,
            memory_digest,
            obs_payload,
            history_digest,
            reservoir: ParticleReservoir::new(capacity),
            n: 0,
            actions: Vec::new(),
            prior: Vec::new(),
            regret: Vec::new(),
            avg_strategy: Vec::new(),
            enemy_tables: Vec::new(),
            children: digest_map(),
            network_value,
            pending_pins: Vec::new(),
            expanded: false,
            cached_policy_prior: None,
            enemy_hash_version: -1,
            enemy_hash_cache: Vec::new(),
        }
    }

    pub fn table(&self, info_hash: &Digest) -> Option<&EnemyTable> {
        self.enemy_tables.iter().find(|t| &t.info_hash == info_hash)
    }

    pub fn table_mut(&mut self, info_hash: &Digest) -> Option<&mut EnemyTable> {
        self.enemy_tables
            .iter_mut()
            .find(|t| &t.info_hash == info_hash)
    }

    pub fn widen_self(&mut self, action: usize, prior_mass: f64) -> usize {
        if let Some(at) = self.actions.iter().position(|&a| a == action) {
            return at;
        }
        self.actions.push(action);
        self.prior.push(prior_mass.max(0.0));
        let positive: Vec<f64> = self.prior.iter().map(|&v| v.max(0.0)).collect();
        let total = npsum(&positive);
        if total > 0.0 {
            self.prior = positive.iter().map(|&v| v / total).collect();
        }
        self.regret.push(0.0);
        self.avg_strategy.push(0.0);
        let n_self = self.actions.len();
        for table in &mut self.enemy_tables {
            table.ensure_self_rows(n_self);
        }
        n_self - 1
    }

    /// Replace candidate prior masses from a full-length network prior.
    ///
    /// Used when the legal set grows or the root is reused with a fresh
    /// evaluate. Without it, widening rebuilds mass only from old candidates
    /// and newly legal expands get prior 0 — the search locks onto pass.
    pub fn refresh_self_priors(&mut self, prior: &[f64]) {
        if self.actions.is_empty() {
            return;
        }
        let masses: Vec<f64> = self
            .actions
            .iter()
            .map(|&a| prior.get(a).copied().unwrap_or(0.0).max(0.0))
            .collect();
        let total = npsum(&masses);
        self.prior = if total > 0.0 {
            masses.iter().map(|&v| v / total).collect()
        } else {
            vec![1.0 / self.actions.len() as f64; self.actions.len()]
        };
    }

    /// Evict the lowest-score unpinned, unprotected table if over capacity.
    fn retention_evict(&mut self, max_tables: usize, protect: &Digest) -> Option<EnemyTable> {
        if self.enemy_tables.len() <= max_tables {
            return None;
        }
        let mut best: Option<usize> = None;
        let mut best_score = f64::INFINITY;
        for (at, table) in self.enemy_tables.iter().enumerate() {
            if self.pending_pins.contains(&table.info_hash) || &table.info_hash == protect {
                continue;
            }
            let score = table.retention_score();
            // `min` keeps the first minimum, so replace only on strictly less.
            if best.is_none() || score < best_score {
                best_score = score;
                best = Some(at);
            }
        }
        best.map(|at| self.enemy_tables.remove(at))
    }
}

// ----------------------------------------------------------------- the tree

/// A bounded forest with a single active root.
pub struct SearchTree {
    pub max_nodes: usize,
    pub max_enemy_tables: usize,
    pub seat: usize,
    pub root: Option<u32>,
    pub nodes: Vec<InfoNode>,
    index: DigestMap<u32>,
    pub eviction_loss: f64,
    pub total_joint_visits: f64,
    pub table_hits: u64,
    pub table_misses: u64,
    pub completed_simulations: u64,
}

impl SearchTree {
    pub fn new(max_nodes: usize, max_enemy_tables: usize, seat: usize) -> Self {
        Self {
            max_nodes,
            max_enemy_tables,
            seat,
            root: None,
            nodes: Vec::new(),
            index: digest_map(),
            eviction_loss: 0.0,
            total_joint_visits: 0.0,
            table_hits: 0,
            table_misses: 0,
            completed_simulations: 0,
        }
    }

    pub fn clear(&mut self) {
        self.root = None;
        self.nodes.clear();
        self.index.clear();
        self.eviction_loss = 0.0;
        self.total_joint_visits = 0.0;
        self.table_hits = 0;
        self.table_misses = 0;
        self.completed_simulations = 0;
    }

    #[inline]
    pub fn node(&self, at: u32) -> &InfoNode {
        &self.nodes[at as usize]
    }

    #[inline]
    pub fn node_mut(&mut self, at: u32) -> &mut InfoNode {
        &mut self.nodes[at as usize]
    }

    pub fn find(&self, key: &Digest) -> Option<u32> {
        self.index.get(key).copied()
    }

    /// Register a node, or refuse past the cap.
    ///
    /// The Python raises `MemoryError`; `search.py` catches it and turns the
    /// path into a leaf evaluation, so the refusal is part of the contract
    /// rather than a crash.
    #[allow(clippy::too_many_arguments)]
    pub fn make_node(
        &mut self,
        key: Digest,
        turn: i32,
        memory_digest: Digest,
        obs_payload: Vec<u8>,
        history_digest: Digest,
        network_value: f64,
        capacity: usize,
    ) -> Result<u32, ()> {
        if let Some(&at) = self.index.get(&key) {
            return Ok(at);
        }
        if self.nodes.len() >= self.max_nodes {
            return Err(());
        }
        let at = self.nodes.len() as u32;
        self.nodes.push(InfoNode::new(
            key,
            turn,
            memory_digest,
            obs_payload,
            history_digest,
            network_value,
            capacity,
        ));
        self.index.insert(key, at);
        Ok(at)
    }

    pub fn set_root(&mut self, at: u32) {
        self.root = Some(at);
        let key = self.nodes[at as usize].key;
        self.index.entry(key).or_insert(at);
    }

    pub fn matches_info(
        &self,
        at: u32,
        turn: i32,
        memory_digest: &Digest,
        obs_payload: &[u8],
    ) -> bool {
        let node = self.node(at);
        node.turn == turn
            && &node.memory_digest == memory_digest
            && node.obs_payload == obs_payload
    }

    /// Particle mass behind each installed enemy table.
    ///
    /// The per-particle enemy hashes are cached against the reservoir version,
    /// so a resample invalidates them and a repeated backup does not re-emit
    /// eight observations.
    pub fn enemy_weights(&mut self, at: u32) -> (Vec<Digest>, Vec<f64>) {
        let seat = self.seat;
        let node = &mut self.nodes[at as usize];
        let hashes: Vec<Digest> = node.enemy_tables.iter().map(|t| t.info_hash).collect();
        if hashes.is_empty() {
            return (Vec::new(), Vec::new());
        }
        let version = node.reservoir.version as i64;
        if node.enemy_hash_version != version || node.enemy_hash_cache.len() != node.reservoir.n() {
            let enemy_seat = 1 - seat;
            let mut cache = Vec::with_capacity(node.reservoir.n());
            for particle in &node.reservoir.particles {
                let enemy_obs = emit_observation(&particle.state, enemy_seat);
                let enemy_mem = update_memory(&particle.enemy_memory, &enemy_obs);
                cache.push((
                    particle.weight.max(0.0),
                    enemy_info_hash(&enemy_obs, &enemy_mem),
                ));
            }
            node.enemy_hash_cache = cache;
            node.enemy_hash_version = version;
        }

        let mut mass = vec![0.0f64; hashes.len()];
        for &(weight, ref h) in &node.enemy_hash_cache {
            if let Some(at) = hashes.iter().position(|k| k == h) {
                mass[at] += weight;
            }
        }
        let total = npsum(&mass);
        let weights = if total > 0.0 {
            mass.iter().map(|&v| v / total).collect()
        } else {
            vec![1.0 / hashes.len() as f64; hashes.len()]
        };
        (hashes, weights)
    }

    /// Fetch or install one enemy table, then evict down to capacity.
    pub fn get_or_create_enemy_table(
        &mut self,
        at: u32,
        info_hash: Digest,
        actions: &[usize],
        prior: &[f64],
    ) {
        let node_n = self.nodes[at as usize].n;
        if let Some(table) = self.nodes[at as usize].table_mut(&info_hash) {
            self.table_hits += 1;
            table.touch(node_n);
            return;
        }
        self.table_misses += 1;
        let node = &mut self.nodes[at as usize];
        let n_self = node.actions.len().max(1);
        let mut table = EnemyTable::create(info_hash, actions, prior, n_self);
        if !node.actions.is_empty() {
            table.ensure_self_rows(node.actions.len());
        }
        table.touch(node_n);
        node.enemy_tables.push(table);

        let max_tables = self.max_enemy_tables;
        loop {
            let evicted = self.nodes[at as usize].retention_evict(max_tables, &info_hash);
            match evicted {
                Some(table) => self.eviction_loss += table.eviction_loss(),
                None => break,
            }
        }
    }

    pub fn pin_enemy(&mut self, at: u32, info_hash: Digest) {
        let node = &mut self.nodes[at as usize];
        if !node.pending_pins.contains(&info_hash) {
            node.pending_pins.push(info_hash);
        }
    }

    pub fn clear_pins(&mut self, at: u32) {
        self.nodes[at as usize].pending_pins.clear();
    }

    /// Fully back up one simulation step at a node, in the root perspective.
    pub fn backup_node(
        &mut self,
        at: u32,
        h_star: &Digest,
        a_idx: usize,
        b_idx: usize,
        leaf_value: f64,
    ) {
        let n_self = self.nodes[at as usize].actions.len();
        let node_n = self.nodes[at as usize].n;
        {
            let node = &mut self.nodes[at as usize];
            let table = node
                .table_mut(h_star)
                .expect("backup on a node without the sampled enemy table");
            table.ensure_self_rows(n_self);
            let n_enemy = table.n_enemy();
            apply_joint_backup(
                &mut table.visits,
                &mut table.value_sum,
                &mut table.q,
                n_enemy,
                a_idx,
                b_idx,
                leaf_value,
            );
            table.touch(node_n);
        }
        self.total_joint_visits += 1.0;

        let network_value = self.nodes[at as usize].network_value;
        let sigma_self = {
            let node = &self.nodes[at as usize];
            mixed_strategy(&node.regret, &node.prior, node.n)
        };
        let (hashes, weights) = self.enemy_weights(at);

        let mut enemy_sigmas: Vec<Vec<f64>> = Vec::with_capacity(hashes.len());
        let mut q_eff_list: Vec<Vec<f64>> = Vec::with_capacity(hashes.len());
        for h in &hashes {
            let node = &mut self.nodes[at as usize];
            let node_n = node.n;
            let table = node.table_mut(h).expect("hash came from this node");
            table.ensure_self_rows(n_self);
            enemy_sigmas.push(mixed_strategy(&table.regret, &table.prior, node_n));
            q_eff_list.push(effective_q(&table.visits, &table.q, network_value));
        }
        let (u_self, v) =
            aggregate_self_utilities(&sigma_self, &weights, &enemy_sigmas, &q_eff_list);

        // Enemy utilities on the sampled hash only.
        let node_n = self.nodes[at as usize].n;
        let (u_enemy, sigma_b_star) = {
            let node = &self.nodes[at as usize];
            let table = node.table(h_star).expect("sampled table still installed");
            let q_star = effective_q(&table.visits, &table.q, network_value);
            let sigma_b_star = mixed_strategy(&table.regret, &table.prior, node_n);
            let (_, u_enemy, _) = matrix_utilities(&sigma_self, &sigma_b_star, &q_star);
            (u_enemy, sigma_b_star)
        };

        let node = &mut self.nodes[at as usize];
        node.regret = regret_plus_update(&node.regret, &u_self, v, true);
        accumulate_average_strategy(&mut node.avg_strategy, &sigma_self);
        let table = node.table_mut(h_star).expect("sampled table still installed");
        table.regret = regret_plus_update(&table.regret, &u_enemy, v, false);
        accumulate_average_strategy(&mut table.avg_strategy, &sigma_b_star);
        node.n += 1;
    }

    /// Marginal self visits at the root, summed over enemy tables.
    pub fn root_marginal_visits(&self) -> Vec<f64> {
        let root = match self.root {
            Some(root) => self.node(root),
            None => return Vec::new(),
        };
        let n_self = root.actions.len();
        let mut visits = vec![0.0f64; n_self];
        for table in &root.enemy_tables {
            if table.n_self != n_self {
                continue;
            }
            let n_b = table.n_enemy();
            for (i, slot) in visits.iter_mut().enumerate() {
                *slot += npsum(&table.visits[i * n_b..(i + 1) * n_b]);
            }
        }
        visits
    }

    pub fn root_action_index(&self) -> Result<usize, String> {
        let root = match self.root {
            Some(root) => self.node(root),
            None => return Err("empty root".to_string()),
        };
        if root.actions.is_empty() {
            return Err("empty root".to_string());
        }
        let visits = self.root_marginal_visits();
        Ok(select_root_action(
            &root.avg_strategy,
            &visits,
            &root.prior,
            None,
        ))
    }

    pub fn eviction_loss_rate(&self) -> f64 {
        self.eviction_loss / self.total_joint_visits.max(1.0)
    }

    pub fn table_hit_rate(&self) -> f64 {
        let total = self.table_hits + self.table_misses;
        if total == 0 {
            return 0.0;
        }
        self.table_hits as f64 / total as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: u8) -> Digest {
        [byte; 32]
    }

    fn tree() -> SearchTree {
        SearchTree::new(MAX_TREE_NODES, MAX_ENEMY_TABLES, 0)
    }

    fn node(tree: &mut SearchTree, byte: u8) -> u32 {
        tree.make_node(
            digest(byte),
            0,
            digest(byte),
            vec![byte],
            [0; 32],
            0.0,
            8,
        )
        .unwrap()
    }

    #[test]
    fn a_repeated_key_returns_the_same_handle() {
        let mut tree = tree();
        let a = node(&mut tree, 1);
        let b = node(&mut tree, 1);
        assert_eq!(a, b);
        assert_eq!(tree.nodes.len(), 1);
    }

    #[test]
    fn the_node_cap_refuses_rather_than_panics() {
        let mut tree = SearchTree::new(2, MAX_ENEMY_TABLES, 0);
        node(&mut tree, 1);
        node(&mut tree, 2);
        assert!(tree
            .make_node(digest(3), 0, digest(3), vec![3], [0; 32], 0.0, 8)
            .is_err());
    }

    #[test]
    fn widening_self_renormalizes_and_pads_every_table() {
        let mut tree = tree();
        let at = node(&mut tree, 1);
        tree.node_mut(at).widen_self(5, 1.0);
        tree.get_or_create_enemy_table(at, digest(9), &[3, 4], &[0.5, 0.5]);
        assert_eq!(tree.node(at).table(&digest(9)).unwrap().n_self, 1);
        tree.node_mut(at).widen_self(6, 3.0);
        let table = tree.node(at).table(&digest(9)).unwrap();
        assert_eq!(table.n_self, 2);
        assert_eq!(table.visits.len(), 4);
        let prior = &tree.node(at).prior;
        assert!((prior[0] + prior[1] - 1.0).abs() < 1e-15);
        assert!((prior[1] - 0.75).abs() < 1e-15);
    }

    #[test]
    fn widening_an_enemy_column_keeps_the_existing_statistics() {
        let mut table = EnemyTable::create(digest(1), &[7, 8], &[1.0, 1.0], 2);
        table.visits[0] = 3.0; // (self 0, enemy 0)
        table.visits[3] = 5.0; // (self 1, enemy 1)
        table.widen_enemy(9, 2.0);
        assert_eq!(table.n_enemy(), 3);
        assert_eq!(table.visits.len(), 6);
        assert_eq!(table.visits[0], 3.0);
        assert_eq!(table.visits[4], 5.0);
        assert_eq!(table.visits[2], 0.0);
    }

    #[test]
    fn eviction_takes_the_lowest_retention_score_and_never_the_protected_one() {
        let mut tree = SearchTree::new(MAX_TREE_NODES, 1, 0);
        let at = node(&mut tree, 1);
        tree.get_or_create_enemy_table(at, digest(10), &[1], &[1.0]);
        tree.node_mut(at).n = 50;
        tree.get_or_create_enemy_table(at, digest(11), &[1], &[1.0]);
        // The first table was touched at N = 0, the second at N = 50, and the
        // second is the protected one, so the first goes.
        assert_eq!(tree.node(at).enemy_tables.len(), 1);
        assert!(tree.node(at).table(&digest(11)).is_some());
    }

    #[test]
    fn a_pinned_table_survives_eviction_pressure() {
        let mut tree = SearchTree::new(MAX_TREE_NODES, 1, 0);
        let at = node(&mut tree, 1);
        tree.get_or_create_enemy_table(at, digest(10), &[1], &[1.0]);
        tree.pin_enemy(at, digest(10));
        tree.node_mut(at).n = 50;
        tree.get_or_create_enemy_table(at, digest(11), &[1], &[1.0]);
        assert_eq!(tree.node(at).enemy_tables.len(), 2);
    }
}
