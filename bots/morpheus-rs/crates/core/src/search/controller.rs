//! Building the tree, rooting it, and filling in enemy priors.
//!
//! Everything that *adds* to the tree lives here: the root evaluation and the
//! reuse-or-reset decision on a new observation, self-candidate expansion,
//! and the batched enemy-prior materialisation that selection is forbidden to
//! do for itself. The cross-turn LRU is here too — without it every turn
//! re-ran the same enemy-prior forwards, measured at ~36 ms/turn.
//!
//! The helpers `select` and `backup` need are `pub(super)`: sibling modules do
//! not see each other's private items, and `pub(super)` is the narrowest thing
//! that lets them through without widening the crate's API surface.

use crate::belief::{BeliefState, Particle};
use crate::board::action::{legal_mask, live_build_cost, N_ACTIONS};
use crate::board::hashing::{
    child_edge_key, enemy_info_hash_prehashed, info_state_key_prehashed, memory_digest,
    observation_payload, roll_history_digest,
};
use crate::board::memory::{update_memory, VisibleMemory};
use crate::board::observe::emit_observation;
use crate::belief::Action5;
use crate::io::wire::Observation;
use crate::search::matrix::{enemy_widening_limit, self_widening_limit};
use crate::search::tree::{Digest, SearchTree};
use crate::search::{
    EnemyPriorRequest, EvalItem, SearchConfig, SearchController, SearchEvaluator, ZERO_DIGEST,
};
use crate::support::rng::SharedRng;
use crate::tactics::{mandatory_action_indices, play_mask, policy_ordered_candidates};

impl SearchController {
    pub fn new(seat: usize, config: SearchConfig, rng: SharedRng) -> Self {
        Self {
            seat,
            config,
            tree: SearchTree::new(config.max_nodes, config.max_enemy_tables, seat),
            history_digest: ZERO_DIGEST,
            memory: None,
            rng,
            last_root_prior: None,
            last_select_enemy_prior_forwards: 0,
            enemy_prior_cache: Vec::new(),
            enemy_prior_cache_hits: 0,
            enemy_prior_cache_misses: 0,
        }
    }

    pub(super) fn cached_enemy_prior(&mut self, info_hash: &Digest) -> Option<Vec<f64>> {
        let at = self
            .enemy_prior_cache
            .iter()
            .position(|(key, _)| key == info_hash)?;
        let entry = self.enemy_prior_cache.remove(at);
        let prior: Vec<f64> = entry.1.iter().map(|&v| v as f64).collect();
        self.enemy_prior_cache.push(entry);
        self.enemy_prior_cache_hits += 1;
        Some(prior)
    }

    fn store_enemy_prior(&mut self, info_hash: Digest, prior: &[f64]) {
        if let Some(at) = self
            .enemy_prior_cache
            .iter()
            .position(|(key, _)| key == &info_hash)
        {
            self.enemy_prior_cache.remove(at);
        }
        self.enemy_prior_cache
            .push((info_hash, prior.iter().map(|&v| v as f32).collect()));
        while self.enemy_prior_cache.len() > self.config.enemy_prior_cache_size {
            self.enemy_prior_cache.remove(0);
        }
    }

    // ------------------------------------------------------------------ root

    pub fn ensure_root<E: SearchEvaluator + ?Sized>(
        &mut self,
        evaluator: &mut E,
        obs: &Observation,
        memory: &VisibleMemory,
        belief: &BeliefState,
    ) -> u32 {
        self.memory = Some(memory.clone());
        let obs_payload = observation_payload(obs);
        let mem_d = memory_digest(memory);
        let key = info_state_key_prehashed(obs.turn, &mem_d, &obs_payload, &self.history_digest);
        let (prior, value) = evaluator.evaluate(obs, memory, belief, true, true);
        self.last_root_prior = Some(prior.clone());
        let at = self
            .tree
            .make_node(
                key,
                obs.turn,
                mem_d,
                obs_payload,
                self.history_digest,
                value,
                self.config.n_particles,
            )
            .expect("the root always fits: a fresh tree has room");
        if !self.tree.node(at).expanded {
            self.expand_self_candidates(at, obs, memory, &prior);
            let mut rng = self.rng.handle();
            self.tree
                .node_mut(at)
                .reservoir
                .replace_from_belief(belief, &mut rng);
            let node = self.tree.node_mut(at);
            node.expanded = true;
            node.network_value = value;
        }
        self.tree.set_root(at);
        at
    }

    /// Follow the child for `sent_action` plus the new observation, or restart.
    pub fn reuse_or_reset<E: SearchEvaluator + ?Sized>(
        &mut self,
        evaluator: &mut E,
        sent_action: Action5,
        obs: &Observation,
        memory: &VisibleMemory,
        belief: &BeliefState,
    ) -> u32 {
        let obs_payload = observation_payload(obs);
        self.history_digest = roll_history_digest(&self.history_digest, sent_action, &obs_payload);
        let edge = child_edge_key(sent_action, &obs_payload);
        let child = self
            .tree
            .root
            .and_then(|root| self.tree.node(root).children.get(&edge).copied());
        let mem_d = memory_digest(memory);

        if let Some(child) = child {
            if self.tree.matches_info(child, obs.turn, &mem_d, &obs_payload) {
                let mut rng = self.rng.handle();
                self.tree
                    .node_mut(child)
                    .reservoir
                    .replace_from_belief(belief, &mut rng);
                let (prior_e, value) = evaluator.evaluate(obs, memory, belief, true, true);
                self.tree.node_mut(child).network_value = value;
                self.last_root_prior = Some(prior_e.clone());
                // Always refresh masses and widen from the live network prior.
                // Skipping when the child already had candidates left pass-only
                // priors stuck after early turns where only pass was legal.
                self.expand_self_candidates(child, obs, memory, &prior_e);
                self.memory = Some(memory.clone());
                self.tree.set_root(child);
                return child;
            }
        }

        // Exact match failed — new root, and the history restarts from this
        // observation alone.
        self.tree.clear();
        self.history_digest = roll_history_digest(&ZERO_DIGEST, sent_action, &obs_payload);
        self.ensure_root(evaluator, obs, memory, belief)
    }

    pub(super) fn expand_self_candidates(
        &mut self,
        at: u32,
        obs: &Observation,
        memory: &VisibleMemory,
        prior: &[f64],
    ) {
        let cost_grid = live_build_cost(obs, memory);
        let mask = play_mask(obs, memory, Some(&cost_grid));
        let limit = self_widening_limit(self.tree.node(at).n);
        // Cache the full network prior so later widening does not rebuild mass
        // only from already-accepted candidates, which zeroes new expands.
        let cached: Vec<f64> = prior.to_vec();
        {
            let node = self.tree.node_mut(at);
            node.cached_policy_prior = Some(cached.clone());
            if !node.actions.is_empty() {
                node.refresh_self_priors(&cached);
            }
        }
        let mandatory = mandatory_action_indices(obs, memory, &mask);
        let candidates = policy_ordered_candidates(prior, &mask, &mandatory, limit);
        let node = self.tree.node_mut(at);
        for index in candidates {
            let mass = prior.get(index).copied().unwrap_or(0.0);
            node.widen_self(index, mass);
        }
        // Re-normalize from the live prior after any new append, so early
        // pass-only mass cannot dominate newly added expands.
        if !node.actions.is_empty() {
            node.refresh_self_priors(&cached);
        }
    }

    pub(super) fn enemy_view(&self, particle: &Particle) -> (Observation, VisibleMemory, Digest) {
        let enemy_seat = 1 - self.seat;
        let enemy_obs = emit_observation(&particle.state, enemy_seat);
        let enemy_mem = update_memory(&particle.enemy_memory, &enemy_obs);
        let h = enemy_info_hash_prehashed(&observation_payload(&enemy_obs), &memory_digest(&enemy_mem));
        (enemy_obs, enemy_mem, h)
    }

    /// Create an enemy table from an already-evaluated prior. No network.
    pub(super) fn install_enemy_table(
        &mut self,
        at: u32,
        info_hash: Digest,
        enemy_obs: &Observation,
        enemy_mem: &VisibleMemory,
        prior_e: &[f64],
    ) {
        let cost_grid = live_build_cost(enemy_obs, enemy_mem);
        let mask = legal_mask(enemy_obs, enemy_mem, Some(&cost_grid));
        let limit = enemy_widening_limit(self.tree.node(at).n);
        let mandatory = mandatory_action_indices(enemy_obs, enemy_mem, &mask);
        let candidates = policy_ordered_candidates(prior_e, &mask, &mandatory, limit);
        let priors: Vec<f64> = candidates
            .iter()
            .map(|&i| prior_e.get(i).copied().unwrap_or(0.0))
            .collect();
        self.tree
            .get_or_create_enemy_table(at, info_hash, &candidates, &priors);
    }

    /// Batch-evaluate missing enemy priors and install their tables.
    ///
    /// Returns the number of network forwards. Selection must never call this.
    pub fn materialize_enemy_priors<E: SearchEvaluator + ?Sized>(
        &mut self,
        evaluator: &mut E,
        requests: Vec<EnemyPriorRequest>,
        belief: &BeliefState,
    ) -> usize {
        self.last_select_enemy_prior_forwards = 0;
        if requests.is_empty() {
            return 0;
        }
        let enemy_seat = 1 - self.seat;
        let mut ordered: Vec<EnemyPriorRequest> = Vec::new();
        for req in requests {
            let at = req.node();
            if self.tree.node(at).table(&req.info_hash).is_some() {
                continue;
            }
            if ordered
                .iter()
                .any(|seen| seen.node() == at && seen.info_hash == req.info_hash)
            {
                continue;
            }
            ordered.push(req);
        }
        if ordered.is_empty() {
            return 0;
        }

        let items: Vec<EvalItem> = ordered
            .iter()
            .map(|req| EvalItem {
                obs: req.enemy_obs.clone(),
                memory: req.enemy_mem.clone(),
                belief: BeliefState {
                    seat: enemy_seat,
                    particles: vec![req.particle.clone()],
                    config: belief.config,
                    collapsed: false,
                },
                from_root: false,
                shape: false,
            })
            .collect();
        let priors = evaluator.policy_priors_many(&items);
        self.last_select_enemy_prior_forwards = priors.len();

        for (req, prior) in ordered.iter().zip(&priors) {
            let at = req.node();
            self.store_enemy_prior(req.info_hash, prior);
            self.enemy_prior_cache_misses += 1;
            if self.tree.node(at).table(&req.info_hash).is_some() {
                continue;
            }
            self.install_enemy_table(at, req.info_hash, &req.enemy_obs, &req.enemy_mem, prior);
        }
        self.last_select_enemy_prior_forwards
    }

    pub(super) fn widen_enemy_if_needed(
        &mut self,
        at: u32,
        info_hash: &Digest,
        enemy_obs: &Observation,
        enemy_mem: &VisibleMemory,
    ) {
        let limit = enemy_widening_limit(self.tree.node(at).n);
        {
            let table = match self.tree.node(at).table(info_hash) {
                Some(table) => table,
                None => return,
            };
            if table.actions.len() >= limit {
                return;
            }
        }
        let cost_grid = live_build_cost(enemy_obs, enemy_mem);
        let mask = legal_mask(enemy_obs, enemy_mem, Some(&cost_grid));
        let mandatory = mandatory_action_indices(enemy_obs, enemy_mem, &mask);
        // Rebuild a full-length prior vector for ordering. The Python has a
        // dead `enemy_prior` branch here that no caller reaches; the live path
        // is always this reconstruction from the table's own masses.
        let mut full = vec![0.0f64; N_ACTIONS];
        {
            let table = self.tree.node(at).table(info_hash).unwrap();
            for (i, &action) in table.actions.iter().enumerate() {
                if i < table.prior.len() {
                    full[action] = table.prior[i];
                }
            }
        }
        let candidates = policy_ordered_candidates(&full, &mask, &mandatory, limit);
        let table = self.tree.node_mut(at).table_mut(info_hash).unwrap();
        for index in candidates {
            if table.actions.contains(&index) {
                continue;
            }
            table.widen_enemy(index, full[index]);
            if table.actions.len() >= limit {
                break;
            }
        }
    }
}
