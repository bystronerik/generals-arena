//! Root-sampled information-set MCTS with simultaneous matrices.
//!
//! Port of `bots/morpheus/search.py`. Selection performs **zero network
//! forwards**: when an enemy table's prior is missing it returns an
//! [`EnemyPriorRequest`] so the caller can batch the forwards and resume,
//! which is what lets the runtime charge inference to one admitted component
//! instead of scattering it through the tree walk.
//!
//! M0's baseline made two of these numbers worth restating: `selection`
//! measured 9.6× its shipped p99 on the M3 Pro and 13.5× on x86, and `backup`
//! 2.9×/4.7×. rewrite-plan §7 filed both as "small absolute numbers"; they are
//! not, and this port treats them as first-class rather than as the tail of the
//! search work.

use std::rc::Rc;

use crate::board::action::{legal_mask, live_build_cost, N_ACTIONS};
use crate::belief::{Action5, BeliefConfig, BeliefState, Particle, PASS_ACTION};
use crate::board::hashing::{
    child_edge_key, enemy_info_hash_prehashed, info_state_key_prehashed, memory_digest,
    observation_payload, roll_history_digest,
};
use crate::matrix::{enemy_widening_limit, mixed_strategy, sample_index, self_widening_limit};
use crate::board::memory::{update_memory, VisibleMemory};
use crate::board::observe::emit_observation;
use crate::support::rng::SharedRng;
use crate::board::state::GameState;
use crate::tactics::{mandatory_action_indices, play_mask, policy_ordered_candidates};
use crate::board::transition::{transition, Actions};
use crate::tree::{Digest, SearchTree};

pub const SEARCH_DEPTH: usize = 16;
pub const PENDING_LEAF_BATCH: usize = 4;
pub const ZERO_DIGEST: Digest = [0u8; 32];

/// One `(obs, memory, belief)` triple the evaluator is asked about.
pub struct EvalItem {
    pub obs: Observation,
    pub memory: VisibleMemory,
    pub belief: BeliefState,
    /// The *perspective* flag: the returned value is already root-perspective.
    pub from_root: bool,
    /// The *root-prior shaping* flag. Only the real root evaluation sets it —
    /// conflating the two made every leaf pay the heuristic blend.
    pub shape: bool,
}

use crate::io::wire::Observation;

/// Injectable policy/value interface for the search.
pub trait SearchEvaluator {
    fn evaluate(
        &mut self,
        obs: &Observation,
        memory: &VisibleMemory,
        belief: &BeliefState,
        from_root: bool,
        shape: bool,
    ) -> (Vec<f64>, f64);

    /// Batched evaluation. The default loops, as the Python's fallback does.
    fn evaluate_many(&mut self, items: &[EvalItem]) -> Vec<(Vec<f64>, f64)> {
        items
            .iter()
            .map(|item| {
                self.evaluate(
                    &item.obs,
                    &item.memory,
                    &item.belief,
                    item.from_root,
                    item.shape,
                )
            })
            .collect()
    }

    /// Policy-only priors for enemy tables. Enemy tables discard the value, so
    /// a network evaluator skips the WDL head here.
    fn policy_priors_many(&mut self, items: &[EvalItem]) -> Vec<Vec<f64>> {
        items
            .iter()
            .map(|item| {
                self.evaluate(&item.obs, &item.memory, &item.belief, false, false)
                    .0
            })
            .collect()
    }

    fn set_previous_action(&mut self, _action: Option<Action5>) {}

    /// The evaluator itself, viewed as a belief-proposal policy.
    ///
    /// The Python resolves this with `getattr(evaluator, "policy_logits")` and
    /// hands the *same object* to both roles, which Rust cannot express as two
    /// mutable borrows — so the role is a method instead. `None` means the
    /// belief advances on uniform legal enemy actions, which is what
    /// `deployment.json` ships.
    fn as_proposal_policy(&mut self) -> Option<&mut (dyn crate::proposal::ProposalPolicy + '_)> {
        None
    }

    /// The raw legal-normalized network prior from the last root evaluation,
    /// before the shaping blend — the "who's deciding" probe reads this.
    fn last_unshaped_prior(&self) -> Option<&[f64]> {
        None
    }

    fn clear_last_unshaped_prior(&mut self) {}
}

#[derive(Clone, Copy)]
pub struct SearchConfig {
    pub depth: usize,
    pub pending_batch: usize,
    pub max_nodes: usize,
    pub max_enemy_tables: usize,
    pub n_particles: usize,
    /// Cross-turn LRU of enemy priors keyed by enemy info hash. The tree is
    /// rebuilt on most turns, so without this every turn re-runs the same
    /// enemy-prior forwards (~36 ms/turn measured).
    pub enemy_prior_cache_size: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            depth: SEARCH_DEPTH,
            pending_batch: PENDING_LEAF_BATCH,
            max_nodes: 4096,
            max_enemy_tables: 8,
            n_particles: 64,
            enemy_prior_cache_size: 512,
        }
    }
}

/// One selected simulation path awaiting leaf evaluation and backup.
pub struct PendingPath {
    pub nodes: Vec<u32>,
    /// `(enemy info hash, self index, enemy index, child edge key)`.
    pub edges: Vec<(Digest, usize, usize, Digest)>,
    pub particle: Particle,
    pub leaf_state: Rc<GameState>,
    pub leaf_node: Option<u32>,
    pub needs_expand: bool,
    pub terminal_value: Option<f64>,
}

/// Selection paused until an enemy-table prior is materialised.
pub struct EnemyPriorRequest {
    pub nodes: Vec<u32>,
    pub edges: Vec<(Digest, usize, usize, Digest)>,
    pub particle: Particle,
    pub state: Rc<GameState>,
    pub depth: usize,
    pub freeze_widening: bool,
    pub info_hash: Digest,
    pub enemy_obs: Observation,
    pub enemy_mem: VisibleMemory,
}

impl EnemyPriorRequest {
    pub fn node(&self) -> u32 {
        *self.nodes.last().expect("a request always has its node")
    }
}

pub enum Selection {
    Path(Box<PendingPath>),
    NeedsPrior(Box<EnemyPriorRequest>),
}

/// Owns the tree and runs batched simulations.
pub struct SearchController {
    pub seat: usize,
    pub config: SearchConfig,
    pub tree: SearchTree,
    pub history_digest: Digest,
    pub memory: Option<VisibleMemory>,
    pub rng: SharedRng,
    pub last_root_prior: Option<Vec<f64>>,
    /// Enemy-prior network time, charged by `materialize_enemy_priors` only.
    pub last_select_enemy_prior_forwards: usize,
    /// Content-addressed LRU, oldest first. The hash covers the enemy
    /// observation and memory, so entries never go stale — they only cost
    /// memory, and they deliberately survive `tree.clear()`.
    enemy_prior_cache: Vec<(Digest, Vec<f32>)>,
    pub enemy_prior_cache_hits: u64,
    pub enemy_prior_cache_misses: u64,
}

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

    fn cached_enemy_prior(&mut self, info_hash: &Digest) -> Option<Vec<f64>> {
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

    fn expand_self_candidates(
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

    fn enemy_view(&self, particle: &Particle) -> (Observation, VisibleMemory, Digest) {
        let enemy_seat = 1 - self.seat;
        let enemy_obs = emit_observation(&particle.state, enemy_seat);
        let enemy_mem = update_memory(&particle.enemy_memory, &enemy_obs);
        let h = enemy_info_hash_prehashed(&observation_payload(&enemy_obs), &memory_digest(&enemy_mem));
        (enemy_obs, enemy_mem, h)
    }

    /// Create an enemy table from an already-evaluated prior. No network.
    fn install_enemy_table(
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

    fn widen_enemy_if_needed(
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

    /// Select one simulation path from a frozen statistics snapshot.
    ///
    /// Zero network calls. A missing enemy prior returns
    /// [`Selection::NeedsPrior`]; an unexpanded node stops for leaf evaluation.
    /// With `freeze_widening`, progressive widening is skipped so the matrix
    /// width stays fixed (the runtime forecasts fewer than 16 simulations).
    pub fn select_path(
        &mut self,
        mut freeze_widening: bool,
        resume: Option<EnemyPriorRequest>,
    ) -> Selection {
        let root = self.tree.root.expect("select_path before a root exists");
        let memory = self
            .memory
            .clone()
            .expect("select_path before the root memory was set");

        let (mut node, mut particle, mut state, mut nodes, mut edges, mut depth) = match resume {
            Some(resume) => {
                freeze_widening = resume.freeze_widening;
                (
                    resume.node(),
                    resume.particle,
                    resume.state,
                    resume.nodes,
                    resume.edges,
                    resume.depth,
                )
            }
            None => {
                let mut rng = self.rng.handle();
                let particle = self
                    .tree
                    .node(root)
                    .reservoir
                    .sample(&mut rng)
                    .expect("the root reservoir is filled by ensure_root")
                    .clone();
                let state = Rc::clone(&particle.state);
                (root, particle, state, vec![root], Vec::new(), 0)
            }
        };

        while depth < self.config.depth {
            if state.winner >= 0 {
                let seat_win = if state.winner as usize == self.seat {
                    1.0
                } else {
                    -1.0
                };
                return Selection::Path(Box::new(PendingPath {
                    nodes,
                    edges,
                    particle,
                    leaf_state: state,
                    leaf_node: Some(node),
                    needs_expand: false,
                    terminal_value: Some(seat_win),
                }));
            }

            // Unexpanded node: stop for leaf evaluation, still without a
            // network call.
            if self.tree.node(node).actions.is_empty() {
                return Selection::Path(Box::new(PendingPath {
                    nodes,
                    edges,
                    particle,
                    leaf_state: state,
                    leaf_node: Some(node),
                    needs_expand: true,
                    terminal_value: None,
                }));
            }

            let my_obs = emit_observation(&state, self.seat);
            let my_mem = if node == root {
                memory.clone()
            } else {
                update_memory(&memory, &my_obs)
            };

            let (enemy_obs, enemy_mem, h) = self.enemy_view(&particle);
            if self.tree.node(node).table(&h).is_none() {
                // Cross-turn cache first: installing from a cached prior needs
                // no network, so selection keeps its zero-forward invariant.
                if let Some(cached) = self.cached_enemy_prior(&h) {
                    self.install_enemy_table(node, h, &enemy_obs, &enemy_mem, &cached);
                }
            }
            if self.tree.node(node).table(&h).is_none() {
                // Pin before materialisation so a later install cannot evict it.
                self.tree.pin_enemy(node, h);
                return Selection::NeedsPrior(Box::new(EnemyPriorRequest {
                    nodes,
                    edges,
                    particle,
                    state,
                    depth,
                    freeze_widening,
                    info_hash: h,
                    enemy_obs,
                    enemy_mem,
                }));
            }
            {
                // Re-fetching through the tree records the hit and touches LRU.
                let (actions, prior) = {
                    let table = self.tree.node(node).table(&h).unwrap();
                    (table.actions.clone(), table.prior.clone())
                };
                self.tree
                    .get_or_create_enemy_table(node, h, &actions, &prior);
            }
            self.tree.pin_enemy(node, h);

            if !freeze_widening {
                // Prefer a full-length network prior: rebuilding from
                // `node.prior` alone assigns mass 0 to newly legal expands,
                // which is the pass lock.
                let prior_for_self: Vec<f64> = if node == root && self.last_root_prior.is_some() {
                    self.last_root_prior.clone().unwrap()
                } else if let Some(cached) = self.tree.node(node).cached_policy_prior.clone() {
                    cached
                } else {
                    let node_ref = self.tree.node(node);
                    let mut full = vec![0.0f64; N_ACTIONS];
                    for (i, &action) in node_ref.actions.iter().enumerate() {
                        if i < node_ref.prior.len() {
                            full[action] = node_ref.prior[i];
                        }
                    }
                    full
                };
                if self.tree.node(node).actions.len() < self_widening_limit(self.tree.node(node).n)
                {
                    self.expand_self_candidates(node, &my_obs, &my_mem, &prior_for_self);
                }
                self.widen_enemy_if_needed(node, &h, &enemy_obs, &enemy_mem);
            }

            let (sigma_a, sigma_b, node_actions, table_actions) = {
                let node_ref = self.tree.node(node);
                let table = node_ref.table(&h).unwrap();
                (
                    mixed_strategy(&node_ref.regret, &node_ref.prior, node_ref.n),
                    mixed_strategy(&table.regret, &table.prior, node_ref.n),
                    node_ref.actions.clone(),
                    table.actions.clone(),
                )
            };
            let (a_idx, b_idx) = {
                let mut rng = self.rng.handle();
                (
                    sample_index(&sigma_a, &mut rng),
                    sample_index(&sigma_b, &mut rng),
                )
            };
            let a = crate::board::action::decode_action(node_actions[a_idx]).unwrap_or(PASS_ACTION);
            let b = crate::board::action::decode_action(table_actions[b_idx]).unwrap_or(PASS_ACTION);

            let mut actions: Actions = [PASS_ACTION; 2];
            actions[self.seat] = a;
            actions[1 - self.seat] = b;
            let (next_state, info) = transition(&state, &actions);
            let next_state = Rc::new(next_state);

            let enemy_seat = 1 - self.seat;
            // One observation pair per depth step; reused for the hash, the
            // memory fold and the edge key.
            let child_obs = emit_observation(&next_state, self.seat);
            let child_payload = observation_payload(&child_obs);
            let enemy_next_obs = emit_observation(&next_state, enemy_seat);
            let enemy_next_mem = update_memory(&particle.enemy_memory, &enemy_next_obs);

            if info.is_done || next_state.winner >= 0 {
                let term = if next_state.winner < 0 {
                    0.0
                } else if next_state.winner as usize == self.seat {
                    1.0
                } else {
                    -1.0
                };
                let edge = child_edge_key(a, &child_payload);
                edges.push((h, a_idx, b_idx, edge));
                return Selection::Path(Box::new(PendingPath {
                    nodes,
                    edges,
                    particle: Particle {
                        state: Rc::clone(&next_state),
                        weight: particle.weight,
                        enemy_memory: Rc::new(enemy_next_mem),
                        enemy_prev_action: Some(b),
                        history: particle.history.clone(),
                    },
                    leaf_state: next_state,
                    leaf_node: None,
                    needs_expand: false,
                    terminal_value: Some(term),
                }));
            }

            let edge = child_edge_key(a, &child_payload);
            edges.push((h, a_idx, b_idx, edge));

            let existing = self.tree.node(node).children.get(&edge).copied();
            let child = match existing {
                Some(child) => child,
                None => {
                    let child_mem = update_memory(&my_mem, &child_obs);
                    let child_mem_d = memory_digest(&child_mem);
                    let child_hist = roll_history_digest(
                        &self.tree.node(node).history_digest,
                        a,
                        &child_payload,
                    );
                    let child_key = info_state_key_prehashed(
                        child_obs.turn,
                        &child_mem_d,
                        &child_payload,
                        &child_hist,
                    );
                    let made = self.tree.make_node(
                        child_key,
                        child_obs.turn,
                        child_mem_d,
                        child_payload.clone(),
                        child_hist,
                        0.0,
                        self.config.n_particles,
                    );
                    let child = match made {
                        Ok(child) => child,
                        Err(()) => {
                            // The node cap is a refusal, not a crash: the path
                            // becomes a leaf evaluation at the parent.
                            return Selection::Path(Box::new(PendingPath {
                                nodes,
                                edges,
                                particle,
                                leaf_state: next_state,
                                leaf_node: Some(node),
                                needs_expand: true,
                                terminal_value: None,
                            }));
                        }
                    };
                    self.tree.node_mut(node).children.insert(edge, child);
                    let arriving = Particle {
                        state: Rc::clone(&next_state),
                        weight: 1.0,
                        enemy_memory: Rc::new(enemy_next_mem),
                        enemy_prev_action: Some(b),
                        history: particle.history.clone(),
                    };
                    let mut rng = self.rng.handle();
                    self.tree
                        .node_mut(child)
                        .reservoir
                        .admit(&arriving, &mut rng);
                    nodes.push(child);
                    // A new child is unexpanded: stop for leaf evaluation.
                    return Selection::Path(Box::new(PendingPath {
                        nodes,
                        edges,
                        particle: arriving,
                        leaf_state: next_state,
                        leaf_node: Some(child),
                        needs_expand: true,
                        terminal_value: None,
                    }));
                }
            };

            let arriving = Particle {
                state: Rc::clone(&next_state),
                weight: particle.weight,
                enemy_memory: Rc::new(enemy_next_mem),
                enemy_prev_action: Some(b),
                history: particle.history.clone(),
            };
            let mut rng = self.rng.handle();
            self.tree
                .node_mut(child)
                .reservoir
                .admit(&arriving, &mut rng);
            node = child;
            particle = arriving;
            state = next_state;
            nodes.push(node);
            depth += 1;
        }

        Selection::Path(Box::new(PendingPath {
            nodes,
            edges,
            particle,
            leaf_state: state,
            leaf_node: Some(node),
            needs_expand: false,
            terminal_value: None,
        }))
    }

    /// Select one path, materialising enemy priors as needed.
    pub fn complete_select_path<E: SearchEvaluator + ?Sized>(
        &mut self,
        evaluator: &mut E,
        belief: &BeliefState,
        freeze_widening: bool,
    ) -> PendingPath {
        let mut outcome = self.select_path(freeze_widening, None);
        loop {
            match outcome {
                Selection::Path(path) => return *path,
                Selection::NeedsPrior(request) => {
                    let request = *request;
                    let resume = EnemyPriorRequest {
                        nodes: request.nodes.clone(),
                        edges: request.edges.clone(),
                        particle: request.particle.clone(),
                        state: Rc::clone(&request.state),
                        depth: request.depth,
                        freeze_widening: request.freeze_widening,
                        info_hash: request.info_hash,
                        enemy_obs: request.enemy_obs.clone(),
                        enemy_mem: request.enemy_mem.clone(),
                    };
                    self.materialize_enemy_priors(evaluator, vec![request], belief);
                    outcome = self.select_path(freeze_widening, Some(resume));
                }
            }
        }
    }

    /// Evaluate many leaves, batching network forwards when the evaluator can.
    pub fn evaluate_leaves<E: SearchEvaluator + ?Sized>(
        &mut self,
        evaluator: &mut E,
        paths: &mut [PendingPath],
        belief: &BeliefState,
    ) -> Vec<f64> {
        let mut values: Vec<Option<f64>> = vec![None; paths.len()];
        let mut pending_idx: Vec<usize> = Vec::new();
        let mut items: Vec<EvalItem> = Vec::new();

        for (i, path) in paths.iter().enumerate() {
            if let Some(terminal) = path.terminal_value {
                values[i] = Some(terminal);
                continue;
            }
            let winner = path.leaf_state.winner;
            if winner >= 0 {
                values[i] = Some(if winner as usize == self.seat { 1.0 } else { -1.0 });
                continue;
            }
            let obs = emit_observation(&path.leaf_state, self.seat);
            let mem = match &self.memory {
                Some(memory) => update_memory(memory, &obs),
                None => unreachable!("evaluate_leaves before the root memory was set"),
            };
            pending_idx.push(i);
            // Root perspective (no sign flip) but NOT shaped: leaves keep the
            // raw network prior so the tree interior stays faithful to the net.
            items.push(EvalItem {
                obs,
                memory: mem,
                belief: BeliefState {
                    seat: self.seat,
                    particles: vec![path.particle.clone()],
                    config: belief.config,
                    collapsed: false,
                },
                from_root: true,
                shape: false,
            });
        }

        if !items.is_empty() {
            let results = evaluator.evaluate_many(&items);
            for (local, (&i, (prior, value))) in pending_idx.iter().zip(&results).enumerate() {
                values[i] = Some(*value);
                let path = &paths[i];
                if let (Some(leaf), true) = (path.leaf_node, path.needs_expand) {
                    self.tree.node_mut(leaf).network_value = *value;
                    if self.tree.node(leaf).actions.is_empty() {
                        let item = &items[local];
                        let (obs, mem) = (item.obs.clone(), item.memory.clone());
                        self.expand_self_candidates(leaf, &obs, &mem, prior);
                        self.tree.node_mut(leaf).expanded = true;
                    }
                }
            }
        }

        values.into_iter().map(|v| v.expect("every leaf scored")).collect()
    }

    /// Back up from leaf to root. Only fully completed paths call this.
    pub fn backup_path(&mut self, path: &PendingPath, leaf_value: f64) {
        for i in (0..path.edges.len()).rev() {
            let node = path.nodes[i];
            let (h, a_idx, b_idx, _edge) = path.edges[i];
            self.tree.backup_node(node, &h, a_idx, b_idx, leaf_value);
            self.tree.clear_pins(node);
        }
        self.tree.completed_simulations += 1;
    }

    /// Select up to `pending_batch` paths, evaluate them, back them up in order.
    pub fn run_batch<E: SearchEvaluator + ?Sized>(
        &mut self,
        evaluator: &mut E,
        belief: &BeliefState,
        n_sims: Option<usize>,
        freeze_widening: bool,
    ) -> usize {
        assert!(self.tree.root.is_some(), "root not set");
        let batch_n = n_sims
            .unwrap_or(self.config.pending_batch)
            .min(self.config.pending_batch);
        let mut paths: Vec<PendingPath> = Vec::with_capacity(batch_n);
        for _ in 0..batch_n {
            paths.push(self.complete_select_path(evaluator, belief, freeze_widening));
        }
        let mut completed = 0;
        for mut path in paths {
            let value = {
                let mut one = [path];
                let values = self.evaluate_leaves(evaluator, &mut one, belief);
                path = one.into_iter().next().unwrap();
                values[0]
            };
            self.backup_path(&path, value);
            completed += 1;
        }
        completed
    }

    pub fn root_marginal_visits(&self) -> Vec<f64> {
        self.tree.root_marginal_visits()
    }

    pub fn best_action(&self) -> Option<Action5> {
        let index = self.tree.root_action_index().ok()?;
        let root = self.tree.node(self.tree.root?);
        crate::board::action::decode_action(root.actions[index])
    }

    /// Highest marginal visit, ties by root prior — the 1–7 simulation band.
    pub fn best_action_by_visits(&self) -> Option<Action5> {
        let root = self.tree.node(self.tree.root?);
        if root.actions.is_empty() {
            return None;
        }
        let visits = self.root_marginal_visits();
        let mut best = 0usize;
        for i in 1..root.actions.len() {
            if (visits[i], root.prior[i]) > (visits[best], root.prior[best]) {
                best = i;
            }
        }
        crate::board::action::decode_action(root.actions[best])
    }

    pub fn best_action_or_pass(&self) -> Action5 {
        let root = match self.tree.root {
            Some(root) => self.tree.node(root),
            None => return PASS_ACTION,
        };
        if root.actions.is_empty() {
            return PASS_ACTION;
        }
        if self.tree.completed_simulations == 0 {
            // Highest prior among candidates.
            let mut best = 0usize;
            for i in 1..root.prior.len() {
                if root.prior[i] > root.prior[best] {
                    best = i;
                }
            }
            return crate::board::action::decode_action(root.actions[best]).unwrap_or(PASS_ACTION);
        }
        self.best_action().unwrap_or(PASS_ACTION)
    }
}

/// Deterministic stub: a uniform legal prior and a constant value.
pub struct UniformEvaluator {
    pub value: f64,
}

impl SearchEvaluator for UniformEvaluator {
    fn evaluate(
        &mut self,
        obs: &Observation,
        memory: &VisibleMemory,
        _belief: &BeliefState,
        _from_root: bool,
        _shape: bool,
    ) -> (Vec<f64>, f64) {
        let mask = legal_mask(obs, memory, None);
        let live = mask.iter().filter(|&&m| m).count();
        let mut prior = vec![0.0f64; N_ACTIONS];
        if live == 0 {
            prior[crate::board::action::PASS_INDEX] = 1.0;
        } else {
            for (i, &m) in mask.iter().enumerate() {
                if m {
                    prior[i] = 1.0 / live as f64;
                }
            }
        }
        (prior, self.value)
    }
}

/// Test helper: a fixed prior vector and value, masked and renormalized.
///
/// The Python's `ScriptedEvaluator`. It exists here for the same reason: the
/// `search` parity surface needs an evaluator whose answers are *identical* on
/// both sides, and the network's are not — they agree to 6.6e-7, which is
/// enough to reorder a near-tie in the candidate list and turn a search
/// comparison into an argument about the last bit of a softmax. With a scripted
/// prior, any disagreement in the tree is the tree's.
pub struct ScriptedEvaluator {
    pub prior: Vec<f64>,
    pub value: f64,
}

impl SearchEvaluator for ScriptedEvaluator {
    fn evaluate(
        &mut self,
        obs: &Observation,
        memory: &VisibleMemory,
        _belief: &BeliefState,
        _from_root: bool,
        _shape: bool,
    ) -> (Vec<f64>, f64) {
        let mask = legal_mask(obs, memory, None);
        let masked: Vec<f64> = (0..N_ACTIONS)
            .map(|i| if mask[i] { self.prior[i] } else { 0.0 })
            .collect();
        let total = crate::support::rng::npsum(&masked);
        if total <= 0.0 {
            let live = mask.iter().filter(|&&m| m).count().max(1);
            let prior = (0..N_ACTIONS)
                .map(|i| if mask[i] { 1.0 / live as f64 } else { 0.0 })
                .collect();
            return (prior, self.value);
        }
        (masked.iter().map(|&v| v / total).collect(), self.value)
    }
}

/// A belief holding exactly the particles handed to it, for leaf evaluation.
pub fn singleton_belief(seat: usize, particle: &Particle, config: BeliefConfig) -> BeliefState {
    BeliefState {
        seat,
        particles: vec![particle.clone()],
        config,
        collapsed: false,
    }
}
