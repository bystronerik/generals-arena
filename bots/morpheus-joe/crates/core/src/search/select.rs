//! The tree walk: one simulation path, zero network forwards.
//!
//! [`SearchController::select_path`] is the 300-line loop the rest of the
//! search is arranged around, and its invariant is the reason it is arranged
//! that way — it never calls the network. A missing enemy prior suspends the
//! walk as an [`EnemyPriorRequest`] instead, which is what lets the runtime
//! charge every forward to one admitted component.
//! [`SearchController::complete_select_path`] is the same walk with the
//! materialise-and-resume loop closed around it.

use std::rc::Rc;

use crate::belief::{BeliefState, Particle, PASS_ACTION};
use crate::board::action::N_ACTIONS;
use crate::board::hashing::{
    child_edge_key, info_state_key_prehashed, memory_digest, observation_payload,
    roll_history_digest,
};
use crate::board::memory::update_memory;
use crate::board::observe::emit_observation;
use crate::board::transition::{transition, Actions};
use crate::search::matrix::{mixed_strategy, sample_index, self_widening_limit};
use crate::search::{
    EnemyPriorRequest, PendingPath, SearchController, SearchEvaluator, Selection,
};

impl SearchController {
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
}
