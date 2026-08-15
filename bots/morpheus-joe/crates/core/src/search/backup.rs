//! Leaf evaluation, backup, and reading the answer off the root.
//!
//! [`SearchController::evaluate_leaves`] is the only place in the search that
//! asks the evaluator for a value, and it asks for all of them at once so the
//! runtime's leaf batch stays a scheduling unit. Backup then folds each value
//! from leaf to root, and the `best_action*` family reads the decision back
//! out — including the two degraded readings for the 0- and 1-to-7-simulation
//! bands, which exist because the deadline sometimes leaves nothing else.

use crate::belief::{BeliefState, PASS_ACTION};
use crate::board::memory::update_memory;
use crate::board::observe::emit_observation;
use crate::belief::Action5;
use crate::search::{EvalItem, PendingPath, SearchController, SearchEvaluator};

impl SearchController {
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
