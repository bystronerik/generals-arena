//! The tree, the regret arithmetic, and the runtime around them.

use crate::board::action::N_ACTIONS;
use crate::search::matrix;
use crate::support::rng::Replay;
use crate::parity::codec::*;
use crate::parity::ints::Ints;
use crate::parity::bench::VaryingEvaluator;
use crate::parity::Ctx;

/// strategy vectors and one joint matrix -> the whole regret cycle
///
/// The one surface in the port with a **tolerance on an integer-free
/// path**, and the reason is the oracle: NumPy sends `@` on f64 to
/// BLAS, whose reduction order is the vendor's. See `matrix.rs`.
pub(in crate::parity) fn matrix(
    ints: &mut Ints,
    out: &mut Vec<i64>,
    _ctx: &mut Ctx,
) -> Result<(), String> {
    let n_a = ints.n()?;
    let n_b = ints.n()?;
    let n_hashes = ints.n()?;
    let n = ints.next()?;
    let regrets = ints.f64s(n_a)?;
    let prior = ints.f64s(n_a)?;
    let avg_strategy = ints.f64s(n_a)?;
    let marginal_visits = ints.f64s(n_a)?;
    let first_play = ints.f64s(1)?[0];
    let enemy_weights = ints.f64s(n_hashes)?;
    let mut enemy_regrets = Vec::with_capacity(n_hashes);
    let mut enemy_priors = Vec::with_capacity(n_hashes);
    let mut visits = Vec::with_capacity(n_hashes);
    let mut q = Vec::with_capacity(n_hashes);
    for _ in 0..n_hashes {
        enemy_regrets.push(ints.f64s(n_b)?);
        enemy_priors.push(ints.f64s(n_b)?);
        visits.push(ints.f64s(n_a * n_b)?);
        q.push(ints.f64s(n_a * n_b)?);
    }
    let a_idx = ints.n()?;
    let b_idx = ints.n()?;
    let leaf_value = ints.f64s(1)?[0];

    out.extend([
        matrix::self_widening_limit(n) as i64,
        matrix::enemy_widening_limit(n) as i64,
    ]);
    push_f64(out, &[matrix::exploration_epsilon(n)]);

    let sigma_self = matrix::mixed_strategy(&regrets, &prior, n);
    push_f64(out, &sigma_self);
    let mut enemy_sigmas = Vec::with_capacity(n_hashes);
    let mut q_eff_list = Vec::with_capacity(n_hashes);
    for h in 0..n_hashes {
        let sigma_b = matrix::mixed_strategy(&enemy_regrets[h], &enemy_priors[h], n);
        push_f64(out, &sigma_b);
        let q_eff = matrix::effective_q(&visits[h], &q[h], first_play);
        push_f64(out, &q_eff);
        enemy_sigmas.push(sigma_b);
        q_eff_list.push(q_eff);
    }
    let (u_self, v) = matrix::aggregate_self_utilities(
        &sigma_self,
        &enemy_weights,
        &enemy_sigmas,
        &q_eff_list,
    );
    push_f64(out, &u_self);
    push_f64(out, &[v]);
    let (u_h, u_enemy, v_h) =
        matrix::matrix_utilities(&sigma_self, &enemy_sigmas[0], &q_eff_list[0]);
    push_f64(out, &u_h);
    push_f64(out, &u_enemy);
    push_f64(out, &[v_h]);
    push_f64(
        out,
        &matrix::regret_plus_update(&regrets, &u_self, v, true),
    );
    push_f64(
        out,
        &matrix::regret_plus_update(&enemy_regrets[0], &u_enemy, v, false),
    );

    let mut visits_0 = visits[0].clone();
    let mut value_sum = vec![0.0f64; n_a * n_b];
    let mut q_0 = q[0].clone();
    matrix::apply_joint_backup(
        &mut visits_0,
        &mut value_sum,
        &mut q_0,
        n_b,
        a_idx,
        b_idx,
        leaf_value,
    );
    push_f64(out, &visits_0);
    push_f64(out, &value_sum);
    push_f64(out, &q_0);

    push_f64(out, &matrix::normalize_average_strategy(&avg_strategy));
    out.push(matrix::select_root_action(
        &avg_strategy,
        &marginal_visits,
        &prior,
        None,
    ) as i64);
    Ok(())
}

// `decide` — "the whole no-search decision, network included" — was the
// harness's tier 3 and it is **retired here with no replacement**, as a named
// and accepted cost (joe-net-plan §8.4). It needs an oracle, and no Python
// program plays this combination: building one means porting joe's JAX net
// into Python morpheus, whose only product would be the oracle. What covers
// the ground instead, in descending order of what it proves: N3's new `prior`
// surface over the genuinely new logic; determinism (same frame twice, same
// seed twice); and transitively, §8.2 for the forward plus the surviving
// surfaces for the search. The residue — "did the combination decide
// correctly" — is answered by the N6 rating round and by nothing else.

/// a root frame + a scripted evaluator + a recorded draw stream ->
/// the whole tree after N batches
///
/// The surface that proves the *search*, which no other one reaches:
/// `decide` runs at zero simulations by design, and `matrix` checks
/// the arithmetic without the storage that feeds it. Here the tree
/// is built for real — selection, progressive widening, enemy-table
/// installation and eviction, leaf expansion, backup — and every
/// statistic it accumulates is compared.
///
/// The evaluator is *scripted*, not the network. The two engines'
/// priors agree to 6.6e-7, which is enough to reorder a near-tie in
/// the candidate list, and a search comparison that could fail on
/// the last bit of a softmax would prove nothing about the search.
/// With identical priors on both sides, any disagreement here is the
/// tree's.
pub(in crate::parity) fn search(
    ints: &mut Ints,
    out: &mut Vec<i64>,
    _ctx: &mut Ctx,
) -> Result<(), String> {
    let obs = read_observation(ints)?;
    let memory = read_memory(ints)?;
    let belief = read_belief(ints)?;
    let prior = ints.f64s(N_ACTIONS)?;
    let value = ints.f64s(1)?[0];
    let config = crate::search::SearchConfig {
        depth: ints.n()?,
        pending_batch: ints.n()?,
        n_particles: ints.n()?,
        max_nodes: ints.n()?,
        max_enemy_tables: ints.n()?,
        ..Default::default()
    };
    let batches = ints.n()?;
    let freeze = ints.next()? != 0;
    let vary = ints.next()? != 0;
    let rng = crate::support::rng::SharedRng::new(Box::new(Replay::new(read_draws(ints)?)));
    let scripted = crate::search::ScriptedEvaluator { prior, value };
    let mut evaluator: Box<dyn crate::search::SearchEvaluator> = if vary {
        Box::new(VaryingEvaluator { inner: scripted })
    } else {
        Box::new(scripted)
    };
    let mut controller =
        crate::search::SearchController::new(belief.seat, config, rng.handle());
    controller.ensure_root(evaluator.as_mut(), &obs, &memory, &belief);
    for _ in 0..batches {
        controller.run_batch(evaluator.as_mut(), &belief, None, freeze);
    }

    let tree = &controller.tree;
    out.extend([
        tree.completed_simulations as i64,
        tree.nodes.len() as i64,
        tree.table_hits as i64,
        tree.table_misses as i64,
    ]);
    push_f64(out, &[tree.eviction_loss, tree.total_joint_visits]);
    let root = tree.node(tree.root.expect("ensure_root always sets one"));
    out.push(root.n);
    out.push(root.actions.len() as i64);
    out.extend(root.actions.iter().map(|&a| a as i64));
    push_f64(out, &root.prior);
    push_f64(out, &root.regret);
    push_f64(out, &root.avg_strategy);
    out.push(root.enemy_tables.len() as i64);
    for table in &root.enemy_tables {
        out.push(table.actions.len() as i64);
        out.extend(table.actions.iter().map(|&a| a as i64));
        push_f64(out, &table.prior);
        push_f64(out, &table.regret);
        push_f64(out, &table.avg_strategy);
        out.extend([table.n_self as i64, table.last_used, table.touch_count]);
        push_f64(out, &table.visits);
        push_f64(out, &table.q);
    }
    push_f64(out, &tree.root_marginal_visits());
    match tree.root_action_index() {
        Ok(index) => out.push(index as i64),
        Err(_) => out.push(-1),
    }
    match controller.best_action() {
        Some(action) => {
            out.push(1);
            out.extend(action.iter().map(|&v| v as i64));
        }
        None => out.extend([0, -1, -1, -1, -1, -1]),
    }
    // The degradation path, on the tree that was just built. It
    // lives here rather than in its own surface because its
    // interesting branch — "the search says pass, the policy
    // fallback does not" — needs a real root to say pass.
    for (completed, has_root, fallback) in [
        (0u64, false, [1, 0, 0, 0, 0]),
        (0, true, [0, 1, 1, 0, 0]),
        (tree.completed_simulations, true, [0, 1, 1, 0, 0]),
        (tree.completed_simulations, true, [1, 0, 0, 0, 0]),
    ] {
        let (action, level) = crate::runtime::select_degraded_action(
            completed,
            has_root,
            Some(fallback),
            &controller,
        );
        out.extend(action.iter().map(|&v| v as i64));
        out.push(level as i64);
    }

    // Every node, not only the root — M6's correction to this
    // surface. A leaf value comes from the evaluator and is applied
    // unchanged to every edge on the path, so nothing a child
    // computes ever reaches the root's statistics; and `Replay`
    // hands back the oracle's sampled index whatever distribution
    // this side built, so a divergence inside a child does not even
    // change the tree's shape. Comparing the root alone therefore
    // could not see the enemy-hash cache, the memory fold, or any
    // other per-node arithmetic. The nodes are in creation order on
    // both sides.
    let tree = &controller.tree;
    out.push(tree.nodes.len() as i64);
    for node in &tree.nodes {
        out.extend([node.n, node.turn as i64]);
        out.extend(node.memory_digest.iter().map(|&b| b as i64));
        out.push(node.reservoir.n() as i64);
        out.push(node.actions.len() as i64);
        out.extend(node.actions.iter().map(|&a| a as i64));
        push_f64(out, &node.prior);
        push_f64(out, &node.regret);
        push_f64(out, &node.avg_strategy);
        out.push(node.enemy_tables.len() as i64);
        for table in &node.enemy_tables {
            out.extend([table.n_self as i64, table.last_used, table.touch_count]);
            out.push(table.actions.len() as i64);
            out.extend(table.actions.iter().map(|&a| a as i64));
            push_f64(out, &table.prior);
            push_f64(out, &table.regret);
            push_f64(out, &table.avg_strategy);
            push_f64(out, &table.visits);
            push_f64(out, &table.q);
        }
    }
    out.push(crate::support::rng::Rng::consumed(&rng) as i64);
    Ok(())
}

/// The eviction decision, driven directly.
///
/// M6 added this for the reason the `runtime` surface exists: two of
/// the retention rule's four behaviours cannot be reached by running
/// a search at all. The score is `last_used + 0.25 * ln1p(touches)`,
/// so the touch term can only decide a tie in `last_used` — and
/// `last_used` is the node's visit counter at the table's last
/// backup, which advances on every backup, so no two tables in a
/// real tree ever hold the same one. Stating the table set is the
/// only way to ask the question.
pub(in crate::parity) fn evict(
    ints: &mut Ints,
    out: &mut Vec<i64>,
    _ctx: &mut Ctx,
) -> Result<(), String> {
    let count = ints.n()?;
    let node_n = ints.next()?;
    let max_tables = ints.n()?;
    let mut tree = crate::search::tree::SearchTree::new(1024, max_tables, 0);
    let at = tree
        .make_node([0u8; 32], 0, [0u8; 32], Vec::new(), [0u8; 32], 0.0, 1)
        .map_err(|_| "the first node always fits".to_string())?;
    let mut pins: Vec<[u8; 32]> = Vec::new();
    for _ in 0..count {
        let seed = ints.next()? as u8;
        let last_used = ints.next()?;
        let touch_count = ints.next()?;
        let pinned = ints.next()? != 0;
        let hash = [seed; 32];
        tree.get_or_create_enemy_table(at, hash, &[1], &[1.0]);
        let table = tree
            .node_mut(at)
            .table_mut(&hash)
            .expect("just installed");
        table.last_used = last_used;
        table.touch_count = touch_count;
        if pinned {
            pins.push(hash);
        }
    }
    for hash in pins {
        tree.pin_enemy(at, hash);
    }
    tree.node_mut(at).n = node_n;
    let arriving = [ints.next()? as u8; 32];
    tree.get_or_create_enemy_table(at, arriving, &[1], &[1.0]);
    let node = tree.node(at);
    out.push(node.enemy_tables.len() as i64);
    for table in &node.enemy_tables {
        out.extend([
            table.info_hash[0] as i64,
            table.last_used,
            table.touch_count,
        ]);
    }
    Ok(())
}

/// scalar controller arithmetic with no state behind it
///
/// `nearest_rank_p99` and `highest_prior_legal` are two of the three
/// places the runtime decides something on its own, and neither is
/// reachable through `decide`: the prior it is handed is already
/// legal-normalized, so the mask it applies never binds, and the
/// percentile only shows up in an admission decision the parity
/// harness does not replay. Feeding both directly is the only way
/// the harness can see them at all.
pub(in crate::parity) fn runtime(
    ints: &mut Ints,
    out: &mut Vec<i64>,
    _ctx: &mut Ctx,
) -> Result<(), String> {
    let count = ints.n()?;
    let samples = ints.f64s(count)?;
    match crate::runtime::nearest_rank_p99(&samples) {
        Ok(value) => {
            out.push(1);
            push_f64(out, &[value]);
        }
        Err(_) => out.extend([0, 0]),
    }
    let prior = ints.f64s(N_ACTIONS)?;
    let mask_ints = ints.ints(N_ACTIONS)?;
    let mask: Vec<bool> = mask_ints.iter().map(|v| *v != 0).collect();
    let action = crate::runtime::highest_prior_legal(&prior, &mask);
    out.extend(action.iter().map(|&v| v as i64));
    Ok(())
}
