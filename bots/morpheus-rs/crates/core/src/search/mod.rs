//! Root-sampled information-set MCTS with simultaneous matrices.
//!
//! Port of `bots/morpheus/search.py`. Selection performs **zero network
//! forwards**: when an enemy table's prior is missing it returns an
//! [`EnemyPriorRequest`] so the caller can batch the forwards and resume,
//! which is what lets the runtime charge inference to one admitted component
//! instead of scattering it through the tree walk.
//!
//! M0's baseline made two of these numbers worth restating: `selection`
//! measured 9.6x its shipped p99 on the M3 Pro and 13.5x on x86, and `backup`
//! 2.9x/4.7x. rewrite-plan §7 filed both as "small absolute numbers"; they are
//! not, and this port treats them as first-class rather than as the tail of the
//! search work.
//!
//! ## The directory
//!
//! This file holds the shared types — the config, the two path types, and the
//! [`SearchController`]'s fields — and the rest is one subject per file:
//! [`controller`] builds and roots the tree and materialises enemy priors,
//! [`select`] walks it, [`backup`] scores the leaves and folds the values
//! home, [`evaluator`] is every implementation of [`SearchEvaluator`],
//! [`tree`] is the arena the nodes live in, and [`matrix`] the regret
//! arithmetic each node runs.
//!
//! `select` and `backup` carry `impl SearchController` blocks against fields
//! declared here. That costs nothing in visibility: a child module can see its
//! ancestors' private items.

pub mod backup;
pub mod controller;
pub mod evaluator;
pub mod matrix;
pub mod select;
pub mod tree;

pub use evaluator::{ScriptedEvaluator, SearchEvaluator, UniformEvaluator};

use std::rc::Rc;

use crate::belief::{BeliefConfig, BeliefState, Particle};
use crate::board::memory::VisibleMemory;
use crate::board::state::GameState;
use crate::io::wire::Observation;
use crate::search::tree::{Digest, SearchTree};
use crate::support::rng::SharedRng;

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

/// A belief holding exactly the particles handed to it, for leaf evaluation.
pub fn singleton_belief(seat: usize, particle: &Particle, config: BeliefConfig) -> BeliefState {
    BeliefState {
        seat,
        particles: vec![particle.clone()],
        config,
        collapsed: false,
    }
}
