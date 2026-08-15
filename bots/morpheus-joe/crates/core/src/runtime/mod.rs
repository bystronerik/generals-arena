//! The runtime controller: deadlines, admission, degradation, telemetry.
//!
//! Port of `bots/morpheus/runtime.py`. rewrite-plan §7 files this as
//! "translation only, no gain" — and that is the point. The controller *is* the
//! deadline behaviour, so it is ported faithfully rather than improved, and its
//! telemetry schema is kept identical so the existing analysis tooling reads
//! both bots.
//!
//! **M4 found the one number here that is a lie by construction.** The
//! controller charges `filter_step` and the `recover_belief` it may trigger to
//! a single component, `particle_transitions`, so its p99 describes a cost
//! distribution that is bimodal by a factor of a hundred (1.1 ms filtered,
//! 127 ms recovered on the Python). That accounting is reproduced, because
//! changing it would silently invalidate every recorded trace — but M7 should
//! reserve for the two paths separately.
//!
//! ## The directory
//!
//! [`config`] is the knobs and the component vocabulary, [`clock`] the time
//! source behind a trait, [`estimator`] the per-component p99 forecast
//! admission control runs on, [`metrics`] the passive telemetry schema,
//! [`degrade`] what to send when the deadline wins, and [`controller`] the
//! thing that spends the budget.
//!
//! [`deployment`] and [`telemetry`] are here for the same reason: the
//! manifest exists to override this directory's knobs, and the trace's key set
//! *is* the controller's component list.

pub mod clock;
pub mod config;
pub mod controller;
pub mod degrade;
pub mod deployment;
pub mod estimator;
pub mod metrics;
pub mod telemetry;

pub use clock::*;
pub use config::*;
pub use controller::*;
pub use degrade::*;
pub use estimator::*;
pub use metrics::*;
