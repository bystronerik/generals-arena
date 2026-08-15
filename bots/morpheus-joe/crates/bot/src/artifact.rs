//! Finding joe's artifact, and refusing to start on the wrong one.
//!
//! What is left of morpheus's `nn/inference.rs` after the port. Its manifest
//! guardrails — tensor schema, action schema, architecture version, the four
//! architecture dimensions, the army-bin edges — described the 249,316-
//! parameter CNN and went with it; `joenet::nn::net::Net::load` applies joe's
//! own equivalents (`tensor_schema`, then `pad_to`, `history_size`, `depth`,
//! `embed_dim`, `n_head`, `ff_factor`, `patch_size`, `num_bins`, then every
//! tensor's name, shape and dtype, then a refusal if any tensor is left over).
//!
//! Two things it does **not** do, which is why this file exists:
//!
//! * **The digest.** `Net::load` reads `model.safetensors` without checking it
//!   against the `safetensors_sha256` the manifest beside it pins. That is the
//!   only check that notices a *corrupt* or *half-copied* file rather than a
//!   mislabelled one, and this bot's weights arrive by a copy —
//!   `scripts/joe_artifact_fanout.py` — that a killed process can interrupt.
//!   morpheus-rs verified its digest at load; so does this. It costs about a
//!   millisecond per 34 MB out of an 8.5 s first-move grace.
//! * **The search.** joe-rs resolves one exe-relative candidate because its
//!   layout is fixed. This bot runs from two: `run.sh` execs
//!   `target/release/morpheus-joe` from inside the bot directory, and the
//!   submission zip puts the binary beside `artifact/`. Both shapes are tried,
//!   plus the current directory, so the same binary works under the repo
//!   harness and in the sandbox without a wrapper passing paths in.

use std::path::{Path, PathBuf};

use joenet::io::json::{parse, Json};
use joenet::nn::net::Net;
use morpheus_joe_core::support::sha256::sha256;

pub const MANIFEST_NAME: &str = "manifest.json";
pub const ARTIFACT_DIRNAME: &str = "artifact";

/// What the manifest says about the checkpoint on disk, for the trace and the
/// selfcheck. Provenance only — nothing branches on it.
pub struct Provenance {
    pub safetensors_sha256: String,
    pub tensor_schema: String,
    pub checkpoint_run: String,
    pub checkpoint_step: String,
}

fn hex(bytes: &[u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for b in bytes {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

/// `bots/morpheus-joe/artifact/`, resolved from the running binary.
pub fn default_artifact_dir() -> Result<PathBuf, String> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        // .../bots/morpheus-joe/target/release/morpheus-joe
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join(ARTIFACT_DIRNAME));
            if let Some(up) = dir.parent().and_then(|p| p.parent()) {
                candidates.push(up.join(ARTIFACT_DIRNAME));
            }
        }
    }
    candidates.push(PathBuf::from(ARTIFACT_DIRNAME));
    for candidate in &candidates {
        if candidate.join(MANIFEST_NAME).is_file() {
            return Ok(candidate.clone());
        }
    }
    Err(format!(
        "no artifact directory with a {MANIFEST_NAME} among {candidates:?}"
    ))
}

fn read_manifest(dir: &Path) -> Result<Json, String> {
    let path = dir.join(MANIFEST_NAME);
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("reading {}: {e}", path.display()))?;
    parse(&text).map_err(|e| format!("{}: {e}", path.display()))
}

fn str_at(manifest: &Json, key: &str) -> String {
    manifest
        .get(key)
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

/// Read the manifest's provenance without loading 34 MB of weights.
pub fn provenance(dir: &Path) -> Result<Provenance, String> {
    let manifest = read_manifest(dir)?;
    let checkpoint = manifest.get("checkpoint");
    Ok(Provenance {
        safetensors_sha256: str_at(&manifest, "safetensors_sha256"),
        tensor_schema: str_at(&manifest, "tensor_schema"),
        checkpoint_run: checkpoint
            .and_then(|c| c.get("run_name"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        checkpoint_step: checkpoint
            .and_then(|c| c.get("global_step"))
            .and_then(|v| v.as_i64())
            .map_or_else(|| "null".into(), |v| v.to_string()),
    })
}

/// Verify the weights against the digest the manifest pins, then load them.
///
/// The digest first, deliberately: a truncated file is likelier to fail
/// `Net::load`'s shape checks with a confusing message about one tensor than
/// to be recognised as the copy that never finished.
pub fn load(dir: &Path) -> Result<Net, String> {
    let manifest = read_manifest(dir)?;
    let claimed = manifest
        .get("safetensors_sha256")
        .and_then(|v| v.as_str())
        .ok_or("manifest has no safetensors_sha256 to verify against")?
        .to_string();
    let name = manifest
        .get("safetensors")
        .and_then(|v| v.as_str())
        .unwrap_or("model.safetensors");
    let weights = dir.join(name);
    let raw = std::fs::read(&weights).map_err(|e| format!("reading {}: {e}", weights.display()))?;
    let actual = hex(&sha256(&[&raw]));
    if actual != claimed {
        return Err(format!(
            "{}: SHA-256 {actual} but the manifest pins {claimed}. The artifact is \
             corrupt or the copy from joe-rs was interrupted; re-run \
             `python scripts/joe_artifact_fanout.py --bot morpheus-joe`",
            weights.display()
        ));
    }
    Net::load(dir)
}
