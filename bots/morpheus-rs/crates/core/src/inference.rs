//! Loading the artifact, and the guardrails that decide whether to play at all.
//!
//! Port of `bots/morpheus/inference.py`, minus the parts that only exist
//! because the Python ships three TorchScript modules: there is one artifact
//! here, one graph, and an entry-point switch ([`network::Heads`]).
//!
//! What survives unchanged is the refusal to run on an artifact that does not
//! match the code. `validate_manifest` on the Python side checks the manifest
//! version, the tensor and action schema tags, the architecture version, the
//! four architecture dimensions the graph assumes, the army-bin edges, and the
//! weights digest. All of it is checked here too, against the manifest
//! `tools/convert_artifact.py` copied next to the weights, because the failure
//! it prevents — a bot that starts happily on last month's checkpoint and
//! plays a rated game with the wrong priors — is silent in every other way.
//!
//! The digest is the expensive one: SHA-256 over a 1 MB file, using this
//! crate's own implementation. It costs a millisecond of the ~8.5 s first-move
//! grace and it is the only check that notices a *corrupt* file rather than a
//! mislabelled one.

use std::path::{Path, PathBuf};

use crate::json::{parse, Json};
use crate::network::{
    Heads, Network, Output, BOARD, IN_CHANNELS, N_ARMY_BINS, N_BLOCKS,
};
use crate::safetensors::SafeTensors;
use crate::sha256::sha256;

pub const MANIFEST_NAME: &str = "manifest.json";
pub const ARTIFACT_DIRNAME: &str = "artifact";

pub const MANIFEST_VERSION: &str = "1";
pub const TENSOR_SCHEMA_VERSION: &str = "morpheus-tensor-v1";
pub const ACTION_SCHEMA_VERSION: &str = "morpheus-action-v1";
pub const ARCHITECTURE_VERSION: &str = "morpheus-net-v1";

/// The 17 logarithmic army-bin edge values the tensor builder and the
/// auxiliary heads share.
///
/// Compared against the manifest with a *relative* tolerance, not bit-exactly,
/// even though the Python compares its own copy exactly. Both sides evaluate
/// `exp(ln(4096) · i / 16)`, and `exp` is a libm function: glibc on x86 and
/// Apple's libm need not agree in the last bit for the same double. An exact
/// comparison would therefore be a check that refuses to start the bot on a
/// platform difference of one ulp. What the check is *for* — a manifest from a
/// different army scale or a different bin count — moves these values by
/// percent, not by ulps.
fn army_bin_edges() -> Vec<f64> {
    let army_max: f64 = 4096.0;
    let mut edges = vec![0.0f64];
    let log_max = army_max.ln();
    for i in 1..=N_ARMY_BINS {
        edges.push((log_max * i as f64 / N_ARMY_BINS as f64).exp());
    }
    edges[N_ARMY_BINS] = army_max;
    edges
}

fn hex(bytes: &[u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for b in bytes {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

pub struct Session {
    pub network: Network,
    pub army_scale: f64,
    pub checkpoint_id: String,
    pub weights_sha256: String,
    output: Option<Output>,
}

impl Session {
    /// Load and validate `<dir>/manifest.json` plus the weights it names.
    pub fn load(artifact_dir: &Path) -> Result<Self, String> {
        let manifest_path = artifact_dir.join(MANIFEST_NAME);
        let text = std::fs::read_to_string(&manifest_path)
            .map_err(|e| format!("reading {}: {e}", manifest_path.display()))?;
        let manifest = parse(&text)
            .map_err(|e| format!("{}: {e}", manifest_path.display()))?;

        let artifact_file = manifest.str_field("artifact_file")?;
        let weights_path = artifact_dir.join(artifact_file);
        let raw = std::fs::read(&weights_path)
            .map_err(|e| format!("reading {}: {e}", weights_path.display()))?;
        let digest = hex(&sha256(&[&raw]));
        validate_manifest(&manifest, &digest)?;

        let st = SafeTensors::from_bytes(&raw)
            .map_err(|e| format!("{}: {e}", weights_path.display()))?;
        // The safetensors metadata block is written by the converter and is
        // not the authority — the manifest is. It is checked anyway, because
        // the two travelling separately is exactly how a weights file and a
        // manifest end up describing different checkpoints.
        if let Some(tag) = st.metadata("architecture_version") {
            if tag != ARCHITECTURE_VERSION {
                return Err(format!(
                    "weights metadata architecture_version {tag:?} != {ARCHITECTURE_VERSION:?}"
                ));
            }
        }
        let network = Network::from_safetensors(&st)?;

        let checkpoint_id = manifest
            .get("training_run")
            .and_then(|r| r.get("checkpoint_id"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        Ok(Self {
            network,
            army_scale: manifest
                .field("army_scale")?
                .as_f64()
                .ok_or("army_scale is not a number")?,
            checkpoint_id,
            weights_sha256: digest,
            output: Some(Output::new()),
        })
    }

    /// `bots/morpheus-rs/artifact/`, resolved from the running binary.
    ///
    /// `run.sh` execs `target/release/morpheus-rs` from inside the bot
    /// directory, and the submission zip puts the binary beside `artifact/`.
    /// Both shapes are tried, plus the current directory, so the same binary
    /// works under the repo harness and in the sandbox without a wrapper
    /// passing paths in.
    pub fn default_artifact_dir() -> Result<PathBuf, String> {
        let mut candidates: Vec<PathBuf> = Vec::new();
        if let Ok(exe) = std::env::current_exe() {
            // .../bots/morpheus-rs/target/release/morpheus-rs
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
            "no artifact directory with a {MANIFEST_NAME} among {:?}",
            candidates
        ))
    }

    pub fn load_default() -> Result<Self, String> {
        Self::load(&Self::default_artifact_dir()?)
    }

    /// Run every shape the match path uses, before the clock matters.
    ///
    /// The Python warms batches 1, 4, and 64 to make TorchScript specialise
    /// its kernels. Nothing here specialises at runtime — the shapes are
    /// compiled in — so this is warming caches and touching every page of the
    /// weights, which is worth doing once and not worth doing sixty-four
    /// times. The batch dimension is a loop here, so one sample is the only
    /// shape there is.
    pub fn warmup(&mut self) {
        let x = vec![0f32; IN_CHANNELS * BOARD * BOARD];
        for heads in [Heads::Policy, Heads::PolicyWdl, Heads::All] {
            for _ in 0..2 {
                self.forward(&x, heads);
            }
        }
    }

    /// One forward pass, reusing the session's output buffer.
    ///
    /// The `take`/put-back dance exists to hand `forward_into` a `&mut Output`
    /// while `&mut self.network` is also borrowed. An `Option` rather than
    /// `mem::replace(_, Output::new())`, because the latter allocates thirteen
    /// thousand floats on every call — on the per-turn path, which §6 says
    /// allocates nothing.
    pub fn forward(&mut self, x: &[f32], heads: Heads) -> &Output {
        let mut out = self
            .output
            .take()
            .expect("output buffer is only absent inside this function");
        self.network.forward_into(x, heads, &mut out);
        self.output = Some(out);
        self.output.as_ref().unwrap()
    }
}

/// Every guardrail `inference.validate_manifest` applies, in the same order.
pub fn validate_manifest(manifest: &Json, weights_digest: &str) -> Result<(), String> {
    for key in [
        "manifest_version",
        "tensor_schema",
        "action_schema",
        "architecture_version",
        "architecture",
        "quantization",
        "weights_sha256",
        "army_bin_edges",
        "artifact_file",
        "runtime",
        "army_scale",
    ] {
        if manifest.get(key).is_none() {
            return Err(format!("manifest missing {key:?}"));
        }
    }
    for (key, want) in [
        ("manifest_version", MANIFEST_VERSION),
        ("tensor_schema", TENSOR_SCHEMA_VERSION),
        ("action_schema", ACTION_SCHEMA_VERSION),
        ("architecture_version", ARCHITECTURE_VERSION),
    ] {
        let got = manifest.str_field(key)?;
        if got != want {
            return Err(format!("{key} mismatch: {got:?} != {want:?}"));
        }
    }

    let arch = manifest.field("architecture")?;
    for (key, want) in [
        ("in_channels", IN_CHANNELS as i64),
        ("board", BOARD as i64),
        ("n_army_bins", N_ARMY_BINS as i64),
        ("n_blocks", N_BLOCKS as i64),
    ] {
        let got = arch.int_field(key)?;
        if got != want {
            return Err(format!("architecture {key} mismatch: {got} != {want}"));
        }
    }

    let quant = manifest.field("quantization")?;
    for key in ["engine", "format"] {
        quant.str_field(key)?;
    }
    // The Python selects a quantized backend for int8 artifacts. There is no
    // int8 path here, so an int8 artifact is a refusal rather than a silent
    // float reinterpretation of quantized bytes.
    let format = quant.str_field("format")?;
    if format != "float32" {
        return Err(format!("quantization format {format:?} is not float32"));
    }

    let edges = manifest
        .field("army_bin_edges")?
        .as_array()
        .ok_or("army_bin_edges is not an array")?;
    let want_edges = army_bin_edges();
    if edges.len() != want_edges.len() {
        return Err(format!(
            "army_bin_edges has {} entries, expected {}",
            edges.len(),
            want_edges.len()
        ));
    }
    for (i, (got, want)) in edges.iter().zip(&want_edges).enumerate() {
        let got = got.as_f64().ok_or("army_bin_edges holds a non-number")?;
        if (got - want).abs() > 1e-12 * want.abs().max(1.0) {
            return Err(format!("army_bin_edges[{i}]: {got} != {want}"));
        }
    }

    let claimed = manifest.str_field("weights_sha256")?;
    if claimed != weights_digest {
        return Err(format!(
            "weights SHA-256 mismatch: manifest {claimed} != file {weights_digest}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn good_manifest(digest: &str) -> String {
        let edges: Vec<String> = army_bin_edges().iter().map(|e| format!("{e:?}")).collect();
        format!(
            r#"{{"manifest_version":"1","tensor_schema":"morpheus-tensor-v1",
                 "action_schema":"morpheus-action-v1",
                 "architecture_version":"morpheus-net-v1",
                 "architecture":{{"in_channels":49,"board":21,"n_army_bins":16,"n_blocks":12}},
                 "quantization":{{"engine":"none","format":"float32"}},
                 "weights_sha256":"{digest}","army_bin_edges":[{}],
                 "artifact_file":"model.safetensors","runtime":"x","army_scale":4096.0}}"#,
            edges.join(",")
        )
    }

    #[test]
    fn accepts_a_well_formed_manifest() {
        let manifest = parse(&good_manifest("abc")).unwrap();
        validate_manifest(&manifest, "abc").unwrap();
    }

    #[test]
    fn refuses_every_way_the_artifact_can_be_wrong() {
        let base = good_manifest("abc");
        let cases = [
            // a digest that does not match the file on disk
            (base.clone(), "different-digest"),
            // an architecture the graph does not implement
            (base.replace(r#""n_blocks":12"#, r#""n_blocks":10"#), "abc"),
            (base.replace(r#""in_channels":49"#, r#""in_channels":64"#), "abc"),
            // a schema the tensor builder no longer produces
            (
                base.replace("morpheus-tensor-v1", "morpheus-tensor-v2"),
                "abc",
            ),
            (base.replace("morpheus-net-v1", "morpheus-net-v2"), "abc"),
            (base.replace("morpheus-action-v1", "morpheus-action-v2"), "abc"),
            // a quantized artifact, which this bot cannot read
            (base.replace(r#""format":"float32""#, r#""format":"int8""#), "abc"),
            // an edge table from a different army scale
            (base.replace("4096.0]", "4095.0]"), "abc"),
            // a required key removed outright
            (base.replace(r#""runtime":"x","#, ""), "abc"),
        ];
        for (text, digest) in cases {
            let manifest = parse(&text).unwrap();
            assert!(
                validate_manifest(&manifest, digest).is_err(),
                "should have refused: {text}"
            );
        }
    }

    #[test]
    fn army_bin_edges_match_the_python_schema() {
        // Values from `bots/morpheus/artifact/manifest.json`, which the Python
        // derives from `logarithmic_army_bin_edges`.
        let edges = army_bin_edges();
        assert_eq!(edges.len(), 17);
        assert_eq!(edges[0], 0.0);
        assert_eq!(edges[16], 4096.0);
        assert!((edges[1] - 1.681792830507429).abs() < 1e-15);
        assert!((edges[8] - 63.99999999999998).abs() < 1e-13);
    }
}
