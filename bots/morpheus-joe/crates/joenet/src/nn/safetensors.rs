//! A read-only safetensors loader for f32 weights.
//!
//! Ported from morpheus-rs (`crates/core/src/nn/safetensors.rs`). The format
//! is `<u64 little-endian header length><JSON header><tensor data>`, where
//! each header entry names a dtype, a row-major shape, and a `[begin, end)`
//! byte range into the data region. That is the whole specification this bot
//! needs, and `tools/convert_artifact.py` writes the file the reference
//! Python implementation reads back unchanged — so the layout assumptions here
//! are checked by a third party at conversion time, not only by their mirror
//! image in the writer.
//!
//! No mmap. The port plan (§5) said weights load by mmap, but reading the file
//! into one owned `Vec<f32>` at startup is the better shape here as it was for
//! morpheus: the artifact is ~33 MB, it is read once inside the ~10 s
//! first-move grace, and an owned buffer gives aligned `&[f32]` slices with no
//! unsafe and no lifetime tied to a mapping. This is also what dropped the
//! `memmap2` dependency.
//!
//! Loudness over leniency: a truncated file, an overlapping tensor, a dtype
//! that is not F32, or a shape whose product disagrees with its byte range are
//! all errors at load. The alternative is a bot that starts and plays nonsense.

use std::collections::BTreeMap;
use std::path::Path;

use crate::io::json::{parse, Json};

/// One tensor: its shape and its slice of the flat f32 payload.
pub struct TensorView<'a> {
    pub shape: Vec<usize>,
    pub data: &'a [f32],
}

pub struct SafeTensors {
    data: Vec<f32>,
    /// name -> (shape, offset in `data` measured in f32 elements, length)
    index: BTreeMap<String, (Vec<usize>, usize, usize)>,
    metadata: BTreeMap<String, String>,
}

impl SafeTensors {
    pub fn load(path: &Path) -> Result<Self, String> {
        let raw = std::fs::read(path)
            .map_err(|e| format!("reading {}: {e}", path.display()))?;
        Self::from_bytes(&raw)
    }

    pub fn from_bytes(raw: &[u8]) -> Result<Self, String> {
        if raw.len() < 8 {
            return Err("safetensors file shorter than its length prefix".to_string());
        }
        let header_len = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
        let body_start = 8usize
            .checked_add(header_len)
            .ok_or_else(|| "header length overflows".to_string())?;
        if body_start > raw.len() {
            return Err(format!(
                "header claims {header_len} bytes but the file has {}",
                raw.len() - 8
            ));
        }
        let header_text = std::str::from_utf8(&raw[8..body_start])
            .map_err(|_| "header is not UTF-8".to_string())?;
        let header = parse(header_text)?;
        let entries = match &header {
            Json::Object(map) => map,
            _ => return Err("header is not a JSON object".to_string()),
        };

        let body = &raw[body_start..];
        if body.len() % 4 != 0 {
            return Err(format!("data region {} bytes is not a whole f32 count", body.len()));
        }

        let mut metadata = BTreeMap::new();
        let mut index: BTreeMap<String, (Vec<usize>, usize, usize)> = BTreeMap::new();
        // Byte ranges must tile the data region without overlapping. Two
        // tensors sharing bytes is legal in the format and never legal here;
        // it would mean the writer lost track of an offset.
        let mut claimed: Vec<(usize, usize, String)> = Vec::new();

        for (name, entry) in entries {
            if name == "__metadata__" {
                if let Json::Object(meta) = entry {
                    for (k, v) in meta {
                        let text = v
                            .as_str()
                            .ok_or_else(|| format!("__metadata__.{k} is not a string"))?;
                        metadata.insert(k.clone(), text.to_string());
                    }
                    continue;
                }
                return Err("__metadata__ is not an object".to_string());
            }
            let dtype = entry.str_field("dtype").map_err(|e| format!("{name}: {e}"))?;
            if dtype != "F32" {
                return Err(format!("{name}: dtype {dtype} is not F32"));
            }
            let shape: Vec<usize> = entry
                .field("shape")
                .map_err(|e| format!("{name}: {e}"))?
                .as_array()
                .ok_or_else(|| format!("{name}: shape is not an array"))?
                .iter()
                .map(|v| {
                    v.as_i64()
                        .filter(|n| *n >= 0)
                        .map(|n| n as usize)
                        .ok_or_else(|| format!("{name}: shape holds a non-integer"))
                })
                .collect::<Result<_, _>>()?;
            let offsets = entry
                .field("data_offsets")
                .map_err(|e| format!("{name}: {e}"))?
                .as_array()
                .ok_or_else(|| format!("{name}: data_offsets is not an array"))?;
            if offsets.len() != 2 {
                return Err(format!("{name}: data_offsets needs exactly two entries"));
            }
            let begin = offsets[0]
                .as_i64()
                .filter(|n| *n >= 0)
                .ok_or_else(|| format!("{name}: bad data_offsets"))? as usize;
            let end = offsets[1]
                .as_i64()
                .filter(|n| *n >= 0)
                .ok_or_else(|| format!("{name}: bad data_offsets"))? as usize;
            if end < begin || end > body.len() {
                return Err(format!(
                    "{name}: data_offsets [{begin}, {end}) outside a {} byte payload",
                    body.len()
                ));
            }
            if begin % 4 != 0 || (end - begin) % 4 != 0 {
                return Err(format!("{name}: data_offsets not f32-aligned"));
            }
            let count: usize = shape.iter().product();
            if (end - begin) / 4 != count {
                return Err(format!(
                    "{name}: shape {shape:?} needs {count} floats, range holds {}",
                    (end - begin) / 4
                ));
            }
            claimed.push((begin, end, name.clone()));
            index.insert(name.clone(), (shape, begin / 4, count));
        }

        claimed.sort();
        for pair in claimed.windows(2) {
            if pair[0].1 > pair[1].0 {
                return Err(format!(
                    "tensors {:?} and {:?} share bytes",
                    pair[0].2, pair[1].2
                ));
            }
        }

        let mut data = vec![0f32; body.len() / 4];
        for (i, chunk) in body.chunks_exact(4).enumerate() {
            data[i] = f32::from_le_bytes(chunk.try_into().unwrap());
        }

        Ok(Self { data, index, metadata })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.index.keys().map(|s| s.as_str())
    }

    #[allow(dead_code)]
    pub fn metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(|s| s.as_str())
    }

    pub fn get(&self, name: &str) -> Result<TensorView<'_>, String> {
        let (shape, offset, count) = self
            .index
            .get(name)
            .ok_or_else(|| format!("artifact has no tensor {name:?}"))?;
        Ok(TensorView {
            shape: shape.clone(),
            data: &self.data[*offset..*offset + *count],
        })
    }

    /// Fetch with the shape the caller expects, so a re-trained checkpoint with
    /// different widths fails here rather than as silent garbage downstream.
    pub fn get_shaped(&self, name: &str, shape: &[usize]) -> Result<&[f32], String> {
        let view = self.get(name)?;
        if view.shape != shape {
            return Err(format!(
                "{name}: shape {:?} != expected {shape:?}",
                view.shape
            ));
        }
        Ok(view.data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blob(header: &str, floats: &[f32]) -> Vec<u8> {
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(header.as_bytes());
        for f in floats {
            out.extend_from_slice(&f.to_le_bytes());
        }
        out
    }

    #[test]
    fn reads_shapes_metadata_and_values() {
        let header = r#"{"__metadata__":{"k":"v"},"a":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"b":{"dtype":"F32","shape":[1],"data_offsets":[16,20]}}"#;
        let st = SafeTensors::from_bytes(&blob(header, &[1.0, 2.0, 3.0, 4.0, 9.5])).unwrap();
        assert_eq!(st.metadata("k"), Some("v"));
        assert_eq!(st.get_shaped("a", &[2, 2]).unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(st.get_shaped("b", &[1]).unwrap(), &[9.5]);
        assert!(st.get_shaped("a", &[4]).is_err());
        assert!(st.get("missing").is_err());
    }

    #[test]
    fn rejects_broken_files() {
        let cases = [
            // dtype we cannot read
            r#"{"a":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}"#,
            // shape disagrees with the byte range
            r#"{"a":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}}"#,
            // range runs past the payload
            r#"{"a":{"dtype":"F32","shape":[8],"data_offsets":[0,32]}}"#,
            // two tensors over the same bytes
            r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}}"#,
        ];
        for header in cases {
            let raw = blob(header, &[1.0, 2.0, 3.0]);
            assert!(
                SafeTensors::from_bytes(&raw).is_err(),
                "should have rejected {header}"
            );
        }
        assert!(SafeTensors::from_bytes(&[0u8; 4]).is_err());
        // header length longer than the file
        let mut truncated = (999u64).to_le_bytes().to_vec();
        truncated.extend_from_slice(b"{}");
        assert!(SafeTensors::from_bytes(&truncated).is_err());
    }
}
