//! The two things the process reads from outside itself.
//!
//! `wire` is the judge's stdio protocol; `json` is the format the config,
//! the deployment manifest and the telemetry trace are written in. Both are
//! format codecs with no game knowledge at all — nothing here knows what a
//! general is.

pub mod json;
pub mod wire;
