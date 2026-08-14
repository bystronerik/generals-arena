//! Strip a Rust source of everything the judge does not need to compile it.
//!
//! Reads one `.rs` file on stdin, writes the minified equivalent on stdout.
//! `tools/package_submission.py` runs it over `crates/**/*.rs` **on the way
//! into the zip only** — the repo sources are never touched, exactly as
//! `arena/bundle.py` treats the Python bots.
//!
//! Why this exists: the vendored variant ships source, and this crate's
//! comments are not incidental. They carry the FMA bug and its 277 ms
//! measurement, the BLAS reduction that the oracle cannot pin, NumPy's
//! unstable `argsort`, and the reasoning behind every tactical rule. A
//! submission is readable by whoever holds it; the design record does not have
//! to be part of it.
//!
//! The transform is a parse and a re-print, not a text rewrite. Ordinary `//`
//! comments never reach the AST, so they vanish for free; doc comments survive
//! as `#[doc]` attributes and `remove_docs` is what takes those. A regex over
//! the same problem eats code the first time a `//` appears inside a string
//! literal.
//!
//! What it deliberately does not do: rename anything. Python's bundler renames
//! locals because Python ships names at runtime; a compiled Rust binary does
//! not, and `strip = "symbols"` in the release profile removes the ones the
//! object file would have carried.

use std::io::{self, Read, Write};

fn main() {
    let mut source = String::new();
    if let Err(err) = io::stdin().read_to_string(&mut source) {
        eprintln!("[minify] reading stdin: {err}");
        std::process::exit(2);
    }

    // A parse failure is a hard stop, never a fallback to the original text.
    // Silently shipping the un-minified file would leave the packager
    // reporting a minified bundle it did not build.
    let parsed = match syn::parse_file(&source) {
        Ok(parsed) => parsed,
        Err(err) => {
            eprintln!("[minify] parse: {err}");
            std::process::exit(1);
        }
    };

    let minified = rustminify::minify_file(&rustminify::remove_docs(parsed));

    // Round-trip check, because the output is what gets compiled at intake and
    // a build failure there is a rejected submission. Cheap: one more parse.
    if let Err(err) = syn::parse_file(&minified) {
        eprintln!("[minify] output does not re-parse: {err}");
        std::process::exit(1);
    }

    let mut stdout = io::stdout();
    if stdout.write_all(minified.as_bytes()).is_err() || stdout.flush().is_err() {
        std::process::exit(2);
    }
}
