#!/usr/bin/env python3
"""What would a dependency actually cost in the submission zip?

    python bots/morpheus-joe/tools/dependency_budget.py
    python bots/morpheus-joe/tools/dependency_budget.py --crate sha2=0.10 --crate tract-onnx=0.21

Writes docs/research/measurements/morpheus-rs-dependency-budget.{json,md}.

The submission limits are 50 MB zipped, 512 MB unpacked, and **10,000 files**
(RULES.md §08). The file count is the one that binds, and until M2 the plan
asserted that `cargo vendor` "blows past 10k files easily" without anyone
having measured it. It does not: `sha2` is 5.6% of the cap. That wrong belief
had already been used to justify a decision, which is the reason this exists as
a tool rather than a number in a comment — the next dependency argument should
start from a measurement that can be re-run.

Each crate is vendored in isolation, so the figures are *marginal-if-first*.
Real graphs share crates, so adding two of these costs less than their sum.
Needs network (it resolves from crates.io); nothing here touches the bot.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BOT_DIR = Path(__file__).resolve().parents[1]
REPO = BOT_DIR.parents[1]
MEASUREMENTS = REPO / "docs" / "research" / "measurements"

# Judge limits, mirrored from arena.bundle so the two cannot drift.
sys.path.insert(0, str(REPO))
from arena.bundle import MAX_FILES, MAX_UNPACKED_BYTES  # noqa: E402

# Candidates worth knowing the price of: the hash we hand-rolled, and the
# inference engines §3 is choosing between at M3.
DEFAULT_CRATES = (
    "sha2=0.10",
    "candle-core=0.9",
)


def _cargo_env() -> dict[str, str]:
    env = os.environ.copy()
    cargo_bin = Path.home() / ".cargo" / "bin"
    if cargo_bin.is_dir():
        env["PATH"] = f"{cargo_bin}{os.pathsep}{env.get('PATH', '')}"
    return env


def measure(spec: str) -> dict:
    """Vendor one crate into a scratch workspace and weigh the result."""
    name, _, version = spec.partition("=")
    with tempfile.TemporaryDirectory(prefix="dep-budget-") as tmp:
        root = Path(tmp)
        (root / "src").mkdir()
        (root / "src" / "main.rs").write_text("fn main() {}\n", encoding="utf-8")
        dep = f'{name} = {{ version = "{version or "*"}", default-features = false }}'
        (root / "Cargo.toml").write_text(
            "[package]\n"
            'name = "budget-probe"\nversion = "0.0.0"\nedition = "2021"\n\n'
            f"[dependencies]\n{dep}\n",
            encoding="utf-8",
        )
        vendor = subprocess.run(
            ["cargo", "vendor", "--versioned-dirs", "vendor"],
            cwd=str(root),
            env=_cargo_env(),
            capture_output=True,
            text=True,
        )
        if vendor.returncode != 0:
            return {"crate": name, "error": vendor.stderr[-500:].strip()}

        vendor_dir = root / "vendor"
        files = [p for p in vendor_dir.rglob("*") if p.is_file()]
        per_crate = sorted(
            (
                {
                    "crate": d.name,
                    "files": sum(1 for p in d.rglob("*") if p.is_file()),
                    "bytes": sum(p.stat().st_size for p in d.rglob("*") if p.is_file()),
                }
                for d in vendor_dir.iterdir()
                if d.is_dir()
            ),
            key=lambda row: -row["files"],
        )
        total_files = len(files)
        total_bytes = sum(p.stat().st_size for p in files)
        return {
            "crate": name,
            "version": version or "latest",
            "crates": len(per_crate),
            "files": total_files,
            "bytes": total_bytes,
            "files_pct_of_cap": round(100.0 * total_files / MAX_FILES, 2),
            "bytes_pct_of_cap": round(100.0 * total_bytes / MAX_UNPACKED_BYTES, 2),
            # The heaviest few explain most of the total and are usually where
            # a feature flag could cut it down.
            "heaviest": per_crate[:5],
        }


def _markdown(report: dict) -> str:
    lines = [
        "# Morpheus-rs dependency budget",
        "",
        "What a crate would cost in the submission zip, measured rather than",
        "assumed. Limits: 50 MB zipped, 512 MB unpacked, **10,000 files**",
        "(RULES.md §08); the file count is the binding one.",
        "",
        f"- Measured: {report['measured_at']}",
        f"- Toolchain: `{report['cargo']}`",
        "",
        "Each crate is vendored **in isolation**, with default features off, so",
        "these are marginal-if-first figures. Real graphs share crates, so two",
        "of them together cost less than the sum of their rows.",
        "",
        "| crate | crates in graph | files | % of 10k cap | unpacked |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["crates"]:
        if "error" in row:
            lines.append(f"| `{row['crate']}` | — | — | — | failed: {row['error'][:40]} |")
            continue
        lines.append(
            f"| `{row['crate']}` {row['version']} | {row['crates']} | {row['files']:,} "
            f"| {row['files_pct_of_cap']}% | {row['bytes'] / 1e6:.1f} MB |"
        )
    lines += [
        "",
        "## Why this exists",
        "",
        "The plan's §1 asserted that `cargo vendor` \"blows past 10k files",
        "easily\", and that claim had already been used to justify hand-writing",
        "SHA-256 rather than depending on `sha2`. The measurement does not",
        "support it: `sha2` is 5.6% of the cap. The hand-written implementation",
        "is still the right call for other reasons — hashing costs 0.001 ms p99,",
        "so the library's speed advantage is worthless, and the digests are",
        "dictionary keys rather than a security boundary — but the budget was",
        "not one of them.",
        "",
        "The budget is real where it binds: an inference crate is two orders of",
        "magnitude heavier than a hash, and M3 picks one.",
        "",
        "## Heaviest crates per graph",
        "",
    ]
    for row in report["crates"]:
        if "error" in row:
            continue
        lines += [f"**`{row['crate']}`**", ""]
        for sub in row["heaviest"]:
            lines.append(
                f"- `{sub['crate']}` — {sub['files']:,} files, "
                f"{sub['bytes'] / 1e6:.1f} MB"
            )
        lines.append("")
    lines += [
        "A single crate usually dominates. `sha2` pulls `cpufeatures` → `libc`",
        "for runtime CPU detection, and `libc` alone is 404 of its 565 files —",
        "worth knowing before assuming a small crate is small.",
        "",
        "## Re-running",
        "",
        "```bash",
        "python bots/morpheus-joe/tools/dependency_budget.py --crate tract-onnx=0.21",
        "```",
        "",
        "Needs network. Nothing here touches the bot or its lock file; every",
        "probe is vendored in a scratch directory and thrown away.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    import datetime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crate", action="append", default=None, help="name=version")
    parser.add_argument("--output", type=Path, default=MEASUREMENTS / "morpheus-joe-dependency-budget.json")
    args = parser.parse_args(argv)

    if shutil.which("cargo", path=_cargo_env()["PATH"]) is None:
        print("cargo not found", file=sys.stderr)
        return 2

    specs = args.crate or list(DEFAULT_CRATES)
    cargo = subprocess.run(
        ["cargo", "--version"], env=_cargo_env(), capture_output=True, text=True
    ).stdout.strip()

    rows = []
    for spec in specs:
        print(f"[budget] vendoring {spec}…", flush=True)
        rows.append(measure(spec))

    report = {
        "measured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "cargo": cargo,
        "limits": {"files": MAX_FILES, "unpacked_bytes": MAX_UNPACKED_BYTES},
        "crates": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"wrote {args.output.with_suffix('.md')}")
    for row in rows:
        if "error" not in row:
            print(f"  {row['crate']:<14} {row['files']:>5} files  {row['files_pct_of_cap']:>5}% of cap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
