"""Break-the-source check for the unclejoe parity harness (port-plan §6).

A green parity run proves the two implementations agree on the cases it ran;
it cannot say whether those cases reach the behavior under test. This
applies one deliberate source mutation at a time, rebuilds on the fast
`mutation` profile (same float semantics — Rust does not reassociate floats
across opt levels), runs the smoke-scope parity drivers, and asserts they
FAIL. A mutation the harness misses is a hole in the corpus, not a pass.

A clean baseline build must pass first, or every "kill" is meaningless.

Usage: .venv/bin/python bots/unclejoe/tools/mutation_check.py [--json OUT]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BOT_DIR = Path(__file__).resolve().parent.parent
SRC = BOT_DIR / "src"

# (name, file, old, new, what-it-breaks)
MUTATIONS = [
    (
        "history-roll-direction",
        "board/obs.rs",
        "next.army_stack[CELLS..HISTORY * CELLS]\n"
        "        .copy_from_slice(&state.army_stack[..(HISTORY - 1) * CELLS]);",
        "next.army_stack[..(HISTORY - 1) * CELLS]\n"
        "        .copy_from_slice(&state.army_stack[CELLS..HISTORY * CELLS]);",
        "army history stack rolls the wrong way",
    ),
    (
        "seen-accumulation-or",
        "board/obs.rs",
        "next.seen[idx] = state.seen[idx] || scratch.visible[idx];",
        "next.seen[idx] = scratch.visible[idx];",
        "seen memory forgets instead of accumulating",
    ),
    (
        "pad-mountain-rule",
        "board/obs.rs",
        "state.mountains[idx] || (pad_mask(i, j) && scratch.visible[idx]);",
        "state.mountains[idx];",
        "visible padding never becomes a confirmed mountain",
    ),
    (
        "normalize-divisor",
        "board/obs.rs",
        "*v *= RECIP_50;",
        "*v *= RECIP_5;",
        "a divide-by-50 site divides by 5",
    ),
    (
        "timestep-decay-off-by-one",
        "board/obs.rs",
        "state.last_enemy_army_seen_timestep[idx] + 1.0;",
        "state.last_enemy_army_seen_timestep[idx] + 2.0;",
        "channel 21's counter counts twice as fast",
    ),
    (
        "qk-proj-swap",
        "nn/net.rs",
        "q: take_linear(&mut tensors, &format!(\"{p}.attn.q_proj\"), EMBED, EMBED)?,\n"
        "                k: take_linear(&mut tensors, &format!(\"{p}.attn.k_proj\"), EMBED, EMBED)?,",
        "q: take_linear(&mut tensors, &format!(\"{p}.attn.k_proj\"), EMBED, EMBED)?,\n"
        "                k: take_linear(&mut tensors, &format!(\"{p}.attn.q_proj\"), EMBED, EMBED)?,",
        "the R2 leaf-misalignment bug, planted deliberately",
    ),
    (
        "softmax-scale",
        "nn/net.rs",
        "let scale = (HEAD_DIM as f64).sqrt();",
        "let scale = HEAD_DIM as f64;",
        "attention scale sqrt(48) becomes 48",
    ),
    (
        "argmax-tie-break",
        "board/action.rs",
        "if v > best_v {",
        "if v >= best_v {",
        "argmax returns the last maximum instead of the first",
    ),
    (
        "mask-pass-channel",
        "board/obs.rs",
        "for v in &mut penalties[8 * CELLS..9 * CELLS] {\n        *v = 0.0;\n    }",
        "",
        "the pass action is masked out",
    ),
    # Selection-plan S1 plants. All four are killed by the crate's unit
    # tests in `board/select.rs` — the corpus cannot see them: a turn-blind
    # seed still varies with the board, a masked logit already carries −1e9,
    # and the sign/scale of the noise only shows in the draw's distribution.
    (
        "select-turn-blind",
        "board/select.rs",
        "let seed = splitmix64(digest ^ (turn as u32 as u64).wrapping_mul(TURN_SALT));",
        "let seed = splitmix64(digest);",
        "the draw ignores the turn, so a revisited position repeats its noise",
    ),
    (
        "select-masked-noise",
        "board/select.rs",
        "if penalty != 0.0 {\n            continue;\n        }",
        "let _ = penalty;",
        "noise reaches masked entries, so an illegal action can be lifted",
    ),
    (
        "select-temperature-scale",
        "board/select.rs",
        "score += t * gumbel(seed, i);",
        "score += gumbel(seed, i);",
        "the noise ignores the temperature",
    ),
    (
        "select-noise-sign",
        "board/select.rs",
        "score += t * gumbel(seed, i);",
        "score -= t * gumbel(seed, i);",
        "subtracted noise samples a distribution that is not softmax(logits/T)",
    ),
    # S4 trail plants (docs/research/strategies/joe-rs-noundo.md), killed
    # by the crate's unit tests: the penalty must tax exactly the moves
    # landing on trail cells, the ring must hold the whole window, and the
    # sign must be a tax.
    (
        "trail-approach",
        "board/select.rs",
        "let (sr, sc) = (r as i32 - dr, c as i32 - dc);",
        "let (sr, sc) = (r as i32 + dr, c as i32 + dc);",
        "the penalty lands on moves leaving the trail cell, not entering it",
    ),
    (
        "trail-window",
        "board/select.rs",
        "self.cells.push(cell);\n        if self.cells.len() > self.cap {\n            self.cells.remove(0);\n        }",
        "self.cells.clear();\n        self.cells.push(cell);",
        "the ring remembers one source, so a period-4 circle closes untaxed",
    ),
    (
        "trail-sign",
        "board/select.rs",
        "logits[d * CELLS + src] -= delta;",
        "logits[d * CELLS + src] += delta;",
        "the penalty rewards re-entering the trail instead of taxing it",
    ),
    (
        "trail-ownership",
        "board/select.rs",
        "if r < h && c < w && raw[CH_OWNED * h * w + r * w + c] != 0.0 && !taxed.contains(&(r, c)) {",
        "if r < h && c < w && !taxed.contains(&(r, c)) {",
        "a trail cell the enemy captured stays taxed, so the retake is throttled",
    ),
    (
        "trail-exemption",
        "board/select.rs",
        "let mut has_free_exit = false;",
        "let mut has_free_exit = true;",
        "a stack encircled by hills has its only way out taxed",
    ),
]

PYTEST = [".venv/bin/pytest", "bots/unclejoe/tests/", "-m", "joe", "-q", "-x"]


def build(profile: str) -> bool:
    r = subprocess.run(
        ["cargo", "build", "--profile", profile],
        cwd=BOT_DIR, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-1500:])
    return r.returncode == 0


def run_harness(binary: Path) -> bool:
    """The full harness: parity drivers plus the crate's unit tests. The
    unit tests are load-bearing for behaviors no fixture can reach — an
    exact logit tie never occurs in real frames, so only
    `action::tests::argmax_is_first_max` can see a tie-break flip."""
    env = dict(os.environ)
    env["JOE_RS_BIN"] = str(binary)
    env["JOE_RS_PARITY_SMOKE"] = "1"
    r = subprocess.run(PYTEST, cwd=BOT_DIR.parent.parent, env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        return False
    r = subprocess.run(["cargo", "test", "--profile", "mutation"],
                       cwd=BOT_DIR, capture_output=True, text=True)
    return r.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None,
                        help="write the per-mutation record here")
    args = parser.parse_args()

    binary = BOT_DIR / "target" / "mutation" / "unclejoe"

    print("[mutation] baseline build + smoke parity (must pass)")
    assert build("mutation"), "baseline build failed"
    assert run_harness(binary), (
        "baseline smoke parity FAILED on the mutation profile — every kill "
        "below would be meaningless")
    print("[mutation] baseline green")

    results = []
    killed = 0
    for name, fname, old, new, what in MUTATIONS:
        path = SRC / fname
        original = path.read_text()
        if old not in original:
            raise SystemExit(f"{name}: pattern not found in {fname} — source drifted")
        path.write_text(original.replace(old, new, 1))
        try:
            t0 = time.time()
            if not build("mutation"):
                raise SystemExit(f"{name}: mutated source failed to BUILD — "
                                 f"rewrite the mutation to be type-correct")
            caught = not run_harness(binary)
            dt = time.time() - t0
        finally:
            path.write_text(original)
        status = "KILLED" if caught else "SURVIVED"
        print(f"[mutation] {name}: {status} ({what}) [{dt:.0f}s]")
        results.append({"name": name, "file": fname, "breaks": what, "killed": caught})
        killed += caught

    # Restore a clean release-profile state for whoever runs next.
    build("mutation")

    print(f"[mutation] {killed}/{len(MUTATIONS)} killed")
    if args.json:
        args.json.write_text(json.dumps(
            {"total": len(MUTATIONS), "killed": killed, "mutations": results},
            indent=2) + "\n")
        print(f"[mutation] wrote {args.json}")
    if killed != len(MUTATIONS):
        survivors = [r["name"] for r in results if not r["killed"]]
        sys.exit(f"survivors: {survivors} — the harness has blind spots")


if __name__ == "__main__":
    main()
