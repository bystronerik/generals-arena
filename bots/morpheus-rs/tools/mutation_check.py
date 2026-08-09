#!/usr/bin/env python3
"""Does the parity harness actually notice when the port is wrong?

    python bots/morpheus-rs/tools/mutation_check.py
    python bots/morpheus-rs/tools/mutation_check.py --full   # against the corpus

A green parity run proves the two implementations agree on the cases it ran.
It says nothing about whether those cases *reach* the behaviour under test —
and on this port that gap was real twice over. Dropping the 50-tick army
growth passed 672 recorded transition cases, because no recorded state sat at
`time % 50 == 49`. Dropping the NumPy negative-index wrap passed too: the wrap
reads row `h-1`, and the competition preset pads smaller boards to 21×21 with
mountains, so on a padded board that row is border and can never be owned.

So this deliberately breaks one behaviour at a time, in the real source, and
checks the harness reports a mismatch. A mutation that **survives** is not a
bug in the port — it means nothing in the case set can tell the two behaviours
apart, which is a hole in the harness and gets recorded as one.

Sources are restored from an in-memory copy in a `finally`, and the binary is
rebuilt at the end, so an interrupted run leaves the tree clean.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

BOT_DIR = Path(__file__).resolve().parents[1]
REPO = BOT_DIR.parents[1]
SRC = BOT_DIR / "crates" / "core" / "src"


# Which parity surfaces can possibly see a break in each source file.
#
# Running all twenty surfaces per mutation was the honest default until M4
# made it the dominant cost: eighty-odd mutations times a torch import and two
# million integers through a pipe is hours, and the surfaces a `reservoir.rs`
# mutation could ever reach is one. The map is deliberately **generous** — the
# board layer feeds everything downstream, so `transition.rs` still runs the
# whole set — because an over-narrow entry would report a caught mutation as a
# survivor and be read as a coverage hole.
FILE_SURFACES: dict[str, tuple[str, ...]] = {
    # The board layer is upstream of every belief and every tensor.
    "transition.rs": (),
    "state.rs": (),
    "observe.rs": (),
    "action.rs": (),
    "memory.rs": (),
    "hashing.rs": ("hash", "propose"),
    "tensor.rs": ("tensor", "net", "prior"),
    "symmetry.rs": ("symmetry",),
    "network.rs": ("net", "prior"),
    "gemm.rs": ("net", "prior"),
    "rng.rs": ("npsum", "argsort", "summary", "propose", "filter",
               "rejuvenate", "maxent", "reservoir", "toplegal", "initbelief"),
    "belief.rs": ("summary", "filter", "rejuvenate", "maxent", "reservoir",
                  "initbelief"),
    "particle_summary.rs": ("summary",),
    "proposal.rs": ("propose", "rejuvenate", "maxent", "toplegal"),
    "recovery.rs": ("rejuvenate", "maxent"),
    "reservoir.rs": ("reservoir",),
    # M5. `tactics.rs` reaches the decision through five surfaces and is also
    # upstream of the search's candidate sets, so it maps to all of them.
    "matrix.rs": ("matrix",),
    "tactics.rs": ("playmask", "candidates", "planners", "shaping", "constrain",
                   "decide"),
    "runtime.rs": ("decide", "runtime", "search"),
    "tree.rs": ("search",),
    "search.rs": ("search",),
}


@dataclass(frozen=True)
class Mutation:
    """One behaviour, deleted. `note` explains what it costs if it survives."""

    name: str
    file: str
    before: str
    after: str
    note: str = ""

    def surfaces(self) -> tuple[str, ...]:
        """The parity kinds worth running for this mutation; empty means all."""
        return FILE_SURFACES.get(self.file, ())


MUTATIONS: tuple[Mutation, ...] = (
    Mutation(
        "50-tick army growth",
        "transition.rs",
        "if state.time % 50 == 0 {",
        "if false && state.time % 50 == 0 {",
    ),
    Mutation(
        "even-tick structure growth",
        "transition.rs",
        "if state.time % 2 == 0 {",
        "if false && state.time % 2 == 0 {",
    ),
    Mutation(
        "numpy negative-index wrap",
        "state.rs",
        "let r = if row < 0 { row + self.h as i32 } else { row };",
        "let r = row;",
    ),
    Mutation(
        "deathtouch turn boundary",
        "transition.rs",
        "state.time >= turn",
        "state.time > turn",
    ),
    Mutation(
        "attacker wins ties",
        "transition.rs",
        "let attacker_wins = army > target_army;",
        "let attacker_wins = army >= target_army;",
    ),
    Mutation(
        "half-split rounds up",
        "transition.rs",
        "source_army.div_euclid(2)",
        "(source_army + 1).div_euclid(2)",
    ),
    Mutation(
        "a move may empty its source",
        "transition.rs",
        "raw.min(source_army - 1).max(0)",
        "raw.min(source_army).max(0)",
        note=(
            "Equivalent mutant, not a harness hole. `army_to_move` guards its "
            "invariant twice over — once in the `raw` formula and again in the "
            "clamp — so either alone is sufficient and deleting either is "
            "unobservable. The Python is written the same way; the redundancy "
            "is being ported faithfully, not introduced here."
        ),
    ),
    Mutation(
        "build ignores affordability",
        "transition.rs",
        "let affords = state.armies[at] >= cost[at];",
        "let affords = true;",
    ),
    Mutation(
        "build proximity surcharge dropped",
        "transition.rs",
        "cost[(r * w + c) as usize] += surcharge;",
        "cost[(r * w + c) as usize] += 0;",
    ),
    Mutation(
        "loser cells not transferred",
        "transition.rs",
        "state.ownership[winner][i] |= loser_mask[i];",
        "state.ownership[winner][i] |= false;",
    ),
    Mutation(
        "neutral flag survives a loss",
        "transition.rs",
        "state.ownership_neutral[i] &= !loser_mask[i];",
        "state.ownership_neutral[i] &= true;",
        note=(
            "Unobservable by construction: `ownership_neutral` and "
            "`ownership[seat]` are disjoint on every well-formed state, so the "
            "mask it clears is always empty. Faithful to the Python, and only "
            "a malformed state could tell the difference."
        ),
    ),
    Mutation(
        "visibility is 4-neighbour",
        "observe.rs",
        "for dr in -1..=1 {",
        "for dr in 0..=0 {",
    ),
    Mutation(
        "structures under fog read as fog",
        "observe.rs",
        "TYPE_STRUCTURE_FOG\n        } else {",
        "TYPE_FOG\n        } else {",
    ),
    Mutation(
        "fogged armies leak",
        "observe.rs",
        "obs.army_grid[i] = if visible[i] { state.armies[i] } else { 0 };",
        "obs.army_grid[i] = state.armies[i];",
    ),
    Mutation(
        "half-split legal at two army",
        "action.rs",
        "let allow_half = src_army > 2;",
        "let allow_half = src_army >= 2;",
    ),
    Mutation(
        "build on unexplored ground",
        "action.rs",
        "if !memory.known_passable_base[at] {",
        "if false && !memory.known_passable_base[at] {",
    ),
    Mutation(
        "moves into mountains are legal",
        "action.rs",
        "if dest_t == TYPE_MOUNTAIN || dest_t == TYPE_STRUCTURE_FOG {",
        "if false {",
    ),
    # --- M2: memory, hashing, tensor, symmetry ---
    Mutation(
        "a fogged structure is always a mountain",
        "memory.rs",
        "next.known_castle[i] = was_passable;\n                next.known_mountain[i] = !was_passable;",
        "next.known_castle[i] = false;\n                next.known_mountain[i] = true;",
    ),
    Mutation(
        "plain fog erases a known castle",
        "memory.rs",
        "next.known_passable_base[i] = true;\n                next.known_mountain[i] = false;\n            }",
        "next.known_passable_base[i] = true;\n                next.known_mountain[i] = false;\n                next.known_castle[i] = false;\n            }",
    ),
    Mutation(
        "castle owner refreshes on any sighting",
        "memory.rs",
        "if t == TYPE_CASTLE {\n                next.remembered_castle_owner[i] = owner;\n            }",
        "next.remembered_castle_owner[i] = owner;",
    ),
    Mutation(
        "dynamic memory refreshes through fog",
        "memory.rs",
        "if visible {\n            next.ever_visible[i] = true;",
        "if true {\n            next.ever_visible[i] = true;",
    ),
    Mutation(
        "owner planes hashed as int32",
        "hashing.rs",
        "hasher.update(&[value as i8 as u8]);\n    }\n    for &value in &memory.remembered_army[..n] {",
        "hasher.update(&value.to_le_bytes());\n    }\n    for &value in &memory.remembered_army[..n] {",
    ),
    Mutation(
        "board dimensions left out of the digest",
        "hashing.rs",
        "hasher.update(&(memory.h as i32).to_le_bytes());",
        "hasher.update(&[]);",
    ),
    Mutation(
        "army compression in single precision",
        "tensor.rs",
        "let out = x.max(0.0).ln_1p() / denom;",
        "let out = ((x.max(0.0) as f32).ln_1p() / denom as f32) as f64;",
    ),
    Mutation(
        "coordinate planes in double precision",
        "tensor.rs",
        "set(P_ROW_FROM_GENERAL, r, c, (r as f32 - gr as f32) / 20.0);",
        "set(P_ROW_FROM_GENERAL, r, c, ((r as f64 - gr as f64) / 20.0) as f32);",
        note=(
            "Equivalent on this board: rows run 0..20 and the general sits on "
            "an integer cell, so (r - gr)/20 is exactly representable in f32 "
            "and the two widths agree bit for bit. The width still matters as "
            "documentation of what the Python does — a plane with non-integer "
            "inputs would diverge."
        ),
    ),
    Mutation(
        "sight age left unclamped",
        "tensor.rs",
        "set(P_SIGHT_AGE, r, c, age.clamp(0.0, 1.0) as f32);",
        "set(P_SIGHT_AGE, r, c, age as f32);",
        note=(
            "Unreachable by construction. Age is only computed for a cell that "
            "has been seen, so `last_seen_turn` is a real turn in [0, turn]; "
            "the numerator is therefore in [0, 1200] and the quotient already "
            "in [0, 1]. The clamp can only bind on a memory the game cannot "
            "produce. Faithful to the Python, which clamps for the same "
            "belt-and-braces reason."
        ),
    ),
    Mutation(
        "neutral plane ignores mountains",
        "tensor.rs",
        "let passable_vis = visible && !memory.known_mountain[i];",
        "let passable_vis = visible;",
    ),
    Mutation(
        "half-splits paint as full moves",
        "tensor.rs",
        "if split == 1 { 0.5 } else { 1.0 };",
        "1.0;",
    ),
    Mutation(
        "bulk growth countdown off by one",
        "tensor.rs",
        "(((50 - ((turn + 1) % 50)) % 50) as f64) / 49.0",
        "(((50 - (turn % 50)) % 50) as f64) / 49.0",
    ),
    Mutation(
        "constants painted over the padding",
        "tensor.rs",
        "for r in 0..h {\n            for c in 0..w {\n                tensor[plane * PLANE_CELLS + r * PAD + c] = v;",
        "for r in 0..PAD {\n            for c in 0..PAD {\n                tensor[plane * PLANE_CELLS + r * PAD + c] = v;",
    ),
    Mutation(
        "rot90+flip composed in the other order",
        "symmetry.rs",
        "Symmetry::Rot90FlipH => {\n                let (r2, c2) = rot90_rc(r, c);\n                flip_h_rc(r2, c2)\n            }",
        "Symmetry::Rot90FlipH => {\n                let (r2, c2) = flip_h_rc(r, c);\n                rot90_rc(r2, c2)\n            }",
    ),
    Mutation(
        "directions do not follow rotations",
        "symmetry.rs",
        "Symmetry::Rot90 => rot90[d],",
        "Symmetry::Rot90 => d,",
    ),
    # --- M3: the network graph and the arithmetic around it ------------------
    #
    # The `net` surface compares against TorchScript with a *tolerance*, not
    # bit-exactly, which makes these mutations more load-bearing than the
    # tier-1 ones: a budget wide enough to absorb a real mistake is a budget
    # that proves nothing. Several below are deliberately small — an eps, a
    # padding column, a precision — to find out where the budget stops seeing.
    Mutation(
        "group norm epsilon an order of magnitude out",
        "network.rs",
        "const GROUP_NORM_EPS: f64 = 1e-5;",
        "const GROUP_NORM_EPS: f64 = 1e-4;",
    ),
    Mutation(
        "group norm sums over the pad columns",
        "network.rs",
        "let plane = &buf[c * STRIDE..c * STRIDE + CELLS];",
        "let plane = &buf[c * STRIDE..(c + 1) * STRIDE];",
        note=(
            "Equivalent, and equivalent for a reason worth having: the seven "
            "pad columns per plane are zero, so summing them adds nothing to "
            "either accumulator. This mutation surviving is evidence *for* the "
            "zero-padding invariant the module doc claims. The observable half "
            "of 'include the pad columns' is the divisor, which is the next "
            "mutation and is caught."
        ),
    ),
    Mutation(
        "group norm divides by the padded width",
        "network.rs",
        "let n = (per_group * CELLS) as f64;",
        "let n = (per_group * STRIDE) as f64;",
    ),
    Mutation(
        "group norm over the whole layer, not eight groups",
        "network.rs",
        "pub const GROUP_NORM_GROUPS: usize = 8;",
        "pub const GROUP_NORM_GROUPS: usize = 4;",
    ),
    Mutation(
        "dilation cycle flattened to 1",
        "network.rs",
        "pub const DILATION_CYCLE: [usize; 3] = [1, 2, 4];",
        "pub const DILATION_CYCLE: [usize; 3] = [1, 1, 1];",
    ),
    Mutation(
        "ReLU6 with no upper clamp",
        "network.rs",
        "*v = v.clamp(0.0, 6.0);",
        "*v = v.max(0.0);",
    ),
    Mutation(
        "residual connection dropped",
        "network.rs",
        "s.trunk[i] += s.residual[i];",
        "s.trunk[i] += 0.0;",
    ),
    Mutation(
        # Re-aimed at M4. The M3 addendum's halo rewrite replaced the `ky`/`kx`
        # loop with a flat tap index, so the old pattern stopped matching and
        # the tool reported it `stale` — which is the point of failing on a
        # misaimed mutation rather than counting it as a survivor.
        "depthwise kernel transposed",
        "network.rs",
        "let dy = (tap / 3) as isize - 1;\n            let dx = (tap % 3) as isize - 1;",
        "let dy = (tap % 3) as isize - 1;\n            let dx = (tap / 3) as isize - 1;",
    ),
    Mutation(
        "global mean ignores the board mask",
        "network.rs",
        "sum += (v * mask[p]) as f64;",
        "sum += v as f64;",
    ),
    Mutation(
        "global max ignores the board mask",
        "network.rs",
        "let candidate = if mask[p] < 0.5 { MASK_FILL } else { v };",
        "let candidate = v;",
    ),
    Mutation(
        "global mean channels reversed",
        "network.rs",
        "self.scratch.feat[c] = (sum / safe) as f32;",
        "self.scratch.feat[TRUNK - 1 - c] = (sum / safe) as f32;",
    ),
    Mutation(
        "stem weights laid out with the tap axis outermost",
        "network.rs",
        "stem[oc * IN_CHANNELS * 9 + ic * 9 + k] =",
        "stem[oc * IN_CHANNELS * 9 + k * IN_CHANNELS + ic] =",
    ),
    Mutation(
        "1x1 head bias dropped",
        "network.rs",
        "Some(&head.bias),",
        "None,",
    ),
    Mutation(
        "softmax in double precision",
        "network.rs",
        "let e = (v - max).exp();",
        "let e = ((v - max) as f64).exp() as f32;",
        note=(
            "Equivalent at this width. The result is rounded back to f32 "
            "immediately, and `exp` is correctly rounded closely enough that "
            "computing it in double and narrowing lands on the same float for "
            "every logit the network produces. What the f32 contract actually "
            "pins is the *sum and the division*, which this does not touch; "
            "widening those needs `exps` widened too, which is a rewrite "
            "rather than a one-line break, and the `prior` surface's 5e-6 cap "
            "is what holds them."
        ),
    ),
    Mutation(
        "illegal actions keep their softmax mass",
        "network.rs",
        ".map(|i| if mask[i] { (exps[i] / total) as f64 } else { 0.0 })",
        ".map(|i| (exps[i] / total) as f64)",
        note=(
            "Equivalent because the fill underflows. An illegal logit is set "
            "to `f32::MIN`, so `exp(MIN - max)` is exactly 0.0 and the "
            "multiply by the mask removes nothing. The Python multiplies too, "
            "for the same non-reason. Keeping the explicit zero is still "
            "right: it makes 'illegal means zero' a property of the code "
            "rather than of float underflow, and the harness checks it "
            "without tolerance."
        ),
    ),
    Mutation(
        "backup value not negated off the root",
        "network.rs",
        "    if from_root {\n        v\n    } else {\n        -v\n    }",
        "    let _ = from_root;\n    v",
    ),
    Mutation(
        "WDL value is win minus draw",
        "network.rs",
        "((exps[0] / total) - (exps[2] / total)) as f64",
        "((exps[0] / total) - (exps[1] / total)) as f64",
    ),
    Mutation(
        "GEMM bias applied to the wrong row",
        "gemm.rs",
        "let add = bias.map_or(0.0, |v| v[m0 + i]);",
        "let add = bias.map_or(0.0, |v| v[m0]);",
    ),
    Mutation(
        "linear head drops its bias",
        "gemm.rs",
        "y[m] = sum + bias[m];",
        "y[m] = sum;",
    ),
    # --- M4: the belief filter, and the NumPy behaviours under it ------------
    #
    # This milestone's surfaces have a new failure mode the earlier ones did
    # not: a wrong answer can hide behind the *replay*. `Replay` hands back the
    # oracle's index whatever the port computed, so a mutation that corrupts a
    # probability, a weight, or an ESS can still sample the same action. The
    # mutations below deliberately target that seam.
    Mutation(
        "numpy sum reassociated as a plain loop",
        "rng.rs",
        "let mut r = [0.0f64; 8];",
        "let mut res = 0.0;\n        for &value in values {\n            res += value;\n        }\n        return res;\n        #[allow(unreachable_code)]\n        let mut r = [0.0f64; 8];",
    ),
    Mutation(
        "argsort made stable",
        "rng.rs",
        "aquicksort(&negated, &mut order);",
        "order.sort_by(|&a, &b| negated[a].partial_cmp(&negated[b]).unwrap());",
    ),
    Mutation(
        "argsort skips the heapsort depth limit",
        "rng.rs",
        "if cdepth < 0 {",
        "if false && cdepth < 0 {",
        note=(
            "Equivalent on every case here. The depth limit is `2*floor(log2 "
            "n)` and median-of-three quicksort only exceeds it on inputs built "
            "to defeat the pivot choice; a probability vector, uniform or not, "
            "is not one. The fallback is ported because NumPy has it and a "
            "future adversarial input would need it, not because anything "
            "reaches it today."
        ),
    ),
    Mutation(
        "ESS normalizes before squaring, not after",
        "belief.rs",
        "let squares: Vec<f64> = normalized.iter().map(|w| w * w).collect();",
        "let squares: Vec<f64> = weights.iter().map(|w| w * w).collect();",
    ),
    Mutation(
        "normalize_weights keeps negative weights",
        "belief.rs",
        ".map(|p| p.with_weight(p.weight.max(0.0) / total))",
        ".map(|p| p.with_weight(p.weight / total))",
        note=(
            "Equivalent mutant. Every caller filters first: `filter_step`, "
            "`recover_belief` and `replace_from_belief` all pass a set already "
            "restricted to `weight > 0`, and the two reconstruction paths build "
            "their particles at `1/n`. So a negative weight never reaches the "
            "clamp. The Python clamps for the same belt-and-braces reason and "
            "the redundancy is ported, not introduced."
        ),
    ),
    Mutation(
        "resample uses np.sum where the Python uses the builtin",
        "belief.rs",
        "let mut total = 0.0f64;\n    for particle in particles {\n        total += particle.weight.max(0.0);\n    }",
        "let total = npsum(&particles.iter().map(|p| p.weight.max(0.0)).collect::<Vec<f64>>());",
    ),
    Mutation(
        "ESS threshold compared with a strict inequality",
        "belief.rs",
        "if ess(&belief.weights()) >= threshold {",
        "if ess(&belief.weights()) > threshold {",
        note=(
            "Unreachable, and provably so for the equal-weight case. "
            "`maybe_resample` only ever sees a set `filter_step` has already "
            "padded to `n_particles`, so its weights are `1/n` and its ESS is "
            "exactly `n` — against a threshold of `n/2`. Reaching equality "
            "needs `k >= n_particles` surviving particles whose ESS is exactly "
            "`n_particles/2`, which equal weights make impossible (ESS would be "
            "`k >= n_particles`) and unequal weights make an exact float "
            "coincidence. The boundary is copied from the Python either way."
        ),
    ),
    Mutation(
        "a refuted particle is downweighted, not killed",
        "belief.rs",
        "survivors.push(particle.with_weight(0.0));",
        "survivors.push(particle.with_weight(particle.weight * 1e-9));",
    ),
    Mutation(
        "a refuted particle advances anyway",
        "belief.rs",
        "if !observations_match(&emit_observation(&next_state, belief.seat), real_obs) {",
        "if false {",
    ),
    Mutation(
        "history keeps the newest frames from the wrong end",
        "belief.rs",
        "merged.drain(..merged.len() - lag);",
        "merged.truncate(lag);",
    ),
    Mutation(
        "the initial prior ignores the separation distance",
        "belief.rs",
        "if dist[i] < min_distance {",
        "if false {",
    ),
    Mutation(
        "the initial prior offers cells we can see",
        "belief.rs",
        "if !passable[i] || vis[i] {",
        "if !passable[i] {",
        note=(
            "Equivalent on any observation the engine can emit. A visible cell "
            "is typed plain, mountain, castle or general — never `TYPE_FOG` — "
            "and the very next test drops everything that is not `TYPE_FOG`. So "
            "the vision check is redundant with the type check unless the frame "
            "is hand-built, which the M2 crafted-memory pairs show is a real "
            "class of case but not one a first frame belongs to. Both checks are "
            "in the Python."
        ),
    ),
    Mutation(
        "the belief mean uses the unnormalized weights",
        "particle_summary.rs",
        "raw.iter().map(|value| value / total).collect()",
        "raw.clone()",
    ),
    Mutation(
        "belief planes accumulated in single precision",
        "particle_summary.rs",
        "army_mean[i] += weight * army;",
        "army_mean[i] += (weight as f32 * army as f32) as f64;",
    ),
    Mutation(
        "army variance left unclamped",
        "particle_summary.rs",
        "let var = (army_sq[i] - army_mean[i] * army_mean[i]).max(0.0);",
        "let var = army_sq[i] - army_mean[i] * army_mean[i];",
        note=(
            "Equivalent on every case the harness reaches, and kept for the "
            "same reason the Python keeps it. `E[x^2] - E[x]^2` is "
            "non-negative in exact arithmetic; the clamp only binds when "
            "cancellation drives it a few ulps below zero, which needs a "
            "plane where the particles nearly agree on a large army. Nothing "
            "in the corpus or the synthetic beliefs produces one, and "
            "`sqrt` of a negative would be a NaN in a network input — a "
            "belt-and-braces guard whose value is that it never fires."
        ),
    ),
    Mutation(
        "the enemy vision plane counts owned cells, not sight",
        "particle_summary.rs",
        "let vision = visibility_from_owned(owned, h, w);",
        "let vision = *owned;",
    ),
    Mutation(
        "the uniform proposal is not uniform",
        "proposal.rs",
        "let share = 1.0 / n as f64;",
        "let share = 1.0 / (n as f64 + 1.0);",
    ),
    Mutation(
        "the proposal key ignores the previous action",
        "proposal.rs",
        "sha256(&[&payload, &digest, &prev_idx.to_le_bytes()])",
        "sha256(&[&payload, &digest])",
    ),
    Mutation(
        "the masked softmax shifts by the unmasked maximum",
        "proposal.rs",
        "        .map(|(&l, &m)| if m { l } else { -1e9 })",
        "        .map(|(&l, &_m)| l)",
    ),
    Mutation(
        "the masked softmax normalizes over illegal mass too",
        "proposal.rs",
        "        .map(|(&c, &m)| if m { (c - peak).exp() } else { 0.0 })",
        "        .map(|(&c, &_m)| (c - peak).exp())",
        note=(
            "Equivalent because the clip underflows, the same way the network's "
            "`f32::MIN` fill does. An illegal logit is set to -1e9 before the "
            "shift, so `exp(-1e9 - peak)` is exactly zero and contributes "
            "nothing to the denominator. The explicit zero stays because it "
            "makes 'illegal means zero' a property of the code rather than of "
            "float underflow."
        ),
    ),
    Mutation(
        "the softmax denominator is a plain loop",
        "proposal.rs",
        "    let total = npsum(&exp);",
        "    let total: f64 = exp.iter().sum();",
    ),
    Mutation(
        "top_legal_actions drops the pass it leads with",
        "proposal.rs",
        "if mask[PASS_INDEX] {\n        out.push(PASS_ACTION);\n    }",
        "if false {\n        out.push(PASS_ACTION);\n    }",
    ),
    Mutation(
        "top_legal_actions returns k, not k plus pass",
        "proposal.rs",
        "if out.len() >= k + 1 {",
        "if out.len() >= k {",
    ),
    Mutation(
        "vision-changing compares the enemy's sight, not ours",
        "recovery.rs",
        "visibility_mask(&next_b, seat) != visibility_mask(&next_p, seat)",
        "visibility_mask(&next_b, 1 - seat) != visibility_mask(&next_p, 1 - seat)",
    ),
    Mutation(
        "rejuvenation accepts a history it never verified",
        "recovery.rs",
        "if !observations_match(\n                        &emit_observation(&next_state, seat),\n                        &frame.observation_after,\n                    ) {",
        "if false {",
    ),
    Mutation(
        "the replay beam keeps its worst branches",
        "recovery.rs",
        "next_beam.sort_by(|a, b| (-a.weight).partial_cmp(&(-b.weight)).unwrap());",
        "next_beam.sort_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap());",
    ),
    Mutation(
        "the acceptance weight has no floor",
        "recovery.rs",
        "let new_weight = entry.weight * p_b.max(1e-12);",
        "let new_weight = entry.weight * p_b;",
        note=(
            "Unreachable on the deployed path, and deliberately kept for the "
            "other one. Every candidate rejuvenation replays comes from the "
            "particle's own legal mask, and under the uniform proposal every "
            "legal action carries `1/n` — never zero. The floor binds only "
            "under the policy proposal, where a softmax can underflow an "
            "improbable-but-real action to zero and silently zero a history "
            "that explains every observation. `use_policy_proposal` is false "
            "today; the floor is what M7 would need if it were not."
        ),
    ),
    Mutation(
        "the replay budget is counted per turn, not per transition",
        "recovery.rs",
        "transitions_used += 1;",
        "transitions_used += 0;",
    ),
    Mutation(
        "the reconstruction spreads the army unevenly",
        "recovery.rs",
        "state.armies[at] = (base + if (i as i64) < rem { 1 } else { 0 }) as i32;\n            }\n            state.generals[general_at] = true;",
        "state.armies[at] = base as i32;\n            }\n            state.generals[general_at] = true;",
    ),
    Mutation(
        "the reconstruction skips its observation check",
        "recovery.rs",
        "if !observations_match(&emit_observation(&state, seat), obs) {\n            continue;\n        }",
        "",
        note=(
            "Reachable in principle, unreachable from the corpus, and named "
            "here rather than crafted. `_hidden_cells` only offers cells the "
            "seat cannot see, so enemy land placed there stays invisible — "
            "*unless* the same reconstruction also places our own hidden land "
            "next to it, which would give us vision of a cell the real frame "
            "showed as fog. That needs `need_own_land > 0`, meaning Morpheus "
            "holds land it cannot see, which happens in no frame of the M0 "
            "corpus. The check is the guard for the case; a crafted frame that "
            "forces it is the obvious next addition if the branch ever matters."
        ),
    ),
    Mutation(
        "a reconstruction is not marked collapsed",
        "recovery.rs",
        "collapsed: true,",
        "collapsed: false,",
    ),
    Mutation(
        "structure fog over proven ground reads as mountain",
        "recovery.rs",
        "if !memory.known_passable_base[i] && !memory.known_castle[i] {\n                mountains[i] = true;\n            }",
        "mountains[i] = true;",
    ),
    Mutation(
        "the reservoir admits at the arriving weight",
        "reservoir.rs",
        "let arriving = particle.with_weight(1.0);",
        "let arriving = particle.with_weight(particle.weight);",
    ),
    Mutation(
        "Algorithm R accepts everything",
        "reservoir.rs",
        "if rng.random_one() < (self.capacity as f64 / t) {",
        "if rng.random_one() < 1.0 {",
    ),
    Mutation(
        "replace_from_belief ignores the belief's own count",
        "reservoir.rs",
        "let n = self.capacity.min(belief.config.n_particles);",
        "let n = self.capacity;",
    ),
# --- M5: the search math, the tactical layer, the decision -------------
    Mutation(
        "self widening floor",
        "matrix.rs",
        "widening_limit(n, SELF_WIDENING_COEFF, SELF_WIDENING_CAP).max(SELF_WIDENING_FLOOR)",
        "widening_limit(n, SELF_WIDENING_COEFF, SELF_WIDENING_CAP)",
        "the floor keeps the root from locking its average strategy onto pass",
    ),
    Mutation(
        "widening cap",
        "matrix.rs",
        "(raw as usize).min(cap)",
        "raw as usize",
    ),
    Mutation(
        "exploration epsilon floor",
        "matrix.rs",
        "(EXPLORATION_NUMERATOR / (1.0 + n).sqrt()).max(EXPLORATION_FLOOR)",
        "EXPLORATION_NUMERATOR / (1.0 + n).sqrt()",
    ),
    Mutation(
        "regret matching clips negatives",
        "matrix.rs",
        "let positive = clamp_positive(regrets);\n    let total = npsum(&positive);\n    if total <= 0.0 {\n        return normalized_or_uniform(prior);",
        "let positive = regrets.to_vec();\n    let total = npsum(&positive);\n    if total <= 0.0 {\n        return normalized_or_uniform(prior);",
    ),
    Mutation(
        "regret matching plus clips at zero",
        "matrix.rs",
        "updated.max(0.0)",
        "updated",
    ),
    Mutation(
        "the mixed strategy blends the prior back in",
        "matrix.rs",
        "(1.0 - eps) * r + eps * q",
        "r",
    ),
    Mutation(
        "unvisited joint entries take first-play urgency",
        "matrix.rs",
        "if n > 0.0 { value } else { first_play }",
        "value",
    ),
    Mutation(
        "the dot product reduces pairwise, as numpy does",
        "matrix.rs",
        "let products: Vec<f64> = a.iter().zip(b).map(|(&x, &y)| x * y).collect();\n    npsum(&products)",
        "a.iter().zip(b).map(|(&x, &y)| x * y).sum()",
        "measured 1 ulp against Accelerate: the surface's tolerance may absorb it",
    ),
    Mutation(
        "the root selector drops actions the live prior zeroed",
        "matrix.rs",
        "if any && !all {",
        "if false && any && !all {",
    ),
    Mutation(
        "the root selector breaks ties by visits then prior",
        "matrix.rs",
        "let better = (avg[i], visits[i], prior[i]) > (avg[best], visits[best], prior[best]);",
        "let better = avg[i] > avg[best];",
    ),
    Mutation(
        "pass is illegal when anything else is playable",
        "tactics.rs",
        "if mask[..PASS_INDEX].iter().any(|&v| v) {\n        mask[PASS_INDEX] = false;\n    }",
        "if false {\n        mask[PASS_INDEX] = false;\n    }",
    ),
    Mutation(
        "pre-contact ban on stacking onto own structures",
        "tactics.rs",
        "if own_struct[g.at(tr, tc)] {\n            mask[index] = false;\n        }",
        "if false && own_struct[g.at(tr, tc)] {\n            mask[index] = false;\n        }",
    ),
    Mutation(
        "the garrison floor is threat-aware",
        "tactics.rs",
        "let floor = garrison_floor(own_total).max(max_threat_arrival(obs, memory) + 1);",
        "let floor = garrison_floor(own_total);",
        "the static floor guards fog rushes; the threat term guards a visible stack",
    ),
    Mutation(
        "the garrison floor exempts a winning capture",
        "tactics.rs",
        "if onto_gen && moved > defender {\n                continue;\n            }",
        "if false && onto_gen && moved > defender {\n                continue;\n            }",
    ),
    Mutation(
        "a half-split leaves ceil(a/2) on the general",
        "tactics.rs",
        "let remaining = if is_half { ga - ga / 2 } else { 1 };",
        "let remaining = if is_half { ga / 2 } else { 1 };",
    ),
    Mutation(
        "the castle anchor yields to a general threat",
        "tactics.rs",
        "if general_threat(obs, memory).is_some() {\n        // Defense of the general outranks castle savings: the anchored pile may\n        // be the reinforcement that saves the game.\n        return;\n    }",
        "if false {\n        return;\n    }",
    ),
    Mutation(
        "the castle anchor lets combat moves through",
        "tactics.rs",
        "let dest_enemy = g.inside(tr, tc) && g.owner(tr, tc) == OWNER_ENEMY;",
        "let dest_enemy = false;",
    ),
    Mutation(
        "the build site must cost exactly the base price",
        "tactics.rs",
        "&& cost[i] == BASE_COST",
        "&& cost[i] >= BASE_COST",
    ),
    Mutation(
        "the build site keeps its distance from visible enemies",
        "tactics.rs",
        "if d_enemy < CASTLE_SAFE_ENEMY_DIST as i64 {\n                continue;\n            }",
        "if false {\n                continue;\n            }",
    ),
    Mutation(
        "the build site is sticky on an existing pile",
        "tactics.rs",
        "let key = (-pile, d_gen, r, c);",
        "let key = (0, d_gen, r, c);",
    ),
    Mutation(
        "a threat that ties still counts",
        "tactics.rs",
        "arrival >= garrison + (d / 2) as i64",
        "arrival > garrison + (d / 2) as i64",
    ),
    Mutation(
        "reinforcement must arrive before the threat",
        "tactics.rs",
        "if !(1..=d - 1).contains(&rr) {",
        "if !(1..=d).contains(&rr) {",
    ),
    Mutation(
        "the kill march refuses to price a fogged cell",
        "tactics.rs",
        "if t == TYPE_FOG || t == TYPE_STRUCTURE_FOG {\n                    continue;\n                }",
        "if false {\n                    continue;\n                }",
    ),
    Mutation(
        "the kill march pays for the garrison's growth",
        "tactics.rs",
        "garrison + ((steps + 1) / 2) as i64",
        "garrison",
    ),
    Mutation(
        "the kill march prefers collecting own armies",
        "tactics.rs",
        "let key = if o == 1 { (0, -a_n) } else { (1, a_n) };",
        "let key = (1, a_n);",
    ),
    Mutation(
        "an enemy take needs a surplus to score as one",
        "tactics.rs",
        "if dest_owner == OWNER_ENEMY && surplus > 0.0 {",
        "if dest_owner == OWNER_ENEMY {",
    ),
    Mutation(
        "border chew is damped once the general is known",
        "tactics.rs",
        "enemy_score *= if is_gen_touch || progress > 0.0 {\n                    1.35\n                } else {\n                    GENERAL_CHEW_DAMP\n                };",
        "enemy_score *= 1.35;",
    ),
    Mutation(
        "the hunt bonus only rewards progress",
        "tactics.rs",
        "let hunt_factor = 1.0 + HUNT_PROGRESS_BONUS * hunt_prog.max(0.0);",
        "let hunt_factor = 1.0 + HUNT_PROGRESS_BONUS * hunt_prog;",
    ),
    Mutation(
        "deathtouch outranks the surplus gate",
        "tactics.rs",
        "if turn >= DEATHTOUCH_TURN && is_gen_touch {\n                score = DEATHTOUCH_SCORE;\n            }",
        "if false {\n                score = DEATHTOUCH_SCORE;\n            }",
    ),
    Mutation(
        "the savings catchment overrides the front assembly",
        "tactics.rs",
        "if src_b >= 0 && src_b <= CASTLE_CATCHMENT {",
        "if false {",
    ),
    Mutation(
        "commitment hysteresis only amplifies positive scores",
        "tactics.rs",
        "if out[index] > 0.0 {\n                        out[index] *= CONTINUATION_BONUS;\n                    }",
        "out[index] *= CONTINUATION_BONUS;",
        "equivalent by arithmetic: scores are clamped non-negative, so the only "
        "value the guard excludes is zero and 0 * 1.5 is zero. The guard is "
        "documentation of intent, and the Python writes it the same way.",
    ),
    Mutation(
        "the blend centres by the geometric mean before clipping",
        "tactics.rs",
        "(h[i].ln() - log_gmean).clamp(-log_clip, log_clip)",
        "h[i].ln().clamp(-log_clip, log_clip)",
        "without centring every action saturates the clip and the term is constant",
    ),
    Mutation(
        "the blend clip bounds the nudge",
        "tactics.rs",
        "let log_h = if scored[i] {\n            (h[i].ln() - log_gmean).clamp(-log_clip, log_clip)",
        "let log_h = if scored[i] {\n            (h[i].ln() - log_gmean)",
    ),
    Mutation(
        "the prior floor stops the blend resurrecting a network zero",
        "tactics.rs",
        "let floor = floor_abs.max(floor_frac * prior_max);",
        "let floor = 0.0;",
    ),
    Mutation(
        "unscored actions sit at the bottom of the clip range",
        "tactics.rs",
        "(f64::NEG_INFINITY).clamp(-log_clip, log_clip)",
        "f64::NEG_INFINITY",
    ),
    Mutation(
        "an underpowered general capture is not forced",
        "tactics.rs",
        "capture_moved_army(index, obs) > g.army(d_r, d_c)",
        "true",
    ),
    Mutation(
        "the biggest capture wins, first on a tie",
        "tactics.rs",
        "if army > best_army {\n                    best_army = army;\n                    best = index;\n                }",
        "if army >= best_army {\n                    best_army = army;\n                    best = index;\n                }",
    ),
    Mutation(
        "a tie in the kill/defence race goes to the kill",
        "tactics.rs",
        "if threat.map_or(true, |(_, d)| steps <= d) && mask[encode_action(first)] {",
        "if threat.map_or(true, |(_, d)| steps < d) && mask[encode_action(first)] {",
    ),
    Mutation(
        "forced defence needs an imminent threat",
        "tactics.rs",
        "if threat.1 <= DEFENSE_FORCE_WITHIN {",
        "if true {",
    ),
    Mutation(
        "the tithe yields to an enemy take",
        "tactics.rs",
        "if !chosen_is_take && turn % CASTLE_TITHE_PERIOD == 0 {",
        "if turn % CASTLE_TITHE_PERIOD == 0 {",
    ),
    Mutation(
        "the garrison release stands down during a threat",
        "tactics.rs",
        "if threat.is_none() && (GARRISON_FLOOR_FROM..DEATHTOUCH_TURN).contains(&turn) {",
        "if (GARRISON_FLOOR_FROM..DEATHTOUCH_TURN).contains(&turn) {",
    ),
    Mutation(
        "the garrison release needs the release factor",
        "tactics.rs",
        "if ga as f64 >= GARRISON_RELEASE_FACTOR * floor as f64 && !chosen_from_gen {",
        "if ga as f64 >= floor as f64 && !chosen_from_gen {",
    ),
    Mutation(
        "the release picks the highest-prior direction",
        "tactics.rs",
        "if prior[index] > prior[best] {\n                                best = index;\n                            }",
        "if false {\n                                best = index;\n                            }",
    ),
    Mutation(
        "a reverse onto enemy land is not oscillation",
        "tactics.rs",
        "if g.owner(tr, tc) == OWNER_ENEMY {\n        return false;\n    }",
        "if false {\n        return false;\n    }",
    ),
    Mutation(
        "a reverse that unlocks vision is not oscillation",
        "tactics.rs",
        "if revealed > 0 {\n        return false;\n    }",
        "if false {\n        return false;\n    }",
    ),
    Mutation(
        "the oscillation window is the last eight moves",
        "tactics.rs",
        "out[out.len() - OSCILLATION_HISTORY..].to_vec()",
        "out[..OSCILLATION_HISTORY].to_vec()",
    ),
    Mutation(
        "the redirect skips own-land retreats",
        "tactics.rs",
        "if dest_o == 1 && prog <= 0.0 {\n                continue;\n            }",
        "if false {\n                continue;\n            }",
    ),
    Mutation(
        "mandatory actions come before the policy order",
        "tactics.rs",
        "for &index in mandatory {\n        if seen[index] || !mask[index] {",
        "for &index in &[] as &[usize] {\n        if seen[index] || !mask[index] {",
    ),
    Mutation(
        "the candidate order is a stable sort on the prior",
        "tactics.rs",
        "        (-pa).partial_cmp(&(-pb)).unwrap_or(std::cmp::Ordering::Equal)",
        "        pa.partial_cmp(&pb).unwrap_or(std::cmp::Ordering::Equal)",
    ),
    Mutation(
        "frontier and attack lists rank by source army",
        "tactics.rs",
        "scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));",
        "scored.sort_by(|a, b| a.1.cmp(&b.1));",
    ),
    Mutation(
        "pass is mandatory only when it is alone",
        "tactics.rs",
        "if mask[PASS_INDEX] && !has_nonpass {",
        "if mask[PASS_INDEX] {",
    ),
    Mutation(
        "the reveal grid ignores cells we already own",
        "tactics.rs",
        "if owned[at] {\n                continue;\n            }",
        "if false {\n                continue;\n            }",
        "equivalent: owning a cell makes its whole 3x3 box visible, so the "
        "invisible count there is already zero and the early return only saves "
        "the work. The Python skips for the same reason.",
    ),
    Mutation(
        "path progress punishes leaving the goal component",
        "tactics.rs",
        "if after < 0 {\n        return -2.0;\n    }",
        "if after < 0 {\n        return 2.0;\n    }",
        "unreachable on a legal move: a destination is adjacent to a reachable "
        "source, so it is reachable too unless it is mountain or structure-fog "
        "- which the legal mask already refuses. Only a hand-built field with "
        "an isolated cell could tell.",
    ),
    Mutation(
        "the seek goals prefer enemy land over the posterior",
        "tactics.rs",
        "if !enemy.is_empty() {\n        return enemy;\n    }",
        "if false {\n        return enemy;\n    }",
        "enemy land is what keeps incursions near home scored as progress",
    ),
    Mutation(
        "the believed general is the weighted mode",
        "tactics.rs",
        "Some(current) => (entry.1, entry.2) > (current.1, current.2),",
        "Some(current) => entry.2 > current.2,",
    ),
    Mutation(
        "the assembly point is the frontmost own cell",
        "tactics.rs",
        "let front: Vec<usize> = (0..n)\n        .filter(|&i| own[i] && field.dist[i] == dmin)\n        .collect();",
        "let front: Vec<usize> = (0..n).filter(|&i| own[i]).collect();",
    ),
    Mutation(
        "the movable-stack exclusion drops the floored general",
        "tactics.rs",
        "    if !(GARRISON_FLOOR_FROM..DEATHTOUCH_TURN).contains(&turn) {\n        return None;\n    }\n    own_general_cell(obs, memory)",
        "    let _ = turn;\n    None",
    ),
    Mutation(
        "the highest-prior fallback respects the mask",
        "runtime.rs",
        "let score = |i: usize| if mask[i] { values[i] } else { -1.0 };",
        "let score = |i: usize| values[i];",
    ),
    Mutation(
        "the degradation path prefers a non-pass fallback",
        "runtime.rs",
        "if action == PASS_ACTION {\n        if let Some(fallback) = policy_fallback {",
        "if false {\n        if let Some(fallback) = policy_fallback {",
    ),
    Mutation(
        "the nearest-rank p99 rounds the rank up",
        "runtime.rs",
        "let rank = (0.99 * samples.len() as f64).ceil() as usize;",
        "let rank = (0.99 * samples.len() as f64).floor().max(1.0) as usize;",
    ),
    Mutation(
        "eviction protects the table just installed",
        "tree.rs",
        "if self.pending_pins.contains(&table.info_hash) || &table.info_hash == protect {",
        "if self.pending_pins.contains(&table.info_hash) {",
    ),
    Mutation(
        "eviction spares pinned tables",
        "tree.rs",
        "if self.pending_pins.contains(&table.info_hash) || &table.info_hash == protect {\n                continue;",
        "if &table.info_hash == protect {\n                continue;",
    ),
    Mutation(
        "retention weighs touch count as well as recency",
        "tree.rs",
        "self.last_used as f64 + LRU_TOUCH_WEIGHT * (self.touch_count as f64).ln_1p()",
        "self.last_used as f64",
    ),
    Mutation(
        "eviction takes the first minimum",
        "tree.rs",
        "if best.is_none() || score < best_score {",
        "if best.is_none() || score <= best_score {",
    ),
    Mutation(
        "widening an enemy column keeps the existing statistics",
        "tree.rs",
        "grown[row * new_b..row * new_b + old_b]\n                    .copy_from_slice(&plane[row * old_b..(row + 1) * old_b]);",
        "let _ = row;",
    ),
    Mutation(
        "widening self renormalizes the candidate prior",
        "tree.rs",
        "if total > 0.0 {\n            self.prior = positive.iter().map(|&v| v / total).collect();\n        }\n        self.regret.push(0.0);\n        self.avg_strategy.push(0.0);\n        let n_self = self.actions.len();",
        "if false {\n            self.prior = positive.iter().map(|&v| v / total).collect();\n        }\n        self.regret.push(0.0);\n        self.avg_strategy.push(0.0);\n        let n_self = self.actions.len();",
    ),
    Mutation(
        "refresh_self_priors rebuilds mass from the full network prior",
        "tree.rs",
        "let masses: Vec<f64> = self\n            .actions\n            .iter()\n            .map(|&a| prior.get(a).copied().unwrap_or(0.0).max(0.0))\n            .collect();",
        "let masses: Vec<f64> = self.prior.clone();",
        "without it, newly legal expands get prior 0 and the search locks onto pass",
    ),
    Mutation(
        "the enemy-hash cache is keyed on the reservoir version",
        "tree.rs",
        "if node.enemy_hash_version != version || node.enemy_hash_cache.len() != node.reservoir.n() {",
        "if node.enemy_hash_cache.is_empty() {",
    ),
    Mutation(
        "marginal visits skip tables of the wrong width",
        "tree.rs",
        "if table.n_self != n_self {\n                continue;\n            }",
        "if false {\n                continue;\n            }",
    ),
    Mutation(
        "backup pads every table to the current candidate width",
        "tree.rs",
        "table.ensure_self_rows(n_self);\n            let n_enemy = table.n_enemy();",
        "let n_enemy = table.n_enemy();",
    ),
    Mutation(
        "selection stops at an unexpanded node",
        "search.rs",
        "if self.tree.node(node).actions.is_empty() {",
        "if false {",
    ),
    Mutation(
        "freeze_widening actually freezes widening",
        "search.rs",
        "if !freeze_widening {",
        "if true {",
    ),
    Mutation(
        "the cross-turn prior cache installs without a network call",
        "search.rs",
        "if let Some(cached) = self.cached_enemy_prior(&h) {",
        "if let Some(cached) = None::<Vec<f64>> {",
        "only a slower path, unless it also changes the draw sequence",
    ),
    Mutation(
        "a child's memory folds forward from the root's, not the parent's",
        "search.rs",
        "update_memory(&memory, &my_obs)",
        "memory.clone()",
    ),
    Mutation(
        "an arriving particle is admitted to the child reservoir",
        "search.rs",
        "self.tree\n                        .node_mut(child)\n                        .reservoir\n                        .admit(&arriving, &mut rng);\n                    nodes.push(child);",
        "nodes.push(child);",
    ),
    Mutation(
        "a new child arrives at weight 1, not the parent's",
        "search.rs",
        "let arriving = Particle {\n                        state: Rc::clone(&next_state),\n                        weight: 1.0,",
        "let arriving = Particle {\n                        state: Rc::clone(&next_state),\n                        weight: particle.weight,",
    ),
    Mutation(
        "a terminal leaf takes the seat's own result",
        "search.rs",
        "let seat_win = if state.winner as usize == self.seat {\n                    1.0\n                } else {\n                    -1.0\n                };",
        "let seat_win = 1.0;",
    ),
    Mutation(
        "a drawn finish scores zero, not a loss",
        "search.rs",
        "let term = if next_state.winner < 0 {\n                    0.0\n                } else if next_state.winner as usize == self.seat {",
        "let term = if false {\n                    0.0\n                } else if next_state.winner as usize == self.seat {",
    ),
    Mutation(
        "backup walks the path from leaf to root",
        "search.rs",
        "for i in (0..path.edges.len()).rev() {",
        "for i in 0..path.edges.len() {",
        "the node statistics are per-edge, so the order may be unobservable",
    ),
    Mutation(
        "a leaf evaluation expands the node it stopped at",
        "search.rs",
        "if self.tree.node(leaf).actions.is_empty() {",
        "if false {",
    ),
    Mutation(
        "the root prior drives widening at the root",
        "search.rs",
        "let prior_for_self: Vec<f64> = if node == root && self.last_root_prior.is_some() {",
        "let prior_for_self: Vec<f64> = if false {",
    ),
)


def _cargo_env() -> dict[str, str]:
    env = os.environ.copy()
    cargo_bin = Path.home() / ".cargo" / "bin"
    if cargo_bin.is_dir():
        env["PATH"] = f"{cargo_bin}{os.pathsep}{env.get('PATH', '')}"
    return env


# Built once per mutation, so the link is the run's dominant cost. The
# `mutation` profile drops fat LTO and the single codegen unit; see the comment
# on it in Cargo.toml for why that is safe here and checked rather than assumed.
PROFILE = "mutation"
MUTATION_BINARY = BOT_DIR / "target" / PROFILE / "morpheus-rs"


def _build() -> bool:
    result = subprocess.run(
        [
            "cargo", "build", "--profile", PROFILE,
            "--manifest-path", str(BOT_DIR / "Cargo.toml"),
        ],
        env=_cargo_env(),
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _parity(args: list[str]) -> tuple[bool, str]:
    """`(all kinds agreed, output)` for one parity run."""
    env = os.environ.copy()
    env["MORPHEUS_RS_BINARY"] = str(MUTATION_BINARY)
    result = subprocess.run(
        [sys.executable, str(BOT_DIR / "tests" / "parity_cases.py"), *args],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0, result.stdout + result.stderr


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="use the whole corpus")
    parser.add_argument(
        "--only",
        default=None,
        help="substring filter on mutation names; for iterating on a new one",
    )
    parser.add_argument("--output", type=Path, default=None, help="write a JSON report")
    args = parser.parse_args(argv)
    parity_args = [] if args.full else ["--smoke"]

    selected = [m for m in MUTATIONS if not args.only or args.only in m.name]
    if not selected:
        print(f"no mutation matches {args.only!r}", file=sys.stderr)
        return 2
    originals = {name: (SRC / name).read_text() for name in {m.file for m in selected}}
    results = []
    try:
        if not _build():
            print("baseline build failed", file=sys.stderr)
            return 2
        clean, output = _parity(parity_args)
        if not clean:
            # Also the guard on the `mutation` profile: if dropping LTO ever
            # changed an answer, this is where it surfaces, before any mutation
            # has been applied and could be blamed.
            print(
                f"baseline parity is already failing on the {PROFILE} profile; "
                "fix that first:\n" + output
            )
            return 2
        print(
            f"baseline: parity clean ({'corpus' if args.full else 'smoke'}, "
            f"{PROFILE} profile)\n"
        )

        for mutation in selected:
            path = SRC / mutation.file
            source = originals[mutation.file]
            if mutation.before not in source:
                results.append({"name": mutation.name, "outcome": "stale"})
                print(f"  STALE     {mutation.name} (pattern no longer in {mutation.file})")
                continue
            # A mutation that lands in `#[cfg(test)]` breaks a test instead of
            # the bot, and then "survives" for a reason that says nothing about
            # the harness. That happened once — the pad-column mutation matched
            # a line in `group_norm`'s own unit test, sailed through, and was
            # about to be written up as a coverage hole. `replace(..., 1)` hits
            # the first occurrence, so the check is simply whether that
            # occurrence is below the test module.
            tests_at = source.find("#[cfg(test)]")
            if tests_at >= 0 and source.index(mutation.before) > tests_at:
                results.append({"name": mutation.name, "outcome": "in-test"})
                print(f"  IN-TEST   {mutation.name} (pattern only matches inside #[cfg(test)])")
                continue
            path.write_text(source.replace(mutation.before, mutation.after, 1))
            surfaces = mutation.surfaces()
            scoped = parity_args + (["--kinds", *surfaces] if surfaces else [])
            try:
                if not _build():
                    outcome = "uncompilable"
                else:
                    agreed, _ = _parity(scoped)
                    outcome = "survived" if agreed else "caught"
                    # A survivor is the expensive claim, so it is not allowed to
                    # rest on the scoping: re-run the *whole* set before
                    # recording one. A narrow map then costs a slow run, not a
                    # false coverage hole.
                    if outcome == "survived" and surfaces:
                        agreed, _ = _parity(parity_args)
                        outcome = "survived" if agreed else "caught"
            finally:
                path.write_text(source)
            results.append(
                {
                    "name": mutation.name,
                    "outcome": outcome,
                    "note": mutation.note,
                    "surfaces": list(surfaces) or "all",
                }
            )
            label = {"caught": "CAUGHT  ", "survived": "SURVIVED", "uncompilable": "NOBUILD "}[outcome]
            print(f"  {label}  {mutation.name}")
            if outcome == "survived" and mutation.note:
                print(f"              {mutation.note.splitlines()[0]}")
    finally:
        for name, source in originals.items():
            (SRC / name).write_text(source)
        _build()

    caught = sum(1 for r in results if r["outcome"] == "caught")
    survived = [r for r in results if r["outcome"] == "survived"]
    print(f"\n{caught}/{len(results)} caught, {len(survived)} survived")

    if args.output and not args.only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {"scope": "corpus" if args.full else "smoke", "results": results},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.output}")
    elif args.output:
        print("--only: report not written (it would record a partial run)")

    # Survivors with a recorded explanation are accepted; an unexplained one is
    # a hole in the harness and fails the run.
    unexplained = [r for r in survived if not r.get("note")]
    for r in unexplained:
        print(f"unexplained survivor: {r['name']}", file=sys.stderr)
    # A mutation that never reached production code proves nothing either way.
    misaimed = [r for r in results if r["outcome"] in ("stale", "in-test")]
    for r in misaimed:
        print(f"misaimed mutation: {r['name']} ({r['outcome']})", file=sys.stderr)
    return 1 if unexplained or misaimed else 0


if __name__ == "__main__":
    raise SystemExit(main())
