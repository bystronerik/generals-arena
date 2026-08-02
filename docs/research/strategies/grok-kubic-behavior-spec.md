# Kubic behavioral specification (observational)

> Superseded for implementation reading by the merge
> [`kubic-behavior-spec.md`](kubic-behavior-spec.md) (defense =
> present-but-rare). Keep this file as the grok evidence trail.

Implementation-grade reading of `competition-replays/Kubic/` under competition
rules (`RULES.md`). This is **not** a bot to ship. It is a reverse-engineered
decision procedure so a developer can reimplement move-for-move behavior without
watching replays.

Primary evidence: **341 fit wins**. Holdout: **37 wins**. Losses (**11**) and
draws (**1**) are skimmed for failure modes only and are excluded from
threshold fitting unless a claim says otherwise.

Reproduce every quantitative claim by re-running the named script under
`scripts/`. Aggregates live under `docs/research/measurements/grok-kubic-*.json`.

---

## Corpus and split

| Item | Value |
| --- | --- |
| Folder files | 384 win / 11 lose / 1 draw (each match = `.json` + `.meta.json`; a naive file count looks like 768/22/2) |
| Seat-resolved played | **378 win / 11 lose / 1 draw**, plus **6 forfeits** (`total_ticks <= 1`) |
| Split | Wins sorted by `match_id`; index `i` with `i % 10 == 9` → holdout; else fit |
| Fit / holdout | **341 / 37** wins |
| Outcome source | `Replay.outcome` (name → seat), **never** the scraper folder — see `docs/engine/leaderboard-replays.md` |

Scripts: `scripts/grok-analyze/kubic_corpus.py` → `docs/research/measurements/grok-kubic-corpus-split.json`.

**Do not** write these games into `data/games/`, `data/ratings/`, or
`data/remote_games/`.

---

## Evidence tags

Every claim below is tagged:

- **MEASURED** — distribution or rate with sample size from the fit set (or
  holdout, when stated).
- **INFERRED** — candidate rule; supporting and contradicting counts stated.
- **UNKNOWN** — not recoverable from frame-only replays.
- **REFUTED / WEAK (holdout)** — fit-derived rule that failed the holdout gate;
  kept with counterexamples (never silently dropped).

Holdout gates: `scripts/grok-analyze/verify_kubic_holdout.py` →
`docs/research/measurements/grok-kubic-holdout-verification.{json,md}`.
Confirmed = agreement ≥ 0.80; weak = [0.60, 0.80); refuted = < 0.60.

---

## Named constants (fit-derived)

Confidence: **high** = holdout confirmed and fit n≥300; **medium** = holdout
confirmed but smaller conditional n, or fit-only with clear band; **low** =
INFERRED / under-sampled / holdout weak.

| Constant | Value | Unit | Confidence | Source |
| --- | --- | --- | --- | --- |
| `PASS_TICKS_FORCE` | `{0, 1}` | tick | high | timing / opening; holdout H1 37/37 |
| `OPEN_FIRST_GAIN_TICK` | **3** (p=0.985) | tick | high | opening (rewritten) |
| `OPEN_PULSE_TICKS` | `{3, 6, 9}` then pause | tick | high | opening |
| `OPEN_FLOOD_START` | **27** (±1 in 94.4%) | tick | high | opening |
| `OPEN_TILES_T50` | 24 (band 20–25) | tiles | high | opening / expansion; H2 37/37 |
| `OPEN_ARMY_T50` | 50 (band ~41–51) | army | high | opening |
| `OPEN_TIP_DIST_HOME_EARLY` | median **1** during pulse; tip often on gen | tiles | medium | opening |
| `OPEN_FIRST_STEP` | open neighbour minimizing dist to **map center** (hit 0.909) | — | medium | opening INFERRED |
| `CASTLE_IN_FIRST_50` | **0** (all 341 fit wins) | builds | high | opening spend-detector |
| `CASTLE_FIRST_REAL_TICK` | median **138** (spend drop ≥30; EventLog tick-10 is FP) | tick | high | opening; see disagreement |
| `CASTLE_DIST_TO_GENERAL` | median **9** (real builds) | Manhattan | medium | opening n≈281 |
| `CASTLE_ARMY_BEFORE` | median **37**; drop median **35** | army | medium | opening |
| `CONTACT_TICK` | median **82** (p25=69, p75=91) | tick | high | expansion / timing |
| `CONTACT_ARMY_RATIO_MIN` | **1.0** (median 1.05) | us_army/them_army | high | attack; H4 37/37 |
| `CONTACT_STACK` | median **15** | army on tip | medium | attack / army |
| `CONTACT_TO_CAPTURE` | median **2** | ticks | medium | attack |
| `EXPAND_COMPONENTS` | **1** through expansion end | count | high | expansion; H3 37/37 |
| `POST_CONTACT_NEUTRAL_SHARE` | median **0.63** of gains in next 50 ticks | fraction | medium | expansion |
| `GATHER_WAVE_MIN` | ≥1 per win; median **4** waves | waves | high | army; H7 37/37 |
| `GATHER_STREAK_MIN` | 5 (detector) | consecutive growing tip moves | medium | events API |
| `SUSTAINED_MARCH_START` | median tip **15** (p25=9, p75=25) | army | medium | army |
| `HOME_BANK_PEAK` | median **27** | army on general | medium | army / defense |
| `HOME_BANK_AT_CONTACT` | median **8** | army | medium | army |
| `HOME_BANK_AT_SIGHT` | median **14** | army | medium | army |
| `MOVE_MODE_DEFAULT` | **full** (leave-1); half ≈1.5% of classifiable | — | medium† | army / timing; H9 36/37 |
| `PASS_RATE_MAX` | **< 0.05** (median 0.011) | fraction of ticks | high | timing; H8 37/37 |
| `REACT_LATENCY_P90` | **≤ 3** ticks after first visible enemy | ticks | **weak holdout** | timing; H10 29/37 |
| `SIGHT_TICK` | median **180** (~0.89 of game length) | tick | high | timing |
| `TIP_AT_SIGHT` | median **23** (use ≥10 as floor) | army | high | army / attack; H11 37/37 |
| `SIGHT_TO_KILL` | median **24** (p90≈101; gate ≤120) | ticks | high | attack / timing; H12 36/37 |
| `TOWARD_AFTER_SIGHT` | median **0.93** (gate ≥0.70) | directed-move fraction | high | attack; H6 31/37 |
| `COMMIT_HELD_GE10` | send ≥ held; median surplus **+14.5** | army | medium | attack n=282 |
| `RECALL_PROX_D` | under-sampled; when enemy_home_dist≤3, away-stack recall ≈83% in wins | Manhattan | low | defense |
| `RECALL_LATENCY` | median **5** | ticks | low | defense |
| `ENEMY_HOME_DIST_WIN` | median min dist **7**; ≤1 in only 3.8% of fit wins | Manhattan | high | defense; H13 36/37 |

† Move inference marks ~30% of ticks `multi`/`unknown`. Full/half rates are
conditional on classifiable sends.

---

## State the bot appears to track

| State | Update rule | Tag |
| --- | --- | --- |
| Own general cell | Latch from spawn | MEASURED (always known) |
| Own territory / frontier | Every owned cell | MEASURED |
| Built castles | Persist after real spend (drop ≥~35); **do not** trust early `castle_built` EventLog stamps | MEASURED via spend-detector; EventLog tick~10 is FP |
| Tip stack identity | Largest owned army cell (tie-stable) | MEASURED (path tool) |
| Enemy land in vision | 3×3 Chebyshev vision each tick | MEASURED (fog API) |
| Enemy general cell | Latch forever after first vision | **INFERRED** — 100% of fit/holdout wins reach sight then drive tip toward that cell (toward_frac≈0.93); engine fog fades but strike continues |
| Gather / strike phase | Pre-sight gather waves; post-sight toward-general marches | MEASURED |
| Home threat distance | Enemy owned tile Manhattan to own general | **INFERRED** for defense redirects; exact threshold **UNKNOWN** |

**UNKNOWN:** section priors, mountain-pocket tables, explicit MCTS trees, and
any hidden belief distribution over unseen general cells (only the post-sight
latch is strongly evidenced).

---

## Decision procedure (priority order)

Resolve **one** action per tick. Higher priority wins. Tie-breaks are stated
inside each step.

```
function kubic_act(state, vision, memory):
    # --- hard terminals ---
    if can_capture_enemy_general_this_tick(state):          # deathtouch from 800
        return full_move(tip_or_adjacent_killer → enemy_general)

    if tick in {0, 1}:
        return PASS                                         # MEASURED 100%

    # --- defense ---
    d_home = min Manhattan(enemy_owned_visible_or_known, own_general)
    if d_home is not None and d_home <= RECALL_PROX_D:       # D under-sampled; see open Q
        if tip_dist(own_general) > 2:
            return full_move(tip toward own_general)        # INFERRED; latency~5
        return reinforce_threatened_cells()

    # --- strike (enemy general known) ---
    if memory.enemy_general_known:
        if tip_army < TIP_AT_SIGHT_FLOOR:                   # operating ~23; floor 10
            return gather_into_tip(full_moves=True)         # MEASURED gather waves
        return full_move(tip toward memory.enemy_general)   # toward_frac~0.93
        # no new castle projects after sight (INFERRED from timing)

    # --- contact / contest (enemy land seen, general unknown) ---
    if vision.has_enemy_tile:
        # react within a few ticks (fit p90=3; holdout WEAK — see H10)
        if can_commit_capture(src, dst):
            # held>=10: require sent >= held (usually surplus ~+15)
            # held==1: roll with tip (ratio meaningless)
            return full_move(src → dst)
        if gather_wave_open:
            return continue_gather_wave()                   # MEASURED 11/11 wait
        # default front class = push (76%), else trade / gather_then_push
        if army_ratio >= CONTACT_ARMY_RATIO_MIN:            # ~1.0
            return push_or_expand_mix()
            # MEASURED: next 50 ticks still ~63% neutral gains — do not hard-pivot
        return expand_neutrals()                            # stay busy; never idle

    # --- mid-game castle (NOT in first 50) ---
    if tick >= ~116 and castles_built < 1 and can_fund_castle(plain):
        # real first build median tick 138; dist_to_general median 9; spend ~35
        # NEVER trust EventLog castle_built @ tick 10 — false positive (190 on fit)
        return BUILD_CASTLE(plain_near_price)

    # --- opening / expansion (blind) ---
    return open_then_expand():
        # ticks 0–1: already passed above
        # pulse captures at ticks 3, 6, 9 from general (full leave-1)
        #   first step: open neighbour minimizing Manhattan to map CENTER (INFERRED)
        # pause net tile growth until flood_start ≈ 27
        # from ~27: nearly every-tick neutral flood until contact
        # keep components == 1; land @50 ≈ 24 tiles / army ≈ 50
        # tip stays near general in the pulse phase
        # use FULL (leave-1) by default; half rare
```

### Action density

**MEASURED:** after ticks 0–1, act nearly every tick. Mid-game buckets 51–200
have pass rate ≈ 0. Pass before bulk-growth ticks (`t % 50 == 49`) is **not**
elevated — do not idle for +1 land growth.

---

## Phase profiles

### Opening (ticks 0–50)

| Claim | Tag | Detail |
| --- | --- | --- |
| Pass ticks 0–1 | MEASURED | Holdout 37/37 |
| First gain at tick 3 (gen army ≥2) | MEASURED | p=0.985; cex 5 games |
| Pulse 3/6/9 then pause | MEASURED | p_gain 0.985/0.935/0.739; near-zero at tick 20 |
| Flood start ≈ tick 27 | MEASURED | 94.4% within ±1; army≈14, stack≈10, tiles≈4 |
| Land @50 ≈ 24 tiles, army ≈ 50 | MEASURED | Holdout band 37/37 |
| First step toward map center | INFERRED | center-neighbour hit 0.909 > egen 0.859 (fog-legal) |
| Land stays in own half | MEASURED | enemy-half frac@50 median 0; enemy captures 10/7884 |
| **No castle in first 50** | MEASURED | 341/341; EventLog `castle_built`@~10 is **false positive** (190 cases, spend drop &lt;30) |
| First **real** castle | MEASURED | median stamp **138**, dist median **9**, army_before median **37**, drop median **35** |

Script: `scripts/grok-analyze/analyze_kubic_opening.py` (spend-detector version from opening analyst).

**REFUTED (detector artifact):** “first castle production at tick 10 / adjacent to general.” That claim came from EventLog `castle_built` without a spend check. Timing/attack aggregates that used the raw event are contaminated for early ticks. Keep the claim only as a warning.

### Expansion until contact

| Claim | Tag | Detail |
| --- | --- | --- |
| Stop pure expansion at first orthogonal contact | MEASURED | phase boundary; contact median tick 82 |
| Single contiguous component | MEASURED | holdout 37/37 |
| Blind path (no visible enemy target) | MEASURED | blind_share median 1.0 |
| Multi-option capture: fill + toward-egen | MEASURED | best-option rates ~0.77 / ~0.77 |
| Stray from territory | MEASURED | orthogonal captures only (Manhattan 1) |

Script: `scripts/grok-analyze/analyze_kubic_expansion.py`.

### Army routing

| Claim | Tag | Detail |
| --- | --- | --- |
| Tip carries strike mass; general is modest bank | MEASURED | mean tip frac ~0.25 vs gen ~0.13 |
| ≥1 gather wave every win | MEASURED | median 4; ~72% pre-sight |
| Sustained marches start ~15 army | MEASURED | |
| Full ≫ half | INFERRED | 98.5% of classifiable; 32% ticks ambiguous |

Script: `scripts/grok-analyze/analyze_kubic_army.py`.

### Attack / contest / strike

| Claim | Tag | Detail |
| --- | --- | --- |
| Contact near army parity (≥1.0) | MEASURED | holdout 37/37 |
| Front window: push 76% / trade 13% / gather_then_push 10% | MEASURED | |
| Commit on held≥10: sent≥held | INFERRED | 282/282 samples; surplus median +14.5 |
| After sight: march toward general | MEASURED | toward median 0.93; holdout 31/37 ≥0.70 |
| Sight→kill median 24 ticks | MEASURED | tip@sight median 23 |
| Enemy castle capture optional | MEASURED | 25% of wins; usually after own castle |

Script: `scripts/grok-analyze/analyze_kubic_attack.py`.

### Defense

| Claim | Tag | Detail |
| --- | --- | --- |
| Wins keep enemy far from home | MEASURED | min dist median 7; ≤1 in 3.8% wins vs 100% losses |
| Incursion → redirect tip home | MEASURED | 92% of away-stack samples; latency median 5 |
| Exact recall radius D | UNKNOWN | under-sampled in wins |
| Loss mode | MEASURED | stack away (median dist 12) while gen still has army; often never sight; bist cluster also has pass_rate≥0.30 (tempo collapse / possible disconnect) |

Script: `scripts/grok-analyze/analyze_kubic_defense.py`.

### Timing

Script: `scripts/grok-analyze/analyze_kubic_timing.py` (pass rates, buckets, reaction, milestones).

---

## Holdout verification summary

| Rule | Rate | Status |
| --- | ---: | --- |
| H1 pass ticks 0–1 | 1.000 | confirmed |
| H2 tiles@50 in [20,25] | 1.000 | confirmed |
| H3 contiguous expansion | 1.000 | confirmed |
| H4 army_ratio@contact ≥1 | 1.000 | confirmed |
| H5 has general sight | 1.000 | confirmed |
| H6 toward_after_sight ≥0.7 | 0.838 | confirmed |
| H7 ≥1 gather wave | 1.000 | confirmed |
| H8 pass_rate <0.05 | 1.000 | confirmed |
| H9 full/(full+half) ≥0.90 | 0.973 | confirmed |
| H10 react ≤3 ticks | **0.784** | **WEAK** — keep rule, mark soft; fails listed in holdout md |
| H11 tip@sight ≥10 | 1.000 | confirmed |
| H12 sight→kill ≤120 | 0.973 | confirmed |
| H13 enemy never adj home | 0.973 | confirmed |
| H14 legacy EventLog castle==10 | 0.955 | **refuted_policy_artifact** (FP detector, not policy) |
| H14b no real castle spend by t50 | 1.000 | confirmed |
| H15 pulse tile-gain on tick 3 | 1.000 | confirmed |
| H16 flood start within 27±1 | **0.622** | **WEAK** — fit used a stricter flood detector; holdout first-gain-after-20 is softer |

**WEAK rule H10** stays soft (≤3 preferred, ≤10 acceptable).

**WEAK rule H16:** keep flood≈27 as the fit operating point, but do not hard-assert ±1 on holdout with the coarse verifier. Prefer the opening script’s flood detector for implementation.

**H14:** EventLog tick-10 castle is a **false positive**; policy is “no castle in first 50,” confirmed by H14b.

---

## Analyst disagreements (open — do not average)

1. **Castle timing (RESOLVED against EventLog).** Timing/attack used
   `castle_built` stamps → median tick 10. Opening spend-detector shows **0**
   real builds in ticks 1–50 and first real build median **138**. Independent
   re-check: 190 early stamps with drop&lt;30, **0** early stamps with drop≥30.
   **Spec follows the spend-detector.** Raw EventLog early stamps are FP.
2. **Post-contact priority.** Expansion: continue neutrals (median 63% of gains).
   Attack: front class is push (76%). Compatible if push and neutral grab interleave;
   incompatible if read as exclusive modes. **Open:** exact arbiter between
   “take adjacent enemy” and “take adjacent neutral” on the same tick.
3. **Defense radius vs attack push.** Defense wants tip home when enemy near;
   attack wants tip on the front. Wins rarely create the conflict (enemy stays
   far). **Open:** priority when both fire.
4. **Half-move role.** Army/timing: half is rare (~1.5%). Opening may differ in
   the pulse phase. **Open:** deliberate early probes vs inference noise.
5. **First-step target.** Opening: map-center neighbour (0.909). Expansion
   multi-option: toward-egen (0.767) + fill (0.775). **Open:** whether center
   bias is only the first step or a lasting prior.
6. **Castle skip.** Some wins never show a real spend. **Open:** skip condition.

---

## Loss / draw skim (excluded from thresholds)

| Pattern | Evidence |
| --- | --- |
| Never sight enemy general | 9/11 losses |
| Enemy reaches general (dist 0/1) | 11/11 losses |
| Zero gather waves | bist 24184–24189, 24988 |
| Tempo collapse (pass_rate ≥0.30) | bist 24184–24189 — **treat as non-policy** (disconnect / stall), not intentional idling |
| Early contact with low land | e.g. 20595, 20923 |
| Draw | 22221 only — not used for rules |

---

## Open questions / cannot determine from replays

1. Exact expansion score weights (fill vs toward-egen vs fog value).
2. Fog-memory implementation details (what else is latched besides enemy general).
3. Belief / hunt target over unseen general cells before first sight.
4. Exact integer `RECALL_PROX_D` and whether recall is vision-gated.
5. True actions on `multi`-inferred ticks (simultaneous sources vs combat).
6. Exact castle cell scorer (dist≈9, price≈35) and skip condition.
7. Pulse-phase pause intent (ticks 10–26): pathfinding, stack growth, or scripted wait.
8. Whether “no capture for ≥40 ticks after contact” (9% of wins) is a hold rule
   or a map artifact.
9. Search algorithm (MCTS vs greedy BFS) — structure invisible in frames.
10. Time-budget / fault behavior under the 150 ms limit.
11. Why EventLog marks adjacent cells as castles at tick ~10 (production-shaped
    noise vs real structure) — tooling bug, not bot policy.

---

## File index

| Artifact | Path |
| --- | --- |
| This spec | `docs/research/strategies/grok-kubic-behavior-spec.md` |
| Corpus split | `docs/research/measurements/grok-kubic-corpus-split.json` |
| Opening | `scripts/grok-analyze/analyze_kubic_opening.py` → `grok-kubic-opening.{json,md}` |
| Expansion | `scripts/grok-analyze/analyze_kubic_expansion.py` → `grok-kubic-expansion.{json,md}` |
| Army | `scripts/grok-analyze/analyze_kubic_army.py` → `grok-kubic-army.{json,md}` |
| Attack | `scripts/grok-analyze/analyze_kubic_attack.py` → `grok-kubic-attack.{json,md}` |
| Defense | `scripts/grok-analyze/analyze_kubic_defense.py` → `grok-kubic-defense.{json,md}` |
| Timing | `scripts/grok-analyze/analyze_kubic_timing.py` → `grok-kubic-timing.{json,md}` |
| Holdout | `scripts/grok-analyze/verify_kubic_holdout.py` → `grok-kubic-holdout-verification.{json,md}` |
| Shared helpers | `scripts/grok-analyze/kubic_corpus.py`, `scripts/grok-analyze/kubic_moves.py` |
| Prior batch sketch | `scripts/grok-analyze/analyze_kubic.py` → `kubic_full_analysis.json` |
