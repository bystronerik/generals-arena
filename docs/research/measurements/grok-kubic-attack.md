# Kubic attack behavior (fit set)

Fit wins n=341; holdout excluded n=37; losses skimmed n=11.

Reproduce: `python scripts/grok-analyze/analyze_kubic_attack.py`

## Top rules

- **ENGAGE_RATIO** [MEASURED, n=341]: First contact typically occurs near parity or slight army lead; median army_ratio=1.05, 97% of fit wins have army_ratio>=1.0; median stack_at_contact=15.
- **COMMIT_SENT_GE_HELD** [INFERRED, n=282]: On held>=10 cells, sent/held median=1.70, frac(sent>=held)=1.00; median surplus=14.5. First-capture sent/held median=13.00 (first held median=1, usually a 1-army frontier tile).
- **POST_SIGHT_STRIKE** [MEASURED, n=304]: After first_general_sight, drive largest stack toward enemy general (median toward_fraction=0.93; median sight→kill=24 ticks; median stack_at_sight=23).
- **FRONT_PUSH_DEFAULT** [MEASURED, n=341]: In the 30-tick window after contact, dominant class is 'push' (260/341). Trade=44, gather_then_push=34, passive=3.
- **GATHER_THEN_STRIKE** [MEASURED, n=11]: When a gather_wave is open at contact (n=11), first_capture waited for end in 11/11. Gather completed between contact and first_capture in 87/341 games. Gather open at sight: 0/341 (typically gather finishes before sight).
- **CASTLE_THEN_RAID** [MEASURED, n=85]: Own castle production precedes enemy-castle capture when both occur (median own castles already built=1). Enemy castle captures in 85/341 wins — not required to win. Median first enemy-castle-capture tick=203.

## Key distributions

- `army_ratio_contact`: n=341, median=1.052, p25=1.016, p75=1.205, mean=1.188
- `tile_ratio_contact`: n=341, median=1.163, p25=1.022, p75=1.533, mean=3.115
- `stack_at_contact`: n=341, median=15.000, p25=10.000, p75=20.000, mean=18.229
- `contact_to_capture`: n=341, median=2.000, p25=1.000, p75=21.000, mean=12.771
- `commit_sent_over_held`: n=9060, median=14.000, p25=7.000, p75=26.000, mean=18.649
- `commit_sent_over_held_held1`: n=5970, median=16.000, p25=8.000, p75=30.000, mean=21.364
- `commit_sent_over_held_2_9`: n=2808, median=12.333, p25=6.400, p75=19.500, mean=14.535
- `commit_sent_over_held_ge10`: n=282, median=1.703, p25=1.287, p75=2.381, mean=2.139
- `commit_surplus_held_ge10`: n=282, median=14.500, p25=7.000, p75=27.750, mean=21.926
- `first_capture_sent_over_held`: n=341, median=13.000, p25=6.000, p75=20.000, mean=14.335
- `sight_to_kill`: n=341, median=24.000, p25=2.000, p75=59.000, mean=39.698
- `stack_at_sight`: n=341, median=23.000, p25=15.000, p75=33.000, mean=28.446
- `toward_after_sight`: n=304, median=0.929, p25=0.824, p75=1.000, mean=0.898

## Front behavior (post-contact window)

- push: 260
- trade: 44
- gather_then_push: 34
- passive: 3

## Tagged claims

- [MEASURED] (n=341) At first_contact, Kubic's army/opponent army median is 1.052 (p25=1.016, p75=1.205).
- [MEASURED] (n=341) At first_contact, tile ratio median is 1.163; max_stack median is 15.
- [MEASURED] (n=341) Contact→first_capture latency median is 2 ticks (p10=1, p90=38).
- [MEASURED] (n=336) At first_capture, army ratio median is 1.059; stack median 15.
- [MEASURED] (n=9060) Capture commit sent/held (all sampled): median 14.000. Stratified — held==1 median 16.000 (n=5970); held 2–9 median 12.333 (n=2808); held>=10 median 1.703 (n=282); surplus(sent-held) at held>=10 median 14. First-capture sent/held median 13.000.
- [INFERRED] (n=282) Candidate commit rule on defended cells (held>=10): send at least held army (100.0% of samples) and typically a large surplus (median surplus=14.5). On held==1 frontier cells the ratio is huge because a full stack rolls light tiles — do not treat the all-sample median as a threshold.
- [MEASURED] (n=341) Front-window (30 ticks post-contact) classes: gather_then_push=34 (10.0%), passive=3 (0.9%), push=260 (76.2%), trade=44 (12.9%)
- [MEASURED] (n=341) Sight→kill latency median 24 ticks (p25=2, p75=59, p90=101); stack_at_sight median 23; toward_fraction after sight median 0.929.
- [INFERRED] (n=304) After first_general_sight, Kubic's largest stack moves toward the enemy general on a median 93% of directed moves (p10=73%) — strike is the default post-sight mode.
- [MEASURED] (n=341) Gather open at first_contact in 11/341 games; among those, first_capture waited for gather end in 11/11 (100.0%). Gather completed between contact and capture in 87/341. Gather open at sight in 0/341; among measurable strike waits 0/0.
- [INFERRED] (n=11) When a gather_wave is open at contact, first enemy capture usually waits for that wave to end (11/11).
- [MEASURED] (n=341) Enemy castle captures occur in 85/341 fit wins (24.9%). When they occur, own castles already built before that capture: median 1. First own castle tick median 10; first enemy-castle-capture tick median 203.
- [MEASURED] (n=341) Contact without our capture for >=40 ticks (or never): 31/341 (9.1%). Quick capture (<=10 ticks): 229/341. Median army_ratio at contact when delayed=1.0303030303030303; when quick=1.0615384615384615.
- [INFERRED] (n=31) Delayed capture after contact is not explained by being army-behind alone: delayed median army_ratio=1.030 vs quick=1.062. Likely causes include map chokepoints, unfinished gather, or castle-build priority (see UNKNOWN).
- [MEASURED] (n=0) Counterexample set: push/gather_then_push while army_ratio_contact < 0.8 — 0 games.

## Counterexamples

- Push while army_ratio_contact < 0.8: 0 (e.g. none)
- Sight→kill > 150 ticks: 10 sample rows in JSON
- Undercommit captures (sent/held < 1): 11 games
- Delayed capture after contact: 30 listed

## Loss skim — attack conditions that failed

- `never_killed`: 11
- `never_saw_general`: 9
- `army_behind_at_contact`: 6
- `contact_no_capture`: 2
- `small_stack_at_sight`: 1

- match 20583: contact=74, army_ratio=1.0161290322580645, sight=None, toward=None, fails=['never_saw_general', 'never_killed']
- match 20595: contact=31, army_ratio=1.0, sight=350, toward=1.0, fails=['never_killed']
- match 20603: contact=87, army_ratio=1.0, sight=None, toward=None, fails=['never_saw_general', 'never_killed']
- match 20923: contact=34, army_ratio=1.0, sight=73, toward=0.7297297297297297, fails=['never_killed', 'small_stack_at_sight']
- match 24184: contact=206, army_ratio=0.48214285714285715, sight=None, toward=None, fails=['never_saw_general', 'army_behind_at_contact', 'never_killed']
- match 24185: contact=209, army_ratio=0.37637362637362637, sight=None, toward=None, fails=['never_saw_general', 'army_behind_at_contact', 'never_killed']
- match 24186: contact=123, army_ratio=0.60431654676259, sight=None, toward=None, fails=['never_saw_general', 'contact_no_capture', 'army_behind_at_contact', 'never_killed']
- match 24187: contact=145, army_ratio=0.5878378378378378, sight=None, toward=None, fails=['never_saw_general', 'army_behind_at_contact', 'never_killed']
- match 24188: contact=232, army_ratio=0.40302267002518893, sight=None, toward=None, fails=['never_saw_general', 'army_behind_at_contact', 'never_killed']
- match 24189: contact=157, army_ratio=0.432, sight=None, toward=None, fails=['never_saw_general', 'army_behind_at_contact', 'never_killed']
- match 24988: contact=82, army_ratio=1.0, sight=None, toward=None, fails=['never_saw_general', 'contact_no_capture', 'never_killed']

## Cannot determine

- Intentional fog memory vs re-sight each tick (engine fades fog; bot may remember).
- Exact priority between castle-build spend and front push on the same tick.
- Whether half-moves are used specifically for probing vs economy (move inference ambiguous).
- True action when multi-stack combat occurs (largest-stack path only).
- Whether 'no attack despite contact' is a deliberate hold rule or a pathing/map artifact.
- Opponent-visible army estimation error under fog (we use true replay armies when adjacent).
- All-sample sent/held median is dominated by held==1 tiles; use held>=10 stratum for thresholds.

## Paths

- script: `scripts/grok-analyze/analyze_kubic_attack.py`
- json: `docs/research/measurements/grok-kubic-attack.json`
- md: `docs/research/measurements/grok-kubic-attack.md`
- corpus: `scripts/grok-analyze/kubic_corpus.py`
- moves: `scripts/grok-analyze/kubic_moves.py`
- split: `docs/research/measurements/grok-kubic-corpus-split.json`
