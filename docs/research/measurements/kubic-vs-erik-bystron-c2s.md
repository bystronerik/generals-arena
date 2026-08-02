# Kubic vs erik.bystron — contact→sight (macaria-era proxy)

Date: 2026-08-02  
Filter: competition replays where `players` contains both `Kubic` and
`erik.bystron`. At that time this account ran **macaria**.  
Corpus: union of `competition-replays/Kubic/` and
`competition-replays/erik.bystron/` (12 unique match ids).  
Method: `arena.instrument.replay.analyze` with `queried_player=Kubic`.
Seat-resolved winner, not folder label.

Machine-readable: [`kubic-vs-erik-bystron-c2s.json`](kubic-vs-erik-bystron-c2s.json)

## Outcome

| Side | W–L |
| --- | --- |
| Kubic | **11–1** |

## Timing (Kubic seat)

| Metric | This subset (wins) | All Kubic fit wins (docs) | sosipolis vs macaria c2s | macaria vs sosipolis c2s |
| --- | ---: | ---: | ---: | ---: |
| Contact median | **51** | 82 | 83 | 83 |
| Sight rate | **11/11** | ~100% | 3/10 (30%) | 8/10 (80%) |
| Contact→sight median | **110** | 93 | 115 (n=3) | 192.5 (n=8) |
| Sight→kill median | **50** | 20–24 | 77 (n=2) | 1 (n=8) |

n=12 is small. Treat as a matched-opponent skim, not a full corpus claim.

## Reading

1. Against erik.bystron/macaria, Kubic still **always sights** on wins and
   still **wins 11–1**.
2. Contact→sight vs this opponent (**110**) is close to sosipolis when sosipolis
   sights (**115**). Sosipolis’s main gap remains **sight rate** (30% vs 100%).
3. Sight→kill vs this opponent (**50**) is slower than Kubic’s all-opponent
   median (~20–24), so the all-opponent Kubic finish number is optimistic for
   macaria matchups.
4. Arena macaria vs sosipolis shows a different finish shape: long C→S, then
   near-instant S→K (median 1).

## Per-game

| id | out | ticks | C | S | C→S | S→K |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 24988 | lose | 88 | 82 | — | — | — |
| 24989 | win | 250 | 51 | 135 | 84 | 115 |
| 24990 | win | 195 | 78 | 194 | 116 | 1 |
| 24991 | win | 134 | 32 | 78 | 46 | 56 |
| 24992 | win | 246 | 79 | 185 | 106 | 61 |
| 24993 | win | 395 | 88 | 394 | 306 | 1 |
| 39664 | win | 191 | 36 | 139 | 103 | 52 |
| 39665 | win | 229 | 81 | 227 | 146 | 2 |
| 39666 | win | 232 | 39 | 149 | 110 | 83 |
| 39667 | win | 182 | 73 | 132 | 59 | 50 |
| 39668 | win | 291 | 36 | 256 | 220 | 35 |
| 39669 | win | 217 | 36 | 175 | 139 | 42 |
