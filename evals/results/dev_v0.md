# Eval report: `dev_v0`

12 prompts × 4 configs = 45 playlists. Agent/scorer: `ollama:llama3.1:8b`, judge: `None`, catalog: `data/catalog_sample.csv (20k-track genre-stratified sample)`.

> **Partial run**: partial: stopped at 45/56 playlists (system memory pressure); 11 prompts complete for all configs. Code: `349f2a9 (before the eval-driven retrieval/era/arc fixes)`.

## Configs

- **baseline**: Keyword retrieval, heuristic ranking (no LLM after intent parsing)
- **llm_rank**: Keyword retrieval + batched LLM relevance scoring
- **agent**: Tool-calling retrieval agent + LLM relevance scoring
- **agent_seq**: The agent config's playlists re-ordered along the intent's energy arc

## Results (mean ± standard error)

| Metric | baseline | llm_rank | agent | agent_seq |
|---|---|---|---|---|
| Gold score ↑ | 0.43 ± 0.08 | 0.59 ± 0.08 | 0.58 ± 0.08 | 0.63 ± 0.07 |
| Genre hit ↑ | 0.24 ± 0.07 | 0.37 ± 0.17 | 0.43 ± 0.12 | 0.43 ± 0.12 |
| Era hit ↑ | 0.07 ± 0.07 | 0.27 ± 0.07 | 0.37 ± 0.17 | 0.37 ± 0.17 |
| Feature fit ↑ | 0.65 ± 0.07 | 0.68 ± 0.09 | 0.68 ± 0.10 | 0.68 ± 0.10 |
| Arc (Spearman) ↑ | -0.62 ± 0.19 | 0.12 ± 0.01 | -0.14 ± 0.29 | 0.37 ± 0.34 |
| Artist diversity ↑ | 0.88 ± 0.01 | 0.87 ± 0.02 | 0.85 ± 0.03 | 0.85 ± 0.03 |
| Duplicate songs ↓ | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Familiar artists | 0.63 ± 0.02 | 0.43 ± 0.07 | 0.18 ± 0.08 | 0.18 ± 0.08 |
| Latency (s) ↓ | 2.13 ± 0.43 | 40.88 ± 4.14 | 50.03 ± 5.06 | 50.03 ± 5.06 |

## Paired difference vs `baseline` (95% bootstrap CI, per-prompt win rate)

| Config | Gold score Δ | Judge precision Δ | Intent alignment Δ |
|---|---|---|---|
| llm_rank | +0.17 [+0.08, +0.27], wins 82% | – | – |
| agent | +0.16 [+0.07, +0.26], wins 73% | – | – |
| agent_seq | +0.21 [+0.10, +0.35], wins 82% | – | – |

## Gold score by category

| Category | n | baseline | llm_rank | agent | agent_seq |
|---|---|---|---|---|---|
| activity | 2 | 0.47 | 0.53 | 0.62 | 0.62 |
| arc | 2 | 0.19 | 0.56 | 0.43 | 0.69 |
| genre_era | 2 | 0.12 | 0.25 | 0.40 | 0.40 |
| mood | 2 | 0.66 | 0.82 | 0.66 | 0.66 |
| niche | 1 | 0.40 | 0.87 | 0.67 | 0.67 |
| nuanced | 2 | 0.67 | 0.70 | 0.89 | 0.89 |
| personalized | 1 | 0.59 | 0.59 | 0.59 | 0.59 |

## Lowest gold scores

- `genre-01` × `baseline` (0.00): *80s synth-pop and new wave classics* → The Less I Know The Better (Tame Impala); Around the World (Daft Punk); No. 1 Party Anthem (Arctic Monkeys); Can I Believe You (Fleet Foxes)…
- `arc-01` × `baseline` (0.09): *a workout that starts with a warmup, peaks hard in the middle, then cools down* → Around the World (Daft Punk); The Less I Know The Better (Tame Impala); No. 1 Party Anthem (Arctic Monkeys); The Chain - 2004 Remaster (Fleetwood Mac)…
- `genre-01` × `llm_rank` (0.17): *80s synth-pop and new wave classics* → Around the World (Daft Punk); This Charming Man - 2011 Remaster (The Smiths); You're My Heart, You're My Soul (Modern Talking); Cheri Cheri Lady (Modern Talking)…
- `act-02` × `baseline` (0.22): *deep focus music for studying, no distracting vocals* → Do I Wanna Know? (Arctic Monkeys); Giorgio by Moroder (Daft Punk); No. 1 Party Anthem (Arctic Monkeys); Can I Believe You (Fleet Foxes)…
- `genre-01` × `agent` (0.23): *80s synth-pop and new wave classics* → Around the World (Daft Punk); Save a Prayer - 2009 Remaster (Duran Duran); You're My Heart, You're My Soul (Modern Talking); Cemetry Gates - 2011 Remaster (The Smiths)…
