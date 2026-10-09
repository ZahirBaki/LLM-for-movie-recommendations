# Experimental ranking mode

Movie Agent now accepts `baseline` (default) or `genre_votes` per chat request. The selector is development-only and does not reset messages, session ID, watched/excluded identities, seeds, pending clarification, preferences, country, cards, ordinal references, or expiry. The LLM does not choose the mode or reorder results. Request-local ContextVars propagate through the assistant, MovieTools, card service, and AnyIO worker threads; tokens reset in `finally` to isolate concurrent requests. Every request explicitly selects a mode; omitting it restores baseline.

## Enable and switch

In the backend PowerShell terminal, from the project root:

```powershell
$env:ENVIRONMENT = 'development'
$env:MOVIE_EXPERIMENTAL_RANKING = 'true'
.venv\Scripts\python.exe -m uvicorn backend.api.app:app --host 127.0.0.1 --port 8000
```

In a separate frontend terminal:

```powershell
cd frontend
$env:NEXT_PUBLIC_MOVIE_EXPERIMENTAL_RANKING = 'true'
npm.cmd run dev
```

Open http://localhost:3000 and use **Active ranking** in Movie Agent: **Original** or **Experimental (Genre + Votes)**. Original starts selected. Switching preserves the same conversation and applies to the next request, including a pending title answer or follow-up. The selector is disabled during a request.

The backend requires an explicit `development` value in ENVIRONMENT, APP_ENV, or NODE_ENV plus the true feature flag. Any `production` value in those variables overrides enablement. The frontend requires Next development mode plus its public flag. Production builds hide the selector regardless of its public flag. Restart Next after changing environment variables.

To disable: set both feature flags to `false` and restart the corresponding servers. Backend-disabled experimental requests use baseline, including forged requests. No algorithm is deployed or made the production default.

## Scope and exact scoring

The active path is `/movie-chat` ? `respond_to_movie_chat` / `handle_session_turn` ? MovieTools / ChatRecommender ? `run_intent_recommendation` ? `recommend_cards` ? RecommendationEngine. Historical prototypes are untouched.

Local content-only seeds, verified external TMDB seeds, and mixed seeds use the experimental reranker. Existing routing selects collaborative primary when at least two local favorites are SVD-usable and no external queries are present; these requests retain baseline, including content/cold-item fillers, with `collaborative_primary` as the fallback reason. External movies receive no fabricated collaborative score.

The reranker considers only the baseline content top 200 eligible candidates after existing hard filters and session exclusions. It shares the existing evaluation scoring function:

`0.65 * baseline rank percentile + 0.20 * seed-union genre Jaccard + 0.10 * votes/(votes+500) + 0.05 * normalized log1p(popularity)`

Popularity uses the pool's 95th percentile denominator and clipping to [0,1]. Rank percentile is `1 - index/max(pool_size-1,1)`. Final ties sort by ascending TMDB ID, matching evaluation. Missing genres/votes/popularity contribute zero. Empty pools return no results and record `empty_candidate_pool` without claiming reranking was applied. Production content weights and SVD scoring remain unchanged.

## Diagnostics

When enabled in development, `logs/ranking.jsonl` records engine events: request/session identifier, requested/applied modes, lane, application boolean, fallback, seed IDs, candidate count, ranked IDs, and available score components. Session delivery events record the actual final card IDs after exclusions/count truncation. Disabled experimental requests also record their fallback in development. The file is Git-ignored; delete it to clear local diagnostics after stopping the backend. No prompts, titles, credentials, API keys, IP addresses, or external personal identifiers are recorded. The `movie.ranking` Python logger also emits these events for configured local logging. Production does not write the local diagnostic file.

## Compare identical inputs offline

```powershell
.venv\Scripts\python.exe -m evaluation.scripts.compare_ranking --favorites 157336
.venv\Scripts\python.exe -m evaluation.scripts.compare_ranking --favorites 157336 1248832 --exclude 11802
.venv\Scripts\python.exe -m evaluation.scripts.compare_ranking --favorites 157336 --filters-json '{"min_year":1990,"max_year":1999}'
```

The command uses the actual engine for both modes, the same seed IDs, constraints, and exclusions, without provider/OpenAI calls. External metadata uses the existing verified snapshot and requires the compatible cached embedding model. Set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 to prohibit downloads. It prints top-ten title/year/ID/rank/scores, applied-mode diagnostics, shared/added/removed IDs, and position changes. Collaborative inputs correctly show identical baseline outputs. For a chat comparison, use fresh sessions with identical inputs because an existing session deliberately excludes already shown movies.

Measured local example for Interstellar (157336):

| Movie | Year | TMDB ID | Original rank | Experimental rank | Content score | Experiment score |
|---|---:|---:|---:|---:|---:|---:|
| The Great Flood | 2025 | 982843 | 4 | 1 | 0.521409 | 0.941552 |
| Spaceman | 2024 | 636706 | 9 | 2 | 0.501464 | 0.929512 |
| Star Trek Beyond | 2016 | 188927 | 7 | 3 | 0.501873 | 0.873647 |
| Crumbs | 2015 | 322614 | 2 | 4 | 0.524591 | 0.870119 |
| Marooned | 1969 | 40140 | 8 | 5 | 0.501637 | 0.868978 |

Original #1: Space Chimps (2008), ID 11802, content 0.557372. Five shared titles; added IDs 871, 2160, 25392, 401513, 916401; removed IDs 11802, 16911, 70981, 454439, 877269. Full output: `evaluation/results/experimental-comparison-example.txt`.

## Files

Created: `backend/recommendation/experimental_ranking.py` (shared scorer, gating, contexts, diagnostics), `evaluation/scripts/compare_ranking.py` (offline live-engine comparison), `tests/backend/test_experimental_ranking.py`, `tests/browser/check_experimental_ranking_browser.py`, this document, and local comparison/test output artifacts.

Modified: `evaluation/scripts/recommendation_quality.py` (calls the shared scorer without changing saved evaluation data), `backend/recommendation/recommendation_engine.py` (content-only opt-in), `backend/api/app.py` (validated mode and request context), `backend/sessions/movie_session.py` (experimental exclusions and delivery diagnostics), `frontend/app/components/MovieChat.tsx` (selector/request field), `frontend/app/globals.css` (compact selector styling), `.gitignore` (local diagnostics).

## Tests and limitations

```powershell
.venv\Scripts\python.exe -m unittest discover
cd frontend
npx.cmd tsc --noEmit
```

Browser tests require Chrome, Playwright, the flagged Next dev server, and the existing offline browser fixture backend:

```powershell
.venv\Scripts\python.exe -m tests.browser.check_movie_chat_browser --serve
# In another terminal:
.venv\Scripts\python.exe -m tests.browser.check_movie_chat_browser
.venv\Scripts\python.exe -m tests.browser.check_movie_chat_presentation_browser
.venv\Scripts\python.exe -m tests.browser.check_experimental_ranking_browser
```

Regression coverage includes unchanged/disabled baseline, backend production enforcement, local/external/mixed parity against evaluation, collaborative fallback, session mode switching, watched/excluded identities, pending clarification, deterministic ranking, top-200 bound, count and card alignment, metadata gaps and empty pools. Existing external/session tests are also exercised with experimental mode enabled. Tests mock OpenAI, TMDB, and encoding. No paid OpenAI calls, catalog edits, embedding regeneration, SVD training, or human-rating edits occurred.

Final validation (2026-10-09):

- Full backend suite: **361 passed**, zero failures/errors (`logs/archive/experimental-backend-tests.log`). This includes inherited external regression tests rerun with experimental mode enabled.
- TypeScript: `npx.cmd tsc --noEmit` passed.
- Existing `tests/browser/check_movie_chat_browser.py`: all checks passed with real engine ranking and mocked external services.
- Existing `tests/browser/check_movie_chat_presentation_browser.py`: all checks passed, including desktop, small mobile, and landscape dimensions.
- New `tests/browser/check_experimental_ranking_browser.py`: desktop/mobile selector default, mode submission, history preservation, and stable session ID passed.
- Real-catalog local comparison and mixed Interstellar/Digger comparison with cached offline encoding and exclusion 11802 completed. Mixed run returned ten cards in each mode with no collaborative scores; five titles overlapped. Full output: `evaluation/results/experimental-mixed-comparison.txt`.
- `git diff --check` passed. An initial selector layout regression and a test-fixture/locator error were corrected; no outstanding test failures remain.


This is an experimental opt-in supported by preliminary single-evaluator evidence, not proof of universal improvement. Genre union does not model mood, tone, or audience. Popularity/votes can favor mainstream older films. Candidates outside the top 200 cannot enter. Existing baseline follow-up retrieval remains bounded to 20 candidates, so it can return fewer than requested when many movies are blocked; experimental content removes those identities before forming its pool. Both modes may return fewer when hard filters leave too few eligible movies. Session memory remains process-local and expires under the existing policy. No new mood/tone model is included.
