# Project structure and operating guide

> Final GitHub cleanup (2026-10-09): the prior three ESLint errors and five warnings are resolved. ESLint, TypeScript, production build, all 371 backend tests, all five browser suites and protected-artifact verification pass. See [final cleanup review](FINAL_CLEANUP_REVIEW.md) and [GitHub readiness](GITHUB_READINESS.md). No files were deleted or Git changes staged, committed or pushed during this final task. Earlier verification statements below describe the reorganization milestone.

This milestone moves existing files into logical packages without changing application behavior. The complete old?new mapping and pre-move protected hashes are in [REORGANIZATION_MANIFEST.json](REORGANIZATION_MANIFEST.json). The root README contains the exact installation/startup/test commands; feature guides in this directory have updated commands and keep their historical measured conclusions.

## Architecture and responsibilities

The frontend still calls the existing FastAPI routes. `backend/api/app.py` initializes the saved engine, binds TMDB/provider/collection adapters, and exposes the same thirteen FastAPI routes including generated documentation routes. No routes were split or redesigned.

`backend/services/movie_assistant.py` owns conversational tool dispatch. `movie_chat.py` and `chat_service.py` retain their existing controller contracts and prompts; `movie_tools.py` and `chat_recommender.py` resolve verified identities and orchestrate recommendations; `movie_collections.py` handles collection membership.

`backend/sessions/movie_session.py` owns the process-local session store, expiration, clarification state, ordinal references, preferences, watched/excluded identities, and follow-ups. The word-count multi-intent fix is unchanged: ten results followed by ?I've already seen the first three. Give me five more.? produces five new cards in both modes.

`backend/recommendation/recommendation_engine.py` selects the existing content/collaborative lane. Content and saved SVD implementations, runtime external-query encoding, and the shared experimental scorer live alongside it. `hybrid_recommender.py` remains available to historical evaluation scripts; it is not the active API engine. Original remains default; the experimental content-only top-200 formula, production gating and collaborative fallback are unchanged.

`backend/integrations/movie_research.py` implements the existing official web research provider. TMDB and streaming adapters remain bound in the API application to avoid changing initialization, tests, or response behavior. Extracting them is a possible future refactor, not part of this move.

`backend/core/movie_guardrails.py` retains policy checks; `paths.py` resolves resource paths from its source file. Absolute caller-supplied paths remain supported. Twelve legacy artifact labels are resolved to their actual files so old fingerprint keys in evaluation JSON remain meaningful. Historical JSON and human-review payloads were not migrated or edited internally.

## Data and models

- `data/raw/ml-32m/`: original ratings, movies, links, tags and upstream checksum/README files.
- `data/raw/movies.csv`: unique historical TMDB sample export.
- `data/processed/tmdb_movies.csv`: existing 62,398-row runtime catalog.
- `data/processed/movie_mapping.csv`: existing MovieLens/TMDB mapping.
- `data/models/movie_embeddings.npy`: unchanged (62,398, 384) array aligned with catalog row order.
- `data/models/svd_movie_factors.npy`: unchanged (16,034, 50) saved factors.
- `data/models/svd_movie_index.csv`: corresponding item index.

All 29 protected data/model/evaluation files match pre-move SHA-256 checksums. No data/model operation regenerated, retrained, sorted or overwrote these assets. Large files moved on the same filesystem with checked, non-overwriting destinations.

## Evaluation, tests, development tools and history

`evaluation/scripts/` contains quality diagnostics, ranking comparison, report generation, blind review/analysis, and the prior collaborative/hybrid/cold-start/tuning scripts. Historical evaluation code may fit models as part of its original procedure: do not execute those scripts as application startup or as a structural smoke test.

`evaluation/fixtures/external_movies.json` is the unchanged verified external metadata snapshot. `evaluation/results/` retains existing CSV/JSON/text measurements. `evaluation/results/human/` retains genuine anonymous sessions and completed exports privately, with the same content and blind assignments. Browser fixtures use separate `logs/human-browser-test/` storage.

`tests/backend/` contains all existing test modules and shared fixtures. Mock paths and dynamic imports use qualified packages; discovery still finds 371 cases. `tests/browser/` contains five existing browser scripts. Their real API fixture server uses port 8012 and the human review service uses port 8013.

`scripts/data/` retains explicit catalog/embedding/mapping/model builders. `scripts/diagnostics/` retains manual diagnostics, including live checks that can make paid requests; none were run during cleanup. `scripts/legacy/` retains old Streamlit, TMDB, content, KNN and matrix-factorization prototypes without deleting or merging their unique behavior. Use package/module commands from the repository root. The old script paths intentionally no longer exist.

`docs/` contains feature Markdown guides, this document, the cleanup audit, and the move manifest. Historical free-text notes remain in `docs/historical/`. `logs/archive/` preserves prior debugging/regression logs; new local verification and ranking diagnostics go in `logs/`, which is Git-ignored. Existing frontend folders/configuration remain intact.

## Commands

From the project root, backend:

```powershell
.venv\Scripts\python.exe -m uvicorn backend.api.app:app --host 127.0.0.1 --port 8000
```

Frontend (another terminal):

```powershell
cd frontend
npm.cmd run dev
```

Experimental backend: set ENVIRONMENT=development and MOVIE_EXPERIMENTAL_RANKING=true before starting uvicorn. Experimental frontend: set NEXT_PUBLIC_MOVIE_EXPERIMENTAL_RANKING=true before npm run dev. Original remains selected initially. Set flags false and restart to disable. Production values override backend enablement; Next production builds hide the selector.

Backend tests and integrity verification:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests/backend -t .
.venv\Scripts\python.exe -m scripts.verify_project_structure
```

Browser fixture server and checks (separate terminals; flagged Next dev and Chrome required):

```powershell
$env:MOVIE_EXPERIMENTAL_RANKING='true'
.venv\Scripts\python.exe -m tests.browser.check_movie_chat_browser --serve
# Another terminal:
.venv\Scripts\python.exe -m tests.browser.check_movie_chat_browser
.venv\Scripts\python.exe -m tests.browser.check_movie_chat_presentation_browser
.venv\Scripts\python.exe -m tests.browser.check_experimental_ranking_browser
.venv\Scripts\python.exe -m tests.browser.check_word_count_followups_browser
# Human browser service uses isolated storage; start in another terminal:
.venv\Scripts\python.exe -m evaluation.scripts.human_evaluation serve --store logs/human-browser-test
# Another terminal:
.venv\Scripts\python.exe -m tests.browser.check_human_evaluation_browser
```

Ranking comparison:

```powershell
.venv\Scripts\python.exe -m evaluation.scripts.compare_ranking --favorites 157336 1248832 --exclude 11802
```

The [README](../README.md) adds installation, environment configuration, TypeScript/lint commands and safe fresh evaluation output commands.

## Verification results

- All 371 existing backend cases discoverable and passing, without skips or paid requests.
- TypeScript and all five desktop/mobile browser scripts passed.
- Real-catalog offline quality diagnostic and cached mixed-seed comparison completed.
- API OpenAPI schema SHA-256 unchanged; all thirteen routes remain available. The relocated backend was restarted at localhost:8000 and its root/OpenAPI endpoints passed live health checks.
- Saved catalog, embeddings, SVD and all historical evaluation/human records have unchanged protected checksums.
- Loading/importing with a different working directory passed.
- Runtime function ASTs match before/after except the intended diagnostics file path, after ignoring relocated imports. Runtime prompt/text hashes match across nine modules.
- Existing frontend lint errors/warnings remain; see the audit. No frontend design or app source changes were made.

No commit, push, destructive Git operation, paid OpenAI request, embedding rebuild or SVD training occurred.
