# Movie AI Recommender

> Final GitHub cleanup (2026-10-09): the prior three ESLint errors and five warnings are resolved. ESLint, TypeScript, production build, all 371 backend tests, all five browser suites and protected-artifact verification pass. See [final cleanup review](docs/FINAL_CLEANUP_REVIEW.md) and [GitHub readiness](docs/GITHUB_READINESS.md). No files were deleted or Git changes staged, committed or pushed during this final task. Earlier verification statements below describe the reorganization milestone.

A conversational movie recommender with a Next.js interface, FastAPI backend, a 62,398-movie catalog, local content embeddings, MovieLens SVD recommendations, verified external TMDB seeds, streaming availability, and bounded session memory. Original ranking is the default; Genre + Votes is an optional development-only content reranker.

## Project layout

```text
backend/                 Python application packages
  api/app.py             FastAPI application and existing routes/TMDB adapters
  core/                  Guardrails and project-root path configuration
  recommendation/        Content, SVD, hybrid, runtime embeddings and ranking
  services/              Assistant, chat, tools and collections
  integrations/          External movie research
  sessions/              Session memory and pending/follow-up handling
frontend/                Existing Next.js application
data/                   Existing datasets and model artifacts
  raw/                   MovieLens originals and historical sample export
  processed/             Catalog and movie mapping
  models/                Embeddings, SVD factors and saved item index
evaluation/
  scripts/               Offline quality, comparison, blind review and legacy metrics
  fixtures/              Verified external movie snapshot
  results/               Preserved measured outputs and private human reviews
tests/
  backend/               All 371 existing offline backend tests and fixtures
  browser/               Movie Agent, presentation, ranking and blind-review checks
scripts/
  data/                  Explicit rebuild commands; never run by normal startup
  diagnostics/           Opt-in diagnostics, including paid/live checks
  legacy/                Preserved standalone historical applications/prototypes
  verify_project_structure.py
docs/                   Architecture, feature guides and cleanup audit
logs/                    Local diagnostics, archived logs and isolated test outputs
```

See [project structure](docs/PROJECT_STRUCTURE.md), [cleanup audit](docs/CLEANUP_AUDIT.md), and the [complete move/integrity manifest](docs/REORGANIZATION_MANIFEST.json).

## Installation

Use the existing virtual environment when available. On a fresh checkout:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
cd frontend
npm.cmd ci
cd ..
```

The datasets and saved models are local assets; a new clone may not include ignored large files. Supply the existing aligned artifacts in their documented locations. Do not regenerate or retrain them as part of ordinary installation. `requirements-chat.txt` retains the existing chat/embedding requirements. Browser checks need Playwright and an installed Chrome browser. Legacy rebuilding/evaluation scripts can train models or call TMDB; they are opt-in tools.

Keep the existing root `.env`. On a fresh checkout only, copy `.env.example` to `.env` and supply your own TMDB_API_KEY and OPENAI_API_KEY. Never commit credentials. MOVIE_CHAT_MOCK defaults to false; true is a development fixture mode with limited supported phrases, not live AI. Live chat and web research can use paid OpenAI requests; automated regression tests use mocks.

## Start the application

Run these commands from the project root in separate terminals.

Backend:

```powershell
.venv\Scripts\python.exe -m uvicorn backend.api.app:app --host 127.0.0.1 --port 8000
```

Frontend:

```powershell
cd frontend
npm.cmd run dev
```

Open http://localhost:3000. The API remains http://127.0.0.1:8000. Endpoints and request/response schemas are unchanged. The old `api:app` module path has been replaced by `backend.api.app:app`; no root compatibility copies are required.

Data and root `.env` loading resolve from the repository rather than the working directory. Module commands need the repository on Python's import path: invoke `python -m ...` from the root, or supply `uvicorn --app-dir <project-directory>` from another directory.

## Enable Experimental ranking

In the backend terminal, before startup:

```powershell
$env:ENVIRONMENT = 'development'
$env:MOVIE_EXPERIMENTAL_RANKING = 'true'
.venv\Scripts\python.exe -m uvicorn backend.api.app:app --host 127.0.0.1 --port 8000
```

In the frontend terminal:

```powershell
cd frontend
$env:NEXT_PUBLIC_MOVIE_EXPERIMENTAL_RANKING = 'true'
npm.cmd run dev
```

Use the Movie Agent ranking selector. Original starts selected. Switching keeps the same session. Any production value in ENVIRONMENT, APP_ENV, or NODE_ENV blocks backend experimental ranking; the frontend control is hidden outside Next development mode. Collaborative-primary requests keep Original ranking. Disable both flags and restart to disable the selector/reranker. See [ranking details](docs/EXPERIMENTAL_RANKING_MODE.md).

## Tests and integrity checks

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests/backend -t .
.venv\Scripts\python.exe -m scripts.verify_project_structure
cd frontend
npx.cmd tsc --noEmit
npm.cmd run lint
```

`python -m unittest discover` from the root also discovers all 371 tests. All tests remain enabled. The verifier checks preserved artifact hashes, runtime text, API schemas, discovery count and loading from a different working directory.

For browser tests, start the flagged frontend above. In a separate project-root terminal, start the offline fixture backend:

```powershell
$env:MOVIE_EXPERIMENTAL_RANKING = 'true'
.venv\Scripts\python.exe -m tests.browser.check_movie_chat_browser --serve
```

Then run:

```powershell
.venv\Scripts\python.exe -m tests.browser.check_movie_chat_browser
.venv\Scripts\python.exe -m tests.browser.check_movie_chat_presentation_browser
.venv\Scripts\python.exe -m tests.browser.check_experimental_ranking_browser
.venv\Scripts\python.exe -m tests.browser.check_word_count_followups_browser
```

These use mock services and real local rankings; the fixture backend binds port 8012. For the human review browser check, use a separate terminal and isolated synthetic storage:

```powershell
.venv\Scripts\python.exe -m evaluation.scripts.human_evaluation serve --store logs/human-browser-test
# Another terminal:
.venv\Scripts\python.exe -m tests.browser.check_human_evaluation_browser
```

Never run synthetic browser ratings against genuine `evaluation/results/human` storage.

## Offline evaluation and ranking comparisons

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
.venv\Scripts\python.exe -m evaluation.scripts.compare_ranking --favorites 157336
.venv\Scripts\python.exe -m evaluation.scripts.compare_ranking --favorites 157336 1248832 --exclude 11802
.venv\Scripts\python.exe -m evaluation.scripts.recommendation_quality diagnose --favorites 157336 --top-n 10 --output logs/quality-diagnostic.json
```

External seeds require the already-cached compatible local encoder and verified fixture. These commands do not query OpenAI/providers, regenerate embeddings, or train SVD. To rerun the full quality evaluation, specify fresh output/report destinations rather than overwriting historical measurements:

```powershell
.venv\Scripts\python.exe -m evaluation.scripts.recommendation_quality evaluate --output logs/quality-evaluation.json --report logs/quality-evaluation.md
```

Normal human review uses `python -m evaluation.scripts.human_evaluation serve`, preserving private reviews in `evaluation/results/human`. See [human evaluation](docs/HUMAN_RECOMMENDATION_EVALUATION.md) for analysis/export commands.

## Reorganization validation

All 371 backend tests, TypeScript, all five browser suites, offline diagnostic/comparison commands, and artifact/API integrity checks passed. Catalog, embeddings and saved SVD load with their original alignment; genuine human records remain unchanged. No training, paid OpenAI calls, Git commit, or push occurred.

Existing ESLint findings remain: three `react-hooks/set-state-in-effect` errors and five warnings in unchanged frontend files. They are documented in the cleanup audit and were not suppressed or fixed during this structural refactor.
