# Cleanup audit

> Final GitHub cleanup (2026-10-09): the prior three ESLint errors and five warnings are resolved. ESLint, TypeScript, production build, all 371 backend tests, all five browser suites and protected-artifact verification pass. See [final cleanup review](FINAL_CLEANUP_REVIEW.md) and [GitHub readiness](GITHUB_READINESS.md). No files were deleted or Git changes staged, committed or pushed during this final task. Earlier verification statements below describe the reorganization milestone.

This is a structural refactor. No source, test, documentation, dataset, model artifact, measured result, or genuine human-review record was deleted. No major duplicate business logic was merged. All original modified/untracked files were treated as project assets rather than disposal candidates. The pre-move Git worktree was clean; Git history remains intact and no changes were staged, committed, or pushed by this task.

The repository-wide inventory covered every root Python module with AST imports and top-level definitions, runtime/dynamic imports, mock targets, file readers/writers, standalone commands, frontend API/configuration, feature docs, local datasets, saved models and evaluation records. The [move manifest](REORGANIZATION_MANIFEST.json) lists all 145 moves and protected hashes. It includes unique standalone code even when no module imports it.

## A. Files safely removed

Only 47 generated root Python cache files and the task's empty scan file were removed. No logs were deleted: their possible debugging/reproducibility value led to preservation under `logs/archive/`. Eighteen clearly unused import bindings were also removed after checking in-file usage and cross-module/re-export references; no registration/initialization import was removed. Details are retained in [UNUSED_IMPORT_CANDIDATES.json](UNUSED_IMPORT_CANDIDATES.json). The active ranking log moved to `logs/ranking.jsonl`; new local diagnostic events may append to it. Framework/venv caches were left intact.

| Original path | Reason and evidence |
|---|---|
| `__pycache__/api.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/chat_recommender.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/chat_service.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/check_chat_live.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/check_movie_assistant_live.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/check_movie_chat_browser.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/check_movie_guardrails_live.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/content_recommender.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/experimental_ranking.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/human_evaluation.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/hybrid_recommender.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/movie_assistant.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/movie_chat.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/movie_collections.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/movie_guardrails.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/movie_research.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/movie_session.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/movie_tools.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/recommendation_engine.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/recommendation_quality.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/recommendation_quality_report.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/research_test_fixtures.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/runtime_movie_embeddings.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/svd_recommender.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/test_chat.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/test_chat_recommender.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/test_conversation_policy.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/test_experimental_ranking.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/test_external_movie_embeddings.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/test_franchise_routing.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/test_human_evaluation.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/test_live_movie_resolution.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/test_movie_assistant.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/test_movie_chat.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/test_movie_collections.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/test_movie_research.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/test_movie_session.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/test_movie_tools.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/test_multi_intent_followups.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/test_pending_clarification.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/test_project_hail_mary.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/test_recommendation_quality.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/test_recommendation_references.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/test_research_diagnostics.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/test_response_polish.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/test_watch_order_consistency.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `__pycache__/test_word_count_followups.cpython-314.pyc` | Generated CPython bytecode; the corresponding source is preserved and Python recreates it. |
| `dependency-inventory.txt` | Empty scan output produced during this task; zero bytes, no references or unique data. |

Obsolete empty parent directories were removed only after their contents had been moved. Every move/deletion target was resolved inside the workspace. Moved source files were edited only for imports, resource paths and command references; no dataset/model destination was overwritten.

## B. Potentially unused files preserved

| Current path | Purpose and reference evidence | Why preserved |
|---|---|---|
| `scripts/legacy/streamlit_app.py` | Previous Streamlit UI; imports the active RecommendationEngine but is not imported by the FastAPI/Next runtime. Standalone entry point. | Unique UI and historical workflow; removal requires approval. |
| `scripts/legacy/tmdb_cli.py` | Previous standalone TMDB sample exporter; writes the preserved `data/raw/movies.csv`. No runtime importers. | API/data rebuild provenance and a unique sample dataset. |
| `scripts/legacy/content_demo.py` | Earlier interactive content recommender; no active runtime/test/evaluation importer found. | Unique interactive behavior and older scoring implementation; not assumed dead. |
| `scripts/legacy/collaborative_demo.py` | Standalone KNN prototype/evaluation; no active application importer. | Historical comparison/reference behavior. |
| `scripts/legacy/matrix_factorization.py` | Standalone factorization prototype; not the saved SVD loader used by the API. | Training/evaluation provenance; no execution during refactor. |
| `backend/recommendation/hybrid_recommender.py` | Used by historical comparison, hybrid evaluation and weight-tuning scripts; not by the active API RecommendationEngine. | It is demonstrably used by offline tools, so not disposable. |
| `scripts/diagnostics/check_*_live.py` and `diagnose_movie_research.py` | Manual HTTP/OpenAI/TMDB diagnostics, referenced by guides; standalone entry points rather than production imports. | Useful support/reproduction tools; some are paid/live and remain opt-in. |
| `tests/backend/test_project_hail_mary.py` | Standalone recommendation diagnostic; its import-only behavior is explicitly regression-tested by pending-clarification tests. | Cannot be removed merely because unittest finds no cases in that module. |
| `evaluation/scripts/evaluate_*.py`, `tune_*.py`, `compare_models.py`, `analyze_content_model.py` | Standalone historical benchmarks and tuning scripts, associated with retained results CSVs. | Unique evaluation protocols/provenance; some fit models, so none were run during structural validation. |

All root CSV/text evaluation measurements, prior logs and genuine human records were retained. No missing-import evidence was used as proof of dispensability. A reference found only in docs or direct execution is still a valid dependency.

## C. Potential duplicate implementations

- `scripts/legacy/content_demo.py` and the active `backend/recommendation/content_recommender.py` overlap in content similarity/preferences. They represent different development stages; merging could change scoring and was deliberately avoided.
- `backend/recommendation/hybrid_recommender.py` and `recommendation_engine.py` both combine content/collaborative evidence but serve different historical/runtime callers. Keep distinct unless a separately evaluated migration is approved.
- KNN/factorization prototypes and the multiple collaborative/SVD evaluation scripts overlap in sampling, scoring and metric helpers. Their versions/protocols/results are distinct; preserve reproducibility before consolidating helpers.
- `recall_at_k` and `ndcg_at_k` helpers recur across evaluation scripts. A future shared metrics module is possible only after verifying each script's denominator, handling of missing relevance, rank slots and split protocol.
- TMDB sampling and catalog-building code overlap, but write different datasets and support different completeness workflows.
- Existing cross-test fixture reuse/inheritance intentionally reruns external regressions in several settings. All 371 discovered cases were preserved, not deduplicated to reduce coverage.

## D. Outdated documentation

Feature Markdown guides moved into `docs/` with startup/module/data references updated. Their milestone-specific counts (for example 287, 304 or 361 tests), capability statements and quality judgments are historical evidence rather than current global validation. The current architecture/commands/results live in PROJECT_STRUCTURE.md and the root README. Earlier parser-only/chat-only guides remain useful historical descriptions and already identify successor guides.

`docs/historical/evaluation results and details.txt` and `movie_recommender_hybrid_summary.txt` retain original free-text provenance and may contain old root filenames/commands. They are archived notes, not current startup instructions; do not follow those paths. They were preserved rather than rewritten or deleted.

`docs/EXPERIMENTAL_RANKING_MODE.md` preserves the earlier measured comparison and limitations, with new file/module/log paths. Saved evaluation JSON keeps its original artifact label keys; `backend/core/paths.py` resolves those labels without modifying any measured JSON or human rating.

## E. Files requiring approval before possible removal

The five `scripts/legacy/` prototypes, historical evaluation versions/results, older diagnostic scripts, archived free-text documentation and offline hybrid class could be considered for a future retirement review. Every one remains present. Removal would require explicit user approval plus review of standalone workflows/provenance; this audit is not approval to delete them. No deletion approval is required to finish the current task because those candidates were preserved.

## F. Recommended future cleanup

- Extract TMDB/provider adapters and route wiring from the API application in a separate behavior-tested refactor.
- Give shared test fixtures their own modules rather than importing test classes; retain equivalent case coverage and guard against accidental duplicate collection.
- Review historical benchmark protocols before consolidating metric helpers or retiring prototype commands.
- Add a separately reviewed dependency lock for reproducible fresh installs; dependencies were not installed/upgraded during this move.
- Treat the three existing frontend `react-hooks/set-state-in-effect` errors and five warnings as a separate UI task. The errors occur in unchanged `frontend/app/components/ManualRecommend.tsx` at lines 167 and 1081 and `frontend/app/evaluation/EvaluationReview.tsx` at line 83. No lint rules were disabled and no app logic was changed to make lint pass.
- Decide a retention policy for archived debugging logs after reviewing their reproduction value. Existing logs were not assumed disposable.
- A conservative top-level import scan is retained in [UNUSED_IMPORT_CANDIDATES.json](UNUSED_IMPORT_CANDIDATES.json), including the disposition of eighteen safely removed bindings. Future scans must still check re-exports and registration/initialization side effects; an unused binding does not prove an entire module is dead code. Unused newly introduced path imports were also removed.

## Validation and remaining issues

All 371 backend tests, TypeScript, five browser suites, API schema comparison, non-root resource loading, offline quality diagnostic, real mixed-seed ranking comparison, prompt/text integrity and protected artifact checks passed. All 29 protected files retain checksums, including two genuine human-review files. No test was skipped or disabled.

Existing ESLint findings are the only outstanding check failures (three errors, five warnings in unchanged frontend files). Initial migration failures were structural logger/mock/script-path references and were fixed without changing test assertions about application behavior. A local backend holding the diagnostics log was stopped to permit the safe move, then restarted using the new module command. The relocated API is healthy at http://127.0.0.1:8000; its root and OpenAPI endpoints were verified. No paid APIs, training, regeneration, Git commit or push occurred.
