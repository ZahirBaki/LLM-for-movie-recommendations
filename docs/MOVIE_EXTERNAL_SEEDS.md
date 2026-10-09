# Temporary external movie recommendation seeds

MovieTools resolves exact TMDB identity before scoring. External seeds use the
same RecommendationEngine and card/provider enrichment as local seeds. They
are never appended to the catalog, embedding matrix, or MovieLens/SVD model.

## Encoding compatibility

`scripts/data/build_embeddings.py` encodes only `overview.fillna("").astype(str)` with
`sentence-transformers/all-MiniLM-L6-v2`, `normalize_embeddings=True`, producing
384-dimensional vectors. No title, genre, actor or director text is embedded.
Runtime encoding uses the raw verified overview, the same model and encode
options. Metadata supplies the existing separate genre/cast/director features.
Every vector must have exactly 384 finite values and unit norm. Empty overviews,
missing identity data, invalid vectors and unavailable models stop scoring.

The model loads lazily on CPU, from provisioned local Hugging Face files only,
and is reused per backend process. Before first use, three catalog overviews are
encoded and compared to their stored vectors (absolute tolerance 1e-4, relative
1e-3). This detects incompatible weights/tokenizers: the original build did not
record a model revision. No OpenAI embeddings or runtime model downloads occur.
Install `requirements-chat.txt` and provision the named model before deployment.
A failed initialization has a 60-second retry cooldown.

## Ranking and mixed favorites

`RecommendationEngine.recommend(external_queries=...)` accepts temporary verified
query representations. ContentRecommender compares local and external seed
vectors together against its existing normalized matrix using dot products and
the existing 70% average / 30% strongest-match blend. Existing story normalization,
feature inference, content weights and hard filters apply unchanged.

Requests containing external seeds use the engine's existing content-only
ranking path, including requests with multiple local favorites. External seeds
have no collaborative interaction vectors. No SVD scores are manufactured;
returned cards have null collaborative scores. Local-only SVD and content paths
are unchanged. All candidates come from the local catalog and all seed IDs are
excluded. The session layer continues to filter seen, excluded and already shown
movies, within its existing 20-candidate bound.

## Cache and sessions

A thread-safe LRU cache holds up to 128 verified external query representations
per process for one hour. Entries include identity, metadata and a read-only
384-dimensional vector. Metadata changes invalidate the vector via a fingerprint.
Cache hits avoid repeated encoding and qualified-title/detail TMDB calls.
Unqualified title searches still check ambiguity. Cache eviction/expiry can cause
re-verification/re-encoding, using the already loaded model.

Session state stores up to 20 backend-resolved active seed identities, never
vectors. Pending clarification, watched/excluded tracking, country, more results
and ordinal detail/provider lookups continue through the existing session flow.
Session TTL and isolation remain unchanged. No cache or session data is persisted.

## Validation and limits

Offline tests mock TMDB, the encoder and OpenAI while exercising actual engine
scoring and card enrichment. A separate real-model smoke check reproduced three
stored probe vectors; reported maximum difference for the first was 7.8e-08.

Usable verified TMDB metadata and locally provisioned model files are required.
External/mixed requests use content-only ranking. Catalog/provider data retain
existing snapshot/availability limitations. The original build has no model
revision manifest; probe comparisons provide a compatibility check, not exhaustive
reproduction of every stored vector. Caches are bounded and process-local.


## Files and final regression results

Production: `backend/recommendation/runtime_movie_embeddings.py`, `backend/recommendation/content_recommender.py`,
`backend/recommendation/recommendation_engine.py`, `backend/api/app.py`, `backend/services/movie_tools.py`, `backend/services/movie_assistant.py`,
`backend/sessions/movie_session.py`, `requirements-chat.txt`.
Tests: `tests/backend/test_external_movie_embeddings.py`, `tests/browser/check_movie_chat_browser.py`.
Documentation: `MOVIE_EXTERNAL_SEEDS.md`, `MOVIE_SESSION_MEMORY.md`.

Full backend discovery: 246 passed, zero failures, zero collection errors;
this includes all original 222 tests plus 24 external-query regressions.
TypeScript `tsc --noEmit` passed. Both browser scripts passed desktop/mobile,
external seed cards, null SVD scores, more/seen/detail follow-ups, pending
clarification, providers, navigation and presentation/scroll regressions.
TMDB, OpenAI and encoder calls in regressions were mocked; the separate cached
model compatibility smoke test used the real local encoder, without paid calls.
Catalog, embedding and SVD artifacts, ranking constants and frontend layout
were unchanged in this milestone.
