# Milestone 2: structured intent to recommendations

`backend/services/chat_recommender.py` provides `ChatRecommender.recommend_from_intent(intent)`.
The controller resolves catalog identities and maps existing preferences. It
does not call OpenAI or implement scoring. `backend/api/app.py` injects a runner that invokes
the unchanged `recommend(RecommendationRequest(...))` flow, reusing its engine
call, TMDB metadata, scores, movie-card fields, and provider enrichment.

`POST /chat` remains parser-only. `POST /recommend-from-intent` accepts a
structured MovieIntent directly and can be exercised without OpenAI credit:

```json
{
  "intent": "recommend",
  "favorite_movies": ["Interstellar"],
  "top_n": 3,
  "country_code": "SE",
  "natural_language_preferences": ["less depressing"]
}
```

Use `/docs` to submit the request, or call the shared controller directly:

```python
from api import chat_recommender
from chat_service import MovieIntent

result = chat_recommender.recommend_from_intent(
    MovieIntent(intent="recommend", favorite_movies=["Interstellar"], top_n=3)
)
print(result.model_dump())
```

API initialization loads the existing persisted engine artifacts once. Direct
calls and this endpoint require no OpenAI requests; real provider enrichment
still uses the existing TMDB API behavior. Tests replace enrichment to stay offline.

## Resolution policy

The read-only title index is built from `engine.content.movies`, the same catalog
used by website search. Matching uses case-insensitive exact titles with whitespace
normalization. No fuzzy, substring, external TMDB, or title-only scoring fallback
is used. Duplicate catalog rows are deduplicated by TMDB ID.

- A single exact identity resolves to `{tmdb_id, title, year}`.
- A title such as `Crash (1996)` explicitly restricts the reference release year.
  Literal complete titles are checked first. Result year filters never determine
  the reference movie's identity.
- Multiple exact identities require clarification unless one overwhelmingly
  dominates *both* catalog recognition signals: at least 1,000 votes, at least
  10 times every competitor's votes, and at least 5 times every competitor's
  popularity. Missing recognition data does not justify automatic selection.
  These thresholds are named constants, not recommendation weights. They are
  an entity-resolution heuristic; a caller can specify the year to remove doubt.
- Unmatched names return `unresolved_movies`, without substituting another movie.
- Any unresolved or ambiguous favorite blocks ranking entirely; the controller
  reports all issues alongside already resolved movies, rather than using a subset.

In the current catalog, these rules resolve Coco to 354912 (2017), Inside Out to
150540 (2015), and Zootopia to 269149 (2016). Crash remains ambiguous between
884 (1996) and 1640 (2005); the local catalog labels the latter 2005, not 2004.

## Mapping and results

All existing scoring/filter inputs map directly: resolved `favorite_movies`,
`favorite_genres`, `favorite_actors`, `favorite_directors`, `required_genres`,
`required_actors`, `required_directors`, `exclude_genres`, `exclude_actors`,
`exclude_directors`, `min_year`, `max_year`, and `top_n`. Unspecified `top_n`
preserves the existing request default of 10. No preferences are inferred from
reference movies by this controller or promoted from required to favorite.

`country_code` defaults to the existing `SE` region and is passed to the existing
API enrichment flow; it is never an engine ranking argument. `movie_titles` and
`natural_language_preferences` are returned under `unsupported_preferences`,
with their original field names; they do not influence scores.

Result fields are `status`, `message`, `resolved_movies`, `clarifications`,
`unresolved_movies`, `recommendations`, `unsupported_preferences`, and
`country_code`. Recommendation cards have exactly the existing `/recommend`
format. The endpoint returns these normal controller outcomes as HTTP 200:

| Status | Meaning |
| --- | --- |
| `success` | Existing flow ran; recommendations may be empty when filters find no matches. |
| `needs_clarification` | Select among identities in each clarification's `options`. |
| `unresolved` | One or more titles are absent from the local catalog. |
| `needs_preferences` | No reference movie or soft preference was supplied. |
| `unsupported_intent` | Search, information, and unknown intents are not executed here. |

Invalid MovieIntent bodies return HTTP 422; unexpected execution errors return a
sanitized HTTP 500. No clarification UI, conversation state, or selection endpoint
was added. For now callers can resubmit a year-qualified title.

**Existing engine limitation:** Required filters alone are not scoring seeds.
The current content model requires a reference movie or a favorite genre/actor/
director, and the engine can fail on a metadata-free empty frame otherwise.
The controller therefore returns `needs_preferences` for a request containing
only required filters, years, or unsupported nuances. It does not change scoring
or silently convert required filters into favorites. Existing `/recommend`
behavior is unchanged.

## Offline tests

From the project root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.backend.test_chat_recommender test_chat -v
```

These tests use existing catalog, embeddings, and persisted SVD factors, replace
provider calls, and block network access. They verify correct reference IDs reach
the real engine/content model, actual recommendation output, required/excluded
filters, year bounds, ambiguity, unresolved names, unsupported metadata, and
unchanged rankings/scores across regions. A real `/recommend` response is compared
with the controller's movie-card list to verify website response compatibility.
Milestone 1 tests continue to verify that `/chat` does not invoke recommendations.
