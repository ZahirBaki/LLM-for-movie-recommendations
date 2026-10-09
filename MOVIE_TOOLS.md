# Conversational integration

Grounded collection tools and normalization now extend these capabilities; see
[MOVIE_COLLECTIONS.md](MOVIE_COLLECTIONS.md). No scoring/artifact changes are involved.

The capabilities below are now exposed through the real tool-calling controller
documented in [MOVIE_ASSISTANT.md](MOVIE_ASSISTANT.md). Their backend implementations
and ranking behavior are unchanged.

# Future Movie AI tool contracts

These are tested Python capabilities, not active OpenAI tools. No LLM tool calling,
conversational answers, memory, or new HTTP routes were added. Existing chat routes
and frontend behavior are unchanged. `movie_tools.py` has no API import, model
initialization, or OpenAI request. Application binding happens once in `api.py`:

```python
from api import movie_tools
from chat_service import MovieIntent

matches = movie_tools.search_movies("Interstellar")
identity = movie_tools.resolve_movie_title("Interstellar")
if identity.status == "success":
    facts = movie_tools.get_movie_details(identity.movie.tmdb_id)
    watch = movie_tools.get_watch_providers(identity.movie.tmdb_id, "SE")
recommendations = movie_tools.recommend_movies(
    MovieIntent(intent="recommend", favorite_movies=["Interstellar"], top_n=3)
)
```

Importing `api` loads the existing persisted recommendation artifacts and the
existing `.env` configuration. Tests can construct `MovieTools` with an injected
catalog/controller/search runner/provider runner instead. The bound methods are
the primary reusable tool functions; no new development endpoints are needed.

## search_movies

**Purpose:** Find local catalog candidates while preserving TMDB identities.

**Input:** `query: str`, `limit: int = 10`. Queries require at least two non-padding
characters; limits are 1–10, matching the current website search's capacity.

**Output:** `SearchResult` with status (`success`, `not_found`, `error`), query,
matches, and optional message. Each match contains tmdb_id, title, year,
original_title, genres, and the existing w185 poster URL.

**Use for:** “Find Interstellar”, “Find movies called Crash”, catalog discovery.

**Do not use for:** Ranking recommendations, automatically resolving ambiguity,
or answering factual questions from a title alone.

The tool delegates to existing `/movies/search` functionality, preserving its
prefix/substring search, exact-match/popularity ordering and ten-result cap.
Meaningful same-title movies are separate results. Increasing search capacity or
searching original-title aliases is future work; no conflicting search algorithm
was added here.

## resolve_movie_title (shared supporting capability)

**Purpose:** Safely convert a title into a canonical movie identity before lookup.

**Input:** `title: str`, optionally year-qualified such as `Crash (1996)`.

**Output:** `ResolutionResult`: success plus one movie, not_found, error, or
needs_clarification plus options. Movie identities contain tmdb_id, title, year.

**Use for:** A factual/watch request that names a movie but has no TMDB ID yet.

**Do not use for:** Choosing arbitrarily among clarification options or resolving
pronouns/context such as “that one”. Search first if the title is incomplete.

It delegates directly to the same `ChatRecommender.resolve_title` used for
recommendations. Exact matching, year qualification, and documented recognition
dominance thresholds remain unchanged. Crash needs clarification; Coco resolves
to 354912 (2017), Inside Out to 150540 (2015), Zootopia to 269149 (2016).

## get_movie_details

**Purpose:** Retrieve factual metadata grounded in the current local TMDB catalog.

**Input:** `tmdb_id: int`, positive. Resolve ambiguous titles before calling.

**Output:** `DetailsResult`: status (`success`, `not_found`, `error`), movie or null,
and optional message. MovieDetails contains tmdb_id, title, original_title,
release_date, release_year, genres, director, actors, overview, original_language,
popularity, vote_average, vote_count, runtime, poster_path, w500 poster_url, and
`rating_source = "TMDB"`. Genres/director/actors preserve catalog string formats.

**Use for:** Rating, cast, director, overview, release date/year, genre, runtime.

**Do not use for:** Recommendation ranking or claiming an IMDb rating. Ratings
are the TMDB user-vote average recorded in this catalog, not LLM memory or IMDb.

Missing/empty/nonfinite values become null; no facts are inferred. Metadata is a
catalog snapshot and may be incomplete or stale. No live TMDB detail fallback
or catalog rebuild was added. Unknown IDs return not_found.

## get_watch_providers

**Purpose:** Retrieve regional availability through the existing TMDB fetcher.

**Input:** `tmdb_id: int`, `country_code: str` (two ASCII letters, normalized to
uppercase). This milestone requires the movie to exist in the local catalog.

**Output:** `WatchResult` with status (success, not_found, not_available, error),
tmdb_id, message, and `watch_providers`. Provider data preserves country_code,
stream, free, ads, rent, buy, link. Each provider has provider_id, name, logo_url,
and display_priority; missing metadata remains null.

**Use for:** “Where can I watch Dune in Sweden?”, streaming/rental/purchase lookup.

**Do not use for:** Recommendation ranking, assuming a subscription is owned,
or inventing availability when no data is returned.

The tool delegates to the existing fetcher; TMDB flatrate still maps to stream.
Country affects enrichment only. No provider API implementation was duplicated.
The fetcher already returns empty data for absent regional results, missing TMDB
credentials, and request failures. `not_available` means data is unavailable,
not proof that the movie cannot be watched anywhere. Unexpected/malformed
provider results produce a sanitized error with empty categories. Raw exception
logging was removed from the existing fetcher to avoid leaking request URLs/keys.

## recommend_movies

**Purpose:** Execute the existing controller and recommendation pipeline.

**Input:** Validated `MovieIntent`, with reference movies, favorite/required/
excluded genres/actors/directors, inclusive year bounds, top_n, and country_code.

**Output:** Existing IntentRecommendationResult fields, with an additional error
status for sanitized execution failure: status, message, resolved_movies,
clarifications, unresolved_movies, recommendations, unsupported_preferences,
country_code. Existing movie-card fields and recommendation order are preserved.

**Use for:** Movie recommendation requests.

**Do not use for:** Factual lookups, casual conversation, unsupported semantic
reranking, or ranking movies in the LLM itself.

The wrapper calls `ChatRecommender.recommend_from_intent`, which reuses the
existing `/recommend` execution/formatting and RecommendationEngine scoring.
No second algorithm, popularity fallback or filter-to-favorite conversion exists.
`natural_language_preferences` and `movie_titles` remain unsupported metadata.
Required-only category requests such as “Recommend a comedy” return
`needs_preferences` with the existing explanatory reason. No scoring changes
were made to work around this limitation.

## Serialization, errors and future control

All public results are Pydantic models. `.model_dump()` produces clean Python
data suitable for `json.dumps(..., allow_nan=False)`; `.model_dump_json()` is also
supported. Pandas missing values, NumPy scalars and NaN/Infinity are normalized.
Missing numeric values are null rather than fabricated scores/ratings. Invalid
inputs and unexpected failures return structured, sanitized statuses/messages.

Next milestone: an LLM controller may choose a tool, use tool-supplied facts, or
answer conversationally without a tool. Casual messages must not be forced into
recommendation/search. This milestone does not wire that routing or tool calling.

## Offline verification

```powershell
.\.venv\Scripts\python.exe -m unittest test_movie_tools test_movie_chat test_chat_recommender test_chat -v
```

The tests compare Interstellar metadata against catalog rows, expose Crash search
duplicates/clarification, preserve animated-film identities, mock the existing
TMDB HTTP response to verify all provider categories, confirm unchanged ranking
between regions, compare recommendation wrapper/controller output, and enforce
strict JSON serialization for missing/nonfinite data. Existing route tests cover
search, people, genres, recommendations, parser-only chat, structured controller,
and mock orchestration. No OpenAI credit, retraining or regeneration is required.
The existing `check_movie_chat_browser.py` harness remains usable for website
regression checks with fixture provider/poster assets and real ranking.

Validation for this milestone: all 38 backend tests passed, including nine new
movie-tools tests. Python compilation and whitespace checks passed. FastAPI
completed Uvicorn startup, and the browser harness passed manual and mock chat,
movie/actor/director search, genre selection, country/outside-click behavior,
streaming logos, rent/buy/watch links, error handling, mobile layout and refresh
reset. Provider/poster browser data were fixtures; recommendation ranking used
the persisted real artifacts. No live OpenAI requests were made.
