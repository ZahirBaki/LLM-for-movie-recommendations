# Movie Assistant

Current policy: [MOVIE_POLICY.md](MOVIE_POLICY.md), superseding earlier scope
restrictions and allowing interpretive story chronology from general knowledge.

Collection support and scope/cost guardrails now extend this architecture; see
[MOVIE_COLLECTIONS.md](MOVIE_COLLECTIONS.md) for current limits, title normalization,
seven tools, backend-formatted collection replies and the new milestone checks.

The `/movie-chat` endpoint now uses real OpenAI Responses API function calling.
The stateless path is user message + selected country → `MovieAssistant` →
validated allowlisted `api.movie_tools` capabilities → OpenAI explanation plus
backend-owned structured results. There is no conversation/session memory.
The legacy parser `/chat`, structured `/recommend-from-intent`, and manual
recommender remain available.

`OPENAI_MODEL` is configured once in `chat_service.py`: environment override or
`gpt-4.1-mini`. The controller reuses the installed asynchronous OpenAI SDK,
`store=False`, no automatic retries, a 30-second per-request timeout and a
100-second total live-controller deadline.

## Capabilities and grounding

| Tool | Selection and source |
| --- | --- |
| `search_movies` | Catalog title discovery, up to 10 matches; preserves duplicate identities. |
| `resolve_movie_title` | Existing safe exact-title/year resolver; unique results authorize IDs for later facts/providers. |
| `get_movie_details` | Local catalog facts, TMDB user ratings/vote counts, cast, director, runtime, release date, overview and genres. Missing data remains unknown. |
| `get_watch_providers` | Existing backend/TMDB integration; stream, free, ads, rent, buy and JustWatch link. |
| `recommend_movies` | Existing ChatRecommender → RecommendationEngine → existing card formatting/enrichment. Engine order and scores are preserved. |
| `search_movie_collection` | Resolve a TMDB collection name; preserve ambiguous groups and partial search coverage. |
| `get_movie_collection` | TMDB membership sorted by release date in the backend; does not establish broad-franchise completeness or story chronology. |

The system instructions require tools for authoritative backend/TMDB facts,
catalog searches and providers. General discussion may use model knowledge.
Titles become TMDB IDs through backend resolution, never model
knowledge. Fact/provider calls with an ID not uniquely resolved within the current
turn are rejected. Search alone does not authorize an ambiguous selection.
Recommendation titles resolve within the existing recommendation controller.
Crash ambiguity produces actual catalog choices and disables further tools for
that turn; year-qualified requests resolve the selected identity. Clarification
buttons submit a fresh complete request with the title qualified by year, preserving
the original factual/provider/recommendation request and other constraints.

Country defaults to the selected frontend region. The model extracts an explicit
user country as an ISO code and that takes precedence. Region only affects
provider lookup/enrichment; it never changes ranking. Unavailable provider data is
not proof of absence. The existing fetcher cannot distinguish all missing-data,
credential and upstream-failure cases; this limitation remains.

Casual greetings, thanks and capability questions can produce text without tools.
Ordinary unrelated questions receive normal concise answers. Required-only genre/year
requests still return `needs_preferences`; there is no popularity fallback.
Unsupported natural-language nuances remain metadata, not new ranking filters.

## Safety and request size

Tool names use an explicit runner allowlist. Pydantic validates JSON arguments,
forbids extra fields and coercion, checks IDs, country format, year ranges and
counts, and caps recommendation cards at 20. Argument payloads are capped at
16,000 characters. Parallel calls are disabled and multiple calls in one response
are rejected. At most five tool rounds and ten total tool calls are allowed,
with one final synthesis request. Malformed calls, unknown tools, timeouts,
authentication/quota/network failures and unexpected exceptions yield a sanitized
HTTP 503 response. Backend structured tool errors can yield HTTP 200 `error`.
No exception text or API key crosses the frontend boundary.

Only the current request, instructions, tool definitions and relevant results go
to OpenAI. Image URLs, ranking scores and internal engine fields are omitted from
model-facing tool results. Full catalog, embeddings and SVD factors are never sent.
Movie cards come directly from backend results; the model cannot modify them.
Generated prose is instructed to use tool facts, but is still model-generated and
not independently fact-checked sentence by sentence.

## Response contract

Existing fields remain: `status`, `assistant_message`, `recommendations`,
`clarifications`, `resolved_movies`, `unresolved_movies`, `unsupported_preferences`,
`country_code`, `message`, `parsed_intent`, and `mock_mode`.
Statuses remain `success`, `needs_clarification`, `unresolved`, `needs_preferences`,
`unsupported_intent`, and `error`. Factual/casual requests normally have no
recommendation cards. `parsed_intent` is populated for recommendation tool calls.

Additive fields: `tool_results` contains allowlisted tool names and structured
backend results; `openai_requests` counts completed OpenAI requests for the turn.
`collection` and `collection_choices` contain verified collection results and
collection alternatives; collection text is formatted entirely by the backend.
Tool names are metadata and never rendered as chat text. Provider replies render
regional categories/link/attribution without creating recommendation cards.
On an exception the sanitized fallback does not preserve partial request counts.

## Development and verification

Keep `.env` credentials local. API startup owns dotenv loading.
`MOVIE_CHAT_MOCK=true` enables deterministic development phrases without OpenAI;
production ENVIRONMENT/APP_ENV/NODE_ENV disables fixtures. Existing recommendation
phrases still use real ranking. Added fixtures include `Hello`, `How are you?`,
`Thanks`, `What can you do?`, `Who directed Interstellar?`,
`What's Interstellar's rating?`, `Who stars in Dune (2021)?`,
`Tell me about Crash`, `Tell me about Crash (1996)`, `Find movies called Crash`,
and `Where can I watch Interstellar in Sweden?`. Mock provider calls still use the
backend unless patched by offline tests/browser harness. This is a fixture mode,
not a full natural-language parser. Unknown mock phrases fail cleanly.

```powershell
# Offline backend checks; no OpenAI requests
.\.venv\Scripts\python.exe -m unittest test_movie_assistant test_movie_tools test_movie_chat test_chat_recommender test_chat -v
# Frontend type check
cd frontend
npx.cmd tsc --noEmit
cd ..
# Backend
.\.venv\Scripts\python.exe -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
# Frontend, in another terminal
cd frontend
npm.cmd run dev
```

For fixture browser checks run `check_movie_chat_browser.py --serve` in a separate
terminal (port 8012), keep the frontend on port 3000, then run
`.\.venv\Scripts\python.exe check_movie_chat_browser.py`. Requires Playwright and
installed Chrome. It uses real ranking and fixture providers/posters.

Only after offline checks pass, explicitly run the optional paid check:
`.\.venv\Scripts\python.exe check_movie_assistant_live.py`. It sends five independent
messages once, prints HTTP/status/text/cards/tool metadata and request count,
and stops on failure. It is not part of unit tests.

Validated for this milestone: 51 backend tests passed, frontend TypeScript passed,
and browser checks passed manual/search/people/genre/country controls, chat cards,
provider UI, factual/casual text, factual and recommendation Crash selection,
loading/duplicate suppression, errors, mobile layout and refresh reset.
Five live messages all returned HTTP 200/success: casual (0 cards), Interstellar
rating (0), director (0), recommendation (5), Sweden providers (0).
The live check made 12 actual OpenAI requests. Rating output was 8.5/10; director
output Christopher Nolan. Provider output used SE despite selected US. Live provider
names/availability are a test-time snapshot, not a guarantee of future availability.

No scoring, model artifacts, catalog, embeddings or SVD factors were changed.
No accounts, database, follow-up references or conversation memory were added.
Clarification buttons still require a release year; same-title/same-year collisions
need future explicit-ID selection support. Existing ranking quality is unchanged.
