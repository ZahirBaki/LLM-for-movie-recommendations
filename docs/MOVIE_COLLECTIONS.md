# Grounded collections and Movie AI guardrails

Current policy: [MOVIE_POLICY.md](MOVIE_POLICY.md). Earlier movie-only scope and
assistant chronology restrictions below are superseded. Collection tools still
provide release order only; general story chronology can use model knowledge.

This milestone keeps requests independent. No session memory, database, accounts,
history, scoring changes, retraining, catalog rebuild or embedding rebuild was added.

## Response polish

Collection presentation now leads with a friendly release-order introduction and
uses `Title (year)` by default. Exact release dates appear when requested. Technical
source/coverage/chronology fields remain unchanged in structured results; they are
not automatically copied into normal prose. Explicit broad-franchise requests
still receive a short warning that the collection cannot verify every related
film. Explicit story-order requests retain a natural explanation that a reliable
story chronology is unavailable. Source questions can mention TMDB naturally.
Ambiguity, missing dates and truncated results still receive useful short notices.

The formatter receives only the current user message to decide presentation;
this adds no conversation memory. Collection membership, sorting and trusted
backend rendering are unchanged. General assistant style instructions now favor
direct answers, useful labels such as "TMDB rating", and natural country/provider,
search and recommendation language without tool/engine/status terminology.

Polish verification: **83 offline tests passed**, including metadata/card equality,
no extra collection films, ordinary versus exact-date lists, contextual caveats,
explicit source wording and unchanged guardrails. The existing browser harness
passed without changing the frontend or response contract. One live run using
`scripts/diagnostics/check_movie_guardrails_live.py --polish` tested X-Men order (two OpenAI requests)
and Interstellar rating (three), **five actual OpenAI requests total**, no reruns.
The 900-token default remains unchanged. The live list contained X-Men (2000),
X2 (2003), The Last Stand (2006), First Class (2011), Days of Future Past (2014),
Apocalypse (2016) and Dark Phoenix (2019). The rating reply was "Interstellar has
a TMDB rating of 8.5 out of 10 based on over 41,000 votes."

## X-Men identity investigation

Direct inspection of `data/processed/tmdb_data/raw/movies.csv` found:

| TMDB ID | Title | Original title | Release date | Release year |
| --- | --- | --- | --- | --- |
| 36657 | X-Men | X-Men | 2000-07-13 | 2000 |

The old resolver successfully handled `X-Men` and `X-Men (2000)`. It reproducibly
failed for `x-men 2000`, `X Men`, and `xmen`. Its normalization handled case and
whitespace only; its year parser accepted parentheses only. There is no evidence
of a corrupt/missing catalog artifact. The original browser run did not preserve
tool arguments, so its exact failed argument cannot be reconstructed; the
reproducible resolver gaps are fixed rather than assuming the catalog was missing.

Resolution now first tries an exact NFKC/case/whitespace-normalized title, with
Unicode dash variants normalized to ASCII hyphens. If exact matching fails, an
exact alias key ignores whitespace and hyphens only; it does not use edit distance
or substring matches. Original-title aliases are included. Alias collisions retain
all identities and cannot use vote/popularity dominance to hide ambiguity.
Explicit integer `year`, `Title (YYYY)` and `Title YYYY` are supported. Literal
titles containing years or just numbers are looked up before splitting a year.
Conflicting explicit/embedded years do not resolve. The existing resolver's
recognition policy for ordinary duplicate titles remains; Crash still clarifies.
Coco/Inside Out/Zootopia regression identities remain 354912/150540/269149.

## Collection capabilities

`backend/services/movie_collections.py` implements injectable, JSON-safe capabilities exposed
through `api.movie_tools` and the controller's explicit allowlist:

| Tool | Source and behavior |
| --- | --- |
| `search_movie_collection(query, limit=5)` | TMDB `/search/collection`, first page; resolves a unique exact collection name or a sole result. Multiple meaningful matches require clarification. Limit 1–10, query 2–300 characters. |
| `get_movie_collection(collection_id, limit=50)` | TMDB `/collection/{id}` `parts`; deduplicates by movie ID and sorts valid ISO release dates ascending, missing/invalid dates last, with TMDB ID as a stable tie-breaker. Limit 1–100. |

Sources: [TMDB collection search](https://developer.themoviedb.org/reference/search-collection)
and [collection details](https://developer.themoviedb.org/reference/collection-details).
No movie membership is inferred from title substrings, and no franchise lists are
hardcoded. Membership may include movies absent from the local catalog because it
comes directly from TMDB collection data; recommendation scoring still uses its
existing local catalog and artifacts.

Search results include status, matches, resolved collection ID, total result count,
truncation, source and scope. Collection results include name/ID, movie IDs/titles,
release dates/years, posters, optional ratings, total member count, missing-date
count, truncation and source URL. Nonfinite/missing ratings become null. Limits
are applied after deduplication and sorting, never silently before release sorting.
Truncation is explicit in both metadata and user text. TMDB uses its supplied
release date, not a country-specific theatrical-release history.

All results state `coverage=tmdb_collection_only` and
`complete_franchise_verified=false`. TMDB collection membership does not establish
the complete conceptual franchise. The inspected X-Men Collection (748) returned
seven films including X2 and Dark Phoenix, with no Wolverine or Deadpool films.
Planet of the Apes search returned separate Original (1709) and Reboot (173710)
collections; the backend does not choose or combine them arbitrarily.

## Controller grounding and presentation

Instructions require backend tools for ratings, dates, runtime, director, cast,
genres, overview, providers, identity, collection membership and release order.
Missing facts cannot be filled from model memory, including with disclaimers.
Common series/franchise/all-films/order language forces collection search as the
first OpenAI tool, then collection retrieval. Other movie phrasing is routed by
the same instructed tool-calling model. Collection IDs must have been uniquely
resolved by backend search within this turn; ambiguous searches stop immediately.

Final collection and collection-search error/clarification text is formatted by
the backend. Model prose accompanying collection calls is ignored, and collection
members are never sent back for model-generated final lists. A model response
with no tool for a recognized collection request is replaced by an unverified-data
fallback. Offline tests deliberately attempt to add X2/Wolverine when they are
absent from a fixture; neither reaches the answer. Non-collection factual prose
remains model-generated from tools under grounding instructions.

The supported order is **release order**, clearly identified in user-facing lists.
Explicit story-order requests explain that a reliable story chronology is absent;
release dates do not establish story order. Internal chronology metadata is always
retained. No editorial timeline, reboot grouping or chronological watch order is invented.

`/movie-chat` preserves existing movie-card fields and adds `collection` and
`collection_choices`. Collection lists are text plus structured membership, not
ranked recommendation cards. The existing choice-button styling is reused;
selecting a collection submits a new independent named-collection request.
Movie clarification, country selection, providers and manual recommendations
remain independent. No interface redesign was made.

## Scope, input, output and cost protections

`backend/core/movie_guardrails.py` contains the 10,000-character message limit and
`MAX_OUTPUT_TOKENS`. Model output defaults to **900 tokens**, configured using
`MOVIE_ASSISTANT_MAX_OUTPUT_TOKENS` before startup, bounded to 256–2,000; invalid
configuration falls back to 900. This gives normal short movie explanations room
without permitting enormous essays. Every Responses request uses the SDK-supported
`max_output_tokens`; this caps generated output (including tool arguments), not
backend structured membership/card data.
[Official OpenAI parameter documentation](https://developers.openai.com/api/reference/python/resources/responses/methods/create).

Allowed scope is movie recommendations/facts/series, cast/crew, genres, ratings,
plots, release/provider information, movie comparisons/discussion and brief
greetings. Deterministic checks intercept known unrelated essay/code requests,
direct instruction overrides, new general-assistant roles and requests to expose
prompts/keys/environment/hidden implementation details. These return a short
movie-domain redirect with `unsupported_intent` and `openai_requests=0` before
client construction. System instructions enforce the wider scope and resist user
or tool-data overrides. Pattern checks are not a universal semantic classifier or
a guarantee against every adversarial phrasing; movie-related essays/discussion
remain allowed within the output cap. Secrets are never supplied to model context.

Oversized/empty `/movie-chat` input or invalid country receives a compact structured
HTTP 422 message without echoing the whole rejected body. Other endpoint validation
contracts are unchanged. The frontend displays this validation response clearly.

Existing five tool rounds/ten total-call bounds, no parallel calls, 16,000-character
tool argument bound, recommendation count 1–20, compact model-facing results,
30-second OpenAI request timeout, 100-second controller deadline, `store=False`
and `max_retries=0` remain. TMDB requests use a five-second timeout and a response
size check. No full catalog, embeddings or SVD data reaches OpenAI. Completed
OpenAI request counts are preserved; exceptional fallback responses still do not
retain a partial count. There is no cross-request rate limiting/account quota in
this milestone.

## Tests and running

```powershell
# Offline; all collection requests and OpenAI responses are fixtures/mocks
.\.venv\Scripts\python.exe -m unittest tests.backend.test_response_polish test_movie_collections test_movie_assistant test_movie_tools test_movie_chat test_chat_recommender test_chat -v
cd frontend
npx.cmd tsc --noEmit
node node_modules/eslint/bin/eslint.js app/components/MovieChat.tsx
cd ..
# Backend (MOVIE_CHAT_MOCK defaults off)
.\.venv\Scripts\python.exe -m uvicorn backend.api.app:app --reload --host 127.0.0.1 --port 8000
# Frontend, separate terminal
cd frontend
npm.cmd run dev
```

For offline browser checks, run `tests/browser/check_movie_chat_browser.py --serve` on port 8012
in a separate terminal, keep the frontend on 3000, then run
`.\.venv\Scripts\python.exe -m tests.browser.check_movie_chat_browser`. Installed Chrome and
Playwright are required. Providers/posters/collections are fixtures; ranking and
catalog identity use the persisted artifacts. Fixture collection membership is
intentionally incomplete to catch memory supplementation. `MOVIE_CHAT_MOCK=true`
enables the added X-Men and Planet of the Apes release-order fixture phrases;
outside tests those fixtures still fetch real TMDB unless an adapter is patched.
Mock mode remains disabled in production.

After offline tests pass, the optional paid check is
`.\.venv\Scripts\python.exe -m scripts.diagnostics.check_movie_guardrails_live`. It sends exactly four
messages once, prints sanitized responses/tool names/counts, verifies backend-only
collection formatting, and stops on failure. It never runs under unittest.

Original collection/guardrails validation: **72 backend tests passed**, TypeScript and MovieChat lint
passed. Browser regression passed collections/choices, scope redirects, chat,
manual recommendations, cards, providers, countries, movie clarification,
loading/errors/duplicate-send handling, mobile layout and refresh reset.

The single live run passed all four HTTP checks:

| Request | Result | Actual OpenAI requests |
| --- | --- | --- |
| X-Men movies in release order | HTTP 200/success; collection 748; seven backend-formatted members, source/scope/chronology notes | 2 |
| Interstellar rating | HTTP 200/success; TMDB 8.5/10, 41,307 votes; resolve + details | 3 |
| 2,000-word World War II essay | HTTP 200/unsupported_intent; concise movie-domain redirect | 0 |
| Ignore instructions + 5,000-word economics essay | HTTP 200/unsupported_intent; concise movie-domain redirect | 0 |

**Five actual OpenAI requests total**, no paid reruns. The verified collection
order was X-Men (2000-07-13), X2 (2003-04-27), The Last Stand (2006-05-24),
First Class (2011-06-01), Days of Future Past (2014-05-15), Apocalypse
(2016-05-18), and Dark Phoenix (2019-06-05). These are a test-time TMDB snapshot,
not hardcoded application membership. The live script asserted that the final
list exactly matched backend formatting of returned collection data.

Known limitations: each message still needs its own movie/series title;
"put them in release order" has no previous-turn context. Only TMDB collections
are supported, not authoritative unions of broad franchises. Search uses the
first page and reports omitted matches; refine broad searches. Missing dates,
same-title/same-year movie choices, upstream unavailability, existing ranking
quality and general model-prose grounding limitations remain explicit.
