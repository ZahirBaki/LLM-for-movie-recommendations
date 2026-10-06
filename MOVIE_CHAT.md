# Movie AI chat (milestone 3)

`POST /movie-chat` accepts `{message, country_code}`. It reuses
`parse_movie_message` and `ChatRecommender.recommend_from_intent`, with the selected
website region overriding the parsed region for provider enrichment. Blocking
recommendation/provider work runs in FastAPI's thread pool. Assistant messages
come from deterministic templates, without a second OpenAI call.

Responses consistently contain `status`, `assistant_message`, `parsed_intent`,
`resolved_movies`, `recommendations`, `clarifications`, `unresolved_movies`,
`unsupported_preferences`, `country_code`, `message`, and `mock_mode`.
Normal outcomes include success (also an empty-match success), needs clarification,
unresolved, needs preferences, and unsupported intent. Upstream/execution failures
return HTTP 503 with an error response and a generic message. Invalid request
bodies return HTTP 422. Neither keys nor internal/upstream exception text is shown.

## Start in development mock mode

PowerShell from the project root:

```powershell
$env:ENVIRONMENT = "development"
$env:MOVIE_CHAT_MOCK = "true"
.\.venv\Scripts\python.exe -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
cd frontend
npm.cmd run dev
```

Open http://localhost:3000 and scroll to **Movie AI**. The chat uses the existing
streaming-country selector and leaves the manual form/results independent.
`npm.cmd` avoids the local PowerShell script execution-policy restriction.
No `.env` changes or API key changes are needed.

Mock parsing is disabled by default, and blocked when `ENVIRONMENT`, `APP_ENV`, or
`NODE_ENV` is `production`. Set `ENVIRONMENT=production` for a production backend.
The production frontend's build mode alone does not configure a separate backend.
Unset `MOVIE_CHAT_MOCK` or set it to `false` and restart to use real parsing later.
Real parsing can incur OpenAI API usage; it was not activated for this milestone.

Mock mode never calls OpenAI and never falls back to heuristic parsing. It
case-folds and normalizes whitespace, then looks up an exact predefined phrase:

- `Something like Interstellar`
- `Something like Coco, Inside Out and Zootopia`
- `Something like Interstellar with Ryan Gosling after 2015`
- `Something like Crash`
- `Something like Crash (1996)`
- `Something like Crash (2005)`
- `Something like Nonexistent Movie xyz987654`
- `Recommend a comedy` (existing engine's needs-preferences case)
- `Who directed Interstellar?` (unsupported intent)

Other phrases receive a clear development mock-mode message. Recommendations are
real engine results, not mock rankings. Provider enrichment uses the existing
TMDB flow and can still require TMDB network access; only browser/backend tests
substitute provider fixtures to remain fully offline.

## Frontend and clarification

`MovieChat.tsx` stores user/assistant messages, card results, and clarification
choices in page memory. It sends only the current message and selected country.
It supports Enter, Shift+Enter, disabled controls/loading, a synchronous duplicate
send guard, a 120-second request deadline, and safe malformed-response handling.
Refresh resets messages. `MovieCard.tsx` and `movie-types.ts` were extracted from
the manual results and are shared by both flows, preserving scores, streaming
logos, rent/buy display, watch links, and JustWatch attribution.

Clicking a Crash option submits a new independent phrase such as
`Something like Crash (1996)`. No previous constraints or conversational context
are carried over. Resolution policy and TMDB IDs remain as in milestone 2.
Options with no release year cannot be selected by this first version, and
same-title/same-year collisions still require future ID-based selection support.
Unsupported nuances are preserved by the backend but not semantically reranked.

## Checks

```powershell
.\.venv\Scripts\python.exe -m unittest test_movie_chat test_chat_recommender test_chat -v
cd frontend
node node_modules/typescript/bin/tsc --noEmit
node node_modules/eslint/bin/eslint.js app/components
```

Optional offline browser harness, with Next.js already running on port 3000:

```powershell
# Terminal one, project root: test-only server with fixture provider data
.\.venv\Scripts\python.exe check_movie_chat_browser.py --serve
# Terminal two, project root
.\.venv\Scripts\python.exe check_movie_chat_browser.py
```

This browser harness requires Chrome and `python -m pip install playwright`
in the test environment. Playwright 1.63.0 was installed into `.venv` for these
checks; it is not a runtime dependency. Browser API requests are routed to the
test server on port 8012, leaving existing application servers alone. Provider
and poster assets are offline fixtures, while catalog resolution/ranking are real.

Validation: all 29 backend tests passed. Browser checks passed for all five main
mock phrases, both Crash choices (backend), clickable Crash selection (browser),
nonexistent titles, needs preferences, unsupported/mock-unavailable phrases,
malformed/network errors, duplicate-send protection, multiline input, mobile
width, and refresh reset. Existing movie/actor/director search, genre selection,
country selection/outside click, manual recommendation results, and provider/link
UI passed browser checks. TypeScript passed and component lint reported only two
existing-style `<img>` optimization warnings after card extraction.

No scoring, model, embeddings, catalog generation, conversation memory, accounts,
database, persistent history, or localStorage chat persistence was added/changed.
