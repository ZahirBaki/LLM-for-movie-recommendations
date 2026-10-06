# Backend chat parser

`POST /chat` converts one message into validated preferences. It does not call
RecommendationEngine, recommend movies, resolve TMDB IDs, or save conversation
history. `api.py` keeps ownership of the existing `load_dotenv()` configuration.
Never commit `.env` or print its contents.

The parser uses the official OpenAI Python SDK's Responses API
`responses.parse(text_format=MovieIntent)` and reads `output_parsed`.
`OPENAI_MODEL` in `chat_service.py` is the single model setting: `gpt-4.1-mini`,
a low-cost model supporting structured outputs.

## Environment and commands (PowerShell, project root)

The project previously had no Python dependency manifest. `requirements-chat.txt`
is additive to the existing FastAPI/Pydantic v2 environment, not a replacement
for its recommender dependencies. A `.venv` was created with access to the
existing Python 3.14 packages, without upgrading those packages.

```powershell
py -3.14 -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-chat.txt
.\.venv\Scripts\python.exe -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

The first two commands are only needed to recreate/install the environment.
`OPENAI_API_KEY` comes from the existing ignored `.env`. Missing keys do not
prevent the rest of the API from starting.

## Testing through /docs

Open http://127.0.0.1:8000/docs, expand `POST /chat`, select **Try it out**,
enter this request, and click **Execute**:

```json
{"message": "Recommend something like Interstellar with Ryan Gosling from after 2015"}
```

Expected response (illustrative until verified live):

```json
{
  "message": "Recommend something like Interstellar with Ryan Gosling from after 2015",
  "parsed": {
    "intent": "recommend",
    "favorite_movies": ["Interstellar"],
    "movie_titles": [],
    "favorite_genres": [],
    "favorite_actors": [],
    "favorite_directors": [],
    "required_genres": [],
    "required_actors": ["Ryan Gosling"],
    "required_directors": [],
    "exclude_genres": [],
    "exclude_actors": [],
    "exclude_directors": [],
    "min_year": 2016,
    "max_year": null,
    "country_code": null,
    "top_n": null,
    "natural_language_preferences": []
  }
}
```

## Schema and interpretation

- `intent`: `recommend`, `search`, `information`, or `unknown` (default).
- `favorite_movies`: unresolved reference titles (`list[str]`). The LLM does not
  generate canonical IDs; future catalog resolution must supply TMDB IDs.
- `movie_titles`: subjects of search/information requests (`list[str]`).
- `favorite_genres`, `favorite_actors`, `favorite_directors`: soft preferences.
- `required_genres`, `required_actors`, `required_directors`: requested constraints.
- `exclude_genres`, `exclude_actors`, `exclude_directors`: explicit exclusions.
- `min_year`, `max_year`: nullable integers, inclusive bounds, valid ordered range.
- `country_code`: nullable uppercase two-letter streaming region. The region is
  supported by the existing API, not a recommender scoring parameter.
- `top_n`: nullable positive integer; unspecified count stays null.
- `natural_language_preferences`: unsupported nuances preserved for a future stage.

All lists default to `[]`; optional scalars default to `None`. Unknown extra fields
are rejected. Genres use TMDB terminology, such as `Science Fiction`.
Explicit requests such as "with Ryan Gosling" use `required_actors`; general likes
use `favorite_actors`. Reference titles do not imply additional cast/genre wishes.

## Error contract

Errors contain a deliberately sanitized `detail` string. Blank/whitespace messages
and invalid request bodies return 422. Messages are limited to 10,000 characters.
Missing keys, authentication failures, exhausted credit/quota, and connection
failures return 503; ordinary rate limits return 429; upstream API errors,
refusals/incomplete output and invalid parsed preferences return 502; timeouts
return 504; unexpected errors return 500. Requests time out after 30 seconds and
SDK retries are disabled. Raw upstream exceptions are never returned or logged.

## Validation commands

```powershell
.\.venv\Scripts\python.exe -m py_compile api.py chat_service.py test_chat.py check_chat_live.py
.\.venv\Scripts\python.exe -m unittest test_chat -v
.\.venv\Scripts\python.exe check_chat_live.py
```

Offline tests exercise the actual SDK with a mocked HTTP transport, schema
validation, request validation, errors, existing read endpoints and the absence
of recommender calls. They do not establish live semantic extraction quality.
The optional live checker includes all seven milestone prompts and stops on the
first unavailable API response to avoid repeated failed or billed requests.
Quota/billing failures require API credit; they are separate from code validation.

Validation in this workspace: syntax/import checks passed; all 10 offline tests
passed; Uvicorn completed startup and served `/` and `/docs` successfully. All
pre-existing API functions and request models were verified unchanged. Installed
OpenAI SDK version: 2.54.0 (with its missing transitive dependencies in `.venv`).
The permitted live attempt reached OpenAI but returned HTTP 429, classified as
a rate limit rather than a recognized billing/quota code. No live parsed result
was obtained, so semantic correctness of the seven prompts remains unverified.
No recommendation algorithms, catalog/model artifacts, or frontend files were
changed for this milestone.
