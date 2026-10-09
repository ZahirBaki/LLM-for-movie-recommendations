# Bounded movie research

## Direct diagnostic follow-up

The explicit `scripts/diagnostics/diagnose_movie_research.py` script loads the same project dotenv
configuration before importing model settings, creates the same AsyncOpenAI client
(30-second timeout, zero retries, 100-second total deadline), and calls the provider
directly once. It never imports the API, controller or recommendation engine.
It refuses execution when production disables development diagnostics.

Local details require `MOVIE_RESEARCH_DIAGNOSTICS=true` and non-production
ENVIRONMENT/APP_ENV/NODE_ENV. Diagnostic logs include exception class, numeric
HTTP error status, scalar error type/code, sanitized message and request ID; response
metadata includes status/incomplete reason, output types/count, search status,
text presence/length and citation counts/spans. No web text, prompts, queries,
request bodies, headers, cookies, credentials or environment dumps are logged.
Known configured keys and credential patterns are redacted. Production browser
responses remain sanitized with no diagnostic fields.

One direct request was run, unchanged:
"Verify whether The Lord of the Rings: The War of the Rohirrim was released as a
theatrical Middle-earth film and give the release year."

It succeeded on gpt-4.1-mini: completed response, one completed web_search_call,
one message, 1,229 text characters, four URL-citation annotations, two parsed
sources. No exception or OpenAI error code occurred. HTTP error status was absent;
the diagnostic retained SDK result metadata rather than raw HTTP response headers.
The official request parameters and parser worked for this query. No model/tool
selection or parsing correction was justified; the prior full-franchise failure
was not reproduced and its discarded cause cannot be reconstructed from this run.
This confirms search access for the account now, not the cause of the earlier
unresolved answer or guaranteed completeness of broad research.

No final application-level live test was run: the user's authorization limited it
to a diagnostic-proven code/configuration fix, and none was identified. Total live
OpenAI requests in this diagnostic task: one, no retries. No account/billing change
is indicated by this success. To investigate the original broad case further, the
next step is an application-level reproduction with these diagnostics enabled;
that would require a separately authorized live request.

Six new offline tests cover the observed SDK object shape with neutralized text,
URLs and IDs; diagnostic gating; error metadata/redaction; and safe incomplete,
uncited and malformed results. Search remains non-live in automated tests.
All 130 relevant backend tests passed after this diagnostic change. Research
request construction, parsing, routing, recommendation behavior, frontend, limits
and model/catalog/embedding artifacts were not changed. No RISEN or memory work
was performed.

The installed OpenAI Python SDK is 2.54.0. `requirements-chat.txt` already permits
OpenAI 2.x; no provider/dependency was added. The existing Responses integration
uses `OPENAI_MODEL` (default `gpt-4.1-mini`), the existing local API credential,
`store=False`, no retries, 30-second request timeout and 100-second turn deadline.

[Official web-search documentation](https://developers.openai.com/api/docs/guides/tools-web-search)
lists `gpt-4.1-mini` support and specifies `web_search`, `search_context_size`,
mandatory tool choice and URL citation annotations. Installed SDK documentation
defines `max_tool_calls` as the total built-in-tool limit for a response.
Account/network access is established only by the optional live check, not by
the existence of SDK types. No alternate model/provider is silently substituted.

Milestone result: the sole live Middle-earth attempt returned HTTP 200/unresolved,
no sources, and the honest verification-unavailable message. One OpenAI request
was attempted; no paid rerun was made. The initial provider did not retain a
failure category, so this outcome cannot establish whether account/API access,
timeout, incomplete output or citation validation caused the failure. Actual
grounded franchise completeness, including The War of the Rohirrim, remains
unverified live. Safe request/incomplete/invalid-result diagnostic categories and
numeric HTTP status logging are now available for subsequent requests; raw
provider errors/results are never logged or exposed. No claim of successful
account activation is made. Current activation requirement is usable Responses
web_search access on the existing configured OpenAI account/project/model; if
the request fails, its safe diagnostic identifies the next configuration check.

`backend/integrations/movie_research.py` isolates a mockable ResearchProvider request/result contract.
OpenAIWebResearch uses one Responses request, only the official hosted
`web_search` tool, forced tool choice, low context and at most two hosted calls.
There are no application search/retry loops, scrapers, raw page downloads, or
handoffs to movie functions. Low context does not guarantee a particular number
of internal search results/queries. Output retains the default 900-token cap.
Up to five unique cited public HTTP(S) source URLs and thirty inline spans are
returned; missing search/citations, unsafe links, incomplete responses or excessive
sources fail closed with a concise verification-unavailable message. No model-memory
answer is substituted or described as researched. `MOVIE_RESEARCH_ENABLED=false`
disables research with the same honest message and zero requests.

Routing priority: long-form/secret guards, personalized recommendations, backend
facts/providers/explicit collections, then research for movie/franchise information
with all/every/complete/entire/whole/full/including or freshness signals. Universe/
franchise list requests imply broader completeness even without "all". Plain
trilogy lists, discussion and general questions retain normal knowledge answers.
Common personal recommendations remain forced through recommend_movies and the
existing engine. No research is used to pick personalized cards. Stateless requests
still need explicit franchise names/context; "this franchise" alone has no memory.

Source policy prefers official studio/franchise/movie pages, reputable databases
and established film publications, with multiple sources when practical. Membership
disagreements and coverage uncertainty must be stated; search does not prove
absolute completeness. Released theatrical scope includes relevant animation.
Upcoming/announced titles stay separate; rumors are not confirmed releases.

Web data is untrusted: the provider instructs the model to ignore page instructions,
secrets requests, ranking bypasses and policy overrides. Only hosted search is
available; no environment contents, backend tools, catalog or model artifacts are
sent. Raw search actions/results and hidden reasoning are not exposed. This limits
execution privileges but cannot prove the model is immune to misleading web text.

Research prose follows the watch-order/scope policy. Numbered title (year/date)
release-list runs are conservatively sorted in the backend and citation offsets
remapped. Exact dates sort within a year only when all rows in that year have dates.
Unknown same-year order is preserved. Upcoming/story/editorial sections and
cross-line citations are not reordered. Other prose is not parsed into a fake
database; supplied years/membership remain model-extracted facts, not independently
verified by this sorter. TMDB collection sorting/data remains unchanged.

`/movie-chat` adds optional-use `sources` and `citations` arrays (empty for other
responses). Citation offsets refer to Unicode characters in assistant_message.
The frontend renders escaped text with clickable inline citations and compact
source links; no HTML/markdown execution or redesign. Unavailable research uses
existing `unresolved` status. Research request accounting includes a failed attempt
as one request; normal controller accounting remains unchanged.

Offline tests mock all search responses and network calls. The optional
`scripts/diagnostics/check_movie_research_live.py` runs the single user-authorized Middle-earth query
once after the full relevant suite passes. If unavailable, activation needs working
OpenAI Responses web_search access for the configured model/account/project,
billing/quota and network access; no third-party dependency/provider is required.

No memory, accounts, history, scoring changes, retraining, catalog or embedding
rebuilds, franchise tables or fake mega-collections are introduced.

Validation: 124 relevant backend tests passed after the final diagnostic change (123
passed before the live attempt). TypeScript
passed, and Chrome regressions passed source links, inline citations, manual/chat
recommendations, collections, providers, clarification, errors and mobile layout.
The added failure-diagnostic regression passed offline afterward.
