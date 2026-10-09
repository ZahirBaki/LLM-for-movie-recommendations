# Session conversation memory V1

`/movie-chat` accepts an optional `session_id`. The chat component creates one
cryptographically random UUID v4 on its first send and reuses it until that
component is remounted or the page is refreshed. `crypto.randomUUID()` is preferred;
the fallback uses `crypto.getRandomValues()` and UUID v4 bits. There is no cookie,
localStorage, account, fingerprint, or cross-browser identity. The backend accepts
only UUID v4 strings of 36 characters, normalizes case, and rejects malformed IDs
with HTTP 422. IDs never appear in the normal UI or model-facing context.
Requests without an ID remain backward-compatible and stateless.

## Store and lifecycle

`backend/sessions/movie_session.py` owns `MovieSessionState`, `SessionStore`, reference resolution,
preference edits, and integration helpers. Defaults are named constants:

- `SESSION_TTL_SECONDS`: 1,800 seconds / 30 minutes of inactivity.
- `MAX_SESSIONS`: 500 sessions per backend process.
- `MAX_RECENT_MESSAGES`: 8 messages, each at most 500 characters.
- `MAX_MOVIE_REFERENCES`: 20 entries in the most recent relevant movie list.
- `MAX_TRACKED_MOVIE_IDS`: 200 each for shown, seen, and excluded IDs.
- `MAX_CONTEXT_CHARS`: 6,000 characters of model-facing context.
- `MAX_CANDIDATES`: 20 recommendation candidates, matching the existing tool cap.
- `MAX_SESSION_DATA_CALLS`: 5 backend lookups for a deterministic comparison.

The store removes expired inactive entries on access, transaction completion, or
inspection. No background scheduler is required. Expired entries cannot be read
on the next request. At capacity, the least recently completed inactive session
is evicted. Active/queued entries are leased and cannot be evicted mid-turn; if
all slots are busy, the existing sanitized unavailable response is returned.
Per-session async locks serialize turns. A short thread lock protects the index;
state is deep-copied on checkout and commit so callers cannot mutate committed
objects. An exception rolls back the transaction. Separate sessions do not share
state or preferences. State is neither logged nor written to disk.

Restarting the backend clears all sessions. Memory is process-local: multiple
workers do not share it, so V1 assumes one backend worker. The opaque ID is a
temporary capability, not authenticated user identity.

## Stored state

Each state contains only:

- `recent_messages`: bounded user/assistant role and visible text excerpts.
- `current_movie`: title, year, and optional backend TMDB ID.
- `last_movie_list` and `last_list_kind`: recommendation, collection, research,
  or knowledge references, in the displayed order.
- `active_preferences`: the existing bounded `MovieIntent` representation.
- `shown_movie_ids`, `seen_movie_ids`, `excluded_movie_ids`: bounded ID lists.
- `country_code` and `last_intent`.

The store entry separately holds a monotonic last-update time, lease count, and
lock. It does not store tools' raw outputs, credentials, headers, hidden prompts,
web page contents, or chain-of-thought. Only visible answer excerpts are retained.
Common credential/sensitive markers are omitted from text storage; this lightweight
filter is not a general-purpose personal-data detector. Avoid entering sensitive
information into the chat.

## References and continuity

First/second/third through tenth, `number N`, and `last one` resolve against the
latest relevant list. `it`, `its`, `that movie`, and `that one` require a clear
current movie or a one-movie list. Ambiguous or out-of-range references ask for
clarification. Successful factual/provider lookups establish the current movie;
an explicit new movie lookup replaces it. A multi-movie result clears singular
selection rather than picking a movie arbitrarily.
New general discussion clears stale singular movie selection; its pronouns remain
with the conversational controller rather than being forced into movie lookups.

Backend recommendation and collection results supply exact IDs. Numbered
title/year entries in research or general answers may be retained with **no ID**.
Before a factual/provider lookup, those references go through the existing title
resolver with the stored year. Ambiguity is preserved; no IDs are inferred from
prose. Unrecognized prose formats are not converted into a list.

Clear factual/provider references use existing tools directly without an extra
LLM call. Highest-rated, newest, and shortest comparisons use actual details,
including TMDB ratings. Lookups stop at five calls; missing or unqueried entries
are explicitly excluded from the claim. Ties currently select the first matching
entry. A failed lookup does not fabricate values.

Explicit current-turn country takes precedence over remembered country, then the
frontend/default region. Common country names and uppercase two-letter codes are
handled directly; other names remain with the existing LLM extraction. Remembered
region may override a subsequent frontend selection until explicitly changed in
chat, as specified by this precedence. No IP/location inference is used.

## Recommendation context

Memory determines current preferences; the existing RecommendationEngine still
determines the candidates and ranking. `Give me 5 more` preserves `MovieIntent`
and filters shown/seen/excluded IDs after one engine call. The adapter requests
`min(20, requested_count + blocked_count)` candidates, preserves their relative
order, and takes up to the requested count. There is no scoring change, huge
candidate request, or retry loop. Fewer available matches are explained rather
than filled with model-generated films. This bound means an unseen lower-ranked
film may exist beyond the candidate window.

Simple year edits preserve other fields and replace the opposite year bound.
`after 2020` means 2021 onward; `before 2010` means through 2009. Simple genre
replacement clears the old required/favorite genre; exclusion removes that genre
from positive fields, and re-inclusion clears its exclusion. An existing expressed
favorite-genre seed can be updated, while required-only requests retain the
existing needs-preferences behavior. Other natural preference refinements use the
existing controller with compact context: nonempty extracted fields replace the
corresponding prior fields, while unspecified fields remain unchanged. Complex
clearing/replacement paraphrases still depend on model extraction.

Seen/exclusion commands resolve references before updating state. `I've seen the
first two` adds those exact IDs to seen and excluded lists. `Don't recommend Dune`
uses a unique matching stored reference, otherwise the existing resolver and
clarification. Explicit new recommendation requests reset active preferences and
shown IDs; seen/excluded IDs remain for the chat. General questions are not forced
into a recommendation route. Old IDs eventually leave the 200-ID bounded window.

## Model context and guardrails

The RISEN sections remain intact. Compact JSON includes movie references,
preferences, country, recent text, and abbreviated exclusion/shown lists. It is
sent as a separate **untrusted user-data item**, never interpolated into system
instructions. Size reduction removes old message excerpts and list tails, then
oversized preferences if necessary; omissions are labelled. Full structured
preferences and exclusion filtering remain backend-owned. Only backend-resolved
stored references seed the controller's allowed movie-ID set.

The 10,000-character current-message limit, long-form/secret interception,
900-token default model output cap, existing tool rounds, and research bounds
remain. Guards run before deterministic memory edits. Tool schemas and research
architecture are unchanged. Responses expose no session state or debug endpoint.
No additional parser or summarization AI calls are added. The frontend changes
only session ID submission and its continuity description; cards and citations
retain their existing rendering.

## Validation and next steps

Run the existing relevant backend suites plus `test_movie_session`, and the
offline `tests/browser/check_movie_chat_browser.py` fixture server/browser check. Session tests
cover creation/isolation, expiry/eviction, transaction serialization/rollback,
exact and ambiguous references, preferences, seen/more, regions, list data,
privacy/context bounds, real-engine integration, API validation and legacy use.
Browser checks cover a stable hidden ID, refresh reset, follow-ups, cards, and
sources. No paid live calls are required.

Milestone validation: all **173 relevant offline backend tests passed** (131
existing plus 42 session tests). `npx.cmd tsc --noEmit` passed. The full offline
Chrome browser regression passed, including session continuity, cards, sources,
error handling, mobile layout, and a fresh ID after refresh. Live OpenAI requests:
**zero**. Recommendation/scoring/model/catalog/embedding artifacts and the research
provider implementation were not changed.

V2 could add an explicit new-chat control, better typed preference patches and
confirmation for complex edits, clearer pending-clarification handling, cached
bounded list metadata for comparisons, and tests for more reference paraphrases.
Persistent profiles/accounts and shared-worker storage are separate decisions;
they are not implemented in V1.

## Pending recommendation clarifications

`pending_task` retains one bounded recommendation intent (including count, filters,
and country), backend-resolved seed IDs, up to 20 ambiguous references with at
most 20 candidates each, and whether the original seeds were watched. It shares
the session's 30-minute inactivity TTL, eviction and transaction isolation. It
stores no external metadata snapshots or persistent profile.

Title/year, unique title, release year, displayed ordinal and `both` (exactly two
candidates) replies resume the saved task without an OpenAI call. Invalid or still
ambiguous replies keep the task and show its candidate choices. Multiple ambiguous
seeds resolve in sequence; scoring waits until all identities are resolved. A
verified title/year correction can use the existing bounded TMDB lookup. Local
IDs bypass repeated duplicate-name resolution only after matching catalog identity.
External movies use the compatible runtime query component when available;
metadata/model failures produce an explicit explanation and no fabricated cards. Transient scoring errors retain the pending task for retry.

Clear new questions, recommendation requests and cancellation override the task.
Recommendation clarification buttons submit the selected qualified title so an
invalid reply or a second ambiguity cannot replace the original task with a movie
information request. Information clarification buttons retain their existing flow.
Successful completion clears the task; existing seen, excluded, provider and more
recommendation behavior continues unchanged.


External favorites retain their backend-resolved identities in `active_seed_movies`
(up to 20 entries). Embeddings and full external metadata stay in the separate
bounded runtime cache, never in session state. More-result requests reuse those
identities and cache entries; new recommendation contexts clear active seeds.
See `MOVIE_EXTERNAL_SEEDS.md` for compatibility checks and content-only ranking.


## Combined list updates and recommendation continuation

A complete continuation command is parsed before changing state. Supported forms
include seen/watched references to the first N entries or numbered entries, or
exclusion of an ordinal such as the last movie, followed by a request for more or
other movies. A period/semicolon or `and` separates instructions. A requested
count and after/before/since/from (or newer/older than) year bound can be applied
alongside the exclusion. Existing seeds, filters and country remain active.

References are resolved as a group before any are marked seen/excluded. Invalid
numbers cannot partly change memory. Updates precede the existing engine call;
shown/seen/excluded filtering and the latest list update remain authoritative.
Response title lists come directly from the structured engine results. If the
bounded candidate pool cannot supply the count, only valid cards are returned
with a shortage explanation. Unsupported combined phrasing asks for the complete
command rather than claiming only the exclusion was completed.

Exact local titles with strong recognition evidence use the existing recognition
threshold (at least 1,000 votes, positive popularity). A duplicate must also meet
the existing vote/popularity dominance rules. Those confident matches do not get
reopened by live search. Low-recognition matches, local misses and explicit years
continue through the existing resolution policy. No titles are hardcoded and no
recommendation scoring, embeddings or ranking weights change.
