# Conversational Movie AI policy

Temporary session continuity is now supported when `/movie-chat` receives an
opaque `session_id`; legacy requests without it remain stateless. See
[MOVIE_SESSION_MEMORY.md](MOVIE_SESSION_MEMORY.md) for bounded state, references,
preference updates, privacy, and process-local limitations. Earlier statements
below that no memory exists describe historical milestones.

The main controller prompt, `movie_assistant.PROMPT` in `movie_assistant.py`,
uses RISEN as an organizational structure:

- **ROLE:** a general conversational assistant with specialized movie capabilities.
- **INSTRUCTIONS:** general discussion, engine recommendations, authoritative movie
  data, providers, collections, and grounded research rules.
- **STEPS:** action selection with personalized recommendations taking precedence
  over research; appropriate tools or general knowledge, followed by a natural answer.
- **END GOAL:** one coherent conversational experience with authoritative cards and
  preserved research sources/citations.
- **NARROWING:** grounding, watch-order labels, requested scope, untrusted content,
  secrets, long-form restrictions, and existing cost limits.

This is prompt organization only, with no new runtime component, dependency,
routing layer, or AI call. Deterministic protections remain in `movie_guardrails.py`
and the existing controller. Tool schemas, research implementation, recommendation
engine, and limits remain unchanged by the prompt-organization refactor. The later
session milestone adds temporary context within these same five sections.
The structure regression checks section order and nonempty content without
depending on paragraph wording or indentation. Refactor validation: all 131
relevant offline backend tests passed; no live OpenAI requests were made.

Current completeness/freshness extension: [MOVIE_RESEARCH.md](MOVIE_RESEARCH.md).
These franchise requests now require bounded cited research; ordinary movie
discussion and narrower lists retain general knowledge. Earlier live reports below
describe their historical milestones, not guarantees of current research coverage.

Watch-order consistency: release order means earliest actual release to latest,
including date order within a year when known. Story chronology means narrative
timeline; recommended watch order is editorial. These labels are not equivalent.
Unspecified "in order" defaults to clearly labelled release order. Answer the
requested trilogy/films first; offer related films briefly rather than silently
expand scope or dump a larger universe list. The same date/label consistency
applies to requested broader lists. General lists remain model-generated, while
verified collections retain unchanged backend date sorting. No franchise lists
are hardcoded in production. The optional `check_watch_order_live.py` checks one
LOTR/Hobbit release-order response after offline regressions pass.
Validation for this consistency milestone: all 109 relevant backend tests passed.
One live request returned HTTP 200/success, no tools, and the requested six films
in release sequence 2001, 2002, 2003, 2012, 2013, 2014. Exactly one OpenAI request
was used, with no paid reruns. Mocked tests verify controller instructions and
routes; they do not prove accuracy of future model generations. There is no
deterministic verifier of release years in general-knowledge prose, so such
answers can still contain model errors. No frontend, guardrail thresholds,
recommendation code, ranking artifacts, or memory behavior changed.

This policy supersedes earlier movie-only scope and assistant chronology limits.
Explicit collection wording selects the verified TMDB collection path. Broader
franchise/universe/series lists, completeness language (all/every/entire/whole/
complete movies) and lists including related films use general movie knowledge.
They cannot silently be narrowed to one database collection. Unspecified order
defaults to release order; story chronology remains interpretive. Personalized
recommendations take precedence and keep engine routing. No franchise membership
is hardcoded, and TMDB collection membership is unchanged. Deterministic signals
are generic; uncommon semantic phrasing remains covered by prompt instructions.
Broad routes also receive a final per-request instruction to supply the list
directly, rather than promise a future collection lookup while tools are disabled.
Tool definitions are omitted only for this informational route; all existing
tools remain available for recommendations, explicit collections and facts.

Franchise milestone validation: 101 backend regression tests passed. Full
unittest discovery additionally imports the existing standalone
`test_project_hail_mary.py` diagnostic, which fails with `KeyError: tmdb_id`;
that scoring diagnostic was not changed. No frontend contract changes or browser
rerun were needed. Two live messages used three OpenAI requests: explicit X-Men
collection succeeded with search/details tools and seven verified films; the broad
request used no tools but initially returned a lookup promise instead of a list.
The final route-specific instruction/tool omission addresses that failure offline;
its generated list quality has not been retested live, honoring no paid reruns.
Normal conversation, general explanations, short writing, coding and movie
discussion may use model knowledge. Story chronology may use a commonly used,
interpretive watch order; it must not be attributed to TMDB. Pure story-order
requests disable collection tools. Those tools still provide verified TMDB
membership and release order, with existing coverage limitations.

Catalog searches, authoritative backend/TMDB facts, streaming availability and
verified collection release order retain their tools and validation.
Personalized recommendations must use `recommend_movies`, the existing
RecommendationEngine and unchanged structured cards. Common phrasings force
that named function at the API boundary; model-only prose and other tools are
rejected on that route. Other phrasings remain covered by system instructions.
Ambiguous titles still require clarification. Required-only constraints still
need preferences, with no new ranking fallback.
The controller restores a uniquely stated parenthesized release year when the
model drops it from an extracted favorite title; it never infers a year or ID.

The pre-API guard detects generation intent plus explicit requested output of
**1,000 or more words / 3 or more pages**, including comma-separated counts and
hyphenated units. Explicit multi-page/long-form generation and attempts to remove
output limits are also intercepted. It offers concise help or outlines without
a movie redirect. Numeric questions, short requests and summaries or outlines
of long source documents remain allowed. Subject-based restrictions and blanket
`ignore`/role-change checks were removed; secret protections remain.

The output cap remains **900 tokens by default**, with the existing bounded
environment configuration unchanged. Input remains 10,000 characters; tool rounds
remain five. Existing result bounds, timeouts, no automatic retries, no parallel
calls and request accounting remain. No full catalog, embeddings or SVD factors
are sent to OpenAI. No memory or persistent history was added.

Offline checks: run all eight suites including `test_conversation_policy`.
The optional `check_conversation_policy_live.py` sends four messages once after
offline checks pass. It uses Dune (2020), explicitly selected by the user.
Unqualified Dune has a safe ambiguity regression.

Limitations: deterministic routing/detection covers common English phrases;
prompt policy covers other paraphrases. General knowledge can be inaccurate;
story timelines can conflict. Requests below the word threshold can still exceed
the token cap. Existing provider-data and same-title/same-year limitations remain.
Follow-up references require full context because the assistant is stateless.

Validation: all 94 offline backend tests and browser regressions passed. The
2,000-word essay returned HTTP 200/unsupported_intent with zero OpenAI requests.
The first live run used four OpenAI
requests: machine learning and X-Men chronology answered without tools; the
recommendation called the correct tool but dropped the Dune (2020) qualifier,
causing safe clarification and no cards. The controller fix is covered offline
with real-engine cards; no paid rerun was made. The chronology answer had weak
timeline caveats, so the prompt now explicitly requires an interpretive opening.
