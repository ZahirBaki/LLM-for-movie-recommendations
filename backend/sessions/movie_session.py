"""Bounded process-local continuity; no persistence, ranking or extra AI calls."""
import asyncio
import json
import re
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from backend.services.chat_service import MovieIntent
from backend.services.chat_recommender import MovieClarification, ResolvedMovie
from backend.core.movie_guardrails import is_recommendation_request, has_current_movie_preferences, scope_redirect

SESSION_TTL_SECONDS = 30 * 60
MAX_SESSIONS = 500
MAX_RECENT_MESSAGES = 8
MAX_MESSAGE_CHARS = 500
MAX_MOVIE_REFERENCES = 20
MAX_TRACKED_MOVIE_IDS = 200
MAX_CONTEXT_CHARS = 6000
MAX_CANDIDATES = 20  # Same bound as the existing recommendation tool.
MAX_SESSION_DATA_CALLS = 5


class MovieReference(BaseModel):
    tmdb_id: int | None = Field(default=None, gt=0)
    title: str = Field(max_length=300)
    year: int | None = None

    def qualified_title(self):
        return f"{self.title} ({self.year})" if self.year else self.title


class RecentMessage(BaseModel):
    role: str
    content: str


class PendingTask(BaseModel):
    intent: MovieIntent
    resolved_movies: list[ResolvedMovie] = Field(default_factory=list, max_length=20)
    clarifications: list[MovieClarification] = Field(default_factory=list, max_length=20)
    watched: bool = False


class MovieSessionState(BaseModel):
    pending_task: PendingTask | None = None
    recent_messages: list[RecentMessage] = Field(default_factory=list)
    current_movie: MovieReference | None = None
    last_movie_list: list[MovieReference] = Field(default_factory=list)
    last_list_kind: str | None = None
    active_preferences: MovieIntent | None = None
    active_seed_movies: list[ResolvedMovie] = Field(default_factory=list, max_length=20)
    shown_movie_ids: list[int] = Field(default_factory=list)
    seen_movie_ids: list[int] = Field(default_factory=list)
    excluded_movie_ids: list[int] = Field(default_factory=list)
    country_code: str | None = None
    last_intent: str | None = None


@dataclass
class _Entry:
    state: MovieSessionState = field(default_factory=MovieSessionState)
    updated_at: float = 0
    users: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class SessionStore:
    """Serialize turns per session; lease entries so expiry/eviction cannot race.

    State is copied and committed only on a completed transaction. Short index
    operations use a thread lock; no lock is held while performing network calls.
    """
    def __init__(self, ttl=SESSION_TTL_SECONDS, max_sessions=MAX_SESSIONS, clock=time.monotonic):
        if ttl <= 0 or max_sessions < 1:
            raise ValueError("Session bounds must be positive")
        self.ttl, self.max_sessions, self.clock = ttl, max_sessions, clock
        self._entries = OrderedDict()
        self._index_lock = threading.RLock()

    def _expire(self):
        now = self.clock()
        for key, entry in list(self._entries.items()):
            if not entry.users and now - entry.updated_at >= self.ttl:
                del self._entries[key]

    @asynccontextmanager
    async def transaction(self, session_id):
        with self._index_lock:
            self._expire()
            if session_id not in self._entries:
                if len(self._entries) >= self.max_sessions:
                    victim = next((key for key, entry in self._entries.items() if not entry.users), None)
                    if victim is None:
                        raise RuntimeError("Session capacity is temporarily busy")
                    del self._entries[victim]
                self._entries[session_id] = _Entry(updated_at=self.clock())
            entry = self._entries[session_id]
            entry.users += 1
        try:
            async with entry.lock:
                state = entry.state.model_copy(deep=True)
                yield state
                with self._index_lock:
                    entry.state = state.model_copy(deep=True)
                    entry.updated_at = self.clock()
                    self._entries.move_to_end(session_id)
        finally:
            with self._index_lock:
                entry.users -= 1
                self._expire()

    def __len__(self):
        with self._index_lock:
            self._expire()
            return len(self._entries)


session_store = SessionStore()


def safe_text(text, limit=MAX_MESSAGE_CHARS):
    """Do not retain common credential/sensitive payloads, even in short history."""
    if re.search(r"sk-[\w-]+|\b(?:bearer|authorization|api[_ -]?key|password|secret|cookie|ssn|credit card|social security|bank account|iban|phone number|home address|diagnosis)\b|[\w.+-]+@[\w.-]+\.[a-z]{2,}", text, re.I):
        return ""
    return text[:limit]


def bounded_intent(intent):
    data = intent.model_dump()
    for key, value in data.items():
        if isinstance(value, list):
            data[key] = [clean for item in value[:20] if (clean := safe_text(item, 300))]
    data["top_n"] = min(intent.top_n or 10, MAX_CANDIDATES)
    return MovieIntent.model_validate(data)


def model_context(state):
    data = {
        "current_movie": state.current_movie.model_dump() if state.current_movie else None,
        "last_movie_list": [movie.model_dump() for movie in state.last_movie_list],
        "last_list_kind": state.last_list_kind,
        "active_preferences": state.active_preferences.model_dump(exclude_defaults=True) if state.active_preferences else None,
        "country_code": state.country_code,
        "recent_messages": [message.model_dump() for message in state.recent_messages],
    }
    # IDs here are data, not new authority to execute backend calls. The controller
    # separately seeds its allowlist only from stored backend-resolved references.
    for name in ("shown_movie_ids", "seen_movie_ids", "excluded_movie_ids"):
        data[name] = getattr(state, name)[-50:]
    if data["active_preferences"]:
        for key, value in data["active_preferences"].items():
            if isinstance(value, list):
                data["active_preferences"][key] = [item[:120] for item in value[:8]]
    encode = lambda: json.dumps(data, ensure_ascii=False)
    while len(encode()) > MAX_CONTEXT_CHARS and data["recent_messages"]:
        data["recent_messages"].pop(0)
    while len(encode()) > MAX_CONTEXT_CHARS and data["last_movie_list"]:
        data["last_movie_list"].pop()
        data["list_truncated"] = True
    if len(encode()) > MAX_CONTEXT_CHARS:
        data["active_preferences"] = None
        data["preferences_omitted_for_size"] = True
    for name in ("shown_movie_ids", "seen_movie_ids", "excluded_movie_ids"):
        while len(encode()) > MAX_CONTEXT_CHARS and data[name]:
            data[name].pop(0)
    return encode()


def track_ids(previous, additions):
    return list(dict.fromkeys([*previous, *additions]))[-MAX_TRACKED_MOVIE_IDS:]


_ORDINALS = {"first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4,
             "sixth": 5, "seventh": 6, "eighth": 7, "ninth": 8, "tenth": 9, "last": -1}
_REFERENCE = re.compile(r"\b(?:the\s+)?(?:(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last)\s+(?:one|movie|film)|number\s+(\d{1,3}))\b", re.I)
_SINGULAR = re.compile(r"\b(?:it|its|that movie|that one|that)\b", re.I)


def resolve_reference(message, state):
    match = _REFERENCE.search(message)
    if match:
        index = _ORDINALS[match[1].lower()] if match[1] else int(match[2]) - 1
        if (match[2] and int(match[2]) < 1) or not state.last_movie_list or index >= len(state.last_movie_list) or index < -1:
            return None, True
        return state.last_movie_list[index], True
    if _SINGULAR.search(message):
        if state.last_intent == "conversation":
            return None, False  # Let recent general conversation resolve its own pronouns.
        movie = state.current_movie
        if movie is None and len(state.last_movie_list) == 1:
            movie = state.last_movie_list[0]
        return movie, True
    return None, False


GENRES = {"sci-fi": "Science Fiction", "science fiction": "Science Fiction",
          "comedy": "Comedy", "drama": "Drama", "horror": "Horror", "thriller": "Thriller",
          "action": "Action", "adventure": "Adventure", "animation": "Animation",
          "romance": "Romance", "fantasy": "Fantasy", "documentary": "Documentary"}


def preference_update(message, base):
    """A small high-confidence command grammar; other language stays with the LLM."""
    text = message.lower().strip().rstrip(".!?")
    text = re.sub(r"\bnewer than (\d{4})\b", r"after \1", text)
    text = re.sub(r"\bolder than (\d{4})\b", r"before \1", text)
    data = base.model_dump()
    more = re.fullmatch(r"(?:give me|show me|recommend)?\s*(\d{1,2})?\s*more(?: movies| films)?", text)
    year = re.fullmatch(r"(?:actually )?(?:make (?:them|it) |only |something )?(?:newer[, ]+)?(?:than )?(after|before|since|from) (\d{4})", text)
    genre = re.fullmatch(r"(?:actually |only |make (?:them|it) |remove |no )?([a-z -]+?)(?: instead| movies| films)?", text)
    if more:
        data["top_n"] = max(1, min(int(more[1] or data["top_n"] or 10), MAX_CANDIDATES))
    elif year:
        value = int(year[2])
        if year[1] == "before":
            data.update(max_year=value - 1, min_year=None)
        else:
            data.update(min_year=value + (year[1] == "after"), max_year=None)
    elif genre and genre[1] in GENRES:
        name = GENRES[genre[1]]
        if text.startswith(("remove ", "no ")):
            data["exclude_genres"] = list(dict.fromkeys([*data["exclude_genres"], name]))
            for key in ("favorite_genres", "required_genres"):
                data[key] = [item for item in data[key] if item != name]
        else:
            data["required_genres"] = [name]
            data["favorite_genres"] = []
            data["exclude_genres"] = [item for item in data["exclude_genres"] if item != name]
    else:
        return None
    if not any((data["favorite_movies"], data["favorite_genres"], data["favorite_actors"], data["favorite_directors"])) and data["required_genres"]:
        # Only an existing expressed genre preference may be reused as a seed.
        # Do not convert required-only filters into invented favorite preferences.
        if base.favorite_genres:
            data["favorite_genres"] = list(data["required_genres"])
    return bounded_intent(MovieIntent.model_validate(data))


@dataclass
class ContinuationPlan:
    intent: MovieIntent
    reference_text: str | None = None
    seen: bool = False
    recommend: bool = True


_COUNT_WORDS = dict(zip("one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split(), range(1, 21)))
_COUNT_PATTERN = r"(?:\d{1,2}|" + "|".join(_COUNT_WORDS) + r")"
_UPDATE_PATTERN = r"(?:i(?:'ve| have)? (?:already )?(?:seen|watched)|exclude|avoid|remove|don't recommend|do not recommend)"


def continuation_plan(message, base):
    """Parse the complete command before any seen/exclusion state is changed."""
    text = " ".join(message.replace("\u2019", "'").lower().strip().rstrip(".!?").split())
    command = re.fullmatch(
        rf"(?:(?P<update>{_UPDATE_PATTERN} .+?)(?:[.!;]\s*|\s+and\s+))?"
        rf"(?:give me|show(?: me)?|recommend)\s+(?P<count>{_COUNT_PATTERN})?\s*"
        r"(?:more(?: movies| films)?|others|other movies|new movies|new films|something else)"
        r"(?:\s+(?:only )?(?P<year_filter>(?:after|before|since|from|newer than|older than) \d{4}))?", text)
    if not command:
        # Only explicit list references enter the exclusion-only shortcut.
        # Named movie preferences and more complex commands keep their existing routing.
        update = re.fullmatch(rf"{_UPDATE_PATTERN} (?P<reference>.+?)(?: again)?", text)
        if update and re.fullmatch(r"(?:(?:the )?(?:first (?:one|two|three|\d{1,2})(?: movies| films| ones)?|(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last)(?: one| movie| film))|numbers? \d{1,3}(?:\s*(?:,|and|&)\s*\d{1,3})*|these|those|them|those two|these two)", update["reference"]):
            return ContinuationPlan(intent=base, reference_text=update["reference"], seen=text.startswith("i"), recommend=False)
        return None
    count = command["count"]
    count = _COUNT_WORDS.get(count, count) or base.top_n or 10
    intent = preference_update(f"give me {count} more", base)
    if command["year_filter"]:
        intent = preference_update(command["year_filter"], intent)
    update = command["update"]
    seen = bool(update and update.startswith("i"))
    reference_text = re.sub(rf"^{_UPDATE_PATTERN}\s+", "", update) if update else None
    if reference_text:
        reference_text = re.sub(r"\s+again$", "", reference_text)
    return ContinuationPlan(intent=intent, reference_text=reference_text, seen=seen)


def continuation_references(text, state):
    # This command concerns recommendations, never search/research lists or seeds.
    if state.last_list_kind != "recommendations" or not state.last_movie_list:
        return None
    if text in {"these", "those", "them"}:
        return state.last_movie_list
    if text in {"those two", "these two"}:
        return state.last_movie_list if len(state.last_movie_list) == 2 else None
    first = re.fullmatch(r"(?:the )?first (one|two|three|\d{1,2})(?: movies| films| ones)?", text)
    if first:
        count = {"one": 1, "two": 2, "three": 3}.get(first[1], int(first[1]) if first[1].isdigit() else 0)
        return state.last_movie_list[:count] if 0 < count <= len(state.last_movie_list) else None
    numbers = re.fullmatch(r"numbers? (\d{1,3}(?:\s*(?:,|and|&)\s*\d{1,3})*)", text)
    if numbers:
        indices = list(dict.fromkeys(int(value) - 1 for value in re.findall(r"\d+", numbers[1])))
        return [state.last_movie_list[index] for index in indices] if all(0 <= index < len(state.last_movie_list) for index in indices) else None
    movie, has_reference = resolve_reference(text, state.model_copy(update={"current_movie": None}))
    return [movie] if has_reference and movie else None


async def execute_continuation(plan, message, country, state, tools):
    from backend.services.movie_chat import MovieChatResponse
    if plan.recommend and state.active_preferences is None and not plan.reference_text:
        return MovieChatResponse(status="needs_preferences", country_code=country,
            assistant_message="Tell me a movie, favorite genre, actor or director you like to start recommendations.")
    if plan.reference_text:
        targets = continuation_references(plan.reference_text, state)
        if not targets:
            return clarification(country, "Which movies should I exclude? Choose valid numbers from the latest list.")
        if plan.recommend and state.active_preferences is None:
            return clarification(country, "Which movie preferences should I use for the additional recommendations?")
        if sum(target.tmdb_id is None for target in targets) > MAX_SESSION_DATA_CALLS:
            return clarification(country, "Please identify at most five unresolved movies at a time.")
        resolved = []
        for target in targets:
            movie, resolution = await resolve_backend_reference(target, tools)
            if movie is None:
                result = clarification(country)
                if resolution.status == "needs_clarification":
                    result.clarifications = [MovieClarification(query=resolution.query, options=resolution.options)]
                return result
            resolved.append(movie)
        # Resolve every reference first; an invalid choice cannot partly update memory.
        ids = [movie.tmdb_id for movie in resolved]
        state.excluded_movie_ids = track_ids(state.excluded_movie_ids, ids)
        if plan.seen:
            state.seen_movie_ids = track_ids(state.seen_movie_ids, ids)
    if not plan.recommend:
        result = MovieChatResponse(status="success", country_code=country,
            assistant_message="I'll avoid those movies in this chat's recommendations.")
        update_state(state, message, result)
        return result
    plan.intent.country_code = country
    output = await session_recommend(tools, plan.intent, state)
    count = len(output.recommendations)
    text = (f"Here are {count} more movies:\n\n" + "\n".join(f"{index}. {movie['title']}" for index, movie in enumerate(output.recommendations, 1))) if count else "I couldn't find additional movies matching those preferences."
    if output.status != "success":
        text = output.message or "Please clarify your movie preferences."
    elif count < (plan.intent.top_n or 10):
        text += " There weren't enough unshown matches in the bounded candidate results."
    result = MovieChatResponse(**output.model_dump(), parsed_intent=plan.intent, assistant_message=text)
    # This turn's watched references are recommendations, not the original seeds.
    update_state(state, "Recommendation continuation", result)
    return result


def is_preference_followup(message, state):
    return bool(state.active_preferences and (
        preference_update(message, state.active_preferences) is not None
        or re.fullmatch(r"(?:something|make them|make it) (?:more|less) .{1,100}", message.strip().rstrip(".!?"), re.I)))


def merge_preferences(base, incoming, message):
    if base is None:
        return bounded_intent(incoming)
    deterministic = preference_update(message, base)
    if deterministic:
        return deterministic
    data = base.model_dump()
    # Empty/default tool fields mean unchanged. Nonempty fields are explicit edits.
    for key, value in incoming.model_dump().items():
        if value is not None and value != [] and key != "intent":
            data[key] = value
    if incoming.min_year is not None and incoming.max_year is None and base.max_year is not None and incoming.min_year > base.max_year:
        data["max_year"] = None
    if incoming.max_year is not None and incoming.min_year is None and base.min_year is not None and incoming.max_year < base.min_year:
        data["min_year"] = None
    return bounded_intent(MovieIntent.model_validate(data))


async def session_recommend(tools, intent, state, *, verified_movies=None):
    """One bounded engine call; preserve ranking while filtering identities."""
    intent = bounded_intent(intent)
    requested = intent.top_n or 10
    blocked = set(state.shown_movie_ids + state.seen_movie_ids + state.excluded_movie_ids)
    candidates = min(MAX_CANDIDATES, requested + len(blocked))
    if verified_movies is None and state.active_seed_movies:
        verified_movies = state.active_seed_movies
    if verified_movies:
        from backend.services.chat_recommender import compact_title
        titles = []
        for title in intent.favorite_movies:
            qualifier = re.fullmatch(r"(.+?)(?:\s*\((\d{4})\)|\s+(\d{4}))", title)
            base = qualifier[1] if qualifier else title
            year = int(qualifier[2] or qualifier[3]) if qualifier else None
            known = [movie for movie in verified_movies if compact_title(movie.title) == compact_title(base)
                and (year is None or movie.year == year)]
            titles.append((f"{known[0].title} ({known[0].year})" if known[0].year else known[0].title) if len(known) == 1 else title)
        intent = intent.model_copy(update={"favorite_movies": titles})
    kwargs = {"verified_movies": verified_movies} if verified_movies is not None else {}
    from backend.recommendation.experimental_ranking import excluded_context
    token = excluded_context.set(tuple(blocked))
    try:
        output = await run_in_threadpool(tools.recommend_movies, intent.model_copy(update={"top_n": candidates}), **kwargs)
    finally:
        excluded_context.reset(token)
    blocked.update(movie.tmdb_id for movie in output.resolved_movies)
    output.recommendations = [item for item in output.recommendations if item.get("tmdb_id") not in blocked][:requested]
    from backend.recommendation.experimental_ranking import ranking_context, record_diagnostic
    import logging
    import json
    mode, request_id = ranking_context.get()
    delivery = {"request_id": request_id, "requested_mode": mode,
        "final_ranked_tmdb_ids": [item.get("tmdb_id") for item in output.recommendations]}
    logging.getLogger("movie.ranking").info("ranking_delivery %s", json.dumps(delivery))
    record_diagnostic("ranking_delivery", delivery)
    return output


def references_from_response(response):
    if response.recommendations:
        return [MovieReference(tmdb_id=item["tmdb_id"], title=item["title"][:300],
                               year=item.get("release_year", item.get("year")))
                for item in response.recommendations[:MAX_MOVIE_REFERENCES]], "recommendations"
    if response.collection and response.collection.movies:
        return [MovieReference(tmdb_id=item.tmdb_id, title=item.title[:300], year=item.release_year)
                for item in response.collection.movies[:MAX_MOVIE_REFERENCES]], "collection"
    # Only explicit numbered title/year entries are captured. Research/model prose
    # supplies names, never trusted IDs; title resolution is required later.
    movies = []
    for line in response.assistant_message.splitlines():
        match = re.match(r"^\s*\d+[.)]\s+\**(.+?)\**\s+\((\d{4})\)", line)
        if match and (title := safe_text(match[1].strip(" *"), 300)):
            movies.append(MovieReference(title=title, year=int(match[2])))
    return movies[:MAX_MOVIE_REFERENCES], "research" if response.sources else "knowledge"


def update_state(state, message, response):
    if response.status in {"error", "unsupported_intent", "unresolved"}:
        return
    if response.status == "needs_clarification" and response.parsed_intent and response.clarifications:
        state.pending_task = PendingTask(intent=bounded_intent(response.parsed_intent),
            resolved_movies=response.resolved_movies[:20],
            clarifications=[item.model_copy(update={"options": item.options[:20]}) for item in response.clarifications[:20]],
            watched=bool(re.search(r"\b(?:seen|watched)\b", message, re.I)))
    elif response.status == "success" or response.status == "needs_preferences":
        state.pending_task = None
    state.country_code = response.country_code
    if response.parsed_intent:
        state.active_preferences = bounded_intent(response.parsed_intent)
        if response.status in {"success", "needs_preferences"} and response.resolved_movies:
            state.active_seed_movies = response.resolved_movies[:20]
        state.last_intent = "recommend"
        if response.status in {"success", "needs_preferences"} and re.search(r"\b(?:seen|watched)\b", message, re.I):
            ids = [movie.tmdb_id for movie in response.resolved_movies]
            state.seen_movie_ids = track_ids(state.seen_movie_ids, ids)
            state.excluded_movie_ids = track_ids(state.excluded_movie_ids, ids)
    movies, kind = references_from_response(response)
    if movies:
        state.last_movie_list, state.last_list_kind = movies, kind
        state.last_intent = kind
        state.current_movie = movies[0] if len(movies) == 1 else None
        if kind == "recommendations":
            state.shown_movie_ids = track_ids(state.shown_movie_ids, [movie.tmdb_id for movie in movies])
    elif response.recommendations == [] and response.parsed_intent and response.status == "success":
        state.last_movie_list = []
        state.current_movie = None
    elif response.resolved_movies and (not response.parsed_intent or response.status == "needs_preferences"):
        movie = response.resolved_movies[-1]
        state.current_movie = MovieReference(**movie.model_dump())
        state.last_intent = "information"
    elif response.status == "needs_clarification" and response.clarifications:
        state.current_movie = None
    for role, text in (("user", message), ("assistant", response.assistant_message)):
        if clean := safe_text(text):
            state.recent_messages.append(RecentMessage(role=role, content=clean))
    state.recent_messages = state.recent_messages[-MAX_RECENT_MESSAGES:]


def explicit_country(message):
    # This small helper mirrors common mock regions; other explicit countries can
    # still be extracted by the existing LLM provider/recommendation arguments.
    match = re.search(r"\bin (Sweden|the US|the United States|United States|UK|the UK|United Kingdom|France|Germany|Canada|Australia|[A-Z]{2})\b", message)
    if not match:
        match = re.search(r"\bin (sweden|the us|the united states|united states|the uk|uk|united kingdom|france|germany|canada|australia)\b", message, re.I)
    if not match:
        return None
    country = match[1]
    return {"sweden": "SE", "the us": "US", "the united states": "US", "united states": "US", "uk": "GB", "the uk": "GB",
            "united kingdom": "GB", "france": "FR", "germany": "DE", "canada": "CA", "australia": "AU"}.get(country.lower(), country.upper())


async def resolve_backend_reference(movie, tools):
    if movie.tmdb_id is not None:
        return movie, None
    output = await run_in_threadpool(tools.resolve_movie_title, movie.title, movie.year)
    if output.status == "success" and output.movie:
        return MovieReference(**output.movie.model_dump()), output
    return None, output


def clarification(country, text="Which movie do you mean? Choose a title or a number from the last list."):
    from backend.services.movie_chat import MovieChatResponse
    return MovieChatResponse(status="needs_clarification", country_code=country, assistant_message=text)


async def reference_information(message, movie, country, tools):
    from backend.services.movie_chat import MovieChatResponse
    movie, resolution = await resolve_backend_reference(movie, tools)
    result = MovieChatResponse(status="success", country_code=country, assistant_message="")
    if resolution:
        result.tool_results.append({"tool": "resolve_movie_title", "result": resolution.model_dump()})
    if movie is None:
        result.status = "needs_clarification" if resolution.status == "needs_clarification" else "unresolved"
        result.assistant_message = "Which movie do you mean?" if result.status == "needs_clarification" else "I couldn't resolve that movie in the catalog."
        if result.status == "needs_clarification":
            result.clarifications = [MovieClarification(query=resolution.query, options=resolution.options)]
        return result
    result.resolved_movies = [ResolvedMovie(tmdb_id=movie.tmdb_id, title=movie.title, year=movie.year)]
    provider = bool(re.search(r"\b(?:watch|stream|available|how about)\b", message, re.I))
    name = "get_watch_providers" if provider else "get_movie_details"
    output = await run_in_threadpool(getattr(tools, name), movie.tmdb_id, country) if provider else await run_in_threadpool(tools.get_movie_details, movie.tmdb_id)
    result.tool_results.append({"tool": name, "result": output.model_dump()})
    if output.status == "error":
        result.status, result.assistant_message = "error", "Movie information is temporarily unavailable."
    elif provider:
        result.country_code = output.watch_providers.country_code
        sections = []
        for field_name, label in (("stream", "stream"), ("free", "watch free"), ("ads", "watch with ads"), ("rent", "rent"), ("buy", "buy")):
            names = [p.name for p in getattr(output.watch_providers, field_name) if p.name]
            if names:
                sections.append(f"{label} on {', '.join(names)}")
        result.assistant_message = f"In {country}, you can { '; '.join(sections)} for {movie.title}." if sections else f"I don't have availability information for {movie.title} in {country} right now."
    elif output.movie:
        details = output.movie
        text = message.lower()
        if "direct" in text:
            result.assistant_message = f"{movie.title} was directed by {details.director}." if details.director else "Director information is unavailable."
        elif "rat" in text:
            result.assistant_message = f"{movie.title} has a TMDB rating of {details.vote_average}/10." if details.vote_average is not None else "Rating information is unavailable."
        elif "short" in text or "runtime" in text or "long" in text:
            result.assistant_message = f"{movie.title} runs {details.runtime} minutes." if details.runtime is not None else "Runtime information is unavailable."
        else:
            result.assistant_message = details.overview[:1200] if details.overview else "Overview information is unavailable."
    else:
        result.status, result.assistant_message = "unresolved", "Movie details are unavailable."
    return result


async def list_comparison(message, state, country, tools):
    from backend.services.movie_chat import MovieChatResponse
    if not state.last_movie_list:
        return clarification(country)
    field_name, label = ("vote_average", "TMDB rating") if "rat" in message.lower() else (("runtime", "runtime in minutes") if "short" in message.lower() else ("release_date", "release date"))
    result = MovieChatResponse(status="success", country_code=country, assistant_message="")
    candidates = []
    missing = False
    for reference in state.last_movie_list[:MAX_MOVIE_REFERENCES]:
        needed = 1 + (reference.tmdb_id is None)
        if len(result.tool_results) + needed > MAX_SESSION_DATA_CALLS:
            missing = True
            break
        movie, resolution = await resolve_backend_reference(reference, tools)
        if resolution:
            result.tool_results.append({"tool": "resolve_movie_title", "result": resolution.model_dump()})
        if movie is None:
            missing = True
            continue
        output = await run_in_threadpool(tools.get_movie_details, movie.tmdb_id)
        result.tool_results.append({"tool": "get_movie_details", "result": output.model_dump()})
        value = getattr(output.movie, field_name) if output.movie else None
        if value is not None:
            candidates.append((value, movie))
        else:
            missing = True
    if not candidates:
        result.status, result.assistant_message = "unresolved", "I don't have enough movie data to compare that list."
    else:
        value, movie = (min if field_name == "runtime" else max)(candidates, key=lambda pair: pair[0])
        result.resolved_movies = [ResolvedMovie(**movie.model_dump())]
        result.assistant_message = f"{movie.title} ({movie.year}) has the {'lowest' if field_name == 'runtime' else 'highest' if field_name == 'vote_average' else 'latest'} {label}: {value}."
        if missing:
            result.assistant_message += " This compares only movies with available data retrieved within this turn's limit; other list entries may differ."
    return result


def pending_question(task, country):
    result = clarification(country, "Which movie did you mean by " + task.clarifications[0].query + "? Choose a title and year or a number from these options.")
    result.clarifications = [task.clarifications[0]]
    result.parsed_intent = task.intent.model_copy(deep=True)
    result.resolved_movies = task.resolved_movies
    return result


async def resume_pending(message, country, state, tools):
    """Only backend-provided candidates can complete a pending identity slot."""
    from backend.services.movie_chat import MovieChatResponse
    from backend.services.chat_recommender import _normalize, compact_title
    task = state.pending_task.model_copy(deep=True)
    options = task.clarifications[0].options
    text = re.sub(r"^(?:no|yes)[, ]+", "", message.strip().rstrip(".!?"), flags=re.I)
    normalized = _normalize(text)
    selected = []
    if normalized in {"both", "both of them", "both movies"} and len(options) == 2:
        selected = options
    else:
        ordinal = re.fullmatch(r"(?:the )?(first|second|third|fourth|fifth|last)(?: one| movie| film)?|(?:number )?(\d{1,2})", normalized)
        if ordinal:
            index = _ORDINALS[ordinal[1]] if ordinal[1] else int(ordinal[2]) - 1
            if -1 <= index < len(options) and (not ordinal[2] or int(ordinal[2]) > 0):
                selected = [options[index]]
        else:
            year_match = re.fullmatch(r"(?:the )?(\d{4})(?: one| movie| version)?", normalized)
            for movie in options:
                names = {_normalize(movie.title), _normalize(f"{movie.title} ({movie.year})"), _normalize(f"{movie.title} {movie.year}")}
                if normalized in names or (year_match and movie.year == int(year_match[1])):
                    selected.append(movie)
            # A bare duplicate title is still ambiguous.
            if len(selected) > 1:
                selected = []
    if not selected:
        # A correction can name a verified release omitted from bounded options.
        qualified = re.fullmatch(r"(.+?)(?:\s*\((\d{4})\)|\s+(\d{4}))", text)
        if qualified and _normalize(qualified[1]) in {_normalize(task.clarifications[0].query), *(_normalize(movie.title) for movie in options)}:
            resolution = await run_in_threadpool(tools.resolve_movie_title, qualified[1], int(qualified[2] or qualified[3]))
            if resolution.status == "success" and resolution.movie:
                selected = [resolution.movie]
        if not selected:
            return pending_question(task, country)
    replacements = [f"{movie.title} ({movie.year})" if movie.year else movie.title for movie in selected]
    query = task.clarifications[0].query
    if len(task.intent.favorite_movies) + len(replacements) - 1 > MAX_MOVIE_REFERENCES:
        return pending_question(task, country)
    task.clarifications.pop(0)
    task.intent.favorite_movies = [replacement for title in task.intent.favorite_movies
        for replacement in (replacements if title == query else [title])]
    task.resolved_movies = list({movie.tmdb_id: movie for movie in [*task.resolved_movies, *selected]}.values())[:20]
    if task.clarifications:
        state.pending_task = task
        return pending_question(task, country)
    # Replace earlier resolved aliases as well, preserving exact backend identities.
    for index, title in enumerate(task.intent.favorite_movies):
        candidates = tools.controller.resolve_title(title)
        base_title = re.sub(r"(?:\s*\(\d{4}\)|\s+\d{4})$", "", title)
        known = [movie for movie in task.resolved_movies if any(candidate.tmdb_id == movie.tmdb_id for candidate in candidates)
            or compact_title(base_title) == compact_title(movie.title)]
        if len(known) == 1:
            movie = known[0]
            task.intent.favorite_movies[index] = f"{movie.title} ({movie.year})" if movie.year else movie.title
    intent = task.intent.model_copy(deep=True)
    intent.country_code = country
    output = await session_recommend(tools, intent, state, verified_movies=task.resolved_movies)
    result = MovieChatResponse(**output.model_dump(), parsed_intent=intent,
        assistant_message=("Based on " + ", ".join(movie.title for movie in output.resolved_movies) + ", here are movies you might enjoy:\n\n" + "\n".join(f"{i}. {movie['title']}" for i, movie in enumerate(output.recommendations, 1)))
        if output.status == "success" else (output.message or "I couldn't resolve all of the requested movie seeds."))
    if task.watched and output.status in {"success", "needs_preferences"}:
        ids = [movie.tmdb_id for movie in output.resolved_movies]
        state.seen_movie_ids = track_ids(state.seen_movie_ids, ids)
        state.excluded_movie_ids = track_ids(state.excluded_movie_ids, ids)
    if output.status in {"success", "needs_preferences"}:
        state.pending_task = None
    update_state(state, message, result)
    return result


def starts_new_task(message):
    return bool(is_recommendation_request(message) or has_current_movie_preferences(message)
        or re.match(r"^(?:actually[, ]*)?(?:forget\b|instead\b|who\b|what\b|why\b|how\b|where\b|tell me\b|explain\b|find\b|show me\b|give me\b|write\b|can you\b)", message.strip(), re.I))


async def handle_session_turn(message, country, state, controller, tools, responder):
    """Run inside the store transaction. Guard current input before memory edits."""
    from backend.services.movie_chat import MovieChatResponse, mock_enabled
    from backend.core.movie_guardrails import MAX_MESSAGE_LENGTH
    if not message.strip() or len(message) > MAX_MESSAGE_LENGTH:
        raise ValueError("Invalid chat message")
    redirect = scope_redirect(message)
    if redirect:
        state.pending_task = None
        return MovieChatResponse(status="unsupported_intent", country_code=country, assistant_message=redirect, mock_mode=mock_enabled())
    country = explicit_country(message) or state.country_code or country
    if state.pending_task:
        from backend.services.chat_recommender import _normalize
        answer = _normalize(re.sub(r"^(?:no|yes)[, ]+", "", message.strip().rstrip(".!?"), flags=re.I))
        candidate_title = any(answer in {_normalize(movie.title), _normalize(f"{movie.title} ({movie.year})"), _normalize(f"{movie.title} {movie.year}")}
            for movie in state.pending_task.clarifications[0].options)
        if candidate_title or not starts_new_task(message):
            result = await resume_pending(message, country, state, tools)
            result.mock_mode = mock_enabled()
            return result
        state.pending_task = None
        state.active_preferences = None
        state.active_seed_movies = []
        state.current_movie = None
        state.last_intent = None
        message = re.sub(r"^(?:actually[, ]*)?(?:forget that[.! ,]*)(?:instead[, ]*)?", "", message, flags=re.I)
    text = message.lower().strip().rstrip(".!?").replace("’", "'")
    recommendation = is_recommendation_request(message)
    explicit_seeds = has_current_movie_preferences(message)
    reference, has_reference = resolve_reference(message, state)
    plan = continuation_plan(message, state.active_preferences or MovieIntent(intent="recommend"))
    if plan:
        result = await execute_continuation(plan, message, country, state, tools)
        result.mock_mode = mock_enabled()
        return result
    if re.match(rf"^{_UPDATE_PATTERN}\b", text) and re.search(
        rf"\b(?:give me|show(?: me)?|recommend)\s+(?:{_COUNT_PATTERN}\s+)?(?:more|others|new movies|something else)\b", text):
        return clarification(country, "Please restate both instructions using movie numbers, a count and an optional after/before year so I can apply the complete update.")
    seen = bool(re.match(r"^(?:i(?:'ve| have)? (?:already )?(?:seen|watched)|i already watched|not that one)\b", text))
    exclusion = re.fullmatch(r"(?:don't|do not|never) recommend (.+)", text, re.I)
    if state.active_preferences is None and preference_update(message, MovieIntent(intent="recommend")) is not None:
        return MovieChatResponse(status="needs_preferences", country_code=country,
            assistant_message="Tell me a movie, favorite genre, actor or director you like to start recommendations.")
    if (seen and not recommendation and not explicit_seeds) or exclusion:
        targets = []
        first = re.search(r"\bfirst (two|three|\d{1,2})\b", text)
        if first:
            count = {"two": 2, "three": 3}.get(first[1], int(first[1]) if first[1].isdigit() else 0)
            if 0 < count <= len(state.last_movie_list):
                targets = state.last_movie_list[:count]
        elif exclusion:
            # Preserve original casing, including year qualifiers.
            title = re.fullmatch(r"(?:don't|do not|never) recommend (.+)", message.strip().rstrip(".!?"), re.I)[1]
            references = [*state.last_movie_list, *([state.current_movie] if state.current_movie else [])]
            known = list({(movie.tmdb_id, movie.title, movie.year): movie for movie in references
                          if movie.title.casefold() == title.casefold() or movie.qualified_title().casefold() == title.casefold()}.values())
            targets = known if len(known) == 1 else [MovieReference(title=title[:300])]
        elif reference:
            targets = [reference]
        if not targets:
            return clarification(country)
        resolved = []
        for target in targets:
            if len(resolved) >= MAX_SESSION_DATA_CALLS and target.tmdb_id is None:
                return clarification(country, "Please identify at most five unresolved movies at a time.")
            movie, output = await resolve_backend_reference(target, tools)
            if movie is None:
                result = clarification(country)
                if output.status == "needs_clarification":
                    result.clarifications = [MovieClarification(query=output.query, options=output.options)]
                return result
            resolved.append(movie)
        ids = [movie.tmdb_id for movie in resolved]
        state.excluded_movie_ids = track_ids(state.excluded_movie_ids, ids)
        if seen:
            state.seen_movie_ids = track_ids(state.seen_movie_ids, ids)
        result = MovieChatResponse(status="success", country_code=country, assistant_message="I'll avoid those movies in this chat's recommendations.")
    elif re.fullmatch(r"which(?: one)?(?: is)? (?:highest rated|the highest rated|newest|the newest|shortest|the shortest)", text):
        result = await list_comparison(message, state, country, tools)
    elif recommendation and not explicit_seeds and re.search(r"\b(?:similar to|like)\s+(?:those|them|these)\b", text):
        if not state.last_movie_list:
            return clarification(country, "Which movies should I use for similar recommendations? Please name them.")
        # Use the actual referenced list, including year qualifiers. Unknown
        # research identities and duplicate titles still pass through the resolver.
        if re.fullmatch(r"(?:give me|recommend|suggest|show me|i want)?\s*(?:movies|films|something)\s+(?:similar to|like)\s+(?:those|them|these)(?: movies| films| ones)?", text):
            intent = MovieIntent(intent="recommend", country_code=country,
                favorite_movies=[movie.qualified_title() for movie in state.last_movie_list])
            output = await session_recommend(tools, intent, state)
            result = MovieChatResponse(**output.model_dump(), parsed_intent=intent,
                assistant_message="Here are movies similar to that list:" if output.status == "success" else (output.message or "Please clarify the movie titles."))
        else:
            # Let the existing controller extract additional current-turn filters;
            # the simple plural grammar must not silently discard them.
            result = await responder(message, country, controller, tools, session=state)
    elif not explicit_seeds and has_reference and re.search(r"\b(?:watch|stream|directed|director|rating|rated|tell|about|runtime|long|short|how about)\b", text):
        if reference is None:
            return clarification(country)
        if re.search(r"\bin\s+[A-Za-z]", message) and explicit_country(message) is None:
            # Less common region names stay with the existing model extraction,
            # rather than silently applying the wrong remembered region.
            result = await responder(message, country, controller, tools, session=state)
        else:
            result = await reference_information(message, reference, country, tools)
    elif state.active_preferences and (intent := preference_update(message, state.active_preferences)):
        intent.country_code = country
        output = await session_recommend(tools, intent, state)
        count = len(output.recommendations)
        requested = intent.top_n or 10
        result = MovieChatResponse(**output.model_dump(), parsed_intent=intent, assistant_message=(f"Here are {count} more movies:" if count else "I couldn't find additional movies matching those preferences.") if output.status == "success" else (output.message or "Please clarify your movie preferences."))
        if output.status == "success" and count < requested:
            result.assistant_message += " There weren't enough unshown matches in the bounded candidate results."
    else:
        followup = is_preference_followup(message, state)
        if is_recommendation_request(message) and not followup:
            state.active_preferences = None
            state.active_seed_movies = []
            state.shown_movie_ids = []  # New explicit recommendation context.
            state.last_movie_list = []
            state.current_movie = None
            state.last_list_kind = None
            state.last_intent = None
        result = await responder(message, country, controller, tools, session=state)
        if result.status == "success" and not result.parsed_intent and not result.resolved_movies and not result.tool_results:
            # A new conversational subject must not keep a stale movie selected.
            # The relevant movie list/preferences remain available for explicit returns.
            state.current_movie = None
            state.last_intent = "conversation"
    result.mock_mode = mock_enabled()
    update_state(state, message, result)
    return result
