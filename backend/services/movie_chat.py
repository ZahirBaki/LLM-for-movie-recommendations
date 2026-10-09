"""Single-message orchestration with deterministic, opt-in development fixtures."""

import os
from typing import Literal
from pydantic import Field

from fastapi.concurrency import run_in_threadpool
from backend.services.chat_service import MovieIntent
from backend.services.movie_assistant import respond_live
from backend.services.chat_recommender import ChatRecommender, IntentRecommendationResult, MovieClarification
from backend.services.movie_collections import CollectionMatch, CollectionResult, collection_response_text
from backend.core.movie_guardrails import scope_redirect
from backend.integrations.movie_research import ResearchSource, ResearchCitation


MOCK_PHRASES = {
    "something like interstellar": {"favorite_movies": ["Interstellar"]},
    "recommend something like interstellar.": {"favorite_movies": ["Interstellar"]},
    "give me 5 movies like coco and inside out.": {"favorite_movies": ["Coco", "Inside Out"], "top_n": 5},
    "i have watched coco, inside out and zootopia i want movies similar to those": {"favorite_movies": ["Coco", "Inside Out", "Zootopia"]},
    "something like coco, inside out and zootopia": {"favorite_movies": ["Coco", "Inside Out", "Zootopia"]},
    "something like interstellar with ryan gosling after 2015": {"favorite_movies": ["Interstellar"], "required_actors": ["Ryan Gosling"], "min_year": 2016},
    "something like crash": {"favorite_movies": ["Crash"]},
    "something like crash (1996)": {"favorite_movies": ["Crash (1996)"]},
    "something like crash (2005)": {"favorite_movies": ["Crash (2005)"]},
    "something like nonexistent movie xyz987654": {"favorite_movies": ["Nonexistent Movie xyz987654"]},
    "recommend a comedy": {"required_genres": ["Comedy"]},
    "i watched interstellar, arrival and dune. what should i watch next?": {"favorite_movies": ["Interstellar", "Arrival", "Dune"]},
}

# Explicit offline fixtures; general conversation in live mode uses the LLM.
MOCK_GENERAL_PHRASES = {
    "what is machine learning?": "Machine learning is a way for computers to learn patterns from examples and use them to make predictions or decisions.",
    "why is the sky blue?": "Air molecules scatter short blue wavelengths of sunlight more strongly than longer red wavelengths, making the sky look blue.",
    "who was albert einstein?": "Albert Einstein was a physicist best known for developing the theories of relativity.",
    "what is the chronological order of the x-men movies?": "A story watch order is interpretive because the X-Men films have branching timelines. It's general movie knowledge, rather than a TMDB release-order list.",
}


class MovieChatResponse(IntentRecommendationResult):
    status: Literal["success", "needs_clarification", "unresolved", "needs_preferences", "unsupported_intent", "error"]
    assistant_message: str
    parsed_intent: MovieIntent | None = None
    mock_mode: bool = False
    tool_results: list[dict] = Field(default_factory=list)
    openai_requests: int = 0
    collection: CollectionResult | None = None
    collection_choices: list[CollectionMatch] = Field(default_factory=list)
    sources: list[ResearchSource] = Field(default_factory=list)
    citations: list[ResearchCitation] = Field(default_factory=list)


def mock_enabled() -> bool:
    # Disabled by default. Explicit production environments always block fixtures.
    production = any(os.getenv(name, "").casefold() == "production" for name in ("ENVIRONMENT", "APP_ENV", "NODE_ENV"))
    return not production and os.getenv("MOVIE_CHAT_MOCK", "false").casefold() == "true"


async def respond_to_movie_chat(message: str, country_code: str, controller: ChatRecommender, tools=None, *, session=None) -> MovieChatResponse:
    mock = mock_enabled()
    redirect = scope_redirect(message)
    if redirect:
        return MovieChatResponse(status="unsupported_intent", country_code=country_code, mock_mode=mock, assistant_message=redirect)
    if not mock:
        if session is not None:
            return await respond_live(message, country_code, tools, session=session)
        return await respond_live(message, country_code, tools)
    if mock:
        normalized = " ".join(message.split()).casefold()
        if normalized in MOCK_GENERAL_PHRASES:
            return MovieChatResponse(status="success", country_code=country_code, mock_mode=True,
                assistant_message=MOCK_GENERAL_PHRASES[normalized])
        if tools and normalized in {"give me the x-men collection in release order.", "give me the planet of the apes collection in release order.", "give me the planet of the apes (original) collection in release order.", "give me the planet of the apes (reboot) collection in release order."}:
            import re
            name = re.sub(r"^give me the\s+|\s+(?:movies|collection) in release order\.$", "", message.strip(), flags=re.I)
            return await mock_collection(name, country_code, tools, message)
        if normalized in {"hello", "how are you?", "thanks", "what can you do?"}:
            return MovieChatResponse(status="success", country_code=country_code, mock_mode=True,
                assistant_message="I'm ready to help you find movies, check movie facts, or see where to watch them.")
        if tools and normalized.startswith(("who directed ", "what's ", "who stars in ", "tell me about ", "where can i watch ", "find movies called ")):
            return await mock_information(message, country_code, tools)
        fixture = MOCK_PHRASES.get(" ".join(message.split()).casefold())
        if fixture is None:
            return MovieChatResponse(status="error", country_code=country_code, mock_mode=True,
                assistant_message="This phrase is not available in development mock mode. Try ‘Something like Interstellar’ or ‘Something like Crash’.")
        parsed = MovieIntent(**{"intent": "recommend", **fixture})
    # The visible website region is authoritative for watch-provider enrichment.
    parsed = MovieIntent.model_validate({**parsed.model_dump(), "country_code": country_code})
    if session is not None:
        from backend.sessions.movie_session import session_recommend
        result = await session_recommend(tools, parsed, session)
    else:
        result = await run_in_threadpool(controller.recommend_from_intent, parsed)
    if result.status == "success":
        text = f"Here are {len(result.recommendations)} movies I'd recommend:" if result.recommendations else "I couldn't find movies matching those preferences. Try relaxing a filter."
    elif result.status == "needs_clarification":
        titles = ", ".join(item.query for item in result.clarifications)
        text = f"I found more than one movie called {titles}. Which one did you mean?"
    elif result.status == "unresolved":
        text = "I couldn't find that movie in my catalog. Try another title."
    elif result.status == "needs_preferences":
        text = "Tell me a movie you like, or a favorite genre, actor, or director so I can make recommendations."
    else:
        text = "I can help recommend movies. Tell me a movie you like or what you're in the mood for."
    return MovieChatResponse(**result.model_dump(), assistant_message=text, parsed_intent=parsed, mock_mode=mock)


async def mock_information(message, country_code, tools):
    """Small deterministic fixtures; no OpenAI or separate movie-data implementation."""
    import re
    result = MovieChatResponse(status="success", country_code=country_code, mock_mode=True, assistant_message="")
    lower = message.casefold()
    if lower.startswith("find movies called "):
        title = message[len("find movies called "):].rstrip(".?")
        output = await run_in_threadpool(tools.search_movies, title)
        result.tool_results = [{"tool": "search_movies", "result": output.model_dump()}]
        intro = "I found one movie with that title:" if len(output.matches) == 1 else f"I found {len(output.matches)} movies with that title:"
        result.assistant_message = intro + "\n" + "\n".join(f"{m.title} ({m.year})" for m in output.matches) if output.matches else "I couldn't find that title."
        result.status = "success" if output.matches else "unresolved"
        return result
    title = re.sub(r"^(who directed |what's |who stars in |tell me about |where can i watch )", "", message, flags=re.I).rstrip(".?")
    title = re.sub(r"['’]s rating$", "", title, flags=re.I)
    if lower.startswith("where can i watch "):
        match = re.search(r" in (Sweden|the US|the United States)$", title, re.I)
        if match:
            country_code = "SE" if match.group(1).casefold() == "sweden" else "US"
            title = title[:match.start()]
    resolution = await run_in_threadpool(tools.resolve_movie_title, title)
    result.tool_results.append({"tool": "resolve_movie_title", "result": resolution.model_dump()})
    if resolution.status == "needs_clarification":
        result.status = "needs_clarification"
        result.clarifications = [MovieClarification(query=title, options=resolution.options)]
        result.assistant_message = "Which movie do you mean: " + ", ".join(f"{m.title} ({m.year})" for m in resolution.options) + "?"
    elif resolution.status != "success":
        result.status = "unresolved" if resolution.status == "not_found" else "error"
        result.assistant_message = "I couldn't find that movie in my catalog. Try another title."
    else:
        result.resolved_movies = [resolution.movie]
        if lower.startswith("where can i watch "):
            output = await run_in_threadpool(tools.get_watch_providers, resolution.movie.tmdb_id, country_code)
            result.country_code = country_code
            names = [p.name for p in output.watch_providers.stream if p.name]
            country_name = {"SE": "Sweden", "US": "the United States"}.get(country_code, country_code)
            result.assistant_message = f"In {country_name}, you can stream {resolution.movie.title} on {', '.join(names)}." if names else f"I don't have streaming information for {resolution.movie.title} in {country_name} right now."
            name = "get_watch_providers"
        else:
            output = await run_in_threadpool(tools.get_movie_details, resolution.movie.tmdb_id)
            movie = output.movie
            if movie is None:
                result.assistant_message = "Movie details are unavailable."
            elif lower.startswith("who directed "):
                result.assistant_message = f"{movie.title} was directed by {movie.director}." if movie.director else "Director information is unavailable."
            elif lower.startswith("what's "):
                result.assistant_message = f"{movie.title} has a TMDB rating of {movie.vote_average}/10." if movie.vote_average is not None else "Rating information is unavailable."
            elif lower.startswith("who stars in "):
                result.assistant_message = movie.actors or "Cast information is unavailable."
            else:
                result.assistant_message = movie.overview or "Overview information is unavailable."
            name = "get_movie_details"
        result.tool_results.append({"tool": name, "result": output.model_dump()})
        if output.status == "error":
            result.status = "error"
    return result


async def mock_collection(query, country_code, tools, user_message=""):
    search = await run_in_threadpool(tools.search_movie_collection, query)
    result = MovieChatResponse(status="success", country_code=country_code, mock_mode=True, assistant_message="",
        tool_results=[{"tool": "search_movie_collection", "result": search.model_dump()}])
    if search.status == "success":
        collection = await run_in_threadpool(tools.get_movie_collection, search.resolved_collection_id)
        result.collection = collection
        result.tool_results.append({"tool": "get_movie_collection", "result": collection.model_dump()})
        result.status = {"success": "success", "not_found": "unresolved", "error": "error"}[collection.status]
        result.assistant_message = collection_response_text(collection, user_message)
    else:
        result.status = {"not_found": "unresolved", "needs_clarification": "needs_clarification", "error": "error"}[search.status]
        result.collection_choices = search.matches if search.status == "needs_clarification" else []
        result.assistant_message = collection_response_text(search, user_message)
    return result
