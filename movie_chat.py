"""Single-message orchestration with deterministic, opt-in development fixtures."""

import os
from typing import Literal

from fastapi.concurrency import run_in_threadpool
from chat_service import MovieIntent, parse_movie_message
from chat_recommender import ChatRecommender, IntentRecommendationResult


MOCK_PHRASES = {
    "something like interstellar": {"favorite_movies": ["Interstellar"]},
    "something like coco, inside out and zootopia": {"favorite_movies": ["Coco", "Inside Out", "Zootopia"]},
    "something like interstellar with ryan gosling after 2015": {"favorite_movies": ["Interstellar"], "required_actors": ["Ryan Gosling"], "min_year": 2016},
    "something like crash": {"favorite_movies": ["Crash"]},
    "something like crash (1996)": {"favorite_movies": ["Crash (1996)"]},
    "something like crash (2005)": {"favorite_movies": ["Crash (2005)"]},
    "something like nonexistent movie xyz987654": {"favorite_movies": ["Nonexistent Movie xyz987654"]},
    "recommend a comedy": {"required_genres": ["Comedy"]},
    "who directed interstellar?": {"intent": "information", "movie_titles": ["Interstellar"]},
}


class MovieChatResponse(IntentRecommendationResult):
    status: Literal["success", "needs_clarification", "unresolved", "needs_preferences", "unsupported_intent", "error"]
    assistant_message: str
    parsed_intent: MovieIntent | None = None
    mock_mode: bool = False


def mock_enabled() -> bool:
    # Disabled by default. Explicit production environments always block fixtures.
    production = any(os.getenv(name, "").casefold() == "production" for name in ("ENVIRONMENT", "APP_ENV", "NODE_ENV"))
    return not production and os.getenv("MOVIE_CHAT_MOCK", "false").casefold() == "true"


async def respond_to_movie_chat(message: str, country_code: str, controller: ChatRecommender) -> MovieChatResponse:
    mock = mock_enabled()
    if mock:
        fixture = MOCK_PHRASES.get(" ".join(message.split()).casefold())
        if fixture is None:
            return MovieChatResponse(status="error", country_code=country_code, mock_mode=True,
                assistant_message="This phrase is not available in development mock mode. Try ‘Something like Interstellar’ or ‘Something like Crash’.")
        parsed = MovieIntent(**{"intent": "recommend", **fixture})
    else:
        parsed = await parse_movie_message(message)
    # The visible website region is authoritative for watch-provider enrichment.
    parsed = MovieIntent.model_validate({**parsed.model_dump(), "country_code": country_code})
    result = await run_in_threadpool(controller.recommend_from_intent, parsed)
    if result.status == "success":
        text = "I found some movies you might like." if result.recommendations else "I couldn't find movies matching those preferences. Try relaxing a filter."
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
