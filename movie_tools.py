"""OpenAI-independent movie capabilities, bound once to existing API services."""

import math
from collections.abc import Callable
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from chat_recommender import ChatRecommender, IntentRecommendationResult, ResolvedMovie
from chat_service import MovieIntent


def json_safe(value: Any) -> Any:
    """Copy output to JSON primitives; unknown/nonfinite scalar values become null."""
    if isinstance(value, BaseModel):
        return json_safe(value.model_dump())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, str):
        return str(value)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if hasattr(value, "item"):
        item = value.item()
        return json_safe(item) if type(item) is not type(value) else None
    return None


class ToolModel(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)


class MovieDetails(ToolModel):
    tmdb_id: int
    title: str
    original_title: str | None = None
    release_date: str | None = None
    release_year: int | None = None
    genres: str | None = None
    director: str | None = None
    actors: str | None = None
    overview: str | None = None
    original_language: str | None = None
    popularity: float | None = None
    vote_average: float | None = None
    vote_count: int | None = None
    runtime: int | None = None
    poster_path: str | None = None
    poster_url: str | None = None
    rating_source: Literal["TMDB"] = "TMDB"


class MovieSearchMatch(ToolModel):
    tmdb_id: int
    title: str
    year: int | None = None
    poster_url: str | None = None
    original_title: str | None = None
    genres: str | None = None


class SearchResult(ToolModel):
    status: Literal["success", "not_found", "error"]
    query: str
    matches: list[MovieSearchMatch] = Field(default_factory=list)
    message: str | None = None


class DetailsResult(ToolModel):
    status: Literal["success", "not_found", "error"]
    movie: MovieDetails | None = None
    message: str | None = None


class ResolutionResult(ToolModel):
    status: Literal["success", "not_found", "needs_clarification", "error"]
    query: str
    movie: ResolvedMovie | None = None
    options: list[ResolvedMovie] = Field(default_factory=list)
    message: str | None = None


class Provider(ToolModel):
    provider_id: int | None = None
    name: str | None = None
    logo_url: str | None = None
    display_priority: int | None = None


class WatchProviders(ToolModel):
    country_code: str
    stream: list[Provider] = Field(default_factory=list)
    free: list[Provider] = Field(default_factory=list)
    ads: list[Provider] = Field(default_factory=list)
    rent: list[Provider] = Field(default_factory=list)
    buy: list[Provider] = Field(default_factory=list)
    link: str | None = None


class WatchResult(ToolModel):
    status: Literal["success", "not_found", "not_available", "error"]
    tmdb_id: int
    watch_providers: WatchProviders
    message: str | None = None


class RecommendationResult(IntentRecommendationResult):
    status: Literal["success", "needs_clarification", "unresolved", "unsupported_intent", "needs_preferences", "error"]


class MovieTools:
    """Bind the existing search, resolver, enrichment and ranking services.

    No globals, model loading, network calls or LLM calls occur at construction.
    api.movie_tools is the shared application instance. Tests can inject fixtures.
    """

    def __init__(self, catalog: pd.DataFrame, controller: ChatRecommender,
                 search_runner: Callable[[str], list[dict]], provider_runner: Callable[[int, str], dict]):
        self.catalog = catalog
        self.controller = controller
        self.search_runner = search_runner
        self.provider_runner = provider_runner
        self.by_id = {}
        for index, row in enumerate(catalog.itertuples()):
            if pd.notna(row.tmdb_id):
                self.by_id.setdefault(int(row.tmdb_id), index)

    def search_movies(self, query: str, limit: int = 10) -> SearchResult:
        """Reuse local website search order and preserve separate TMDB identities."""
        if not isinstance(query, str) or len(query.strip()) < 2 or type(limit) is not int or not 1 <= limit <= 10:
            return SearchResult(status="error", query=query if isinstance(query, str) else "", message="Use a query of at least two characters and a limit from 1 to 10.")
        try:
            matches = []
            for raw in self.search_runner(query)[:limit]:
                data = json_safe(raw)
                details = self.get_movie_details(data["tmdb_id"])
                if details.movie:
                    data.update(original_title=details.movie.original_title, genres=details.movie.genres)
                matches.append(MovieSearchMatch.model_validate(data))
            return SearchResult(status="success" if matches else "not_found", query=query, matches=matches)
        except Exception:
            return SearchResult(status="error", query=query, message="Movie search is temporarily unavailable.")

    def get_movie_details(self, tmdb_id: int) -> DetailsResult:
        if type(tmdb_id) is not int or tmdb_id <= 0:
            return DetailsResult(status="error", message="Use a positive integer TMDB ID.")
        index = self.by_id.get(tmdb_id)
        if index is None:
            return DetailsResult(status="not_found", message="Movie not found in the local catalog.")
        try:
            row = self.catalog.iloc[index]
            data = {field: json_safe(row.get(field)) for field in MovieDetails.model_fields if field not in {"poster_url", "rating_source"}}
            # Empty strings in catalog metadata are absence, not invented facts.
            data = {key: None if value == "" else value for key, value in data.items()}
            path = data.get("poster_path")
            data["poster_url"] = f"https://image.tmdb.org/t/p/w500{path}" if path else None
            return DetailsResult(status="success", movie=MovieDetails.model_validate(data))
        except Exception:
            return DetailsResult(status="error", message="Movie details are temporarily unavailable.")

    def resolve_movie_title(self, title: str) -> ResolutionResult:
        if not isinstance(title, str) or not title.strip():
            return ResolutionResult(status="error", query=title if isinstance(title, str) else "", message="Provide a movie title.")
        try:
            options = self.controller.resolve_title(title)
            if not options:
                return ResolutionResult(status="not_found", query=title, message="Movie not found in the local catalog.")
            if len(options) > 1:
                return ResolutionResult(status="needs_clarification", query=title, options=options, message="Choose a movie identity before retrieving facts.")
            return ResolutionResult(status="success", query=title, movie=options[0])
        except Exception:
            return ResolutionResult(status="error", query=title, message="Movie resolution is temporarily unavailable.")

    def get_watch_providers(self, tmdb_id: int, country_code: str) -> WatchResult:
        region = country_code.strip().upper() if isinstance(country_code, str) else ""
        empty = WatchProviders(country_code=region)
        if type(tmdb_id) is not int or tmdb_id <= 0 or len(region) != 2 or not region.isascii() or not region.isalpha():
            return WatchResult(status="error", tmdb_id=tmdb_id if type(tmdb_id) is int else 0, watch_providers=empty, message="Provide a positive TMDB ID and a two-letter country code.")
        if tmdb_id not in self.by_id:
            return WatchResult(status="not_found", tmdb_id=tmdb_id, watch_providers=empty, message="Movie not found in the local catalog.")
        try:
            providers = WatchProviders.model_validate({**json_safe(self.provider_runner(tmdb_id, region)), "country_code": region})
            available = any((providers.stream, providers.free, providers.ads, providers.rent, providers.buy))
            return WatchResult(status="success" if available else "not_available", tmdb_id=tmdb_id, watch_providers=providers,
                               message=None if available else "Provider information is unavailable for this region.")
        except Exception:
            return WatchResult(status="error", tmdb_id=tmdb_id, watch_providers=empty, message="Provider information is temporarily unavailable.")

    def recommend_movies(self, intent: MovieIntent) -> RecommendationResult:
        try:
            result = self.controller.recommend_from_intent(intent)
            return RecommendationResult.model_validate(json_safe(result))
        except Exception:
            region = (intent.country_code or "SE") if isinstance(intent, MovieIntent) else "SE"
            return RecommendationResult(status="error", country_code=region, message="Unable to process the recommendation request.")
