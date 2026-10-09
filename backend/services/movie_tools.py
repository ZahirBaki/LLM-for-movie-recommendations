"""OpenAI-independent movie capabilities, bound once to existing API services."""

import math
import re
from datetime import date
from collections.abc import Callable
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from backend.services.chat_recommender import ChatRecommender, IntentRecommendationResult, ResolvedMovie, MovieClarification, _normalize
from backend.services.chat_service import MovieIntent
from backend.services.movie_collections import MovieCollections, CollectionSearchResult, CollectionResult


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
    details: MovieDetails | None = None


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
    external_movies: list[MovieDetails] = Field(default_factory=list)


class MovieTools:
    """Bind the existing search, resolver, enrichment and ranking services.

    No globals, model loading, network calls or LLM calls occur at construction.
    api.movie_tools is the shared application instance. Tests can inject fixtures.
    """

    def __init__(self, catalog: pd.DataFrame, controller: ChatRecommender,
                 search_runner: Callable[[str], list[dict]], provider_runner: Callable[[int, str], dict], collection_runner=None, live_movie_runner=None, external_embeddings=None):
        self.catalog = catalog
        self.controller = controller
        self.search_runner = search_runner
        self.provider_runner = provider_runner
        self.external_embeddings = external_embeddings
        self.live_movie_runner = live_movie_runner
        self.collections = MovieCollections(collection_runner or self._collections_unavailable)
        self.by_id = {}
        for index, row in enumerate(catalog.itertuples()):
            if pd.notna(row.tmdb_id):
                self.by_id.setdefault(int(row.tmdb_id), index)

    @staticmethod
    def _collections_unavailable(path, params):
        raise RuntimeError("Collection data unavailable")

    def search_movie_collection(self, query: str, limit: int = 5) -> CollectionSearchResult:
        return self.collections.search_movie_collection(query, limit)

    def get_movie_collection(self, collection_id: int, limit: int = 50) -> CollectionResult:
        return self.collections.get_movie_collection(collection_id, limit)

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
            cached = self.external_embeddings.cached(tmdb_id) if self.external_embeddings else None
            if cached:
                return DetailsResult(status="success", movie=MovieDetails.model_validate(cached.metadata))
            if self.live_movie_runner:
                try:
                    raw = self.live_movie_runner(f"/movie/{tmdb_id}", {"append_to_response": "credits"})
                    if raw is not None:
                        return DetailsResult(status="success", movie=self._live_details(raw, tmdb_id))
                except Exception:
                    return DetailsResult(status="error", message="Live TMDB movie details are temporarily unavailable.")
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

    def resolve_movie_title(self, title: str, year: int | None = None) -> ResolutionResult:
        if not isinstance(title, str) or not title.strip():
            return ResolutionResult(status="error", query=title if isinstance(title, str) else "", message="Provide a movie title.")
        if year is not None and (type(year) is not int or not 1 <= year <= 9999):
            return ResolutionResult(status="error", query=title, message="Use a valid integer release year.")
        try:
            options = self.controller.resolve_title(title) if year is None else self.controller.resolve_title(title, year)
            # Literal catalog titles ending in digits are not year qualifiers.
            query, requested_year = title.strip(), year
            if _normalize(query) not in self.controller.titles:
                qualified = re.fullmatch(r"(.+?)(?:\s*\((\d{4})\)|\s+(\d{4}))\s*", query)
                if qualified:
                    query = qualified[1]
                    parsed_year = int(qualified[2] or qualified[3])
                    if year is not None and year != parsed_year:
                        return ResolutionResult(status="not_found", query=title, message="Conflicting release years.")
                    requested_year = parsed_year
            if not options and requested_year is not None and self.external_embeddings:
                cached = self.external_embeddings.find(query, requested_year)
                if cached:
                    return ResolutionResult(status="success", query=title,
                        movie=ResolvedMovie(tmdb_id=cached.tmdb_id, title=cached.title, year=cached.year),
                        details=MovieDetails.model_validate(cached.metadata))
            # Preserve confident exact identities and explicit local year matches.
            # Low-recognition unqualified titles may hide a newer absent namesake.
            confident_local = len(options) == 1 and self.controller.confident_exact_match(query, options[0])
            if self.live_movie_runner and (not options or (len(options) == 1 and requested_year is None and not confident_local)):
                return self._resolve_live(query, requested_year, title, options)
            if not options:
                return ResolutionResult(status="not_found", query=title, message="Movie not found in the local catalog.")
            if len(options) > 1:
                return ResolutionResult(status="needs_clarification", query=title, options=options, message="Choose a movie identity before retrieving facts.")
            return ResolutionResult(status="success", query=title, movie=options[0])
        except Exception:
            return ResolutionResult(status="error", query=title, message="Movie resolution is temporarily unavailable.")

    @staticmethod
    def _live_details(raw, expected_id):
        if not isinstance(raw, dict) or type(raw.get("id")) is not int or raw["id"] != expected_id:
            raise ValueError("TMDB identity mismatch")
        title = raw.get("title")
        if not isinstance(title, str) or not title.strip() or len(title) > 300:
            raise ValueError("Missing TMDB title")
        release = raw.get("release_date")
        year = date.fromisoformat(release).year if release else None
        credits = raw.get("credits") or {}
        crew, cast = credits.get("crew", []), credits.get("cast", [])
        path = raw.get("poster_path")
        return MovieDetails.model_validate(json_safe({
            "tmdb_id": expected_id, "title": title, "original_title": raw.get("original_title"),
            "release_date": release or None, "release_year": year,
            "director": ", ".join(p["name"] for p in crew[:1000] if p.get("job") == "Director") or None,
            "actors": ", ".join(p["name"] for p in cast[:10]) or None,
            "genres": ", ".join(g["name"] for g in raw.get("genres", [])[:30]) or None,
            "overview": (raw.get("overview") or "")[:6000] or None,
            "original_language": raw.get("original_language"), "popularity": raw.get("popularity"),
            "vote_average": raw.get("vote_average"), "vote_count": raw.get("vote_count"),
            "runtime": raw.get("runtime"), "poster_path": path,
            "poster_url": f"https://image.tmdb.org/t/p/w500{path}" if path else None}))

    def _resolve_live(self, query, year, original, local_options):
        try:
            params = {"query": query, "page": 1, "include_adult": "false"}
            if year is not None:
                params["year"] = year
            raw = self.live_movie_runner("/search/movie", params)
            if not isinstance(raw, dict) or not isinstance(raw.get("results"), list):
                raise ValueError("Invalid TMDB search")
            matches = {movie.tmdb_id: movie for movie in local_options}
            for candidate in raw["results"][:20]:
                if not isinstance(candidate, dict) or type(candidate.get("id")) is not int or candidate["id"] <= 0:
                    continue
                names = [candidate.get("title"), candidate.get("original_title")]
                if not any(isinstance(name, str) and _normalize(name) == _normalize(query) for name in names):
                    continue  # TMDB search also returns fuzzy/unrelated titles.
                release = candidate.get("release_date")
                candidate_year = date.fromisoformat(release).year if release else None
                if year is not None and candidate_year != year:
                    continue
                matches[candidate["id"]] = ResolvedMovie(tmdb_id=candidate["id"], title=candidate["title"], year=candidate_year)
            options = list(matches.values())
            if len(options) > 1 or (options and raw.get("total_pages", 1) > 1):
                return ResolutionResult(status="needs_clarification", query=original, options=options[:5],
                    message="Several title matches exist. Please choose a title and release year; only the first search page was checked.")
            if not options:
                return ResolutionResult(status="not_found", query=original, message="No exact title/year match was found locally or in the bounded TMDB search.")
            movie = options[0]
            if movie.tmdb_id in self.by_id:
                return ResolutionResult(status="success", query=original, movie=movie)
            details = self.get_movie_details(movie.tmdb_id)
            if details.status != "success":
                return ResolutionResult(status="error", query=original, message=details.message)
            verified = details.movie
            names = [verified.title, verified.original_title]
            if not any(name and _normalize(name) == _normalize(query) for name in names) or verified.release_year != movie.year:
                return ResolutionResult(status="not_found", query=original, message="TMDB search and detailed title/year identity did not agree.")
            return ResolutionResult(status="success", query=original, movie=movie, details=verified)
        except Exception:
            if local_options:
                return ResolutionResult(status="needs_clarification", query=original, options=local_options,
                    message="Live TMDB verification is unavailable. Confirm the local movie's release year before using it.")
            return ResolutionResult(status="error", query=original, message="Live TMDB title lookup is temporarily unavailable; the local catalog has no matching identity.")

    def get_watch_providers(self, tmdb_id: int, country_code: str) -> WatchResult:
        region = country_code.strip().upper() if isinstance(country_code, str) else ""
        empty = WatchProviders(country_code=region)
        if type(tmdb_id) is not int or tmdb_id <= 0 or len(region) != 2 or not region.isascii() or not region.isalpha():
            return WatchResult(status="error", tmdb_id=tmdb_id if type(tmdb_id) is int else 0, watch_providers=empty, message="Provide a positive TMDB ID and a two-letter country code.")
        if tmdb_id not in self.by_id:
            details = self.get_movie_details(tmdb_id)
            if details.status != "success":
                return WatchResult(status=details.status, tmdb_id=tmdb_id, watch_providers=empty, message=details.message)
        try:
            providers = WatchProviders.model_validate({**json_safe(self.provider_runner(tmdb_id, region)), "country_code": region})
            available = any((providers.stream, providers.free, providers.ads, providers.rent, providers.buy))
            return WatchResult(status="success" if available else "not_available", tmdb_id=tmdb_id, watch_providers=providers,
                               message=None if available else "Provider information is unavailable for this region.")
        except Exception:
            return WatchResult(status="error", tmdb_id=tmdb_id, watch_providers=empty, message="Provider information is temporarily unavailable.")

    def recommend_movies(self, intent: MovieIntent, *, verified_movies: list[ResolvedMovie] | None = None) -> RecommendationResult:
        try:
            # Resolve every seed before scoring: never rank from a partial subset
            # or silently substitute an older local namesake.
            if self.live_movie_runner and intent.intent == "recommend" and intent.favorite_movies:
                if len(intent.favorite_movies) > 20:
                    return RecommendationResult(status="error", country_code=intent.country_code or "SE",
                        message="Please provide at most twenty movie seeds per request.")
                result = RecommendationResult(status="success", country_code=intent.country_code or "SE")
                verified_titles = []
                for title in dict.fromkeys(intent.favorite_movies[:20]):
                    known = [movie for movie in (verified_movies or []) if title == (f"{movie.title} ({movie.year})" if movie.year else movie.title)]
                    resolutions = [ResolutionResult(status="success", query=title, movie=movie) for movie in known] if known else [self.resolve_movie_title(title)]
                    for resolution in resolutions:
                        movie = resolution.movie
                        if movie and movie.tmdb_id not in self.by_id and not resolution.details:
                            details = self.get_movie_details(movie.tmdb_id)
                            if details.movie and details.movie.title == movie.title and details.movie.release_year == movie.year:
                                resolution.details = details.movie
                            else:
                                resolution = ResolutionResult(status="error", query=title, message="The previously verified movie identity could not be confirmed.")
                        if resolution.status == "needs_clarification":
                            result.clarifications.append(MovieClarification(query=title, options=resolution.options))
                        elif resolution.status != "success":
                            result.unresolved_movies.append(title)
                            result.message = resolution.message
                            if resolution.status == "error":
                                result.status = "error"
                        else:
                            result.resolved_movies.append(resolution.movie)
                            verified_titles.append(f"{resolution.movie.title} ({resolution.movie.year})" if resolution.movie.year else resolution.movie.title)
                            if resolution.details and resolution.movie.tmdb_id not in self.by_id:
                                result.external_movies.append(resolution.details)
                result.resolved_movies = list({movie.tmdb_id: movie for movie in result.resolved_movies}.values())
                result.external_movies = list({movie.tmdb_id: movie for movie in result.external_movies}.values())
                if result.clarifications:
                    result.status = "needs_clarification"
                    result.message = "Please choose the movie title and year before I recommend similar movies."
                    return result
                if result.unresolved_movies:
                    result.status = "error" if result.status == "error" else "unresolved"
                    return result
                if result.external_movies:
                    from backend.recommendation.runtime_movie_embeddings import ExternalEmbeddingError
                    result.status = "needs_preferences"
                    if self.external_embeddings is None:
                        result.message = "The recommendation engine cannot score external movies because the runtime embedding component is unavailable; the engine requires a compatible catalog-row embedding or runtime query vector."
                        return result
                    try:
                        queries = [self.external_embeddings.get(movie) for movie in result.external_movies]
                    except ExternalEmbeddingError as exc:
                        result.message = str(exc)
                        return result
                    from backend.services.chat_recommender import ENGINE_FIELDS
                    inputs = {field: getattr(intent, field) for field in ENGINE_FIELDS}
                    if inputs["top_n"] is None:
                        inputs.pop("top_n")
                    inputs.update(country_code=result.country_code,
                        favorite_movies=[movie.model_dump() for movie in result.resolved_movies if movie.tmdb_id in self.by_id],
                        external_queries=queries)
                    result.recommendations = self.controller.recommendation_runner(**inputs)
                    result.status = "success"
                    names = ", ".join(f"{movie.title} ({movie.year})" for movie in result.resolved_movies)
                    result.message = (f"Since you enjoyed {names}, here are movies you might like:" if result.recommendations
                        else "I couldn't find catalog movies matching those preferences.")
                    return result
                intent = intent.model_copy(update={"favorite_movies": verified_titles})
            result = self.controller.recommend_from_intent(intent, verified_movies=verified_movies)
            return RecommendationResult.model_validate(json_safe(result))
        except Exception:
            region = (intent.country_code or "SE") if isinstance(intent, MovieIntent) else "SE"
            return RecommendationResult(status="error", country_code=region, message="Unable to process the recommendation request.")
