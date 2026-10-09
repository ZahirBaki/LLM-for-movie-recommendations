"""Translate validated intents into the existing recommendation API flow.

No OpenAI calls, ranking, catalog mutation, or model initialization occurs here.
The injected runner accepts RecommendationRequest-compatible keyword data and
returns the existing movie-card list, including watch-provider enrichment.
"""

import re
import unicodedata
from collections.abc import Callable
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from backend.services.chat_service import MovieIntent

# A duplicate title can resolve automatically only if both catalog signals show
# overwhelming recognition. This is entity resolution, never recommendation scoring.
MIN_DOMINANT_VOTES = 1000
VOTE_DOMINANCE_RATIO = 10
POPULARITY_DOMINANCE_RATIO = 5

ENGINE_FIELDS = (
    "favorite_genres", "favorite_actors", "favorite_directors",
    "required_genres", "required_actors", "required_directors",
    "exclude_genres", "exclude_actors", "exclude_directors",
    "min_year", "max_year", "top_n",
)


class ResolvedMovie(BaseModel):
    tmdb_id: int
    title: str
    year: int | None = None


class MovieClarification(BaseModel):
    type: Literal["movie"] = "movie"
    query: str
    options: list[ResolvedMovie]


class IntentRecommendationResult(BaseModel):
    status: Literal["success", "needs_clarification", "unresolved", "unsupported_intent", "needs_preferences"]
    message: str | None = None
    resolved_movies: list[ResolvedMovie] = Field(default_factory=list)
    clarifications: list[MovieClarification] = Field(default_factory=list)
    unresolved_movies: list[str] = Field(default_factory=list)
    recommendations: list[dict[str, Any]] = Field(default_factory=list)
    unsupported_preferences: dict[str, list[str]] = Field(default_factory=dict)
    country_code: str


def _normalize(title: str) -> str:
    title = unicodedata.normalize("NFKC", title)
    title = "".join("-" if unicodedata.category(char) == "Pd" else char for char in title)
    return " ".join(title.split()).casefold()


def compact_title(title: str) -> str:
    """Exact separator-equivalence fallback, never substring/edit-distance matching."""
    return re.sub(r"[\s-]+", "", _normalize(title))


class ChatRecommender:
    def __init__(self, catalog: pd.DataFrame, recommendation_runner: Callable[..., list[dict[str, Any]]]):
        self.recommendation_runner = recommendation_runner
        # Read-only resolution snapshot; built once for the loaded API engine.
        self.titles: dict[str, list[dict[str, Any]]] = {}
        self.aliases: dict[str, list[dict[str, Any]]] = {}
        for row in catalog.to_dict("records"):
            if pd.isna(row.get("tmdb_id")) or pd.isna(row.get("title")):
                continue
            year = row.get("release_year")
            movie = {
                "tmdb_id": int(row["tmdb_id"]), "title": str(row["title"]),
                "year": int(year) if pd.notna(year) else None,
                "vote_count": self._number(row.get("vote_count")),
                "popularity": self._number(row.get("popularity")),
            }
            self.titles.setdefault(_normalize(movie["title"]), []).append(movie)
            for name in {movie["title"], row.get("original_title")}:
                if isinstance(name, str) and name.strip():
                    self.aliases.setdefault(compact_title(name), []).append(movie)

    @staticmethod
    def _number(value) -> float:
        return float(value) if value is not None and pd.notna(value) else 0.0

    def resolve_title(self, query: str, year: int | None = None) -> list[ResolvedMovie]:
        # First preserve literal titles such as "1917" or a title ending in digits.
        matches = self.titles.get(_normalize(query), [])
        explicit_year = year is not None
        alias_match = False
        if not matches:
            qualified = re.fullmatch(r"(.+?)(?:\s*\((\d{4})\)|\s+(\d{4}))\s*", query.strip())
            if qualified:
                explicit_year = True
                title, parenthesized, trailing = qualified.groups()
                parsed_year = int(parenthesized or trailing)
                if year is not None and year != parsed_year:
                    return []
                query, year = title, parsed_year
                matches = self.titles.get(_normalize(query), [])
            if not matches:
                alias_match = True
                matches = self.aliases.get(compact_title(query), [])
        if year is not None:
            matches = [movie for movie in matches if movie["year"] == year]

        # Duplicate catalog rows do not constitute separate movie identities.
        matches = list({movie["tmdb_id"]: movie for movie in matches}.values())
        matches.sort(key=lambda movie: (-movie["vote_count"], -movie["popularity"], movie["tmdb_id"]))
        if len(matches) > 1 and not explicit_year and not alias_match:
            best, others = matches[0], matches[1:]
            if (
                best["vote_count"] >= MIN_DOMINANT_VOTES
                and best["popularity"] > 0
                and all(
                    best["vote_count"] >= VOTE_DOMINANCE_RATIO * max(other["vote_count"], 1)
                    and best["popularity"] >= POPULARITY_DOMINANCE_RATIO * max(other["popularity"], 0.01)
                    for other in others
                )
            ):
                matches = [best]
        return [ResolvedMovie(**{key: movie[key] for key in ("tmdb_id", "title", "year")}) for movie in matches]

    def confident_exact_match(self, query: str, movie: ResolvedMovie) -> bool:
        """Recognition evidence for an exact local title, never a ranking weight."""
        rows = self.titles.get(_normalize(query), [])
        return bool(rows and self.resolve_title(query) == [movie] and any(
            row["tmdb_id"] == movie.tmdb_id and row["vote_count"] >= MIN_DOMINANT_VOTES
            and row["popularity"] > 0 for row in rows))

    def recommend_from_intent(self, intent: MovieIntent, *, verified_movies: list[ResolvedMovie] | None = None) -> IntentRecommendationResult:
        # Revalidate even if a caller used model_construct or mutated the model.
        intent = MovieIntent.model_validate(intent.model_dump())
        unsupported = {}
        for field in ("natural_language_preferences", "movie_titles"):
            if getattr(intent, field):
                unsupported[field] = list(getattr(intent, field))
        result = IntentRecommendationResult(
            status="success", unsupported_preferences=unsupported,
            country_code=intent.country_code or "SE",
        )
        if intent.intent != "recommend":
            result.status = "unsupported_intent"
            result.message = "Only recommendation intents are supported by this controller."
            return result

        # Session identities must still match a catalog ID, title and year.
        verified_movies = [movie for movie in (verified_movies or [])
            if any(row["tmdb_id"] == movie.tmdb_id and row["year"] == movie.year
                for row in self.titles.get(_normalize(movie.title), []))]
        seen_ids = set()
        for query in dict.fromkeys(intent.favorite_movies):
            verified_candidates = [movie for movie in verified_movies if query == (f"{movie.title} ({movie.year})" if movie.year else movie.title)]
            if verified_candidates:
                for movie in verified_candidates:
                    if movie.tmdb_id not in seen_ids:
                        result.resolved_movies.append(movie)
                        seen_ids.add(movie.tmdb_id)
                continue
            candidates = self.resolve_title(query)
            if not candidates:
                result.unresolved_movies.append(query)
            elif len(candidates) > 1:
                result.clarifications.append(MovieClarification(query=query, options=candidates))
            elif candidates[0].tmdb_id not in seen_ids:
                result.resolved_movies.append(candidates[0])
                seen_ids.add(candidates[0].tmdb_id)

        # Never recommend from only the successfully resolved subset.
        if result.clarifications or result.unresolved_movies:
            result.status = "needs_clarification" if result.clarifications else "unresolved"
            return result

        if not any((result.resolved_movies, intent.favorite_genres, intent.favorite_actors, intent.favorite_directors)):
            result.status = "needs_preferences"
            result.message = "The existing engine requires a reference movie or a favorite genre, actor, or director to rank results. Required filters alone are insufficient."
            return result

        inputs = {field: getattr(intent, field) for field in ENGINE_FIELDS}
        if inputs["top_n"] is None:
            inputs.pop("top_n")  # Preserve the existing request's default count.
        inputs["favorite_movies"] = [movie.model_dump() for movie in result.resolved_movies]
        inputs["country_code"] = result.country_code  # Enrichment only, via the existing API flow.
        result.recommendations = self.recommendation_runner(**inputs)
        return result
