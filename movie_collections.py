"""TMDB-owned collection membership; no inferred or hardcoded franchise groups."""
from datetime import date
import math
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from chat_recommender import compact_title

COLLECTION_SCOPE = "TMDB collection membership only; a collection may not include every film in a broader franchise. No broader franchise membership has been verified."
STORY_LIMITATION = "Story chronology is not represented in this data. This list uses release dates, not story order."


class CollectionModel(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)


class CollectionMatch(CollectionModel):
    collection_id: int
    name: str


class CollectionSearchResult(CollectionModel):
    status: Literal["success", "needs_clarification", "not_found", "error"]
    query: str
    matches: list[CollectionMatch] = Field(default_factory=list)
    resolved_collection_id: int | None = None
    total_results: int = 0
    truncated: bool = False
    source: Literal["TMDB"] = "TMDB"
    coverage: Literal["tmdb_collection_only"] = "tmdb_collection_only"
    complete_franchise_verified: bool = False
    scope_note: str = COLLECTION_SCOPE
    message: str | None = None


class CollectionMovie(CollectionModel):
    tmdb_id: int
    title: str
    release_date: str | None = None
    release_year: int | None = None
    poster_path: str | None = None
    poster_url: str | None = None
    vote_average: float | None = None


class CollectionResult(CollectionModel):
    status: Literal["success", "not_found", "error"]
    collection_id: int
    name: str | None = None
    movies: list[CollectionMovie] = Field(default_factory=list)
    total_movies: int = 0
    truncated: bool = False
    missing_release_dates: int = 0
    source: Literal["TMDB"] = "TMDB"
    source_url: str | None = None
    coverage: Literal["tmdb_collection_only"] = "tmdb_collection_only"
    complete_franchise_verified: bool = False
    scope_note: str = COLLECTION_SCOPE
    order: Literal["release_date"] = "release_date"
    story_order_supported: bool = False
    chronology_note: str = STORY_LIMITATION
    message: str | None = None


def collection_key(name):
    return compact_title(re.sub(r"\s+collection\s*$", "", name, flags=re.I))


class MovieCollections:
    def __init__(self, request_runner):
        # runner(path, params) -> dict or None (404); injected in offline tests.
        self.request_runner = request_runner

    def search_movie_collection(self, query: str, limit: int = 5) -> CollectionSearchResult:
        if not isinstance(query, str) or not 2 <= len(query.strip()) <= 300 or type(limit) is not int or not 1 <= limit <= 10:
            return CollectionSearchResult(status="error", query=query if isinstance(query, str) else "", message="Provide a collection query of 2–300 characters and a limit of 1–10.")
        try:
            data = self.request_runner("/search/collection", {"query": query.strip(), "page": 1, "include_adult": "false"})
            if data is None:
                return CollectionSearchResult(status="not_found", query=query)
            raw = data["results"]
            total = int(data["total_results"])
            matches = list({item["id"]: CollectionMatch(collection_id=item["id"], name=item["name"][:300])
                            for item in raw if type(item.get("id")) is int and item["id"] > 0}.values())
            exact = [item for item in matches if collection_key(item.name) == collection_key(query)]
            selected = exact[0] if len(exact) == 1 and total <= len(raw) else (matches[0] if len(matches) == total == 1 else None)
            # Preserve a resolved exact match even if it was outside the display bound.
            if selected:
                matches = [selected] + [item for item in matches if item != selected]
            return CollectionSearchResult(status="success" if selected else ("needs_clarification" if matches else "not_found"),
                query=query, matches=matches[:limit], resolved_collection_id=selected.collection_id if selected else None,
                total_results=total, truncated=total > min(len(matches), limit))
        except Exception:
            return CollectionSearchResult(status="error", query=query, message="Collection search is temporarily unavailable.")

    def get_movie_collection(self, collection_id: int, limit: int = 50) -> CollectionResult:
        if type(collection_id) is not int or collection_id <= 0 or type(limit) is not int or not 1 <= limit <= 100:
            return CollectionResult(status="error", collection_id=collection_id if type(collection_id) is int else 0, message="Provide a positive collection ID and a limit of 1–100.")
        try:
            data = self.request_runner(f"/collection/{collection_id}", {})
            if data is None:
                return CollectionResult(status="not_found", collection_id=collection_id, message="Collection not found.")
            if data["id"] != collection_id:
                raise ValueError("Mismatched collection identity")
            movies = {}
            for part in data["parts"]:
                movie_date = None
                try:
                    candidate = part.get("release_date")
                    if isinstance(candidate, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", candidate):
                        movie_date = date.fromisoformat(candidate).isoformat()
                except ValueError:
                    pass
                path = part.get("poster_path")
                path = path if isinstance(path, str) and path.startswith("/") else None
                rating = part.get("vote_average")
                rating = rating if type(rating) in {int, float} and math.isfinite(rating) and 0 <= rating <= 10 else None
                if type(part.get("id")) is not int or part["id"] <= 0 or not isinstance(part.get("title"), str):
                    raise ValueError("Invalid collection member")
                movies.setdefault(part["id"], CollectionMovie(tmdb_id=part["id"], title=part["title"][:300],
                    release_date=movie_date, release_year=int(movie_date[:4]) if movie_date else None,
                    poster_path=path, poster_url=f"https://image.tmdb.org/t/p/w500{path}" if path else None, vote_average=rating))
            # Unknown/invalid dates sort last and never acquire invented chronology.
            ordered = sorted(movies.values(), key=lambda movie: (movie.release_date is None, movie.release_date or "", movie.tmdb_id))
            return CollectionResult(status="success", collection_id=collection_id, name=data["name"][:300],
                movies=ordered[:limit], total_movies=len(ordered), truncated=len(ordered) > limit,
                missing_release_dates=sum(movie.release_date is None for movie in ordered),
                source_url=f"https://www.themoviedb.org/collection/{collection_id}")
        except Exception:
            return CollectionResult(status="error", collection_id=collection_id, message="Collection details are temporarily unavailable.")


def collection_response_text(result, user_message: str = ""):
    """Trusted formatting prevents model-added membership, dates and ordering."""
    if isinstance(result, CollectionSearchResult):
        if result.status == "needs_clarification":
            names = "; ".join(match.name for match in result.matches)
            suffix = " More matches exist; try a more specific collection name." if result.truncated else ""
            return f"I found several collections with that name: {names}. Which one do you mean?{suffix}"
        if result.status == "not_found":
            return "I couldn't find a movie collection with that name. Try a more specific series name."
        return result.message or "Please specify a collection name."
    if result.status != "success":
        return result.message or "I couldn't retrieve that collection."
    request = user_message.casefold()
    name = re.sub(r"\s+collection\s*$", "", result.name or "this collection", flags=re.I)
    story_order = bool(re.search(r"\bchronolog\w*\b|\bstory\b.{0,30}\border\b|\btimeline\b", request))
    exact_dates = bool(re.search(r"\brelease dates?\b|\b(?:exact|full|specific)\b.{0,30}\bdates?\b", request))
    source_requested = bool(re.search(r"\b(?:source|sources|tmdb|provenance)\b|\bwhere\b.{0,60}\b(?:get|got|from)\b", request))
    broad_request = bool(re.search(r"\b(?:all|every|entire|complete|including|whole)\b", request))
    if story_order and not result.story_order_supported:
        lines = [f"I can give you the {name} movies in release order, but I don't have a reliable source for their story chronology:"]
    else:
        lines = [f"Sure — here are the movies in the {name} collection in release order:"]
    if broad_request and not result.complete_franchise_verified:
        lines.append("This covers the collection, but I can't verify every related film in the wider series from this list.")
    lines.append("")
    for index, movie in enumerate(result.movies, 1):
        released = (movie.release_date or "release date unavailable") if exact_dates else (movie.release_year or "release year unavailable")
        lines.append(f"{index}. {movie.title} ({released})")
    if not result.movies:
        lines.append("I couldn't find any movies in this collection.")
    if result.truncated:
        lines.append(f"Showing {len(result.movies)} of {result.total_movies} movies in this collection.")
    if result.missing_release_dates:
        lines.append("Some release dates are missing, so those movies appear last; I can't confirm their order.")
    if source_requested:
        lines.append("The collection and release information come from TMDB.")
    return "\n".join(lines)
