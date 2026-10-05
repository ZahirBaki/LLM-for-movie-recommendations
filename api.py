from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

import pandas as pd

from recommendation_engine import RecommendationEngine
import os
import requests

from dotenv import load_dotenv

# ==========================================================
# FASTAPI APP
# ==========================================================
# ==========================================================
# ENVIRONMENT
# ==========================================================

load_dotenv()

TMDB_API_KEY = os.getenv(
    "TMDB_API_KEY"
)

TMDB_BASE_URL = (
    "https://api.themoviedb.org/3"
)

app = FastAPI(
    title="Movie Recommender API",
    version="1.1"
)


# ==========================================================
# CORS
#
# Allows our Next.js frontend on port 3000 to communicate
# with the Python backend on port 8000.
# ==========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================
# LOAD RECOMMENDATION ENGINE ONCE
# ==========================================================

print("\nStarting recommendation API...")
print("Loading recommendation engine...\n")

engine = RecommendationEngine()

print("\nRecommendation API ready.\n")


# ==========================================================
# PREPARE MOVIE SEARCH DATA
# ==========================================================

search_movies_df = engine.content.movies[
    [
        "tmdb_id",
        "title",
        "release_year",
        "popularity",
        "poster_path"
    ]
].copy()


search_movies_df["search_title"] = (
    search_movies_df["title"]
    .fillna("")
    .astype(str)
    .str.lower()
)


search_movies_df["popularity"] = pd.to_numeric(
    search_movies_df["popularity"],
    errors="coerce"
).fillna(0)


# ==========================================================
# PREPARE ACTOR / DIRECTOR SEARCH DATA
# ==========================================================

def build_people_index(column_name):
    """
    Build a searchable actor/director index.

    Each person receives:
    - appearance_count: number of catalog movies they appear in
    - popularity_score: summed TMDB popularity of those movies

    These values help rank well-known/relevant people higher
    when several names match the same search.
    """

    people = {}

    for _, row in engine.content.movies.iterrows():

        value = row.get(column_name)

        if pd.isna(value):
            continue

        popularity = pd.to_numeric(
            row.get("popularity", 0),
            errors="coerce"
        )

        if pd.isna(popularity):
            popularity = 0

        for person in str(value).split(","):

            person = person.strip()

            if not person:
                continue

            key = person.lower()

            if key not in people:
                people[key] = {
                    "name": person,
                    "search_name": key,
                    "appearance_count": 0,
                    "popularity_score": 0.0,
                }

            people[key]["appearance_count"] += 1

            people[key]["popularity_score"] += float(
                popularity
            )

    return list(people.values())


actor_search_index = build_people_index(
    "actors"
)


director_search_index = build_people_index(
    "director"
)


# ==========================================================
# PREPARE GENRES
# ==========================================================

genre_set = set()


for value in engine.content.movies["genres"].dropna():

    for genre in str(value).split(","):

        genre = genre.strip()

        if genre:
            genre_set.add(genre)


available_genres = sorted(
    genre_set
)


# ==========================================================
# REQUEST MODEL
# ==========================================================

class FavoriteMovie(BaseModel):
    tmdb_id: int
    title: str
    year: int | None = None


class RecommendationRequest(BaseModel):

    # ------------------------------------------------------
    # FAVORITES / SOFT PREFERENCES
    # ------------------------------------------------------

    favorite_movies: List[FavoriteMovie] = []

    favorite_genres: List[str] = []

    favorite_actors: List[str] = []

    favorite_directors: List[str] = []


    # ------------------------------------------------------
    # HARD REQUIREMENTS
    # ------------------------------------------------------

    required_genres: List[str] = []

    required_actors: List[str] = []

    required_directors: List[str] = []


    # ------------------------------------------------------
    # EXCLUSIONS
    # ------------------------------------------------------

    exclude_genres: List[str] = []

    exclude_actors: List[str] = []

    exclude_directors: List[str] = []


    # ------------------------------------------------------
    # RELEASE YEAR
    # ------------------------------------------------------

    min_year: int | None = None

    max_year: int | None = None

    # ------------------------------------------------------
    # COUNTRY / STREAMING REGION
    # ------------------------------------------------------

    country_code: str = "SE"
    # ------------------------------------------------------
    # NUMBER OF RESULTS
    # ------------------------------------------------------

    top_n: int = 10


# ==========================================================
# HOME
# ==========================================================

@app.get("/")
def home():

    return {
        "message": "Movie Recommender API is running"
    }


# ==========================================================
# MOVIE SEARCH
# ==========================================================

@app.get("/movies/search")
def search_movies(
    q: str = Query(
        min_length=2
    )
):

    query = (
        q.strip().lower()
    )


    # ------------------------------------------------------
    # TITLES STARTING WITH QUERY
    # ------------------------------------------------------

    starts_mask = (
        search_movies_df["search_title"]
        .str.startswith(query)
    )


    starts = search_movies_df.loc[
        starts_mask
    ].copy()


    starts = starts.sort_values(
        "popularity",
        ascending=False
    )


    # ------------------------------------------------------
    # TITLES CONTAINING QUERY
    # ------------------------------------------------------

    if len(starts) < 10:

        contains_mask = (
            search_movies_df["search_title"]
            .str.contains(
                query,
                regex=False
            )
        )


        extra = search_movies_df.loc[
            contains_mask
            & ~starts_mask
        ].copy()


        extra = extra.sort_values(
            "popularity",
            ascending=False
        )


        matches = pd.concat(
            [
                starts,
                extra
            ]
        )

    else:

        matches = starts


    # ------------------------------------------------------
    # EXACT MATCH FIRST
    # ------------------------------------------------------

    matches = (
        matches
        .head(50)
        .copy()
    )


    matches["exact_match"] = (
        matches["search_title"]
        == query
    )


    matches = (
        matches
        .sort_values(
            [
                "exact_match",
                "popularity"
            ],
            ascending=[
                False,
                False
            ]
        )
        .head(10)
    )


    # ------------------------------------------------------
    # RESPONSE
    # ------------------------------------------------------

    results = []


    for row in matches.itertuples():

        year = None


        if pd.notna(
            row.release_year
        ):

            try:

                year = int(
                    row.release_year
                )

            except:

                pass


        poster_url = None


        if (
            pd.notna(row.poster_path)
            and str(row.poster_path).strip()
        ):

            poster_url = (
                "https://image.tmdb.org/t/p/w185"
                + str(row.poster_path)
            )


        results.append(
            {
                "tmdb_id": int(
                    row.tmdb_id
                ),

                "title": str(
                    row.title
                ),

                "year": year,

                "poster_url": poster_url
            }
        )


    return results


# ==========================================================
# ACTOR SEARCH
#
# Matches ANY part of the actor's name.
#
# Examples:
#
#   bale     -> Christian Bale
#   christian -> Christian Bale
#   jackson  -> Samuel L. Jackson
#
# ==========================================================

@app.get("/actors/search")
def search_actors(
    q: str = Query(min_length=2)
):
    query = q.strip().lower()

    if len(query) < 2:
        return []

    ranked_results = []

    for actor in actor_search_index:

        name = actor["search_name"]
        display_name = actor["name"]

        words = name.split()

        if not words:
            continue

        first_name = words[0]
        last_name = words[-1]

        # ==================================================
        # TEXT MATCH RANK
        #
        # Lower = better
        # ==================================================

        if name == query:
            match_rank = 0

        # bale -> Christian Bale
        elif last_name == query:
            match_rank = 1

        # christian -> Christian Bale
        elif first_name == query:
            match_rank = 2

        # bal -> Christian Bale
        elif last_name.startswith(query):
            match_rank = 3

        elif first_name.startswith(query):
            match_rank = 4

        elif any(
            word.startswith(query)
            for word in words
        ):
            match_rank = 5

        elif query in name:
            match_rank = 6

        else:
            continue

        ranked_results.append(
            {
                "name": display_name,
                "match_rank": match_rank,
                "appearance_count":
                    actor["appearance_count"],
                "popularity_score":
                    actor["popularity_score"],
            }
        )

    # ======================================================
    # SORTING
    #
    # 1. Quality of text match
    # 2. Movie popularity
    # 3. Number of movie appearances
    # 4. Alphabetical as final tie breaker
    # ======================================================

    ranked_results.sort(
        key=lambda person: (
            person["match_rank"],
            -person["popularity_score"],
            -person["appearance_count"],
            person["name"].lower(),
        )
    )

    return [
        person["name"]
        for person in ranked_results[:10]
    ]


# ==========================================================
# DIRECTOR SEARCH
#
# Matches ANY part of the director's name.
#
# Examples:
#
#   nolan       -> Christopher Nolan
#   christopher -> Christopher Nolan
#   spielberg   -> Steven Spielberg
#
# ==========================================================

@app.get("/directors/search")
def search_directors(
    q: str = Query(min_length=2)
):
    query = q.strip().lower()

    if len(query) < 2:
        return []

    ranked_results = []

    for director in director_search_index:

        name = director["search_name"]
        display_name = director["name"]

        words = name.split()

        if not words:
            continue

        first_name = words[0]
        last_name = words[-1]

        # ==================================================
        # TEXT MATCH RANK
        # ==================================================

        if name == query:
            match_rank = 0

        elif last_name == query:
            match_rank = 1

        elif first_name == query:
            match_rank = 2

        elif last_name.startswith(query):
            match_rank = 3

        elif first_name.startswith(query):
            match_rank = 4

        elif any(
            word.startswith(query)
            for word in words
        ):
            match_rank = 5

        elif query in name:
            match_rank = 6

        else:
            continue

        ranked_results.append(
            {
                "name": display_name,
                "match_rank": match_rank,
                "appearance_count":
                    director["appearance_count"],
                "popularity_score":
                    director["popularity_score"],
            }
        )

    ranked_results.sort(
        key=lambda person: (
            person["match_rank"],
            -person["popularity_score"],
            -person["appearance_count"],
            person["name"].lower(),
        )
    )

    return [
        person["name"]
        for person in ranked_results[:10]
    ]


# ==========================================================
# GENRES
# ==========================================================

@app.get("/genres")
def get_genres():

    return available_genres

# ==========================================================
# TMDB WATCH PROVIDERS
# ==========================================================

def get_watch_providers(
    tmdb_id,
    country_code="SE"
):

    # ------------------------------------------------------
    # DEFAULT EMPTY RESULT
    # ------------------------------------------------------

    empty_result = {
        "country_code": country_code,
        "stream": [],
        "free": [],
        "ads": [],
        "rent": [],
        "buy": [],
        "link": None,
    }


    if (
        tmdb_id is None
        or not TMDB_API_KEY
    ):
        return empty_result


    # ------------------------------------------------------
    # NORMALIZE COUNTRY CODE
    # ------------------------------------------------------

    country_code = (
        str(country_code)
        .strip()
        .upper()
    )

    if not country_code:
        country_code = "SE"

    empty_result[
        "country_code"
    ] = country_code


    # ------------------------------------------------------
    # TMDB REQUEST
    # ------------------------------------------------------

    url = (
        f"{TMDB_BASE_URL}"
        f"/movie/{int(tmdb_id)}"
        f"/watch/providers"
    )


    try:

        response = requests.get(
            url,
            params={
                "api_key":
                    TMDB_API_KEY
            },
            timeout=5,
        )

        response.raise_for_status()

        data = response.json()


    except requests.RequestException as error:

        print(
            f"Watch provider request failed "
            f"for TMDB {tmdb_id}: {error}"
        )

        return empty_result


    # ------------------------------------------------------
    # GET SELECTED COUNTRY
    # ------------------------------------------------------

    country_data = (
        data
        .get("results", {})
        .get(country_code)
    )


    if not country_data:

        return empty_result


    # ------------------------------------------------------
    # CLEAN PROVIDERS
    # ------------------------------------------------------

    def clean_providers(
        providers
    ):

        cleaned = []

        for provider in (
            providers or []
        ):

            logo_path = provider.get(
                "logo_path"
            )

            logo_url = None

            if logo_path:

                logo_url = (
                    "https://image.tmdb.org"
                    "/t/p/w92"
                    + logo_path
                )


            cleaned.append(
                {
                    "provider_id":
                        provider.get(
                            "provider_id"
                        ),

                    "name":
                        provider.get(
                            "provider_name"
                        ),

                    "logo_url":
                        logo_url,

                    "display_priority":
                        provider.get(
                            "display_priority"
                        ),
                }
            )


        return cleaned


    # ------------------------------------------------------
    # RESPONSE
    # ------------------------------------------------------

    return {

        "country_code":
            country_code,

        # TMDB calls subscription streaming "flatrate"
        "stream":
            clean_providers(
                country_data.get(
                    "flatrate"
                )
            ),

        "free":
            clean_providers(
                country_data.get(
                    "free"
                )
            ),

        "ads":
            clean_providers(
                country_data.get(
                    "ads"
                )
            ),

        "rent":
            clean_providers(
                country_data.get(
                    "rent"
                )
            ),

        "buy":
            clean_providers(
                country_data.get(
                    "buy"
                )
            ),

        "link":
            country_data.get(
                "link"
            ),
    }
# ==========================================================
# RECOMMENDATIONS
# ==========================================================

@app.post("/recommend")
def recommend(
    request: RecommendationRequest
):

    favorite_movies = [
        {
            "tmdb_id": movie.tmdb_id,
            "title": movie.title,
            "year": movie.year,
        }
        for movie in request.favorite_movies
    ]

    recommendations = engine.recommend(

    # ------------------------------------------------------
    # FAVORITES / SOFT PREFERENCES
    # ------------------------------------------------------

    favorite_movies=
        favorite_movies,

    favorite_genres=
        request.favorite_genres,

    favorite_actors=
        request.favorite_actors,

    favorite_directors=
        request.favorite_directors,


    # ------------------------------------------------------
    # HARD REQUIREMENTS
    # ------------------------------------------------------

    required_genres=
        request.required_genres,

    required_actors=
        request.required_actors,

    required_directors=
        request.required_directors,


    # ------------------------------------------------------
    # EXCLUSIONS
    # ------------------------------------------------------

    exclude_genres=
        request.exclude_genres,

    exclude_actors=
        request.exclude_actors,

    exclude_directors=
        request.exclude_directors,


    # ------------------------------------------------------
    # YEAR RANGE
    # ------------------------------------------------------

    min_year=
        request.min_year,

    max_year=
        request.max_year,


    # ------------------------------------------------------
    # RESULTS
    # ------------------------------------------------------

    top_n=
        request.top_n
)


    results = []


    for _, movie in recommendations.iterrows():

        tmdb_id = movie.get(
            "tmdb_id"
        )

        # --------------------------------------------------
        # WATCH PROVIDERS
        # --------------------------------------------------

        watch_providers = (
            get_watch_providers(

                tmdb_id,

                request.country_code
            )
        )
        # --------------------------------------------------
        # GET FULL TMDB METADATA
        # --------------------------------------------------

        movie_info = None


        if (
            tmdb_id is not None
            and pd.notna(tmdb_id)
        ):

            match = engine.content.movies[
                pd.to_numeric(
                    engine.content.movies[
                        "tmdb_id"
                    ],
                    errors="coerce"
                )
                == int(tmdb_id)
            ]


            if not match.empty:

                movie_info = (
                    match.iloc[0]
                )


        # --------------------------------------------------
        # DEFAULT VALUES
        # --------------------------------------------------

        title = movie.get(
            "title",
            "Unknown"
        )


        year = movie.get(
            "release_year"
        )


        genres = movie.get(
            "genres",
            ""
        )


        rating = None

        overview = ""

        poster_url = None


        # --------------------------------------------------
        # TMDB METADATA
        # --------------------------------------------------

        if movie_info is not None:

            title = movie_info.get(
                "title",
                title
            )


            year = movie_info.get(
                "release_year",
                year
            )


            genres = movie_info.get(
                "genres",
                genres
            )


            rating = movie_info.get(
                "vote_average"
            )


            overview = movie_info.get(
                "overview",
                ""
            )


            poster_path = (
                movie_info.get(
                    "poster_path"
                )
            )


            if (
                poster_path is not None
                and pd.notna(
                    poster_path
                )
                and str(
                    poster_path
                ).strip()
            ):

                poster_url = (
                    "https://image.tmdb.org/t/p/w500"
                    + str(poster_path)
                )


        # --------------------------------------------------
        # CLEAN YEAR
        # --------------------------------------------------

        clean_year = None


        if (
            year is not None
            and pd.notna(year)
        ):

            try:

                clean_year = int(
                    year
                )

            except:

                pass


        # --------------------------------------------------
        # CLEAN RATING
        # --------------------------------------------------

        clean_rating = None


        if (
            rating is not None
            and pd.notna(rating)
        ):

            try:

                clean_rating = float(
                    rating
                )

            except:

                pass


        # --------------------------------------------------
        # CLEAN SCORES
        # --------------------------------------------------

        def clean_score(
            value
        ):

            if (
                value is None
                or pd.isna(value)
            ):

                return None


            try:

                return float(
                    value
                )

            except:

                return None


        # --------------------------------------------------
        # ADD RESULT
        # --------------------------------------------------

        results.append(
            {
                "tmdb_id": (
                    int(tmdb_id)

                    if (
                        tmdb_id is not None
                        and pd.notna(tmdb_id)
                    )

                    else None
                ),

                "title": str(
                    title
                ),

                "year":
                    clean_year,

                "genres": (
                    str(genres)

                    if (
                        genres is not None
                        and not pd.isna(genres)
                    )

                    else ""
                ),

                "rating":
                    clean_rating,

                "overview": (
                    str(overview)

                    if (
                        overview is not None
                        and not pd.isna(overview)
                    )

                    else ""
                ),

                "poster_url":
                    poster_url,
                "watch_providers":
                    watch_providers,
                "recommendation_source":
                    str(
                        movie.get(
                            "recommendation_source",
                            ""
                        )
                    ),

                "engine_mode":
                    str(
                        movie.get(
                            "engine_mode",
                            ""
                        )
                    ),

                "content_score":
                    clean_score(
                        movie.get(
                            "content_score"
                        )
                    ),

                "svd_score":
                    clean_score(
                        movie.get(
                            "svd_score"
                        )
                    ),

                "final_score":
                    clean_score(
                        movie.get(
                            "final_score"
                        )
                    )
            }
        )


    return results
