import pandas as pd

from content_recommender import ContentRecommender
from svd_recommender import SVDRecommender


# ==========================================================
# SETTINGS
# ==========================================================

DEFAULT_TOP_N = 10

# Minimum number of favorite movies that must be usable
# by SVD before SVD becomes the primary recommendation model.
MIN_SVD_FAVORITES = 2

SVD_CANDIDATE_POOL = 500

SOFT_CONTENT_BONUS = 0.10


# ==========================================================
# COLD-ITEM CONTENT CANDIDATES
# ==========================================================
#
# Movies unavailable to SVD (for example new 2025/2026
# releases) can still enter through the content model.
#
# We only consider a small number of the strongest content
# candidates. They are not treated as if SVD disliked them;
# they simply have no collaborative score.

CONTENT_COLD_CANDIDATE_POOL = 50

# A cold movie must be among the strongest content matches
# relative to the current request before it can compete.
#
# We use rank rather than pretending that content_score and
# svd_score are directly comparable.

COLD_CONTENT_MAX_RANK = 10


# ==========================================================
# RECOMMENDATION ENGINE
# ==========================================================

class RecommendationEngine:

    def __init__(self):

        print("\nInitializing recommendation engine...")

        # --------------------------------------------------
        # CONTENT MODEL
        # --------------------------------------------------

        self.content = ContentRecommender()

        # --------------------------------------------------
        # SVD MODEL
        # --------------------------------------------------

        self.svd = SVDRecommender()
        self.svd.load_model()

        print("\nRecommendation engine ready.")


    # ======================================================
    # FIND FAVORITE TMDB IDS
    # ======================================================

    def get_favorite_tmdb_ids(
        self,
        favorite_movies
    ):

        tmdb_ids = []

        for movie in favorite_movies:
            if not isinstance(movie, dict):
                continue

            tmdb_id = movie.get("tmdb_id")
            if tmdb_id is None:
                continue

            try:
                tmdb_ids.append(int(tmdb_id))
            except (TypeError, ValueError):
                continue

        return tmdb_ids


    # ======================================================
    # GET CONTENT SCORES
    # ======================================================

    def get_content_scores(
        self,
        favorite_movies,
        favorite_tmdb_ids,
        favorite_genres,
        favorite_actors,
        favorite_directors
    ):

        return self.content.get_scores(

            favorite_movies=favorite_movies,

            favorite_tmdb_ids=favorite_tmdb_ids,

            favorite_genres=favorite_genres,

            favorite_actors=favorite_actors,

            favorite_directors=favorite_directors,

            top_n=None
        )


    # ======================================================
    # COUNT FAVORITES USABLE BY SVD
    # ======================================================

    def get_usable_svd_favorites(
        self,
        favorite_tmdb_ids
    ):

        if not favorite_tmdb_ids:
            return []

        usable = []

        mapping = self.svd.mapping.copy()

        if "tmdb_id" not in mapping.columns:
            return usable

        mapping = (
            mapping
            .dropna(subset=["tmdb_id"])
            .copy()
        )

        mapping["tmdb_id"] = (
            mapping["tmdb_id"]
            .astype(int)
        )

        if hasattr(
            self.svd,
            "movie_to_index"
        ):

            usable_movie_ids = set(
                self.svd.movie_to_index.keys()
            )

            mapping = mapping[
                mapping["movieId"]
                .isin(usable_movie_ids)
            ]

        available_tmdb_ids = set(
            mapping["tmdb_id"]
            .tolist()
        )

        for tmdb_id in favorite_tmdb_ids:

            if tmdb_id in available_tmdb_ids:
                usable.append(tmdb_id)

        return usable


    # ======================================================
    # ADD TMDB METADATA
    # ======================================================

    def add_metadata(
        self,
        dataframe
    ):

        if dataframe.empty:
            return dataframe

        result = dataframe.copy()

        metadata_columns = [
            "tmdb_id",
            "genres",
            "director",
            "actors"
        ]

        metadata = (
            self.content.movies[
                metadata_columns
            ]
            .copy()
        )

        metadata["tmdb_id"] = (
            metadata["tmdb_id"]
            .astype(int)
        )

        metadata = (
            metadata
            .drop_duplicates(
                subset=["tmdb_id"]
            )
        )

        result["tmdb_id"] = (
            result["tmdb_id"]
            .astype(int)
        )

        columns_to_add = [
            column
            for column in [
                "genres",
                "director",
                "actors"
            ]
            if column not in result.columns
        ]

        if columns_to_add:

            result = (
                result.merge(

                    metadata[
                        ["tmdb_id"]
                        + columns_to_add
                    ],

                    on="tmdb_id",

                    how="left"
                )
            )

        return result


    # ======================================================
    # METADATA HELPERS
    # ======================================================

    def get_field_values(
        self,
        value
    ):

        if pd.isna(value):
            return set()

        return {
            item.strip().lower()
            for item in str(value).split(",")
            if item.strip()
        }


    def field_contains_any(
        self,
        value,
        requested_values
    ):

        if not requested_values:
            return False

        values = self.get_field_values(
            value
        )

        requested = {
            item.strip().lower()
            for item in requested_values
            if item.strip()
        }

        return bool(
            values.intersection(
                requested
            )
        )


    def field_contains_all(
        self,
        value,
        requested_values
    ):

        if not requested_values:
            return True

        values = self.get_field_values(
            value
        )

        requested = {
            item.strip().lower()
            for item in requested_values
            if item.strip()
        }

        return requested.issubset(
            values
        )


    # ======================================================
    # APPLY HARD FILTERS
    # ======================================================

    def apply_filters(
        self,
        dataframe,

        min_year=None,
        max_year=None,

        required_genres=None,
        required_actors=None,
        required_directors=None,

        exclude_genres=None,
        exclude_actors=None,
        exclude_directors=None
    ):

        result = dataframe.copy()

        required_genres = required_genres or []
        required_actors = required_actors or []
        required_directors = required_directors or []

        exclude_genres = exclude_genres or []
        exclude_actors = exclude_actors or []
        exclude_directors = exclude_directors or []


        # --------------------------------------------------
        # YEAR
        # --------------------------------------------------

        if min_year is not None:

            result = result[
                result["release_year"]
                >= min_year
            ]

        if max_year is not None:

            result = result[
                result["release_year"]
                <= max_year
            ]


        # --------------------------------------------------
        # REQUIRED GENRES
        # --------------------------------------------------

        if required_genres:

            result = result[
                result["genres"].apply(

                    lambda value:

                    self.field_contains_all(
                        value,
                        required_genres
                    )
                )
            ]


        # --------------------------------------------------
        # REQUIRED ACTORS
        # --------------------------------------------------

        if required_actors:

            result = result[
                result["actors"].apply(

                    lambda value:

                    self.field_contains_all(
                        value,
                        required_actors
                    )
                )
            ]


        # --------------------------------------------------
        # REQUIRED DIRECTORS
        # --------------------------------------------------

        if required_directors:

            result = result[
                result["director"].apply(

                    lambda value:

                    self.field_contains_all(
                        value,
                        required_directors
                    )
                )
            ]


        # --------------------------------------------------
        # EXCLUDED GENRES
        # --------------------------------------------------

        if exclude_genres:

            result = result[
                ~result["genres"].apply(

                    lambda value:

                    self.field_contains_any(
                        value,
                        exclude_genres
                    )
                )
            ]


        # --------------------------------------------------
        # EXCLUDED ACTORS
        # --------------------------------------------------

        if exclude_actors:

            result = result[
                ~result["actors"].apply(

                    lambda value:

                    self.field_contains_any(
                        value,
                        exclude_actors
                    )
                )
            ]


        # --------------------------------------------------
        # EXCLUDED DIRECTORS
        # --------------------------------------------------

        if exclude_directors:

            result = result[
                ~result["director"].apply(

                    lambda value:

                    self.field_contains_any(
                        value,
                        exclude_directors
                    )
                )
            ]

        return result


    # ======================================================
    # CONTENT RECOMMENDATIONS
    # ======================================================

    def recommend_content(
        self,
        content_scores,
        top_n
    ):

        if content_scores.empty:
            return pd.DataFrame()

        result = (
            content_scores
            .sort_values(
                "content_score",
                ascending=False
            )
            .head(top_n)
            .copy()
        )

        result[
            "recommendation_source"
        ] = "content"

        return result


    # ======================================================
    # GET SVD CANDIDATES
    # ======================================================

    def get_svd_candidates(
        self,
        favorite_tmdb_ids,
        content_scores
    ):

        svd_results = (
            self.svd
            .get_scores_for_favorite_tmdb_movies(

                favorite_tmdb_ids,

                top_n=None
            )
        )

        if svd_results.empty:
            return pd.DataFrame()

        svd_results = (
            svd_results.copy()
        )

        svd_results["tmdb_id"] = (
            svd_results["tmdb_id"]
            .astype(int)
        )


        # --------------------------------------------------
        # ADD TMDB METADATA
        # --------------------------------------------------

        movie_info = (
            self.content.movies[
                [
                    "tmdb_id",
                    "release_year",
                    "genres",
                    "director",
                    "actors"
                ]
            ]
            .copy()
        )

        movie_info["tmdb_id"] = (
            movie_info["tmdb_id"]
            .astype(int)
        )

        movie_info = (
            movie_info
            .drop_duplicates(
                subset=["tmdb_id"]
            )
        )

        svd_results = (
            svd_results.merge(

                movie_info,

                on="tmdb_id",

                how="left"
            )
        )


        # --------------------------------------------------
        # ADD CONTENT SCORE
        # --------------------------------------------------

        if not content_scores.empty:

            content_lookup = (
                content_scores[
                    [
                        "tmdb_id",
                        "content_score"
                    ]
                ]
                .copy()
            )

            content_lookup["tmdb_id"] = (
                content_lookup["tmdb_id"]
                .astype(int)
            )

            content_lookup = (
                content_lookup
                .drop_duplicates(
                    subset=["tmdb_id"]
                )
            )

            svd_results = (
                svd_results.merge(

                    content_lookup,

                    on="tmdb_id",

                    how="left"
                )
            )

        else:

            svd_results[
                "content_score"
            ] = 0.0


        svd_results[
            "content_score"
        ] = (

            pd.to_numeric(
                svd_results["content_score"],
                errors="coerce"
            )
            .fillna(0.0)
        )


        # --------------------------------------------------
        # KEEP STRONGEST SVD CANDIDATES
        # --------------------------------------------------

        svd_results = (
            svd_results
            .sort_values(
                "svd_score",
                ascending=False
            )
            .head(SVD_CANDIDATE_POOL)
            .copy()
        )

        return svd_results


    # ======================================================
    # SOFT PREFERENCE RERANKING
    # ======================================================

    def rerank_svd_with_soft_preferences(
        self,
        svd_results,
        has_soft_preferences
    ):

        if svd_results.empty:
            return svd_results

        result = svd_results.copy()

        # --------------------------------------------------
        # NO EXPLICIT SOFT PREFERENCES
        #
        # Keep pure SVD ordering.
        # --------------------------------------------------

        if not has_soft_preferences:

            result[
                "final_score"
            ] = result["svd_score"]

            return result


        # --------------------------------------------------
        # SOFT PREFERENCE BONUS
        #
        # SVD remains the main signal.
        # Content only gives a small bonus.
        # --------------------------------------------------

        result[
            "final_score"
        ] = (

            result["svd_score"]

            +

            (
                SOFT_CONTENT_BONUS
                *
                result["content_score"]
            )
        )

        return result


    # ======================================================
    # ADD STRONG COLD CONTENT CANDIDATES
    # ======================================================

    def add_cold_content_candidates(
        self,
        svd_results,
        content_scores,
        has_soft_preferences
    ):

        # --------------------------------------------------
        # Only use this lane when the user supplied an
        # explicit soft preference.
        #
        # Without an explicit preference, SVD remains pure.
        # --------------------------------------------------

        if not has_soft_preferences:
            return svd_results

        if content_scores.empty:
            return svd_results


        # --------------------------------------------------
        # TAKE STRONGEST CONTENT CANDIDATES
        # --------------------------------------------------

        content_candidates = (
            content_scores
            .sort_values(
                "content_score",
                ascending=False
            )
            .head(CONTENT_COLD_CANDIDATE_POOL)
            .copy()
        )

        content_candidates[
            "content_rank"
        ] = range(
            1,
            len(content_candidates) + 1
        )


        # --------------------------------------------------
        # KEEP ONLY VERY STRONG CONTENT RESULTS
        # --------------------------------------------------

        content_candidates = (
            content_candidates[
                content_candidates["content_rank"]
                <= COLD_CONTENT_MAX_RANK
            ]
            .copy()
        )

        if content_candidates.empty:
            return svd_results


        # --------------------------------------------------
        # REMOVE MOVIES ALREADY AVAILABLE THROUGH SVD
        #
        # We only want genuinely SVD-unavailable movies in
        # this extra lane.
        # --------------------------------------------------

        mapping = self.svd.mapping.copy()

        mapping = (
            mapping
            .dropna(
                subset=["tmdb_id"]
            )
            .copy()
        )

        mapping["tmdb_id"] = (
            mapping["tmdb_id"]
            .astype(int)
        )

        usable_movie_ids = set(
            self.svd.movie_to_index.keys()
        )

        mapping = mapping[
            mapping["movieId"]
            .isin(usable_movie_ids)
        ]

        svd_available_tmdb_ids = set(
            mapping["tmdb_id"]
            .tolist()
        )

        content_candidates["tmdb_id"] = (
            content_candidates["tmdb_id"]
            .astype(int)
        )

        cold_candidates = (
            content_candidates[
                ~content_candidates["tmdb_id"]
                .isin(
                    svd_available_tmdb_ids
                )
            ]
            .copy()
        )

        if cold_candidates.empty:
            return svd_results


        # --------------------------------------------------
        # REMOVE ANY DUPLICATES ALREADY IN CURRENT RESULTS
        # --------------------------------------------------

        if not svd_results.empty:

            existing_ids = set(
                svd_results["tmdb_id"]
                .astype(int)
                .tolist()
            )

            cold_candidates = (
                cold_candidates[
                    ~cold_candidates["tmdb_id"]
                    .isin(existing_ids)
                ]
                .copy()
            )

        if cold_candidates.empty:
            return svd_results


        # --------------------------------------------------
        # COLD-ITEM COMPETITION SCORE
        #
        # IMPORTANT:
        #
        # Missing SVD score does NOT mean SVD score = 0.
        #
        # Instead, strong cold items enter according to
        # their CONTENT RANK.
        #
        # #1 content candidate receives a score slightly
        # above the current 10th recommendation.
        #
        # Lower content ranks receive progressively smaller
        # scores.
        # --------------------------------------------------

        if svd_results.empty:

            return svd_results


        ranked_svd = (
            svd_results
            .sort_values(
                "final_score",
                ascending=False
            )
            .copy()
        )


        # Score around the Top-N boundary.
        #
        # We use the 10th SVD result when available.
        # This allows exceptional cold items to compete for
        # Top 10 without automatically replacing the best
        # collaborative recommendations.

        boundary_position = min(
            DEFAULT_TOP_N,
            len(ranked_svd)
        )

        boundary_score = float(
            ranked_svd
            .iloc[
                boundary_position - 1
            ]["final_score"]
        )


        # --------------------------------------------------
        # Rank-based cold score
        # --------------------------------------------------

        cold_candidates[
            "cold_strength"
        ] = (

            (
                COLD_CONTENT_MAX_RANK
                - cold_candidates[
                    "content_rank"
                ]
                + 1
            )

            / COLD_CONTENT_MAX_RANK
        )


        # Best cold candidate can rise modestly above the
        # current Top-10 boundary.
        #
        # Lower-ranked content candidates fall below it.

        cold_candidates[
            "final_score"
        ] = (

            boundary_score

            * (

                0.90

                + (

                    0.15

                    * cold_candidates[
                        "cold_strength"
                    ]
                )

            )

        )


        cold_candidates[
            "svd_score"
        ] = pd.NA


        cold_candidates[
            "recommendation_source"
        ] = "content_cold"


        # --------------------------------------------------
        # COMBINE
        # --------------------------------------------------

        combined = pd.concat(
            [
                svd_results,
                cold_candidates
            ],
            ignore_index=True,
            sort=False
        )


        combined = (
            combined
            .sort_values(
                "final_score",
                ascending=False
            )
            .drop_duplicates(
                subset=["tmdb_id"],
                keep="first"
            )
            .reset_index(drop=True)
        )


        return combined

    # ======================================================
    # FILL MISSING RESULTS FROM CONTENT
    # ======================================================

    def fill_with_content(
        self,
        svd_results,
        content_scores,
        top_n
    ):

        # --------------------------------------------------
        # START WITH AVAILABLE SVD RESULTS
        # --------------------------------------------------

        if svd_results.empty:

            current = pd.DataFrame()

        else:

            current = (
                svd_results
                .copy()
            )

            if (
                "recommendation_source"
                not in current.columns
            ):

                current[
                    "recommendation_source"
                ] = "svd"

            else:

                current[
                    "recommendation_source"
                ] = (
                    current[
                        "recommendation_source"
                    ]
                    .fillna("svd")
                )


        # --------------------------------------------------
        # ENOUGH RESULTS ALREADY
        # --------------------------------------------------

        if len(current) >= top_n:

            return (
                current
                .head(top_n)
                .copy()
            )


        # --------------------------------------------------
        # NUMBER OF RESULTS STILL NEEDED
        # --------------------------------------------------

        needed = (
            top_n
            - len(current)
        )


        if content_scores.empty:

            return (
                current
                .head(top_n)
                .copy()
            )


        # --------------------------------------------------
        # EXCLUDE MOVIES ALREADY RETURNED
        # --------------------------------------------------

        content_candidates = (
            content_scores
            .copy()
        )

        if not current.empty:

            used_tmdb_ids = set(
                current["tmdb_id"]
                .astype(int)
                .tolist()
            )

            content_candidates = (
                content_candidates[
                    ~content_candidates[
                        "tmdb_id"
                    ]
                    .astype(int)
                    .isin(used_tmdb_ids)
                ]
            )


        # --------------------------------------------------
        # GET BEST CONTENT FALLBACKS
        # --------------------------------------------------

        content_candidates = (
            content_candidates
            .sort_values(
                "content_score",
                ascending=False
            )
            .head(needed)
            .copy()
        )

        content_candidates[
            "recommendation_source"
        ] = "content_fallback"


        # --------------------------------------------------
        # COMBINE
        # --------------------------------------------------

        if current.empty:

            return content_candidates


        return pd.concat(
            [
                current,
                content_candidates
            ],
            ignore_index=True,
            sort=False
        )


    # ======================================================
    # MAIN RECOMMEND FUNCTION
    # ======================================================

    def recommend(
        self,

        favorite_movies=None,

        # --------------------------------------------------
        # SOFT PREFERENCES
        # --------------------------------------------------

        favorite_genres=None,
        favorite_actors=None,
        favorite_directors=None,

        # --------------------------------------------------
        # HARD REQUIREMENTS
        # --------------------------------------------------

        required_genres=None,
        required_actors=None,
        required_directors=None,

        # --------------------------------------------------
        # HARD EXCLUSIONS
        # --------------------------------------------------

        exclude_genres=None,
        exclude_actors=None,
        exclude_directors=None,

        # --------------------------------------------------
        # YEAR CONSTRAINTS
        # --------------------------------------------------

        min_year=None,
        max_year=None,

        top_n=DEFAULT_TOP_N
    ):


        # ==================================================
        # NORMALIZE INPUTS
        # ==================================================

        favorite_movies = favorite_movies or []

        favorite_movie_titles = [
            movie["title"]
            for movie in favorite_movies
            if isinstance(movie, dict)
            and movie.get("title")
        ]

        favorite_genres = favorite_genres or []
        favorite_actors = favorite_actors or []
        favorite_directors = favorite_directors or []

        required_genres = required_genres or []
        required_actors = required_actors or []
        required_directors = required_directors or []

        exclude_genres = exclude_genres or []
        exclude_actors = exclude_actors or []
        exclude_directors = exclude_directors or []


        # ==================================================
        # PRINT REQUEST
        # ==================================================

        print("\n")
        print("=" * 75)
        print("RECOMMENDATION REQUEST")
        print("=" * 75)

        print(
            "\nFavorite movies:",
            favorite_movies
        )

        print(
            "Soft genres:",
            favorite_genres
        )

        print(
            "Soft actors:",
            favorite_actors
        )

        print(
            "Soft directors:",
            favorite_directors
        )

        print(
            "Required genres:",
            required_genres
        )

        print(
            "Required actors:",
            required_actors
        )

        print(
            "Required directors:",
            required_directors
        )

        print(
            "Excluded genres:",
            exclude_genres
        )

        print(
            "Excluded actors:",
            exclude_actors
        )

        print(
            "Excluded directors:",
            exclude_directors
        )

        print(
            "Year range:",
            min_year,
            "-",
            max_year
        )


        # ==================================================
        # 1. CONTENT SCORES
        # ==================================================

        favorite_tmdb_ids = (
            self.get_favorite_tmdb_ids(
                favorite_movies
            )
        )

        print(
            "\nCalculating content scores..."
        )

        content_scores = (
            self.get_content_scores(

                favorite_movie_titles,

                favorite_tmdb_ids,

                favorite_genres,

                favorite_actors,

                favorite_directors
            )
        )


        # ==================================================
        # 2. ADD METADATA
        # ==================================================

        content_scores = (
            self.add_metadata(
                content_scores
            )
        )


        # ==================================================
        # 3. APPLY HARD FILTERS TO CONTENT
        # ==================================================

        filtered_content = (
            self.apply_filters(

                content_scores,

                min_year=min_year,
                max_year=max_year,

                required_genres=
                    required_genres,

                required_actors=
                    required_actors,

                required_directors=
                    required_directors,

                exclude_genres=
                    exclude_genres,

                exclude_actors=
                    exclude_actors,

                exclude_directors=
                    exclude_directors
            )
        )


        # ==================================================
        # 4. PRINT FAVORITE TMDB IDS
        # ==================================================

        print(
            "\nFavorite TMDB IDs:",
            favorite_tmdb_ids
        )


        # ==================================================
        # 5. CHECK SVD COVERAGE
        # ==================================================

        usable_svd_favorites = (
            self.get_usable_svd_favorites(
                favorite_tmdb_ids
            )
        )

        print(
            "Favorites usable by SVD:",
            len(usable_svd_favorites),
            "/",
            len(favorite_tmdb_ids)
        )


        enough_svd_favorites = (

            len(usable_svd_favorites)

            >= MIN_SVD_FAVORITES
        )


        # ==================================================
        # 6. DETERMINE REQUEST TYPE
        # ==================================================

        has_soft_preferences = (

            bool(favorite_genres)

            or bool(favorite_actors)

            or bool(favorite_directors)
        )


        has_hard_constraints = (

            bool(required_genres)

            or bool(required_actors)

            or bool(required_directors)

            or bool(exclude_genres)

            or bool(exclude_actors)

            or bool(exclude_directors)

            or min_year is not None

            or max_year is not None
        )


        # ==================================================
        # 7. SVD PRIMARY
        # ==================================================

        if enough_svd_favorites:

            print(
                "\nUsing SVD as primary model..."
            )

            svd_results = (
                self.get_svd_candidates(

                    favorite_tmdb_ids,

                    content_scores
                )
            )


            # ----------------------------------------------
            # APPLY HARD FILTERS TO SVD CANDIDATES
            # ----------------------------------------------

            if has_hard_constraints:

                svd_results = (
                    self.apply_filters(

                        svd_results,

                        min_year=min_year,
                        max_year=max_year,

                        required_genres=
                            required_genres,

                        required_actors=
                            required_actors,

                        required_directors=
                            required_directors,

                        exclude_genres=
                            exclude_genres,

                        exclude_actors=
                            exclude_actors,

                        exclude_directors=
                            exclude_directors
                    )
                )


            # ----------------------------------------------
            # SOFT PREFERENCE RERANK
            # ----------------------------------------------

            svd_results = (
                self.rerank_svd_with_soft_preferences(

                    svd_results,

                    has_soft_preferences
                )
            )


            # ----------------------------------------------
            # SORT
            # ----------------------------------------------

            svd_results = (
                svd_results
                .sort_values(
                    "final_score",
                    ascending=False
                )
                .copy()
            )

            # ----------------------------------------------
            # ADD STRONG SVD-UNAVAILABLE CONTENT CANDIDATES
            # ----------------------------------------------

            svd_results = (
                self.add_cold_content_candidates(

                    svd_results,

                    filtered_content,

                    has_soft_preferences
                )
            )

            # ----------------------------------------------
            # FILL WITH CONTENT IF SVD DOES NOT HAVE
            # ENOUGH ELIGIBLE MOVIES
            # ----------------------------------------------

            recommendations = (
                self.fill_with_content(

                    svd_results,

                    filtered_content,

                    top_n
                )
            )


            if has_hard_constraints:

                mode = (
                    "SVD PRIMARY + HARD FILTERS"
                )

            elif has_soft_preferences:

                mode = (
                    "SVD PRIMARY + SOFT CONTENT RERANK"
                )

            else:

                mode = (
                    "SVD — collaborative primary"
                )


        # ==================================================
        # 8. CONTENT FALLBACK
        # ==================================================

        else:

            print(
                "\nNot enough usable SVD favorites."
            )

            recommendations = (
                self.recommend_content(

                    filtered_content,

                    top_n
                )
            )


            if has_hard_constraints:

                mode = (
                    "CONTENT — hard constraints"
                )

            elif has_soft_preferences:

                mode = (
                    "CONTENT — soft preferences"
                )

            else:

                mode = (
                    "CONTENT — SVD fallback"
                )


        # ==================================================
        # 9. FINAL OUTPUT
        # ==================================================

        print(
            "\nSelected mode:",
            mode
        )


        if recommendations.empty:

            print(
                "No recommendations satisfy "
                "the request."
            )

            return recommendations


        recommendations = (
            recommendations
            .head(top_n)
            .reset_index(drop=True)
        )


        recommendations[
            "recommendation_rank"
        ] = (
            recommendations.index + 1
        )


        recommendations[
            "engine_mode"
        ] = mode


        return recommendations


# ==========================================================
# TESTS
# ==========================================================

if __name__ == "__main__":

    engine = RecommendationEngine()


    # ======================================================
    # TEST 1
    # PURE SVD
    # ======================================================

    print(
        "\n\nTEST 1 — PURE SVD"
    )

    test1 = engine.recommend(

        favorite_movies=[
            "Interstellar",
            "The Godfather",
            "The Dark Knight"
        ],

        top_n=10
    )

    if not test1.empty:

        print(
            test1[
                [
                    "recommendation_rank",
                    "title",
                    "release_year",
                    "recommendation_source",
                    "svd_score"
                ]
            ].to_string(index=False)
        )


    # ======================================================
    # TEST 2
    # SVD + SOFT PREFERENCES
    # ======================================================

    print(
        "\n\nTEST 2 — SVD + SOFT PREFERENCES"
    )

    test2 = engine.recommend(

        favorite_movies=[
            "The Dark Knight",
            "Interstellar",
            "The Godfather"
        ],

        favorite_genres=[
            "Science Fiction"
        ],

        favorite_actors=[
            "Christian Bale"
        ],

        favorite_directors=[
            "Christopher Nolan"
        ],

        top_n=10
    )

    if not test2.empty:

        print(
            test2[
                [
                    "recommendation_rank",
                    "title",
                    "release_year",
                    "svd_score",
                    "content_score",
                    "final_score",
                    "recommendation_source"
                ]
            ].to_string(index=False)
        )


    # ======================================================
    # TEST 3
    # SVD + HARD GENRE
    # ======================================================

    print(
        "\n\nTEST 3 — SVD + REQUIRED SCIENCE FICTION"
    )

    test3 = engine.recommend(

        favorite_movies=[
            "The Dark Knight",
            "Interstellar",
            "The Godfather"
        ],

        required_genres=[
            "Science Fiction"
        ],

        top_n=10
    )

    if not test3.empty:

        print(
            test3[
                [
                    "recommendation_rank",
                    "title",
                    "release_year",
                    "genres",
                    "recommendation_source"
                ]
            ].to_string(index=False)
        )


    # ======================================================
    # TEST 4
    # SVD + HARD DIRECTOR EXCLUSION
    # ======================================================

    print(
        "\n\nTEST 4 — SVD + EXCLUDE NOLAN"
    )

    test4 = engine.recommend(

        favorite_movies=[
            "The Dark Knight",
            "Interstellar",
            "The Godfather"
        ],

        exclude_directors=[
            "Christopher Nolan"
        ],

        top_n=10
    )

    if not test4.empty:

        print(
            test4[
                [
                    "recommendation_rank",
                    "title",
                    "release_year",
                    "director",
                    "recommendation_source"
                ]
            ].to_string(index=False)
        )


    # ======================================================
    # TEST 5
    # CONTENT FALLBACK
    # ======================================================

    print(
        "\n\nTEST 5 — CONTENT FALLBACK"
    )

    test5 = engine.recommend(

        favorite_movies=[
            "Interstellar"
        ],

        required_genres=[
            "Science Fiction"
        ],

        min_year=2024,

        top_n=10
    )

    if not test5.empty:

        print(
            test5[
                [
                    "recommendation_rank",
                    "title",
                    "release_year",
                    "genres",
                    "recommendation_source"
                ]
            ].to_string(index=False)
        )
