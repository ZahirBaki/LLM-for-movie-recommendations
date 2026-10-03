import pandas as pd

from content_recommender import ContentRecommender
from svd_recommender import SVDRecommender


# ==========================================================
# SETTINGS
# ==========================================================

DEFAULT_TOP_N = 10

# Minimum number of favorite movies that must be usable
# by SVD before we trust SVD as the primary recommender.
MIN_SVD_FAVORITES = 2


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
        self.svd.fit()

        print("\nRecommendation engine ready.")


    # ======================================================
    # FIND FAVORITE TMDB IDS
    # ======================================================

    def get_favorite_tmdb_ids(
        self,
        favorite_movies
    ):

        tmdb_ids = []

        movie_titles = (
            self.content.movies["title"]
            .tolist()
        )

        for movie_name in favorite_movies:

            match = self.content.find_best_match(
                movie_name,
                movie_titles,
                "Movie"
            )

            if match is None:
                continue

            movie_match = (
                self.content.movies[
                    self.content.movies[
                        "title"
                    ].str.lower()
                    == match.lower()
                ]
            )

            if movie_match.empty:
                continue

            tmdb_id = int(
                movie_match.iloc[0]["tmdb_id"]
            )

            tmdb_ids.append(tmdb_id)

        return tmdb_ids


    # ======================================================
    # GET CONTENT SCORES
    # ======================================================

    def get_content_scores(
        self,
        favorite_movies,
        favorite_genres,
        favorite_actors,
        favorite_directors
    ):

        return self.content.get_scores(

            favorite_movies=favorite_movies,

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

        # Only MovieLens movies surviving the SVD
        # minimum-rating filter are actually usable.

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

                usable.append(
                    tmdb_id
                )

        return usable


    # ======================================================
    # ADD TMDB METADATA TO CONTENT RESULTS
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

        # Only add metadata columns that are not
        # already present in the score dataframe.

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
    # HELPER — SPLIT METADATA FIELD
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


    # ======================================================
    # HELPER — CONTAINS ANY
    # ======================================================

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


    # ======================================================
    # HELPER — CONTAINS ALL
    # ======================================================

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

        required_genres = (
            required_genres or []
        )

        required_actors = (
            required_actors or []
        )

        required_directors = (
            required_directors or []
        )

        exclude_genres = (
            exclude_genres or []
        )

        exclude_actors = (
            exclude_actors or []
        )

        exclude_directors = (
            exclude_directors or []
        )


        # --------------------------------------------------
        # YEAR FILTERS
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
    # SVD RECOMMENDATIONS
    # ======================================================

    def recommend_svd(
        self,
        favorite_tmdb_ids,
        content_scores,
        top_n
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
        # ADD CONTENT SCORE FOR INSPECTION
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
            ] = pd.NA


        svd_results[
            "recommendation_source"
        ] = "svd"


        return (
            svd_results
            .sort_values(
                "svd_score",
                ascending=False
            )
            .head(top_n)
            .copy()
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


        # --------------------------------------------------
        # NORMALIZE INPUTS
        # --------------------------------------------------

        favorite_movies = (
            favorite_movies or []
        )

        favorite_genres = (
            favorite_genres or []
        )

        favorite_actors = (
            favorite_actors or []
        )

        favorite_directors = (
            favorite_directors or []
        )

        required_genres = (
            required_genres or []
        )

        required_actors = (
            required_actors or []
        )

        required_directors = (
            required_directors or []
        )

        exclude_genres = (
            exclude_genres or []
        )

        exclude_actors = (
            exclude_actors or []
        )

        exclude_directors = (
            exclude_directors or []
        )


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
        # 1. CALCULATE CONTENT SCORES
        # ==================================================

        print(
            "\nCalculating content scores..."
        )

        content_scores = (
            self.get_content_scores(

                favorite_movies,

                favorite_genres,

                favorite_actors,

                favorite_directors
            )
        )


        # ==================================================
        # 2. ADD RAW TMDB METADATA
        # ==================================================

        # get_scores() contains scores such as actor_score
        # and director_score, but hard filtering needs the
        # actual actor/director/genre metadata.

        content_scores = (
            self.add_metadata(
                content_scores
            )
        )


        # ==================================================
        # 3. APPLY HARD CONSTRAINTS
        # ==================================================

        content_scores = (
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
        # 4. FAVORITE TMDB IDS
        # ==================================================

        favorite_tmdb_ids = (
            self.get_favorite_tmdb_ids(
                favorite_movies
            )
        )

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


        enough_svd_favorites = (

            len(usable_svd_favorites)

            >= MIN_SVD_FAVORITES
        )


        # ==================================================
        # 7. CHOOSE RECOMMENDATION ENGINE
        # ==================================================


        # --------------------------------------------------
        # HARD CONSTRAINT MODE
        # --------------------------------------------------

        if has_hard_constraints:

            mode = (
                "CONTENT — hard constraints"
            )

            recommendations = (
                self.recommend_content(

                    content_scores,

                    top_n
                )
            )


        # --------------------------------------------------
        # SOFT PREFERENCE MODE
        # --------------------------------------------------

        elif has_soft_preferences:

            mode = (
                "CONTENT — soft preference"
            )

            recommendations = (
                self.recommend_content(

                    content_scores,

                    top_n
                )
            )


        # --------------------------------------------------
        # SVD PRIMARY MODE
        # --------------------------------------------------

        elif enough_svd_favorites:

            mode = (
                "SVD — collaborative primary"
            )

            recommendations = (
                self.recommend_svd(

                    favorite_tmdb_ids,

                    content_scores,

                    top_n
                )
            )


        # --------------------------------------------------
        # CONTENT FALLBACK
        # --------------------------------------------------

        else:

            mode = (
                "CONTENT — SVD fallback"
            )

            recommendations = (
                self.recommend_content(

                    content_scores,

                    top_n
                )
            )


        # ==================================================
        # 8. FINAL OUTPUT
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
    # TEST 1 — NORMAL FAVORITES
    #
    # Expected:
    # SVD collaborative primary
    # ======================================================

    print(
        "\n\nTEST 1 — NORMAL FAVORITES"
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
                    "recommendation_source"
                ]
            ].to_string(index=False)
        )


    # ======================================================
    # TEST 2 — HARD ACTOR REQUIREMENT
    #
    # Every recommendation MUST contain Christian Bale.
    # ======================================================

    print(
        "\n\nTEST 2 — REQUIRE CHRISTIAN BALE"
    )

    test2 = engine.recommend(

        favorite_movies=[
            "The Dark Knight"
        ],

        required_actors=[
            "Christian Bale"
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
                    "actors",
                    "content_score"
                ]
            ].to_string(index=False)
        )


    # ======================================================
    # TEST 3 — HARD DIRECTOR REQUIREMENT
    #
    # Every recommendation MUST be directed by Nolan.
    # ======================================================

    print(
        "\n\nTEST 3 — REQUIRE CHRISTOPHER NOLAN"
    )

    test3 = engine.recommend(

        favorite_movies=[
            "Interstellar"
        ],

        required_directors=[
            "Christopher Nolan"
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
                    "director",
                    "content_score"
                ]
            ].to_string(index=False)
        )


    # ======================================================
    # TEST 4 — EXCLUDE DIRECTOR
    #
    # Similar to Interstellar,
    # but Christopher Nolan must NOT appear.
    # ======================================================

    print(
        "\n\nTEST 4 — LIKE INTERSTELLAR BUT NOT NOLAN"
    )

    test4 = engine.recommend(

        favorite_movies=[
            "Interstellar"
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
                    "content_score"
                ]
            ].to_string(index=False)
        )


    # ======================================================
    # TEST 5 — MULTIPLE HARD CONSTRAINTS
    #
    # Must:
    # 1. Be Science Fiction
    # 2. Be from 2024 or later
    # ======================================================

    print(
        "\n\nTEST 5 — RECENT SCIENCE FICTION"
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
                    "content_score"
                ]
            ].to_string(index=False)
        )


    # ======================================================
    # TEST 6 — SOFT ACTOR PREFERENCE
    #
    # Christian Bale is preferred,
    # but NOT required.
    # ======================================================

    print(
        "\n\nTEST 6 — SOFT CHRISTIAN BALE PREFERENCE"
    )

    test6 = engine.recommend(

        favorite_movies=[
            "The Dark Knight"
        ],

        favorite_actors=[
            "Christian Bale"
        ],

        top_n=10
    )

    if not test6.empty:

        print(
            test6[
                [
                    "recommendation_rank",
                    "title",
                    "release_year",
                    "actors",
                    "content_score"
                ]
            ].to_string(index=False)
        )