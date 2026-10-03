import pandas as pd

from content_recommender import ContentRecommender
from svd_recommender import SVDRecommender


# ==========================================================
# SETTINGS
# ==========================================================

# Hybrid V3 does NOT use a fixed weighted average.
#
# Instead:
#
# - SVD is the primary recommender when collaborative
#   information is available.
#
# - Content is used to introduce movies that SVD cannot
#   represent, especially cold/new movies.

DEFAULT_TOP_N = 10

# 20% of the final recommendation list is reserved
# for content-only / cold-item recommendations.
#
# For Top 10:
#
# 8 SVD recommendations
# 2 content cold-item recommendations

CONTENT_EXPLORATION_FRACTION = 0.20


# ==========================================================
# HYBRID RECOMMENDER V3
# ==========================================================

class HybridRecommender:

    def __init__(self):

        print(
            "\nInitializing Hybrid Recommender V3..."
        )

        # --------------------------------------------------
        # Content model
        # --------------------------------------------------

        self.content = ContentRecommender()

        # --------------------------------------------------
        # SVD model
        # --------------------------------------------------

        self.svd = SVDRecommender()

        self.svd.fit()

        print(
            "\nHybrid Recommender V3 ready."
        )


    # ======================================================
    # FIND TMDB IDS FOR FAVORITE MOVIES
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

            movie_match = self.content.movies[
                self.content.movies[
                    "title"
                ].str.lower()
                == match.lower()
            ]

            if movie_match.empty:
                continue

            tmdb_id = int(
                movie_match.iloc[0][
                    "tmdb_id"
                ]
            )

            tmdb_ids.append(
                tmdb_id
            )

        return tmdb_ids


    # ======================================================
    # MOVIE INFORMATION
    # ======================================================

    def get_movie_info(self):

        movie_info = (
            self.content.movies[
                [
                    "tmdb_id",
                    "title",
                    "release_year"
                ]
            ]
            .copy()
        )

        movie_info["tmdb_id"] = (
            movie_info["tmdb_id"]
            .astype(int)
        )

        return movie_info


    # ======================================================
    # RECOMMEND
    # ======================================================

    def recommend(
        self,
        favorite_movies=None,
        favorite_genres=None,
        favorite_actors=None,
        favorite_directors=None,
        top_n=DEFAULT_TOP_N
    ):

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


        # ==================================================
        # 1. CONTENT SCORES
        # ==================================================

        print(
            "\nCalculating content scores..."
        )

        content_scores = (
            self.content.get_scores(

                favorite_movies=
                    favorite_movies,

                favorite_genres=
                    favorite_genres,

                favorite_actors=
                    favorite_actors,

                favorite_directors=
                    favorite_directors,

                top_n=None
            )
        )


        if content_scores.empty:

            print(
                "No content recommendations available."
            )

            return pd.DataFrame()


        content_scores = (
            content_scores
            .copy()
        )

        content_scores["tmdb_id"] = (
            content_scores["tmdb_id"]
            .astype(int)
        )


        # ==================================================
        # 2. FAVORITE TMDB IDS
        # ==================================================

        favorite_tmdb_ids = (
            self.get_favorite_tmdb_ids(
                favorite_movies
            )
        )

        favorite_tmdb_set = set(
            favorite_tmdb_ids
        )

        print(
            "\nFavorite TMDB IDs:",
            favorite_tmdb_ids
        )


        # ==================================================
        # 3. SVD SCORES
        # ==================================================

        print(
            "\nCalculating collaborative SVD scores..."
        )

        if favorite_tmdb_ids:

            svd_scores = (
                self.svd
                .get_scores_for_favorite_tmdb_movies(

                    favorite_tmdb_ids,

                    top_n=None
                )
            )

        else:

            svd_scores = pd.DataFrame()


        if not svd_scores.empty:

            svd_scores = (
                svd_scores
                .copy()
            )

            svd_scores["tmdb_id"] = (
                svd_scores["tmdb_id"]
                .astype(int)
            )


        # ==================================================
        # 4. REMOVE FAVORITES
        # ==================================================

        content_scores = (
            content_scores[
                ~content_scores[
                    "tmdb_id"
                ].isin(
                    favorite_tmdb_set
                )
            ]
            .copy()
        )


        if not svd_scores.empty:

            svd_scores = (
                svd_scores[
                    ~svd_scores[
                        "tmdb_id"
                    ].isin(
                        favorite_tmdb_set
                    )
                ]
                .copy()
            )


        # ==================================================
        # 5. IF SVD IS NOT AVAILABLE
        # ==================================================

        # If the user's favorite movies cannot be mapped
        # into the collaborative model, fall back completely
        # to the content recommender.

        if svd_scores.empty:

            print(
                "\nNo usable SVD recommendations."
            )

            print(
                "Using content-only fallback."
            )

            result = (
                content_scores
                .sort_values(
                    "content_score",
                    ascending=False
                )
                .head(top_n)
                .copy()
            )

            result["source"] = (
                "content_fallback"
            )

            result["svd_score"] = pd.NA

            result["hybrid_rank"] = (
                range(
                    1,
                    len(result) + 1
                )
            )

            columns = [
                "tmdb_id",
                "title",
                "release_year",
                "content_score",
                "svd_score",
                "source",
                "hybrid_rank"
            ]

            return (
                result[columns]
                .reset_index(drop=True)
            )


        # ==================================================
        # 6. DETERMINE FINAL SLOT COUNTS
        # ==================================================

        content_slots = int(
            round(
                top_n
                * CONTENT_EXPLORATION_FRACTION
            )
        )

        # If top_n is reasonably large,
        # reserve at least one content slot.

        if (
            top_n >= 5
            and content_slots == 0
        ):

            content_slots = 1


        # Never let content consume the whole list
        # when SVD is available.

        if content_slots >= top_n:

            content_slots = (
                top_n - 1
            )


        svd_slots = (
            top_n
            - content_slots
        )


        print(
            "\nHybrid V3 allocation:"
        )

        print(
            "SVD slots:",
            svd_slots
        )

        print(
            "Cold-content slots:",
            content_slots
        )


        # ==================================================
        # 7. FIND MOVIES SVD CAN REPRESENT
        # ==================================================

        # Because we requested top_n=None from SVD,
        # this set represents the TMDB movies for which
        # the SVD recommender can currently produce a score.

        svd_available_ids = set(
            svd_scores[
                "tmdb_id"
            ]
            .astype(int)
            .tolist()
        )


        print(
            "\nMovies with SVD scores:",
            len(svd_available_ids)
        )


        # ==================================================
        # 8. PRIMARY SVD RECOMMENDATIONS
        # ==================================================

        svd_primary = (
            svd_scores
            .sort_values(
                "svd_score",
                ascending=False
            )
            .head(
                svd_slots
            )
            .copy()
        )


        svd_primary["source"] = (
            "svd"
        )


        # ==================================================
        # 9. CONTENT COLD-ITEM CANDIDATES
        # ==================================================

        # These are movies in our TMDB catalog that are NOT
        # represented by the SVD recommendation universe.
        #
        # This is where content-based recommendation has a
        # unique advantage.

        cold_content = (
            content_scores[
                ~content_scores[
                    "tmdb_id"
                ].isin(
                    svd_available_ids
                )
            ]
            .copy()
        )


        cold_content = (
            cold_content
            .sort_values(
                "content_score",
                ascending=False
            )
        )


        print(
            "Content-only cold candidates:",
            len(cold_content)
        )


        # ==================================================
        # 10. SELECT CONTENT COLD ITEMS
        # ==================================================

        content_exploration = (
            cold_content
            .head(
                content_slots
            )
            .copy()
        )


        content_exploration["source"] = (
            "content_cold"
        )


        # ==================================================
        # 11. IF NOT ENOUGH COLD ITEMS
        # ==================================================

        # Normally the TMDB catalog gives us plenty.
        #
        # But if fewer cold items exist than requested,
        # give the unused slots back to SVD.

        missing_content_slots = (
            content_slots
            - len(
                content_exploration
            )
        )


        if missing_content_slots > 0:

            extra_svd = (
                svd_scores
                .sort_values(
                    "svd_score",
                    ascending=False
                )
                .iloc[
                    svd_slots:
                    svd_slots
                    + missing_content_slots
                ]
                .copy()
            )

            extra_svd["source"] = (
                "svd"
            )

            svd_primary = pd.concat(
                [
                    svd_primary,
                    extra_svd
                ],
                ignore_index=True
            )


        # ==================================================
        # 12. PREPARE SVD RESULTS
        # ==================================================

        movie_info = (
            self.get_movie_info()
        )


        svd_result = (
            svd_primary[
                [
                    "tmdb_id",
                    "svd_score",
                    "source"
                ]
            ]
            .copy()
        )


        svd_result = (
            svd_result.merge(
                movie_info,
                on="tmdb_id",
                how="left"
            )
        )


        # Get content scores for SVD movies too.
        #
        # We are NOT using them to override the SVD ranking.
        # They are included for inspection/explanation.

        content_lookup = (
            content_scores[
                [
                    "tmdb_id",
                    "content_score"
                ]
            ]
            .drop_duplicates(
                subset=["tmdb_id"]
            )
        )


        svd_result = (
            svd_result.merge(
                content_lookup,
                on="tmdb_id",
                how="left"
            )
        )


        # ==================================================
        # 13. PREPARE CONTENT RESULTS
        # ==================================================

        if not content_exploration.empty:

            content_result = (
                content_exploration[
                    [
                        "tmdb_id",
                        "title",
                        "release_year",
                        "content_score",
                        "source"
                    ]
                ]
                .copy()
            )

            content_result[
                "svd_score"
            ] = pd.NA

        else:

            content_result = pd.DataFrame(
                columns=[
                    "tmdb_id",
                    "title",
                    "release_year",
                    "content_score",
                    "source",
                    "svd_score"
                ]
            )


        # ==================================================
        # 14. BUILD FINAL LIST
        # ==================================================

        # First keep the strong SVD recommendations.
        #
        # Then append the content cold-item discoveries.
        #
        # Later we can experiment with interleaving, but
        # first we want a simple and interpretable V3.

        final = pd.concat(
            [
                svd_result,
                content_result
            ],
            ignore_index=True
        )


        # Remove any accidental duplicates.

        final = (
            final
            .drop_duplicates(
                subset=["tmdb_id"],
                keep="first"
            )
            .head(top_n)
            .reset_index(drop=True)
        )


        final["hybrid_rank"] = (
            final.index + 1
        )


        # ==================================================
        # 15. RETURN
        # ==================================================

        columns = [
            "tmdb_id",
            "title",
            "release_year",
            "content_score",
            "svd_score",
            "source",
            "hybrid_rank"
        ]


        return (
            final[columns]
            .copy()
        )


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    recommender = (
        HybridRecommender()
    )


    recommendations = (
        recommender.recommend(

            favorite_movies=[
                "Interstellar",
                "The Godfather",
                "The Dark Knight"
            ],

            favorite_genres=[],

            favorite_actors=[],

            favorite_directors=[],

            top_n=10
        )
    )


    print("\n")

    print("=" * 110)

    print(
        "HYBRID RECOMMENDATIONS - VERSION 3"
    )

    print("=" * 110)


    if recommendations.empty:

        print(
            "No recommendations available."
        )

    else:

        print(
            recommendations.to_string(
                index=False
            )
        )