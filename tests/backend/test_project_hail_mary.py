import pandas as pd

from backend.recommendation.recommendation_engine import RecommendationEngine


# ==========================================================
# TEST CASE
# ==========================================================

FAVORITE_MOVIES = [
    "Interstellar",
    "Gravity",
    "The Martian",
]

FAVORITE_ACTORS = []

TARGET_TITLE = "Project Hail Mary"

TOP_N = 20


# ==========================================================
# START ENGINE
# ==========================================================

def main():
    print("\n" + "=" * 80)
    print("PROJECT HAIL MARY DIAGNOSTIC")
    print("=" * 80)

    print("\nInitializing recommendation engine...")

    engine = RecommendationEngine()


    # ==========================================================
    # INPUT
    # ==========================================================

    print("\n" + "=" * 80)
    print("TEST INPUT")
    print("=" * 80)

    print("\nFavorite movies:")

    for movie in FAVORITE_MOVIES:
        print(" -", movie)

    print("\nSoft actor preference:")

    for actor in FAVORITE_ACTORS:
        print(" -", actor)


    # ==========================================================
    # FIND TARGET MOVIE IN TMDB CATALOG
    # ==========================================================

    movies = engine.content.movies.copy()

    target_matches = movies[
        movies["title"]
        .astype(str)
        .str.lower()
        .eq(TARGET_TITLE.lower())
    ].copy()


    print("\n" + "=" * 80)
    print("TARGET MOVIE IN TMDB CATALOG")
    print("=" * 80)


    if target_matches.empty:

        print(
            f"\nERROR: '{TARGET_TITLE}' was not found "
            "in the TMDB catalog."
        )

        raise SystemExit


    target = target_matches.iloc[0]

    target_tmdb_id = int(target["tmdb_id"])


    print("\nTitle:", target["title"])

    print(
        "TMDB ID:",
        target_tmdb_id
    )

    print(
        "Year:",
        target.get("release_year")
    )

    print(
        "Genres:",
        target.get("genres")
    )

    print(
        "Director:",
        target.get("director")
    )

    print(
        "Actors:",
        target.get("actors")
    )


    # ==========================================================
    # CHECK RYAN GOSLING METADATA
    # ==========================================================

    print("\n" + "=" * 80)
    print("ACTOR METADATA CHECK")
    print("=" * 80)


    actors_text = str(
        target.get("actors", "")
    )


    if "ryan gosling" in actors_text.lower():

        print(
            "\nPASS: Ryan Gosling exists in "
            "Project Hail Mary's actor metadata."
        )

    else:

        print(
            "\nWARNING: Ryan Gosling was NOT found "
            "in the movie's actor metadata."
        )

        print(
            "\nStored actors:",
            actors_text
        )


    # ==========================================================
    # CONTENT MODEL
    # ==========================================================

    print("\n" + "=" * 80)
    print("CONTENT MODEL")
    print("=" * 80)


    content_scores = engine.content.get_scores(

        favorite_movies=
            FAVORITE_MOVIES,

        favorite_genres=
            [],

        favorite_actors=
            FAVORITE_ACTORS,

        favorite_directors=
            []
    )


    # Make sure we have a clean ranking.
    content_scores = (
        content_scores
        .sort_values(
            "content_score",
            ascending=False
        )
        .reset_index(drop=True)
    )

    content_scores[
        "diagnostic_rank"
    ] = range(
        1,
        len(content_scores) + 1
    )


    print(
        f"\nTop {TOP_N} content recommendations:\n"
    )


    columns_to_show = [
        column
        for column in [
            "diagnostic_rank",
            "title",
            "release_year",
            "story_score",
            "genre_score",
            "actor_score",
            "director_score",
            "content_score",
        ]
        if column in content_scores.columns
    ]


    print(
        content_scores[
            columns_to_show
        ]
        .head(TOP_N)
        .to_string(index=False)
    )


    # ==========================================================
    # TARGET CONTENT RANK
    # ==========================================================

    target_content = content_scores[
        pd.to_numeric(
            content_scores["tmdb_id"],
            errors="coerce"
        )
        == target_tmdb_id
    ]


    print("\n" + "-" * 80)
    print("PROJECT HAIL MARY — CONTENT RESULT")
    print("-" * 80)


    if target_content.empty:

        print(
            "\nProject Hail Mary does NOT appear "
            "in the content score table."
        )

    else:

        row = target_content.iloc[0]

        print(
            "\nContent rank:",
            int(row["diagnostic_rank"])
        )

        for column in [
            "story_score",
            "genre_score",
            "actor_score",
            "director_score",
            "content_score",
        ]:

            if column in row.index:

                print(
                    f"{column}:",
                    row[column]
                )


    # ==========================================================
    # CHECK FAVORITE TMDB IDS
    # ==========================================================

    print("\n" + "=" * 80)
    print("FAVORITE MOVIE MAPPING")
    print("=" * 80)


    favorite_tmdb_ids = (
        engine.get_favorite_tmdb_ids(
            FAVORITE_MOVIES
        )
    )


    print(
        "\nFavorite TMDB IDs:",
        favorite_tmdb_ids
    )


    usable_svd_favorites = (
        engine.get_usable_svd_favorites(
            favorite_tmdb_ids
        )
    )


    print(
        "SVD-usable favorites:",
        usable_svd_favorites
    )

    print(
        "Usable favorite count:",
        len(usable_svd_favorites)
    )


    # ==========================================================
    # CHECK IF PROJECT HAIL MARY EXISTS IN SVD
    # ==========================================================

    print("\n" + "=" * 80)
    print("SVD AVAILABILITY")
    print("=" * 80)


    mapping = engine.svd.mapping.copy()


    mapping_tmdb = pd.to_numeric(
        mapping["tmdb_id"],
        errors="coerce"
    )


    target_mapping = mapping[
        mapping_tmdb == target_tmdb_id
    ]


    if target_mapping.empty:

        print(
            "\nProject Hail Mary has NO "
            "MovieLens/TMDB mapping."
        )

        print(
            "SVD availability: NO"
        )

    else:

        print(
            "\nMovieLens mapping found:"
        )

        print(
            target_mapping.to_string(
                index=False
            )
        )


        usable = False

        if "movieId" in target_mapping.columns:

            for movie_id in (
                target_mapping["movieId"]
                .dropna()
                .tolist()
            ):

                try:
                    movie_id = int(movie_id)

                    if (
                        movie_id
                        in engine.svd.movie_to_index
                    ):
                        usable = True

                except Exception:
                    pass


        print(
            "\nSVD availability:",
            "YES" if usable else "NO"
        )


    # ==========================================================
    # FINAL RECOMMENDATION ENGINE
    # ==========================================================

    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATION ENGINE")
    print("=" * 80)


    final_results = engine.recommend(

        favorite_movies=
            FAVORITE_MOVIES,

        favorite_genres=
            [],

        favorite_actors=
            FAVORITE_ACTORS,

        favorite_directors=
            [],

        top_n=10
    )


    print("\nFinal Top 10:\n")


    final_columns = [
        column
        for column in [
            "recommendation_rank",
            "title",
            "release_year",
            "recommendation_source",
            "content_score",
            "svd_score",
            "final_score",
        ]
        if column in final_results.columns
    ]


    print(
        final_results[
            final_columns
        ].to_string(index=False)
    )


    # ==========================================================
    # CHECK TARGET IN FINAL RESULTS
    # ==========================================================

    final_target = final_results[
        pd.to_numeric(
            final_results["tmdb_id"],
            errors="coerce"
        )
        == target_tmdb_id
    ]


    print("\n" + "-" * 80)
    print("FINAL PROJECT HAIL MARY RESULT")
    print("-" * 80)


    if final_target.empty:

        print(
            "\nProject Hail Mary is NOT "
            "in the final Top 10."
        )

    else:

        row = final_target.iloc[0]

        if "recommendation_rank" in row.index:

            print(
                "\nProject Hail Mary final rank:",
                row["recommendation_rank"]
            )

        else:

            print(
                "\nProject Hail Mary IS "
                "in the final Top 10."
            )


    # ==========================================================
    # SUMMARY
    # ==========================================================

    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)


    if not target_content.empty:

        content_rank = int(
            target_content.iloc[0][
                "diagnostic_rank"
            ]
        )

        print(
            "\nProject Hail Mary content rank:",
            content_rank
        )

    else:

        print(
            "\nProject Hail Mary content rank: "
            "NOT FOUND"
        )


    print(
        "Project Hail Mary in final Top 10:",
        "YES"
        if not final_target.empty
        else "NO"
    )


    print("\nDiagnostic complete.")

if __name__ == "__main__":
    main()
