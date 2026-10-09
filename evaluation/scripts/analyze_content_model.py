from backend.core.paths import project_path
import pandas as pd
import numpy as np

from backend.recommendation.content_recommender import ContentRecommender


# ==========================================================
# SETTINGS
# ==========================================================

RATINGS_FILE = project_path('data/raw/ml-32m/ratings.csv')
MAPPING_FILE = project_path('data/processed/movie_mapping.csv')
TMDB_FILE = project_path('data/processed/tmdb_movies.csv')

N_EVAL_USERS = 1000
N_FAVORITES = 5
TOP_K = 10
RANDOM_STATE = 42

POSITIVE_RATING = 4.0


# ==========================================================
# NDCG
# ==========================================================

def ndcg_at_k(recommended_ids, relevant_ids, k=10):

    recommended_ids = recommended_ids[:k]

    dcg = 0.0

    for rank, tmdb_id in enumerate(
        recommended_ids,
        start=1
    ):

        if tmdb_id in relevant_ids:

            dcg += 1.0 / np.log2(
                rank + 1
            )

    ideal_hits = min(
        len(relevant_ids),
        k
    )

    if ideal_hits == 0:
        return 0.0

    idcg = sum(
        1.0 / np.log2(rank + 1)

        for rank in range(
            1,
            ideal_hits + 1
        )
    )

    return dcg / idcg


# ==========================================================
# GET TOP K FOR ONE COMPONENT
# ==========================================================

def get_component_top_k(
    scores,
    score_column,
    k
):

    # Only consider movies where this component
    # actually produced a positive score.
    #
    # This prevents zero-score components from
    # accidentally returning the existing dataframe order.

    valid = scores[
        scores[score_column] > 0
    ].copy()

    if valid.empty:
        return []

    valid = valid.sort_values(
        score_column,
        ascending=False
    )

    return (
        valid
        .head(k)["tmdb_id"]
        .astype(int)
        .tolist()
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    print(
        "\n" + "=" * 75
    )

    print(
        "CONTENT MODEL COMPONENT EVALUATION"
    )

    print(
        "=" * 75
    )


    # ======================================================
    # LOAD DATA
    # ======================================================

    print(
        "\nLoading data..."
    )

    ratings = pd.read_csv(
        RATINGS_FILE,
        usecols=[
            "userId",
            "movieId",
            "rating"
        ]
    )

    mapping = pd.read_csv(
        MAPPING_FILE
    )

    tmdb_movies = pd.read_csv(
        TMDB_FILE,
        usecols=[
            "tmdb_id",
            "title",
            "release_year"
        ]
    )


    # ======================================================
    # CLEAN MAPPING
    # ======================================================

    mapping = mapping.dropna(
        subset=["tmdb_id"]
    ).copy()

    mapping["tmdb_id"] = (
        mapping["tmdb_id"]
        .astype(int)
    )

    tmdb_movies = tmdb_movies.dropna(
        subset=["tmdb_id"]
    ).copy()

    tmdb_movies["tmdb_id"] = (
        tmdb_movies["tmdb_id"]
        .astype(int)
    )


    # Keep only mappings that exist
    # in our TMDB catalog.

    valid_tmdb_ids = set(
        tmdb_movies["tmdb_id"]
        .tolist()
    )

    mapping = mapping[
        mapping["tmdb_id"]
        .isin(valid_tmdb_ids)
    ].copy()


    # ======================================================
    # POSITIVE RATINGS
    # ======================================================

    print(
        "Preparing positive ratings..."
    )

    positive = ratings[
        ratings["rating"]
        >= POSITIVE_RATING
    ].copy()


    # Add TMDB IDs

    positive = positive.merge(

        mapping[
            [
                "movieId",
                "tmdb_id"
            ]
        ],

        on="movieId",
        how="inner"
    )


    # ======================================================
    # REMOVE DUPLICATE TMDB MOVIES PER USER
    # ======================================================

    # A user can theoretically have multiple MovieLens
    # movieIds mapping to the same TMDB movie.
    #
    # For this content evaluation we want each TMDB movie
    # to count only once.

    positive = positive.drop_duplicates(
        subset=[
            "userId",
            "tmdb_id"
        ]
    )


    # ======================================================
    # ELIGIBLE USERS
    # ======================================================

    positive_counts = (
        positive
        .groupby("userId")
        .size()
    )

    eligible_users = (
        positive_counts[
            positive_counts
            >= N_FAVORITES + 1
        ]
        .index
        .to_numpy()
    )

    print(
        "Eligible users:",
        len(eligible_users)
    )


    # ======================================================
    # SAMPLE EVALUATION USERS
    # ======================================================

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    if len(eligible_users) > N_EVAL_USERS:

        eval_users = rng.choice(
            eligible_users,
            size=N_EVAL_USERS,
            replace=False
        )

    else:

        eval_users = eligible_users


    print(
        "Evaluation users:",
        len(eval_users)
    )

    print(
        "Favorites per user:",
        N_FAVORITES
    )


    # ======================================================
    # TITLE LOOKUP
    # ======================================================

    title_lookup = (

        tmdb_movies[
            [
                "tmdb_id",
                "title"
            ]
        ]

        .drop_duplicates(
            subset=["tmdb_id"]
        )

        .set_index("tmdb_id")[
            "title"
        ]

        .to_dict()
    )


    # ======================================================
    # LOAD CONTENT MODEL
    # ======================================================

    print(
        "\nLoading content model..."
    )

    content_model = (
        ContentRecommender()
    )


    # ======================================================
    # RESULT STORAGE
    # ======================================================

    model_names = [

        "Story Only",
        "Genre Only",
        "Actor Only",
        "Director Only",
        "Current Combined"
    ]


    recalls = {

        name: []

        for name
        in model_names
    }


    ndcgs = {

        name: []

        for name
        in model_names
    }


    hits = {

        name: 0

        for name
        in model_names
    }


    evaluated_users = 0
    skipped_users = 0


    # ======================================================
    # EVALUATE USERS
    # ======================================================

    print(
        "\nEvaluating users..."
    )


    for user_number, user_id in enumerate(
        eval_users,
        start=1
    ):

        user_tmdb_ids = (
            positive.loc[
                positive["userId"] == user_id,
                "tmdb_id"
            ]
            .drop_duplicates()
            .astype(int)
            .tolist()
        )

        if len(user_tmdb_ids) < (
            N_FAVORITES + 1
        ):
            skipped_users += 1
            continue


        # --------------------------------------------------
        # Deterministic random favorite selection
        # --------------------------------------------------

        user_rng = np.random.default_rng(
            RANDOM_STATE + int(user_id)
        )

        favorite_tmdb_ids = (

            user_rng.choice(
                user_tmdb_ids,
                size=N_FAVORITES,
                replace=False
            )

            .astype(int)
            .tolist()
        )


        # Everything else is hidden/relevant.

        favorite_set = set(
            favorite_tmdb_ids
        )

        hidden_tmdb_ids = set(

            tmdb_id

            for tmdb_id
            in user_tmdb_ids

            if tmdb_id
            not in favorite_set
        )


        if not hidden_tmdb_ids:

            skipped_users += 1
            continue


        # --------------------------------------------------
        # Convert favorite TMDB IDs to titles
        # --------------------------------------------------

        favorite_titles = [

            title_lookup[tmdb_id]

            for tmdb_id
            in favorite_tmdb_ids

            if tmdb_id
            in title_lookup
        ]


        if len(favorite_titles) != N_FAVORITES:

            skipped_users += 1
            continue


        # --------------------------------------------------
        # Calculate content scores
        # --------------------------------------------------

        scores = (
            content_model.get_scores(

                favorite_movies=(
                    favorite_titles
                ),

                favorite_genres=[],

                favorite_actors=[],

                favorite_directors=[],

                top_n=None
            )
        )


        if scores.empty:

            skipped_users += 1
            continue


        # --------------------------------------------------
        # STORY
        # --------------------------------------------------

        story_top = get_component_top_k(
            scores,
            "story_score",
            TOP_K
        )


        # --------------------------------------------------
        # GENRE
        # --------------------------------------------------

        genre_top = get_component_top_k(
            scores,
            "genre_score",
            TOP_K
        )


        # --------------------------------------------------
        # ACTOR
        # --------------------------------------------------

        actor_top = get_component_top_k(
            scores,
            "actor_score",
            TOP_K
        )


        # --------------------------------------------------
        # DIRECTOR
        # --------------------------------------------------

        director_top = (
            get_component_top_k(
                scores,
                "director_score",
                TOP_K
            )
        )


        # --------------------------------------------------
        # COMBINED
        # --------------------------------------------------

        combined_top = (
            get_component_top_k(
                scores,
                "content_score",
                TOP_K
            )
        )


        recommendations = {

            "Story Only":
                story_top,

            "Genre Only":
                genre_top,

            "Actor Only":
                actor_top,

            "Director Only":
                director_top,

            "Current Combined":
                combined_top
        }


        # --------------------------------------------------
        # METRICS
        # --------------------------------------------------

        for model_name, recommended in (
            recommendations.items()
        ):

            model_hits = sum(

                1

                for tmdb_id
                in recommended

                if tmdb_id
                in hidden_tmdb_ids
            )


            recall = (

                model_hits

                / len(
                    hidden_tmdb_ids
                )
            )


            ndcg = ndcg_at_k(

                recommended,
                hidden_tmdb_ids,
                TOP_K
            )


            recalls[
                model_name
            ].append(
                recall
            )


            ndcgs[
                model_name
            ].append(
                ndcg
            )


            hits[
                model_name
            ] += model_hits


        evaluated_users += 1


        # --------------------------------------------------
        # PROGRESS
        # --------------------------------------------------

        if (
            user_number % 50 == 0
            or user_number
            == len(eval_users)
        ):

            print(
                f"Processed "
                f"{user_number}/"
                f"{len(eval_users)} users"
            )


    # ======================================================
    # FINAL RESULTS
    # ======================================================

    print(
        "\n" + "=" * 75
    )

    print(
        "CONTENT COMPONENT RESULTS"
    )

    print(
        "=" * 75
    )

    print(
        f"\nUsers evaluated: "
        f"{evaluated_users}"
    )

    print(
        f"Users skipped: "
        f"{skipped_users}"
    )

    print(
        f"Favorites per user: "
        f"{N_FAVORITES}"
    )


    print(
        "\n"
        f"{'Model':<25}"
        f"{'Recall@10':>15}"
        f"{'NDCG@10':>15}"
        f"{'Total Hits':>15}"
    )

    print(
        "-" * 70
    )


    summary_rows = []


    for model_name in model_names:

        mean_recall = (

            np.mean(
                recalls[
                    model_name
                ]
            )

            if recalls[
                model_name
            ]

            else 0.0
        )


        mean_ndcg = (

            np.mean(
                ndcgs[
                    model_name
                ]
            )

            if ndcgs[
                model_name
            ]

            else 0.0
        )


        total_hits = (
            hits[
                model_name
            ]
        )


        print(

            f"{model_name:<25}"

            f"{mean_recall:>15.6f}"

            f"{mean_ndcg:>15.6f}"

            f"{total_hits:>15}"
        )


        summary_rows.append({

            "model":
                model_name,

            "recall_at_10":
                mean_recall,

            "ndcg_at_10":
                mean_ndcg,

            "total_hits":
                total_hits,

            "users_evaluated":
                evaluated_users,

            "favorites_per_user":
                N_FAVORITES
        })


    # ======================================================
    # SAVE RESULTS
    # ======================================================

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        project_path('evaluation/results/content_component_analysis.csv'),
        index=False
    )


    print(
        "\nSaved: "
        "content_component_analysis.csv"
    )

    print(
        "\nEvaluation complete."
    )


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":

    main()