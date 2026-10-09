from backend.core.paths import project_path
import pandas as pd
import numpy as np

from backend.recommendation.hybrid_recommender import HybridRecommender


# ==========================================================
# SETTINGS
# ==========================================================

RATINGS_FILE = project_path('data/raw/ml-32m/ratings.csv')
LINKS_FILE = project_path('data/raw/ml-32m/links.csv')

# Positive rating = movie the user liked
POSITIVE_RATING = 4.0

# User must have enough liked movies for us to split
MIN_POSITIVE_MOVIES = 10

# Start small because hybrid evaluation is much heavier
# than SVD-only evaluation.
N_EVAL_USERS = 100

TEST_FRACTION = 0.20

TOP_K = 10

RANDOM_STATE = 42


# ==========================================================
# RECALL@K
# ==========================================================

def recall_at_k(
    recommended_tmdb_ids,
    relevant_tmdb_ids,
    k=10
):

    recommended = set(
        recommended_tmdb_ids[:k]
    )

    relevant = set(
        relevant_tmdb_ids
    )

    if len(relevant) == 0:
        return 0.0

    hits = len(
        recommended.intersection(
            relevant
        )
    )

    return hits / len(relevant)


# ==========================================================
# NDCG@K
# ==========================================================

def ndcg_at_k(
    recommended_tmdb_ids,
    relevant_tmdb_ids,
    k=10
):

    relevant = set(
        relevant_tmdb_ids
    )

    if len(relevant) == 0:
        return 0.0

    dcg = 0.0

    for rank, movie_id in enumerate(
        recommended_tmdb_ids[:k],
        start=1
    ):

        if movie_id in relevant:

            dcg += (
                1.0
                / np.log2(rank + 1)
            )

    # ------------------------------------------------------
    # Ideal ranking
    # ------------------------------------------------------

    ideal_hits = min(
        len(relevant),
        k
    )

    idcg = sum(

        1.0 / np.log2(rank + 1)

        for rank in range(
            1,
            ideal_hits + 1
        )
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg


# ==========================================================
# LOAD MOVIELENS DATA
# ==========================================================

print(
    "Loading MovieLens ratings..."
)

ratings = pd.read_csv(
    RATINGS_FILE
)

links = pd.read_csv(
    LINKS_FILE
)


print(
    "Ratings:",
    len(ratings)
)

print(
    "Users:",
    ratings["userId"].nunique()
)


# ==========================================================
# PREPARE MOVIELENS -> TMDB MAPPING
# ==========================================================

links = links.dropna(
    subset=["tmdbId"]
).copy()


links["tmdbId"] = (
    links["tmdbId"]
    .astype(int)
)


movie_to_tmdb = dict(

    zip(
        links["movieId"],
        links["tmdbId"]
    )
)


# ==========================================================
# KEEP POSITIVE RATINGS
# ==========================================================

positive_ratings = ratings[
    ratings["rating"]
    >= POSITIVE_RATING
].copy()


print(
    "\nPositive ratings:",
    len(positive_ratings)
)


# ==========================================================
# ONLY MOVIES THAT HAVE A TMDB ID
# ==========================================================

positive_ratings = positive_ratings[
    positive_ratings["movieId"]
    .isin(movie_to_tmdb)
].copy()


print(
    "Positive ratings with TMDB mapping:",
    len(positive_ratings)
)


# ==========================================================
# FIND ELIGIBLE USERS
# ==========================================================

positive_counts = (

    positive_ratings

    .groupby("userId")

    .size()
)


eligible_users = (

    positive_counts[
        positive_counts
        >= MIN_POSITIVE_MOVIES
    ]

    .index
    .to_numpy()
)


print(
    "Eligible users:",
    len(eligible_users)
)


# ==========================================================
# SAMPLE EVALUATION USERS
# ==========================================================

rng = np.random.default_rng(
    RANDOM_STATE
)


number_to_sample = min(
    N_EVAL_USERS,
    len(eligible_users)
)


eval_users = rng.choice(

    eligible_users,

    size=number_to_sample,

    replace=False
)


print(
    "Evaluation users:",
    len(eval_users)
)


# ==========================================================
# INITIALIZE HYBRID MODEL ONCE
# ==========================================================

print(
    "\nInitializing hybrid model..."
)


recommender = HybridRecommender()


# ==========================================================
# TMDB CATALOG IDS
# ==========================================================

catalog_tmdb_ids = set(

    recommender
    .content
    .movies["tmdb_id"]
    .astype(int)
    .tolist()
)


# ==========================================================
# EVALUATION
# ==========================================================

results = []


print(
    "\nStarting hybrid evaluation..."
)

print(
    "=" * 70
)


for position, user_id in enumerate(
    eval_users,
    start=1
):

    # ------------------------------------------------------
    # Get user's positive movies
    # ------------------------------------------------------

    user_positive = (

        positive_ratings[
            positive_ratings["userId"]
            == user_id
        ]

        .copy()
    )


    # ------------------------------------------------------
    # Convert MovieLens IDs -> TMDB IDs
    # ------------------------------------------------------

    user_positive["tmdb_id"] = (

        user_positive["movieId"]
        .map(movie_to_tmdb)
    )


    user_positive = (
        user_positive
        .dropna(
            subset=["tmdb_id"]
        )
        .copy()
    )


    user_positive["tmdb_id"] = (
        user_positive["tmdb_id"]
        .astype(int)
    )


    # ------------------------------------------------------
    # Keep only movies that exist in our TMDB catalog
    # ------------------------------------------------------

    user_positive = user_positive[
        user_positive["tmdb_id"]
        .isin(catalog_tmdb_ids)
    ].copy()


    if len(user_positive) < MIN_POSITIVE_MOVIES:

        continue


    # ------------------------------------------------------
    # Shuffle user's movies reproducibly
    # ------------------------------------------------------

    user_positive = (

        user_positive

        .sample(
            frac=1.0,
            random_state=(
                RANDOM_STATE
                + int(user_id)
            )
        )

        .reset_index(drop=True)
    )


    # ------------------------------------------------------
    # 80 / 20 split
    # ------------------------------------------------------

    test_size = max(

        1,

        int(
            round(
                len(user_positive)
                * TEST_FRACTION
            )
        )
    )


    test_movies = (
        user_positive
        .iloc[:test_size]
        .copy()
    )


    favorite_movies = (
        user_positive
        .iloc[test_size:]
        .copy()
    )


    if favorite_movies.empty:
        continue


    # ------------------------------------------------------
    # IMPORTANT:
    #
    # Hybrid recommender currently accepts favorite movie
    # titles.
    #
    # We therefore look up the exact TMDB titles for the
    # training/favorite movies.
    # ------------------------------------------------------

    favorite_tmdb_ids = (

        favorite_movies[
            "tmdb_id"
        ]

        .tolist()
    )


    favorite_titles = (

        recommender
        .content
        .movies[

            recommender
            .content
            .movies["tmdb_id"]
            .astype(int)
            .isin(
                favorite_tmdb_ids
            )
        ]

        ["title"]

        .tolist()
    )


    relevant_tmdb_ids = (

        test_movies[
            "tmdb_id"
        ]

        .tolist()
    )


    if len(favorite_titles) == 0:
        continue


    # ------------------------------------------------------
    # GET HYBRID TOP 10
    # ------------------------------------------------------

    try:

        recommendations = (
            recommender.recommend(

                favorite_movies=
                    favorite_titles,

                favorite_genres=[],

                favorite_actors=[],

                favorite_directors=[],

                top_n=TOP_K
            )
        )

    except Exception as error:

        print(
            f"\nUser {user_id} failed:"
        )

        print(error)

        continue


    if recommendations.empty:
        continue


    recommended_tmdb_ids = (

        recommendations[
            "tmdb_id"
        ]

        .astype(int)

        .tolist()
    )


    # ------------------------------------------------------
    # METRICS
    # ------------------------------------------------------

    recall = recall_at_k(

        recommended_tmdb_ids,

        relevant_tmdb_ids,

        TOP_K
    )


    ndcg = ndcg_at_k(

        recommended_tmdb_ids,

        relevant_tmdb_ids,

        TOP_K
    )


    hits = len(

        set(
            recommended_tmdb_ids
        )

        .intersection(

            set(
                relevant_tmdb_ids
            )
        )
    )


    results.append({

        "userId":
            int(user_id),

        "favorites":
            len(favorite_titles),

        "hidden_movies":
            len(relevant_tmdb_ids),

        "hits_at_10":
            hits,

        "recall_at_10":
            recall,

        "ndcg_at_10":
            ndcg
    })


    # ------------------------------------------------------
    # PROGRESS
    # ------------------------------------------------------

    print(

        f"User {position}/{len(eval_users)} | "
        f"favorites={len(favorite_titles)} | "
        f"hidden={len(relevant_tmdb_ids)} | "
        f"hits={hits} | "
        f"Recall@10={recall:.4f} | "
        f"NDCG@10={ndcg:.4f}"
    )


# ==========================================================
# FINAL RESULTS
# ==========================================================

results_df = pd.DataFrame(
    results
)


print("\n")
print("=" * 70)

print(
    "HYBRID EVALUATION RESULTS"
)

print("=" * 70)


if results_df.empty:

    print(
        "No users were successfully evaluated."
    )


else:

    mean_recall = (
        results_df[
            "recall_at_10"
        ]
        .mean()
    )


    mean_ndcg = (
        results_df[
            "ndcg_at_10"
        ]
        .mean()
    )


    total_hits = (
        results_df[
            "hits_at_10"
        ]
        .sum()
    )


    print(
        "Users evaluated:",
        len(results_df)
    )


    print(
        "Total hits:",
        total_hits
    )


    print(
        f"Mean Recall@10: "
        f"{mean_recall:.6f}"
    )


    print(
        f"Mean NDCG@10: "
        f"{mean_ndcg:.6f}"
    )


    # ------------------------------------------------------
    # SAVE RESULTS
    # ------------------------------------------------------

    output_file = (
        project_path('evaluation/results/hybrid_evaluation_results.csv')
    )


    results_df.to_csv(
        output_file,
        index=False
    )


    print(
        "\nSaved:",
        output_file
    )