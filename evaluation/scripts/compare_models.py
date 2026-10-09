from backend.core.paths import project_path
import pandas as pd
import numpy as np

from backend.recommendation.hybrid_recommender import HybridRecommender


# ==========================================================
# SETTINGS
# ==========================================================

RATINGS_FILE = project_path('data/raw/ml-32m/ratings.csv')
LINKS_FILE = project_path('data/raw/ml-32m/links.csv')

POSITIVE_RATING = 4.0
MIN_POSITIVE_MOVIES = 10

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
# HELPER FOR METRICS
# ==========================================================

def calculate_metrics(
    recommendations,
    relevant_tmdb_ids
):

    recall = recall_at_k(
        recommendations,
        relevant_tmdb_ids,
        TOP_K
    )

    ndcg = ndcg_at_k(
        recommendations,
        relevant_tmdb_ids,
        TOP_K
    )

    hits = len(
        set(recommendations[:TOP_K])
        .intersection(
            set(relevant_tmdb_ids)
        )
    )

    return hits, recall, ndcg


# ==========================================================
# LOAD DATA
# ==========================================================

print(
    "Loading MovieLens data..."
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
# MOVIELENS -> TMDB MAPPING
# ==========================================================

links = (
    links
    .dropna(
        subset=["tmdbId"]
    )
    .copy()
)

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
# POSITIVE RATINGS
# ==========================================================

positive_ratings = (

    ratings[
        ratings["rating"]
        >= POSITIVE_RATING
    ]

    .copy()
)

positive_ratings = (

    positive_ratings[
        positive_ratings["movieId"]
        .isin(movie_to_tmdb)
    ]

    .copy()
)

print(
    "\nPositive ratings:",
    len(positive_ratings)
)


# ==========================================================
# INITIALIZE MODELS ONCE
# ==========================================================

print(
    "\nInitializing models..."
)

recommender = HybridRecommender()


# ==========================================================
# TMDB CATALOG
# ==========================================================

catalog = (
    recommender
    .content
    .movies
    .copy()
)

catalog["tmdb_id"] = (
    catalog["tmdb_id"]
    .astype(int)
)

catalog_tmdb_ids = set(
    catalog["tmdb_id"]
    .tolist()
)


# ==========================================================
# KEEP POSITIVE MOVIES THAT EXIST IN OUR TMDB CATALOG
# ==========================================================

positive_ratings["tmdb_id"] = (

    positive_ratings["movieId"]
    .map(movie_to_tmdb)
)

positive_ratings = (

    positive_ratings
    .dropna(
        subset=["tmdb_id"]
    )
    .copy()
)

positive_ratings["tmdb_id"] = (
    positive_ratings["tmdb_id"]
    .astype(int)
)

positive_ratings = (

    positive_ratings[
        positive_ratings["tmdb_id"]
        .isin(catalog_tmdb_ids)
    ]

    .copy()
)


# ==========================================================
# FIND ELIGIBLE USERS AFTER CATALOG FILTERING
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
# SAMPLE SAME 100 USERS
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
# RESULTS
# ==========================================================

results = []


print("\n")
print("=" * 80)
print("STARTING FAIR MODEL COMPARISON")
print("=" * 80)


# ==========================================================
# EVALUATE USERS
# ==========================================================

for position, user_id in enumerate(
    eval_users,
    start=1
):

    # ------------------------------------------------------
    # USER'S POSITIVE MOVIES
    # ------------------------------------------------------

    user_positive = (

        positive_ratings[
            positive_ratings["userId"]
            == user_id
        ]

        .copy()
    )


    # ------------------------------------------------------
    # SHUFFLE REPRODUCIBLY
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
    # 80 / 20 SPLIT
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


    favorite_tmdb_ids = (
        favorite_movies[
            "tmdb_id"
        ]
        .astype(int)
        .tolist()
    )

    relevant_tmdb_ids = (
        test_movies[
            "tmdb_id"
        ]
        .astype(int)
        .tolist()
    )


    # ======================================================
    # GET FAVORITE TITLES
    # ======================================================

    favorite_titles = (

        catalog[
            catalog["tmdb_id"]
            .isin(
                favorite_tmdb_ids
            )
        ]

        ["title"]

        .tolist()
    )

    if len(favorite_titles) == 0:
        continue


    # ======================================================
    # 1. CONTENT-ONLY
    # ======================================================

    content_results = (
        recommender.content.get_scores(

            favorite_movies=
                favorite_titles,

            favorite_genres=[],

            favorite_actors=[],

            favorite_directors=[],

            top_n=TOP_K
        )
    )

    content_ids = (

        content_results[
            "tmdb_id"
        ]

        .astype(int)

        .tolist()
    )


    # ======================================================
    # 2. NEW-USER SVD
    # ======================================================

    svd_results = (

        recommender.svd
        .get_scores_for_favorite_tmdb_movies(

            favorite_tmdb_ids,

            top_n=TOP_K
        )
    )


    if svd_results.empty:

        svd_ids = []

    else:

        svd_ids = (

            svd_results[
                "tmdb_id"
            ]

            .astype(int)

            .tolist()
        )


    # ======================================================
    # 3. HYBRID V2
    # ======================================================

    hybrid_results = (
        recommender.recommend(

            favorite_movies=
                favorite_titles,

            favorite_genres=[],

            favorite_actors=[],

            favorite_directors=[],

            top_n=TOP_K
        )
    )

    hybrid_ids = (

        hybrid_results[
            "tmdb_id"
        ]

        .astype(int)

        .tolist()
    )


    # ======================================================
    # CALCULATE METRICS
    # ======================================================

    (
        content_hits,
        content_recall,
        content_ndcg

    ) = calculate_metrics(

        content_ids,
        relevant_tmdb_ids
    )


    (
        svd_hits,
        svd_recall,
        svd_ndcg

    ) = calculate_metrics(

        svd_ids,
        relevant_tmdb_ids
    )


    (
        hybrid_hits,
        hybrid_recall,
        hybrid_ndcg

    ) = calculate_metrics(

        hybrid_ids,
        relevant_tmdb_ids
    )


    # ======================================================
    # SAVE USER RESULT
    # ======================================================

    results.append({

        "userId":
            int(user_id),

        "favorites":
            len(favorite_titles),

        "hidden_movies":
            len(relevant_tmdb_ids),

        # Content
        "content_hits":
            content_hits,

        "content_recall":
            content_recall,

        "content_ndcg":
            content_ndcg,

        # SVD
        "svd_hits":
            svd_hits,

        "svd_recall":
            svd_recall,

        "svd_ndcg":
            svd_ndcg,

        # Hybrid
        "hybrid_hits":
            hybrid_hits,

        "hybrid_recall":
            hybrid_recall,

        "hybrid_ndcg":
            hybrid_ndcg
    })


    # ======================================================
    # PROGRESS
    # ======================================================

    print(

        f"User {position}/{len(eval_users)} | "

        f"C hits={content_hits} | "

        f"SVD hits={svd_hits} | "

        f"H hits={hybrid_hits}"
    )


# ==========================================================
# CREATE RESULTS DATAFRAME
# ==========================================================

results_df = pd.DataFrame(
    results
)


# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n")
print("=" * 80)
print("FAIR MODEL COMPARISON")
print("=" * 80)


if results_df.empty:

    print(
        "No users were successfully evaluated."
    )

else:

    content_recall = (
        results_df[
            "content_recall"
        ]
        .mean()
    )

    content_ndcg = (
        results_df[
            "content_ndcg"
        ]
        .mean()
    )


    svd_recall = (
        results_df[
            "svd_recall"
        ]
        .mean()
    )

    svd_ndcg = (
        results_df[
            "svd_ndcg"
        ]
        .mean()
    )


    hybrid_recall = (
        results_df[
            "hybrid_recall"
        ]
        .mean()
    )

    hybrid_ndcg = (
        results_df[
            "hybrid_ndcg"
        ]
        .mean()
    )


    print(
        "Users evaluated:",
        len(results_df)
    )

    print()

    print(
        f"{'Model':<20}"
        f"{'Recall@10':>15}"
        f"{'NDCG@10':>15}"
        f"{'Total Hits':>15}"
    )

    print(
        "-" * 65
    )

    print(
        f"{'Content':<20}"
        f"{content_recall:>15.6f}"
        f"{content_ndcg:>15.6f}"
        f"{results_df['content_hits'].sum():>15}"
    )

    print(
        f"{'SVD New-User':<20}"
        f"{svd_recall:>15.6f}"
        f"{svd_ndcg:>15.6f}"
        f"{results_df['svd_hits'].sum():>15}"
    )

    print(
        f"{'Hybrid V2':<20}"
        f"{hybrid_recall:>15.6f}"
        f"{hybrid_ndcg:>15.6f}"
        f"{results_df['hybrid_hits'].sum():>15}"
    )


    # ======================================================
    # SAVE DETAILED RESULTS
    # ======================================================

    results_df.to_csv(
        project_path('evaluation/results/model_comparison_results.csv'),
        index=False
    )


    # ======================================================
    # SAVE SUMMARY
    # ======================================================

    summary_df = pd.DataFrame({

        "Model": [
            "Content",
            "SVD New-User",
            "Hybrid V2"
        ],

        "Recall@10": [
            content_recall,
            svd_recall,
            hybrid_recall
        ],

        "NDCG@10": [
            content_ndcg,
            svd_ndcg,
            hybrid_ndcg
        ],

        "Total Hits": [

            results_df[
                "content_hits"
            ].sum(),

            results_df[
                "svd_hits"
            ].sum(),

            results_df[
                "hybrid_hits"
            ].sum()
        ]
    })


    summary_df.to_csv(
        project_path('evaluation/results/model_comparison_summary.csv'),
        index=False
    )


    print(
        "\nSaved:"
    )

    print(
        project_path('evaluation/results/model_comparison_results.csv')
    )

    print(
        project_path('evaluation/results/model_comparison_summary.csv')
    )