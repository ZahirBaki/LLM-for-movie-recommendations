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

N_EVAL_USERS = 1000
TEST_FRACTION = 0.20

TOP_K = 10
CANDIDATES_PER_MODEL = 200

RANDOM_STATE = 42


# ==========================================================
# WEIGHTS TO TEST
# ==========================================================

CONTENT_WEIGHTS = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0
]


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
# CREATE RANK SCORE
# ==========================================================

def add_rank_score(
    dataframe,
    score_column,
    rank_column,
    rank_score_column
):

    dataframe = (

        dataframe

        .sort_values(
            score_column,
            ascending=False
        )

        .reset_index(drop=True)

        .copy()
    )

    dataframe[rank_column] = (
        dataframe.index + 1
    )

    number_of_movies = len(
        dataframe
    )

    if number_of_movies > 1:

        dataframe[
            rank_score_column
        ] = (

            number_of_movies
            - dataframe[rank_column]

        ) / (

            number_of_movies - 1
        )

    else:

        dataframe[
            rank_score_column
        ] = 1.0

    return dataframe


# ==========================================================
# LOAD MOVIELENS DATA
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


# ==========================================================
# INITIALIZE MODELS ONCE
# ==========================================================

print(
    "\nInitializing recommender models..."
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
# KEEP ONLY MOVIES AVAILABLE IN TMDB CATALOG
# ==========================================================

positive_ratings = (

    positive_ratings[
        positive_ratings["tmdb_id"]
        .isin(catalog_tmdb_ids)
    ]

    .copy()
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
# SAMPLE USERS
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
# STORAGE
# ==========================================================
#
# Each weight will have its own list of:
#
# Recall@10
# NDCG@10
# Hits
#
# ==========================================================

weight_results = {}

for content_weight in CONTENT_WEIGHTS:

    svd_weight = (
        1.0 - content_weight
    )

    weight_results[
        content_weight
    ] = {

        "svd_weight":
            svd_weight,

        "recalls":
            [],

        "ndcgs":
            [],

        "hits":
            []
    }


# ==========================================================
# START EVALUATION
# ==========================================================

print("\n")
print("=" * 80)
print("TUNING HYBRID WEIGHTS")
print("=" * 80)


successful_users = 0


# ==========================================================
# LOOP THROUGH USERS
# ==========================================================

for position, user_id in enumerate(
    eval_users,
    start=1
):

    # ------------------------------------------------------
    # USER POSITIVE MOVIES
    # ------------------------------------------------------

    user_positive = (

        positive_ratings[
            positive_ratings["userId"]
            == user_id
        ]

        .copy()
    )


    # ------------------------------------------------------
    # SAME REPRODUCIBLE SHUFFLE
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
    # SAME 80 / 20 SPLIT
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
    # CONTENT SCORES
    #
    # IMPORTANT:
    # Only calculated ONCE for this user.
    # ======================================================

    content_scores = (

        recommender
        .content
        .get_scores(

            favorite_movies=
                favorite_titles,

            favorite_genres=[],

            favorite_actors=[],

            favorite_directors=[],

            top_n=None
        )
    )


    if content_scores.empty:
        continue


    # ======================================================
    # CONTENT TOP 200
    # ======================================================

    content_candidates = (

        content_scores

        .sort_values(
            "content_score",
            ascending=False
        )

        .head(
            CANDIDATES_PER_MODEL
        )

        .copy()
    )


    content_candidates = (
        add_rank_score(

            content_candidates,

            "content_score",

            "content_rank",

            "content_rank_score"
        )
    )


    # ======================================================
    # SVD SCORES
    #
    # Also calculated ONCE for this user.
    # ======================================================

    svd_scores = (

        recommender
        .svd
        .get_scores_for_favorite_tmdb_movies(

            favorite_tmdb_ids,

            top_n=None
        )
    )


    if svd_scores.empty:
        continue


    # ======================================================
    # SVD TOP 200
    # ======================================================

    svd_candidates = (

        svd_scores

        .sort_values(
            "svd_score",
            ascending=False
        )

        .head(
            CANDIDATES_PER_MODEL
        )

        .copy()
    )


    svd_candidates = (
        add_rank_score(

            svd_candidates,

            "svd_score",

            "svd_rank",

            "svd_rank_score"
        )
    )


    # ======================================================
    # PREPARE CONTENT CANDIDATES
    # ======================================================

    content_part = (

        content_candidates[
            [
                "tmdb_id",
                "content_rank_score"
            ]
        ]

        .copy()
    )


    content_part["tmdb_id"] = (
        content_part["tmdb_id"]
        .astype(int)
    )


    # ======================================================
    # PREPARE SVD CANDIDATES
    # ======================================================

    svd_part = (

        svd_candidates[
            [
                "tmdb_id",
                "svd_rank_score"
            ]
        ]

        .copy()
    )


    svd_part["tmdb_id"] = (
        svd_part["tmdb_id"]
        .astype(int)
    )


    # ======================================================
    # CREATE UNION OF BOTH CANDIDATE SETS
    # ======================================================

    candidates = content_part.merge(

        svd_part,

        on="tmdb_id",

        how="outer"
    )


    # ------------------------------------------------------
    # Not nominated by Content -> 0
    #
    # Not nominated by SVD -> 0
    # ------------------------------------------------------

    candidates[
        "content_rank_score"
    ] = (

        candidates[
            "content_rank_score"
        ]

        .fillna(0.0)
    )


    candidates[
        "svd_rank_score"
    ] = (

        candidates[
            "svd_rank_score"
        ]

        .fillna(0.0)
    )


    # ======================================================
    # TEST EVERY WEIGHT
    # ======================================================

    for content_weight in CONTENT_WEIGHTS:

        svd_weight = (
            1.0 - content_weight
        )


        # --------------------------------------------------
        # Hybrid score for THIS weight combination
        # --------------------------------------------------

        candidates[
            "temporary_hybrid_score"
        ] = (

            candidates[
                "content_rank_score"
            ]

            * content_weight

            +

            candidates[
                "svd_rank_score"
            ]

            * svd_weight
        )


        # --------------------------------------------------
        # Top 10
        # --------------------------------------------------

        top_recommendations = (

            candidates

            .sort_values(
                "temporary_hybrid_score",
                ascending=False
            )

            .head(TOP_K)
        )


        recommended_ids = (

            top_recommendations[
                "tmdb_id"
            ]

            .astype(int)

            .tolist()
        )


        # --------------------------------------------------
        # Recall
        # --------------------------------------------------

        recall = recall_at_k(

            recommended_ids,

            relevant_tmdb_ids,

            TOP_K
        )


        # --------------------------------------------------
        # NDCG
        # --------------------------------------------------

        ndcg = ndcg_at_k(

            recommended_ids,

            relevant_tmdb_ids,

            TOP_K
        )


        # --------------------------------------------------
        # Hits
        # --------------------------------------------------

        hits = len(

            set(
                recommended_ids
            )

            .intersection(

                set(
                    relevant_tmdb_ids
                )
            )
        )


        # --------------------------------------------------
        # Store
        # --------------------------------------------------

        weight_results[
            content_weight
        ][
            "recalls"
        ].append(
            recall
        )


        weight_results[
            content_weight
        ][
            "ndcgs"
        ].append(
            ndcg
        )


        weight_results[
            content_weight
        ][
            "hits"
        ].append(
            hits
        )


    successful_users += 1


    # ======================================================
    # PROGRESS
    # ======================================================

    print(

        f"User "
        f"{position}/{len(eval_users)} "
        f"completed"
    )


# ==========================================================
# BUILD SUMMARY
# ==========================================================

summary_rows = []


for content_weight in CONTENT_WEIGHTS:

    data = weight_results[
        content_weight
    ]

    svd_weight = (
        data["svd_weight"]
    )


    mean_recall = np.mean(
        data["recalls"]
    )


    mean_ndcg = np.mean(
        data["ndcgs"]
    )


    total_hits = np.sum(
        data["hits"]
    )


    summary_rows.append({

        "Content Weight":
            content_weight,

        "SVD Weight":
            svd_weight,

        "Recall@10":
            mean_recall,

        "NDCG@10":
            mean_ndcg,

        "Total Hits":
            int(total_hits)
    })


summary_df = pd.DataFrame(
    summary_rows
)


# ==========================================================
# PRINT RESULTS
# ==========================================================

print("\n")
print("=" * 85)

print(
    "HYBRID WEIGHT TUNING RESULTS"
)

print("=" * 85)

print(
    "Users evaluated:",
    successful_users
)

print()


print(
    f"{'Content':>10}"
    f"{'SVD':>10}"
    f"{'Recall@10':>15}"
    f"{'NDCG@10':>15}"
    f"{'Hits':>10}"
)

print(
    "-" * 60
)


for _, row in summary_df.iterrows():

    print(

        f"{row['Content Weight']:>10.1f}"

        f"{row['SVD Weight']:>10.1f}"

        f"{row['Recall@10']:>15.6f}"

        f"{row['NDCG@10']:>15.6f}"

        f"{int(row['Total Hits']):>10}"
    )


# ==========================================================
# SAVE RESULTS
# ==========================================================

output_file = (
    project_path('evaluation/results/hybrid_weight_tuning_results.csv')
)

summary_df.to_csv(
    output_file,
    index=False
)


print(
    "\nSaved:",
    output_file
)