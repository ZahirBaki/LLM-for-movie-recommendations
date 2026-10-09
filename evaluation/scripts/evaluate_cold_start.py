from backend.core.paths import project_path
import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from backend.recommendation.content_recommender import ContentRecommender


# ==========================================================
# SETTINGS
# ==========================================================

RATINGS_FILE = project_path('data/raw/ml-32m/ratings.csv')
MOVIES_FILE = project_path('data/raw/ml-32m/movies.csv')
LINKS_FILE = project_path('data/raw/ml-32m/links.csv')

POSITIVE_RATING = 4.0
MIN_MOVIE_RATINGS = 50

N_EVAL_USERS = 1000
N_FAVORITES = 5

TOP_K = 10

N_FACTORS = 50
RANDOM_STATE = 42


# ==========================================================
# METRICS
# ==========================================================

def recall_at_k(recommended_ids, relevant_ids, k=10):

    recommended = set(recommended_ids[:k])
    relevant = set(relevant_ids)

    if len(relevant) == 0:
        return 0.0

    hits = len(
        recommended.intersection(relevant)
    )

    return hits / len(relevant)


def ndcg_at_k(recommended_ids, relevant_ids, k=10):

    relevant = set(relevant_ids)

    if len(relevant) == 0:
        return 0.0

    dcg = 0.0

    for rank, movie_id in enumerate(
        recommended_ids[:k],
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


def count_hits(
    recommended_ids,
    relevant_ids
):

    return len(
        set(recommended_ids[:TOP_K])
        .intersection(
            set(relevant_ids)
        )
    )


# ==========================================================
# LOAD DATA
# ==========================================================

print("=" * 70)
print("REALISTIC COLD-START EVALUATION")
print("=" * 70)

print("\nLoading MovieLens data...")

ratings = pd.read_csv(
    RATINGS_FILE
)

movies = pd.read_csv(
    MOVIES_FILE
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

print(
    "Movies:",
    ratings["movieId"].nunique()
)


# ==========================================================
# INITIALIZE CONTENT RECOMMENDER
# ==========================================================

print(
    "\nLoading content recommender..."
)

content_model = ContentRecommender()

catalog = content_model.movies.copy()

catalog["tmdb_id"] = (
    catalog["tmdb_id"]
    .astype(int)
)

catalog_tmdb_ids = set(
    catalog["tmdb_id"]
    .tolist()
)

print(
    "TMDB catalog movies:",
    len(catalog)
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
# FIND POSITIVE RATINGS THAT CAN BE EVALUATED
# ==========================================================

print(
    "\nPreparing positive ratings..."
)

positive_ratings = (
    ratings[
        ratings["rating"]
        >= POSITIVE_RATING
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

# Keep movies that exist in our TMDB catalog
positive_ratings = (
    positive_ratings[
        positive_ratings["tmdb_id"]
        .isin(catalog_tmdb_ids)
    ]
    .copy()
)


# ==========================================================
# ELIGIBLE COLD-START USERS
# ==========================================================
#
# Need at least:
#
# 5 favorites
# +
# at least 1 hidden positive movie
#
# ==========================================================

minimum_positive_movies = (
    N_FAVORITES + 1
)

positive_counts = (
    positive_ratings
    .groupby("userId")
    .size()
)

eligible_users = (
    positive_counts[
        positive_counts
        >= minimum_positive_movies
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

eval_user_set = set(
    eval_users.tolist()
)

print(
    "Cold-start evaluation users:",
    len(eval_users)
)


# ==========================================================
# CRITICAL STEP:
# REMOVE ALL EVALUATION USERS FROM SVD TRAINING
# ==========================================================

print(
    "\nRemoving evaluation users "
    "from SVD training data..."
)

train_ratings = (
    ratings[
        ~ratings["userId"]
        .isin(eval_user_set)
    ]
    .copy()
)

print(
    "Original ratings:",
    len(ratings)
)

print(
    "Training ratings after removing "
    "evaluation users:",
    len(train_ratings)
)

remaining_eval_ratings = (
    train_ratings[
        train_ratings["userId"]
        .isin(eval_user_set)
    ]
)

print(
    "Evaluation-user ratings remaining "
    "in SVD training:",
    len(remaining_eval_ratings)
)

if len(remaining_eval_ratings) != 0:

    raise ValueError(
        "DATA LEAKAGE DETECTED!"
    )

print(
    "Leakage check passed."
)


# ==========================================================
# FILTER MOVIES USING TRAINING DATA ONLY
# ==========================================================

print(
    "\nFiltering movies..."
)

movie_rating_counts = (
    train_ratings
    .groupby("movieId")
    .size()
)

popular_movie_ids = (
    movie_rating_counts[
        movie_rating_counts
        >= MIN_MOVIE_RATINGS
    ]
    .index
)

train_filtered = (
    train_ratings[
        train_ratings["movieId"]
        .isin(popular_movie_ids)
    ]
    .copy()
)

print(
    "Movies surviving filter:",
    train_filtered[
        "movieId"
    ].nunique()
)

print(
    "Training ratings:",
    len(train_filtered)
)


# ==========================================================
# CREATE USER AND MOVIE INDEXES
# ==========================================================

print(
    "\nCreating SVD indexes..."
)

user_ids = (
    train_filtered[
        "userId"
    ]
    .unique()
)

movie_ids = (
    train_filtered[
        "movieId"
    ]
    .unique()
)


user_to_index = {
    user_id: index
    for index, user_id
    in enumerate(user_ids)
}


movie_to_index = {
    movie_id: index
    for index, movie_id
    in enumerate(movie_ids)
}


index_to_movie = {
    index: movie_id
    for movie_id, index
    in movie_to_index.items()
}


# ==========================================================
# CREATE SPARSE USER-MOVIE MATRIX
# ==========================================================

print(
    "Creating sparse matrix..."
)

row_indices = (
    train_filtered["userId"]
    .map(user_to_index)
)

column_indices = (
    train_filtered["movieId"]
    .map(movie_to_index)
)

user_movie_matrix = csr_matrix(
    (
        train_filtered[
            "rating"
        ].values,

        (
            row_indices,
            column_indices
        )
    ),

    shape=(
        len(user_ids),
        len(movie_ids)
    )
)

print(
    "Matrix shape:",
    user_movie_matrix.shape
)

print(
    "Stored ratings:",
    user_movie_matrix.nnz
)


# ==========================================================
# TRAIN SVD
# ==========================================================

print(
    "\nTraining SVD..."
)

svd = TruncatedSVD(
    n_components=N_FACTORS,
    random_state=RANDOM_STATE
)

user_factors = svd.fit_transform(
    user_movie_matrix
)

movie_factors = (
    svd.components_.T
)

print(
    "SVD training complete."
)

print(
    "User factors:",
    user_factors.shape
)

print(
    "Movie factors:",
    movie_factors.shape
)

print(
    "Explained variance:",
    svd.explained_variance_ratio_.sum()
)


# ==========================================================
# SVD MOVIE -> TMDB LOOKUP
# ==========================================================

svd_movie_tmdb = {}

for movie_id in movie_ids:

    tmdb_id = movie_to_tmdb.get(
        movie_id
    )

    if tmdb_id is None:
        continue

    tmdb_id = int(tmdb_id)

    if tmdb_id not in catalog_tmdb_ids:
        continue

    svd_movie_tmdb[
        movie_id
    ] = tmdb_id


# ==========================================================
# SVD COLD-START FUNCTION
# ==========================================================

def recommend_svd_cold_start(
    favorite_tmdb_ids,
    top_n=10
):

    favorite_movie_indices = []

    favorite_tmdb_set = set(
        favorite_tmdb_ids
    )

    # ------------------------------------------------------
    # Find MovieLens/SVD movies matching the 5 favorites
    # ------------------------------------------------------

    for movie_id, tmdb_id in (
        svd_movie_tmdb.items()
    ):

        if tmdb_id in favorite_tmdb_set:

            matrix_index = (
                movie_to_index.get(
                    movie_id
                )
            )

            if matrix_index is not None:

                favorite_movie_indices.append(
                    matrix_index
                )


    if len(
        favorite_movie_indices
    ) == 0:

        return []


    # ------------------------------------------------------
    # Build temporary new-user profile
    #
    # Average latent vectors of the user's 5 favorites.
    # ------------------------------------------------------

    favorite_vectors = (
        movie_factors[
            favorite_movie_indices
        ]
    )

    temporary_user_vector = (
        favorite_vectors.mean(
            axis=0
        )
    )


    # ------------------------------------------------------
    # Score all SVD movies
    # ------------------------------------------------------

    scores = (
        movie_factors
        @ temporary_user_vector
    )


    ranked_indices = np.argsort(
        scores
    )[::-1]


    recommendations = []

    seen_tmdb_ids = set()


    for matrix_index in ranked_indices:

        movie_id = (
            index_to_movie[
                int(matrix_index)
            ]
        )

        tmdb_id = (
            svd_movie_tmdb.get(
                movie_id
            )
        )

        if tmdb_id is None:
            continue

        # Don't recommend favorites
        if tmdb_id in favorite_tmdb_set:
            continue

        # Prevent duplicate TMDB movies
        if tmdb_id in seen_tmdb_ids:
            continue

        seen_tmdb_ids.add(
            tmdb_id
        )

        recommendations.append(
            tmdb_id
        )

        if len(
            recommendations
        ) >= top_n:

            break


    return recommendations


# ==========================================================
# RESULTS STORAGE
# ==========================================================

results = []


# ==========================================================
# EVALUATE COLD-START USERS
# ==========================================================

print("\n")
print("=" * 70)
print("EVALUATING COLD-START USERS")
print("=" * 70)


for position, user_id in enumerate(
    eval_users,
    start=1
):

    user_positive = (
        positive_ratings[
            positive_ratings["userId"]
            == user_id
        ]
        .copy()
    )


    # ------------------------------------------------------
    # Shuffle reproducibly
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
    # EXACTLY 5 FAVORITES
    # ------------------------------------------------------

    favorites = (
        user_positive
        .iloc[:N_FAVORITES]
        .copy()
    )


    # ------------------------------------------------------
    # EVERYTHING ELSE IS HIDDEN
    # ------------------------------------------------------

    hidden = (
        user_positive
        .iloc[N_FAVORITES:]
        .copy()
    )


    if hidden.empty:
        continue


    favorite_tmdb_ids = (
        favorites["tmdb_id"]
        .astype(int)
        .tolist()
    )


    hidden_tmdb_ids = (
        hidden["tmdb_id"]
        .astype(int)
        .tolist()
    )


    # ======================================================
    # FAVORITE TITLES FOR CONTENT MODEL
    # ======================================================

    favorite_catalog = (
        catalog[
            catalog["tmdb_id"]
            .isin(
                favorite_tmdb_ids
            )
        ]
        .copy()
    )


    favorite_titles = (
        favorite_catalog[
            "title"
        ]
        .tolist()
    )


    if len(
        favorite_titles
    ) == 0:

        continue


    # ======================================================
    # CONTENT RECOMMENDATIONS
    # ======================================================

    content_results = (
        content_model.get_scores(

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
    # SVD COLD-START RECOMMENDATIONS
    # ======================================================

    svd_ids = (
        recommend_svd_cold_start(

            favorite_tmdb_ids,

            top_n=TOP_K
        )
    )


    # ======================================================
    # METRICS
    # ======================================================

    content_hits = count_hits(
        content_ids,
        hidden_tmdb_ids
    )

    content_recall = recall_at_k(
        content_ids,
        hidden_tmdb_ids,
        TOP_K
    )

    content_ndcg = ndcg_at_k(
        content_ids,
        hidden_tmdb_ids,
        TOP_K
    )


    svd_hits = count_hits(
        svd_ids,
        hidden_tmdb_ids
    )

    svd_recall = recall_at_k(
        svd_ids,
        hidden_tmdb_ids,
        TOP_K
    )

    svd_ndcg = ndcg_at_k(
        svd_ids,
        hidden_tmdb_ids,
        TOP_K
    )


    # ======================================================
    # SAVE RESULT
    # ======================================================

    results.append({

        "userId":
            int(user_id),

        "favorites":
            len(favorite_tmdb_ids),

        "hidden_movies":
            len(hidden_tmdb_ids),

        "content_hits":
            content_hits,

        "content_recall":
            content_recall,

        "content_ndcg":
            content_ndcg,

        "svd_hits":
            svd_hits,

        "svd_recall":
            svd_recall,

        "svd_ndcg":
            svd_ndcg
    })


    # ======================================================
    # PROGRESS
    # ======================================================

    if (
        position % 10 == 0
        or position == 1
    ):

        print(

            f"User "
            f"{position}/{len(eval_users)}"

            f" | Content hits: "
            f"{content_hits}"

            f" | SVD hits: "
            f"{svd_hits}"
        )


# ==========================================================
# RESULTS DATAFRAME
# ==========================================================

results_df = pd.DataFrame(
    results
)


# ==========================================================
# FINAL RESULTS
# ==========================================================

print("\n")
print("=" * 70)
print("REALISTIC COLD-START RESULTS")
print("=" * 70)


if results_df.empty:

    print(
        "No users were evaluated."
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

    content_hits = (
        results_df[
            "content_hits"
        ]
        .sum()
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

    svd_hits = (
        results_df[
            "svd_hits"
        ]
        .sum()
    )


    print(
        "Users evaluated:",
        len(results_df)
    )

    print(
        "Favorites per user:",
        N_FAVORITES
    )

    print(
        "Evaluation users included "
        "in SVD training: 0"
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

        f"{content_hits:>15}"
    )


    print(

        f"{'SVD Cold-Start':<20}"

        f"{svd_recall:>15.6f}"

        f"{svd_ndcg:>15.6f}"

        f"{svd_hits:>15}"
    )


    # ======================================================
    # SAVE DETAILED RESULTS
    # ======================================================

    results_df.to_csv(

        project_path('evaluation/results/cold_start_evaluation_results.csv'),

        index=False
    )


    # ======================================================
    # SAVE SUMMARY
    # ======================================================

    summary_df = pd.DataFrame({

        "Model": [
            "Content",
            "SVD Cold-Start"
        ],

        "Recall@10": [
            content_recall,
            svd_recall
        ],

        "NDCG@10": [
            content_ndcg,
            svd_ndcg
        ],

        "Total Hits": [
            content_hits,
            svd_hits
        ]
    })


    summary_df.to_csv(

        project_path('evaluation/results/cold_start_evaluation_summary.csv'),

        index=False
    )


    print(
        "\nSaved:"
    )

    print(
        project_path('evaluation/results/cold_start_evaluation_results.csv')
    )

    print(
        project_path('evaluation/results/cold_start_evaluation_summary.csv')
    )