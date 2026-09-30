import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


# ==========================================================
# SETTINGS
# ==========================================================

RATINGS_FILE = "data/ml-32m/ratings.csv"

POSITIVE_THRESHOLD = 4.0
MIN_POSITIVE_RATINGS = 10
MIN_MOVIE_RATINGS = 50

N_EVALUATION_USERS = 1000
TEST_FRACTION = 0.20

TOP_K = 10

N_FACTORS = 50
RANDOM_STATE = 42

RESULTS_FILE = "svd_evaluation_results.csv"


# ==========================================================
# 1. LOAD RATINGS
# ==========================================================

print("Loading MovieLens ratings...")

ratings = pd.read_csv(RATINGS_FILE)

print("Ratings:", len(ratings))
print("Users:", ratings["userId"].nunique())
print("Movies:", ratings["movieId"].nunique())


# ==========================================================
# 2. FIND POSITIVE RATINGS
# ==========================================================

print("\nFinding positive ratings...")

positive_ratings = ratings[
    ratings["rating"] >= POSITIVE_THRESHOLD
].copy()


positive_per_user = (
    positive_ratings
    .groupby("userId")
    .size()
)


eligible_users = positive_per_user[
    positive_per_user >= MIN_POSITIVE_RATINGS
].index


print("Eligible users:", len(eligible_users))


# ==========================================================
# 3. SELECT SAME 1,000 EVALUATION USERS
# ==========================================================

evaluation_users = (
    pd.Series(eligible_users)
    .sample(
        n=N_EVALUATION_USERS,
        random_state=RANDOM_STATE
    )
    .tolist()
)


print("Evaluation users:", len(evaluation_users))


# ==========================================================
# 4. CREATE SAME TRAIN / TEST SPLIT
# ==========================================================

print("\nCreating train/test split...")

test_rows = []


for user_id in evaluation_users:

    user_positive = positive_ratings[
        positive_ratings["userId"] == user_id
    ]

    user_test = user_positive.sample(
        frac=TEST_FRACTION,
        random_state=RANDOM_STATE
    )

    test_rows.append(user_test)


test_ratings = pd.concat(test_rows)


train_ratings = ratings.drop(
    index=test_ratings.index
).copy()


print("Training ratings:", len(train_ratings))
print("Hidden test ratings:", len(test_ratings))


# Check for leakage.

leakage = train_ratings.index.intersection(
    test_ratings.index
)

print(
    "Hidden rows still in training:",
    len(leakage)
)


# Free memory.

del positive_ratings
del ratings


# ==========================================================
# 5. FILTER MOVIES USING TRAINING DATA ONLY
# ==========================================================

print("\nFiltering movies...")


ratings_per_movie = (
    train_ratings
    .groupby("movieId")
    .size()
)


valid_movie_ids = ratings_per_movie[
    ratings_per_movie >= MIN_MOVIE_RATINGS
].index


filtered_train = train_ratings[
    train_ratings["movieId"].isin(
        valid_movie_ids
    )
].copy()


print(
    "Movies remaining:",
    filtered_train["movieId"].nunique()
)

print(
    "Ratings remaining:",
    len(filtered_train)
)


del train_ratings


# ==========================================================
# 6. CREATE USER / MOVIE INDEXES
# ==========================================================

print("\nCreating indexes...")


user_ids = filtered_train[
    "userId"
].unique()


movie_ids = filtered_train[
    "movieId"
].unique()


user_to_index = {
    user_id: index
    for index, user_id in enumerate(user_ids)
}


movie_to_index = {
    movie_id: index
    for index, movie_id in enumerate(movie_ids)
}


index_to_movie = {
    index: movie_id
    for movie_id, index in movie_to_index.items()
}


# ==========================================================
# 7. CREATE USER-MOVIE MATRIX
# ==========================================================

print("\nCreating sparse user-movie matrix...")


row_indices = (
    filtered_train["userId"]
    .map(user_to_index)
    .to_numpy()
)


column_indices = (
    filtered_train["movieId"]
    .map(movie_to_index)
    .to_numpy()
)


user_movie_matrix = csr_matrix(
    (
        filtered_train["rating"].to_numpy(),
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
# 8. TRAIN SVD
# ==========================================================

print(
    f"\nTraining Truncated SVD "
    f"with {N_FACTORS} factors..."
)


svd = TruncatedSVD(
    n_components=N_FACTORS,
    random_state=RANDOM_STATE
)


user_factors = svd.fit_transform(
    user_movie_matrix
)


movie_factors = svd.components_.T


print("\nSVD training complete.")

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
# 9. PREPARE USER HISTORIES
# ==========================================================

print("\nPreparing evaluation data...")


evaluation_train = filtered_train[
    filtered_train["userId"].isin(
        evaluation_users
    )
]


user_seen = (
    evaluation_train
    .groupby("userId")["movieId"]
    .apply(set)
    .to_dict()
)


# Hidden relevant movies.

user_relevant = {}


for user_id, group in test_ratings.groupby(
    "userId"
):

    relevant = {
        movie_id
        for movie_id in group["movieId"]
        if movie_id in movie_to_index
    }

    user_relevant[user_id] = relevant


# ==========================================================
# 10. RECOMMENDATION FUNCTION
# ==========================================================

def recommend_for_user(
    user_id,
    top_k=10
):

    if user_id not in user_to_index:
        return []


    user_index = user_to_index[
        user_id
    ]


    user_vector = user_factors[
        user_index
    ]


    # Score every movie using latent factors.

    scores = (
        movie_factors
        @ user_vector
    )


    seen = user_seen.get(
        user_id,
        set()
    )


    # We do not want to sort every movie using Python.
    # Get a larger candidate pool quickly first.

    candidate_count = min(
        len(scores),
        top_k + len(seen) + 100
    )


    candidate_indices = np.argpartition(
        scores,
        -candidate_count
    )[-candidate_count:]


    # Sort only those candidates.

    candidate_indices = candidate_indices[
        np.argsort(
            scores[candidate_indices]
        )[::-1]
    ]


    recommendations = []


    for movie_index in candidate_indices:

        movie_id = index_to_movie[
            movie_index
        ]


        if movie_id in seen:
            continue


        recommendations.append(
            movie_id
        )


        if len(recommendations) >= top_k:
            break


    return recommendations


# ==========================================================
# 11. RECALL@10
# ==========================================================

def recall_at_k(
    recommendations,
    relevant
):

    if len(relevant) == 0:
        return None


    hits = len(
        set(recommendations)
        .intersection(relevant)
    )


    return (
        hits
        / len(relevant)
    )


# ==========================================================
# 12. NDCG@10
# ==========================================================

def ndcg_at_k(
    recommendations,
    relevant
):

    if len(relevant) == 0:
        return None


    dcg = 0.0


    for rank, movie_id in enumerate(
        recommendations,
        start=1
    ):

        if movie_id in relevant:

            dcg += (
                1
                / np.log2(rank + 1)
            )


    ideal_hits = min(
        len(relevant),
        TOP_K
    )


    idcg = 0.0


    for rank in range(
        1,
        ideal_hits + 1
    ):

        idcg += (
            1
            / np.log2(rank + 1)
        )


    if idcg == 0:
        return 0.0


    return dcg / idcg


# ==========================================================
# 13. EVALUATE SVD
# ==========================================================

print("\nEvaluating SVD...")


recalls = []
ndcgs = []


for number, user_id in enumerate(
    evaluation_users,
    start=1
):

    relevant = user_relevant.get(
        user_id,
        set()
    )


    if len(relevant) == 0:
        continue


    recommendations = recommend_for_user(
        user_id,
        TOP_K
    )


    recall = recall_at_k(
        recommendations,
        relevant
    )


    ndcg = ndcg_at_k(
        recommendations,
        relevant
    )


    if recall is not None:
        recalls.append(recall)


    if ndcg is not None:
        ndcgs.append(ndcg)


    if number % 100 == 0:

        print(
            f"Processed "
            f"{number}/"
            f"{len(evaluation_users)}"
        )


# ==========================================================
# 14. FINAL SVD RESULTS
# ==========================================================

mean_recall = np.mean(recalls)
mean_ndcg = np.mean(ndcgs)


print("\n" + "=" * 70)

print(
    "TRUNCATED SVD EVALUATION"
)

print("=" * 70)


print(
    f"Users evaluated: {len(recalls)}"
)

print(
    f"Recall@10: {mean_recall:.6f}"
)

print(
    f"NDCG@10:   {mean_ndcg:.6f}"
)


# ==========================================================
# 15. COMPARE WITH PREVIOUS MODELS
# ==========================================================

print("\n" + "=" * 70)

print(
    "CURRENT MODEL COMPARISON"
)

print("=" * 70)


print(
    f"{'Model':<25}"
    f"{'Recall@10':>12}"
    f"{'NDCG@10':>12}"
)


print("-" * 49)


print(
    f"{'Popularity':<25}"
    f"{0.108621:>12.6f}"
    f"{0.175059:>12.6f}"
)


print(
    f"{'Raw KNN':<25}"
    f"{0.175385:>12.6f}"
    f"{0.258277:>12.6f}"
)


print(
    f"{'Mean-Centered KNN':<25}"
    f"{0.145365:>12.6f}"
    f"{0.228090:>12.6f}"
)


print(
    f"{'Truncated SVD (50)':<25}"
    f"{mean_recall:>12.6f}"
    f"{mean_ndcg:>12.6f}"
)


# ==========================================================
# 16. SAVE RESULTS
# ==========================================================

results = pd.DataFrame(
    [
        {
            "Model": "Truncated SVD",
            "Factors": N_FACTORS,
            "Users": len(recalls),
            "Recall@10": mean_recall,
            "NDCG@10": mean_ndcg,
            "Random State": RANDOM_STATE
        }
    ]
)


results.to_csv(
    RESULTS_FILE,
    index=False
)


print(
    f"\nResults saved to: "
    f"{RESULTS_FILE}"
)


print("\nEvaluation complete.")