import pandas as pd
import numpy as np
import time

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

RANDOM_STATE = 42

# The hyperparameter we are testing.
FACTOR_VALUES = [
    20,
    50,
    100,
    200
]

RESULTS_FILE = "svd_factor_tuning_results.csv"


# ==========================================================
# 1. LOAD DATA
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


print(
    "Eligible users:",
    len(eligible_users)
)


# ==========================================================
# 3. SELECT SAME EVALUATION USERS
# ==========================================================

evaluation_users = (
    pd.Series(eligible_users)
    .sample(
        n=N_EVALUATION_USERS,
        random_state=RANDOM_STATE
    )
    .tolist()
)


print(
    "Evaluation users:",
    len(evaluation_users)
)


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


test_ratings = pd.concat(
    test_rows
)


train_ratings = ratings.drop(
    index=test_ratings.index
).copy()


print(
    "Training ratings:",
    len(train_ratings)
)

print(
    "Hidden test ratings:",
    len(test_ratings)
)


leakage = train_ratings.index.intersection(
    test_ratings.index
)


print(
    "Hidden rows still in training:",
    len(leakage)
)


del positive_ratings
del ratings


# ==========================================================
# 5. FILTER MOVIES
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
# 6. CREATE INDEXES
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
    for index, user_id in enumerate(
        user_ids
    )
}


movie_to_index = {
    movie_id: index
    for index, movie_id in enumerate(
        movie_ids
    )
}


index_to_movie = {
    index: movie_id
    for movie_id, index
    in movie_to_index.items()
}


# ==========================================================
# 7. CREATE SPARSE MATRIX
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
# 8. PREPARE EVALUATION DATA
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
# 9. METRICS
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


    return hits / len(relevant)


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
# 10. EVALUATE ONE FACTOR SETTING
# ==========================================================

def train_and_evaluate(n_factors):

    print("\n" + "=" * 70)

    print(
        f"TESTING {n_factors} LATENT FACTORS"
    )

    print("=" * 70)


    start_time = time.time()


    # ------------------------------------------------------
    # TRAIN SVD
    # ------------------------------------------------------

    print(
        f"\nTraining SVD with "
        f"{n_factors} factors..."
    )


    svd = TruncatedSVD(
        n_components=n_factors,
        random_state=RANDOM_STATE
    )


    user_factors = svd.fit_transform(
        user_movie_matrix
    )


    movie_factors = (
        svd.components_.T
    )


    explained_variance = (
        svd
        .explained_variance_ratio_
        .sum()
    )


    print(
        "Explained variance:",
        explained_variance
    )


    # ------------------------------------------------------
    # EVALUATE USERS
    # ------------------------------------------------------

    print("\nEvaluating users...")


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


        if user_id not in user_to_index:
            continue


        user_index = user_to_index[
            user_id
        ]


        user_vector = user_factors[
            user_index
        ]


        scores = (
            movie_factors
            @ user_vector
        )


        seen = user_seen.get(
            user_id,
            set()
        )


        candidate_count = min(
            len(scores),
            TOP_K + len(seen) + 100
        )


        candidate_indices = np.argpartition(
            scores,
            -candidate_count
        )[-candidate_count:]


        candidate_indices = (
            candidate_indices[
                np.argsort(
                    scores[
                        candidate_indices
                    ]
                )[::-1]
            ]
        )


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


            if (
                len(recommendations)
                >= TOP_K
            ):
                break


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


        if number % 200 == 0:

            print(
                f"Processed "
                f"{number}/"
                f"{len(evaluation_users)}"
            )


    mean_recall = np.mean(
        recalls
    )

    mean_ndcg = np.mean(
        ndcgs
    )


    elapsed_time = (
        time.time()
        - start_time
    )


    print(
        f"\nResults for "
        f"{n_factors} factors:"
    )

    print(
        f"Recall@10: "
        f"{mean_recall:.6f}"
    )

    print(
        f"NDCG@10:   "
        f"{mean_ndcg:.6f}"
    )

    print(
        f"Time: "
        f"{elapsed_time:.1f} seconds"
    )


    return {
        "Factors": n_factors,
        "Recall@10": mean_recall,
        "NDCG@10": mean_ndcg,
        "Explained Variance":
            explained_variance,
        "Time Seconds":
            elapsed_time
    }


# ==========================================================
# 11. RUN ALL FACTOR SETTINGS
# ==========================================================

results = []


for n_factors in FACTOR_VALUES:

    result = train_and_evaluate(
        n_factors
    )

    results.append(
        result
    )


# ==========================================================
# 12. RESULTS TABLE
# ==========================================================

results_df = pd.DataFrame(
    results
)


print("\n" + "=" * 70)

print(
    "SVD HYPERPARAMETER TUNING RESULTS"
)

print("=" * 70)


print(
    results_df.to_string(
        index=False
    )
)


# ==========================================================
# 13. BEST SETTINGS
# ==========================================================

best_recall_row = results_df.loc[
    results_df["Recall@10"].idxmax()
]


best_ndcg_row = results_df.loc[
    results_df["NDCG@10"].idxmax()
]


print("\n" + "=" * 70)

print("BEST RESULTS")

print("=" * 70)


print(
    "\nHighest Recall@10:"
)

print(
    f"Factors: "
    f"{int(best_recall_row['Factors'])}"
)

print(
    f"Recall@10: "
    f"{best_recall_row['Recall@10']:.6f}"
)


print(
    "\nHighest NDCG@10:"
)

print(
    f"Factors: "
    f"{int(best_ndcg_row['Factors'])}"
)

print(
    f"NDCG@10: "
    f"{best_ndcg_row['NDCG@10']:.6f}"
)


# ==========================================================
# 14. SAVE RESULTS
# ==========================================================

results_df.to_csv(
    RESULTS_FILE,
    index=False
)


print(
    f"\nResults saved to: "
    f"{RESULTS_FILE}"
)


print("\nTuning complete.")