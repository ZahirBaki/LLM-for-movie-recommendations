from backend.core.paths import project_path
import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


# ==========================================================
# SETTINGS
# ==========================================================

RATINGS_FILE = project_path('data/raw/ml-32m/ratings.csv')
MOVIES_FILE = project_path('data/raw/ml-32m/movies.csv')

POSITIVE_THRESHOLD = 4.0

MIN_POSITIVE_RATINGS = 10
MIN_MOVIE_RATINGS = 50

N_EVALUATION_USERS = 1000

TEST_FRACTION = 0.20

TOP_K = 10

NEIGHBORS_PER_SEED = 50
MAX_SEEDS = 10

RANDOM_STATE = 42

RESULTS_FILE = project_path('evaluation/results/collaborative_evaluation_results.csv')


# ==========================================================
# 1. LOAD DATA
# ==========================================================

print("Loading MovieLens data...")

ratings = pd.read_csv(RATINGS_FILE)
movies = pd.read_csv(MOVIES_FILE)

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
# 3. SELECT EVALUATION USERS
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
# 4. CREATE TEST SET
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


# Remove hidden rows from training data.

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


# We no longer need these large objects.

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


popular_movie_ids = ratings_per_movie[
    ratings_per_movie >= MIN_MOVIE_RATINGS
].index


filtered_train = train_ratings[
    train_ratings["movieId"].isin(
        popular_movie_ids
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


# We can now remove the larger unfiltered training DataFrame.

del train_ratings


# ==========================================================
# 6. MEAN-CENTER RATINGS
# ==========================================================

print("\nMean-centering ratings...")

user_means = (
    filtered_train
    .groupby("userId")["rating"]
    .mean()
)


filtered_train["user_mean"] = (
    filtered_train["userId"]
    .map(user_means)
)


filtered_train["centered_rating"] = (
    filtered_train["rating"]
    - filtered_train["user_mean"]
)


print("Mean-centering complete.")


# ==========================================================
# 7. CREATE MATRIX INDEXES
# ==========================================================

print("\nCreating matrix indexes...")

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


row_indices = filtered_train[
    "movieId"
].map(movie_to_index).to_numpy()


column_indices = filtered_train[
    "userId"
].map(user_to_index).to_numpy()


# ==========================================================
# 8. CREATE SPARSE MATRICES
# ==========================================================

print("\nCreating raw matrix...")

raw_matrix = csr_matrix(
    (
        filtered_train["rating"].to_numpy(),
        (
            row_indices,
            column_indices
        )
    ),
    shape=(
        len(movie_ids),
        len(user_ids)
    )
)


print(
    "Raw matrix:",
    raw_matrix.shape
)


print("\nCreating mean-centered matrix...")

centered_matrix = csr_matrix(
    (
        filtered_train[
            "centered_rating"
        ].to_numpy(),
        (
            row_indices,
            column_indices
        )
    ),
    shape=(
        len(movie_ids),
        len(user_ids)
    )
)


print(
    "Centered matrix:",
    centered_matrix.shape
)


# ==========================================================
# 9. TRAIN KNN MODELS
# ==========================================================

print("\nPreparing KNN models...")


raw_model = NearestNeighbors(
    metric="cosine",
    algorithm="brute"
)

raw_model.fit(raw_matrix)


centered_model = NearestNeighbors(
    metric="cosine",
    algorithm="brute"
)

centered_model.fit(centered_matrix)


print("KNN models ready.")


# ==========================================================
# 10. PREPARE EVALUATION HISTORIES
# ==========================================================

print("\nPreparing evaluation histories...")


evaluation_train = filtered_train[
    filtered_train["userId"].isin(
        evaluation_users
    )
].copy()


# Store seen movies for every evaluation user.

user_seen = (
    evaluation_train
    .groupby("userId")["movieId"]
    .apply(set)
    .to_dict()
)


# Positive training movies.

positive_train = evaluation_train[
    evaluation_train["rating"]
    >= POSITIVE_THRESHOLD
].copy()


# Choose the seed movies once.

user_seeds = {}


for user_id, group in positive_train.groupby(
    "userId"
):

    seeds = (
        group
        .sort_values(
            "rating",
            ascending=False
        )
        .head(MAX_SEEDS)["movieId"]
        .tolist()
    )

    user_seeds[user_id] = seeds


# Hidden relevant movies for every user.

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
# 11. FIND ALL UNIQUE SEED MOVIES
# ==========================================================

print("\nCollecting unique seed movies...")


all_seed_movies = set()


for seeds in user_seeds.values():

    for movie_id in seeds:

        if movie_id in movie_to_index:

            all_seed_movies.add(movie_id)


all_seed_movies = list(
    all_seed_movies
)


print(
    "Unique seed movies:",
    len(all_seed_movies)
)


seed_indices = np.array(
    [
        movie_to_index[movie_id]
        for movie_id in all_seed_movies
    ]
)


# ==========================================================
# 12. BATCH NEIGHBOR CALCULATION
# ==========================================================

def calculate_neighbors(
    model,
    matrix,
    model_name
):

    print(
        f"\nCalculating neighbors for {model_name}..."
    )


    distances, indices = model.kneighbors(
        matrix[seed_indices],
        n_neighbors=NEIGHBORS_PER_SEED + 1
    )


    neighbor_lookup = {}


    for row_number, seed_movie_id in enumerate(
        all_seed_movies
    ):

        movie_neighbors = []


        for neighbor_index, distance in zip(
            indices[row_number],
            distances[row_number]
        ):

            neighbor_movie_id = (
                index_to_movie[
                    neighbor_index
                ]
            )


            # Skip the seed movie itself.

            if (
                neighbor_movie_id
                == seed_movie_id
            ):
                continue


            similarity = 1 - distance


            if similarity <= 0:
                continue


            movie_neighbors.append(
                (
                    neighbor_movie_id,
                    similarity
                )
            )


        neighbor_lookup[
            seed_movie_id
        ] = movie_neighbors[
            :NEIGHBORS_PER_SEED
        ]


    print(
        f"{model_name} neighbor lookup ready."
    )


    return neighbor_lookup


# ==========================================================
# 13. CREATE NEIGHBOR LOOKUPS
# ==========================================================

raw_neighbors = calculate_neighbors(
    raw_model,
    raw_matrix,
    "Raw KNN"
)


centered_neighbors = calculate_neighbors(
    centered_model,
    centered_matrix,
    "Mean-Centered KNN"
)


# ==========================================================
# 14. GENERATE USER RECOMMENDATIONS
# ==========================================================

def recommend_from_lookup(
    user_id,
    neighbor_lookup
):

    seen = user_seen.get(
        user_id,
        set()
    )


    seeds = user_seeds.get(
        user_id,
        []
    )


    candidate_scores = {}


    for seed_movie_id in seeds:

        neighbors = neighbor_lookup.get(
            seed_movie_id,
            []
        )


        for (
            candidate_movie_id,
            similarity
        ) in neighbors:


            if candidate_movie_id in seen:
                continue


            candidate_scores[
                candidate_movie_id
            ] = (
                candidate_scores.get(
                    candidate_movie_id,
                    0.0
                )
                + similarity
            )


    ranked = sorted(
        candidate_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )


    return [
        movie_id
        for movie_id, score
        in ranked[:TOP_K]
    ]


# ==========================================================
# 15. METRIC FUNCTIONS
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


def ndcg_at_k(
    recommendations,
    relevant
):

    if len(relevant) == 0:
        return None


    # ------------------------------
    # DCG
    # ------------------------------

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


    # ------------------------------
    # IDEAL DCG
    # ------------------------------

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
# 16. EVALUATE KNN MODEL
# ==========================================================

def evaluate_knn(
    neighbor_lookup,
    model_name
):

    print(
        f"\nEvaluating {model_name}..."
    )


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


        recommendations = (
            recommend_from_lookup(
                user_id,
                neighbor_lookup
            )
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


    return {
        "Model": model_name,
        "Users": len(recalls),
        "Recall@10": np.mean(recalls),
        "NDCG@10": np.mean(ndcgs)
    }


# ==========================================================
# 17. EVALUATE RAW KNN
# ==========================================================

raw_results = evaluate_knn(
    raw_neighbors,
    "Raw KNN"
)


# ==========================================================
# 18. EVALUATE MEAN-CENTERED KNN
# ==========================================================

centered_results = evaluate_knn(
    centered_neighbors,
    "Mean-Centered KNN"
)


# ==========================================================
# 19. POPULARITY BASELINE
# ==========================================================

print("\nEvaluating popularity baseline...")


movie_popularity = (
    filtered_train
    .groupby("movieId")
    .size()
    .sort_values(
        ascending=False
    )
)


popularity_ranking = (
    movie_popularity
    .index
    .tolist()
)


popularity_recalls = []
popularity_ndcgs = []


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


    seen = user_seen.get(
        user_id,
        set()
    )


    recommendations = []


    for movie_id in popularity_ranking:

        if movie_id in seen:
            continue


        recommendations.append(
            movie_id
        )


        if len(recommendations) >= TOP_K:
            break


    recall = recall_at_k(
        recommendations,
        relevant
    )


    ndcg = ndcg_at_k(
        recommendations,
        relevant
    )


    popularity_recalls.append(
        recall
    )

    popularity_ndcgs.append(
        ndcg
    )


    if number % 100 == 0:

        print(
            f"Processed "
            f"{number}/"
            f"{len(evaluation_users)}"
        )


popularity_results = {
    "Model": "Popularity Baseline",
    "Users": len(
        popularity_recalls
    ),
    "Recall@10": np.mean(
        popularity_recalls
    ),
    "NDCG@10": np.mean(
        popularity_ndcgs
    )
}


# ==========================================================
# 20. FINAL RESULTS
# ==========================================================

results = pd.DataFrame(
    [
        popularity_results,
        raw_results,
        centered_results
    ]
)


print("\n" + "=" * 70)

print(
    "FINAL COLLABORATIVE FILTERING EVALUATION"
)

print("=" * 70)


print(
    results.to_string(
        index=False
    )
)


# ==========================================================
# 21. SAVE RESULTS
# ==========================================================

results.to_csv(
    RESULTS_FILE,
    index=False
)


print(
    f"\nResults saved to: "
    f"{RESULTS_FILE}"
)


print("\nEvaluation complete.")