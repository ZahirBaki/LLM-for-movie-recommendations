import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import numpy as np
# --------------------------------------------------
# 1. LOAD MOVIELENS RATINGS
# --------------------------------------------------

print("Loading MovieLens ratings...")

ratings = pd.read_csv(
    "data/ml-32m/ratings.csv"
)

print("Ratings loaded.")
print("Total ratings:", len(ratings))
print("Users:", ratings["userId"].nunique())
print("Movies:", ratings["movieId"].nunique())


# --------------------------------------------------
# 2. DEFINE POSITIVE RATINGS
# --------------------------------------------------

# For evaluation, ratings of 4.0 or higher
# are considered movies the user liked.

POSITIVE_RATING_THRESHOLD = 4.0

positive_ratings = ratings[
    ratings["rating"] >= POSITIVE_RATING_THRESHOLD
].copy()

print(
    f"\nPositive ratings "
    f"(>= {POSITIVE_RATING_THRESHOLD}):"
)

print("Ratings:", len(positive_ratings))
print(
    "Users:",
    positive_ratings["userId"].nunique()
)
print(
    "Movies:",
    positive_ratings["movieId"].nunique()
)


# --------------------------------------------------
# 3. COUNT POSITIVE RATINGS PER USER
# --------------------------------------------------

positive_per_user = (
    positive_ratings
    .groupby("userId")
    .size()
)

print("\nPositive ratings per user:")
print(positive_per_user.describe())


# --------------------------------------------------
# 4. SELECT USERS FOR EVALUATION
# --------------------------------------------------

# Users need enough liked movies so that we can
# hide some and still leave useful training data.

MIN_POSITIVE_RATINGS = 10

# Start with 1,000 users so evaluation is manageable.
N_EVALUATION_USERS = 1000

# Makes our experiment reproducible.
RANDOM_STATE = 42


eligible_users = positive_per_user[
    positive_per_user >= MIN_POSITIVE_RATINGS
].index


print(
    f"\nUsers with at least "
    f"{MIN_POSITIVE_RATINGS} positive ratings:"
)

print(len(eligible_users))


# Randomly choose evaluation users.
evaluation_users = (
    pd.Series(eligible_users)
    .sample(
        n=N_EVALUATION_USERS,
        random_state=RANDOM_STATE
    )
    .tolist()
)


print(
    "Users selected for evaluation:",
    len(evaluation_users)
)


# --------------------------------------------------
# 5. SELECT HIDDEN TEST MOVIES
# --------------------------------------------------

print("\nSelecting hidden test movies...")

test_rows = []


for user_id in evaluation_users:

    # Get all positively rated movies for this user.
    user_positive = positive_ratings[
        positive_ratings["userId"] == user_id
    ]

    # Randomly hide 20% of the movies they liked.
    #
    # We keep the original DataFrame index here.
    # This will allow us to remove exactly these
    # ratings from the training set later.
    user_test = user_positive.sample(
        frac=0.20,
        random_state=RANDOM_STATE
    )

    test_rows.append(user_test)


# Combine all hidden ratings into one DataFrame.
#
# IMPORTANT:
# Do NOT use ignore_index=True here because we want
# to preserve the original indices from ratings.
test_ratings = pd.concat(
    test_rows
)


print(
    "Hidden test ratings selected:",
    len(test_ratings)
)


# --------------------------------------------------
# 6. CREATE TRAINING DATA
# --------------------------------------------------

print("\nCreating training data...")

# Remove the hidden ratings from the original
# MovieLens ratings.
#
# Everything else remains available for training,
# including lower ratings.

train_ratings = ratings.drop(
    index=test_ratings.index
).copy()


print("Training data created.")


# --------------------------------------------------
# 7. VERIFY TRAIN / TEST SPLIT
# --------------------------------------------------

print("\n--- Train/Test Split ---")

print(
    "Original ratings:",
    len(ratings)
)

print(
    "Train ratings:",
    len(train_ratings)
)

print(
    "Hidden test ratings:",
    len(test_ratings)
)

print(
    "Evaluation users:",
    test_ratings["userId"].nunique()
)


# --------------------------------------------------
# 8. CHECK THAT THE NUMBERS ADD UP
# --------------------------------------------------

expected_total = (
    len(train_ratings)
    + len(test_ratings)
)

print("\n--- Verification ---")

print(
    "Train + test:",
    expected_total
)

print(
    "Original:",
    len(ratings)
)


if expected_total == len(ratings):

    print(
        "SUCCESS: Train and test sizes "
        "add up correctly."
    )

else:

    raise ValueError(
        "ERROR: Train/test sizes do not "
        "match the original dataset."
    )


# --------------------------------------------------
# 9. CHECK FOR DATA LEAKAGE
# --------------------------------------------------

# The exact ratings hidden for testing should no
# longer exist in the training DataFrame.

overlap = train_ratings.index.intersection(
    test_ratings.index
)


print(
    "Hidden rows still in training:",
    len(overlap)
)


if len(overlap) == 0:

    print(
        "SUCCESS: No hidden test rows "
        "remain in training."
    )

else:

    raise ValueError(
        "ERROR: Test ratings leaked into "
        "the training data."
    )


# --------------------------------------------------
# 10. SHOW EXAMPLE EVALUATION USER
# --------------------------------------------------

example_user = evaluation_users[0]

example_train = train_ratings[
    train_ratings["userId"] == example_user
]

example_test = test_ratings[
    test_ratings["userId"] == example_user
]


print("\n" + "=" * 60)

print(
    "EXAMPLE EVALUATION USER:",
    example_user
)

print("=" * 60)


print("\nRatings available during training:")

print(
    example_train[
        ["movieId", "rating"]
    ].head(20)
)


print("\nHidden liked movies:")

print(
    example_test[
        ["movieId", "rating"]
    ]
)


print("\nEvaluation split ready.")

# --------------------------------------------------
# 11. FILTER TRAINING MOVIES
# --------------------------------------------------

print("\nPreparing collaborative training data...")

MIN_MOVIE_RATINGS = 50

train_ratings_per_movie = (
    train_ratings
    .groupby("movieId")
    .size()
)

popular_movie_ids = train_ratings_per_movie[
    train_ratings_per_movie >= MIN_MOVIE_RATINGS
].index

filtered_train_ratings = train_ratings[
    train_ratings["movieId"].isin(popular_movie_ids)
].copy()

print(
    "Movies remaining:",
    filtered_train_ratings["movieId"].nunique()
)

print(
    "Ratings remaining:",
    len(filtered_train_ratings)
)

print(
    "Users remaining:",
    filtered_train_ratings["userId"].nunique()
)


# --------------------------------------------------
# 12. MEAN-CENTER TRAINING RATINGS
# --------------------------------------------------

print("\nMean-centering training ratings...")

user_mean_ratings = (
    filtered_train_ratings
    .groupby("userId")["rating"]
    .mean()
)

filtered_train_ratings["user_mean"] = (
    filtered_train_ratings["userId"]
    .map(user_mean_ratings)
)

filtered_train_ratings["centered_rating"] = (
    filtered_train_ratings["rating"]
    - filtered_train_ratings["user_mean"]
)

print("Mean-centering complete.")


# --------------------------------------------------
# 13. CREATE MATRIX INDEXES
# --------------------------------------------------

print("\nCreating matrix indexes...")

user_ids = filtered_train_ratings[
    "userId"
].unique()

movie_ids = filtered_train_ratings[
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

row_indices = filtered_train_ratings[
    "movieId"
].map(movie_to_index)

column_indices = filtered_train_ratings[
    "userId"
].map(user_to_index)


# --------------------------------------------------
# 14. CREATE RAW TRAINING MATRIX
# --------------------------------------------------

print("\nCreating raw training matrix...")

raw_matrix = csr_matrix(
    (
        filtered_train_ratings["rating"].values,
        (row_indices, column_indices)
    ),
    shape=(
        len(movie_ids),
        len(user_ids)
    )
)

print("Raw matrix created.")
print("Shape:", raw_matrix.shape)
print("Stored ratings:", raw_matrix.nnz)


# --------------------------------------------------
# 15. CREATE MEAN-CENTERED TRAINING MATRIX
# --------------------------------------------------

print("\nCreating mean-centered training matrix...")

centered_matrix = csr_matrix(
    (
        filtered_train_ratings[
            "centered_rating"
        ].values,
        (row_indices, column_indices)
    ),
    shape=(
        len(movie_ids),
        len(user_ids)
    )
)

print("Mean-centered matrix created.")
print("Shape:", centered_matrix.shape)
print("Stored ratings:", centered_matrix.nnz)


print("\nTraining matrices ready for evaluation.")

# --------------------------------------------------
# 16. TRAIN RAW AND MEAN-CENTERED MODELS
# --------------------------------------------------

print("\nTraining collaborative models...")

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

print("Collaborative models ready.")


# --------------------------------------------------
# 17. PREPARE EVALUATION DATA
# --------------------------------------------------

TOP_K = 10

# Number of similar movies retrieved for every
# positively rated seed movie.
NEIGHBORS_PER_SEED = 50

# To keep the first evaluation reasonably fast,
# use at most 10 liked movies from each user's
# training history as recommendation seeds.
MAX_SEEDS = 10


print("\nPreparing evaluation users...")


# Only ratings belonging to our evaluation users.
evaluation_train = filtered_train_ratings[
    filtered_train_ratings["userId"].isin(
        evaluation_users
    )
].copy()


# These are the movies the user liked and that
# are still available during training.
evaluation_positive_train = evaluation_train[
    evaluation_train["rating"]
    >= POSITIVE_RATING_THRESHOLD
].copy()


# --------------------------------------------------
# 18. RECOMMENDATION FUNCTION
# --------------------------------------------------

def recommend_for_user(
    user_id,
    model,
    matrix,
    top_k=10
):

    # ----------------------------------------------
    # Movies already seen by this user
    # ----------------------------------------------

    user_history = evaluation_train[
        evaluation_train["userId"] == user_id
    ]

    seen_movies = set(
        user_history["movieId"].tolist()
    )


    # ----------------------------------------------
    # Find movies the user liked
    # ----------------------------------------------

    user_positive = evaluation_positive_train[
        evaluation_positive_train["userId"]
        == user_id
    ]


    if len(user_positive) == 0:
        return []


    # Prefer the user's highest-rated movies.
    #
    # If they have many liked movies, use at most
    # MAX_SEEDS to keep evaluation manageable.
    user_positive = user_positive.sort_values(
        "rating",
        ascending=False
    ).head(MAX_SEEDS)


    candidate_scores = {}


    # ----------------------------------------------
    # Find neighbors of each liked movie
    # ----------------------------------------------

    for row in user_positive.itertuples():

        movie_id = row.movieId


        # Movie might not exist in our filtered
        # collaborative matrix.
        if movie_id not in movie_to_index:
            continue


        movie_index = movie_to_index[movie_id]


        distances, indices = model.kneighbors(
            matrix[movie_index],
            n_neighbors=min(
                NEIGHBORS_PER_SEED + 1,
                matrix.shape[0]
            )
        )


        # kneighbors returns distances.
        #
        # cosine similarity = 1 - cosine distance
        similarities = 1 - distances[0]


        for neighbor_index, similarity in zip(
            indices[0],
            similarities
        ):

            neighbor_movie_id = (
                index_to_movie[neighbor_index]
            )


            # Do not recommend movies the user
            # has already rated.
            if neighbor_movie_id in seen_movies:
                continue


            # Ignore negative similarities.
            if similarity <= 0:
                continue


            # If several seed movies point toward
            # the same candidate, their similarity
            # scores are added together.
            candidate_scores[
                neighbor_movie_id
            ] = (
                candidate_scores.get(
                    neighbor_movie_id,
                    0
                )
                + similarity
            )


    # ----------------------------------------------
    # SORT CANDIDATES
    # ----------------------------------------------

    ranked_movies = sorted(
        candidate_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )


    recommendations = [
        movie_id
        for movie_id, score
        in ranked_movies[:top_k]
    ]


    return recommendations


# --------------------------------------------------
# 19. RECALL@10
# --------------------------------------------------

def calculate_recall_at_10(
    model,
    matrix,
    model_name
):

    print(
        f"\nEvaluating {model_name}..."
    )


    recalls = []

    evaluated_users = 0


    for number, user_id in enumerate(
        evaluation_users,
        start=1
    ):

        # ------------------------------------------
        # Hidden liked movies for this user
        # ------------------------------------------

        user_test = test_ratings[
            test_ratings["userId"] == user_id
        ]


        # Only evaluate hidden movies that exist
        # in the collaborative catalog.
        relevant_movies = set(
            movie_id
            for movie_id
            in user_test["movieId"].tolist()
            if movie_id in movie_to_index
        )


        if len(relevant_movies) == 0:
            continue


        # ------------------------------------------
        # GENERATE TOP-10
        # ------------------------------------------

        recommendations = recommend_for_user(
            user_id=user_id,
            model=model,
            matrix=matrix,
            top_k=TOP_K
        )


        if len(recommendations) == 0:
            continue


        # ------------------------------------------
        # CALCULATE RECALL
        # ------------------------------------------

        recommended_set = set(
            recommendations
        )


        hits = len(
            relevant_movies.intersection(
                recommended_set
            )
        )


        recall = (
            hits
            / len(relevant_movies)
        )


        recalls.append(recall)

        evaluated_users += 1


        # Show progress every 100 users.
        if number % 100 == 0:

            print(
                f"Processed "
                f"{number}/"
                f"{len(evaluation_users)} users..."
            )


    # ----------------------------------------------
    # AVERAGE RECALL
    # ----------------------------------------------

    if len(recalls) == 0:

        return 0.0, 0


    average_recall = np.mean(recalls)


    return (
        average_recall,
        evaluated_users
    )


# --------------------------------------------------
# 20. EVALUATE RAW MODEL
# --------------------------------------------------

raw_recall, raw_users = (
    calculate_recall_at_10(
        model=raw_model,
        matrix=raw_matrix,
        model_name="RAW collaborative model"
    )
)


# --------------------------------------------------
# 21. EVALUATE MEAN-CENTERED MODEL
# --------------------------------------------------

centered_recall, centered_users = (
    calculate_recall_at_10(
        model=centered_model,
        matrix=centered_matrix,
        model_name="MEAN-CENTERED collaborative model"
    )
)


# --------------------------------------------------
# 22. SHOW RESULTS
# --------------------------------------------------

print("\n" + "=" * 60)

print("COLLABORATIVE FILTERING EVALUATION")

print("=" * 60)


print(
    f"\nRaw model:"
    f"\n  Users evaluated: {raw_users}"
    f"\n  Recall@10: {raw_recall:.4f}"
)


print(
    f"\nMean-centered model:"
    f"\n  Users evaluated: {centered_users}"
    f"\n  Recall@10: {centered_recall:.4f}"
)


print("\nEvaluation complete.")