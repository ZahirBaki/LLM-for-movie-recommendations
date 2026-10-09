from backend.core.paths import project_path
import pandas as pd
import numpy as np


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

POSITIVE_RATING_THRESHOLD = 4.0
MIN_POSITIVE_RATINGS = 10
N_EVALUATION_USERS = 1000
RANDOM_STATE = 42

TOP_K = 10
MIN_MOVIE_RATINGS = 50


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

print("Loading MovieLens data...")

ratings = pd.read_csv(
    project_path('data/raw/ml-32m/ratings.csv')
)

movies = pd.read_csv(
    project_path('data/raw/ml-32m/movies.csv')
)

print("Ratings loaded:", len(ratings))


# --------------------------------------------------
# 2. FIND POSITIVE RATINGS
# --------------------------------------------------

positive_ratings = ratings[
    ratings["rating"] >= POSITIVE_RATING_THRESHOLD
].copy()

positive_per_user = (
    positive_ratings
    .groupby("userId")
    .size()
)


# --------------------------------------------------
# 3. SELECT SAME TYPE OF EVALUATION USERS
# --------------------------------------------------

eligible_users = positive_per_user[
    positive_per_user >= MIN_POSITIVE_RATINGS
].index

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


# --------------------------------------------------
# 4. HIDE 20% OF POSITIVE RATINGS
# --------------------------------------------------

print("Creating test set...")

test_rows = []

for user_id in evaluation_users:

    user_positive = positive_ratings[
        positive_ratings["userId"] == user_id
    ]

    user_test = user_positive.sample(
        frac=0.20,
        random_state=RANDOM_STATE
    )

    test_rows.append(user_test)


test_ratings = pd.concat(test_rows)


# Remove hidden ratings from training data.

train_ratings = ratings.drop(
    index=test_ratings.index
).copy()


print(
    "Hidden test ratings:",
    len(test_ratings)
)


# --------------------------------------------------
# 5. APPLY SAME 50-RATING FILTER
# --------------------------------------------------

print("Filtering movies...")

ratings_per_movie = (
    train_ratings
    .groupby("movieId")
    .size()
)

popular_movie_ids = ratings_per_movie[
    ratings_per_movie >= MIN_MOVIE_RATINGS
].index


filtered_train_ratings = train_ratings[
    train_ratings["movieId"].isin(
        popular_movie_ids
    )
].copy()


print(
    "Movies remaining:",
    filtered_train_ratings[
        "movieId"
    ].nunique()
)


# --------------------------------------------------
# 6. CREATE POPULARITY RANKING
# --------------------------------------------------

print("Building popularity baseline...")

# Count how many training ratings each movie has.
#
# More ratings = more popular.

movie_popularity = (
    filtered_train_ratings
    .groupby("movieId")
    .size()
    .sort_values(
        ascending=False
    )
)


# Convert ranking to a normal list.

popularity_ranking = (
    movie_popularity
    .index
    .tolist()
)


print(
    "Popularity ranking created."
)


# --------------------------------------------------
# 7. PREPARE USER HISTORIES
# --------------------------------------------------

evaluation_train = filtered_train_ratings[
    filtered_train_ratings["userId"].isin(
        evaluation_users
    )
]


user_seen_movies = (
    evaluation_train
    .groupby("userId")["movieId"]
    .apply(set)
    .to_dict()
)


# --------------------------------------------------
# 8. POPULARITY RECOMMENDATION FUNCTION
# --------------------------------------------------

def recommend_popular_movies(
    user_id,
    top_k=10
):

    seen_movies = user_seen_movies.get(
        user_id,
        set()
    )

    recommendations = []


    for movie_id in popularity_ranking:

        # Don't recommend something the user
        # has already rated.

        if movie_id in seen_movies:
            continue


        recommendations.append(
            movie_id
        )


        if len(recommendations) >= top_k:
            break


    return recommendations


# --------------------------------------------------
# 9. CALCULATE RECALL@10
# --------------------------------------------------

print("\nEvaluating popularity baseline...")

recalls = []

evaluated_users = 0


for number, user_id in enumerate(
    evaluation_users,
    start=1
):

    user_test = test_ratings[
        test_ratings["userId"] == user_id
    ]


    # Only test movies that survived the
    # minimum-50-ratings filter.

    relevant_movies = set(
        movie_id
        for movie_id
        in user_test["movieId"].tolist()
        if movie_id in popular_movie_ids
    )


    if len(relevant_movies) == 0:
        continue


    recommendations = recommend_popular_movies(
        user_id,
        top_k=TOP_K
    )


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


    if number % 100 == 0:

        print(
            f"Processed "
            f"{number}/"
            f"{len(evaluation_users)} users..."
        )


# --------------------------------------------------
# 10. FINAL RESULT
# --------------------------------------------------

popularity_recall = np.mean(
    recalls
)


print("\n" + "=" * 60)

print(
    "POPULARITY BASELINE EVALUATION"
)

print("=" * 60)


print(
    f"\nUsers evaluated: "
    f"{evaluated_users}"
)

print(
    f"Recall@10: "
    f"{popularity_recall:.4f}"
)


print("\n" + "=" * 60)

print(
    "CURRENT MODEL COMPARISON"
)

print("=" * 60)


print(
    f"\nPopularity baseline: "
    f"{popularity_recall:.4f}"
)

print(
    "Raw KNN:             "
    "0.1754"
)

print(
    "Mean-centered KNN:   "
    "0.1454"
)


print("\nEvaluation complete.")