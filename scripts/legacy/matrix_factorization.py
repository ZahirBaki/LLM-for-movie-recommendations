from backend.core.paths import project_path
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


# ==========================================================
# SETTINGS
# ==========================================================

RATINGS_FILE = project_path('data/raw/ml-32m/ratings.csv')
MOVIES_FILE = project_path('data/raw/ml-32m/movies.csv')

MIN_MOVIE_RATINGS = 50

# Number of latent factors learned by the model.
N_FACTORS = 50

RANDOM_STATE = 42

TOP_K = 10


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
# 2. FILTER VERY RARE MOVIES
# ==========================================================

print("\nFiltering movies...")

ratings_per_movie = (
    ratings
    .groupby("movieId")
    .size()
)

popular_movie_ids = ratings_per_movie[
    ratings_per_movie >= MIN_MOVIE_RATINGS
].index

filtered_ratings = ratings[
    ratings["movieId"].isin(popular_movie_ids)
].copy()

print(
    "Movies remaining:",
    filtered_ratings["movieId"].nunique()
)

print(
    "Ratings remaining:",
    len(filtered_ratings)
)


# ==========================================================
# 3. CREATE USER AND MOVIE INDEXES
# ==========================================================

print("\nCreating indexes...")

user_ids = filtered_ratings["userId"].unique()
movie_ids = filtered_ratings["movieId"].unique()


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
# 4. CREATE USER-MOVIE MATRIX
# ==========================================================

print("\nCreating sparse user-movie matrix...")


row_indices = (
    filtered_ratings["userId"]
    .map(user_to_index)
    .to_numpy()
)

column_indices = (
    filtered_ratings["movieId"]
    .map(movie_to_index)
    .to_numpy()
)


user_movie_matrix = csr_matrix(
    (
        filtered_ratings["rating"].to_numpy(),
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
# 5. TRAIN MATRIX FACTORIZATION MODEL
# ==========================================================

print(
    f"\nTraining matrix factorization "
    f"with {N_FACTORS} latent factors..."
)


svd = TruncatedSVD(
    n_components=N_FACTORS,
    random_state=RANDOM_STATE
)


user_factors = svd.fit_transform(
    user_movie_matrix
)


movie_factors = svd.components_.T


print("\nTraining complete.")

print(
    "User factors shape:",
    user_factors.shape
)

print(
    "Movie factors shape:",
    movie_factors.shape
)

print(
    "Explained variance:",
    svd.explained_variance_ratio_.sum()
)


# ==========================================================
# 6. RECOMMEND MOVIES FOR A USER
# ==========================================================

def recommend_for_user(user_id, top_k=10):

    if user_id not in user_to_index:
        return []


    user_index = user_to_index[user_id]


    # User's latent vector.
    user_vector = user_factors[
        user_index
    ]


    # Predict preference scores for every movie.
    predicted_scores = (
        movie_factors @ user_vector
    )


    # Movies already rated by this user.
    user_history = filtered_ratings[
        filtered_ratings["userId"]
        == user_id
    ]


    seen_movies = set(
        user_history["movieId"]
    )


    candidates = []


    for movie_index, score in enumerate(
        predicted_scores
    ):

        movie_id = index_to_movie[
            movie_index
        ]


        if movie_id in seen_movies:
            continue


        candidates.append(
            (
                movie_id,
                score
            )
        )


    candidates.sort(
        key=lambda x: x[1],
        reverse=True
    )


    return candidates[:top_k]


# ==========================================================
# 7. TEST ONE USER
# ==========================================================

test_user = user_ids[0]


print("\n" + "=" * 60)

print(
    f"TESTING USER {test_user}"
)

print("=" * 60)


# Show some movies this user rated highly.

user_history = filtered_ratings[
    filtered_ratings["userId"]
    == test_user
]


liked_movies = user_history[
    user_history["rating"] >= 4.0
].sort_values(
    "rating",
    ascending=False
)


liked_movies = liked_movies.merge(
    movies[
        ["movieId", "title"]
    ],
    on="movieId",
    how="left"
)


print("\nSome movies the user liked:\n")


for _, row in liked_movies.head(10).iterrows():

    print(
        f"{row['rating']:.1f}  "
        f"{row['title']}"
    )


# ==========================================================
# SHOW RECOMMENDATIONS
# ==========================================================

recommendations = recommend_for_user(
    test_user,
    TOP_K
)


print("\nRecommended movies:\n")


for rank, (
    movie_id,
    score
) in enumerate(
    recommendations,
    start=1
):


    movie_row = movies[
        movies["movieId"]
        == movie_id
    ]


    if len(movie_row) > 0:

        title = movie_row.iloc[0][
            "title"
        ]

    else:

        title = str(movie_id)


    print(
        f"{rank:2}. "
        f"{title} "
        f"(score={score:.4f})"
    )


print("\nDone.")