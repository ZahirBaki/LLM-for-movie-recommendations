import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# --------------------------------------------------
# 1. LOAD MOVIELENS DATA
# --------------------------------------------------

print("Loading MovieLens data...")

ratings = pd.read_csv("data/ml-32m/ratings.csv")
movies = pd.read_csv("data/ml-32m/movies.csv")
links = pd.read_csv("data/ml-32m/links.csv")

print("MovieLens data loaded.\n")


# --------------------------------------------------
# 2. BASIC INFORMATION
# --------------------------------------------------

print("Ratings shape:", ratings.shape)
print("Movies shape:", movies.shape)
print("Links shape:", links.shape)

print("\nRatings columns:")
print(ratings.columns.tolist())

print("\nMovies columns:")
print(movies.columns.tolist())

print("\nLinks columns:")
print(links.columns.tolist())


# --------------------------------------------------
# 3. LOOK AT THE DATA
# --------------------------------------------------

print("\nFirst 5 ratings:")
print(ratings.head())

print("\nFirst 5 movies:")
print(movies.head())

print("\nFirst 5 links:")
print(links.head())


# --------------------------------------------------
# 4. DATASET STATISTICS
# --------------------------------------------------

print("\nNumber of unique users:")
print(ratings["userId"].nunique())

print("\nNumber of movies with ratings:")
print(ratings["movieId"].nunique())

print("\nRating distribution:")
print(ratings["rating"].value_counts().sort_index())


# --------------------------------------------------
# 5. RATING ACTIVITY
# --------------------------------------------------

ratings_per_user = ratings.groupby("userId").size()
ratings_per_movie = ratings.groupby("movieId").size()

print("\n--- Ratings per user ---")
print(ratings_per_user.describe())

print("\n--- Ratings per movie ---")
print(ratings_per_movie.describe())


# --------------------------------------------------
# 6. MOST RATED MOVIES
# --------------------------------------------------

movie_rating_counts = (
    ratings.groupby("movieId")
    .size()
    .reset_index(name="rating_count")
)

most_rated = (
    movie_rating_counts
    .merge(movies, on="movieId")
    .sort_values("rating_count", ascending=False)
)

print("\nTop 10 most-rated movies:")
print(
    most_rated[
        ["movieId", "title", "rating_count"]
    ].head(10)
)

# --------------------------------------------------
# 7. FILTER MOVIES WITH TOO FEW RATINGS
# --------------------------------------------------

MIN_MOVIE_RATINGS = 50

popular_movie_ids = ratings_per_movie[
    ratings_per_movie >= MIN_MOVIE_RATINGS
].index

filtered_ratings = ratings[
    ratings["movieId"].isin(popular_movie_ids)
].copy()

print("\n--- After filtering ---")

print("Minimum ratings required per movie:", MIN_MOVIE_RATINGS)

print(
    "Movies remaining:",
    filtered_ratings["movieId"].nunique()
)

print(
    "Ratings remaining:",
    len(filtered_ratings)
)

print(
    "Users remaining:",
    filtered_ratings["userId"].nunique()
)
# --------------------------------------------------
# 7B. MEAN-CENTER RATINGS BY USER
# --------------------------------------------------

print("\nMean-centering ratings by user...")

# Calculate each user's average rating
user_mean_ratings = (
    filtered_ratings
    .groupby("userId")["rating"]
    .mean()
)

# Add each user's average to their ratings
filtered_ratings["user_mean"] = (
    filtered_ratings["userId"]
    .map(user_mean_ratings)
)

# Rating relative to that user's normal rating
filtered_ratings["centered_rating"] = (
    filtered_ratings["rating"]
    - filtered_ratings["user_mean"]
)

print("Mean-centering complete.")

print("\nExample centered ratings:")
print(
    filtered_ratings[
        [
            "userId",
            "movieId",
            "rating",
            "user_mean",
            "centered_rating"
        ]
    ].head(10)
)

# --------------------------------------------------
# 8. CREATE MOVIE-USER SPARSE MATRIX
# --------------------------------------------------

print("\nCreating movie-user sparse matrix...")

# Convert the real MovieLens IDs into consecutive matrix positions
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

# Convert every rating into its matrix row and column
row_indices = filtered_ratings["movieId"].map(movie_to_index)
column_indices = filtered_ratings["userId"].map(user_to_index)

movie_user_matrix = csr_matrix(
    (
        filtered_ratings["rating"].values,
        (row_indices, column_indices)
    ),
    shape=(len(movie_ids), len(user_ids))
)

print("Sparse matrix created.")

print("Matrix shape:", movie_user_matrix.shape)
print("Stored ratings:", movie_user_matrix.nnz)

# --------------------------------------------------
# 8B. CREATE MEAN-CENTERED MOVIE-USER MATRIX
# --------------------------------------------------

print("\nCreating mean-centered movie-user sparse matrix...")

centered_movie_user_matrix = csr_matrix(
    (
        filtered_ratings["centered_rating"].values,
        (row_indices, column_indices)
    ),
    shape=(len(movie_ids), len(user_ids))
)

print("Mean-centered sparse matrix created.")
print("Matrix shape:", centered_movie_user_matrix.shape)
print("Stored values:", centered_movie_user_matrix.nnz)

# --------------------------------------------------
# 9. TRAIN ITEM-ITEM COLLABORATIVE MODEL
# --------------------------------------------------

print("\nTraining collaborative filtering model...")

model = NearestNeighbors(
    metric="cosine",
    algorithm="brute"
)

model.fit(movie_user_matrix)

print("Collaborative filtering model ready.")

print("\nTraining mean-centered collaborative model...")

centered_model = NearestNeighbors(
    metric="cosine",
    algorithm="brute"
)

centered_model.fit(centered_movie_user_matrix)

print("Mean-centered collaborative model ready.")
# --------------------------------------------------
# 10. FUNCTION TO FIND SIMILAR MOVIES
# --------------------------------------------------

def recommend_similar_movies(movie_title, n=10):

    # Find movies containing the title
    matches = movies[
        movies["title"].str.contains(
            movie_title,
            case=False,
            na=False,
            regex=False
        )
    ]

    if matches.empty:
        print(f"\nMovie not found: {movie_title}")
        return

    # For now, use the first match
    selected_movie = matches.iloc[0]

    movie_id = selected_movie["movieId"]
    full_title = selected_movie["title"]

    # Check whether movie survived our 50-rating filter
    if movie_id not in movie_to_index:
        print(
            f"\n{full_title} does not have enough ratings "
            "for collaborative filtering."
        )
        return

    matrix_index = movie_to_index[movie_id]

    # Find nearest movies
    distances, indices = model.kneighbors(
        movie_user_matrix[matrix_index],
        n_neighbors=n + 1
    )

    print(f"\nMovies collaboratively similar to: {full_title}\n")

    recommendation_number = 1

    for distance, index in zip(
        distances.flatten(),
        indices.flatten()
    ):
        recommended_movie_id = index_to_movie[index]

        # Skip the movie itself
        if recommended_movie_id == movie_id:
            continue

        movie_info = movies[
            movies["movieId"] == recommended_movie_id
        ].iloc[0]

        similarity = 1 - distance

        print(
            f"{recommendation_number}. "
            f"{movie_info['title']} "
            f"| similarity: {similarity:.3f}"
        )

        recommendation_number += 1

        if recommendation_number > n:
            break

# --------------------------------------------------
# 10B. COMPARE RAW VS MEAN-CENTERED MODEL
# --------------------------------------------------

def compare_models(movie_title, n=10):

    matches = movies[
        movies["title"].str.contains(
            movie_title,
            case=False,
            na=False,
            regex=False
        )
    ]

    if matches.empty:
        print(f"\nMovie not found: {movie_title}")
        return

    selected_movie = matches.iloc[0]

    movie_id = selected_movie["movieId"]
    full_title = selected_movie["title"]

    if movie_id not in movie_to_index:
        print(
            f"\n{full_title} does not have enough ratings "
            "for collaborative filtering."
        )
        return

    matrix_index = movie_to_index[movie_id]

    # Raw-rating model
    raw_distances, raw_indices = model.kneighbors(
        movie_user_matrix[matrix_index],
        n_neighbors=n + 1
    )

    # Mean-centered model
    centered_distances, centered_indices = (
        centered_model.kneighbors(
            centered_movie_user_matrix[matrix_index],
            n_neighbors=n + 1
        )
    )

    print("\n" + "=" * 70)
    print(f"COMPARISON FOR: {full_title}")
    print("=" * 70)

    print("\nRAW RATINGS:\n")

    number = 1

    for distance, index in zip(
        raw_distances.flatten(),
        raw_indices.flatten()
    ):

        recommended_movie_id = index_to_movie[index]

        if recommended_movie_id == movie_id:
            continue

        movie_info = movies[
            movies["movieId"] == recommended_movie_id
        ].iloc[0]

        similarity = 1 - distance

        print(
            f"{number}. {movie_info['title']} "
            f"| similarity: {similarity:.3f}"
        )

        number += 1

        if number > n:
            break

    print("\nMEAN-CENTERED RATINGS:\n")

    number = 1

    for distance, index in zip(
        centered_distances.flatten(),
        centered_indices.flatten()
    ):

        recommended_movie_id = index_to_movie[index]

        if recommended_movie_id == movie_id:
            continue

        movie_info = movies[
            movies["movieId"] == recommended_movie_id
        ].iloc[0]

        similarity = 1 - distance

        print(
            f"{number}. {movie_info['title']} "
            f"| similarity: {similarity:.3f}"
        )

        number += 1

        if number > n:
            break

# --------------------------------------------------
# 11. TEST
# --------------------------------------------------

compare_models("Interstellar", n=10)