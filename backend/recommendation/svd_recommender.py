from backend.core.paths import project_path
import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


# ==========================================================
# SETTINGS
# ==========================================================

RATINGS_FILE = project_path('data/raw/ml-32m/ratings.csv')
MOVIES_FILE = project_path('data/raw/ml-32m/movies.csv')
MAPPING_FILE = project_path('data/processed/movie_mapping.csv')

MIN_MOVIE_RATINGS = 50
N_FACTORS = 50
RANDOM_STATE = 42
# ==========================================================
# SAVED MODEL FILES
# ==========================================================

SVD_MOVIE_FACTORS_FILE = project_path('data/models/svd_movie_factors.npy')
SVD_MOVIE_INDEX_FILE = project_path('data/models/svd_movie_index.csv')

# ==========================================================
# SVD RECOMMENDER CLASS
# ==========================================================

class SVDRecommender:

    def __init__(self):

        self.ratings = None
        self.movies = None
        self.mapping = None

        self.user_movie_matrix = None

        self.user_to_index = {}
        self.movie_to_index = {}
        self.index_to_movie = {}

        self.user_factors = None
        self.movie_factors = None

        self.user_seen = {}

        self.svd = None


    # ======================================================
    # LOAD DATA
    # ======================================================

    def load_data(self):

        print("Loading MovieLens data...")

        self.ratings = pd.read_csv(
            RATINGS_FILE
        )

        self.movies = pd.read_csv(
            MOVIES_FILE
        )

        self.mapping = pd.read_csv(
            MAPPING_FILE
        )

        print(
            "Ratings:",
            len(self.ratings)
        )

        print(
            "Users:",
            self.ratings["userId"].nunique()
        )

        print(
            "Movies:",
            self.ratings["movieId"].nunique()
        )


    # ======================================================
    # FILTER MOVIES
    # ======================================================

    def filter_movies(self):

        print("\nFiltering movies...")

        ratings_per_movie = (
            self.ratings
            .groupby("movieId")
            .size()
        )

        valid_movies = ratings_per_movie[
            ratings_per_movie
            >= MIN_MOVIE_RATINGS
        ].index

        self.ratings = self.ratings[
            self.ratings["movieId"].isin(
                valid_movies
            )
        ].copy()

        print(
            "Movies remaining:",
            self.ratings["movieId"].nunique()
        )

        print(
            "Ratings remaining:",
            len(self.ratings)
        )


    # ======================================================
    # CREATE INDEXES
    # ======================================================

    def create_indexes(self):

        print("\nCreating indexes...")

        user_ids = (
            self.ratings["userId"]
            .unique()
        )

        movie_ids = (
            self.ratings["movieId"]
            .unique()
        )

        self.user_to_index = {
            user_id: index
            for index, user_id
            in enumerate(user_ids)
        }

        self.movie_to_index = {
            movie_id: index
            for index, movie_id
            in enumerate(movie_ids)
        }

        self.index_to_movie = {
            index: movie_id
            for movie_id, index
            in self.movie_to_index.items()
        }


    # ======================================================
    # CREATE SPARSE MATRIX
    # ======================================================

    def create_matrix(self):

        print(
            "\nCreating sparse user-movie matrix..."
        )

        row_indices = (
            self.ratings["userId"]
            .map(self.user_to_index)
            .to_numpy()
        )

        column_indices = (
            self.ratings["movieId"]
            .map(self.movie_to_index)
            .to_numpy()
        )

        values = (
            self.ratings["rating"]
            .to_numpy()
        )

        self.user_movie_matrix = csr_matrix(
            (
                values,
                (
                    row_indices,
                    column_indices
                )
            ),
            shape=(
                len(self.user_to_index),
                len(self.movie_to_index)
            )
        )

        print(
            "Matrix shape:",
            self.user_movie_matrix.shape
        )

        print(
            "Stored ratings:",
            self.user_movie_matrix.nnz
        )


    # ======================================================
    # TRAIN SVD
    # ======================================================

    def train(self):

        print(
            f"\nTraining SVD with "
            f"{N_FACTORS} factors..."
        )

        self.svd = TruncatedSVD(
            n_components=N_FACTORS,
            random_state=RANDOM_STATE
        )

        self.user_factors = (
            self.svd.fit_transform(
                self.user_movie_matrix
            )
        )

        self.movie_factors = (
            self.svd.components_.T
        )

        explained_variance = (
            self.svd
            .explained_variance_ratio_
            .sum()
        )

        print("\nTraining complete.")

        print(
            "User factors:",
            self.user_factors.shape
        )

        print(
            "Movie factors:",
            self.movie_factors.shape
        )

        print(
            "Explained variance:",
            explained_variance
        )


    # ======================================================
    # PREPARE USER HISTORY
    # ======================================================

    def prepare_user_history(self):

        print(
            "\nPreparing user history..."
        )

        self.user_seen = (
            self.ratings
            .groupby("userId")["movieId"]
            .apply(set)
            .to_dict()
        )


    # ======================================================
    # BUILD / TRAIN EVERYTHING
    # ======================================================

    def fit(self):

        self.load_data()

        self.filter_movies()

        self.create_indexes()

        self.create_matrix()

        self.train()

        self.prepare_user_history()

        print(
            "\nSVD recommender ready."
        )

    # ======================================================
    # SAVE TRAINED MODEL
    # ======================================================

    def save_model(self):

        if self.movie_factors is None:
            raise ValueError(
                "SVD model has not been trained yet."
            )

        print("\nSaving trained SVD model...")

        # Save latent movie vectors
        np.save(
            SVD_MOVIE_FACTORS_FILE,
            self.movie_factors
        )

        # Save relationship between matrix position
        # and MovieLens movieId
        movie_index = pd.DataFrame({

            "movieId": [
                self.index_to_movie[index]
                for index in range(
                    len(self.index_to_movie)
                )
            ],

            "movie_index": range(
                len(self.index_to_movie)
            )
        })

        movie_index.to_csv(
            SVD_MOVIE_INDEX_FILE,
            index=False
        )

        print(
            "Movie factors saved:",
            self.movie_factors.shape
        )

        print(
            "Movie index saved:",
            len(movie_index)
        )

        print(
            "\nSVD model saved successfully."
        )

    # ======================================================
    # LOAD TRAINED MODEL
    # ======================================================

    def load_model(self):

        print("\nLoading saved SVD model...")

        # --------------------------------------------------
        # Load the small metadata files needed for
        # recommendations.
        # --------------------------------------------------

        self.movies = pd.read_csv(
            MOVIES_FILE
        )

        self.mapping = pd.read_csv(
            MAPPING_FILE
        )


        # --------------------------------------------------
        # Load latent movie vectors
        # --------------------------------------------------

        self.movie_factors = np.load(
            SVD_MOVIE_FACTORS_FILE
        )


        # --------------------------------------------------
        # Rebuild movie indexes
        # --------------------------------------------------

        movie_index = pd.read_csv(
            SVD_MOVIE_INDEX_FILE
        )

        self.movie_to_index = {

            int(row.movieId):
                int(row.movie_index)

            for row
            in movie_index.itertuples()
        }

        self.index_to_movie = {

            int(row.movie_index):
                int(row.movieId)

            for row
            in movie_index.itertuples()
        }


        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        if (
            len(self.movie_to_index)
            != len(self.movie_factors)
        ):

            raise ValueError(
                "Saved SVD movie index does not match "
                "saved movie factors."
            )


        print(
            "Movie factors:",
            self.movie_factors.shape
        )

        print(
            "Movies in collaborative model:",
            len(self.movie_to_index)
        )

        print(
            "\nSaved SVD model loaded."
        )
    # ======================================================
    # EXISTING MOVIELENS USER
    # ======================================================

    def get_scores(
        self,
        user_id,
        top_n=1000
    ):

        if user_id not in self.user_to_index:

            print(
                f"User {user_id} "
                "does not exist in MovieLens."
            )

            return pd.DataFrame()


        # --------------------------------------------------
        # Get user's latent vector
        # --------------------------------------------------

        user_index = (
            self.user_to_index[user_id]
        )

        user_vector = (
            self.user_factors[user_index]
        )


        # --------------------------------------------------
        # Score every movie
        # --------------------------------------------------

        scores = (
            self.movie_factors
            @ user_vector
        )


        result = pd.DataFrame({
            "movie_index":
                np.arange(len(scores)),

            "svd_raw_score":
                scores
        })


        result["movieId"] = (
            result["movie_index"]
            .map(self.index_to_movie)
        )


        # --------------------------------------------------
        # Remove movies already seen
        # --------------------------------------------------

        seen = self.user_seen.get(
            user_id,
            set()
        )

        result = result[
            ~result["movieId"].isin(seen)
        ].copy()


        # --------------------------------------------------
        # Connect MovieLens to TMDB
        # --------------------------------------------------

        mapping = self.mapping[
            self.mapping[
                "found_in_tmdb_catalog"
            ] == True
        ].copy()


        mapping = mapping[
            [
                "movieId",
                "tmdb_id"
            ]
        ].dropna()


        mapping["tmdb_id"] = (
            mapping["tmdb_id"]
            .astype(int)
        )


        result = result.merge(
            mapping,
            on="movieId",
            how="inner"
        )


        # --------------------------------------------------
        # Normalize SVD scores to 0-1
        # --------------------------------------------------

        minimum_score = (
            result["svd_raw_score"].min()
        )

        maximum_score = (
            result["svd_raw_score"].max()
        )


        if maximum_score > minimum_score:

            result["svd_score"] = (
                (
                    result["svd_raw_score"]
                    - minimum_score
                )
                /
                (
                    maximum_score
                    - minimum_score
                )
            )

        else:

            result["svd_score"] = 0.0


        # --------------------------------------------------
        # Add movie titles
        # --------------------------------------------------

        result = result.merge(
            self.movies[
                [
                    "movieId",
                    "title"
                ]
            ],
            on="movieId",
            how="left"
        )


        # --------------------------------------------------
        # Sort
        # --------------------------------------------------

        result = result.sort_values(
            "svd_score",
            ascending=False
        )


        return (
            result[
                [
                    "movieId",
                    "tmdb_id",
                    "title",
                    "svd_raw_score",
                    "svd_score"
                ]
            ]
            .head(top_n)
            .reset_index(drop=True)
        )


    # ======================================================
    # NEW USER
    # ======================================================

    def get_scores_for_favorite_tmdb_movies(
        self,
        favorite_tmdb_ids,
        top_n=None
    ):

        if not favorite_tmdb_ids:

            print(
                "No favorite movies were provided."
            )

            return pd.DataFrame()


        # --------------------------------------------------
        # Prepare MovieLens -> TMDB mapping
        # --------------------------------------------------

        mapping = self.mapping[
            self.mapping[
                "found_in_tmdb_catalog"
            ] == True
        ].copy()


        mapping = mapping.dropna(
            subset=["tmdb_id"]
        )


        mapping["tmdb_id"] = (
            mapping["tmdb_id"]
            .astype(int)
        )


        # --------------------------------------------------
        # Find the favorite movies in MovieLens
        # --------------------------------------------------

        favorite_mapping = mapping[
            mapping["tmdb_id"].isin(
                favorite_tmdb_ids
            )
        ].copy()


        # --------------------------------------------------
        # Only use favorite movies that survived the
        # >= 50 ratings collaborative filter
        # --------------------------------------------------

        favorite_mapping = (
            favorite_mapping[
                favorite_mapping[
                    "movieId"
                ].isin(
                    self.movie_to_index
                )
            ]
        )


        if favorite_mapping.empty:

            print(
                "None of the favorite movies "
                "have usable MovieLens SVD data."
            )

            return pd.DataFrame()


        favorite_movie_ids = (
            favorite_mapping["movieId"]
            .unique()
            .tolist()
        )


        print(
            "\nFavorite movies usable by SVD:",
            len(favorite_movie_ids)
        )


        # --------------------------------------------------
        # Show which favorite movies SVD found
        # --------------------------------------------------

        found_favorites = (
            favorite_mapping[
                [
                    "movieId",
                    "tmdb_id"
                ]
            ]
            .merge(
                self.movies[
                    [
                        "movieId",
                        "title"
                    ]
                ],
                on="movieId",
                how="left"
            )
        )


        print(
            "\nFavorites found in collaborative model:"
        )

        print(
            found_favorites.to_string(
                index=False
            )
        )


        # --------------------------------------------------
        # Get latent vectors for favorite movies
        # --------------------------------------------------

        favorite_indices = [
            self.movie_to_index[
                movie_id
            ]

            for movie_id
            in favorite_movie_ids
        ]


        favorite_vectors = (
            self.movie_factors[
                favorite_indices
            ]
        )


        # --------------------------------------------------
        # Create temporary user vector
        #
        # We average the latent vectors of all favorite
        # movies.
        # --------------------------------------------------

        temporary_user_vector = (
            favorite_vectors.mean(
                axis=0
            )
        )


        # --------------------------------------------------
        # Score every collaborative movie
        # --------------------------------------------------

        scores = (
            self.movie_factors
            @ temporary_user_vector
        )


        result = pd.DataFrame({
            "movie_index":
                np.arange(len(scores)),

            "svd_raw_score":
                scores
        })


        result["movieId"] = (
            result["movie_index"]
            .map(self.index_to_movie)
        )


        # --------------------------------------------------
        # Remove the favorite movies themselves
        # --------------------------------------------------

        result = result[
            ~result["movieId"].isin(
                favorite_movie_ids
            )
        ].copy()


        # --------------------------------------------------
        # Connect recommendations to TMDB
        # --------------------------------------------------

        valid_mapping = mapping[
            [
                "movieId",
                "tmdb_id"
            ]
        ].drop_duplicates(
            subset=["movieId"]
        )


        result = result.merge(
            valid_mapping,
            on="movieId",
            how="inner"
        )


        # --------------------------------------------------
        # Normalize SVD scores to 0-1
        # --------------------------------------------------

        minimum_score = (
            result["svd_raw_score"].min()
        )

        maximum_score = (
            result["svd_raw_score"].max()
        )


        if maximum_score > minimum_score:

            result["svd_score"] = (
                (
                    result["svd_raw_score"]
                    - minimum_score
                )
                /
                (
                    maximum_score
                    - minimum_score
                )
            )

        else:

            result["svd_score"] = 0.0


        # --------------------------------------------------
        # Add movie titles
        # --------------------------------------------------

        result = result.merge(
            self.movies[
                [
                    "movieId",
                    "title"
                ]
            ],
            on="movieId",
            how="left"
        )


        # --------------------------------------------------
        # Sort by collaborative score
        # --------------------------------------------------

        result = result.sort_values(
            "svd_score",
            ascending=False
        )


        # --------------------------------------------------
        # Select useful columns
        # --------------------------------------------------

        result = result[
            [
                "movieId",
                "tmdb_id",
                "title",
                "svd_raw_score",
                "svd_score"
            ]
        ].reset_index(
            drop=True
        )


        # --------------------------------------------------
        # Optional Top-N limit
        # --------------------------------------------------

        if top_n is not None:

            result = result.head(
                top_n
            )


        return result


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    recommender = SVDRecommender()


    # ------------------------------------------------------
    # Train SVD
    # ------------------------------------------------------

    recommender.fit()


    # ------------------------------------------------------
    # TEST A NEW USER
    #
    # TMDB IDs:
    #
    # Interstellar     = 157336
    # The Godfather    = 238
    # The Dark Knight  = 155
    # ------------------------------------------------------

    favorite_tmdb_ids = [
        157336,
        238,
        155
    ]


    recommendations = (
        recommender
        .get_scores_for_favorite_tmdb_movies(
            favorite_tmdb_ids,
            top_n=20
        )
    )


    # ------------------------------------------------------
    # Print recommendations
    # ------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "SVD RECOMMENDATIONS "
        "FOR NEW USER"
    )

    print(
        "=" * 70
    )


    if recommendations.empty:

        print(
            "No collaborative "
            "recommendations available."
        )

    else:

        print(
            recommendations.to_string(
                index=False
            )
        )