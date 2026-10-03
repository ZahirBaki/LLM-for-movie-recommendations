import pandas as pd
import numpy as np
import difflib


# ==========================================================
# SETTINGS
# ==========================================================

MOVIES_FILE = "data/tmdb_movies.csv"
EMBEDDINGS_FILE = "data/movie_embeddings.npy"

STORY_MATCH_MINIMUM = 0.25
GENRE_MATCH_THRESHOLD = 0.25


# ==========================================================
# CONTENT RECOMMENDER
# ==========================================================

class ContentRecommender:

    def __init__(self):

        self.movies = None
        self.overview_embeddings = None

        self.known_directors = []
        self.known_actors = []
        self.known_genres = []

        self.load_data()
        self.build_known_values()


    # ======================================================
    # LOAD DATA
    # ======================================================

    def load_data(self):

        print("Loading content recommender...")

        self.movies = pd.read_csv(
            MOVIES_FILE
        )

        self.movies = (
            self.movies
            .reset_index(drop=True)
        )

        for column in [
            "genres",
            "director",
            "actors",
            "overview"
        ]:

            self.movies[column] = (
                self.movies[column]
                .fillna("")
            )

        self.overview_embeddings = np.load(
            EMBEDDINGS_FILE
        )

        if len(self.overview_embeddings) != len(
            self.movies
        ):

            raise ValueError(
                "Movie catalog and embeddings "
                "do not match. "
                f"Movies: {len(self.movies)}, "
                f"Embeddings: "
                f"{len(self.overview_embeddings)}"
            )

        print(
            "Movies loaded:",
            len(self.movies)
        )

        print(
            "Embeddings:",
            self.overview_embeddings.shape
        )

        print(
            "Content recommender ready."
        )


    # ======================================================
    # NORMALIZE TEXT
    # ======================================================

    @staticmethod
    def normalize(text):

        return str(text).strip().lower()


    # ======================================================
    # BUILD KNOWN VALUES
    # ======================================================

    def build_known_values(self):

        self.known_directors = sorted(
            {
                director.strip()

                for director
                in self.movies["director"]

                if director.strip()
            }
        )

        actors = set()

        for actor_text in self.movies["actors"]:

            for actor in actor_text.split(","):

                actor = actor.strip()

                if actor:
                    actors.add(actor)

        self.known_actors = sorted(
            actors
        )

        genres = set()

        for genre_text in self.movies["genres"]:

            for genre in genre_text.split(","):

                genre = genre.strip()

                if genre:
                    genres.add(genre)

        self.known_genres = sorted(
            genres
        )


    # ======================================================
    # FIND BEST TEXT MATCH
    # ======================================================

    def find_best_match(
        self,
        user_input,
        choices,
        label
    ):

        user_input = str(
            user_input
        ).strip()

        if not user_input:
            return None

        # Exact match
        for choice in choices:

            if (
                self.normalize(choice)
                == self.normalize(user_input)
            ):

                return choice

        # Partial match
        partial_matches = [

            choice

            for choice in choices

            if self.normalize(user_input)
            in self.normalize(choice)

        ]

        if len(partial_matches) == 1:

            match = partial_matches[0]

            print(
                f'{label}: "{user_input}" '
                f'matched to "{match}"'
            )

            return match

        # Fuzzy match
        fuzzy_matches = (
            difflib.get_close_matches(
                user_input,
                choices,
                n=3,
                cutoff=0.6
            )
        )

        if fuzzy_matches:

            best_match = (
                fuzzy_matches[0]
            )

            print(
                f'{label}: "{user_input}" '
                f'matched to "{best_match}"'
            )

            return best_match

        print(
            f'Warning: {label.lower()} '
            f'"{user_input}" was not found.'
        )

        return None


    # ======================================================
    # VALIDATE USER PREFERENCES
    # ======================================================

    def validate_preferences(
        self,
        favorite_movies,
        favorite_genres,
        favorite_actors,
        favorite_directors
    ):

        # Movies
        movie_titles = (
            self.movies[
                "title"
            ].tolist()
        )

        validated_movies = []

        for movie in favorite_movies:

            match = self.find_best_match(
                movie,
                movie_titles,
                "Movie"
            )

            if match:
                validated_movies.append(
                    match
                )

        # Genres
        validated_genres = []

        for genre in favorite_genres:

            match = self.find_best_match(
                genre,
                self.known_genres,
                "Genre"
            )

            if match:
                validated_genres.append(
                    match
                )

        # Actors
        validated_actors = []

        for actor in favorite_actors:

            match = self.find_best_match(
                actor,
                self.known_actors,
                "Actor"
            )

            if match:
                validated_actors.append(
                    match
                )

        # Directors
        validated_directors = []

        for director in favorite_directors:

            match = self.find_best_match(
                director,
                self.known_directors,
                "Director"
            )

            if match:
                validated_directors.append(
                    match
                )

        return (
            validated_movies,
            validated_genres,
            validated_actors,
            validated_directors
        )


    # ======================================================
    # GET CONTENT SCORES
    # ======================================================

    def get_scores(
        self,
        favorite_movies=None,
        favorite_genres=None,
        favorite_actors=None,
        favorite_directors=None,
        top_n=None
    ):

        # --------------------------------------------------
        # Handle missing preferences
        # --------------------------------------------------

        favorite_movies = (
            favorite_movies or []
        )

        favorite_genres = (
            favorite_genres or []
        )

        favorite_actors = (
            favorite_actors or []
        )

        favorite_directors = (
            favorite_directors or []
        )


        # --------------------------------------------------
        # Validate preferences
        # --------------------------------------------------

        (
            favorite_movies,
            favorite_genres,
            favorite_actors,
            favorite_directors

        ) = self.validate_preferences(

            favorite_movies,
            favorite_genres,
            favorite_actors,
            favorite_directors
        )


        if not any([
            favorite_movies,
            favorite_genres,
            favorite_actors,
            favorite_directors
        ]):

            print(
                "No valid content preferences "
                "were provided."
            )

            return pd.DataFrame()


        # --------------------------------------------------
        # Work on copy
        # --------------------------------------------------

        movies = self.movies.copy()


        # --------------------------------------------------
        # Score columns
        # --------------------------------------------------

        movies["story_raw"] = 0.0
        movies["story_score"] = 0.0

        movies["genre_score"] = 0.0
        movies["actor_score"] = 0.0
        movies["director_score"] = 0.0


        # ==================================================
        # FIND FAVORITE MOVIE ROWS
        # ==================================================

        favorite_movie_indices = []

        favorite_movie_rows = []

        for movie_name in favorite_movies:

            matches = movies[
                movies["title"].str.lower()
                == movie_name.lower()
            ]

            if not matches.empty:

                index = matches.index[0]

                favorite_movie_indices.append(
                    index
                )

                favorite_movie_rows.append(
                    movies.loc[index]
                )


        # ==================================================
        # STORY SCORE
        # ==================================================

        if favorite_movie_indices:

            favorite_embeddings = (
                self.overview_embeddings[
                    favorite_movie_indices
                ]
            )

            # Embeddings are normalized,
            # therefore dot product = cosine similarity.

            similarity_matrix = np.dot(
                favorite_embeddings,
                self.overview_embeddings.T
            )

            average_similarity = (
                similarity_matrix.mean(
                    axis=0
                )
            )

            best_similarity = (
                similarity_matrix.max(
                    axis=0
                )
            )

            # 70% average similarity
            # 30% strongest similarity

            movies["story_raw"] = (
                average_similarity * 0.70
                + best_similarity * 0.30
            )


        # ==================================================
        # INFER PREFERENCES FROM FAVORITE MOVIES
        # ==================================================

        inferred_genres = set()
        inferred_actors = set()
        inferred_directors = set()


        for movie_row in favorite_movie_rows:

            # ----------------------------------------------
            # Genres
            # ----------------------------------------------

            genre_text = (
                movie_row["genres"]
            )

            for genre in genre_text.split(","):

                genre = (
                    genre
                    .strip()
                    .lower()
                )

                if genre:

                    inferred_genres.add(
                        genre
                    )


            # ----------------------------------------------
            # Actors
            # ----------------------------------------------

            actor_text = (
                movie_row["actors"]
            )

            for actor in actor_text.split(","):

                actor = (
                    actor
                    .strip()
                    .lower()
                )

                if actor:

                    inferred_actors.add(
                        actor
                    )


            # ----------------------------------------------
            # Director
            # ----------------------------------------------

            director = (
                str(
                    movie_row["director"]
                )
                .strip()
                .lower()
            )

            if director:

                inferred_directors.add(
                    director
                )


        # ==================================================
        # EXPLICIT PREFERENCES
        # ==================================================

        explicit_genres = {
            genre.lower()
            for genre in favorite_genres
        }

        explicit_actors = {
            actor.lower()
            for actor in favorite_actors
        }

        explicit_directors = {
            director.lower()
            for director in favorite_directors
        }


        # ==================================================
        # GENRE SCORE
        # ==================================================

        def calculate_genre_score(
            movie_genres
        ):

            movie_genre_set = {

                genre.strip().lower()

                for genre
                in movie_genres.split(",")

                if genre.strip()
            }

            explicit_score = 0.0

            if explicit_genres:

                explicit_matches = (
                    movie_genre_set
                    .intersection(
                        explicit_genres
                    )
                )

                explicit_score = (
                    len(explicit_matches)
                    / len(explicit_genres)
                )


            inferred_score = 0.0

            if inferred_genres:

                inferred_matches = (
                    movie_genre_set
                    .intersection(
                        inferred_genres
                    )
                )

                inferred_score = (
                    len(inferred_matches)
                    / len(inferred_genres)
                )


            if (
                explicit_genres
                and inferred_genres
            ):

                return (
                    explicit_score * 0.70
                    + inferred_score * 0.30
                )

            elif explicit_genres:

                return explicit_score

            elif inferred_genres:

                return inferred_score

            return 0.0


        movies["genre_score"] = (
            movies["genres"]
            .apply(
                calculate_genre_score
            )
        )


        # ==================================================
        # ACTOR SCORE
        # ==================================================

        def calculate_actor_score(
            movie_actors
        ):

            movie_actor_set = {

                actor.strip().lower()

                for actor
                in movie_actors.split(",")

                if actor.strip()
            }


            # ----------------------------------------------
            # Explicit actors
            # ----------------------------------------------

            explicit_score = 0.0

            if explicit_actors:

                matches = (
                    movie_actor_set
                    .intersection(
                        explicit_actors
                    )
                )

                explicit_score = (
                    len(matches)
                    / len(explicit_actors)
                )


            # ----------------------------------------------
            # Inferred actors
            # ----------------------------------------------

            inferred_score = 0.0

            if inferred_actors:

                matches = (
                    movie_actor_set
                    .intersection(
                        inferred_actors
                    )
                )

                inferred_score = (
                    len(matches)
                    / len(inferred_actors)
                )


            # Explicit user preference gets
            # more importance than inference.

            if (
                explicit_actors
                and inferred_actors
            ):

                return (
                    explicit_score * 0.70
                    + inferred_score * 0.30
                )

            elif explicit_actors:

                return explicit_score

            elif inferred_actors:

                return inferred_score

            return 0.0


        movies["actor_score"] = (
            movies["actors"]
            .apply(
                calculate_actor_score
            )
        )


        # ==================================================
        # DIRECTOR SCORE
        # ==================================================

        def calculate_director_score(
            movie_director
        ):

            movie_director = (
                str(movie_director)
                .strip()
                .lower()
            )


            explicit_score = 0.0

            if explicit_directors:

                if (
                    movie_director
                    in explicit_directors
                ):

                    explicit_score = 1.0


            inferred_score = 0.0

            if inferred_directors:

                if (
                    movie_director
                    in inferred_directors
                ):

                    inferred_score = 1.0


            if (
                explicit_directors
                and inferred_directors
            ):

                return (
                    explicit_score * 0.70
                    + inferred_score * 0.30
                )

            elif explicit_directors:

                return explicit_score

            elif inferred_directors:

                return inferred_score

            return 0.0


        movies["director_score"] = (
            movies["director"]
            .apply(
                calculate_director_score
            )
        )


        # ==================================================
        # REMOVE FAVORITES FROM CANDIDATES
        # ==================================================

        favorite_movies_lower = [

            movie.lower()

            for movie in favorite_movies
        ]

        candidate_mask = (

            ~movies["title"]
            .str.lower()
            .isin(
                favorite_movies_lower
            )
        )


        # ==================================================
        # NORMALIZE STORY SCORE
        # ==================================================

        if favorite_movies:

            candidate_story_scores = (

                movies.loc[
                    candidate_mask,
                    "story_raw"
                ]

                .clip(lower=0)
            )

            max_story_score = (
                candidate_story_scores.max()
            )

            if max_story_score > 0:

                movies["story_score"] = (

                    movies["story_raw"]
                    .clip(lower=0)

                    / max_story_score
                )

                movies["story_score"] = (

                    movies["story_score"]
                    .clip(
                        lower=0,
                        upper=1
                    )
                )


        # ==================================================
        # CONTENT WEIGHTS
        # ==================================================

        base_weights = {

            "story_score": 0.40,

            "genre_score": 0.25,

            "director_score": 0.20,

            "actor_score": 0.15
        }


        # ==================================================
        # DETERMINE ACTIVE PREFERENCES
        # ==================================================

        active_scores = []


        if favorite_movies:

            active_scores.append(
                "story_score"
            )


        if (
            explicit_genres
            or inferred_genres
        ):

            active_scores.append(
                "genre_score"
            )


        if (
            explicit_directors
            or inferred_directors
        ):

            active_scores.append(
                "director_score"
            )


        if (
            explicit_actors
            or inferred_actors
        ):

            active_scores.append(
                "actor_score"
            )


        active_weight_total = sum(

            base_weights[score]

            for score
            in active_scores
        )


        # ==================================================
        # WEIGHTED BASE SCORE
        # ==================================================

        movies["base_score"] = 0.0


        if active_weight_total > 0:

            for score in active_scores:

                adjusted_weight = (

                    base_weights[score]

                    / active_weight_total
                )

                movies["base_score"] += (

                    movies[score]

                    * adjusted_weight
                )


        # ==================================================
        # STORY MATCH THRESHOLD
        # ==================================================

        if favorite_movies:

            candidate_raw_story = (

                movies.loc[
                    candidate_mask,
                    "story_raw"
                ]
            )

            distribution_story_threshold = (

                np.percentile(
                    candidate_raw_story,
                    75
                )
            )

            story_match_threshold = max(

                STORY_MATCH_MINIMUM,

                distribution_story_threshold
            )

        else:

            story_match_threshold = (
                STORY_MATCH_MINIMUM
            )


        # ==================================================
        # COUNT MEANINGFUL MATCHES
        # ==================================================

        movies["match_count"] = 0


        # Story

        if favorite_movies:

            movies.loc[
                movies["story_raw"]
                >= story_match_threshold,

                "match_count"

            ] += 1


        # Genre

        if (
            explicit_genres
            or inferred_genres
        ):

            movies.loc[
                movies["genre_score"]
                >= GENRE_MATCH_THRESHOLD,

                "match_count"

            ] += 1


        # Director

        if (
            explicit_directors
            or inferred_directors
        ):

            movies.loc[
                movies["director_score"] > 0,

                "match_count"

            ] += 1


        # Actor

        if (
            explicit_actors
            or inferred_actors
        ):

            movies.loc[
                movies["actor_score"] > 0,

                "match_count"

            ] += 1


        # ==================================================
        # MULTI-MATCH BONUS
        # ==================================================

        number_of_active_categories = (
            len(active_scores)
        )

        movies[
            "multi_match_bonus"
        ] = 0.0


        if number_of_active_categories > 1:

            movies[
                "multi_match_bonus"
            ] = (

                (
                    movies["match_count"]
                    - 1
                )
                .clip(lower=0)

                /

                (
                    number_of_active_categories
                    - 1
                )

            ) * 0.10


        # ==================================================
        # FINAL CONTENT SCORE
        # ==================================================

        movies["content_score"] = (

            movies["base_score"]

            + movies[
                "multi_match_bonus"
            ]
        )


        # ==================================================
        # REMOVE FAVORITES
        # ==================================================

        result = movies[
            candidate_mask
        ].copy()


        # ==================================================
        # SORT
        # ==================================================

        result = (
            result
            .sort_values(
                "content_score",
                ascending=False
            )
        )


        # ==================================================
        # RETURN COLUMNS
        # ==================================================

        columns = [

            "tmdb_id",
            "title",
            "release_year",

            "content_score",

            "story_score",
            "genre_score",
            "director_score",
            "actor_score",

            "multi_match_bonus"
        ]


        result = (

            result[columns]

            .reset_index(drop=True)
        )


        if top_n is not None:

            result = result.head(
                top_n
            )


        return result


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    recommender = (
        ContentRecommender()
    )

    recommendations = (
        recommender.get_scores(

            favorite_movies=[
                "Interstellar",
                "The Godfather",
                "The Dark Knight"
            ],

            favorite_genres=[],

            favorite_actors=[],

            favorite_directors=[],

            top_n=20
        )
    )


    print(
        "\n" + "=" * 80
    )

    print(
        "CONTENT RECOMMENDATIONS "
        "FOR NEW USER"
    )

    print(
        "=" * 80
    )


    if recommendations.empty:

        print(
            "No content recommendations "
            "available."
        )

    else:

        print(
            recommendations.to_string(
                index=False
            )
        )