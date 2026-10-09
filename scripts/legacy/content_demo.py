from backend.core.paths import project_path
import pandas as pd
import numpy as np
import difflib

from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. Load movies
# --------------------------------------------------

movies = pd.read_csv(project_path('data/processed/tmdb_movies.csv'))

print(f"Movies loaded: {len(movies)}")


# --------------------------------------------------
# 2. Handle missing values
# --------------------------------------------------

for column in ["genres", "director", "actors", "overview"]:
    movies[column] = movies[column].fillna("")


# --------------------------------------------------
# 3. Load semantic model
# --------------------------------------------------

print("\nLoading semantic model...")

semantic_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Semantic model loaded.")


# --------------------------------------------------
# 4. Create semantic embeddings
# --------------------------------------------------

# --------------------------------------------------
# 4. Load precomputed semantic embeddings
# --------------------------------------------------

print("Loading precomputed movie embeddings...")

overview_embeddings = np.load(
    project_path('data/models/movie_embeddings.npy')
)

if len(overview_embeddings) != len(movies):
    raise ValueError(
        "Movie catalog and embeddings do not match. "
        f"Movies: {len(movies)}, "
        f"Embeddings: {len(overview_embeddings)}"
    )

print(
    "Embeddings loaded:",
    overview_embeddings.shape
)

print("Precomputed semantic embeddings ready.")


# --------------------------------------------------
# 5. Helper functions
# --------------------------------------------------

def split_input(text):
    return [
        item.strip()
        for item in text.split(",")
        if item.strip()
    ]


def normalize(text):
    return text.strip().lower()


# --------------------------------------------------
# 6. Build known directors
# --------------------------------------------------

known_directors = sorted(
    {
        director.strip()
        for director in movies["director"]
        if director.strip()
    }
)


# --------------------------------------------------
# 7. Build known actors
# --------------------------------------------------

known_actors = set()

for actor_text in movies["actors"]:

    for actor in actor_text.split(","):

        actor = actor.strip()

        if actor:
            known_actors.add(actor)

known_actors = sorted(known_actors)


# --------------------------------------------------
# 8. Build known genres
# --------------------------------------------------

known_genres = set()

for genre_text in movies["genres"]:

    for genre in genre_text.split(","):

        genre = genre.strip()

        if genre:
            known_genres.add(genre)

known_genres = sorted(known_genres)


# --------------------------------------------------
# 9. Intelligent matching
# --------------------------------------------------

def find_best_match(user_input, choices, label):

    user_input = user_input.strip()

    if not user_input:
        return None


    # Exact case-insensitive match
    for choice in choices:

        if normalize(choice) == normalize(user_input):

            if choice != user_input:
                print(
                    f'{label}: "{user_input}" '
                    f'matched to "{choice}"'
                )

            return choice


    # Partial match
    partial_matches = [
        choice
        for choice in choices
        if normalize(user_input)
        in normalize(choice)
    ]


    if len(partial_matches) == 1:

        match = partial_matches[0]

        print(
            f'{label}: "{user_input}" '
            f'matched to "{match}"'
        )

        return match


    # Fuzzy spelling match
    fuzzy_matches = difflib.get_close_matches(
        user_input,
        choices,
        n=3,
        cutoff=0.6
    )


    if fuzzy_matches:

        best_match = fuzzy_matches[0]

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


# --------------------------------------------------
# 10. Ask user for preferences
# --------------------------------------------------

print("\nTell me about your movie preferences.\n")


favorite_movies_input = input(
    "Favorite movies (separate with commas): "
)

favorite_genres_input = input(
    "Favorite genres (separate with commas): "
)

favorite_actors_input = input(
    "Favorite actors (separate with commas): "
)

favorite_directors_input = input(
    "Favorite directors (separate with commas): "
)


favorite_movies_raw = split_input(
    favorite_movies_input
)

favorite_genres_raw = split_input(
    favorite_genres_input
)

favorite_actors_raw = split_input(
    favorite_actors_input
)

favorite_directors_raw = split_input(
    favorite_directors_input
)


# --------------------------------------------------
# 11. Validate favorite movies
# --------------------------------------------------

favorite_movies = []

movie_titles = movies["title"].tolist()


for movie_name in favorite_movies_raw:

    match = find_best_match(
        movie_name,
        movie_titles,
        "Movie"
    )

    if match:
        favorite_movies.append(match)


# --------------------------------------------------
# 12. Validate genres
# --------------------------------------------------

favorite_genres = []


for genre in favorite_genres_raw:

    match = find_best_match(
        genre,
        known_genres,
        "Genre"
    )

    if match:
        favorite_genres.append(match)


# --------------------------------------------------
# 13. Validate actors
# --------------------------------------------------

favorite_actors = []


for actor in favorite_actors_raw:

    match = find_best_match(
        actor,
        known_actors,
        "Actor"
    )

    if match:
        favorite_actors.append(match)


# --------------------------------------------------
# 14. Validate directors
# --------------------------------------------------

favorite_directors = []


for director in favorite_directors_raw:

    match = find_best_match(
        director,
        known_directors,
        "Director"
    )

    if match:
        favorite_directors.append(match)


# --------------------------------------------------
# 15. Create score columns
# --------------------------------------------------

movies["story_raw"] = 0.0
movies["story_score"] = 0.0

movies["genre_score"] = 0.0
movies["actor_score"] = 0.0
movies["director_score"] = 0.0


# --------------------------------------------------
# 16. Find favorite movie indices
# --------------------------------------------------

favorite_movie_indices = []


for movie_name in favorite_movies:

    matches = movies[
        movies["title"].str.lower()
        == movie_name.lower()
    ]

    if not matches.empty:

        favorite_movie_indices.append(
            matches.index[0]
        )


# --------------------------------------------------
# 17. Semantic story similarity
# --------------------------------------------------

if favorite_movie_indices:

    favorite_embeddings = overview_embeddings[
        favorite_movie_indices
    ]


    # Because embeddings are normalized,
    # dot product works as cosine similarity.
    similarity_matrix = np.dot(
        favorite_embeddings,
        overview_embeddings.T
    )


    # Overall similarity to all favorites
    average_similarity = (
        similarity_matrix.mean(axis=0)
    )


    # Strongest similarity to any one favorite
    best_similarity = (
        similarity_matrix.max(axis=0)
    )


    # Combine both
    movies["story_raw"] = (
        average_similarity * 0.70
        + best_similarity * 0.30
    )


# --------------------------------------------------
# 18. Genres inferred from favorite movies
# --------------------------------------------------

inferred_genres = set()


for movie_name in favorite_movies:

    matched_movie = movies[
        movies["title"].str.lower()
        == movie_name.lower()
    ]

    if not matched_movie.empty:

        genre_text = (
            matched_movie.iloc[0]["genres"]
        )

        genres = [
            genre.strip().lower()
            for genre in genre_text.split(",")
            if genre.strip()
        ]

        inferred_genres.update(
            genres
        )


# --------------------------------------------------
# 19. Explicit genres selected by user
# --------------------------------------------------

explicit_genres = {
    genre.lower()
    for genre in favorite_genres
}


# --------------------------------------------------
# 20. Genre scoring
# --------------------------------------------------

def calculate_genre_score(movie_genres):

    movie_genre_set = {
        genre.strip().lower()
        for genre in movie_genres.split(",")
        if genre.strip()
    }


    # ----------------------------------------------
    # Explicit preferences
    # ----------------------------------------------

    explicit_score = 0.0

    if explicit_genres:

        explicit_matches = (
            movie_genre_set
            .intersection(explicit_genres)
        )

        explicit_score = (
            len(explicit_matches)
            / len(explicit_genres)
        )


    # ----------------------------------------------
    # Genres inferred from favorite movies
    # ----------------------------------------------

    inferred_score = 0.0

    if inferred_genres:

        inferred_matches = (
            movie_genre_set
            .intersection(inferred_genres)
        )

        inferred_score = (
            len(inferred_matches)
            / len(inferred_genres)
        )


    # User-entered genres are more important
    if explicit_genres and inferred_genres:

        return (
            explicit_score * 0.70
            + inferred_score * 0.30
        )


    elif explicit_genres:

        return explicit_score


    elif inferred_genres:

        return inferred_score


    return 0.0


movies["genre_score"] = movies[
    "genres"
].apply(
    calculate_genre_score
)


# --------------------------------------------------
# 21. Actor scoring
# --------------------------------------------------

for actor in favorite_actors:

    actor_matches = (
        movies["actors"]
        .str.lower()
        .str.contains(
            actor.lower(),
            regex=False
        )
    )

    movies.loc[
        actor_matches,
        "actor_score"
    ] += 1


if favorite_actors:

    movies["actor_score"] /= len(
        favorite_actors
    )


# --------------------------------------------------
# 22. Director scoring
# --------------------------------------------------

for director in favorite_directors:

    director_matches = (
        movies["director"].str.lower()
        == director.lower()
    )

    movies.loc[
        director_matches,
        "director_score"
    ] += 1


if favorite_directors:

    movies["director_score"] /= len(
        favorite_directors
    )


# --------------------------------------------------
# 23. Identify candidate movies
# --------------------------------------------------

favorite_movies_lower = [
    movie.lower()
    for movie in favorite_movies
]


candidate_mask = (
    ~movies["title"]
    .str.lower()
    .isin(favorite_movies_lower)
)


# --------------------------------------------------
# 24. Robust story normalization
# --------------------------------------------------

# Previously we divided by the single strongest
# movie.
#
# Now we use the 95th percentile.
#
# This means one extreme movie cannot determine
# the scale for every other movie.


# --------------------------------------------------
# 24. Story normalization
# --------------------------------------------------

if favorite_movies:

    candidate_story_scores = (
        movies.loc[
            candidate_mask,
            "story_raw"
        ]
        .clip(lower=0)
    )

    max_story_score = candidate_story_scores.max()

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


# --------------------------------------------------
# 25. Base weights
# --------------------------------------------------

base_weights = {

    "story_score": 0.40,

    "genre_score": 0.25,

    "director_score": 0.20,

    "actor_score": 0.15
}


# --------------------------------------------------
# 26. Determine active preferences
# --------------------------------------------------

active_scores = []


if favorite_movies:

    active_scores.append(
        "story_score"
    )


if explicit_genres or inferred_genres:

    active_scores.append(
        "genre_score"
    )


if favorite_directors:

    active_scores.append(
        "director_score"
    )


if favorite_actors:

    active_scores.append(
        "actor_score"
    )


active_weight_total = sum(
    base_weights[score]
    for score in active_scores
)


# --------------------------------------------------
# 27. Weighted base score
# --------------------------------------------------

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


# --------------------------------------------------
# 28. Meaningful match thresholds
# --------------------------------------------------

#
# STORY
#
# Use the original semantic similarity here.
#
# We do NOT use normalized story_score for
# deciding whether something genuinely matches.
#

STORY_MATCH_MINIMUM = 0.25


if favorite_movies:

    candidate_raw_story = movies.loc[
        candidate_mask,
        "story_raw"
    ]


    # Also calculate a distribution-based
    # threshold.
    #
    # A story has to be at least in roughly
    # the upper part of the available matches.
    distribution_story_threshold = (
        np.percentile(
            candidate_raw_story,
            75
        )
    )


    STORY_MATCH_THRESHOLD = max(
        STORY_MATCH_MINIMUM,
        distribution_story_threshold
    )


else:

    STORY_MATCH_THRESHOLD = (
        STORY_MATCH_MINIMUM
    )


#
# GENRE
#
# A tiny overlap like 0.04 or 0.08 no longer
# counts as a complete preference match.
#

GENRE_MATCH_THRESHOLD = 0.25


# --------------------------------------------------
# 29. Count meaningful preference matches
# --------------------------------------------------

movies["match_count"] = 0


# Story
if favorite_movies:

    movies.loc[
        movies["story_raw"]
        >= STORY_MATCH_THRESHOLD,
        "match_count"
    ] += 1


# Genre
if explicit_genres or inferred_genres:

    movies.loc[
        movies["genre_score"]
        >= GENRE_MATCH_THRESHOLD,
        "match_count"
    ] += 1


# Director
if favorite_directors:

    movies.loc[
        movies["director_score"] > 0,
        "match_count"
    ] += 1


# Actor
if favorite_actors:

    movies.loc[
        movies["actor_score"] > 0,
        "match_count"
    ] += 1


# --------------------------------------------------
# 30. Multi-match bonus
# --------------------------------------------------

number_of_active_categories = len(
    active_scores
)


movies["multi_match_bonus"] = 0.0


if number_of_active_categories > 1:

    movies["multi_match_bonus"] = (

        (
            movies["match_count"] - 1
        )
        .clip(lower=0)

        /

        (
            number_of_active_categories - 1
        )

    ) * 0.10


# --------------------------------------------------
# 31. Final recommendation score
# --------------------------------------------------

movies["final_score"] = (
    movies["base_score"]
    + movies["multi_match_bonus"]
)


# --------------------------------------------------
# 32. Display scoring setup
# --------------------------------------------------

print("\nWeights being used:\n")


if active_weight_total > 0:

    for score in active_scores:

        adjusted_weight = (
            base_weights[score]
            / active_weight_total
        )

        print(
            f"{score}: "
            f"{adjusted_weight:.1%}"
        )


print(
    "\nStory score = normalized semantic "
    "similarity of movie descriptions"
)


if favorite_movies:

    print(
        "Story normalization reference "
        f"(maximum candidate similarity): "
        f"{max_story_score:.3f}"
    )


    print(
        "Meaningful story match threshold:",
        round(
            STORY_MATCH_THRESHOLD,
            3
        )
    )


if explicit_genres or inferred_genres:

    print(
        "Meaningful genre match threshold:",
        GENRE_MATCH_THRESHOLD
    )


print(
    "Explicit genres = 70% of genre score"
)

print(
    "Favorite-movie genres = 30% "
    "of genre score"
)

print(
    "Multi-match bonus: "
    "up to +0.100"
)


# --------------------------------------------------
# 33. Remove movies already listed as favorites
# --------------------------------------------------

recommendation_pool = movies[
    candidate_mask
].copy()


# --------------------------------------------------
# 34. Sort recommendations
# --------------------------------------------------

recommendations = (
    recommendation_pool
    .sort_values(
        by="final_score",
        ascending=False
    )
    .head(10)
)


# --------------------------------------------------
# 35. Display recommendations
# --------------------------------------------------

print("\nRecommended movies:\n")


for number, (_, movie) in enumerate(
    recommendations.iterrows(),
    start=1
):

    print(
        f"{number}. {movie['title']}"
        f" | final: {movie['final_score']:.3f}"
        f" | base: {movie['base_score']:.3f}"
        f" | bonus: {movie['multi_match_bonus']:.3f}"
        f" | matches: {int(movie['match_count'])}"
        f" | story: {movie['story_score']:.3f}"
        f" | raw_story: {movie['story_raw']:.3f}"
        f" | genre: {movie['genre_score']:.3f}"
        f" | director: {movie['director_score']:.3f}"
        f" | actor: {movie['actor_score']:.3f}"
    )