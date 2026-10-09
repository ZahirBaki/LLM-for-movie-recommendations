from backend.core.paths import project_path
import pandas as pd


# ==========================================================
# FILES
# ==========================================================

TMDB_FILE = project_path('data/processed/tmdb_movies.csv')
MOVIELENS_MOVIES_FILE = project_path('data/raw/ml-32m/movies.csv')
MOVIELENS_LINKS_FILE = project_path('data/raw/ml-32m/links.csv')

OUTPUT_FILE = project_path('data/processed/movie_mapping.csv')


# ==========================================================
# 1. LOAD DATA
# ==========================================================

print("Loading data...")

tmdb_movies = pd.read_csv(TMDB_FILE)
ml_movies = pd.read_csv(MOVIELENS_MOVIES_FILE)
links = pd.read_csv(MOVIELENS_LINKS_FILE)


print("\nTMDB movies:", len(tmdb_movies))
print("MovieLens movies:", len(ml_movies))
print("MovieLens links:", len(links))


# ==========================================================
# 2. PREPARE TMDB IDs
# ==========================================================

# links.csv can contain missing TMDB IDs.
links = links.dropna(subset=["tmdbId"]).copy()

links["tmdbId"] = links["tmdbId"].astype(int)

tmdb_movies["tmdb_id"] = tmdb_movies["tmdb_id"].astype(int)


print("\nMovieLens movies with a TMDB ID:", len(links))


# ==========================================================
# 3. ADD MOVIELENS TITLES
# ==========================================================

mapping = links.merge(
    ml_movies[["movieId", "title"]],
    on="movieId",
    how="left"
)


# ==========================================================
# 4. CONNECT TO OUR TMDB CATALOG
# ==========================================================

mapping = mapping.merge(
    tmdb_movies[
        [
            "tmdb_id",
            "title",
            "release_year"
        ]
    ],
    left_on="tmdbId",
    right_on="tmdb_id",
    how="left",
    suffixes=("_movielens", "_tmdb")
)


# ==========================================================
# 5. CHECK MATCHES
# ==========================================================

mapping["found_in_tmdb_catalog"] = (
    mapping["tmdb_id"].notna()
)


matched = mapping[
    mapping["found_in_tmdb_catalog"]
]

not_matched = mapping[
    ~mapping["found_in_tmdb_catalog"]
]


print("\n" + "=" * 60)
print("MAPPING RESULTS")
print("=" * 60)

print("MovieLens movies with TMDB ID:", len(mapping))
print("Found in our TMDB catalog:", len(matched))
print("Not found in our TMDB catalog:", len(not_matched))

match_percentage = (
    len(matched) / len(mapping)
) * 100

print(
    f"Match percentage: {match_percentage:.2f}%"
)


# ==========================================================
# 6. SHOW EXAMPLES
# ==========================================================

print("\n" + "=" * 60)
print("EXAMPLE MATCHES")
print("=" * 60)

print(
    matched[
        [
            "movieId",
            "tmdbId",
            "title_movielens",
            "title_tmdb",
            "release_year"
        ]
    ].head(20).to_string(index=False)
)


# ==========================================================
# 7. SHOW UNMATCHED EXAMPLES
# ==========================================================

print("\n" + "=" * 60)
print("EXAMPLE MOVIES NOT IN OUR TMDB CATALOG")
print("=" * 60)

print(
    not_matched[
        [
            "movieId",
            "tmdbId",
            "title_movielens"
        ]
    ].head(20).to_string(index=False)
)


# ==========================================================
# 8. SAVE THE MAPPING
# ==========================================================

mapping.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nMapping saved to: {OUTPUT_FILE}"
)

print("\nDone.")