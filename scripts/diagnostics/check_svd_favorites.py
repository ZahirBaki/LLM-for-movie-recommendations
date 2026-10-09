from backend.core.paths import project_path
import pandas as pd


# ==========================================================
# FILES
# ==========================================================

MAPPING_FILE = project_path('data/processed/movie_mapping.csv')
SVD_INDEX_FILE = project_path('data/models/svd_movie_index.csv')
TMDB_FILE = project_path('data/processed/tmdb_movies.csv')


# ==========================================================
# MOVIES TO TEST
# ==========================================================

MOVIES_TO_TEST = [
    "Coco",
    "Inside Out",
    "Zootopia",
]


# ==========================================================
# LOAD DATA
# ==========================================================

mapping = pd.read_csv(MAPPING_FILE)
svd_index = pd.read_csv(SVD_INDEX_FILE)
tmdb = pd.read_csv(TMDB_FILE)


print("\n========================================")
print("SVD FAVORITE DIAGNOSTIC")
print("========================================\n")

print("Mapping columns:")
print(mapping.columns.tolist())

print("\nSVD index columns:")
print(svd_index.columns.tolist())

print("\n")


# ==========================================================
# TEST EACH MOVIE
# ==========================================================

for title in MOVIES_TO_TEST:

    print("=" * 60)
    print(f"MOVIE: {title}")
    print("=" * 60)

    # Find movie in TMDB catalog
    matches = tmdb[
        tmdb["title"]
        .astype(str)
        .str.lower()
        .eq(title.lower())
    ]

    if matches.empty:
        print("❌ Not found in TMDB catalog")
        print()
        continue

    movie = matches.iloc[0]

    tmdb_id = int(movie["tmdb_id"])

    print(f"✓ Found in TMDB")
    print(f"  TMDB ID: {tmdb_id}")
    print(f"  Year: {movie.get('release_year', 'Unknown')}")

    # ------------------------------------------------------
    # Find TMDB -> MovieLens mapping
    # ------------------------------------------------------

    mapping_match = mapping[
        pd.to_numeric(
            mapping["tmdb_id"],
            errors="coerce"
        ) == tmdb_id
    ]

    if mapping_match.empty:
        print("❌ No MovieLens mapping")
        print()
        continue

    print("✓ Found TMDB → MovieLens mapping")

    movie_id = int(
        mapping_match.iloc[0]["movieId"]
    )

    print(f"  MovieLens movieId: {movie_id}")

    # ------------------------------------------------------
    # Check SVD model
    # ------------------------------------------------------

    svd_match = svd_index[
        pd.to_numeric(
            svd_index["movieId"],
            errors="coerce"
        ) == movie_id
    ]

    if svd_match.empty:
        print("❌ NOT present in trained SVD model")
        print("  → Cannot be used as an SVD favorite")

    else:
        print("✓ Present in trained SVD model")
        print("  → CAN be used as an SVD favorite")

    print()


print("=" * 60)
print("DONE")
print("=" * 60)