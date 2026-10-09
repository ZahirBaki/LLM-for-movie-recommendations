from backend.core.paths import project_path
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

CATALOG_FILE = project_path('data/processed/tmdb_movies.csv')
EMBEDDINGS_FILE = project_path('data/models/movie_embeddings.npy')

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 128


# --------------------------------------------------
# LOAD MOVIE CATALOG
# --------------------------------------------------

print("Loading movie catalog...")

movies = pd.read_csv(CATALOG_FILE)

print(f"Movies loaded: {len(movies):,}")


# --------------------------------------------------
# PREPARE OVERVIEWS
# --------------------------------------------------

# Make sure missing values cannot cause problems
movies["overview"] = movies["overview"].fillna("").astype(str)

overviews = movies["overview"].tolist()

print(f"Overviews prepared: {len(overviews):,}")


# --------------------------------------------------
# LOAD SEMANTIC MODEL
# --------------------------------------------------

print()
print("Loading semantic model...")

model = SentenceTransformer(MODEL_NAME)

print("Model loaded.")


# --------------------------------------------------
# CREATE EMBEDDINGS
# --------------------------------------------------

print()
print("Creating movie embeddings...")
print("This may take some time.")

embeddings = model.encode(
    overviews,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)


# --------------------------------------------------
# VERIFY
# --------------------------------------------------

print()
print("Embedding generation complete.")

print(f"Embedding shape: {embeddings.shape}")

expected_shape = (len(movies), 384)

if embeddings.shape != expected_shape:
    raise ValueError(
        f"Unexpected embedding shape. "
        f"Expected {expected_shape}, "
        f"but got {embeddings.shape}."
    )


# --------------------------------------------------
# SAVE
# --------------------------------------------------

os.makedirs(project_path('data'), exist_ok=True)

np.save(EMBEDDINGS_FILE, embeddings)

print()
print(f"Saved embeddings to: {EMBEDDINGS_FILE}")


# --------------------------------------------------
# FINAL CHECK
# --------------------------------------------------

loaded_embeddings = np.load(EMBEDDINGS_FILE)

print()
print("Final verification:")
print(f"Movies:     {len(movies):,}")
print(f"Embeddings: {len(loaded_embeddings):,}")
print(f"Dimensions: {loaded_embeddings.shape[1]}")

if len(movies) == len(loaded_embeddings):
    print()
    print("SUCCESS: Every movie has a matching embedding.")
else:
    raise ValueError(
        "ERROR: Number of movies and embeddings do not match."
    )