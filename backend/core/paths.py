"""Workspace-relative resources; independent of the invoking terminal directory."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_PATHS = {
    'data/tmdb_movies.csv': 'data/processed/tmdb_movies.csv',
    'data/movie_embeddings.npy': 'data/models/movie_embeddings.npy',
    'data/svd_movie_factors.npy': 'data/models/svd_movie_factors.npy',
    'data/svd_movie_index.csv': 'data/models/svd_movie_index.csv',
    'data/movie_mapping.csv': 'data/processed/movie_mapping.csv',
    'recommendation_engine.py': 'backend/recommendation/recommendation_engine.py',
    'content_recommender.py': 'backend/recommendation/content_recommender.py',
    'svd_recommender.py': 'backend/recommendation/svd_recommender.py',
    'runtime_movie_embeddings.py': 'backend/recommendation/runtime_movie_embeddings.py',
    'evaluation_data/external_movies.json': 'evaluation/fixtures/external_movies.json',
    'recommendation_quality.py': 'evaluation/scripts/recommendation_quality.py',
    'recommendation_quality_report.py': 'evaluation/scripts/recommendation_quality_report.py',
}

def project_path(value):
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / LEGACY_PATHS.get(path.as_posix(), path.as_posix())
