"""Process-local, bounded overview queries compatible with build_embeddings.py."""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import threading
import time
import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
MAX_EXTERNAL_EMBEDDINGS = 128
EXTERNAL_EMBEDDING_TTL = 3600


class ExternalEmbeddingError(ValueError):
    pass


@dataclass(frozen=True)
class ExternalMovieQuery:
    tmdb_id: int
    title: str
    year: int | None
    genres: str
    actors: str
    director: str
    embedding: np.ndarray
    metadata: dict


def validate_embedding(vector, dimension):
    vector = np.asarray(vector, dtype=np.float32)
    if dimension != EMBEDDING_DIMENSION or vector.shape != (dimension,):
        raise ExternalEmbeddingError("The runtime embedding dimension does not match the catalog.")
    if not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1, atol=1e-4):
        raise ExternalEmbeddingError("The runtime embedding must be finite and unit normalized.")
    return vector


def load_model():
    # Importing the API or answering local queries never loads torch/model weights.
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME, device="cpu", local_files_only=True)


class RuntimeMovieEmbeddings:
    def __init__(self, dimension=384, *, model_loader=load_model, compatibility_samples=(),
                 max_entries=MAX_EXTERNAL_EMBEDDINGS, ttl=EXTERNAL_EMBEDDING_TTL, clock=time.monotonic):
        if max_entries < 1 or ttl <= 0:
            raise ValueError("Cache bounds must be positive")
        self.dimension = dimension
        self.model_loader = model_loader
        self.compatibility_samples = compatibility_samples
        self.max_entries, self.ttl, self.clock = max_entries, ttl, clock
        self._model = None
        self._retry_at = 0
        self._cache = OrderedDict()
        self._lock = threading.RLock()

    def _expire(self):
        for key, (created, fingerprint, query) in list(self._cache.items()):
            if self.clock() - created >= self.ttl:
                del self._cache[key]

    def cached(self, tmdb_id):
        with self._lock:
            self._expire()
            if tmdb_id not in self._cache:
                return None
            self._cache.move_to_end(tmdb_id)
            return self._cache[tmdb_id][2]

    def find(self, title, year):
        with self._lock:
            self._expire()
            matches = [query for _, _, query in self._cache.values()
                if title.casefold() in {query.title.casefold(), str(query.metadata.get("original_title") or "").casefold()} and query.year == year]
            return self.cached(matches[0].tmdb_id) if len(matches) == 1 else None

    def _encode(self, model, overview):
        # Identical raw overview; no title, metadata prefixes or text normalization.
        raw = model.encode([overview], convert_to_numpy=True, normalize_embeddings=True,
                           show_progress_bar=False)
        if np.asarray(raw).shape != (1, self.dimension):
            raise ExternalEmbeddingError("The runtime embedding dimension does not match the catalog.")
        return validate_embedding(raw[0], self.dimension)

    def get(self, movie):
        metadata = movie.model_dump()
        overview = metadata.get("overview")
        if not isinstance(overview, str) or not overview.strip():
            raise ExternalEmbeddingError("This verified movie has no usable TMDB overview to compare with catalog movies.")
        if type(metadata.get("tmdb_id")) is not int or metadata["tmdb_id"] <= 0 or not metadata.get("title") or not metadata.get("release_year"):
            raise ExternalEmbeddingError("The external movie is missing verified identity metadata.")
        fingerprint = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        with self._lock:
            self._expire()
            cached = self._cache.get(movie.tmdb_id)
            if cached and cached[1] == fingerprint:
                self._cache.move_to_end(movie.tmdb_id)
                return cached[2]
            if self._model is None:
                if self.clock() < self._retry_at:
                    raise ExternalEmbeddingError("The compatible local embedding model is temporarily unavailable.")
                try:
                    model = self.model_loader()
                    for overview_sample, stored in self.compatibility_samples:
                        vector = self._encode(model, overview_sample)
                        if not np.allclose(vector, stored, atol=1e-4, rtol=1e-3):
                            raise ExternalEmbeddingError("The runtime model does not reproduce the stored catalog embeddings.")
                    self._model = model
                except Exception as exc:
                    self._retry_at = self.clock() + 60
                    if isinstance(exc, ExternalEmbeddingError):
                        raise
                    raise ExternalEmbeddingError("The compatible local embedding model is unavailable. Install the runtime dependencies and provision the all-MiniLM-L6-v2 model.") from None
            try:
                vector = self._encode(self._model, overview).copy()
            except ExternalEmbeddingError:
                raise
            except Exception:
                raise ExternalEmbeddingError("The external movie overview could not be encoded; no recommendations were generated.") from None
            vector.setflags(write=False)
            query = ExternalMovieQuery(movie.tmdb_id, movie.title, movie.release_year,
                movie.genres or "", movie.actors or "", movie.director or "", vector, metadata.copy())
            self._cache[movie.tmdb_id] = (self.clock(), fingerprint, query)
            self._cache.move_to_end(movie.tmdb_id)
            while len(self._cache) > self.max_entries:
                self._cache.popitem(last=False)
            return query
