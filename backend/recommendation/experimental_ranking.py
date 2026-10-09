"""Shared offline/live experiment and request-local ranking configuration."""
from backend.core.paths import project_path
import os
from contextvars import ContextVar
import numpy as np
import pandas as pd

excluded_context = ContextVar("ranking_exclusions", default=())

ranking_context = ContextVar("ranking_context", default=("baseline", None))

def enabled():
    environments = [os.getenv(key, "").casefold() for key in ("ENVIRONMENT", "APP_ENV", "NODE_ENV")]
    return "production" not in environments and "development" in environments and os.getenv("MOVIE_EXPERIMENTAL_RANKING", "false").casefold() == "true"

def tags(value):
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return set()
    return {part.strip().casefold() for part in str(value).split(",") if part.strip()}

def genre_votes_scores(base, genre, votes, popularity):
    votes = np.maximum(np.nan_to_num(np.asarray(votes, dtype=float)), 0)
    popular = np.log1p(np.maximum(np.nan_to_num(np.asarray(popularity, dtype=float)), 0))
    popular = np.clip(popular / max(float(np.percentile(popular, 95)), 1e-12), 0, 1)
    reliability = votes / (votes + 500)
    return .65*base + .2*genre + .1*reliability + .05*popular

def rerank(frame, catalog, seed_genres):
    data = frame.head(200).copy().reset_index(drop=True)
    if data.empty:
        return data
    metadata = catalog.drop_duplicates("tmdb_id").set_index("tmdb_id")
    rows = metadata.reindex(data.tmdb_id)
    genre = np.array([len(tags(value) & seed_genres)/len(tags(value) | seed_genres) if tags(value) | seed_genres else 0 for value in rows.get("genres", pd.Series([None]*len(data)))])
    base = 1-np.arange(len(data))/max(len(data)-1,1)
    data["experiment_score"] = genre_votes_scores(base, genre, rows.get("vote_count", pd.Series([0]*len(data))).fillna(0), rows.get("popularity", pd.Series([0]*len(data))).fillna(0))
    return data.sort_values(["experiment_score", "tmdb_id"], ascending=[False, True], kind="stable")


def record_diagnostic(event, data):
    """Development-only structured local log; excludes prompts and credentials."""
    import json
    import logging
    from pathlib import Path
    environments = [os.getenv(key, "").casefold() for key in ("ENVIRONMENT", "APP_ENV", "NODE_ENV")]
    if "production" in environments or "development" not in environments or (not enabled() and data.get("requested_mode") != "genre_votes"):
        return
    logger = logging.getLogger("movie.ranking.local")
    if not logger.handlers:
        handler = logging.FileHandler(project_path("logs/ranking.jsonl"), encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    logger.info(json.dumps({"event": event, **data}, allow_nan=False))
