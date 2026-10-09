"""Offline comparison of the actual live ranking lanes; uses cached artifacts only."""
from backend.core.paths import project_path
import argparse
import contextlib
import io
import os
from unittest.mock import patch
from backend.recommendation.experimental_ranking import excluded_context
from evaluation.scripts.recommendation_quality import QualityDiagnostics, load_snapshots

def compare(diagnostic, ids, filters=None, excluded=()):
    with contextlib.redirect_stdout(io.StringIO()):
        inputs, *_ = diagnostic.prepare(ids, filters or {})
        token = excluded_context.set(tuple(excluded))
        try:
            with patch.dict(os.environ, {"ENVIRONMENT":"development", "APP_ENV":"development", "NODE_ENV":"development", "MOVIE_EXPERIMENTAL_RANKING":"true"}):
                frames = {mode: diagnostic.engine.recommend(**inputs, top_n=10+len(excluded), ranking_mode=mode) for mode in ("baseline", "genre_votes")}
        finally:
            excluded_context.reset(token)
    selected = {mode: frame[~frame.tmdb_id.isin(excluded)].head(10).reset_index(drop=True) for mode,frame in frames.items()}
    for mode, frame in selected.items():
        print("BASELINE TOP 10" if mode == "baseline" else "EXPERIMENTAL TOP 10")
        print("Applied:", frames[mode].attrs.get("ranking_diagnostics"))
        columns = [column for column in ("title", "release_year", "tmdb_id", "content_score", "svd_score", "final_score", "experiment_score") if column in frame]
        print(frame[columns].assign(rank=range(1,len(frame)+1)).to_string(index=False))
    positions = [{int(identity):rank for rank,identity in enumerate(frame.tmdb_id,1)} for frame in selected.values()]
    old,new = positions
    print("COMPARISON")
    print("Both:", sorted(old.keys() & new.keys()))
    print("Added:", sorted(new.keys() - old.keys()))
    print("Removed:", sorted(old.keys() - new.keys()))
    print("Positions (baseline -> experimental):", {identity:(old[identity],new[identity]) for identity in sorted(old.keys() & new.keys())})
    return selected

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--favorites", type=int, nargs="+", required=True)
    parser.add_argument("--exclude", type=int, nargs="*", default=[])
    parser.add_argument("--filters-json", default="{}")
    args = parser.parse_args()
    import json
    from backend.recommendation.recommendation_engine import RecommendationEngine
    compare(QualityDiagnostics(RecommendationEngine(), load_snapshots(project_path('evaluation/fixtures/external_movies.json'))), args.favorites, json.loads(args.filters_json), args.exclude)

if __name__ == "__main__":
    main()
