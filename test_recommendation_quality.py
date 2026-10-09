"""Offline deterministic diagnostics; synthetic relevance labels never enter reports."""
import copy
import json
import math
from pathlib import Path
from contextlib import contextmanager
from uuid import uuid4
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import test_external_movie_embeddings as fixtures
from recommendation_quality import (QualityDiagnostics, artifact_hashes, distribution,
    jaccard, objective_metrics, ranking_metrics, production_content_weights, evaluate)
from recommendation_quality_report import write_report


@contextmanager
def fixture_directory():
    # Python 3.14 mkdtemp's restrictive Windows ACL is incompatible with this sandbox.
    path = Path(__file__).parent / (".quality-test-"+uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        for file in path.iterdir():
            file.unlink()
        path.rmdir()


class DiagnosticTests(unittest.TestCase):
    setUp = fixtures.ExternalTests.setUp

    def diagnostic(self):
        return QualityDiagnostics(self.engine,[self.details],self.cache)

    def test_deterministic_local_diagnostics_and_actual_engine(self):
        diagnostic = self.diagnostic()
        with patch.object(self.engine,"recommend",wraps=self.engine.recommend) as engine:
            first = diagnostic.diagnose([1],5)
        second = diagnostic.diagnose([1],5)
        self.assertEqual(first,second)
        engine.assert_called_once()
        self.assertEqual(len(first["recommendations"]),5)
        self.assertEqual(first["recommendations"][0]["tmdb_id"],2)

    def test_content_breakdown_exactly_reconstructs_score(self):
        result = self.diagnostic().diagnose([1],5)
        for row in result["recommendations"]:
            reconstructed = sum(row["weighted_content_components"].values())+row["multi_match_bonus"]
            self.assertAlmostEqual(reconstructed,row["content_score"],places=6)
            self.assertEqual(row["ranking_score"],row["content_score"])

    def test_unavailable_svd_and_final_scores_remain_null(self):
        result = self.diagnostic().diagnose([900001],5)
        for row in result["recommendations"]:
            self.assertIsNone(row["collaborative_score"])
            self.assertIsNone(row["engine_final_score"])
            self.assertEqual(row["ranking_kind"],"content_only")
        self.assertEqual(result["heldout_ranking_metrics"]["status"],"unavailable")

    def test_collaborative_and_soft_rerank_scores_are_observed(self):
        self.engine.svd.mapping = pd.DataFrame([dict(tmdb_id=i,movieId=i) for i in range(1,16)])
        self.engine.svd.movie_to_index = {i:i-1 for i in range(1,16)}
        self.engine.svd.get_scores_for_favorite_tmdb_movies.return_value = pd.DataFrame([
            dict(tmdb_id=i,title=f"Catalog {i}",svd_score=1-(i-3)/20) for i in range(3,16)])
        diagnostic = self.diagnostic()
        baseline = diagnostic.diagnose([1,2],5)
        expanded = diagnostic.diagnose([1,2],5,experiments=True)
        self.assertEqual(baseline["recommendations"],expanded["recommendations"])
        for row in baseline["recommendations"]:
            self.assertEqual(row["ranking_score"],row["collaborative_score"])
            self.assertEqual(row["ranking_kind"],"collaborative_primary")
        soft = diagnostic.diagnose([1,2],5,filters={"favorite_genres":["Drama"]})
        for row in soft["recommendations"]:
            self.assertAlmostEqual(row["engine_final_score"],row["collaborative_score"]+.1*row["content_score"])

    def test_empty_filtered_results_have_no_invented_metrics(self):
        result = self.diagnostic().diagnose([1],5,filters={"min_year":2030},experiments=True)
        self.assertEqual(result["recommendations"],[])
        self.assertEqual(result["experiments"],{})
        self.assertIsNone(result["metrics"]["genre_consistency_top5"])
        self.assertIsNone(result["metrics"]["popularity"])

    def test_external_and_mixed_seeds_and_raw_similarity(self):
        result = self.diagnostic().diagnose([1,900001],5)
        self.assertEqual([movie["tmdb_id"] for movie in result["seeds"]],[1,900001])
        row = result["recommendations"][0]
        position = self.catalog.index[self.catalog.tmdb_id.eq(row["tmdb_id"])][0]
        expected = np.stack([self.engine.content.overview_embeddings[0],fixtures.vector()]) @ self.engine.content.overview_embeddings[position]
        self.assertAlmostEqual(row["overview_cosine_mean"],float(expected.mean()),places=6)
        self.assertAlmostEqual(row["story_raw"],.7*float(expected.mean())+.3*float(expected.max()),places=6)
        self.assertEqual(len(row["per_seed_overview_similarity"]),2)

    def test_seed_seen_and_explicit_exclusion_correctness(self):
        result = self.diagnostic().diagnose([1,900001],5,excluded=[2,3])
        self.assertTrue(result["metrics"]["seed_exclusion_correct"])
        self.assertTrue(result["metrics"]["explicit_exclusion_correct"])
        self.assertFalse({1,900001,2,3}&{row["tmdb_id"] for row in result["recommendations"]})

    def test_unknown_seed_never_calls_engine_or_fabricates_snapshot(self):
        with patch.object(self.engine,"recommend",wraps=self.engine.recommend) as engine:
            with self.assertRaisesRegex(ValueError,"unavailable"):
                self.diagnostic().diagnose([1,999999],5)
        engine.assert_not_called()

    def test_missing_external_overview_blocks_all_seeds(self):
        diagnostic = QualityDiagnostics(self.engine,[self.details.model_copy(update={"overview":None})],self.cache)
        with self.assertRaisesRegex(ValueError,"no usable"):
            diagnostic.diagnose([1,900001])

    def test_experiments_are_reproducible_and_do_not_change_production(self):
        diagnostic = self.diagnostic()
        catalog = self.engine.content.movies.copy(deep=True)
        matrix = self.engine.content.overview_embeddings.copy()
        first = diagnostic.diagnose([1,900001],5,experiments=True)
        second = diagnostic.diagnose([1,900001],5,experiments=True)
        self.assertEqual(first,second)
        pd.testing.assert_frame_equal(catalog,self.engine.content.movies)
        np.testing.assert_array_equal(matrix,self.engine.content.overview_embeddings)
        baseline = diagnostic.diagnose([1,900001],5)
        self.assertEqual(first["recommendations"],baseline["recommendations"])
        self.assertEqual(set(first["experiments"]),{"genre_aware","popularity_confidence","vote_reliability","genre_votes","audience_proxy_filter","mean_story","strongest_story"})

    def test_all_experiments_preserve_filters_and_exclusions(self):
        result = self.diagnostic().diagnose([1],3,filters={"min_year":2020,"required_genres":["Drama"]},excluded=[10],experiments=True)
        for variant in result["experiments"].values():
            self.assertTrue(all(row["year"]>=2020 and row["tmdb_id"]!=10 for row in variant["recommendations"]))

    def test_invalid_options_are_rejected(self):
        for ids,filters,count in [([],{ },5),([0],{},5),([1],{"new_weight":.2},5),([1],{},0)]:
            with self.assertRaises(ValueError):
                self.diagnostic().diagnose(ids,count,filters)

    def test_missing_metadata_is_not_invented(self):
        row = self.diagnostic().diagnose([1],1)["recommendations"][0]
        self.assertIsNone(row["popularity"])
        self.assertIsNone(row["vote_average"])
        self.assertIsNone(row["vote_count"])

    def test_unavailable_scenario_is_explicit(self):
        scenario = [("Z","Unavailable",[(99999,"Absent",2026)],["Drama"])]
        with patch("recommendation_quality.SCENARIOS",scenario):
            result = evaluate(self.diagnostic())
        self.assertEqual(result["scenarios"][0]["status"],"unavailable")
        self.assertIn("no verified external snapshot",result["scenarios"][0]["reason"])

    def test_report_rendering_and_strict_json(self):
        scenario = [("Z","Fixture",[(1,"Catalog 1",2011)],["Drama"])]
        with patch("recommendation_quality.SCENARIOS",scenario):
            result = evaluate(self.diagnostic())
        json.dumps(result,allow_nan=False)
        with fixture_directory() as directory:
            path = Path(directory)/"report.md"
            write_report(result,path)
            text = path.read_text(encoding="utf-8")
        self.assertIn("Offline relevance metrics",text)
        self.assertIn("not deployed",text)
        self.assertIn("Catalog 2",text)


class MetricTests(unittest.TestCase):
    def test_genre_overlap_and_distribution(self):
        self.assertEqual(jaccard({"drama","comedy"},{"drama","crime"}),1/3)
        self.assertIsNone(jaccard(set(),set()))
        self.assertEqual(distribution([0,None,10])["median"],5)
        self.assertIsNone(distribution([None]))

    def test_metrics_duplicate_exclusion_and_diversity(self):
        rows=[dict(tmdb_id=1,genres="Drama",popularity=2,vote_count=50),
              dict(tmdb_id=1,genres="Comedy",popularity=None,vote_count=None)]
        result=objective_metrics(rows,[1],{"drama"},{"drama"},[1])
        self.assertEqual(result["duplicate_rate"],.5)
        self.assertEqual(result["mean_genre_jaccard"],.5)
        self.assertEqual(result["genre_diversity"],1)
        self.assertFalse(result["seed_exclusion_correct"])
        self.assertFalse(result["explicit_exclusion_correct"])

    def test_recall_ndcg_from_synthetic_ground_truth(self):
        result=ranking_metrics([1,2,3],[2,3],3)
        self.assertEqual(result["recall"],1)
        expected=(1/math.log2(3)+1/math.log2(4))/(1+1/math.log2(3))
        self.assertAlmostEqual(result["ndcg"],expected)
        self.assertEqual(ranking_metrics([1],[]),{"recall":None,"ndcg":None})

    def test_duplicate_relevance_hits_keep_original_rank_slots(self):
        result = ranking_metrics([2,2,3],[2,3],2)
        self.assertEqual(result["recall"],.5)
        self.assertAlmostEqual(result["ndcg"],1/(1+1/math.log2(3)))

    def test_artifact_hashes_and_extracted_weights(self):
        with fixture_directory() as directory:
            path=Path(directory)/"fixture"
            path.write_text("abc")
            first=artifact_hashes([str(path)])
            self.assertEqual(first,artifact_hashes([str(path)]))
            path.write_text("changed")
            self.assertNotEqual(first,artifact_hashes([str(path)]))
        self.assertEqual(production_content_weights(),{"story_score":.4,"genre_score":.25,"director_score":.2,"actor_score":.15})
