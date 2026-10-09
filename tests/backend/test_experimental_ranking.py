"""No network/paid calls: synthetic engine and existing conversation fixtures."""
import contextlib
import io
import os
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from backend.recommendation.experimental_ranking import enabled, rerank, ranking_context, excluded_context
from tests.backend.test_external_movie_embeddings import ExternalTests
from tests.backend.test_movie_assistant import call, response
from evaluation.scripts.recommendation_quality import QualityDiagnostics

class RankingTests(ExternalTests):
    def setUp(self):
        super().setUp()
        patch.dict(os.environ, {"ENVIRONMENT":"development", "APP_ENV":"development", "NODE_ENV":"development", "MOVIE_EXPERIMENTAL_RANKING":"true"}).start()

    def test_default_and_disabled_baseline_identical(self):
        with contextlib.redirect_stdout(io.StringIO()):
            baseline = self.engine.recommend(favorite_movies=[{"tmdb_id":1}], top_n=5)
            with patch.dict(os.environ, {"MOVIE_EXPERIMENTAL_RANKING":"false"}):
                disabled = self.engine.recommend(favorite_movies=[{"tmdb_id":1}], top_n=5, ranking_mode="genre_votes")
        pd.testing.assert_frame_equal(baseline, disabled)
        self.assertEqual(disabled.attrs["ranking_diagnostics"]["fallback_reason"], "feature_disabled")
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(enabled())
        with patch.dict(os.environ, {"APP_ENV":"production"}):
            self.assertFalse(enabled())

    def test_local_external_mixed_exact_count_and_parity(self):
        diagnostic = QualityDiagnostics(self.engine, [self.details], self.cache)
        for ids in ([1], [900001], [1,900001]):
            with contextlib.redirect_stdout(io.StringIO()):
                inputs,*_ = diagnostic.prepare(ids,{})
                frame = self.engine.recommend(**inputs,top_n=5,ranking_mode="genre_votes")
                measured = diagnostic.diagnose(ids,5,experiments=True)
            self.assertEqual(len(frame),5)
            self.assertEqual(frame.tmdb_id.tolist(), [movie["tmdb_id"] for movie in measured["experiments"]["genre_votes"]["recommendations"]])
            self.assertEqual(frame.attrs["ranking_diagnostics"]["applied_mode"],"genre_votes")
            self.assertFalse(frame.tmdb_id.isin(ids).any())

    def test_empty_live_pool_reports_no_application(self):
        with contextlib.redirect_stdout(io.StringIO()):
            frame = self.engine.recommend(favorite_movies=[{"tmdb_id":1}], min_year=2099, ranking_mode="genre_votes")
        self.assertTrue(frame.empty)
        diagnostic = frame.attrs["ranking_diagnostics"]
        self.assertFalse(diagnostic["experimental_applied"])
        self.assertEqual(diagnostic["fallback_reason"], "empty_candidate_pool")

    def test_pool_determinism_missing_metadata_empty(self):
        frame = pd.DataFrame({"tmdb_id":range(1,251),"content_score":np.linspace(1,0,250)})
        catalog = frame.assign(genres=None, vote_count=np.nan, popularity=np.nan)
        first = rerank(frame,catalog,set())
        pd.testing.assert_frame_equal(first,rerank(frame,catalog,set()))
        self.assertEqual(len(first),200)
        self.assertLessEqual(first.tmdb_id.max(),200)
        self.assertTrue(rerank(frame.head(0),catalog,set()).empty)
        self.assertEqual(first.tmdb_id.tolist(),list(range(1,201)))

    def test_request_context_and_cards_align(self):
        token = ranking_context.set(("genre_votes","fixture-request"))
        try:
            result = self.recommend()
        finally:
            ranking_context.reset(token)
        self.assertEqual(len(result.recommendations),5)
        self.assertEqual(ranking_context.get(),("baseline",None))
        with contextlib.redirect_stdout(io.StringIO()):
            frame = self.engine.recommend(external_queries=[self.cache.get(self.details)], top_n=5, ranking_mode="genre_votes")
        self.assertEqual([movie["tmdb_id"] for movie in result.recommendations],frame.tmdb_id.tolist())

    def test_collaborative_fallback_unchanged(self):
        self.engine.svd.mapping = pd.DataFrame([dict(tmdb_id=i,movieId=i) for i in range(1,16)])
        self.engine.svd.movie_to_index = {i:i-1 for i in range(1,16)}
        self.engine.svd.get_scores_for_favorite_tmdb_movies.return_value = pd.DataFrame([dict(tmdb_id=i,title=f"Catalog {i}",svd_score=1-i/20) for i in range(3,16)])
        with contextlib.redirect_stdout(io.StringIO()):
            baseline = self.engine.recommend(favorite_movies=[{"tmdb_id":1},{"tmdb_id":2}],top_n=5)
            experimental = self.engine.recommend(favorite_movies=[{"tmdb_id":1},{"tmdb_id":2}],top_n=5,ranking_mode="genre_votes")
        pd.testing.assert_frame_equal(baseline,experimental)
        self.assertEqual(experimental.attrs["ranking_diagnostics"]["fallback_reason"],"collaborative_primary")

    async def test_endpoint_flag_and_context_cleanup(self):
        import backend.api.app as api
        from backend.services.movie_chat import MovieChatResponse
        async def responder(*args):
            result = self.recommend()
            self.assertEqual(ranking_context.get()[0], "genre_votes")
            return MovieChatResponse(status="success",country_code="SE",assistant_message="Movies", recommendations=result.recommendations)
        with patch.object(api,"respond_to_movie_chat",side_effect=responder), patch.dict(os.environ,{"APP_ENV":"production"}):
            result = await api.movie_chat(api.MovieChatRequest(message="Recommend movies",ranking_mode="genre_votes"))
        self.assertEqual(result.status,"success")
        self.assertEqual(ranking_context.get(),("baseline",None))

    def test_blocked_before_pool(self):
        token = excluded_context.set(tuple(range(2,12)))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                frame = self.engine.recommend(favorite_movies=[{"tmdb_id":1}],top_n=3,ranking_mode="genre_votes")
        finally:
            excluded_context.reset(token)
        self.assertEqual(len(frame),3)
        self.assertFalse(frame.tmdb_id.isin(range(1,12)).any())

    async def test_pending_external_resume_and_seen(self):
        token = ranking_context.set(("genre_votes", "pending-session"))
        try:
            with patch('backend.recommendation.recommendation_engine.rerank', wraps=rerank) as ranker:
                await super().test_pending_external_resume_and_seen()
            ranker.assert_called_once()
        finally:
            ranking_context.reset(token)

    async def test_switch_followup_preserves_memory(self):
        self.client.responses.create.return_value = response(call("recommend_movies", favorite_movies=["External Film (2026)"], top_n=5))
        token = ranking_context.set(("genre_votes","session"))
        try:
            first = await self.turn("I watched External Film (2026). Recommend 5 similar movies.")
            watched = [movie["tmdb_id"] for movie in first.recommendations[:2]]
            ranking_context.set(("baseline","session"))
            second = await self.turn("I have already seen the first two. Give me 5 more.")
            self.assertEqual(len(second.recommendations),5)
            self.assertFalse(set(watched) & {movie["tmdb_id"] for movie in second.recommendations})
            self.assertTrue(self.state.active_seed_movies)
        finally:
            ranking_context.reset(token)

if __name__ == "__main__":
    unittest.main()
