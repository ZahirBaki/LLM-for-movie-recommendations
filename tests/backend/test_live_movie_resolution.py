"""Offline TMDB identity fallback and unsupported external-seed scoring checks."""
import contextlib
import io
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, AsyncMock

import pandas as pd
import numpy as np
import backend.api.app as api
from backend.services.chat_recommender import ChatRecommender
from backend.services.chat_service import MovieIntent
from backend.services.movie_assistant import MovieAssistant
from backend.services.movie_tools import MovieTools
from backend.sessions.movie_session import MovieSessionState, MovieReference, handle_session_turn
from tests.backend.test_movie_assistant import call, response
from tests.backend.test_chat_recommender import empty_providers


def external(identity=1248832, title="Digger", year=2026):
    return {"id": identity, "title": title, "original_title": title,
        "release_date": f"{year}-09-28", "overview": "A powerful man's mission to avert a disaster.",
        "genres": [{"name": "Comedy"}], "runtime": 130, "poster_path": "/fixture.jpg",
        "credits": {"crew": [{"job": "Director", "name": "Alejandro G. Iñárritu"}],
                    "cast": [{"name": "Tom Cruise"}]}}


class LiveMovieTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.catalog = pd.DataFrame([
            {"tmdb_id": 7, "title": "Local Film", "release_year": 2010},
            {"tmdb_id": 656272, "title": "Digger", "release_year": 2021},
            {"tmdb_id": 8, "title": "Duplicate", "release_year": 2000},
            {"tmdb_id": 9, "title": "Duplicate", "release_year": 2020}])
        self.ranking = Mock(return_value=[])
        self.controller = ChatRecommender(self.catalog, self.ranking)
        self.live = Mock(side_effect=self.request)
        self.providers = Mock(side_effect=empty_providers)
        self.tools = MovieTools(self.catalog, self.controller, lambda query: [], self.providers,
                               live_movie_runner=self.live)
        self.client = NS(responses=NS(create=AsyncMock()))

    def request(self, path, params):
        if path == "/search/movie":
            if params["query"].casefold() == "digger":
                values = [external()]
                if "year" not in params:
                    values.append(external(656272, year=2021))
                return {"results": values, "total_pages": 1}
            return {"results": [], "total_pages": 1}
        if path == "/movie/1248832":
            return external()
        return None

    def test_local_movie_found_qualified(self):
        result = self.tools.resolve_movie_title("Local Film (2010)")
        self.assertEqual(result.movie.tmdb_id, 7)
        self.live.assert_not_called()

    def test_missing_local_movie_live_found_exact_metadata(self):
        result = self.tools.resolve_movie_title("Digger 2026")
        self.assertEqual(result.status, "success")
        self.assertEqual((result.movie.tmdb_id, result.movie.year), (1248832, 2026))
        self.assertEqual(result.details.director, "Alejandro G. Iñárritu")
        self.assertEqual(result.details.actors, "Tom Cruise")
        self.assertEqual(self.live.call_count, 2)
        self.assertEqual(self.live.call_args_list[0].args, ("/search/movie", {
            "query": "Digger", "page": 1, "include_adult": "false", "year": 2026}))
        self.assertNotIn(1248832, self.tools.by_id)
        self.assertEqual(len(self.catalog), 4)

    def test_unqualified_old_local_movie_does_not_hide_new_namesake(self):
        result = self.tools.resolve_movie_title("Digger")
        self.assertEqual(result.status, "needs_clarification")
        self.assertEqual({m.tmdb_id for m in result.options}, {656272, 1248832})
        self.assertIsNone(result.movie)

    def test_title_and_year_mismatch_rejected(self):
        for movie in [external(title="Unrelated Film"), external(year=2021)]:
            with self.subTest(movie=movie):
                self.live.side_effect = None
                self.live.return_value = {"results": [movie], "total_pages": 1}
                result = self.tools.resolve_movie_title("Digger", 2026)
                self.assertEqual(result.status, "not_found")
                self.assertIsNone(result.movie)

    def test_detailed_identity_must_match_search(self):
        for movie in [external(title="Wrong Film"), external(year=2025), external(identity=111)]:
            self.live.side_effect = [{"results": [external()], "total_pages": 1}, movie]
            result = self.tools.resolve_movie_title("Digger (2026)")
            self.assertIn(result.status, {"not_found", "error"})
            self.assertIsNone(result.movie)

    def test_local_duplicate_ambiguity_preserved(self):
        result = self.tools.resolve_movie_title("Duplicate")
        self.assertEqual(result.status, "needs_clarification")
        self.assertEqual({m.tmdb_id for m in result.options}, {8, 9})
        self.live.assert_not_called()

    def test_live_duplicate_ambiguity_not_popularity_guess(self):
        self.live.side_effect = None
        self.live.return_value = {"results": [external(), external(identity=999)], "total_pages": 1}
        result = self.tools.resolve_movie_title("Digger (2026)")
        self.assertEqual(result.status, "needs_clarification")
        self.assertEqual({m.tmdb_id for m in result.options}, {1248832, 999})

    def test_live_unavailable_is_not_claimed_missing(self):
        self.live.side_effect = RuntimeError("credential-secret-must-not-leak")
        result = self.tools.resolve_movie_title("Digger (2026)")
        self.assertEqual(result.status, "error")
        self.assertIn("temporarily unavailable", result.message)
        self.assertNotIn("credential-secret", result.model_dump_json())
        self.assertEqual(self.tools.resolve_movie_title("Digger").status, "needs_clarification")

    def test_incomplete_search_page_never_proves_unique_identity(self):
        self.live.side_effect = None
        self.live.return_value = {"results": [external()], "total_pages": 2}
        self.assertEqual(self.tools.resolve_movie_title("Digger (2026)").status, "needs_clarification")
        self.assertEqual(self.live.call_count, 1)

    def test_external_seed_reports_precise_capability_no_cards(self):
        before = self.catalog.copy(deep=True)
        result = self.tools.recommend_movies(MovieIntent(intent="recommend", favorite_movies=["Digger (2026)"]))
        self.assertEqual(result.status, "needs_preferences")
        self.assertEqual(result.resolved_movies[0].tmdb_id, 1248832)
        self.assertIn("catalog-row embedding", result.message)
        self.assertEqual(result.external_movies[0].actors, "Tom Cruise")
        self.assertEqual(result.recommendations, [])
        self.ranking.assert_not_called()
        pd.testing.assert_frame_equal(before, self.catalog)

    def test_mixed_local_external_seeds_never_rank_partial_subset(self):
        result = self.tools.recommend_movies(MovieIntent(intent="recommend",
            favorite_movies=["Local Film (2010)", "Digger (2026)"]))
        self.assertEqual(result.recommendations, [])
        self.ranking.assert_not_called()

    def test_local_seed_still_uses_authoritative_runner(self):
        self.tools.recommend_movies(MovieIntent(intent="recommend", favorite_movies=["Local Film (2010)"]))
        self.ranking.assert_called_once()
        self.assertEqual(self.ranking.call_args.kwargs["favorite_movies"][0]["tmdb_id"], 7)

    def test_external_provider_and_details_use_verified_id(self):
        self.assertEqual(self.tools.get_movie_details(1248832).movie.director, "Alejandro G. Iñárritu")
        result = self.tools.get_watch_providers(1248832, "SE")
        self.assertEqual(result.tmdb_id, 1248832)
        self.providers.assert_called_once_with(1248832, "SE")

    async def test_exact_request_and_year_correction_override_stale_session(self):
        state = MovieSessionState(last_movie_list=[MovieReference(tmdb_id=7, title="Local Film", year=2010)])
        assistant = MovieAssistant(self.tools, self.client)

        async def responder(message, country, controller, tools, *, session):
            return await assistant.respond(message, country, session=session)

        # The unqualified title must clarify the title/year, never the stale list.
        self.client.responses.create.side_effect = [response(call("recommend_movies", favorite_movies=["Digger"])),
            response(text="Which Digger release year do you mean?")]
        result = await handle_session_turn("I watched Digger yesterday and liked it. Recommend similar movies.",
            "SE", state, self.controller, self.tools, responder)
        self.assertEqual(result.status, "needs_clarification")
        self.assertEqual(result.clarifications[0].query, "Digger")
        self.assertEqual({m.tmdb_id for m in result.clarifications[0].options}, {656272, 1248832})
        self.assertNotIn("last list", result.assistant_message)
        self.assertIsNone(state.current_movie)

        # Even if extraction drops the year, restore the user's explicit qualifier.
        self.client.responses.create.side_effect = [response(call("recommend_movies", favorite_movies=["Digger"]))]
        result = await handle_session_turn("No, Digger 2026", "SE", state, self.controller, self.tools, responder)
        self.assertEqual(result.status, "needs_preferences")
        self.assertEqual(result.resolved_movies[0].tmdb_id, 1248832)
        self.assertIn("cannot score", result.assistant_message)
        self.assertEqual(result.recommendations, [])
        self.assertEqual(result.openai_requests, 0)  # Pending task resumes without another model request.
        self.assertEqual(state.current_movie.tmdb_id, 1248832)
        self.ranking.assert_not_called()

    async def test_verified_watched_external_identity_retained_without_fake_ranking(self):
        state = MovieSessionState()
        self.client.responses.create.side_effect = [response(call("recommend_movies", favorite_movies=["Digger (2026)"]))]
        assistant = MovieAssistant(self.tools, self.client)

        async def responder(message, country, controller, tools, *, session):
            return await assistant.respond(message, country, session=session)

        result = await handle_session_turn("I watched Digger (2026) and liked it. Recommend similar movies.",
            "SE", state, self.controller, self.tools, responder)
        self.assertEqual(state.seen_movie_ids, [1248832])
        self.assertEqual(state.excluded_movie_ids, [1248832])
        self.assertEqual(state.last_movie_list, [])
        self.assertEqual(result.recommendations, [])

    def test_real_engine_missing_seed_cannot_score_no_weight_changes(self):
        import backend.recommendation.recommendation_engine as ranking
        weights = {name: value for name, value in vars(ranking).items() if name.isupper() and isinstance(value, (int, float))}
        svd_factors = api.engine.svd.movie_factors.copy()
        with contextlib.redirect_stdout(io.StringIO()):
            result = api.engine.recommend(favorite_movies=[{"tmdb_id": 1248832, "title": "Digger"}], top_n=5)
        # The engine silently ignores the absent ID and can return a fallback
        # list. It must not be presented as similarity scoring for this movie.
        with contextlib.redirect_stdout(io.StringIO()):
            another = api.engine.recommend(favorite_movies=[{"tmdb_id": 999999999, "title": "Another Missing Movie"}], top_n=5)
        self.assertFalse(result.empty)
        self.assertEqual(result.tmdb_id.tolist(), another.tmdb_id.tolist())
        self.assertEqual(result.content_score.tolist(), another.content_score.tolist())
        self.assertEqual(weights, {name: value for name, value in vars(ranking).items() if name.isupper() and isinstance(value, (int, float))})
        np.testing.assert_array_equal(svd_factors, api.engine.svd.movie_factors)
        self.assertNotIn(1248832, set(api.engine.content.movies.tmdb_id))


if __name__ == "__main__":
    unittest.main()
