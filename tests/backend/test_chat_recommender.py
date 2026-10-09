"""Offline controller integration tests using the persisted catalog/SVD artifacts."""

import contextlib
import io
import unittest
from unittest.mock import Mock, patch

import pandas as pd
from fastapi.testclient import TestClient

import backend.api.app as api
from backend.services.chat_recommender import ChatRecommender
from backend.services.chat_service import MovieIntent


def empty_providers(tmdb_id, country_code="SE"):
    return {"country_code": country_code, "stream": [], "free": [], "ads": [], "rent": [], "buy": [], "link": None}


class ControllerTests(unittest.TestCase):
    def setUp(self):
        # Block every outbound HTTP request and every LLM call. Only provider
        # enrichment is substituted; the real recommender and artifacts are used.
        self.provider = patch('backend.api.app.get_watch_providers', side_effect=empty_providers).start()
        self.network = patch("requests.sessions.Session.request", side_effect=AssertionError("Tests must not access the network")).start()
        self.parser = patch('backend.api.app.parse_movie_message', side_effect=AssertionError("Controller must not call OpenAI")).start()
        self.addCleanup(patch.stopall)
        self.client = TestClient(api.app)

    def run_controller(self, intent):
        with contextlib.redirect_stdout(io.StringIO()):
            return api.chat_recommender.recommend_from_intent(intent)

    def test_interstellar_real_recommendations(self):
        result = self.run_controller(MovieIntent(intent="recommend", favorite_movies=["Interstellar"], top_n=3))
        self.assertEqual(result.status, "success")
        self.assertEqual(result.resolved_movies[0].model_dump(), {"tmdb_id": 157336, "title": "Interstellar", "year": 2014})
        self.assertEqual(len(result.recommendations), 3)
        self.assertTrue(all(movie["tmdb_id"] != 157336 for movie in result.recommendations))
        self.assertIn("final_score", result.recommendations[0])

    def test_animation_duplicate_regression_and_ids_reach_engine(self):
        with patch.object(api.engine, "recommend", wraps=api.engine.recommend) as engine_call, patch.object(api.engine.content, "get_scores", wraps=api.engine.content.get_scores) as content_call:
            result = self.run_controller(MovieIntent(intent="recommend", favorite_movies=["Coco", "Inside Out", "Zootopia"], top_n=3))
        expected = [(354912, 2017), (150540, 2015), (269149, 2016)]
        self.assertEqual([(movie.tmdb_id, movie.year) for movie in result.resolved_movies], expected)
        self.assertEqual(result.status, "success")
        self.assertEqual(len(result.recommendations), 3)
        self.assertEqual([movie["tmdb_id"] for movie in engine_call.call_args.kwargs["favorite_movies"]], [value[0] for value in expected])
        self.assertEqual(content_call.call_args.kwargs["favorite_tmdb_ids"], [value[0] for value in expected])

    def test_crash_requires_clarification(self):
        with patch.object(api.engine, "recommend") as engine_call:
            result = self.run_controller(MovieIntent(intent="recommend", favorite_movies=["Crash"]))
        self.assertEqual(result.status, "needs_clarification")
        self.assertEqual({(movie.tmdb_id, movie.year) for movie in result.clarifications[0].options}, {(884, 1996), (1640, 2005)})
        self.assertEqual(result.recommendations, [])
        engine_call.assert_not_called()

    def test_nonexistent_and_partial_resolution(self):
        with patch.object(api.engine, "recommend") as engine_call:
            result = self.run_controller(MovieIntent(intent="recommend", favorite_movies=["Interstellar", "A Completely Nonexistent Movie xyz987654"]))
        self.assertEqual(result.status, "unresolved")
        self.assertEqual(result.unresolved_movies, ["A Completely Nonexistent Movie xyz987654"])
        self.assertEqual(result.resolved_movies[0].tmdb_id, 157336)
        self.assertEqual(result.recommendations, [])
        engine_call.assert_not_called()

    def test_explicit_year_normalization_and_deduplication(self):
        runner = Mock(return_value=[])
        controller = ChatRecommender(api.engine.content.movies, runner)
        result = controller.recommend_from_intent(MovieIntent(intent="recommend", favorite_movies=["  CRASH (1996) ", "Crash (1996)", "Crash (1996)"]))
        self.assertEqual([movie.tmdb_id for movie in result.resolved_movies], [884])
        missing = controller.recommend_from_intent(MovieIntent(intent="recommend", favorite_movies=["Crash (1999)"]))
        self.assertEqual(missing.status, "unresolved")
        self.assertEqual(controller.resolve_title("Interstelar"), [])

    def test_no_recognition_data_never_guesses(self):
        catalog = pd.DataFrame([
            {"tmdb_id": 1, "title": "Duplicate", "release_year": 1990},
            {"tmdb_id": 2, "title": "Duplicate", "release_year": 2000},
        ])
        runner = Mock()
        result = ChatRecommender(catalog, runner).recommend_from_intent(MovieIntent(intent="recommend", favorite_movies=["Duplicate"]))
        self.assertEqual(result.status, "needs_clarification")
        runner.assert_not_called()

    def test_every_supported_field_and_unsupported_metadata(self):
        runner = Mock(return_value=[])
        controller = ChatRecommender(api.engine.content.movies, runner)
        fields = {
            "favorite_genres": ["Adventure"], "favorite_actors": ["Ryan Gosling"],
            "favorite_directors": ["Christopher Nolan"], "required_genres": ["Science Fiction"],
            "required_actors": ["Matthew McConaughey"], "required_directors": ["Christopher Nolan"],
            "exclude_genres": ["Horror"], "exclude_actors": ["Tom Cruise"],
            "exclude_directors": ["Quentin Tarantino"], "min_year": 2000, "max_year": 2020, "top_n": 4,
        }
        result = controller.recommend_from_intent(MovieIntent(intent="recommend", **fields, country_code="US", natural_language_preferences=["less depressing"], movie_titles=["Information subject"]))
        self.assertEqual(runner.call_args.kwargs, {**fields, "country_code": "US", "favorite_movies": []})
        self.assertEqual(result.unsupported_preferences, {"natural_language_preferences": ["less depressing"], "movie_titles": ["Information subject"]})

    def test_actual_required_filters_and_year_bounds(self):
        result = self.run_controller(MovieIntent(intent="recommend", favorite_movies=["Inception"], required_genres=["Science Fiction"], required_actors=["Matthew McConaughey"], required_directors=["Christopher Nolan"], min_year=2014, max_year=2014, top_n=3))
        self.assertEqual(result.status, "success")
        self.assertTrue(result.recommendations)
        self.assertTrue(all(movie["year"] == 2014 for movie in result.recommendations))
        for movie in result.recommendations:
            row = api.engine.content.movies.loc[api.engine.content.movies.tmdb_id == movie["tmdb_id"]].iloc[0]
            self.assertIn("Matthew McConaughey", row.actors)
            self.assertIn("Christopher Nolan", row.director)
            self.assertIn("Science Fiction", row.genres)

    def test_actual_exclusions(self):
        result = self.run_controller(MovieIntent(intent="recommend", favorite_movies=["Interstellar"], exclude_genres=["Horror"], exclude_actors=["Tom Cruise"], exclude_directors=["Christopher Nolan"], min_year=2010, max_year=2020, top_n=3))
        self.assertTrue(result.recommendations)
        for movie in result.recommendations:
            row = api.engine.content.movies.loc[api.engine.content.movies.tmdb_id == movie["tmdb_id"]].iloc[0]
            self.assertTrue(2010 <= movie["year"] <= 2020)
            self.assertNotIn("Horror", row.genres)
            self.assertNotIn("Tom Cruise", row.actors)
            self.assertNotIn("Christopher Nolan", row.director)

    def test_required_only_returns_existing_engine_limitation(self):
        with patch.object(api.engine, "recommend") as engine_call:
            result = self.run_controller(MovieIntent(intent="recommend", required_genres=["Comedy"], min_year=2000))
            self.assertEqual(result.status, "needs_preferences")
            self.assertEqual(result.recommendations, [])
            engine_call.assert_not_called()

    def test_region_affects_only_enrichment_and_website_parity(self):
        intent = MovieIntent(intent="recommend", favorite_movies=["Interstellar"], top_n=2, country_code="SE")
        with patch.object(api.engine, "recommend", wraps=api.engine.recommend) as engine_call:
            se = self.run_controller(intent)
            us = self.run_controller(intent.model_copy(update={"country_code": "US"}))
        self.assertEqual(engine_call.call_args_list[0].kwargs, engine_call.call_args_list[1].kwargs)
        self.assertEqual([movie["tmdb_id"] for movie in se.recommendations], [movie["tmdb_id"] for movie in us.recommendations])
        self.assertEqual([movie["final_score"] for movie in se.recommendations], [movie["final_score"] for movie in us.recommendations])
        self.assertTrue(all(movie["watch_providers"]["country_code"] == "US" for movie in us.recommendations))
        with contextlib.redirect_stdout(io.StringIO()):
            website = self.client.post("/recommend", json={"favorite_movies": [movie.model_dump() for movie in se.resolved_movies], "top_n": 2, "country_code": "SE"})
        self.assertEqual(website.status_code, 200)
        self.assertEqual(website.json(), se.recommendations)

    def test_structured_endpoint_and_non_recommend_intents(self):
        with contextlib.redirect_stdout(io.StringIO()):
            response = self.client.post("/recommend-from-intent", json={"intent": "recommend", "favorite_movies": ["Interstellar"], "top_n": 2})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "success")
        self.assertEqual(len(response.json()["recommendations"]), 2)
        self.assertEqual(self.client.post("/recommend-from-intent", json={"intent": "recommend", "min_year": 2020, "max_year": 2010}).status_code, 422)
        for kind in ("search", "information", "unknown"):
            with patch.object(api.engine, "recommend") as engine_call:
                result = self.run_controller(MovieIntent(intent=kind, movie_titles=["Interstellar"]))
                self.assertEqual(result.status, "unsupported_intent")
                engine_call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
