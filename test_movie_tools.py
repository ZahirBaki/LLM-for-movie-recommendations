"""Offline tests for the tool contracts and existing endpoint compatibility."""

import contextlib
import io
import json
import os
import unittest
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pandas as pd
import requests
from fastapi.testclient import TestClient

import api
from chat_recommender import ChatRecommender, IntentRecommendationResult
from chat_service import MovieIntent
from movie_tools import MovieTools, json_safe
from test_chat_recommender import empty_providers


class MovieToolsTests(unittest.TestCase):
    def setUp(self):
        self.original_provider = api.get_watch_providers
        patch("api.get_watch_providers", side_effect=empty_providers).start()
        patch("requests.sessions.Session.request", side_effect=AssertionError("No network in tool tests")).start()
        patch("movie_chat.parse_movie_message", new_callable=AsyncMock, side_effect=AssertionError("No OpenAI in tool tests")).start()
        self.addCleanup(patch.stopall)
        self.tools = api.movie_tools

    def test_search_interstellar_crash_limits_and_not_found(self):
        result = self.tools.search_movies("Interstellar")
        self.assertEqual(result.status, "success")
        self.assertEqual(result.matches[0].tmdb_id, 157336)
        self.assertEqual(result.matches[0].original_title, self.tools.get_movie_details(157336).movie.original_title)
        crash = self.tools.search_movies("Crash")
        self.assertTrue({884, 1640}.issubset({movie.tmdb_id for movie in crash.matches}))
        self.assertEqual(len(self.tools.search_movies("Crash", limit=1).matches), 1)
        self.assertEqual(self.tools.search_movies("No Such Movie xyz987654").status, "not_found")
        self.assertEqual(self.tools.search_movies("   ").status, "error")
        self.assertEqual(self.tools.search_movies("Crash", limit=11).status, "error")
        self.assertEqual([movie.model_dump(include={"tmdb_id", "title", "year", "poster_url"}) for movie in crash.matches], api.search_movies(q="Crash"))

    def test_details_are_catalog_grounded(self):
        resolved = self.tools.resolve_movie_title("Interstellar")
        result = self.tools.get_movie_details(resolved.movie.tmdb_id)
        self.assertEqual(result.status, "success")
        row = api.engine.content.movies.loc[api.engine.content.movies.tmdb_id == 157336].iloc[0]
        for field in ("title", "original_title", "release_date", "release_year", "director", "actors", "genres", "overview", "original_language", "popularity", "vote_average", "vote_count", "runtime", "poster_path"):
            self.assertEqual(getattr(result.movie, field), json_safe(row[field]), field)
        self.assertEqual(result.movie.rating_source, "TMDB")
        self.assertEqual(result.movie.poster_url, "https://image.tmdb.org/t/p/w500" + row.poster_path)
        self.assertEqual(self.tools.get_movie_details(999999999).status, "not_found")
        self.assertEqual(self.tools.get_movie_details(-1).status, "error")

    def test_shared_resolution_regressions(self):
        for title, tmdb_id, year in (("Coco", 354912, 2017), ("Inside Out", 150540, 2015), ("Zootopia", 269149, 2016)):
            result = self.tools.resolve_movie_title(title)
            self.assertEqual(result.status, "success")
            self.assertEqual((result.movie.tmdb_id, result.movie.year), (tmdb_id, year))
        with patch.object(api.chat_recommender, "resolve_title", wraps=api.chat_recommender.resolve_title) as resolver:
            ambiguous = self.tools.resolve_movie_title("Crash")
            resolver.assert_called_once_with("Crash")
        self.assertEqual(ambiguous.status, "needs_clarification")
        self.assertIsNone(ambiguous.movie)
        self.assertEqual({movie.tmdb_id for movie in ambiguous.options}, {884, 1640})
        self.assertEqual(self.tools.resolve_movie_title("Crash (1996)").movie.tmdb_id, 884)
        self.assertEqual(self.tools.resolve_movie_title("No Such Movie xyz987654").status, "not_found")

    def test_real_provider_mapping_with_offline_tmdb_response(self):
        # Recover the existing fetcher under the test's default enrichment patch.
        original = self.original_provider
        tmdb_provider = {"provider_id": 8, "provider_name": "Fixture", "logo_path": "/fixture.png", "display_priority": 2}
        response = Mock()
        response.json.return_value = {"results": {"US": {name: [tmdb_provider] for name in ("flatrate", "free", "ads", "rent", "buy")}, "SE": {}}}
        response.json.return_value["results"]["US"]["link"] = "https://www.justwatch.com/"
        tools = MovieTools(api.engine.content.movies, api.chat_recommender, api.search_movies, original)
        with patch("api.TMDB_API_KEY", "offline-fixture"), patch("api.requests.get", return_value=response) as fetch:
            result = tools.get_watch_providers(157336, " us ")
            missing = tools.get_watch_providers(157336, "SE")
        self.assertEqual(result.status, "success")
        self.assertEqual(result.watch_providers.country_code, "US")
        for category in ("stream", "free", "ads", "rent", "buy"):
            provider = getattr(result.watch_providers, category)[0]
            self.assertEqual(provider.provider_id, 8)
            self.assertEqual(provider.logo_url, "https://image.tmdb.org/t/p/w92/fixture.png")
        self.assertEqual(result.watch_providers.link, "https://www.justwatch.com/")
        self.assertEqual(missing.status, "not_available")
        self.assertEqual(self.tools.get_watch_providers(999999999, "US").status, "not_found")
        self.assertEqual(self.tools.get_watch_providers(157336, "invalid").status, "error")
        self.assertEqual(fetch.call_count, 2)

    def test_provider_failures_do_not_log_exception_secrets(self):
        original = self.original_provider
        tools = MovieTools(api.engine.content.movies, api.chat_recommender, api.search_movies, original)
        output = io.StringIO()
        with patch("api.TMDB_API_KEY", "offline-fixture"), patch("api.requests.get", side_effect=requests.RequestException("sensitive test marker")), contextlib.redirect_stdout(output):
            result = tools.get_watch_providers(157336, "SE")
        self.assertEqual(result.status, "not_available")
        self.assertNotIn("sensitive test marker", output.getvalue())

    def test_recommendation_parity_and_country_ranking(self):
        intent = MovieIntent(intent="recommend", favorite_movies=["Interstellar"], top_n=2)
        with contextlib.redirect_stdout(io.StringIO()):
            expected = api.chat_recommender.recommend_from_intent(intent)
            actual = self.tools.recommend_movies(intent)
            us = self.tools.recommend_movies(intent.model_copy(update={"country_code": "US"}))
        self.assertEqual(actual.model_dump(), expected.model_dump())
        self.assertEqual([movie["tmdb_id"] for movie in actual.recommendations], [movie["tmdb_id"] for movie in us.recommendations])
        self.assertEqual([movie["final_score"] for movie in actual.recommendations], [movie["final_score"] for movie in us.recommendations])
        self.assertTrue(all(movie["watch_providers"]["country_code"] == "US" for movie in us.recommendations))
        json.dumps(actual.model_dump(), allow_nan=False)

    def test_recommendation_states_and_category_limitation(self):
        for fields, status in (({"favorite_movies": ["Crash"]}, "needs_clarification"), ({"favorite_movies": ["No Such Movie xyz987654"]}, "unresolved"), ({"required_genres": ["Comedy"]}, "needs_preferences")):
            result = self.tools.recommend_movies(MovieIntent(intent="recommend", **fields))
            self.assertEqual(result.status, status)
        self.assertIn("Required filters alone", result.message)
        with patch.object(api.chat_recommender, "recommend_from_intent", side_effect=RuntimeError("sensitive test marker")):
            failed = self.tools.recommend_movies(MovieIntent(intent="recommend", country_code="US"))
        self.assertEqual(failed.status, "error")
        self.assertEqual(failed.country_code, "US")
        self.assertNotIn("sensitive", failed.message)

    def test_json_safety_missing_catalog_values(self):
        row = {"tmdb_id": np.int64(123), "title": "Fixture", "release_year": np.float64(np.nan), "vote_count": np.int64(5), "runtime": pd.NA,
               "popularity": np.float64(np.inf), "vote_average": np.float32(-np.inf), "actors": "", "genres": pd.NA, "release_date": pd.NaT}
        catalog = pd.DataFrame([row])
        tools = MovieTools(catalog, Mock(), Mock(return_value=[]), Mock(side_effect=RuntimeError("sensitive")))
        details = tools.get_movie_details(123)
        self.assertEqual(details.status, "success")
        for field in ("release_year", "runtime", "popularity", "vote_average", "actors", "genres", "release_date", "poster_url"):
            self.assertIsNone(getattr(details.movie, field), field)
        json.dumps(details.model_dump(), allow_nan=False)
        self.assertEqual(tools.get_watch_providers(123, "SE").status, "error")
        dirty = IntentRecommendationResult(status="success", country_code="SE", recommendations=[{"tmdb_id": np.int64(123), "final_score": np.float32(np.inf)}])
        tools.controller.recommend_from_intent.return_value = dirty
        clean = tools.recommend_movies(MovieIntent(intent="recommend", favorite_genres=["Comedy"]))
        self.assertIsNone(clean.recommendations[0]["final_score"])
        json.dumps(clean.model_dump(), allow_nan=False)
        for result in (self.tools.search_movies("Crash"), self.tools.resolve_movie_title("Crash"), self.tools.get_watch_providers(157336, "SE")):
            json.dumps(result.model_dump(), allow_nan=False)

    def test_existing_read_and_chat_endpoints(self):
        client = TestClient(api.app)
        for route in ("/", "/movies/search?q=Interstellar", "/actors/search?q=Bale", "/directors/search?q=Nolan", "/genres"):
            self.assertEqual(client.get(route).status_code, 200, route)
        with patch("api.parse_movie_message", new_callable=AsyncMock, return_value=MovieIntent(intent="information", movie_titles=["Interstellar"])):
            parsed = client.post("/chat", json={"message": "Who directed Interstellar?"})
        self.assertEqual(parsed.status_code, 200)
        self.assertEqual(parsed.json()["parsed"]["intent"], "information")
        with patch.dict(os.environ, {"MOVIE_CHAT_MOCK": "true", "ENVIRONMENT": "development", "APP_ENV": "development", "NODE_ENV": "development"}), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(client.post("/movie-chat", json={"message": "Something like Interstellar", "country_code": "SE"}).json()["status"], "success")


if __name__ == "__main__":
    unittest.main()
