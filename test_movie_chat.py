"""Offline single-message orchestration tests; real persisted recommender."""

import contextlib
import io
import os
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import api
from chat_service import ChatServiceError, MovieIntent
from test_chat_recommender import empty_providers


class MovieChatTests(unittest.TestCase):
    def setUp(self):
        patch.dict(os.environ, {"MOVIE_CHAT_MOCK": "true", "ENVIRONMENT": "development", "APP_ENV": "development", "NODE_ENV": "development"}).start()
        patch("api.get_watch_providers", side_effect=empty_providers).start()
        patch("requests.sessions.Session.request", side_effect=AssertionError("No outbound network in tests")).start()
        self.parser = patch("movie_chat.parse_movie_message", new_callable=AsyncMock, side_effect=AssertionError("No OpenAI in mock mode")).start()
        self.addCleanup(patch.stopall)
        self.client = TestClient(api.app)

    def submit(self, message, country="SE"):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.client.post("/movie-chat", json={"message": message, "country_code": country})

    def test_five_mock_cases(self):
        cases = [
            ("Something like Interstellar", "success", [157336]),
            ("Something like Coco, Inside Out and Zootopia", "success", [354912, 150540, 269149]),
            ("Something like Interstellar with Ryan Gosling after 2015", "success", [157336]),
            ("Something like Crash", "needs_clarification", []),
            ("Something like Nonexistent Movie xyz987654", "unresolved", []),
        ]
        for phrase, status, ids in cases:
            with self.subTest(phrase=phrase):
                response = self.submit(phrase)
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertEqual(data["status"], status)
                self.assertEqual([movie["tmdb_id"] for movie in data["resolved_movies"]], ids)
                self.assertTrue(data["mock_mode"])
                if status == "success":
                    self.assertTrue(data["recommendations"])
                if "Ryan Gosling" in phrase:
                    self.assertEqual(data["parsed_intent"]["required_actors"], ["Ryan Gosling"])
                    self.assertEqual(data["parsed_intent"]["min_year"], 2016)
                    for movie in data["recommendations"]:
                        self.assertGreaterEqual(movie["year"], 2016)
                        row = api.engine.content.movies.loc[api.engine.content.movies.tmdb_id == movie["tmdb_id"]].iloc[0]
                        self.assertIn("Ryan Gosling", row.actors)
        self.parser.assert_not_awaited()

    def test_clarification_followup_is_independent(self):
        result = self.submit("Something like Crash").json()
        options = result["clarifications"][0]["options"]
        self.assertEqual({item["tmdb_id"] for item in options}, {884, 1640})
        for option in options:
            selected = self.submit(f"Something like {option['title']} ({option['year']})").json()
            self.assertEqual(selected["status"], "success")
            self.assertEqual(selected["resolved_movies"][0]["tmdb_id"], option["tmdb_id"])

    def test_other_states_and_normalization(self):
        self.assertEqual(self.submit("  SOMETHING   LIKE Interstellar  ").json()["status"], "success")
        self.assertEqual(self.submit("Recommend a comedy").json()["status"], "needs_preferences")
        self.assertEqual(self.submit("Who directed Interstellar?").json()["status"], "unsupported_intent")
        result = self.submit("An arbitrary mock phrase").json()
        self.assertEqual(result["status"], "error")
        self.assertIn("development mock mode", result["assistant_message"])
        self.parser.assert_not_awaited()

    def test_validation_and_request_region(self):
        self.assertEqual(self.submit("   ").status_code, 422)
        self.assertEqual(self.submit("Something like Interstellar", "invalid").status_code, 422)
        result = self.submit("Something like Interstellar", "US").json()
        self.assertEqual(result["parsed_intent"]["country_code"], "US")
        self.assertTrue(all(movie["watch_providers"]["country_code"] == "US" for movie in result["recommendations"]))

    def test_real_mode_reuses_parser_and_overrides_region(self):
        self.parser.side_effect = None
        self.parser.return_value = MovieIntent(intent="recommend", favorite_movies=["Interstellar"], country_code="US", top_n=2)
        with patch.dict(os.environ, {"MOVIE_CHAT_MOCK": "false"}):
            result = self.submit("Any natural language", "SE").json()
        self.assertEqual(result["status"], "success")
        self.assertFalse(result["mock_mode"])
        self.assertEqual(result["parsed_intent"]["country_code"], "SE")
        self.parser.assert_awaited_once_with("Any natural language")

    def test_production_disables_mock_and_missing_flag_defaults_off(self):
        self.parser.side_effect = ChatServiceError(503, "sensitive upstream detail")
        for config in ({"ENVIRONMENT": "production"}, {"APP_ENV": "production"}, {"NODE_ENV": "production"}, {"MOVIE_CHAT_MOCK": ""}):
            with patch.dict(os.environ, config):
                response = self.submit("Something like Interstellar")
                self.assertEqual(response.status_code, 503)
                self.assertFalse(response.json()["mock_mode"])
                self.assertNotIn("sensitive", response.text)

    def test_errors_never_expose_upstream_details_or_fake_results(self):
        for error in (ChatServiceError(503, "quota secret"), ChatServiceError(503, "auth secret"), ChatServiceError(502, "parse secret"), RuntimeError("internal secret")):
            self.parser.side_effect = error
            with patch.dict(os.environ, {"MOVIE_CHAT_MOCK": "false"}):
                response = self.submit("Something like Interstellar")
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["status"], "error")
            self.assertEqual(response.json()["recommendations"], [])
            self.assertNotIn("secret", response.text)
        with patch.object(api.chat_recommender, "recommend_from_intent", side_effect=RuntimeError("ranking secret")):
            response = self.submit("Something like Interstellar")
            self.assertEqual(response.status_code, 503)
            self.assertNotIn("secret", response.text)


if __name__ == "__main__":
    unittest.main()
