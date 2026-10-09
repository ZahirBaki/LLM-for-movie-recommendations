"""Offline tool-loop tests with scripted OpenAI responses and real backend tools."""
import contextlib
import io
import json
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, patch

import api
from fastapi.testclient import TestClient
from movie_assistant import MovieAssistant, MAX_TOOL_ROUNDS, compact_result
from test_chat_recommender import empty_providers


def call(name, **arguments):
    return NS(type="function_call", name=name, arguments=json.dumps(arguments), call_id="call-test")


def response(*calls, text="Grounded answer"):
    return NS(status="completed", output=list(calls), output_text=text)


class AssistantTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.provider = patch("api.get_watch_providers", side_effect=empty_providers).start()
        patch("requests.sessions.Session.request", side_effect=AssertionError("Offline only")).start()
        patch.object(api.movie_tools, "live_movie_runner", None).start()  # Local-only fixtures; live resolution has separate mocked coverage.
        self.addCleanup(patch.stopall)
        self.client = NS(responses=NS(create=AsyncMock()))
        self.assistant = MovieAssistant(api.movie_tools, self.client)

    async def run_script(self, *responses, message="test", country="US"):
        self.client.responses.create.side_effect = responses
        with contextlib.redirect_stdout(io.StringIO()):
            return await self.assistant.respond(message, country)

    async def test_casual_no_tools(self):
        result = await self.run_script(response(text="Hello! What would you like to watch?"))
        self.assertEqual(result.tool_results, [])
        self.assertEqual(result.recommendations, [])
        self.assertEqual(result.openai_requests, 1)

    async def test_facts_rating_director_cast_grounded_in_catalog(self):
        for title in ("Interstellar", "Dune (2021)"):
            resolved = api.movie_tools.resolve_movie_title(title).movie
            result = await self.run_script(response(call("resolve_movie_title", title=title)),
                response(call("get_movie_details", tmdb_id=resolved.tmdb_id)), response())
            data = result.tool_results[-1]["result"]["movie"]
            self.assertEqual(data, api.movie_tools.get_movie_details(resolved.tmdb_id).movie.model_dump())
            submitted = self.client.responses.create.call_args.kwargs["input"][-1]
            self.assertEqual(json.loads(submitted["output"])["movie"]["vote_average"], data["vote_average"])
            self.assertIn("director", json.loads(submitted["output"])["movie"])
            self.assertIn("actors", json.loads(submitted["output"])["movie"])

    async def test_selected_and_explicit_provider_country(self):
        for region, expected in ((None, "US"), ("SE", "SE")):
            result = await self.run_script(response(call("resolve_movie_title", title="Interstellar")),
                response(call("get_watch_providers", tmdb_id=157336, country_code=region)), response())
            self.assertEqual(result.country_code, expected)
            self.provider.assert_called_with(157336, expected)

    async def test_recommendations_preserve_engine_order_and_animation_ids(self):
        for titles, count in ((["Interstellar"], 3), (["Coco", "Inside Out", "Zootopia"], 5)):
            with patch.object(api.chat_recommender, "recommend_from_intent", wraps=api.chat_recommender.recommend_from_intent) as runner:
                result = await self.run_script(response(call("recommend_movies", favorite_movies=titles, top_n=count)), response())
            self.assertEqual(len(result.recommendations), count)
            self.assertEqual(result.recommendations, result.tool_results[-1]["result"]["recommendations"])
            runner.assert_called_once()
            if titles[0] == "Coco":
                self.assertEqual([m.tmdb_id for m in result.resolved_movies], [354912, 150540, 269149])
            sent = self.client.responses.create.call_args.kwargs["input"][-1]["output"]
            self.assertNotIn("svd_score", sent)
            self.assertEqual(self.client.responses.create.call_args.kwargs["tool_choice"], "none")

    async def test_search_crash(self):
        result = await self.run_script(response(call("search_movies", query="Crash")), response())
        self.assertTrue({884, 1640}.issubset({m["tmdb_id"] for m in result.tool_results[0]["result"]["matches"]}))
        self.assertEqual(result.recommendations, [])

    async def test_ambiguous_stops_before_facts(self):
        result = await self.run_script(response(call("resolve_movie_title", title="Crash")), response())
        self.assertEqual(result.status, "needs_clarification")
        self.assertEqual({o.tmdb_id for o in result.clarifications[0].options}, {884, 1640})
        self.assertEqual(self.client.responses.create.call_args.kwargs["tool_choice"], "none")

    async def test_exact_crash_year(self):
        result = await self.run_script(response(call("resolve_movie_title", title="Crash (1996)")),
            response(call("get_movie_details", tmdb_id=884)), response())
        self.assertEqual(result.resolved_movies[0].tmdb_id, 884)

    async def test_not_found(self):
        result = await self.run_script(response(call("resolve_movie_title", title="Nonexistent xyz987654")), response())
        self.assertEqual(result.status, "unresolved")

    async def test_needs_preferences_no_fallback(self):
        with patch.object(api.engine, "recommend", side_effect=AssertionError("No scoring without preferences")):
            result = await self.run_script(response(call("recommend_movies", required_genres=["Comedy"], min_year=2026)), response())
        self.assertEqual(result.status, "needs_preferences")
        self.assertEqual(result.recommendations, [])

    async def test_invalid_arguments_unknown_tools_and_unresolved_ids(self):
        malformed = call("search_movies", query="Crash")
        malformed.arguments = "not json"
        for item in (malformed, call("__import__", module="os"), call("get_movie_details", tmdb_id=157336),
                     call("search_movies", query="Crash", limit="10"), call("search_movies", query="Crash", extra=True),
                     call("recommend_movies", top_n=500)):
            with self.subTest(item=item), self.assertRaises((ValueError, TypeError)):
                await self.run_script(response(item))

    async def test_loop_limit(self):
        with self.assertRaises(ValueError):
            await self.run_script(*[response(call("search_movies", query="Crash")) for _ in range(MAX_TOOL_ROUNDS + 1)])

    async def test_api_error_sanitized_at_route(self):
        from unittest.mock import patch
        with patch.dict("os.environ", {"MOVIE_CHAT_MOCK": "false"}), patch("movie_chat.respond_live", side_effect=RuntimeError("secret key")):
            result = TestClient(api.app).post("/movie-chat", json={"message": "Hello"})
        self.assertEqual(result.status_code, 503)
        self.assertNotIn("secret", result.text)

    async def test_compaction_keeps_missing_facts(self):
        self.assertEqual(compact_result({"runtime": None, "vote_average": 7, "svd_score": 1}), {"runtime": None, "vote_average": 7})


if __name__ == "__main__":
    unittest.main()
