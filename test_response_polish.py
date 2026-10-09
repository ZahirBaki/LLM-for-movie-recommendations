"""Offline presentation checks without changing backend facts or membership."""
import contextlib
import io
import re
import unittest
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
import api
from movie_assistant import MovieAssistant, PROMPT
from movie_collections import MovieCollections, collection_response_text
from movie_guardrails import MAX_OUTPUT_TOKENS
from test_movie_assistant import call, response
from test_movie_collections import fixture_collection_request
from test_chat_recommender import empty_providers


class CollectionPolishTests(unittest.IsolatedAsyncioTestCase):
    def test_main_prompt_preserves_risen_sections(self):
        sections = re.split(
            r"(?m)^[ \t]*(ROLE|INSTRUCTIONS|STEPS|END[ \t]+GOAL|NARROWING)[ \t]*$",
            PROMPT,
        )
        self.assertEqual(
            [" ".join(header.split()) for header in sections[1::2]],
            ["ROLE", "INSTRUCTIONS", "STEPS", "END GOAL", "NARROWING"],
        )
        for body in sections[2::2]:
            self.assertTrue(body.strip(), "Each RISEN section needs instructions")

    async def answer(self, message):
        client = NS(responses=NS(create=AsyncMock(side_effect=[
            response(call("search_movie_collection", query="X-Men")),
            response(call("get_movie_collection", collection_id=748), text="Add Wolverine and Deadpool from memory")
        ])))
        with patch("api.request_tmdb_collection", side_effect=fixture_collection_request):
            return await MovieAssistant(api.movie_tools, client).respond(message, "SE")

    def assert_no_debug_prose(self, text):
        for phrase in ("provenance", "coverage metadata", "membership only", "backend", "tool returned", "has been verified", "RecommendationEngine"):
            self.assertNotIn(phrase.casefold(), text.casefold())

    async def test_release_list_polished_without_changing_grounded_data(self):
        result = await self.answer("Give me the X-Men collection in release order.")
        expected = MovieCollections(fixture_collection_request).get_movie_collection(748)
        self.assertEqual(result.collection.model_dump(), expected.model_dump())
        self.assertEqual(result.tool_results[-1]["result"], expected.model_dump())
        self.assertTrue(result.assistant_message.startswith("Sure —"))
        self.assertIn("release order:", result.assistant_message)
        self.assertIn("1. X-Men (2000)", result.assistant_message)
        self.assertIn("Dark Phoenix (2019)", result.assistant_message)
        self.assertNotRegex(result.assistant_message, r"\d{4}-\d{2}-\d{2}")
        self.assert_no_debug_prose(result.assistant_message)
        self.assertNotIn("Wolverine", result.assistant_message)
        self.assertNotIn("Deadpool", result.assistant_message)
        self.assertEqual(len(result.collection.movies), 3)
        self.assertFalse(result.collection.complete_franchise_verified)
        self.assertFalse(result.collection.story_order_supported)
        self.assertEqual(result.collection.source, "TMDB")

    async def test_ambiguous_order_defaults_to_release_order(self):
        result = await self.answer("Give me the X-Men collection in order.")
        self.assertIn("release order", result.assistant_message)
        self.assertNotIn("story chronology", result.assistant_message)
        self.assert_no_debug_prose(result.assistant_message)

    async def test_broad_franchise_limit_is_visible_and_no_extra_films(self):
        result = await self.answer("Give me every movie in the X-Men collection including Wolverine and Deadpool.")
        self.assertIn("can't verify every related film in the wider series", result.assistant_message)
        self.assertNotIn("Wolverine", result.assistant_message)
        self.assertNotIn("Deadpool", result.assistant_message)
        self.assertEqual([m.tmdb_id for m in result.collection.movies], [36657, 49538, 320288])
        self.assert_no_debug_prose(result.assistant_message)

    async def test_story_discussion_is_natural_and_not_labeled_as_collection_data(self):
        client = NS(responses=NS(create=AsyncMock(return_value=response(text="A common story watch order is interpretive because of branching timelines."))))
        result = await MovieAssistant(api.movie_tools, client).respond("Give me the X-Men movies in chronological story order.", "SE")
        self.assertIn("interpretive", result.assistant_message)
        self.assertEqual(result.tool_results, [])
        self.assertIsNone(result.collection)
        self.assert_no_debug_prose(result.assistant_message)

    async def test_explicit_source_question_mentions_tmdb_naturally(self):
        result = await self.answer("Where did you get the X-Men list?")
        self.assertIn("The collection and release information come from TMDB.", result.assistant_message)
        self.assert_no_debug_prose(result.assistant_message)

    async def test_exact_release_dates_only_when_requested(self):
        result = await self.answer("Give me the X-Men collection with exact release dates.")
        self.assertIn("X-Men (2000-07-13)", result.assistant_message)
        self.assertEqual(result.collection.movies[0].release_date, "2000-07-13")

    async def test_other_collections_ambiguity_and_material_missing_data(self):
        collections = MovieCollections(fixture_collection_request)
        ambiguous = collections.search_movie_collection("Planet of the Apes")
        text = collection_response_text(ambiguous)
        self.assertIn("Which one do you mean?", text)
        self.assertIn("(Original)", text)
        self.assert_no_debug_prose(text)
        collection = collections.get_movie_collection(748, limit=1)
        self.assertIn("Showing 1 of 3 movies", collection_response_text(collection))
        unknown = collection.model_copy(deep=True)
        unknown.movies[0].release_date = None
        unknown.movies[0].release_year = None
        unknown.missing_release_dates = 1
        self.assertIn("release year unavailable", collection_response_text(unknown))
        self.assertIn("can't confirm their order", collection_response_text(unknown))


class OtherResponsePolishTests(unittest.TestCase):
    def setUp(self):
        patch.dict("os.environ", {"MOVIE_CHAT_MOCK": "true", "ENVIRONMENT": "development", "APP_ENV": "development", "NODE_ENV": "development"}).start()
        patch("api.get_watch_providers", side_effect=empty_providers).start()
        patch("requests.sessions.Session.request", side_effect=AssertionError("Offline only")).start()
        patch.object(api.movie_tools, "live_movie_runner", None).start()  # Local-only fixtures; live resolution has separate mocked coverage.
        self.addCleanup(patch.stopall)
        self.client = TestClient(api.app)

    def answer(self, text):
        with contextlib.redirect_stdout(io.StringIO()):
            response = self.client.post("/movie-chat", json={"message": text, "country_code": "SE"})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        for debug in ("get_movie_details", "get_watch_providers", "RecommendationEngine", "needs_clarification", "vote_average", "tool returned"):
            self.assertNotIn(debug, result["assistant_message"])
        return result

    def test_rating_source_label_and_search_wording(self):
        rating = self.answer("What's Interstellar's rating?")
        actual = api.movie_tools.get_movie_details(157336).movie.vote_average
        self.assertEqual(rating["assistant_message"], f"Interstellar has a TMDB rating of {actual}/10.")
        search = self.answer("Find movies called Crash.")
        self.assertIn("movies with that title:", search["assistant_message"])
        self.assertIn("Crash (1996)", search["assistant_message"])
        self.assertIn("Crash (2005)", search["assistant_message"])

    def test_provider_uses_country_and_real_tool_result(self):
        def providers(tmdb_id, country_code):
            return {**empty_providers(tmdb_id, country_code), "stream": [{"provider_id": 1, "name": "Fixture Stream", "logo_url": None}]}
        with patch("api.get_watch_providers", side_effect=providers):
            result = self.answer("Where can I watch Interstellar in Sweden?")
        self.assertEqual(result["assistant_message"], "In Sweden, you can stream Interstellar on Fixture Stream.")
        self.assertEqual(result["tool_results"][-1]["result"]["watch_providers"]["country_code"], "SE")

    def test_recommendation_prose_and_cards_match_existing_runner(self):
        with contextlib.redirect_stdout(io.StringIO()):
            expected = api.chat_recommender.recommend_from_intent(api.MovieIntent(intent="recommend", favorite_movies=["Interstellar"], country_code="SE"))
        result = self.answer("Something like Interstellar")
        self.assertEqual(result["recommendations"], expected.recommendations)
        self.assertEqual(result["assistant_message"], f"Here are {len(expected.recommendations)} movies I'd recommend:")

    def test_general_style_instructions_preserve_useful_sources_and_cap(self):
        self.assertIn("Lead with the answer", PROMPT)
        self.assertIn("TMDB rating", PROMPT)
        self.assertIn("never permit extra films", PROMPT)
        self.assertEqual(MAX_OUTPUT_TOKENS, 900)


if __name__ == "__main__":
    unittest.main()
