"""Offline current-turn vs session-reference routing regressions using real tools."""
import contextlib
import io
import unittest
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

import api
from chat_service import MovieIntent
from movie_assistant import MovieAssistant
from movie_chat import MovieChatResponse
from movie_session import MovieReference, MovieSessionState, handle_session_turn
from test_chat_recommender import empty_providers
from test_movie_assistant import call, response

REPORTED = "i have watched coco, inside out and zootopia i want movies similar to those"
TITLES = ["Coco", "Inside Out", "Zootopia"]
IDS = [354912, 150540, 269149]


class RecommendationReferenceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        patch("api.get_watch_providers", side_effect=empty_providers).start()
        patch("requests.sessions.Session.request", side_effect=AssertionError("Offline only")).start()
        patch.object(api.movie_tools, "live_movie_runner", None).start()  # Local-only fixtures; live resolution has separate mocked coverage.
        self.addCleanup(patch.stopall)
        self.state = MovieSessionState()
        self.client = NS(responses=NS(create=AsyncMock()))
        self.assistant = MovieAssistant(api.movie_tools, self.client)

    async def turn(self, message, titles=None):
        if titles is not None:
            self.client.responses.create.side_effect = [
                response(call("recommend_movies", favorite_movies=titles, top_n=5)),
                response(text="Here are the engine's recommendations.")]

        async def responder(message, country, controller, tools, *, session):
            return await self.assistant.respond(message, country, session=session)

        with contextlib.redirect_stdout(io.StringIO()):
            return await handle_session_turn(message, "SE", self.state,
                api.chat_recommender, api.movie_tools, responder)

    async def check_explicit(self, message):
        with patch.object(api.engine, "recommend", wraps=api.engine.recommend) as engine:
            result = await self.turn(message, TITLES)
        self.assertEqual(result.status, "success")
        self.assertEqual([(m.tmdb_id, m.year) for m in result.resolved_movies],
                         [(354912, 2017), (150540, 2015), (269149, 2016)])
        engine.assert_called_once()
        self.assertEqual([m["tmdb_id"] for m in engine.call_args.kwargs["favorite_movies"]], IDS)
        self.assertEqual(result.recommendations, result.tool_results[0]["result"]["recommendations"])
        self.assertTrue(result.recommendations)
        self.assertFalse(set(IDS) & {m["tmdb_id"] for m in result.recommendations})
        arguments = self.client.responses.create.call_args_list[-2].kwargs
        self.assertEqual(arguments["tool_choice"], {"type": "function", "name": "recommend_movies"})
        return result

    async def test_exact_reported_message(self):
        await self.check_explicit(REPORTED)

    async def test_capitalization_and_punctuation(self):
        await self.check_explicit("I have watched Coco, Inside Out and Zootopia. I want movies similar to those!")

    async def test_recommend_similar_movies_sentence(self):
        await self.check_explicit("I watched Coco, Inside Out and Zootopia. Recommend similar movies.")

    async def test_what_next(self):
        await self.check_explicit("I watched Coco, Inside Out and Zootopia. What next?")

    async def test_three_movies_what_should_watch_next(self):
        await self.check_explicit("I have watched three movies: Coco, Inside Out and Zootopia. What should I watch next?")

    async def test_named_two_movie_request(self):
        with patch.object(api.engine, "recommend", wraps=api.engine.recommend) as engine:
            result = await self.turn("Recommend movies similar to Coco and Zootopia.", ["Coco", "Zootopia"])
        self.assertEqual([m.tmdb_id for m in result.resolved_movies], [354912, 269149])
        engine.assert_called_once()

    async def test_plural_similar_with_previous_list(self):
        self.state.last_movie_list = [MovieReference(tmdb_id=identity, title=title, year=year)
            for identity, title, year in zip(IDS, TITLES, [2017, 2015, 2016])]
        for pronoun in ("those", "them", "these"):
            with self.subTest(pronoun=pronoun):
                self.state.last_movie_list = [MovieReference(tmdb_id=identity, title=title, year=year)
                    for identity, title, year in zip(IDS, TITLES, [2017, 2015, 2016])]
                with patch.object(api.engine, "recommend", wraps=api.engine.recommend) as engine:
                    result = await self.turn(f"Give me movies similar to {pronoun}")
                self.assertEqual(result.status, "success")
                self.assertEqual([m.tmdb_id for m in result.resolved_movies], IDS)
                engine.assert_called_once()
        self.client.responses.create.assert_not_awaited()

    async def test_plural_without_context(self):
        before = self.state.model_dump()
        with patch.object(api.engine, "recommend") as engine:
            result = await self.turn("Give me movies similar to those")
        self.assertEqual(result.status, "needs_clarification")
        self.assertEqual(self.state.model_dump(), before)
        engine.assert_not_called()
        self.client.responses.create.assert_not_awaited()

    async def test_plural_with_filters_keeps_controller_extraction(self):
        self.state.last_movie_list = [MovieReference(tmdb_id=354912, title="Coco", year=2017)]
        self.client.responses.create.side_effect = [response(call("recommend_movies",
            favorite_movies=["Coco (2017)"], required_actors=["Ryan Gosling"], min_year=2016)), response()]
        result = await self.turn("Give me movies similar to those with Ryan Gosling after 2015")
        self.assertEqual(result.parsed_intent.required_actors, ["Ryan Gosling"])
        self.assertEqual(result.parsed_intent.min_year, 2016)
        self.assertEqual(result.resolved_movies[0].tmdb_id, 354912)

    async def test_ambiguous_singular_reference(self):
        self.state.last_movie_list = [MovieReference(tmdb_id=i, title=t) for i, t in zip(IDS, TITLES)]
        result = await self.turn("Tell me about that one")
        self.assertEqual(result.status, "needs_clarification")
        self.assertEqual(result.tool_results, [])
        self.client.responses.create.assert_not_awaited()

    async def test_new_favorites_override_unrelated_context(self):
        self.state.active_preferences = MovieIntent(intent="recommend", favorite_movies=["Interstellar"],
            required_genres=["Horror"], min_year=2020)
        self.state.current_movie = MovieReference(tmdb_id=157336, title="Interstellar", year=2014)
        self.state.last_movie_list = [self.state.current_movie]
        self.state.shown_movie_ids = [157336]
        result = await self.check_explicit(REPORTED)
        self.assertEqual(self.state.active_preferences.favorite_movies, TITLES)
        self.assertEqual(self.state.active_preferences.required_genres, [])
        self.assertIsNone(self.state.active_preferences.min_year)
        self.assertNotIn(157336, self.state.shown_movie_ids)
        context = self.client.responses.create.call_args_list[-2].kwargs["input"][0]["content"]
        self.assertNotIn("Interstellar", context)
        self.assertEqual(self.state.last_movie_list[0].tmdb_id, result.recommendations[0]["tmdb_id"])

    async def test_duplicate_title_specific_clarification(self):
        with patch.object(api.engine, "recommend") as engine:
            result = await self.turn("I watched Crash. Recommend movies similar to that.", ["Crash"])
        self.assertEqual(result.status, "needs_clarification")
        self.assertEqual(result.clarifications[0].query, "Crash")
        self.assertEqual({m.tmdb_id for m in result.clarifications[0].options}, {884, 1640})
        self.assertNotEqual(result.assistant_message, "Which movie do you mean? Choose a title or a number from the last list.")
        engine.assert_not_called()
        self.assertEqual(self.state.seen_movie_ids, [])

    async def test_session_update_more_and_ordinal(self):
        first = await self.check_explicit(REPORTED)
        self.assertEqual(self.state.seen_movie_ids, IDS)
        self.assertEqual(self.state.excluded_movie_ids, IDS)
        first_ids = {m["tmdb_id"] for m in first.recommendations}
        self.assertEqual(set(self.state.shown_movie_ids), first_ids)
        more = await self.turn("5 more")
        self.assertEqual(more.status, "success")
        self.assertEqual(self.state.active_preferences.favorite_movies, TITLES)
        self.assertFalse((first_ids | set(IDS)) & {m["tmdb_id"] for m in more.recommendations})
        ordinal = await self.turn("Where can I watch the second one in Sweden?")
        self.assertEqual(ordinal.resolved_movies[0].tmdb_id, more.recommendations[1]["tmdb_id"])


if __name__ == "__main__":
    unittest.main()
