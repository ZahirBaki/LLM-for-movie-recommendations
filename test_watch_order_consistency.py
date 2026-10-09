"""Mocked watch-order policy/route checks; membership examples are test-only.

These verify instructions and controller behavior, not model-generated accuracy.
The opt-in live check separately evaluates the actual generated release sequence.
"""
import contextlib
import io
import re
import unittest
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch
import api
from movie_assistant import MovieAssistant
from test_movie_assistant import call, response
from test_movie_collections import fixture_collection_request
from test_chat_recommender import empty_providers
from research_test_fixtures import research_response

LOTR = [("The Lord of the Rings: The Fellowship of the Ring", 2001),
        ("The Lord of the Rings: The Two Towers", 2002),
        ("The Lord of the Rings: The Return of the King", 2003)]
HOBBIT = [("The Hobbit: An Unexpected Journey", 2012),
          ("The Hobbit: The Desolation of Smaug", 2013),
          ("The Hobbit: The Battle of the Five Armies", 2014)]


def movie_list(label, films):
    return label + ':\n' + '\n'.join(f'{i}. {title} ({year})' for i, (title, year) in enumerate(films, 1))


class WatchOrderConsistencyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        patch('api.get_watch_providers', side_effect=empty_providers).start()
        patch('api.request_tmdb_collection', side_effect=fixture_collection_request).start()
        patch('requests.sessions.Session.request', side_effect=AssertionError('Offline only')).start()
        patch.object(api.movie_tools, "live_movie_runner", None).start()  # Local-only fixtures; live resolution has separate mocked coverage.
        self.addCleanup(patch.stopall)
        self.client = NS(responses=NS(create=AsyncMock()))
        self.assistant = MovieAssistant(api.movie_tools, self.client)

    async def answer(self, message, *script):
        self.client.responses.create.reset_mock()
        self.client.responses.create.side_effect = script
        with contextlib.redirect_stdout(io.StringIO()):
            return await self.assistant.respond(message, 'SE')

    def assert_policy_sent(self):
        instructions = ' '.join(self.client.responses.create.call_args_list[0].kwargs['instructions'].split())
        for policy in ['release dates/years run from earliest to latest',
                       'never put later-released prequels first',
                       'must not be labelled release order',
                       'recommended watch order is editorial',
                       "Default ambiguous 'in order' to release order",
                       'do not silently expand it into a larger universe']:
            self.assertIn(policy, instructions)

    async def test_requested_trilogy_release_order_and_default_scope(self):
        for message in ['Give me the LOTR movies in release order.', 'Give me the LOTR movies in order.']:
            text = movie_list('Here they are in release order', LOTR)
            result = await self.answer(message, response(text=text))
            self.assert_policy_sent()
            self.assertEqual(re.findall(r'\((\d{4})\)', result.assistant_message), ['2001', '2002', '2003'])
            self.assertNotIn('Hobbit', result.assistant_message)
            self.assertEqual(result.tool_results, [])
            self.assertEqual(self.client.responses.create.call_args.kwargs['tool_choice'], 'none')

    async def test_expanded_release_order_follows_release_years(self):
        result = await self.answer('Give me LOTR and The Hobbit in release order.', response(text=movie_list('Release order', LOTR + HOBBIT)))
        self.assert_policy_sent()
        self.assertEqual(re.findall(r'\((\d{4})\)', result.assistant_message), ['2001', '2002', '2003', '2012', '2013', '2014'])
        self.assertEqual(result.tool_results, [])

    async def test_story_order_is_distinct_from_release_order(self):
        result = await self.answer('Give me LOTR and The Hobbit in chronological/story order.', response(text=movie_list('Story chronology', HOBBIT + LOTR)))
        self.assert_policy_sent()
        self.assertTrue(result.assistant_message.startswith('Story chronology'))
        self.assertNotIn('release order', result.assistant_message.lower())
        self.assertEqual(re.findall(r'\((\d{4})\)', result.assistant_message), ['2012', '2013', '2014', '2001', '2002', '2003'])
        self.assertEqual(self.client.responses.create.call_args.kwargs['tool_choice'], 'none')

    async def test_broad_middle_earth_and_xmen_keep_general_path(self):
        for message, text in [
            ('Give me all Middle-earth movies in release order.', movie_list('Release order', [('The Hobbit', 1977)] + LOTR + HOBBIT)),
            ('Give me all X-Men universe movies including Deadpool in release order.', movie_list('Release order', [('X-Men', 2000), ('Deadpool', 2016), ('Deadpool & Wolverine', 2024)]))]:
            result = await self.answer(message, research_response(text))
            self.assert_policy_sent()
            years = re.findall(r'\((\d{4})\)', result.assistant_message)
            self.assertEqual(years, sorted(years))
            self.assertIsNone(result.collection)
            self.assertEqual(self.client.responses.create.call_args.kwargs['tool_choice'], {'type': 'web_search'})

    async def test_xmen_chronology_retains_general_knowledge(self):
        result = await self.answer('Give me X-Men movies in chronological order.', response(text='An interpretive story timeline has conflicting branches.'))
        self.assert_policy_sent()
        self.assertEqual(result.tool_results, [])
        self.assertIsNone(result.collection)
        self.assertEqual(self.client.responses.create.call_args.kwargs['tool_choice'], 'none')

    async def test_verified_collection_keeps_backend_date_sorting(self):
        result = await self.answer('Give me the X-Men collection in release order.',
            response(call('search_movie_collection', query='X-Men')), response(call('get_movie_collection', collection_id=748)))
        self.assert_policy_sent()
        dates = [m.release_date for m in result.collection.movies]
        self.assertEqual(dates, sorted(dates))
        self.assertEqual(result.collection.source, 'TMDB')

    async def test_lotr_preferences_still_use_actual_engine(self):
        for message, titles in [
            ('I watched LOTR and The Hobbit. Recommend something.', [LOTR[0][0], HOBBIT[0][0]]),
            ('I like LOTR and The Hobbit. Recommend movies I would like.', [LOTR[0][0], HOBBIT[0][0]]),
            ('Give me movies similar to Fellowship of the Ring.', [LOTR[0][0]])]:
            with patch.object(api.engine, 'recommend', wraps=api.engine.recommend) as engine:
                result = await self.answer(message, response(call('recommend_movies', favorite_movies=titles)), response(text='Here are your recommendations.'))
            engine.assert_called_once()
            self.assertTrue(result.recommendations)
            self.assertEqual(self.client.responses.create.call_args_list[0].kwargs['tool_choice'], {'type': 'function', 'name': 'recommend_movies'})

    async def test_long_form_still_blocked_before_openai(self):
        result = await self.answer('Write a 2000-word essay about Lord of the Rings.')
        self.assertEqual(result.openai_requests, 0)
        self.assertEqual(result.status, 'unsupported_intent')
        self.client.responses.create.assert_not_awaited()
