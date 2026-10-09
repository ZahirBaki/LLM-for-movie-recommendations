"""Offline scope routing; movie examples live only in tests, not application data."""
import contextlib
import io
import unittest
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch
import api
from movie_assistant import MovieAssistant
from movie_guardrails import is_broad_franchise_request, is_collection_request, is_research_request
from research_test_fixtures import research_response
from test_movie_assistant import call, response
from test_movie_collections import fixture_collection_request
from test_chat_recommender import empty_providers


class FranchiseRoutingTests(unittest.IsolatedAsyncioTestCase):
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

    async def test_broad_scope_cannot_be_narrowed(self):
        messages = ['Give me all X-Men movies in release order.',
            'I want all the movies of xmen universe including deadpool for example in order',
            'Give me the entire X-Men universe including Wolverine and Deadpool.',
            'Give me all X-Men movies including Deadpool, Wolverine and New Mutants.',
            'Give me the whole X-Men universe in chronological order.',
            'What order should I watch the whole X-Men franchise?',
            'Give me X-Men, Wolverine and Deadpool movies in order.',
            'Every movie related to the X-Men universe.', 'all MCU movies',
            'entire Batman movie franchise', 'all Spider-Man movies including Venom',
            'whole Conjuring universe', 'all Star Wars movies including spin-offs']
        for message in messages:
            with self.subTest(message=message):
                text = 'A general movie-knowledge list includes Wolverine and Deadpool; story order is interpretive.'
                research = is_research_request(message)
                result = await self.answer(message, research_response(text) if research else response(text=text))
                self.assertTrue(result.assistant_message.startswith(text))
                self.assertEqual(result.tool_results, [])
                self.assertIsNone(result.collection)
                self.assertEqual(result.recommendations, [])
                self.assertEqual(self.client.responses.create.call_args.kwargs['tool_choice'], {'type': 'web_search'} if research else 'none')
                self.assertEqual(self.client.responses.create.call_args.kwargs['tools'], [{'type': 'web_search', 'search_context_size': 'low'}] if research else [])
                self.assertIn("Answer the user's requested list NOW", self.client.responses.create.call_args.kwargs['instructions'])
                self.assertIn('do not promise a future lookup' if research else 'Do not promise to search', self.client.responses.create.call_args.kwargs['instructions'])
                self.assertTrue(is_broad_franchise_request(message))
                self.assertFalse(is_collection_request(message))

    async def test_explicit_collection_keeps_verified_backend_data(self):
        for message in ['Give me the X-Men collection in release order.',
                        'Give me movies in the TMDB X-Men collection in release order.']:
            result = await self.answer(message, response(call('search_movie_collection', query='X-Men')),
                response(call('get_movie_collection', collection_id=748)))
            self.assertEqual(result.collection.source, 'TMDB')
            self.assertEqual([m.tmdb_id for m in result.collection.movies], [36657, 49538, 320288])
            self.assertEqual(self.client.responses.create.call_args_list[0].kwargs['tool_choice'], {'type': 'function', 'name': 'search_movie_collection'})

    async def test_collection_chronology_is_not_backend_chronology(self):
        result = await self.answer('Give me the X-Men collection in chronological order.',
            response(text='A commonly used interpretive story order has conflicting timelines; it is not TMDB chronology.'))
        self.assertEqual(result.tool_results, [])
        self.assertIsNone(result.collection)
        self.assertIn('interpretive', result.assistant_message)

    async def test_personalized_requests_still_rank_with_engine(self):
        for message, favorites in [
            ('Give me movies similar to X-Men.', ['X-Men']),
            ('I watched X-Men, Logan and Deadpool. Recommend something.', ['X-Men', 'Logan', 'Deadpool']),
            ('I love X-Men and Deadpool. Recommend something I would like.', ['X-Men', 'Deadpool']),
            ('Give me movies similar to Deadpool.', ['Deadpool']),
            ('I watched Logan, Deadpool and X-Men. What should I watch next?', ['Logan', 'Deadpool', 'X-Men'])]:
            with patch.object(api.engine, 'recommend', wraps=api.engine.recommend) as engine:
                result = await self.answer(message, response(call('recommend_movies', favorite_movies=favorites)), response(text='Here are your recommendations.'))
            engine.assert_called_once()
            self.assertTrue(result.recommendations)
            self.assertEqual(self.client.responses.create.call_args_list[0].kwargs['tool_choice'], {'type': 'function', 'name': 'recommend_movies'})

    async def test_authoritative_facts_and_provider_tools_remain_available(self):
        for message, title, tmdb_id, tool in [
            ("What's Deadpool's TMDB rating?", 'Deadpool', 293660, 'get_movie_details'),
            ('Where can I watch Deadpool in Sweden?', 'Deadpool', 293660, 'get_watch_providers'),
            ('Where can I watch Logan in Sweden?', 'Logan', 263115, 'get_watch_providers')]:
            result = await self.answer(message, response(call('resolve_movie_title', title=title)),
                response(call(tool, tmdb_id=tmdb_id)), response(text='Grounded answer.'))
            self.assertEqual(result.tool_results[-1]['tool'], tool)
            self.assertEqual(self.client.responses.create.call_args_list[0].kwargs['tool_choice'], 'auto')

    async def test_general_questions_and_long_form_protections(self):
        result = await self.answer('What is machine learning?', response(text='Learning patterns from data.'))
        self.assertEqual(result.status, 'success')
        result = await self.answer('Write a 2000-word essay about X-Men.')
        self.assertEqual(result.openai_requests, 0)
        self.assertEqual(result.status, 'unsupported_intent')

    async def test_unexpected_collection_call_on_broad_route_rejected(self):
        result = await self.answer('Give me all X-Men movies including Deadpool.', response(call('search_movie_collection', query='X-Men')))
        self.assertEqual(result.status, 'unresolved')
        self.assertEqual(result.tool_results, [])
