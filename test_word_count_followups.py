"""Offline regressions for complete natural-language recommendation continuations."""
import contextlib
import io
import os
import unittest
from unittest.mock import patch, Mock
import numpy as np
import pandas as pd
import api
import test_external_movie_embeddings as fixtures
from chat_recommender import ChatRecommender
from chat_service import MovieIntent
from movie_assistant import MovieAssistant
from movie_session import MovieReference, SessionStore
from movie_tools import MovieTools
from experimental_ranking import ranking_context
from test_movie_assistant import call, response

class WordCountFollowups(unittest.IsolatedAsyncioTestCase):
    turn = fixtures.ExternalTests.turn

    def setUp(self):
        fixtures.ExternalTests.setUp(self)
        patch.dict(os.environ, {"ENVIRONMENT":"development", "APP_ENV":"development", "NODE_ENV":"development", "MOVIE_EXPERIMENTAL_RANKING":"true"}).start()
        extra = pd.DataFrame([dict(tmdb_id=i,title=f"Catalog {i}",release_year=1990+i%30,genres="Drama",actors="Actor A",director="Director A",overview=f"Overview {i}") for i in range(16,46)])
        self.catalog = pd.concat([self.catalog,extra],ignore_index=True)
        self.engine.content.movies = self.catalog.copy()
        self.engine.content.overview_embeddings = np.stack([fixtures.vector(i/100) for i in range(1,46)])
        self.engine.content.build_known_values()
        self.controller = ChatRecommender(self.catalog,api.run_intent_recommendation)
        self.tools = MovieTools(self.catalog,self.controller,Mock(),fixtures.empty_providers,live_movie_runner=self.live,external_embeddings=self.cache)
        self.assistant = MovieAssistant(self.tools,self.client)

    async def begin(self, count=10):
        self.client.responses.create.side_effect = [response(call("recommend_movies",favorite_movies=["Catalog 1 (2011)"],favorite_genres=["Drama"],favorite_actors=["Actor A"],favorite_directors=["Director A"],required_genres=["Drama"],min_year=1990,max_year=2050,top_n=count)), response(text="Engine recommendations")]
        initial = await self.turn(f"Recommend {count} movies like Catalog 1 (2011)")
        self.assertEqual(initial.status,"success")
        self.assertEqual(len(initial.recommendations),count)
        return initial

    async def check_combined(self, mode):
        token = ranking_context.set((mode,"word-count-session"))
        try:
            initial = await self.begin()
            before = self.state.active_preferences.model_dump(exclude={"top_n"})
            ids = [movie["tmdb_id"] for movie in initial.recommendations]
            self.state.excluded_movie_ids.append(45)
            self.state.country_code = "US"
            before["country_code"] = "US"
            with patch.object(self.engine,"recommend",wraps=self.engine.recommend) as engine_call:
                result = await self.turn("I've already seen the first three. Give me five more.")
            self.assertEqual(result.status,"success")
            final = [movie["tmdb_id"] for movie in result.recommendations]
            self.assertEqual(len(final),5)
            self.assertEqual(len(set(final)),5)
            self.assertFalse(set(final) & set(ids+[1,45]))
            self.assertEqual(self.state.seen_movie_ids,ids[:3])
            self.assertTrue(set(ids[:3]+[45]) <= set(self.state.excluded_movie_ids))
            self.assertEqual(self.state.active_preferences.model_dump(exclude={"top_n"}),before)
            self.assertEqual(self.state.active_preferences.top_n,5)
            self.assertEqual([movie.tmdb_id for movie in self.state.last_movie_list],final)
            self.assertEqual(self.state.last_list_kind,"recommendations")
            self.assertEqual(result.assistant_message.splitlines()[2:],[f"{index}. {movie['title']}" for index,movie in enumerate(result.recommendations,1)])
            self.assertTrue(all(movie["watch_providers"]["country_code"] == "US" for movie in result.recommendations))
            self.assertEqual(ranking_context.get()[0],mode)
            self.assertEqual(self.client.responses.create.await_count,2)
            # Compare the returned cards with the same unchanged engine ordering.
            from experimental_ranking import excluded_context
            blocked = tuple(set(ids+[45]))
            exclusion_token = excluded_context.set(blocked)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    ranked = self.engine.recommend(**engine_call.call_args.kwargs)
            finally:
                excluded_context.reset(exclusion_token)
            expected = [int(identity) for identity in ranked.tmdb_id if identity not in set(ids+[1,45])][:5]
            self.assertEqual(final,expected)
            # New ordinals must refer to the new list, not the preceding ten.
            await self.turn("I've seen the first one.")
            self.assertIn(final[0],self.state.seen_movie_ids)
        finally:
            ranking_context.reset(token)

    async def test_original_ten_then_word_five(self):
        await self.check_combined("baseline")

    async def test_experimental_ten_then_word_five(self):
        await self.check_combined("genre_votes")

    async def test_exclusion_only_latest_list(self):
        initial = await self.begin()
        with patch.object(self.engine,"recommend",wraps=self.engine.recommend) as engine_call:
            result = await self.turn("Don't recommend these again.")
        self.assertEqual(result.recommendations,[])
        self.assertEqual(self.state.excluded_movie_ids,[movie["tmdb_id"] for movie in initial.recommendations])
        self.assertEqual(len(self.state.last_movie_list),10)
        engine_call.assert_not_called()
        self.assertEqual(self.client.responses.create.await_count,2)

    async def test_word_five_more_without_exclusions(self):
        initial = await self.begin()
        result = await self.turn("Give me five more.")
        self.assertEqual(len(result.recommendations),5)
        self.assertFalse({movie["tmdb_id"] for movie in initial.recommendations} & {movie["tmdb_id"] for movie in result.recommendations})
        self.assertEqual(self.state.seen_movie_ids,[])

    async def test_nonconsecutive_numbers(self):
        initial = await self.begin()
        result = await self.turn("I watched numbers 2 and 4. Recommend three others.")
        self.assertEqual(len(result.recommendations),3)
        self.assertEqual(self.state.seen_movie_ids,[initial.recommendations[index]["tmdb_id"] for index in (1,3)])

    async def test_remove_and_new_movies(self):
        initial = await self.begin()
        result = await self.turn("Remove the first one and give me five new movies.")
        self.assertEqual(len(result.recommendations),5)
        self.assertEqual(self.state.seen_movie_ids,[])
        self.assertIn(initial.recommendations[0]["tmdb_id"],self.state.excluded_movie_ids)

    async def test_missing_ambiguous_and_search_context_do_not_mutate(self):
        await self.begin()
        for text,kind in [("I've seen the first 99. Give me five more.","recommendations"),("I've already watched those two. Show me something else.","recommendations"),("I've seen the first three. Give me five more.","search")]:
            self.state.last_list_kind = kind
            before = self.state.model_dump()
            with patch.object(self.engine,"recommend",wraps=self.engine.recommend) as engine_call:
                result = await self.turn(text)
            self.assertEqual(result.status,"needs_clarification")
            self.assertEqual(self.state.model_dump(),before)
            engine_call.assert_not_called()
        self.state.last_movie_list = []
        result = await self.turn("I've seen the first three. Give me five more.")
        self.assertEqual(result.status,"needs_clarification")

    async def test_unambiguous_those_two(self):
        initial = await self.begin(2)
        result = await self.turn("I've already watched those two. Show me something else.")
        self.assertEqual(len(result.recommendations),2)
        self.assertEqual(self.state.seen_movie_ids,[movie["tmdb_id"] for movie in initial.recommendations])

    async def test_unknown_combined_filters_never_acknowledge_partial_action(self):
        await self.begin()
        result = await self.turn("I've seen the first three. Give me five more with Actor B")
        self.assertEqual(result.status,"needs_clarification")
        self.assertEqual(self.state.seen_movie_ids,[])

    async def test_endpoint_preserves_mode_and_returns_five_cards(self):
        store = SessionStore()
        session_id = "00000000-0000-4000-8000-000000000001"
        await self.begin()
        async with store.transaction(session_id) as state:
            state.__dict__.update(self.state.model_copy(deep=True).__dict__)
        async def responder(message,country,controller,tools,*,session):
            return await self.assistant.respond(message,country,session=session)
        with patch.object(api,"session_store",store),patch.object(api,"movie_tools",self.tools),patch.object(api,"chat_recommender",self.controller),patch.object(api,"respond_to_movie_chat",side_effect=responder):
            result = await api.movie_chat(api.MovieChatRequest(message="I've seen the first three. Give me five more.",session_id=session_id,ranking_mode="genre_votes"))
        payload = result.model_dump(mode="json")
        self.assertEqual(len(payload["recommendations"]),5)
        self.assertTrue(all(movie["tmdb_id"] and movie["poster_url"] is None for movie in payload["recommendations"]))
        self.assertEqual(ranking_context.get(),("baseline",None))
        self.assertEqual(self.client.responses.create.await_count,2)

if __name__ == "__main__":
    unittest.main()
