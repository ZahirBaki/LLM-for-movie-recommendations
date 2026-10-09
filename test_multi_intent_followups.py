"""Offline combined commands and confident title resolution; scoring is unchanged."""
import contextlib
import io
import unittest
from unittest.mock import Mock, patch
import pandas as pd
import api
import test_external_movie_embeddings as fixtures
from chat_service import MovieIntent
from chat_recommender import ChatRecommender, ResolvedMovie, MovieClarification
from movie_session import MovieSessionState, SessionStore, PendingTask, continuation_plan
from movie_tools import MovieTools
from test_movie_assistant import call, response


class FollowupTests(unittest.IsolatedAsyncioTestCase):
    setUp = fixtures.ExternalTests.setUp
    turn = fixtures.ExternalTests.turn
    recommend = fixtures.ExternalTests.recommend

    async def begin(self):
        self.client.responses.create.return_value=response(call("recommend_movies",favorite_movies=["Catalog 1 (2011)","External Film (2026)"],top_n=5))
        initial=await self.turn("I love Catalog 1 and External Film (2026). Recommend 5 movies.")
        self.assertEqual(initial.status,"success")
        self.assertEqual(len(initial.recommendations),5)
        return initial

    async def test_seen_first_two_and_more(self):
        initial=await self.begin()
        first=[m["tmdb_id"] for m in initial.recommendations]
        result=await self.turn("I've already seen the first two. Give me 5 more.")
        self.assertEqual(self.state.seen_movie_ids,first[:2])
        self.assertEqual(len(result.recommendations),5)
        self.assertFalse(set(first)&{m["tmdb_id"] for m in result.recommendations})
        self.assertEqual([m.tmdb_id for m in self.state.last_movie_list],[m["tmdb_id"] for m in result.recommendations])
        self.assertEqual(self.client.responses.create.await_count,1)

    async def test_watched_number_list_recommend_others(self):
        initial=await self.begin()
        result=await self.turn("I watched numbers 1 and 3. Recommend 5 others.")
        self.assertEqual(self.state.seen_movie_ids,[initial.recommendations[i]["tmdb_id"] for i in [0,2]])
        self.assertEqual(len(result.recommendations),5)

    async def test_exclude_last_and_show_more(self):
        initial=await self.begin()
        result=await self.turn("Exclude the last movie and show 3 more.")
        self.assertIn(initial.recommendations[-1]["tmdb_id"],self.state.excluded_movie_ids)
        self.assertEqual(self.state.seen_movie_ids,[])
        self.assertEqual(len(result.recommendations),3)

    async def test_more_without_seen_updates(self):
        await self.begin()
        result=await self.turn("Give me 5 more")
        self.assertEqual(result.status,"success")
        self.assertEqual(len(result.recommendations),5)
        self.assertEqual(self.state.seen_movie_ids,[])

    async def test_combined_exclusions_year_filters_and_shortage(self):
        initial=await self.begin()
        result=await self.turn("I've seen the first two. Give me 5 more after 2022.")
        self.assertEqual(self.state.seen_movie_ids,[m["tmdb_id"] for m in initial.recommendations[:2]])
        self.assertTrue(result.recommendations)
        self.assertLess(len(result.recommendations),5)
        self.assertTrue(all(m["year"]>2022 for m in result.recommendations))
        self.assertIn("weren't enough",result.assistant_message)
        self.assertEqual(self.state.active_preferences.min_year,2023)

    async def test_preserves_mixed_seeds_and_country_actual_engine(self):
        await self.begin()
        original=self.state.active_preferences.model_copy(deep=True)
        self.state.country_code="US"
        self.state.active_preferences.required_genres=["Drama"]
        calls=self.live.call_count
        with patch.object(self.engine,"recommend",wraps=self.engine.recommend) as engine:
            result=await self.turn("I've seen the first two. Give me 5 more")
        self.assertEqual(engine.call_args.kwargs["favorite_movies"][0]["tmdb_id"],1)
        self.assertEqual(engine.call_args.kwargs["external_queries"][0].tmdb_id,900001)
        self.assertEqual(engine.call_args.kwargs["required_genres"],["Drama"])
        self.assertEqual(self.state.active_preferences.favorite_movies,original.favorite_movies)
        self.assertEqual(result.country_code,"US")
        self.assertTrue(all(m["watch_providers"]["country_code"]=="US" for m in result.recommendations))
        self.assertEqual(self.live.call_count,calls)

    async def test_updates_exclusions_before_engine_call(self):
        initial=await self.begin()
        before=[m["tmdb_id"] for m in initial.recommendations[:2]]
        original=self.tools.recommend_movies
        def checked(*args,**kwargs):
            self.assertTrue(set(before)<=set(self.state.seen_movie_ids))
            self.assertTrue(set(before)<=set(self.state.excluded_movie_ids))
            return original(*args,**kwargs)
        with patch.object(self.tools,"recommend_movies",side_effect=checked):
            await self.turn("I've seen the first two. Give me 5 more")

    async def test_structured_card_order_and_no_repeated_titles(self):
        initial=await self.begin()
        result=await self.turn("I've seen the first two. Give me 5 more")
        numbered=[line for line in result.assistant_message.splitlines() if line[:1].isdigit()]
        self.assertEqual(numbered,[f"{i}. {movie['title']}" for i,movie in enumerate(result.recommendations,1)])
        self.assertFalse({m["tmdb_id"] for m in initial.recommendations}&{m["tmdb_id"] for m in result.recommendations})

    async def test_invalid_number_does_not_partly_mark_or_recommend(self):
        await self.begin()
        with patch.object(self.engine,"recommend",wraps=self.engine.recommend) as engine:
            result=await self.turn("I watched numbers 1 and 99. Recommend 5 others")
        self.assertEqual(result.status,"needs_clarification")
        self.assertEqual(self.state.seen_movie_ids,[])
        self.assertEqual(self.state.excluded_movie_ids,[])
        engine.assert_not_called()

    async def test_session_isolation(self):
        await self.begin()
        saved=self.state.model_copy(deep=True)
        store=SessionStore()
        async with store.transaction("a") as state:
            state.active_preferences=saved.active_preferences
            state.active_seed_movies=saved.active_seed_movies
            state.last_movie_list=saved.last_movie_list
            state.shown_movie_ids=saved.shown_movie_ids
        async with store.transaction("b") as state:
            state.active_preferences=saved.active_preferences
            state.last_movie_list=saved.last_movie_list
        async with store.transaction("a") as state:
            self.state=state
            await self.turn("I've seen the first two. Give me 5 more")
        async with store.transaction("b") as state:
            self.assertEqual(state.seen_movie_ids,[])
            self.assertEqual(state.excluded_movie_ids,[])
            self.assertEqual(state.shown_movie_ids,[])

    async def test_pending_external_clarification_still_resumes_then_combined(self):
        self.state.pending_task=PendingTask(intent=self.intent,
            clarifications=[MovieClarification(query="External Film (2026)",options=[ResolvedMovie(tmdb_id=900001,title="External Film",year=2026)])])
        result=await self.turn("the first one")
        self.assertEqual(result.status,"success")
        self.assertIsNone(self.state.pending_task)
        followup=await self.turn("I've seen the first two. Give me 5 more")
        self.assertEqual(len(followup.recommendations),5)
        self.client.responses.create.assert_not_called()

    async def test_unicode_apostrophe_and_no_matching_results(self):
        await self.begin()
        result=await self.turn("I\u2019ve seen the first two. Give me 5 more after 2090")
        self.assertEqual(result.status,"success")
        self.assertEqual(result.recommendations,[])
        self.assertIn("couldn't find",result.assistant_message)
        self.assertIn("weren't enough",result.assistant_message)

    async def test_reported_mixed_titles_and_combined_followup(self):
        self.catalog["vote_count"]=0
        self.catalog["popularity"]=0.0
        self.catalog.loc[0,["tmdb_id","title","release_year","vote_count","popularity"]]=[157336,"Interstellar",2014,41307,70.85]
        self.engine.content.movies=self.catalog.copy()
        self.raw.update(id=1248832,title="Digger")
        self.controller=ChatRecommender(self.catalog,api.run_intent_recommendation)
        self.tools=MovieTools(self.catalog,self.controller,Mock(),fixtures.empty_providers,live_movie_runner=self.live,external_embeddings=self.cache)
        self.assistant=fixtures.MovieAssistant(self.tools,self.client)
        self.client.responses.create.return_value=response(call("recommend_movies",favorite_movies=["Interstellar","Digger (2026)"],top_n=5))
        initial=await self.turn("I love Interstellar and Digger (2026). Recommend 5 movies.")
        self.assertEqual(initial.status,"success")
        self.assertEqual([movie.tmdb_id for movie in initial.resolved_movies],[157336,1248832])
        self.assertEqual(initial.clarifications,[])
        self.assertEqual([request.args[1]["query"] for request in self.live.call_args_list if request.args[0]=="/search/movie"],["Digger"])
        with patch.object(self.engine,"recommend",wraps=self.engine.recommend) as engine:
            followup=await self.turn("I've already seen the first two. Give me 5 more.")
        self.assertEqual(len(followup.recommendations),5)
        self.assertEqual(engine.call_args.kwargs["favorite_movies"][0]["tmdb_id"],157336)
        self.assertEqual(engine.call_args.kwargs["external_queries"][0].tmdb_id,1248832)

    async def test_unrecognized_combination_does_not_acknowledge_partial_completion(self):
        await self.begin()
        result=await self.turn("I've seen the first two. Give me 5 more with Actor B")
        self.assertEqual(result.status,"needs_clarification")
        self.assertEqual(self.state.seen_movie_ids,[])
        self.assertIn("both instructions",result.assistant_message)
        self.assertEqual(self.client.responses.create.await_count,1)

    def test_complete_parse_does_not_drop_unrecognized_filters(self):
        intent=MovieIntent(intent="recommend",favorite_movies=["Catalog 1"])
        self.assertIsNone(continuation_plan("I've seen the first two. Give me 5 more with Actor B",intent))
        self.assertIsNone(continuation_plan("Recommend 5 movies like Another Title",intent))


class ConfidentTitleTests(unittest.TestCase):
    def setUp(self):
        self.live=Mock(side_effect=AssertionError("Confident local match must not call TMDB"))
        self.tools=MovieTools(api.movie_tools.catalog,
            api.chat_recommender,Mock(),Mock(),live_movie_runner=self.live)

    def test_real_catalog_interstellar_auto_resolves(self):
        result=self.tools.resolve_movie_title("Interstellar")
        self.assertEqual(result.status,"success")
        self.assertEqual((result.movie.tmdb_id,result.movie.year),(157336,2014))
        self.live.assert_not_called()

    def test_exact_case_space_and_explicit_year(self):
        for title in ["interstellar"," INTERSTELLAR ","Interstellar (2014)"]:
            self.assertEqual(self.tools.resolve_movie_title(title).movie.tmdb_id,157336)
        self.live.assert_not_called()

    def test_genuine_duplicates_remain_ambiguous(self):
        result=self.tools.resolve_movie_title("Crash")
        self.assertEqual(result.status,"needs_clarification")
        self.assertEqual({movie.year for movie in result.options},{1996,2005})
        self.live.assert_not_called()

    def test_confident_duplicate_dominance_preserved(self):
        rows=pd.DataFrame([dict(tmdb_id=1,title="Example",release_year=2014,vote_count=20000,popularity=100),
            dict(tmdb_id=2,title="Example",release_year=1990,vote_count=10,popularity=1)])
        controller=ChatRecommender(rows,Mock())
        tools=MovieTools(rows,controller,Mock(),Mock(),live_movie_runner=self.live)
        self.assertEqual(tools.resolve_movie_title("Example").movie.tmdb_id,1)
        self.live.assert_not_called()

    def test_low_recognition_local_still_uses_live_fallback(self):
        rows=pd.DataFrame([dict(tmdb_id=1,title="Example",release_year=2021,vote_count=20,popularity=1)])
        live=Mock(return_value={"results":[dict(id=2,title="Example",release_date="2026-01-01")],"total_pages":1})
        tools=MovieTools(rows,ChatRecommender(rows,Mock()),Mock(),Mock(),live_movie_runner=live)
        result=tools.resolve_movie_title("Example")
        self.assertEqual(result.status,"needs_clarification")
        self.assertEqual({movie.tmdb_id for movie in result.options},{1,2})

    def test_explicit_mismatched_year_never_selects_confident_local(self):
        live=Mock(return_value={"results":[],"total_pages":1})
        self.tools.live_movie_runner=live
        result=self.tools.resolve_movie_title("Interstellar",2026)
        self.assertEqual(result.status,"not_found")
        live.assert_called_once()
