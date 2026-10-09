"""Offline pending-task identity and recommendation regressions."""
import contextlib
import io
import unittest
from unittest.mock import AsyncMock, Mock, patch
import pandas as pd
from chat_service import MovieIntent
from chat_recommender import ChatRecommender, MovieClarification, ResolvedMovie
from movie_tools import MovieTools
from movie_chat import MovieChatResponse
from movie_session import MovieSessionState, SessionStore, handle_session_turn


class PendingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch("requests.sessions.Session.request", side_effect=AssertionError("offline only")).start()
        patch.dict("os.environ", {"MOVIE_CHAT_MOCK": "false"}).start()
        rows = [dict(tmdb_id=36657, title="X-Men", release_year=2000),
                dict(tmdb_id=24428, title="The Avengers", release_year=2012),
                dict(tmdb_id=9320, title="The Avengers", release_year=1998)]
        self.runner = Mock(return_value=[dict(tmdb_id=i, title=f"Card {i}", release_year=2020) for i in range(100,120)])
        self.controller = ChatRecommender(pd.DataFrame(rows), self.runner)
        self.tools = MovieTools(pd.DataFrame(rows), self.controller, Mock(), Mock())
        self.state = MovieSessionState()
        self.intent = MovieIntent(intent="recommend", favorite_movies=["X Men", "Avengers"], top_n=5,
            required_genres=["Action"], min_year=2001, country_code="SE")
        self.responder = AsyncMock(side_effect=self.respond)

    async def respond(self, message, country, controller, tools, session=None):
        if message == "Who directed Inception?":
            return MovieChatResponse(status="success", country_code=country, assistant_message="Christopher Nolan.")
        result = self.controller.recommend_from_intent(self.intent.model_copy(update={"favorite_movies": ["X Men", "The Avengers"]}))
        result.clarifications[0].query = "Avengers"
        return MovieChatResponse(**result.model_dump(), parsed_intent=self.intent, assistant_message="Which Avengers?")

    async def turn(self, message):
        return await handle_session_turn(message, "SE", self.state, self.controller, self.tools, self.responder)

    async def begin(self):
        result = await self.turn("I want a movie similar to X Men and Avengers")
        self.assertEqual(result.status, "needs_clarification")
        self.runner.assert_not_called()
        self.assertEqual(self.state.pending_task.resolved_movies[0].tmdb_id,36657)

    async def test_exact_title_and_year_resume_all_seeds_and_filters(self):
        await self.begin()
        result = await self.turn("The Avengers (2012)")
        self.assertEqual(result.status,"success")
        self.assertEqual([m["tmdb_id"] for m in self.runner.call_args.kwargs["favorite_movies"]],[36657,24428])
        self.assertEqual(self.runner.call_args.kwargs["required_genres"],["Action"])
        self.assertEqual(self.runner.call_args.kwargs["min_year"],2001)
        self.assertEqual(len(result.recommendations),5)
        self.assertIn("1. Card 100",result.assistant_message)
        self.assertIsNone(self.state.pending_task)
        self.assertEqual(self.responder.await_count,1)

    async def test_exact_unique_title(self):
        await self.begin()
        next(m for m in self.state.pending_task.clarifications[0].options if m.tmdb_id == 9320).title="Avengers"
        result=await self.turn("The Avengers")
        self.assertEqual(result.status,"success")
        self.assertEqual([m.tmdb_id for m in result.resolved_movies],[36657,24428])

    async def test_year_reply(self):
        await self.begin()
        self.assertEqual((await self.turn("2012")).status,"success")

    async def test_ordinal_reply(self):
        await self.begin()
        self.assertEqual((await self.turn("the first one")).status,"success")

    async def test_both_valid(self):
        await self.begin()
        result=await self.turn("both")
        self.assertEqual({m.tmdb_id for m in result.resolved_movies},{36657,24428,9320})

    async def test_invalid_then_valid(self):
        await self.begin()
        result=await self.turn("number 99")
        self.assertEqual(result.status,"needs_clarification")
        self.assertEqual(result.clarifications[0].query,"Avengers")
        self.runner.assert_not_called()
        self.assertEqual((await self.turn("2012")).status,"success")

    async def test_duplicate_bare_title_stays_ambiguous(self):
        await self.begin()
        self.assertEqual((await self.turn("The Avengers")).status,"needs_clarification")
        self.runner.assert_not_called()

    async def test_new_topic_clears(self):
        await self.begin()
        result=await self.turn("Actually forget that. Who directed Inception?")
        self.assertEqual(result.assistant_message,"Christopher Nolan.")
        self.assertIsNone(self.state.active_preferences)
        self.assertIsNone(self.state.pending_task)
        self.runner.assert_not_called()

    async def test_new_recommendation_overrides(self):
        await self.begin()
        self.intent.favorite_movies=["X Men"]
        self.responder.side_effect=None
        self.responder.return_value=MovieChatResponse(status="success", country_code="SE", assistant_message="New request.")
        await self.turn("Recommend movies like X Men")
        self.assertIsNone(self.state.pending_task)
        self.assertEqual(self.responder.await_count,2)

    async def test_ttl(self):
        await self.begin()
        now=[0];store=SessionStore(ttl=5,clock=lambda:now[0])
        async with store.transaction("a") as state: state.pending_task=self.state.pending_task
        now[0]=6
        async with store.transaction("a") as state: self.assertIsNone(state.pending_task)

    async def test_isolation(self):
        await self.begin()
        store=SessionStore()
        async with store.transaction("a") as state: state.pending_task=self.state.pending_task
        async with store.transaction("b") as state: self.assertIsNone(state.pending_task)
        async with store.transaction("a") as state: self.assertIsNotNone(state.pending_task)

    async def test_more_preserves_seeds(self):
        await self.begin();await self.turn("2012")
        result=await self.turn("5 more")
        self.assertEqual(len(result.recommendations),5)
        self.assertEqual([m["tmdb_id"] for m in self.runner.call_args.kwargs["favorite_movies"]],[36657,24428])
        self.assertTrue(all(m["tmdb_id"]>=105 for m in result.recommendations))

    async def test_seen_first_two(self):
        await self.begin();await self.turn("2012")
        await self.turn("I have seen the first two")
        self.assertEqual(self.state.seen_movie_ids,[100,101])
        result=await self.turn("5 more")
        self.assertFalse({100,101}&{m["tmdb_id"] for m in result.recommendations})

    async def test_live_external_not_substituted_or_ranked(self):
        await self.begin()
        task=self.state.pending_task
        task.clarifications[0].options=[ResolvedMovie(tmdb_id=1248832,title="Digger",year=2026)]
        self.tools.live_movie_runner=Mock(side_effect=lambda path,params: {"results":[dict(id=1248832,title="Digger",release_date="2026-09-28")],"total_pages":1} if path=="/search/movie" else dict(id=1248832,title="Digger",release_date="2026-09-28",overview="Verified movie"))
        result=await self.turn("Digger 2026")
        self.assertEqual(result.status,"needs_preferences")
        self.assertIn("cannot score",result.assistant_message)
        self.assertEqual(result.recommendations,[])
        self.assertIn(1248832,[m.tmdb_id for m in result.resolved_movies])
        self.runner.assert_not_called()

    async def test_unambiguous_request_does_not_create_pending(self):
        self.responder.side_effect=None
        intent=self.intent.model_copy(update={"favorite_movies":["X Men"]})
        output=self.tools.recommend_movies(intent)
        self.responder.return_value=MovieChatResponse(**output.model_dump(),parsed_intent=intent,assistant_message="Cards")
        result=await self.turn("Recommend movies like X Men")
        self.assertEqual(result.status,"success")
        self.assertIsNone(self.state.pending_task)

    async def test_actual_engine_receives_both_ids(self):
        import api
        from test_chat_recommender import empty_providers
        self.tools=api.movie_tools;self.controller=api.chat_recommender
        patch.object(self.tools,"live_movie_runner",None).start()
        patch("api.get_watch_providers",side_effect=empty_providers).start()
        self.responder.side_effect=None
        self.responder.return_value=MovieChatResponse(status="needs_clarification",country_code="SE",assistant_message="Which?",parsed_intent=self.intent,
            resolved_movies=[ResolvedMovie(tmdb_id=36657,title="X-Men",year=2000)],
            clarifications=[MovieClarification(query="Avengers",options=[ResolvedMovie(tmdb_id=24428,title="The Avengers",year=2012),ResolvedMovie(tmdb_id=9320,title="The Avengers",year=1998)])])
        await self.begin()
        with patch.object(api.engine,"recommend",wraps=api.engine.recommend) as engine, contextlib.redirect_stdout(io.StringIO()):
            result=await self.turn("The Avengers (2012)")
        self.assertEqual(result.status,"success")
        self.assertEqual([m["tmdb_id"] for m in engine.call_args.kwargs["favorite_movies"]],[36657,24428])


    async def test_multiple_ambiguities_preserve_partial_answer(self):
        await self.begin()
        from movie_session import PendingTask
        self.state.pending_task.intent.favorite_movies.append("Another")
        self.state.pending_task.clarifications.append(MovieClarification(query="Another", options=[ResolvedMovie(tmdb_id=36657,title="X-Men",year=2000)]))
        self.assertEqual((await self.turn("2012")).status,"needs_clarification")
        self.runner.assert_not_called()
        self.assertEqual(self.state.pending_task.clarifications[0].query,"Another")
        result=await self.turn("the first one")
        self.assertEqual(result.status,"success")
        self.assertEqual({m.tmdb_id for m in result.resolved_movies},{36657,24428})

    async def test_failure_keeps_pending_for_retry(self):
        await self.begin()
        self.runner.side_effect=RuntimeError("temporary failure")
        self.assertEqual((await self.turn("2012")).status,"error")
        self.assertIsNotNone(self.state.pending_task)
        self.runner.side_effect=None
        self.assertEqual((await self.turn("2012")).status,"success")

    async def test_resolved_external_seed_preserves_id_during_other_clarification(self):
        await self.begin()
        task=self.state.pending_task
        task.intent.favorite_movies[0]="Digger"
        task.resolved_movies=[ResolvedMovie(tmdb_id=1248832,title="Digger",year=2026)]
        self.tools.live_movie_runner=Mock(side_effect=lambda path, params: dict(id=1248832,title="Digger",release_date="2026-09-28"))
        result=await self.turn("2012")
        self.assertEqual(result.status,"needs_preferences")
        self.assertEqual({m.tmdb_id for m in result.resolved_movies},{1248832,24428})
        self.assertEqual(result.recommendations,[])
        self.assertEqual([call.args[0] for call in self.tools.live_movie_runner.call_args_list],["/movie/1248832"])
        self.runner.assert_not_called()

    async def test_diagnostic_import_does_not_execute_engine(self):
        import runpy
        with patch("recommendation_engine.RecommendationEngine") as engine:
            module=runpy.run_path("test_project_hail_mary.py",run_name="diagnostic_import_test")
        engine.assert_not_called()
        self.assertTrue(callable(module["main"]))
