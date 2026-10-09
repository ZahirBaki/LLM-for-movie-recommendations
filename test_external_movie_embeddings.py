"""Offline runtime queries: real scoring, mocked TMDB/encoder/OpenAI."""
import contextlib
import io
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, AsyncMock, patch
import numpy as np
import pandas as pd
import api
from chat_service import MovieIntent
from chat_recommender import ChatRecommender, ResolvedMovie, MovieClarification
from content_recommender import ContentRecommender
from recommendation_engine import RecommendationEngine
from runtime_movie_embeddings import RuntimeMovieEmbeddings, ExternalEmbeddingError, validate_embedding
from movie_tools import MovieTools, MovieDetails
from movie_chat import MovieChatResponse
from movie_session import MovieSessionState, handle_session_turn, PendingTask
from movie_assistant import MovieAssistant
from test_movie_assistant import call, response
from test_chat_recommender import empty_providers


def vector(angle=0):
    result=np.zeros(384,dtype=np.float32)
    result[:2]=[np.cos(angle),np.sin(angle)]
    return result


def raw_movie(identity=900001, overview="  A disaster threatens a powerful man.  "):
    return dict(id=identity,title="External Film",release_date="2026-10-02",overview=overview,
        genres=[dict(name="Drama")],credits=dict(crew=[dict(job="Director",name="Director A")],cast=[dict(name="Actor A")]))


class ExternalTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch.dict("os.environ",{"MOVIE_CHAT_MOCK":"false"}).start()
        patch("requests.sessions.Session.request",side_effect=AssertionError("offline tests")).start()
        self.encoder=Mock()
        self.encoder.encode.side_effect=lambda texts,**kwargs:np.stack([vector() for text in texts])
        self.loader=Mock(return_value=self.encoder)
        self.cache=RuntimeMovieEmbeddings(model_loader=self.loader)
        self.catalog=pd.DataFrame([dict(tmdb_id=i,title=f"Catalog {i}",release_year=2010+i,
            genres="Drama",actors="Actor A",director="Director A",overview=f"Overview {i}") for i in range(1,16)])
        content=ContentRecommender.__new__(ContentRecommender)
        content.movies=self.catalog.copy()
        content.overview_embeddings=np.stack([vector(i/20) for i in range(1,16)])
        content.build_known_values()
        self.engine=RecommendationEngine.__new__(RecommendationEngine)
        self.engine.content=content
        self.engine.svd=Mock(mapping=pd.DataFrame(columns=["tmdb_id","movieId"]),movie_to_index={})
        patch.object(api,"engine",self.engine).start()
        patch("api.get_watch_providers",side_effect=empty_providers).start()
        self.controller=ChatRecommender(self.catalog,api.run_intent_recommendation)
        self.raw=raw_movie()
        self.live=Mock(side_effect=lambda path,params:{"results":[self.raw],"total_pages":1} if path=="/search/movie" else self.raw)
        self.tools=MovieTools(self.catalog,self.controller,Mock(),empty_providers,live_movie_runner=self.live,external_embeddings=self.cache)
        self.intent=MovieIntent(intent="recommend",favorite_movies=["External Film (2026)"],top_n=5,country_code="SE")
        self.details=MovieDetails(tmdb_id=900001,title="External Film",release_year=2026,overview=self.raw["overview"],genres="Drama",director="Director A",actors="Actor A")
        self.state=MovieSessionState()
        self.client=NS(responses=NS(create=AsyncMock()))
        self.assistant=MovieAssistant(self.tools,self.client)

    def recommend(self,intent=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.tools.recommend_movies(intent or self.intent)

    async def turn(self,message):
        async def responder(message,country,controller,tools,*,session):
            return await self.assistant.respond(message,country,session=session)
        with contextlib.redirect_stdout(io.StringIO()):
            return await handle_session_turn(message,"SE",self.state,self.controller,self.tools,responder)

    def test_live_identity_is_verified(self):
        result=self.tools.resolve_movie_title("External Film",2026)
        self.assertEqual(result.movie.tmdb_id,900001)
        self.assertEqual(result.details.overview,self.raw["overview"])

    def test_runtime_overview_and_model_lazy_reuse(self):
        self.loader.assert_not_called()
        query=self.cache.get(self.details)
        self.assertEqual(query.embedding.shape,(384,))
        self.encoder.encode.assert_called_once_with([self.raw["overview"]],convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False)
        self.cache.get(self.details)
        self.loader.assert_called_once()
        self.encoder.encode.assert_called_once()

    def test_normalization(self):
        query=self.cache.get(self.details)
        self.assertAlmostEqual(float(np.linalg.norm(query.embedding)),1)
        self.assertFalse(query.embedding.flags.writeable)

    def test_actual_engine_catalog_candidates_and_cards(self):
        with patch.object(self.engine,"recommend",wraps=self.engine.recommend) as engine:
            result=self.recommend()
        self.assertEqual(result.status,"success")
        engine.assert_called_once()
        self.assertEqual(engine.call_args.kwargs["external_queries"][0].tmdb_id,900001)
        self.assertEqual(len(result.recommendations),5)
        self.assertTrue({m["tmdb_id"] for m in result.recommendations}<=set(self.catalog.tmdb_id))
        self.assertTrue(all(m["engine_mode"]=="CONTENT - external movie query" for m in result.recommendations))

    def test_external_seed_excluded_even_when_catalog_contains_id(self):
        query=self.cache.get(self.details)
        self.engine.content.movies.loc[0,"tmdb_id"]=900001
        result=self.engine.content.get_scores(external_queries=[query])
        self.assertNotIn(900001,result.tmdb_id.tolist())

    def test_no_fabricated_collaborative_scores(self):
        result=self.recommend()
        self.engine.svd.get_scores_for_favorite_tmdb_movies.assert_not_called()
        self.assertTrue(all(m["svd_score"] is None for m in result.recommendations))

    def test_mixed_local_external_content_blend(self):
        self.intent.favorite_movies=["Catalog 1 (2011)","External Film (2026)"]
        query=self.cache.get(self.details)
        with patch("content_recommender.np.dot",wraps=np.dot) as dot:
            result=self.recommend()
        self.assertEqual(result.status,"success")
        self.assertEqual(dot.call_args.args[0].shape,(2,384))
        self.assertEqual({m.tmdb_id for m in result.resolved_movies},{1,900001})
        self.assertNotIn(1,[m["tmdb_id"] for m in result.recommendations])

    def test_mixed_two_svd_local_seeds_still_use_content_without_external_svd(self):
        self.engine.svd.mapping=pd.DataFrame([dict(tmdb_id=i,movieId=i) for i in [1,2]])
        self.engine.svd.movie_to_index={1:0,2:1}
        self.intent.favorite_movies=["Catalog 1 (2011)","Catalog 2 (2012)","External Film (2026)"]
        self.assertEqual(self.recommend().status,"success")
        self.engine.svd.get_scores_for_favorite_tmdb_movies.assert_not_called()

    def test_local_only_ranking_unchanged(self):
        local=MovieIntent(intent="recommend",favorite_movies=["Catalog 1 (2011)"],top_n=5,country_code="SE")
        before=self.recommend(local).recommendations
        self.recommend()
        after=self.recommend(local).recommendations
        self.assertEqual(before,after)

    def test_missing_overview(self):
        self.raw["overview"]="  "
        result=self.recommend()
        self.assertEqual(result.status,"needs_preferences")
        self.assertIn("no usable",result.message)
        self.assertEqual(result.recommendations,[])
        self.loader.assert_not_called()

    def test_tmdb_failure_never_ranks_partial_local_seed(self):
        self.live.side_effect=RuntimeError("unavailable")
        self.intent.favorite_movies=["Catalog 1 (2011)","External Film (2026)"]
        with patch.object(self.engine,"recommend",wraps=self.engine.recommend) as engine:
            result=self.recommend()
        self.assertEqual(result.status,"error")
        self.assertEqual(result.recommendations,[])
        engine.assert_not_called()

    def test_model_unavailable_no_generic_fallback(self):
        self.loader.side_effect=RuntimeError("model absent")
        result=self.recommend()
        self.assertEqual(result.status,"needs_preferences")
        self.assertIn("model is unavailable",result.message)
        self.assertEqual(result.recommendations,[])
        self.recommend()
        self.loader.assert_called_once()

    def test_wrong_dimensions(self):
        self.encoder.encode.return_value=None
        self.encoder.encode.side_effect=lambda *args,**kwargs:np.ones((1,768))
        result=self.recommend()
        self.assertEqual(result.status,"needs_preferences")
        self.assertIn("dimension",result.message)
        self.assertEqual(result.recommendations,[])

    def test_nonfinite_zero_and_unnormalized_vectors_rejected(self):
        for raw in [np.full(384,np.nan),np.zeros(384),np.ones(384),np.full(384,np.inf)]:
            with self.assertRaises(ExternalEmbeddingError):validate_embedding(raw,384)

    def test_cache_reuses_metadata_and_avoids_tmdb(self):
        first=self.recommend()
        calls=self.live.call_count
        second=self.recommend()
        self.assertEqual(first.recommendations,second.recommendations)
        self.assertEqual(self.live.call_count,calls)
        self.encoder.encode.assert_called_once()

    def test_bounded_cache_lru_eviction(self):
        cache=RuntimeMovieEmbeddings(model_loader=self.loader,max_entries=2)
        for identity in [1,2]:cache.get(self.details.model_copy(update={"tmdb_id":identity}))
        cache.cached(1)
        cache.get(self.details.model_copy(update={"tmdb_id":3}))
        self.assertIsNone(cache.cached(2))
        self.assertIsNotNone(cache.cached(1))
        self.assertIsNotNone(cache.cached(3))

    def test_cache_expiry_and_metadata_refresh(self):
        now=[0]
        cache=RuntimeMovieEmbeddings(model_loader=self.loader,ttl=5,clock=lambda:now[0])
        cache.get(self.details)
        cache.get(self.details.model_copy(update={"overview":"Different verified overview"}))
        self.assertEqual(self.encoder.encode.call_count,2)
        now[0]=6
        self.assertIsNone(cache.cached(900001))

    def test_model_reproduces_stored_embedding_probe(self):
        cache=RuntimeMovieEmbeddings(model_loader=self.loader,compatibility_samples=[("Stored overview",vector())])
        cache.get(self.details)
        self.assertEqual(self.encoder.encode.call_args_list[0].args[0],["Stored overview"])
        bad=RuntimeMovieEmbeddings(model_loader=self.loader,compatibility_samples=[("Stored overview",vector(1))])
        with self.assertRaisesRegex(ExternalEmbeddingError,"does not reproduce"):bad.get(self.details)

    async def test_pending_external_resume_and_seen(self):
        self.state.pending_task=PendingTask(intent=self.intent,watched=True,
            clarifications=[MovieClarification(query="External Film (2026)",options=[ResolvedMovie(tmdb_id=900001,title="External Film",year=2026)])])
        result=await self.turn("the first one")
        self.assertEqual(result.status,"success")
        self.assertIsNone(self.state.pending_task)
        self.assertIn(900001,self.state.seen_movie_ids)
        self.assertEqual(len(result.recommendations),5)
        self.client.responses.create.assert_not_called()

    async def test_more_after_external_seed_and_seen_excluded_filter(self):
        self.client.responses.create.return_value=response(call("recommend_movies",favorite_movies=["External Film (2026)"],top_n=5))
        first=await self.turn("I watched External Film (2026). Recommend similar movies.")
        self.assertEqual(first.status,"success")
        ids={m["tmdb_id"] for m in first.recommendations}
        self.state.excluded_movie_ids.append(10)
        self.state.seen_movie_ids.append(11)
        count=self.live.call_count
        second=await self.turn("Give me 5 more")
        self.assertEqual(second.status,"success")
        self.assertFalse((ids|{10,11,900001}) & {m["tmdb_id"] for m in second.recommendations})
        self.assertEqual(self.live.call_count,count)
        self.assertEqual(self.client.responses.create.await_count,1)
        self.assertNotIn("embedding",self.state.model_dump_json())

    async def test_structured_text_matches_actual_cards_no_extra_model_text(self):
        self.client.responses.create.return_value=response(call("recommend_movies",favorite_movies=["External Film (2026)"],top_n=5))
        result=await self.turn("Recommend movies like External Film (2026)")
        self.assertEqual(result.status,"success")
        lines=[line for line in result.assistant_message.splitlines() if line[:1].isdigit()]
        self.assertEqual(lines,[f"{i}. {movie['title']}" for i,movie in enumerate(result.recommendations,1)])
        self.assertTrue(all(movie["watch_providers"]["country_code"]=="SE" for movie in result.recommendations))
        self.assertEqual(self.client.responses.create.await_count,1)

    def test_catalog_and_embeddings_not_mutated(self):
        catalog=self.engine.content.movies.copy(deep=True)
        embeddings=self.engine.content.overview_embeddings.copy()
        self.recommend()
        pd.testing.assert_frame_equal(catalog,self.engine.content.movies)
        np.testing.assert_array_equal(embeddings,self.engine.content.overview_embeddings)

    def test_hard_filters_preserved_for_external_query(self):
        self.intent.min_year=2020
        self.intent.max_year=2023
        self.intent.required_genres=["Drama"]
        result=self.recommend()
        self.assertEqual(result.status,"success")
        self.assertTrue(result.recommendations)
        self.assertTrue(all(2020<=movie["year"]<=2023 for movie in result.recommendations))
        self.intent.exclude_directors=["Director A"]
        result=self.recommend()
        self.assertEqual(result.recommendations,[])
        self.assertIn("couldn't find",result.message)

    def test_concurrent_cache_requests_initialize_and_encode_once(self):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=8) as pool:
            results=list(pool.map(lambda _:self.cache.get(self.details),range(8)))
        self.assertTrue(all(query is results[0] for query in results))
        self.loader.assert_called_once()
        self.encoder.encode.assert_called_once()
