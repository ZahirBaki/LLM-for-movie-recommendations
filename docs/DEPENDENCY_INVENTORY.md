# Python dependency inventory

All original modules and standalone entry points are retained. Imports inside functions are included; unused bindings removed after review are separately recorded in UNUSED_IMPORT_CANDIDATES.json.

| Original module | Current source | Local package imports |
|---|---|---|
| `analyze_content_model.py` | `evaluation/scripts/analyze_content_model.py` | `backend.core.paths`, `backend.recommendation.content_recommender` |
| `api.py` | `backend/api/app.py` | `backend.core.movie_guardrails`, `backend.core.paths`, `backend.recommendation.experimental_ranking`, `backend.recommendation.recommendation_engine`, `backend.recommendation.runtime_movie_embeddings`, `backend.services.chat_recommender`, `backend.services.chat_service`, `backend.services.movie_chat`, `backend.services.movie_tools`, `backend.sessions.movie_session` |
| `app.py` | `scripts/legacy/streamlit_app.py` | `backend.recommendation.recommendation_engine` |
| `build_catalog.py` | `scripts/data/build_catalog.py` | `backend.core.paths` |
| `build_embeddings.py` | `scripts/data/build_embeddings.py` | `backend.core.paths` |
| `build_movie_mapping.py` | `scripts/data/build_movie_mapping.py` | `backend.core.paths` |
| `build_svd_model.py` | `scripts/data/build_svd_model.py` | `backend.recommendation.svd_recommender` |
| `chat_recommender.py` | `backend/services/chat_recommender.py` | `backend.services.chat_service` |
| `chat_service.py` | `backend/services/chat_service.py` |  |
| `check_chat_live.py` | `scripts/diagnostics/check_chat_live.py` | `backend.api.app` |
| `check_conversation_policy_live.py` | `scripts/diagnostics/check_conversation_policy_live.py` | `backend.api.app` |
| `check_experimental_ranking_browser.py` | `tests/browser/check_experimental_ranking_browser.py` |  |
| `check_franchise_routing_live.py` | `scripts/diagnostics/check_franchise_routing_live.py` | `backend.api.app` |
| `check_human_evaluation_browser.py` | `tests/browser/check_human_evaluation_browser.py` | `evaluation.scripts.human_evaluation` |
| `check_movie_assistant_live.py` | `scripts/diagnostics/check_movie_assistant_live.py` | `backend.api.app` |
| `check_movie_chat_browser.py` | `tests/browser/check_movie_chat_browser.py` | `backend.api.app`, `backend.recommendation.runtime_movie_embeddings`, `backend.services.chat_recommender`, `backend.services.chat_service`, `backend.services.movie_chat`, `backend.sessions.movie_session`, `tests.backend.test_movie_collections` |
| `check_movie_chat_presentation_browser.py` | `tests/browser/check_movie_chat_presentation_browser.py` |  |
| `check_movie_guardrails_live.py` | `scripts/diagnostics/check_movie_guardrails_live.py` | `backend.api.app`, `backend.services.movie_collections` |
| `check_movie_research_live.py` | `scripts/diagnostics/check_movie_research_live.py` | `backend.api.app` |
| `check_svd_favorites.py` | `scripts/diagnostics/check_svd_favorites.py` | `backend.core.paths` |
| `check_watch_order_live.py` | `scripts/diagnostics/check_watch_order_live.py` | `backend.api.app` |
| `check_word_count_followups_browser.py` | `tests/browser/check_word_count_followups_browser.py` |  |
| `collaborative.py` | `scripts/legacy/collaborative_demo.py` | `backend.core.paths` |
| `compare_models.py` | `evaluation/scripts/compare_models.py` | `backend.core.paths`, `backend.recommendation.hybrid_recommender` |
| `compare_ranking.py` | `evaluation/scripts/compare_ranking.py` | `backend.core.paths`, `backend.recommendation.experimental_ranking`, `backend.recommendation.recommendation_engine`, `evaluation.scripts.recommendation_quality` |
| `content_recommender.py` | `backend/recommendation/content_recommender.py` | `backend.core.paths`, `backend.recommendation.runtime_movie_embeddings` |
| `diagnose_movie_research.py` | `scripts/diagnostics/diagnose_movie_research.py` | `backend.core.paths`, `backend.integrations.movie_research`, `backend.services.chat_service` |
| `evaluate_cold_start.py` | `evaluation/scripts/evaluate_cold_start.py` | `backend.core.paths`, `backend.recommendation.content_recommender` |
| `evaluate_collaborative.py` | `evaluation/scripts/evaluate_collaborative.py` | `backend.core.paths` |
| `evaluate_collaborative_v2.py` | `evaluation/scripts/evaluate_collaborative_v2.py` | `backend.core.paths` |
| `evaluate_hybrid.py` | `evaluation/scripts/evaluate_hybrid.py` | `backend.core.paths`, `backend.recommendation.hybrid_recommender` |
| `evaluate_hybrid_v3.py` | `evaluation/scripts/evaluate_hybrid_v3.py` | `backend.core.paths`, `backend.recommendation.content_recommender` |
| `evaluate_popularity.py` | `evaluation/scripts/evaluate_popularity.py` | `backend.core.paths` |
| `evaluate_svd.py` | `evaluation/scripts/evaluate_svd.py` | `backend.core.paths` |
| `experimental_ranking.py` | `backend/recommendation/experimental_ranking.py` | `backend.core.paths` |
| `human_evaluation.py` | `evaluation/scripts/human_evaluation.py` | `backend.core.paths` |
| `hybrid_recommender.py` | `backend/recommendation/hybrid_recommender.py` | `backend.recommendation.content_recommender`, `backend.recommendation.svd_recommender` |
| `main.py` | `scripts/legacy/tmdb_cli.py` | `backend.core.paths` |
| `matrix_factorization.py` | `scripts/legacy/matrix_factorization.py` | `backend.core.paths` |
| `movie_assistant.py` | `backend/services/movie_assistant.py` | `backend.core.movie_guardrails`, `backend.integrations.movie_research`, `backend.services.chat_recommender`, `backend.services.chat_service`, `backend.services.movie_chat`, `backend.services.movie_collections`, `backend.sessions.movie_session` |
| `movie_chat.py` | `backend/services/movie_chat.py` | `backend.core.movie_guardrails`, `backend.integrations.movie_research`, `backend.services.chat_recommender`, `backend.services.chat_service`, `backend.services.movie_assistant`, `backend.services.movie_collections`, `backend.sessions.movie_session` |
| `movie_collections.py` | `backend/services/movie_collections.py` | `backend.services.chat_recommender` |
| `movie_guardrails.py` | `backend/core/movie_guardrails.py` |  |
| `movie_research.py` | `backend/integrations/movie_research.py` | `backend.core.movie_guardrails`, `backend.services.chat_service` |
| `movie_session.py` | `backend/sessions/movie_session.py` | `backend.core.movie_guardrails`, `backend.recommendation.experimental_ranking`, `backend.services.chat_recommender`, `backend.services.chat_service`, `backend.services.movie_chat` |
| `movie_tools.py` | `backend/services/movie_tools.py` | `backend.recommendation.runtime_movie_embeddings`, `backend.services.chat_recommender`, `backend.services.chat_service`, `backend.services.movie_collections` |
| `recommendation_engine.py` | `backend/recommendation/recommendation_engine.py` | `backend.recommendation.content_recommender`, `backend.recommendation.experimental_ranking`, `backend.recommendation.svd_recommender` |
| `recommendation_quality.py` | `evaluation/scripts/recommendation_quality.py` | `backend.core.paths`, `backend.recommendation.content_recommender`, `backend.recommendation.experimental_ranking`, `backend.recommendation.recommendation_engine`, `backend.recommendation.runtime_movie_embeddings`, `backend.services.chat_recommender`, `backend.services.chat_service`, `backend.services.movie_tools`, `evaluation.scripts.recommendation_quality_report` |
| `recommendation_quality_report.py` | `evaluation/scripts/recommendation_quality_report.py` |  |
| `recommender.py` | `scripts/legacy/content_demo.py` | `backend.core.paths` |
| `research_test_fixtures.py` | `tests/backend/research_test_fixtures.py` |  |
| `runtime_movie_embeddings.py` | `backend/recommendation/runtime_movie_embeddings.py` |  |
| `svd_recommender.py` | `backend/recommendation/svd_recommender.py` | `backend.core.paths` |
| `test_chat.py` | `tests/backend/test_chat.py` | `backend.api.app`, `backend.services.chat_service` |
| `test_chat_recommender.py` | `tests/backend/test_chat_recommender.py` | `backend.api.app`, `backend.services.chat_recommender`, `backend.services.chat_service` |
| `test_conversation_policy.py` | `tests/backend/test_conversation_policy.py` | `backend.api.app`, `backend.core.movie_guardrails`, `backend.services.movie_assistant`, `tests.backend.test_chat_recommender`, `tests.backend.test_movie_assistant` |
| `test_experimental_ranking.py` | `tests/backend/test_experimental_ranking.py` | `backend.api.app`, `backend.recommendation.experimental_ranking`, `backend.services.movie_chat`, `evaluation.scripts.recommendation_quality`, `tests.backend.test_external_movie_embeddings`, `tests.backend.test_movie_assistant` |
| `test_external_movie_embeddings.py` | `tests/backend/test_external_movie_embeddings.py` | `backend.api.app`, `backend.recommendation.content_recommender`, `backend.recommendation.recommendation_engine`, `backend.recommendation.runtime_movie_embeddings`, `backend.services.chat_recommender`, `backend.services.chat_service`, `backend.services.movie_assistant`, `backend.services.movie_tools`, `backend.sessions.movie_session`, `tests.backend.test_chat_recommender`, `tests.backend.test_movie_assistant` |
| `test_franchise_routing.py` | `tests/backend/test_franchise_routing.py` | `backend.api.app`, `backend.core.movie_guardrails`, `backend.services.movie_assistant`, `tests.backend.research_test_fixtures`, `tests.backend.test_chat_recommender`, `tests.backend.test_movie_assistant`, `tests.backend.test_movie_collections` |
| `test_human_evaluation.py` | `tests/backend/test_human_evaluation.py` | `evaluation.scripts.human_evaluation` |
| `test_live_movie_resolution.py` | `tests/backend/test_live_movie_resolution.py` | `backend.api.app`, `backend.recommendation.recommendation_engine`, `backend.services.chat_recommender`, `backend.services.chat_service`, `backend.services.movie_assistant`, `backend.services.movie_tools`, `backend.sessions.movie_session`, `tests.backend.test_chat_recommender`, `tests.backend.test_movie_assistant` |
| `test_movie_assistant.py` | `tests/backend/test_movie_assistant.py` | `backend.api.app`, `backend.services.movie_assistant`, `tests.backend.test_chat_recommender` |
| `test_movie_chat.py` | `tests/backend/test_movie_chat.py` | `backend.api.app`, `backend.services.chat_service`, `backend.services.movie_chat`, `tests.backend.test_chat_recommender` |
| `test_movie_collections.py` | `tests/backend/test_movie_collections.py` | `backend.api.app`, `backend.core.movie_guardrails`, `backend.services.chat_recommender`, `backend.services.movie_assistant`, `backend.services.movie_collections`, `tests.backend.test_movie_assistant` |
| `test_movie_research.py` | `tests/backend/test_movie_research.py` | `backend.api.app`, `backend.core.movie_guardrails`, `backend.integrations.movie_research`, `backend.services.movie_assistant`, `tests.backend.research_test_fixtures`, `tests.backend.test_chat_recommender`, `tests.backend.test_movie_assistant`, `tests.backend.test_movie_collections` |
| `test_movie_session.py` | `tests/backend/test_movie_session.py` | `backend.api.app`, `backend.services.chat_recommender`, `backend.services.chat_service`, `backend.services.movie_assistant`, `backend.services.movie_chat`, `backend.services.movie_tools`, `backend.sessions.movie_session`, `tests.backend.test_chat_recommender`, `tests.backend.test_movie_assistant`, `tests.backend.test_movie_collections` |
| `test_movie_tools.py` | `tests/backend/test_movie_tools.py` | `backend.api.app`, `backend.services.chat_recommender`, `backend.services.chat_service`, `backend.services.movie_tools`, `tests.backend.test_chat_recommender` |
| `test_multi_intent_followups.py` | `tests/backend/test_multi_intent_followups.py` | `backend.api.app`, `backend.services.chat_recommender`, `backend.services.chat_service`, `backend.services.movie_tools`, `backend.sessions.movie_session`, `tests.backend.test_external_movie_embeddings`, `tests.backend.test_movie_assistant` |
| `test_pending_clarification.py` | `tests/backend/test_pending_clarification.py` | `backend.api.app`, `backend.core.paths`, `backend.services.chat_recommender`, `backend.services.chat_service`, `backend.services.movie_chat`, `backend.services.movie_tools`, `backend.sessions.movie_session`, `tests.backend.test_chat_recommender` |
| `test_project_hail_mary.py` | `tests/backend/test_project_hail_mary.py` | `backend.recommendation.recommendation_engine` |
| `test_recommendation_quality.py` | `tests/backend/test_recommendation_quality.py` | `evaluation.scripts.recommendation_quality`, `evaluation.scripts.recommendation_quality_report`, `tests.backend.test_external_movie_embeddings` |
| `test_recommendation_references.py` | `tests/backend/test_recommendation_references.py` | `backend.api.app`, `backend.services.chat_service`, `backend.services.movie_assistant`, `backend.sessions.movie_session`, `tests.backend.test_chat_recommender`, `tests.backend.test_movie_assistant` |
| `test_research_diagnostics.py` | `tests/backend/test_research_diagnostics.py` | `backend.integrations.movie_research` |
| `test_response_polish.py` | `tests/backend/test_response_polish.py` | `backend.api.app`, `backend.core.movie_guardrails`, `backend.services.movie_assistant`, `backend.services.movie_collections`, `tests.backend.test_chat_recommender`, `tests.backend.test_movie_assistant`, `tests.backend.test_movie_collections` |
| `test_watch_order_consistency.py` | `tests/backend/test_watch_order_consistency.py` | `backend.api.app`, `backend.services.movie_assistant`, `tests.backend.research_test_fixtures`, `tests.backend.test_chat_recommender`, `tests.backend.test_movie_assistant`, `tests.backend.test_movie_collections` |
| `test_word_count_followups.py` | `tests/backend/test_word_count_followups.py` | `backend.api.app`, `backend.recommendation.experimental_ranking`, `backend.services.chat_recommender`, `backend.services.movie_assistant`, `backend.services.movie_tools`, `backend.sessions.movie_session`, `tests.backend.test_external_movie_embeddings`, `tests.backend.test_movie_assistant` |
| `tune_hybrid_weights.py` | `evaluation/scripts/tune_hybrid_weights.py` | `backend.core.paths`, `backend.recommendation.hybrid_recommender` |
| `tune_svd.py` | `evaluation/scripts/tune_svd.py` | `backend.core.paths` |
