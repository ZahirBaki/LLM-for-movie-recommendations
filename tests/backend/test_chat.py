"""Offline HTTP/SDK boundary tests. Run: python -m unittest tests.backend.test_chat -v."""

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from fastapi.testclient import TestClient
from openai import AsyncOpenAI
from pydantic import ValidationError

import backend.api.app as api
import backend.services.chat_service as chat_service
from backend.services.chat_service import ChatServiceError, MovieIntent


class ChatTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)

    def test_blank_messages_do_not_call_openai(self):
        with patch('backend.api.app.parse_movie_message', new_callable=AsyncMock) as parser:
            for message in ("", "   ", "\n\t"):
                self.assertEqual(self.client.post("/chat", json={"message": message}).status_code, 422)
            parser.assert_not_awaited()

    def test_response_and_no_recommender_call(self):
        parsed = MovieIntent(intent="recommend", favorite_movies=["Interstellar"], required_actors=["Ryan Gosling"], min_year=2016)
        with patch('backend.api.app.parse_movie_message', new_callable=AsyncMock, return_value=parsed), patch.object(api.engine, "recommend") as recommend:
            result = self.client.post("/chat", json={"message": "Example"})
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.json(), {"message": "Example", "parsed": parsed.model_dump()})
            recommend.assert_not_called()

    def test_http_errors_are_sanitized(self):
        for error, status, detail in (
            (ChatServiceError(503, "Chat unavailable"), 503, "Chat unavailable"),
            (RuntimeError("sensitive internal text"), 500, "An unexpected chat error occurred."),
        ):
            with patch('backend.api.app.parse_movie_message', new_callable=AsyncMock, side_effect=error):
                result = self.client.post("/chat", json={"message": "Example"})
                self.assertEqual(result.status_code, status)
                self.assertEqual(result.json(), {"detail": detail})

    def test_defaults_and_constraints(self):
        self.assertEqual(MovieIntent().favorite_movies, [])
        self.assertIsNone(MovieIntent().top_n)
        for values in ({"intent": "invented"}, {"min_year": 2016, "max_year": 2009}, {"top_n": 0}, {"country_code": "Sweden"}):
            with self.assertRaises(ValidationError):
                MovieIntent(**values)

    def test_existing_read_endpoints(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/genres").status_code, 200)
        schema = self.client.get("/openapi.json").json()
        self.assertIn("/recommend", schema["paths"])
        self.assertIn("/chat", schema["paths"])


class ServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_key(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            with self.assertRaises(ChatServiceError) as caught:
                await chat_service.parse_movie_message("Recommend a comedy")
        self.assertEqual(caught.exception.status_code, 503)

    async def test_real_sdk_structured_parse(self):
        expected = MovieIntent(intent="recommend", required_genres=["Science Fiction"])

        def respond(request):
            payload = json.loads(request.content)
            self.assertEqual(payload["model"], chat_service.OPENAI_MODEL)
            self.assertFalse(payload["store"])
            schema = payload["text"]["format"]["schema"]
            self.assertFalse(schema["additionalProperties"])
            self.assertEqual(set(schema["required"]), set(schema["properties"]))
            return httpx.Response(200, json={
                "id": "resp_test", "object": "response", "created_at": 0,
                "status": "completed", "model": chat_service.OPENAI_MODEL,
                "output": [{"id": "msg_test", "type": "message", "role": "assistant", "status": "completed",
                            "content": [{"type": "output_text", "text": expected.model_dump_json(), "annotations": []}]}],
                "parallel_tool_calls": False, "tool_choice": "auto", "tools": [],
            })

        sdk = AsyncOpenAI(api_key="offline-test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
        with patch.dict(os.environ, {"OPENAI_API_KEY": "offline-test"}), patch('backend.services.chat_service.AsyncOpenAI', return_value=sdk):
            self.assertEqual(await chat_service.parse_movie_message("Recommend a science fiction movie"), expected)

    async def test_upstream_failures(self):
        for kind, expected_status in (("auth", 503), ("quota", 503), ("rate", 429), ("network", 503), ("timeout", 504), ("api", 502)):
            def respond(request):
                if kind == "network":
                    raise httpx.ConnectError("sensitive", request=request)
                if kind == "timeout":
                    raise httpx.ReadTimeout("sensitive", request=request)
                status = {"auth": 401, "quota": 429, "rate": 429, "api": 500}[kind]
                code = "insufficient_quota" if kind == "quota" else "test_error"
                return httpx.Response(status, json={"error": {"message": "sensitive", "type": "test", "code": code}})
            sdk = AsyncOpenAI(api_key="offline-test", max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
            with patch.dict(os.environ, {"OPENAI_API_KEY": "offline-test"}), patch('backend.services.chat_service.AsyncOpenAI', return_value=sdk):
                with self.assertRaises(ChatServiceError) as caught:
                    await chat_service.parse_movie_message("Recommend a comedy")
                self.assertEqual(caught.exception.status_code, expected_status)
                self.assertNotIn("sensitive", caught.exception.detail)

    async def test_refusal_incomplete_and_unexpected(self):
        for output in (SimpleNamespace(status="completed", output_parsed=None), SimpleNamespace(status="incomplete", output_parsed=None), RuntimeError("sensitive")):
            client = AsyncMock()
            client.__aenter__.return_value = client
            if isinstance(output, Exception):
                client.responses.parse.side_effect = output
            else:
                client.responses.parse.return_value = output
            with patch.dict(os.environ, {"OPENAI_API_KEY": "offline-test"}), patch('backend.services.chat_service.AsyncOpenAI', return_value=client):
                with self.assertRaises(ChatServiceError) as caught:
                    await chat_service.parse_movie_message("Recommend a comedy")
                self.assertEqual(caught.exception.status_code, 500 if isinstance(output, Exception) else 502)

    async def test_invalid_structured_output(self):
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.responses.parse.return_value = SimpleNamespace(status="completed", output_parsed={"intent": "invalid"})
        with patch.dict(os.environ, {"OPENAI_API_KEY": "offline-test"}), patch('backend.services.chat_service.AsyncOpenAI', return_value=client):
            with self.assertRaises(ChatServiceError) as caught:
                await chat_service.parse_movie_message("Recommend a comedy")
            self.assertEqual(caught.exception.status_code, 502)


if __name__ == "__main__":
    unittest.main()
