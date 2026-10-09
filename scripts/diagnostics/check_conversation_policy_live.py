"""Opt-in four-case policy check. No retries; excluded from test discovery."""
import contextlib
import io
import os
import sys
from unittest.mock import patch
from fastapi.testclient import TestClient
import backend.api.app as api


def main(essay_only=False):
    # Year explicitly selected by the user; unqualified Dune is ambiguous.
    cases = ["What is machine learning?",
             "What is the chronological order of the X-Men movies?",
             "I watched Interstellar, Arrival and Dune (2020). What should I watch next?",
             "Write me a 2000-word essay about World War II."]
    count = 0
    with patch.dict(os.environ, {"MOVIE_CHAT_MOCK": "false"}), TestClient(api.app) as client:
        for index, message in enumerate(cases):
            if essay_only and index != 3:
                continue
            with patch.object(api.engine, "recommend", wraps=api.engine.recommend) as ranking:
                with contextlib.redirect_stdout(io.StringIO()):
                    response = client.post("/movie-chat", json={"message": message, "country_code": "SE"})
                data = response.json()
                count += data.get("openai_requests", 0)
                print(message)
                print(f"HTTP {response.status_code}; status={data.get('status')}; OpenAI requests={data.get('openai_requests', 0)}")
                print(data.get("assistant_message"))
                print("Tools:", [item["tool"] for item in data.get("tool_results", [])])
                print("Engine calls:", ranking.call_count, "Cards:", [(item["tmdb_id"], item["title"]) for item in data.get("recommendations", [])])
                if response.status_code != 200 or data.get("status") == "error":
                    print(f"Stopped; {count} completed requests (failed attempts may add one).")
                    return 1
                if index < 2:
                    assert data["status"] == "success" and not data["tool_results"] and not data["recommendations"]
                elif index == 2:
                    ranking.assert_called_once()
                    assert data["status"] == "success" and data["recommendations"]
                    assert any(item["tool"] == "recommend_movies" for item in data["tool_results"])
                else:
                    assert data["status"] == "unsupported_intent" and data["openai_requests"] == 0
                    assert "outline" in data["assistant_message"]
    print(f"Actual OpenAI requests: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(essay_only="--essay-only" in sys.argv))
