"""Optional five-message live check; never imported by normal tests."""
import contextlib
import io
import os
from unittest.mock import patch
from fastapi.testclient import TestClient
import api

CASES = ["How are you?", "What's Interstellar's rating?", "Who directed Interstellar?",
         "Recommend something like Interstellar.", "Where can I watch Interstellar in Sweden?"]


def main():
    requests = 0
    with patch.dict(os.environ, {"MOVIE_CHAT_MOCK": "false"}), TestClient(api.app) as client:
        for message in CASES:
            with contextlib.redirect_stdout(io.StringIO()):
                response = client.post("/movie-chat", json={"message": message, "country_code": "US"})
            data = response.json()
            requests += data.get("openai_requests", 0)
            print(f"{message}\nHTTP {response.status_code}; status={data.get('status')}; cards={len(data.get('recommendations', []))}")
            print(data.get("assistant_message"))
            print("Tools:", ", ".join(item["tool"] for item in data.get("tool_results", [])))
            if response.status_code != 200 or data.get("status") == "error":
                print(f"Stopped after failure; {requests} completed OpenAI requests (failed attempts may add one).")
                return 1
    print(f"Actual OpenAI requests: {requests}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
