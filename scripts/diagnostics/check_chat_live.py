"""Optional live /chat checks. Prints only parsed output or sanitized errors."""

from fastapi.testclient import TestClient

from backend.api.app import app

CASES = [
    ("Recommend something like Interstellar with Ryan Gosling from after 2015", {"intent": "recommend", "favorite_movies": "Interstellar", "required_actors": "Ryan Gosling", "min_year": 2016}),
    ("I want a comedy directed by Quentin Tarantino", {"intent": "recommend", "required_genres": "Comedy", "required_directors": "Quentin Tarantino"}),
    ("Something from Christopher Nolan before 2010", {"intent": "recommend", "required_directors": "Christopher Nolan", "max_year": 2009}),
    ("Recommend a science fiction movie", {"intent": "recommend", "required_genres": "Science Fiction"}),
    ("Something like Interstellar but less depressing", {"intent": "recommend", "favorite_movies": "Interstellar"}),
    ("Who directed Interstellar?", {"intent": "information", "movie_titles": "Interstellar"}),
    ("What's the weather tomorrow?", {"intent": "unknown"}),
]


def main():
    with TestClient(app) as client:
        for index, (message, expected) in enumerate(CASES, start=1):
            response = client.post("/chat", json={"message": message})
            print(f"TEST {index}: HTTP {response.status_code}")
            if response.status_code != 200:
                print(response.json()["detail"])
                print("Live checks stopped; remaining cases require a working API connection and credit.")
                return 1
            parsed = response.json()["parsed"]
            for field, value in expected.items():
                assert value in parsed[field] if isinstance(parsed[field], list) else parsed[field] == value, f"Unexpected {field} in test {index}"
            if index == 5:
                assert any("less depressing" in value.lower() for value in parsed["natural_language_preferences"])
            print(parsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
