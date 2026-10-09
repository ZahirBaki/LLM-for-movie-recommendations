from backend.core.paths import PROJECT_ROOT, project_path
import os
import requests
import pandas as pd
from dotenv import load_dotenv


# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("TMDB_API_KEY")

if not API_KEY:
    raise ValueError("TMDB_API_KEY was not found in .env")


# --------------------------------------------------
# 1. Get movies
# --------------------------------------------------

discover_url = "https://api.themoviedb.org/3/discover/movie"

params = {
    "api_key": API_KEY,
    "language": "en-US",
    "sort_by": "popularity.desc",
    "page": 1
}

all_movies = []

for page in range(1, 51):
    params["page"] = page
    response = requests.get(discover_url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    all_movies.extend(data["results"])
    print(f"Loaded movie page {page}/50")

    if page >= data.get("total_pages", 50):
        break

movies = pd.DataFrame(all_movies).drop_duplicates(subset="id")


# --------------------------------------------------
# 2. Get genre names
# --------------------------------------------------

genre_url = "https://api.themoviedb.org/3/genre/movie/list"

genre_params = {
    "api_key": API_KEY,
    "language": "en-US"
}

genre_response = requests.get(
    genre_url,
    params=genre_params
)

genre_data = genre_response.json()

genre_dictionary = {
    genre["id"]: genre["name"]
    for genre in genre_data["genres"]
}


# --------------------------------------------------
# 3. Convert genre IDs to genre names
# --------------------------------------------------

def get_genres(genre_ids):

    return [
        genre_dictionary.get(genre_id, "Unknown")
        for genre_id in genre_ids
    ]


movies["genres"] = movies["genre_ids"].apply(get_genres)


# --------------------------------------------------
# 4. Get actors and director
# --------------------------------------------------

def get_credits(movie_id):

    url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits"

    params = {
        "api_key": API_KEY,
        "language": "en-US"
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        return [], ""

    data = response.json()

    # First 10 actors
    actors = [
        person["name"]
        for person in data.get("cast", [])[:10]
    ]

    # Find director
    directors = [
        person["name"]
        for person in data.get("crew", [])
        if person.get("job") == "Director"
    ]

    director = directors[0] if directors else ""

    return actors, director


# Get credits for every movie
actors_list = []
directors_list = []

for movie_id in movies["id"]:

    actors, director = get_credits(movie_id)

    actors_list.append(actors)
    directors_list.append(director)


movies["actors"] = actors_list
movies["director"] = directors_list


# --------------------------------------------------
# 5. Display the result
# --------------------------------------------------

print("\nMovies with additional information:\n")

print(
    movies[
        [
            "title",
            "release_date",
            "genres",
            "director",
            "actors"
        ]
    ].to_string(index=False)
)

# --------------------------------------------------
# 6. Prepare the data for saving
# --------------------------------------------------

# Convert lists into text so they can be saved in CSV
movies["genres"] = movies["genres"].apply(
    lambda x: ", ".join(x)
)

movies["actors"] = movies["actors"].apply(
    lambda x: ", ".join(x)
)


# --------------------------------------------------
# 7. Select the columns we need
# --------------------------------------------------

movies = movies[
    [
        "id",
        "title",
        "release_date",
        "genres",
        "director",
        "actors",
        "overview",
        "popularity",
        "vote_average",
        "vote_count"
    ]
]


# --------------------------------------------------
# 8. Save as CSV
# --------------------------------------------------

movies.to_csv(
    project_path('data/raw/movies.csv'),
    index=False,
    encoding="utf-8"
)

print("\nMovie data saved to movies.csv")
