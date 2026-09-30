import os
import time
import requests
import pandas as pd

from dotenv import load_dotenv


# ============================================================
# SETTINGS
# ============================================================

START_YEAR = 1950
END_YEAR = 2026

# A movie needs at least this many TMDB votes.
# This helps remove extremely obscure/empty entries.
MIN_VOTE_COUNT = 20

# Number of actors to store for each movie.
MAX_ACTORS = 10

# Save after this many processed movies.
SAVE_EVERY = 100

OUTPUT_FILE = "data/tmdb_movies.csv"

TMDB_BASE_URL = "https://api.themoviedb.org/3"


# ============================================================
# LOAD API KEY
# ============================================================

load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY")

if not API_KEY:
    raise ValueError(
        "TMDB_API_KEY was not found in your .env file."
    )


# ============================================================
# HTTP SESSION
# ============================================================

session = requests.Session()


# ============================================================
# SAFE TMDB REQUEST
# ============================================================

def tmdb_get(endpoint, params=None, retries=5):

    if params is None:
        params = {}

    params["api_key"] = API_KEY

    url = f"{TMDB_BASE_URL}{endpoint}"

    for attempt in range(retries):

        try:

            response = session.get(
                url,
                params=params,
                timeout=30
            )

            # Rate limited
            if response.status_code == 429:

                wait_time = int(
                    response.headers.get(
                        "Retry-After",
                        2
                    )
                )

                print(
                    f"Rate limited. Waiting "
                    f"{wait_time} seconds..."
                )

                time.sleep(wait_time)

                continue

            response.raise_for_status()

            return response.json()

        except requests.RequestException as error:

            print(
                f"Request failed "
                f"(attempt {attempt + 1}/{retries}): "
                f"{error}"
            )

            time.sleep(2 * (attempt + 1))

    print(
        f"Skipping request after "
        f"{retries} failed attempts."
    )

    return None


# ============================================================
# LOAD EXISTING PROGRESS
# ============================================================

if os.path.exists(OUTPUT_FILE):

    print(
        f"\nExisting catalog found: "
        f"{OUTPUT_FILE}"
    )

    existing_df = pd.read_csv(
        OUTPUT_FILE
    )

    movie_records = (
        existing_df
        .to_dict("records")
    )

    processed_ids = set(
        existing_df["tmdb_id"]
        .astype(int)
        .tolist()
    )

    print(
        f"Loaded {len(processed_ids):,} "
        f"existing movies."
    )

else:

    movie_records = []
    processed_ids = set()

    print(
        "\nNo existing catalog found. "
        "Starting a new one."
    )


# ============================================================
# SAVE FUNCTION
# ============================================================

def save_catalog():

    if not movie_records:
        return

    df = pd.DataFrame(
        movie_records
    )

    df = df.drop_duplicates(
        subset="tmdb_id"
    )

    df = df.sort_values(
        by="tmdb_id"
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nSaved {len(df):,} movies "
        f"to {OUTPUT_FILE}\n"
    )


# ============================================================
# GET FULL MOVIE INFORMATION
# ============================================================

def get_movie_details(tmdb_id):

    data = tmdb_get(
        f"/movie/{tmdb_id}",
        {
            "language": "en-US",

            # Get credits in the SAME request.
            "append_to_response": "credits"
        }
    )

    if not data:
        return None


    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    title = (
        data.get("title")
        or ""
    ).strip()

    overview = (
        data.get("overview")
        or ""
    ).strip()

    release_date = (
        data.get("release_date")
        or ""
    ).strip()


    # We need these fields for our recommender.
    if not title:
        return None

    if not overview:
        return None

    if not release_date:
        return None


    # --------------------------------------------------------
    # Genres
    # --------------------------------------------------------

    genre_names = [
        genre["name"]
        for genre in data.get(
            "genres",
            []
        )
        if genre.get("name")
    ]

    genres = ", ".join(
        genre_names
    )


    if not genres:
        return None


    # --------------------------------------------------------
    # Credits
    # --------------------------------------------------------

    credits = data.get(
        "credits",
        {}
    )


    # Actors
    cast = credits.get(
        "cast",
        []
    )

    actors = [
        person.get("name", "")
        for person in cast[:MAX_ACTORS]
        if person.get("name")
    ]

    actor_text = ", ".join(
        actors
    )


    # Director
    crew = credits.get(
        "crew",
        []
    )

    directors = [
        person.get("name", "")
        for person in crew
        if (
            person.get("job")
            == "Director"
            and person.get("name")
        )
    ]

    director_text = ", ".join(
        dict.fromkeys(directors)
    )


    # --------------------------------------------------------
    # Final record
    # --------------------------------------------------------

    return {

        "tmdb_id":
            data.get("id"),

        "title":
            title,

        "original_title":
            data.get(
                "original_title",
                ""
            ),

        "release_date":
            release_date,

        "release_year":
            release_date[:4],

        "genres":
            genres,

        "director":
            director_text,

        "actors":
            actor_text,

        "overview":
            overview,

        "original_language":
            data.get(
                "original_language",
                ""
            ),

        "popularity":
            data.get(
                "popularity",
                0
            ),

        "vote_average":
            data.get(
                "vote_average",
                0
            ),

        "vote_count":
            data.get(
                "vote_count",
                0
            ),

        "runtime":
            data.get(
                "runtime",
                0
            ),

        "poster_path":
            data.get(
                "poster_path",
                ""
            )
    }


# ============================================================
# BUILD CATALOG
# ============================================================

new_movies_since_save = 0


try:

    for year in range(
        END_YEAR,
        START_YEAR - 1,
        -1
    ):

        print(
            "\n"
            + "=" * 60
        )

        print(
            f"Collecting movies from {year}"
        )

        print(
            "=" * 60
        )


        # ----------------------------------------------------
        # First page
        # ----------------------------------------------------

        first_page = tmdb_get(
            "/discover/movie",
            {
                "language":
                    "en-US",

                "include_adult":
                    "false",

                "include_video":
                    "false",

                "primary_release_year":
                    year,

                "sort_by":
                    "vote_count.desc",

                "vote_count.gte":
                    MIN_VOTE_COUNT,

                "page":
                    1
            }
        )


        if not first_page:

            print(
                f"Could not retrieve {year}."
            )

            continue


        total_pages = first_page.get(
            "total_pages",
            0
        )


        # TMDB pagination is limited.
        total_pages = min(
            total_pages,
            500
        )


        print(
            f"{year}: {total_pages} pages"
        )


        # ----------------------------------------------------
        # Process every page
        # ----------------------------------------------------

        for page in range(
            1,
            total_pages + 1
        ):

            print(
                f"\r{year} "
                f"| page {page}/{total_pages} "
                f"| catalog: "
                f"{len(movie_records):,}",
                end="",
                flush=True
            )


            if page == 1:

                page_data = first_page

            else:

                page_data = tmdb_get(
                    "/discover/movie",
                    {
                        "language":
                            "en-US",

                        "include_adult":
                            "false",

                        "include_video":
                            "false",

                        "primary_release_year":
                            year,

                        "sort_by":
                            "vote_count.desc",

                        "vote_count.gte":
                            MIN_VOTE_COUNT,

                        "page":
                            page
                    }
                )


            if not page_data:
                continue


            results = page_data.get(
                "results",
                []
            )


            # ------------------------------------------------
            # Process movies on this page
            # ------------------------------------------------

            for movie in results:

                tmdb_id = movie.get(
                    "id"
                )


                if not tmdb_id:
                    continue


                # Already downloaded.
                if tmdb_id in processed_ids:
                    continue


                # Quick filtering before making
                # the more expensive details request.
                overview = (
                    movie.get("overview")
                    or ""
                ).strip()

                release_date = (
                    movie.get("release_date")
                    or ""
                ).strip()


                if not overview:
                    continue

                if not release_date:
                    continue


                details = get_movie_details(
                    tmdb_id
                )


                if not details:
                    continue


                movie_records.append(
                    details
                )

                processed_ids.add(
                    tmdb_id
                )

                new_movies_since_save += 1


                # --------------------------------------------
                # Save regularly
                # --------------------------------------------

                if (
                    new_movies_since_save
                    >= SAVE_EVERY
                ):

                    save_catalog()

                    new_movies_since_save = 0


                # Small delay so we're polite to TMDB.
                time.sleep(0.03)


        # Save at the end of each year.
        save_catalog()

        new_movies_since_save = 0


# ============================================================
# CTRL+C / ERROR PROTECTION
# ============================================================

except KeyboardInterrupt:

    print(
        "\n\nStopped by user."
    )

    print(
        "Saving everything downloaded "
        "so far..."
    )

    save_catalog()

    print(
        "You can run the script again "
        "later to continue."
    )


except Exception as error:

    print(
        f"\n\nUnexpected error: {error}"
    )

    print(
        "Saving everything downloaded "
        "so far..."
    )

    save_catalog()

    raise


# ============================================================
# FINISHED
# ============================================================

else:

    save_catalog()

    print(
        "\nCatalog build finished!"
    )

    print(
        f"Total movies: "
        f"{len(movie_records):,}"
    )