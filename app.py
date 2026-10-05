import streamlit as st
import pandas as pd

from recommendation_engine import RecommendationEngine


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="AI Movie Recommender",
    page_icon="🎬",
    layout="wide"
)


# ==========================================================
# LOAD RECOMMENDATION ENGINE
# ==========================================================

@st.cache_resource
def load_engine():
    return RecommendationEngine()


engine = load_engine()


# ==========================================================
# PREPARE MOVIE OPTIONS
# ==========================================================

@st.cache_resource
def prepare_movie_options():

    movies = engine.content.movies[
        [
            "tmdb_id",
            "title",
            "release_year",
            "popularity"
        ]
    ].copy()

    # Make sure popularity is numeric
    movies["popularity"] = pd.to_numeric(
        movies["popularity"],
        errors="coerce"
    ).fillna(0)

    # Popular movies first
    movies = movies.sort_values(
        "popularity",
        ascending=False
    )

    movie_options = []
    movie_lookup = {}

    for row in movies.itertuples():

        title = str(row.title)

        # ----------------------------------------------
        # YEAR
        # ----------------------------------------------

        if pd.notna(row.release_year):

            try:
                year = int(row.release_year)

            except:
                year = None

        else:
            year = None


        # ----------------------------------------------
        # LABEL USER SEES
        # ----------------------------------------------

        if year:

            label = (
                f"{title} ({year})"
            )

        else:

            label = title


        # ----------------------------------------------
        # HANDLE DUPLICATE LABELS
        # ----------------------------------------------

        if label in movie_lookup:

            label = (
                f"{label} "
                f"[TMDB {int(row.tmdb_id)}]"
            )


        movie_options.append(
            label
        )

        movie_lookup[label] = {
            "tmdb_id": int(
                row.tmdb_id
            ),
            "title": title,
            "release_year": year
        }


    return (
        movie_options,
        movie_lookup
    )


movie_options, movie_lookup = (
    prepare_movie_options()
)


# ==========================================================
# BASIC INPUT PARSER
# ==========================================================

def parse_input(value):

    if not value:
        return []

    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]


# ==========================================================
# GET MOVIE INFORMATION
# ==========================================================

def get_movie_info(tmdb_id):

    if tmdb_id is None:
        return None

    try:

        tmdb_id = int(
            tmdb_id
        )

    except:

        return None


    movies = engine.content.movies

    match = movies[
        pd.to_numeric(
            movies["tmdb_id"],
            errors="coerce"
        )
        == tmdb_id
    ]


    if match.empty:
        return None


    return match.iloc[0]


# ==========================================================
# DISPLAY MOVIE CARD
# ==========================================================

def display_movie_card(
    movie,
    rank
):

    tmdb_id = movie.get(
        "tmdb_id"
    )

    movie_info = get_movie_info(
        tmdb_id
    )


    # ======================================================
    # TMDB METADATA
    # ======================================================

    if movie_info is not None:

        title = movie_info.get(
            "title",
            movie.get(
                "title",
                "Unknown"
            )
        )

        year = movie_info.get(
            "release_year"
        )

        genres = movie_info.get(
            "genres",
            ""
        )

        rating = movie_info.get(
            "vote_average"
        )

        overview = movie_info.get(
            "overview",
            ""
        )

        poster_path = movie_info.get(
            "poster_path"
        )

    else:

        title = movie.get(
            "title",
            "Unknown"
        )

        year = movie.get(
            "release_year"
        )

        genres = movie.get(
            "genres",
            ""
        )

        rating = None

        overview = ""

        poster_path = None


    # ======================================================
    # POSTER
    # ======================================================

    poster_url = None

    if (
        poster_path is not None
        and not pd.isna(poster_path)
        and str(poster_path).strip()
    ):

        poster_url = (
            "https://image.tmdb.org/t/p/w500"
            + str(poster_path)
        )


    # ======================================================
    # CARD
    # ======================================================

    with st.container(
        border=True
    ):

        poster_col, info_col = (
            st.columns(
                [1, 4],
                gap="medium"
            )
        )


        # --------------------------------------------------
        # POSTER
        # --------------------------------------------------

        with poster_col:

            if poster_url:

                st.image(
                    poster_url,
                    use_container_width=True
                )

            else:

                st.markdown(
                    "## 🎬"
                )

                st.caption(
                    "Poster unavailable"
                )


        # --------------------------------------------------
        # INFORMATION
        # --------------------------------------------------

        with info_col:

            st.subheader(
                f"{rank}. {title}"
            )


            # ----------------------------------------------
            # YEAR + RATING
            # ----------------------------------------------

            details = []


            if pd.notna(year):

                try:

                    details.append(
                        f"📅 {int(year)}"
                    )

                except:
                    pass


            if (
                rating is not None
                and pd.notna(rating)
            ):

                try:

                    details.append(
                        f"⭐ {float(rating):.1f}/10"
                    )

                except:
                    pass


            if details:

                st.write(
                    " • ".join(
                        details
                    )
                )


            # ----------------------------------------------
            # GENRES
            # ----------------------------------------------

            if (
                genres is not None
                and not pd.isna(genres)
                and str(genres).strip()
            ):

                st.caption(
                    f"🎭 {genres}"
                )


            # ----------------------------------------------
            # OVERVIEW
            # ----------------------------------------------

            if (
                overview is not None
                and not pd.isna(overview)
                and str(overview).strip()
            ):

                overview_text = (
                    str(overview)
                    .strip()
                )

                if len(
                    overview_text
                ) > 350:

                    overview_text = (
                        overview_text[:347]
                        + "..."
                    )

                st.write(
                    overview_text
                )


            # ==================================================
            # WHY RECOMMENDED
            # ==================================================

            with st.expander(
                "Why was this recommended?"
            ):

                source = movie.get(
                    "recommendation_source",
                    "unknown"
                )

                st.write(
                    "**Recommendation model:**",
                    str(source).upper()
                )


                # ------------------------------------------
                # ENGINE MODE
                # ------------------------------------------

                engine_mode = movie.get(
                    "engine_mode"
                )

                if (
                    engine_mode is not None
                    and pd.notna(engine_mode)
                ):

                    st.write(
                        "**Engine mode:**",
                        engine_mode
                    )


                # ------------------------------------------
                # CONTENT SCORE
                # ------------------------------------------

                content_score = movie.get(
                    "content_score"
                )

                if (
                    content_score is not None
                    and pd.notna(content_score)
                ):

                    st.write(
                        "**Content score:**",
                        f"{float(content_score):.3f}"
                    )


                # ------------------------------------------
                # SVD SCORE
                # ------------------------------------------

                svd_score = movie.get(
                    "svd_score"
                )

                if (
                    svd_score is not None
                    and pd.notna(svd_score)
                ):

                    st.write(
                        "**SVD score:**",
                        f"{float(svd_score):.3f}"
                    )


                # ------------------------------------------
                # FINAL SCORE
                # ------------------------------------------

                final_score = movie.get(
                    "final_score"
                )

                if (
                    final_score is not None
                    and pd.notna(final_score)
                ):

                    st.write(
                        "**Final ranking score:**",
                        f"{float(final_score):.3f}"
                    )


# ==========================================================
# HEADER
# ==========================================================

st.title(
    "🎬 AI Movie Recommender"
)

st.write(
    "Tell me what you love and I'll help you find your next movie."
)

st.divider()


# ==========================================================
# FAVORITE MOVIES
# ==========================================================

st.header(
    "🍿 Movies you love"
)

st.caption(
    "Search for movies and select as many favorites as you want."
)


# ==========================================================
# SEARCHABLE MULTISELECT
# ==========================================================

selected_movie_labels = st.multiselect(
    "Search for your favorite movies",
    options=movie_options,
    placeholder="Start typing a movie title..."
)


# ==========================================================
# CONVERT SELECTED LABELS TO MOVIES
# ==========================================================

selected_movies = [

    movie_lookup[label]

    for label in selected_movie_labels

    if label in movie_lookup

]


# ==========================================================
# SHOW NUMBER SELECTED
# ==========================================================

if selected_movies:

    st.caption(
        f"✓ {len(selected_movies)} movie"
        f"{'s' if len(selected_movies) != 1 else ''} selected"
    )

else:

    st.caption(
        "No movies selected yet."
    )


st.divider()


# ==========================================================
# OPTIONAL PREFERENCES
# ==========================================================

st.header(
    "🎯 Optional preferences"
)


col1, col2, col3 = (
    st.columns(3)
)


# ----------------------------------------------------------
# GENRE
# ----------------------------------------------------------

with col1:

    genre_text = st.text_input(
        "Genre",
        placeholder="Science Fiction"
    )


# ----------------------------------------------------------
# ACTOR
# ----------------------------------------------------------

with col2:

    actor_text = st.text_input(
        "Actor",
        placeholder="Christian Bale"
    )


# ----------------------------------------------------------
# DIRECTOR
# ----------------------------------------------------------

with col3:

    director_text = st.text_input(
        "Director",
        placeholder="Christopher Nolan"
    )


# ==========================================================
# FIND MOVIES BUTTON
# ==========================================================

st.write("")


find_movies = st.button(
    "🍿 Find Movies",
    type="primary",
    use_container_width=True
)


# ==========================================================
# GENERATE RECOMMENDATIONS
# ==========================================================

if find_movies:

    # ======================================================
    # FAVORITE MOVIES
    # ======================================================

    favorite_movies = [

        movie["title"]

        for movie in selected_movies

    ]


    # ======================================================
    # OPTIONAL PREFERENCES
    # ======================================================

    favorite_genres = (
        parse_input(
            genre_text
        )
    )


    favorite_actors = (
        parse_input(
            actor_text
        )
    )


    favorite_directors = (
        parse_input(
            director_text
        )
    )


    # ======================================================
    # VALIDATION
    # ======================================================

    if not (
        favorite_movies
        or favorite_genres
        or favorite_actors
        or favorite_directors
    ):

        st.warning(
            "Select at least one movie or enter a preference."
        )

        st.stop()


    # ======================================================
    # RECOMMEND
    # ======================================================

    try:

        with st.spinner(
            "Finding movies for you..."
        ):

            recommendations = (
                engine.recommend(

                    favorite_movies=
                        favorite_movies,

                    favorite_genres=
                        favorite_genres,

                    favorite_actors=
                        favorite_actors,

                    favorite_directors=
                        favorite_directors,

                    top_n=10
                )
            )


        # ==================================================
        # NO RESULTS
        # ==================================================

        if recommendations.empty:

            st.warning(
                "I couldn't find recommendations matching your request."
            )


        # ==================================================
        # RESULTS
        # ==================================================

        else:

            st.divider()

            st.header(
                "🍿 Recommended for you"
            )

            st.caption(
                f"{len(recommendations)} "
                "movies selected for you"
            )


            for index, movie in (
                recommendations.iterrows()
            ):

                display_movie_card(
                    movie,
                    index + 1
                )


    # ======================================================
    # ERROR
    # ======================================================

    except Exception as error:

        st.error(
            "Something went wrong while generating recommendations."
        )

        st.exception(
            error
        )