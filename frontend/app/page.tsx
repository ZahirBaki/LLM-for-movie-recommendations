"use client";



import { useEffect, useState } from "react";



const API_URL = "http://127.0.0.1:8000";

const COUNTRIES = [
  { code: "SE", name: "Sweden", flag: "🇸🇪" },
  { code: "US", name: "United States", flag: "🇺🇸" },
  { code: "GB", name: "United Kingdom", flag: "🇬🇧" },
  { code: "DE", name: "Germany", flag: "🇩🇪" },
  { code: "FR", name: "France", flag: "🇫🇷" },
  { code: "DK", name: "Denmark", flag: "🇩🇰" },
  { code: "NO", name: "Norway", flag: "🇳🇴" },
  { code: "FI", name: "Finland", flag: "🇫🇮" },
  { code: "NL", name: "Netherlands", flag: "🇳🇱" },
  { code: "ES", name: "Spain", flag: "🇪🇸" },
  { code: "IT", name: "Italy", flag: "🇮🇹" },
  { code: "CA", name: "Canada", flag: "🇨🇦" },
  { code: "AU", name: "Australia", flag: "🇦🇺" },
];





// ==========================================================

// TYPES

// ==========================================================



type MovieSearchResult = {

  tmdb_id: number;

  title: string;

  year: number | null;

  poster_url: string | null;

};



type WatchProvider = {
  provider_id: number;
  name: string;
  logo_url: string | null;
  display_priority: number | null;
};

type WatchProviders = {
  country_code: string;
  stream: WatchProvider[];
  free: WatchProvider[];
  ads: WatchProvider[];
  rent: WatchProvider[];
  buy: WatchProvider[];
  link: string | null;
};

type Recommendation = {

  tmdb_id: number | null;

  title: string;

  year: number | null;

  genres: string;

  rating: number | null;

  overview: string;

  poster_url: string | null;

  recommendation_source: string;

  engine_mode: string;

  content_score: number | null;

  svd_score: number | null;

  final_score: number | null;
  watch_providers: WatchProviders;

};



type PersonSelectorProps = {

  label: string;

  placeholder: string;

  endpoint: "actors" | "directors";

  selected: string[];

  setSelected: React.Dispatch<React.SetStateAction<string[]>>;

};



type GenreSelectorProps = {

  label: string;

  placeholder: string;

  availableGenres: string[];

  selected: string[];

  setSelected: React.Dispatch<React.SetStateAction<string[]>>;

};





// ==========================================================

// PERSON SELECTOR

// ==========================================================



function PersonSelector({

  label,

  placeholder,

  endpoint,

  selected,

  setSelected,

}: PersonSelectorProps) {

  const [query, setQuery] = useState("");

  const [results, setResults] = useState<string[]>([]);

  const [searching, setSearching] = useState(false);



  useEffect(() => {

    const cleanQuery = query.trim();



    if (cleanQuery.length < 2) {

      setResults([]);

      setSearching(false);

      return;

    }



    const controller = new AbortController();



    const timer = setTimeout(async () => {

      try {

        setSearching(true);



        const response = await fetch(

          `${API_URL}/${endpoint}/search?q=${encodeURIComponent(cleanQuery)}`,

          {

            signal: controller.signal,

          }

        );



        if (!response.ok) {

          throw new Error(`${label} search failed.`);

        }



        const data: string[] = await response.json();



        setResults(

          data.filter(

            (name) => !selected.includes(name)

          )

        );

      } catch (error) {

        if (

          error instanceof Error &&

          error.name !== "AbortError"

        ) {

          console.error(error);

        }

      } finally {

        setSearching(false);

      }

    }, 120);



    return () => {

      clearTimeout(timer);

      controller.abort();

    };

  }, [query, endpoint, label, selected]);



  function addPerson(name: string) {

    if (!selected.includes(name)) {

      setSelected((current) => [

        ...current,

        name,

      ]);

    }



    setQuery("");

    setResults([]);

  }



  function removePerson(name: string) {

    setSelected((current) =>

      current.filter(

        (person) => person !== name

      )

    );

  }



  return (

    <div className="inputGroup">

      <label>{label}</label>



      <div className="selectorWrapper">

        <div className="multiSelectInput">

          {selected.map((name) => (

            <span

              className="selectionChip"

              key={name}

            >

              {name}



              <button

                type="button"

                onClick={() =>

                  removePerson(name)

                }

              >

                ×

              </button>

            </span>

          ))}



          <input

            value={query}

            onChange={(event) =>

              setQuery(event.target.value)

            }

            placeholder={

              selected.length === 0

                ? placeholder

                : "Add another..."

            }

            autoComplete="off"

          />



          {searching && (

            <span className="smallLoader">

              ...

            </span>

          )}

        </div>



        {results.length > 0 && (

          <div className="preferenceDropdown">

            {results.map((name) => (

              <button

                type="button"

                key={name}

                onClick={() =>

                  addPerson(name)

                }

              >

                <span className="personIcon">

                  {endpoint === "actors"

                    ? "★"

                    : "◆"}

                </span>



                <span>{name}</span>

              </button>

            ))}

          </div>

        )}

      </div>

    </div>

  );

}





// ==========================================================

// GENRE SELECTOR

// ==========================================================



function GenreSelector({

  label,

  placeholder,

  availableGenres,

  selected,

  setSelected,

}: GenreSelectorProps) {

  const [query, setQuery] = useState("");

  const [open, setOpen] = useState(false);



  const filteredGenres =

    availableGenres.filter((genre) => {

      const matches =

        genre

          .toLowerCase()

          .includes(

            query.trim().toLowerCase()

          );



      return (

        matches &&

        !selected.includes(genre)

      );

    });



  function addGenre(genre: string) {

    if (!selected.includes(genre)) {

      setSelected((current) => [

        ...current,

        genre,

      ]);

    }



    setQuery("");

    setOpen(false);

  }



  function removeGenre(genre: string) {

    setSelected((current) =>

      current.filter(

        (item) => item !== genre

      )

    );

  }



  return (

    <div className="inputGroup">

      <label>{label}</label>



      <div className="selectorWrapper">

        <div

          className="multiSelectInput"

          onClick={() => setOpen(true)}

        >

          {selected.map((genre) => (

            <span

              className="selectionChip"

              key={genre}

            >

              {genre}



              <button

                type="button"

                onClick={(event) => {

                  event.stopPropagation();

                  removeGenre(genre);

                }}

              >

                ×

              </button>

            </span>

          ))}



          <input

            value={query}

            onFocus={() => setOpen(true)}

            onChange={(event) => {

              setQuery(event.target.value);

              setOpen(true);

            }}

            placeholder={

              selected.length === 0

                ? placeholder

                : "Add another..."

            }

          />

        </div>



        {open &&

          filteredGenres.length > 0 && (

            <div className="preferenceDropdown">

              {filteredGenres.map(

                (genre) => (

                  <button

                    type="button"

                    key={genre}

                    onClick={() =>

                      addGenre(genre)

                    }

                  >

                    <span className="genreIcon">

                      ●

                    </span>



                    <span>{genre}</span>

                  </button>

                )

              )}

            </div>

          )}

      </div>

    </div>

  );

}





// ==========================================================

// MAIN PAGE

// ==========================================================



export default function Home() {



  // ========================================================

  // MOVIE SEARCH

  // ========================================================



  const [search, setSearch] =

    useState("");



  const [

    searchResults,

    setSearchResults,

  ] = useState<MovieSearchResult[]>([]);



  const [searching, setSearching] =

    useState(false);



  const [

    selectedMovies,

    setSelectedMovies,

  ] = useState<MovieSearchResult[]>([]);





  // ========================================================

  // GENRES

  // ========================================================



  const [

    availableGenres,

    setAvailableGenres,

  ] = useState<string[]>([]);





  // ========================================================

  // SOFT PREFERENCES

  // ========================================================



  const [

    selectedGenres,

    setSelectedGenres,

  ] = useState<string[]>([]);



  const [

    selectedActors,

    setSelectedActors,

  ] = useState<string[]>([]);



  const [

    selectedDirectors,

    setSelectedDirectors,

  ] = useState<string[]>([]);





  // ========================================================

  // HARD REQUIREMENTS

  // ========================================================



  const [

    requiredGenres,

    setRequiredGenres,

  ] = useState<string[]>([]);



  const [

    requiredActors,

    setRequiredActors,

  ] = useState<string[]>([]);



  const [

    requiredDirectors,

    setRequiredDirectors,

  ] = useState<string[]>([]);





  // ========================================================

  // EXCLUSIONS

  // ========================================================



  const [

    excludedGenres,

    setExcludedGenres,

  ] = useState<string[]>([]);



  const [

    excludedActors,

    setExcludedActors,

  ] = useState<string[]>([]);



  const [

    excludedDirectors,

    setExcludedDirectors,

  ] = useState<string[]>([]);





  // ========================================================

  // YEAR FILTER

  // ========================================================



  const [minYear, setMinYear] =

    useState("");



  const [maxYear, setMaxYear] =

    useState("");





  // ========================================================

  // COUNTRY / STREAMING REGION

  // ========================================================

  const [countryCode, setCountryCode] =
    useState("SE");



  // ========================================================

  // ADVANCED FILTERS

  // ========================================================



  const [

    advancedOpen,

    setAdvancedOpen,

  ] = useState(false);





  // ========================================================

  // RESULTS

  // ========================================================



  const [

    recommendations,

    setRecommendations,

  ] = useState<Recommendation[]>([]);



  const [loading, setLoading] =

    useState(false);



  const [error, setError] =

    useState("");





  // ========================================================

  // LOAD GENRES

  // ========================================================



  useEffect(() => {

    async function loadGenres() {

      try {

        const response = await fetch(

          `${API_URL}/genres`

        );



        if (!response.ok) {

          throw new Error(

            "Could not load genres."

          );

        }



        const data: string[] =

          await response.json();



        setAvailableGenres(data);

      } catch (error) {

        console.error(error);

      }

    }



    loadGenres();

  }, []);





  // ========================================================

  // MOVIE SEARCH

  // ========================================================



  useEffect(() => {

    const query = search.trim();



    if (query.length < 2) {

      setSearchResults([]);

      setSearching(false);

      return;

    }



    const controller =

      new AbortController();



    const timer = setTimeout(

      async () => {

        try {

          setSearching(true);



          const response = await fetch(

            `${API_URL}/movies/search?q=${encodeURIComponent(query)}`,

            {

              signal:

                controller.signal,

            }

          );



          if (!response.ok) {

            throw new Error(

              "Movie search failed."

            );

          }



          const data:

            MovieSearchResult[] =

            await response.json();



          setSearchResults(

            data.filter(

              (movie) =>

                !selectedMovies.some(

                  (selected) =>

                    selected.tmdb_id ===

                    movie.tmdb_id

                )

            )

          );

        } catch (error) {

          if (

            error instanceof Error &&

            error.name !== "AbortError"

          ) {

            console.error(error);

          }

        } finally {

          setSearching(false);

        }

      },

      120

    );



    return () => {

      clearTimeout(timer);

      controller.abort();

    };

  }, [search, selectedMovies]);





  // ========================================================

  // MOVIE FUNCTIONS

  // ========================================================



  function selectMovie(

    movie: MovieSearchResult

  ) {

    if (

      !selectedMovies.some(

        (selected) =>

          selected.tmdb_id ===

          movie.tmdb_id

      )

    ) {

      setSelectedMovies(

        (current) => [

          ...current,

          movie,

        ]

      );

    }



    setSearch("");

    setSearchResults([]);

  }





  function removeMovie(

    tmdbId: number

  ) {

    setSelectedMovies(

      (current) =>

        current.filter(

          (movie) =>

            movie.tmdb_id !==

            tmdbId

        )

    );

  }





  // ========================================================

  // CHECK IF HARD FILTERS ARE ACTIVE

  // ========================================================



  const activeFilterCount =

    (minYear ? 1 : 0) +

    (maxYear ? 1 : 0) +

    requiredGenres.length +

    requiredActors.length +

    requiredDirectors.length +

    excludedGenres.length +

    excludedActors.length +

    excludedDirectors.length;





  // ========================================================

  // CLEAR HARD FILTERS

  // ========================================================



  function clearFilters() {

    setMinYear("");

    setMaxYear("");



    setRequiredGenres([]);

    setRequiredActors([]);

    setRequiredDirectors([]);



    setExcludedGenres([]);

    setExcludedActors([]);

    setExcludedDirectors([]);

  }





  // ========================================================

  // FIND MOVIES

  // ========================================================



  async function findMovies() {

    setError("");



    const hasSoftPreferences =

      selectedGenres.length > 0 ||

      selectedActors.length > 0 ||

      selectedDirectors.length > 0;



    const hasHardFilters =

      activeFilterCount > 0;



    if (

      selectedMovies.length === 0 &&

      !hasSoftPreferences &&

      !hasHardFilters

    ) {

      setError(

        "Select a favorite movie, add a preference, or use a filter."

      );



      return;

    }





    // ------------------------------------------------------

    // VALIDATE YEARS

    // ------------------------------------------------------



    const parsedMinYear =

      minYear

        ? Number(minYear)

        : null;



    const parsedMaxYear =

      maxYear

        ? Number(maxYear)

        : null;





    if (

      parsedMinYear !== null &&

      parsedMaxYear !== null &&

      parsedMinYear >

        parsedMaxYear

    ) {

      setError(

        "The 'From year' cannot be later than the 'To year'."

      );



      return;

    }





    try {

      setLoading(true);



      setRecommendations([]);





      const response = await fetch(

        `${API_URL}/recommend`,

        {

          method: "POST",



          headers: {

            "Content-Type":

              "application/json",

          },



          body: JSON.stringify({



            // FAVORITES

            favorite_movies:
              selectedMovies.map((movie) => ({
                tmdb_id: movie.tmdb_id,
                title: movie.title,
                year: movie.year,
              })),





            // SOFT PREFERENCES

            favorite_genres:

              selectedGenres,



            favorite_actors:

              selectedActors,



            favorite_directors:

              selectedDirectors,





            // HARD REQUIREMENTS

            required_genres:

              requiredGenres,



            required_actors:

              requiredActors,



            required_directors:

              requiredDirectors,





            // EXCLUSIONS

            exclude_genres:

              excludedGenres,



            exclude_actors:

              excludedActors,



            exclude_directors:

              excludedDirectors,





            // YEAR RANGE

            min_year:

              parsedMinYear,



            max_year:

              parsedMaxYear,



            // STREAMING REGION

            country_code:

              countryCode,



            top_n: 10,

          }),

        }

      );





      if (!response.ok) {

        throw new Error(

          "Recommendation request failed."

        );

      }





      const data:

        Recommendation[] =

        await response.json();





      setRecommendations(data);





      if (data.length === 0) {

        setError(

          "No movies matched all of your filters. Try removing one or more filters."

        );

      }



    } catch (error) {

      console.error(error);



      setError(

        "Something went wrong while generating recommendations."

      );



    } finally {

      setLoading(false);

    }

  }





  // ========================================================

  // PAGE

  // ========================================================



  return (

    <main className="page">



      {/* ===================================================

          HERO

      ==================================================== */}



      <section className="hero">



        <div className="brand">

          <span className="brandIcon">

            ◆

          </span>



          MOVIE AI

        </div>





        <h1>

          Find your next

          <span> great movie.</span>

        </h1>





        <p>

          Pick movies you love, tell us

          what you are in the mood for,

          and let the recommendation

          engine do the rest.

        </p>



      </section>





      {/* ===================================================

          MAIN BOX

      ==================================================== */}



      <section className="recommendationBox">





        {/* =================================================

            FAVORITES

        ================================================== */}



        <div className="sectionHeading">



          <span className="stepNumber">

            1

          </span>



          <div>

            <h2>

              Movies you love

            </h2>



            <p>

              Search for movies you

              already like.

            </p>

          </div>



        </div>





        <div className="searchWrapper">



          <div className="searchInputWrapper">



            <span className="searchIcon">

              ⌕

            </span>





            <input

              value={search}

              onChange={(event) =>

                setSearch(

                  event.target.value

                )

              }

              placeholder="Search for Interstellar, The Godfather..."

              className="movieSearch"

              autoComplete="off"

            />





            {searching && (

              <span className="searchingText">

                Searching...

              </span>

            )}



          </div>





          {searchResults.length > 0 && (

            <div className="searchDropdown">



              {searchResults.map(

                (movie) => (



                  <button

                    type="button"

                    key={movie.tmdb_id}

                    className="searchResult"

                    onClick={() =>

                      selectMovie(movie)

                    }

                  >



                    <div className="searchPoster">



                      {movie.poster_url ? (

                        <img

                          src={

                            movie.poster_url

                          }

                          alt=""

                        />

                      ) : (

                        <div className="posterFallback">

                          🎬

                        </div>

                      )}



                    </div>





                    <div className="searchMovieInfo">



                      <strong>

                        {movie.title}

                      </strong>



                      <span>

                        {movie.year ??

                          "Unknown year"}

                      </span>



                    </div>



                  </button>



                )

              )}



            </div>

          )}



        </div>





        {/* =================================================

            SELECTED MOVIES

        ================================================== */}



        {selectedMovies.length > 0 && (



          <div className="selectedSection">



            <div className="selectedHeader">



              <span>

                {selectedMovies.length}{" "}

                selected

              </span>





              <button

                type="button"

                className="clearButton"

                onClick={() =>

                  setSelectedMovies([])

                }

              >

                Clear all

              </button>



            </div>





            <div className="selectedMovies">



              {selectedMovies.map(

                (movie) => (



                  <div

                    className="selectedMovie"

                    key={movie.tmdb_id}

                  >



                    {movie.poster_url ? (

                      <img

                        src={

                          movie.poster_url

                        }

                        alt={

                          movie.title

                        }

                      />

                    ) : (

                      <div className="selectedFallback">

                        🎬

                      </div>

                    )}





                    <div className="selectedMovieInfo">



                      <strong>

                        {movie.title}

                      </strong>



                      <span>

                        {movie.year ?? ""}

                      </span>



                    </div>





                    <button

                      type="button"

                      className="removeMovie"

                      onClick={() =>

                        removeMovie(

                          movie.tmdb_id

                        )

                      }

                    >

                      ×

                    </button>



                  </div>



                )

              )}



            </div>



          </div>



        )}





        {/* =================================================

            SOFT PREFERENCES

        ================================================== */}



        <div className="preferencesSection">



          <div className="sectionHeading">



            <span className="stepNumber">

              2

            </span>



            <div>



              <h2>

                Anything specific?

              </h2>



              <p>

                These preferences influence

                the ranking, but they are not

                strict requirements.

              </p>



            </div>



          </div>





          <div className="preferenceGrid">



            <GenreSelector

              label="Genres"

              placeholder="Select genres..."

              availableGenres={

                availableGenres

              }

              selected={

                selectedGenres

              }

              setSelected={

                setSelectedGenres

              }

            />





            <PersonSelector

              label="Actors"

              placeholder="Try: Bale"

              endpoint="actors"

              selected={

                selectedActors

              }

              setSelected={

                setSelectedActors

              }

            />





            <PersonSelector

              label="Directors"

              placeholder="Try: Nolan"

              endpoint="directors"

              selected={

                selectedDirectors

              }

              setSelected={

                setSelectedDirectors

              }

            />



          </div>



        </div>





        {/* =================================================

            COUNTRY / STREAMING REGION

        ================================================== */}

        <div className="countrySection">

          <div className="sectionHeading">

            <span className="stepNumber">
              3
            </span>

            <div>
              <h2>Where do you live?</h2>
              <p>
                We&apos;ll show where you can watch each recommendation.
              </p>
            </div>

          </div>

          <div className="countrySelector">
            <label htmlFor="country">Streaming region</label>
            <select
              id="country"
              value={countryCode}
              onChange={(event) => setCountryCode(event.target.value)}
            >
              {COUNTRIES.map((country) => (
                <option key={country.code} value={country.code}>
                  {country.flag} {country.name}
                </option>
              ))}
            </select>
          </div>

        </div>



        {/* =================================================

            ADVANCED FILTERS

        ================================================== */}



        <div className="advancedSection">



          <button

            type="button"

            className="advancedToggle"

            onClick={() =>

              setAdvancedOpen(

                (current) => !current

              )

            }

          >



            <div>



              <span className="filterIcon">

                ⚙

              </span>



              <div>

                <strong>

                  Advanced filters

                </strong>



                <small>

                  Set strict requirements

                  and exclusions

                </small>

              </div>



            </div>





            <div className="advancedToggleRight">



              {activeFilterCount > 0 && (

                <span className="filterCount">

                  {activeFilterCount}

                </span>

              )}



              <span

                className={

                  advancedOpen

                    ? "chevron open"

                    : "chevron"

                }

              >

                ▼

              </span>



            </div>



          </button>





          {advancedOpen && (



            <div className="advancedContent">





              {/* ===========================================

                  YEAR

              ============================================ */}



              <div className="filterBlock">



                <div className="filterBlockHeading">



                  <h3>

                    Release year

                  </h3>



                  <p>

                    Only show movies released

                    within this range.

                  </p>



                </div>





                <div className="yearRange">



                  <div className="yearInput">



                    <label>

                      From

                    </label>



                    <input

                      type="number"

                      min="1900"

                      max="2100"

                      placeholder="e.g. 2010"

                      value={minYear}

                      onChange={(event) =>

                        setMinYear(

                          event.target.value

                        )

                      }

                    />



                  </div>





                  <span className="yearArrow">

                    →

                  </span>





                  <div className="yearInput">



                    <label>

                      To

                    </label>



                    <input

                      type="number"

                      min="1900"

                      max="2100"

                      placeholder="e.g. 2026"

                      value={maxYear}

                      onChange={(event) =>

                        setMaxYear(

                          event.target.value

                        )

                      }

                    />



                  </div>



                </div>



              </div>





              {/* ===========================================

                  MUST INCLUDE

              ============================================ */}



              <div className="filterBlock">



                <div className="filterBlockHeading">



                  <h3>

                    Must include

                  </h3>



                  <p>

                    Recommended movies must

                    match these requirements.

                  </p>



                </div>





                <div className="preferenceGrid">



                  <GenreSelector

                    label="Genres"

                    placeholder="Required genre..."

                    availableGenres={

                      availableGenres

                    }

                    selected={

                      requiredGenres

                    }

                    setSelected={

                      setRequiredGenres

                    }

                  />





                  <PersonSelector

                    label="Actors"

                    placeholder="Required actor..."

                    endpoint="actors"

                    selected={

                      requiredActors

                    }

                    setSelected={

                      setRequiredActors

                    }

                  />





                  <PersonSelector

                    label="Directors"

                    placeholder="Required director..."

                    endpoint="directors"

                    selected={

                      requiredDirectors

                    }

                    setSelected={

                      setRequiredDirectors

                    }

                  />



                </div>



              </div>





              {/* ===========================================

                  EXCLUDE

              ============================================ */}



              <div className="filterBlock exclusionBlock">



                <div className="filterBlockHeading">



                  <h3>

                    Exclude

                  </h3>



                  <p>

                    Movies matching these

                    values will not be shown.

                  </p>



                </div>





                <div className="preferenceGrid">



                  <GenreSelector

                    label="Genres"

                    placeholder="Exclude genre..."

                    availableGenres={

                      availableGenres

                    }

                    selected={

                      excludedGenres

                    }

                    setSelected={

                      setExcludedGenres

                    }

                  />





                  <PersonSelector

                    label="Actors"

                    placeholder="Exclude actor..."

                    endpoint="actors"

                    selected={

                      excludedActors

                    }

                    setSelected={

                      setExcludedActors

                    }

                  />





                  <PersonSelector

                    label="Directors"

                    placeholder="Exclude director..."

                    endpoint="directors"

                    selected={

                      excludedDirectors

                    }

                    setSelected={

                      setExcludedDirectors

                    }

                  />



                </div>



              </div>





              {/* ===========================================

                  CLEAR

              ============================================ */}



              {activeFilterCount > 0 && (



                <div className="clearFiltersRow">



                  <span>

                    {activeFilterCount} active{" "}

                    {activeFilterCount === 1

                      ? "filter"

                      : "filters"}

                  </span>





                  <button

                    type="button"

                    onClick={

                      clearFilters

                    }

                  >

                    Clear filters

                  </button>



                </div>



              )}



            </div>



          )}



        </div>





        {/* =================================================

            ERROR

        ================================================== */}



        {error && (

          <div className="errorMessage">

            {error}

          </div>

        )}





        {/* =================================================

            FIND BUTTON

        ================================================== */}



        <button

          className="findButton"

          onClick={findMovies}

          disabled={loading}

        >



          {loading

            ? "Finding your movies..."

            : "Find movies for me"}



        </button>



      </section>





      {/* ===================================================

          RESULTS

      ==================================================== */}



      {recommendations.length > 0 && (



        <section className="resultsSection">



          <div className="resultsHeading">



            <div>



              <span className="eyebrow">

                PERSONALIZED FOR YOU

              </span>



              <h2>

                Your recommendations

              </h2>



            </div>





            <span className="resultCount">

              {recommendations.length} movies

            </span>



          </div>





          <div className="movieGrid">



            {recommendations.map(

              (movie, index) => (



                <article

                  className="movieCard"

                  key={`${movie.tmdb_id}-${index}`}

                >



                  <div className="posterContainer">



                    {movie.poster_url ? (

                      <img

                        src={

                          movie.poster_url

                        }

                        alt={`${movie.title} poster`}

                      />

                    ) : (

                      <div className="cardPosterFallback">

                        🎬

                      </div>

                    )}





                    <span className="rankBadge">

                      #{index + 1}

                    </span>





                    {movie.rating !== null && (

                      <span className="ratingBadge">

                        ★{" "}

                        {movie.rating.toFixed(

                          1

                        )}

                      </span>

                    )}



                  </div>





                  <div className="movieContent">



                    <h3>

                      {movie.title}

                    </h3>





                    <div className="movieMeta">



                      {movie.year && (

                        <span>

                          {movie.year}

                        </span>

                      )}



                      {movie.genres && (

                        <span>

                          {movie.genres}

                        </span>

                      )}



                    </div>





                    {movie.overview && (

                      <p className="overview">

                        {movie.overview}

                      </p>

                    )}





                    {/* WHERE TO WATCH */}

                    <div className="watchSection">
                      <div className="watchHeading">Where to watch</div>

                      {movie.watch_providers.stream.length > 0 && (
                        <div className="providerRow">
                          <span className="providerLabel">STREAM</span>
                          <div className="providerText">
                            {movie.watch_providers.stream
                              .slice(0, 3)
                              .map((provider) => provider.name)
                              .join(" · ")}
                          </div>
                        </div>
                      )}

                      {movie.watch_providers.free.length > 0 && (
                        <div className="providerRow">
                          <span className="providerLabel">FREE</span>
                          <div className="providerText">
                            {movie.watch_providers.free
                              .slice(0, 3)
                              .map((provider) => provider.name)
                              .join(" · ")}
                          </div>
                        </div>
                      )}

                      {movie.watch_providers.ads.length > 0 && (
                        <div className="providerRow">
                          <span className="providerLabel">WITH ADS</span>
                          <div className="providerText">
                            {movie.watch_providers.ads
                              .map((provider) => provider.name)
                              .join(" · ")}
                          </div>
                        </div>
                      )}

                      {movie.watch_providers.rent.length > 0 && (
                        <div className="providerRow">
                          <span className="providerLabel">RENT</span>
                          <div className="providerText">
                            {movie.watch_providers.rent
                              .slice(0, 3)
                              .map((provider) => provider.name)
                              .join(" · ")}
                          </div>
                        </div>
                      )}

                      {movie.watch_providers.buy.length > 0 && (
                        <div className="providerRow">
                          <span className="providerLabel">BUY</span>
                          <div className="providerText">
                            {movie.watch_providers.buy
                              .slice(0, 3)
                              .map((provider) => provider.name)
                              .join(" · ")}
                          </div>
                        </div>
                      )}

                      {movie.watch_providers.stream.length === 0 &&
                        movie.watch_providers.free.length === 0 &&
                        movie.watch_providers.ads.length === 0 &&
                        movie.watch_providers.rent.length === 0 &&
                        movie.watch_providers.buy.length === 0 && (
                          <div className="noProviders">
                            No watch options found for this region.
                          </div>
                        )}

                      {movie.watch_providers.link && (
                        <a
                          className="watchLink"
                          href={movie.watch_providers.link}
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          View all watch options →
                        </a>
                      )}

                      <div className="justWatchCredit">
                        Streaming availability powered by JustWatch
                      </div>
                    </div>



                    <details className="technicalDetails">



                      <summary>

                        Why was this recommended?

                      </summary>





                      <div>



                        <p>

                          <strong>

                            Source:

                          </strong>{" "}

                          {movie.recommendation_source ||

                            "N/A"}

                        </p>





                        <p>

                          <strong>

                            Mode:

                          </strong>{" "}

                          {movie.engine_mode ||

                            "N/A"}

                        </p>





                        {movie.svd_score !== null && (

                          <p>

                            <strong>

                              SVD score:

                            </strong>{" "}

                            {movie.svd_score.toFixed(

                              3

                            )}

                          </p>

                        )}





                        {movie.content_score !==

                          null && (

                          <p>

                            <strong>

                              Content score:

                            </strong>{" "}

                            {movie.content_score.toFixed(

                              3

                            )}

                          </p>

                        )}





                        {movie.final_score !==

                          null && (

                          <p>

                            <strong>

                              Final score:

                            </strong>{" "}

                            {movie.final_score.toFixed(

                              3

                            )}

                          </p>

                        )}



                      </div>



                    </details>



                  </div>



                </article>



              )

            )}



          </div>



        </section>



      )}





      <footer>

        Movie AI • Hybrid movie recommendation system

      </footer>



    </main>

  );

}
