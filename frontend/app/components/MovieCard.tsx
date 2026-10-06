import type { Recommendation } from "./movie-types";

export default function MovieCard({ movie, index }: { movie: Recommendation; index: number }) {
  return (
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
{/* WHERE TO WATCH */}
<div className="watchSection compactWatch">
  <div className="watchTopRow">
    <div className="watchHeading">
      Where to watch
    </div>
    <span className="watchRegion">
      {movie.watch_providers.country_code}
    </span>
  </div>
  {/* SUBSCRIPTION STREAMING */}
  {movie.watch_providers.stream.length > 0 ? (
    <div className="streamingBlock">
      <span className="providerLabel">
        STREAM
      </span>
      <div className="providerChips">
        {movie.watch_providers.stream
          .slice(0, 3)
          .map((provider) => (
            <div
              className="providerChip"
              key={provider.provider_id}
              title={provider.name}
            >
              {provider.logo_url ? (
                <img
                  src={provider.logo_url}
                  alt={provider.name}
                  title={provider.name}
                  className="providerLogo"
                />
              ) : (
                <span>{provider.name}</span>
              )}
            </div>
          ))}
        {movie.watch_providers.stream.length > 3 && (
          <div className="providerMore">
            +
            {movie.watch_providers.stream.length - 3}
          </div>
        )}
      </div>
    </div>
  ) : (
    <div className="noStreaming">
      No subscription streaming available
    </div>
  )}
  {/* RENT / BUY */}
  {(movie.watch_providers.rent.length > 0 ||
    movie.watch_providers.buy.length > 0) && (
    <div className="purchaseAvailability">
      <span className="purchaseLabel">
        Also available
      </span>
      <div className="purchaseOptions">
        {movie.watch_providers.rent.length > 0 && (
          <span>Rent</span>
        )}
        {movie.watch_providers.rent.length > 0 &&
          movie.watch_providers.buy.length > 0 && (
            <span className="availabilityDot">
              •
            </span>
          )}
        {movie.watch_providers.buy.length > 0 && (
          <span>Buy</span>
        )}
      </div>
    </div>
  )}
  {/* NOTHING AVAILABLE */}
  {movie.watch_providers.stream.length === 0 &&
    movie.watch_providers.free.length === 0 &&
    movie.watch_providers.ads.length === 0 &&
    movie.watch_providers.rent.length === 0 &&
    movie.watch_providers.buy.length === 0 && (
      <div className="noWatchOptions">
        No watch options found for this region.
      </div>
    )}
  {/* LINK */}
  {movie.watch_providers.link && (
    <a
      className="watchLink compactWatchLink"
      href={movie.watch_providers.link}
      target="_blank"
      rel="noopener noreferrer"
    >
      View all watch options →
    </a>
  )}
  <div className="justWatchCredit">
    Availability powered by JustWatch
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
  );
}
