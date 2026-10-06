export type WatchProvider = {
  provider_id: number;
  name: string;
  logo_url: string | null;
  display_priority: number | null;
};

export type WatchProviders = {
  country_code: string;
  stream: WatchProvider[];
  free: WatchProvider[];
  ads: WatchProvider[];
  rent: WatchProvider[];
  buy: WatchProvider[];
  link: string | null;
};

export type Recommendation = {

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



