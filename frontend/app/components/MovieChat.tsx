"use client";

import { useEffect, useRef, useState } from "react";
import MovieCard from "./MovieCard";
import type { Recommendation } from "./movie-types";

type MovieOption = { tmdb_id: number; title: string; year: number | null };
type ChatResponse = {
  status: "success" | "needs_clarification" | "unresolved" | "needs_preferences" | "unsupported_intent" | "error";
  assistant_message: string;
  recommendations: Recommendation[];
  clarifications: { type: "movie"; query: string; options: MovieOption[] }[];
  mock_mode: boolean;
};
type Message = { role: "user" | "assistant"; text: string; result?: ChatResponse };
const UNAVAILABLE = "Movie AI is temporarily unavailable. Please try again later.";

function validResponse(value: unknown): value is ChatResponse {
  if (!value || typeof value !== "object") return false;
  const data = value as ChatResponse;
  return ["success", "needs_clarification", "unresolved", "needs_preferences", "unsupported_intent", "error"].includes(data.status)
    && typeof data.assistant_message === "string"
    && typeof data.mock_mode === "boolean"
    && Array.isArray(data.recommendations) && data.recommendations.every(movie =>
      movie && typeof movie.title === "string" && typeof movie.genres === "string"
      && typeof movie.overview === "string"
      && (movie.tmdb_id === null || Number.isInteger(movie.tmdb_id))
      && (movie.year === null || Number.isInteger(movie.year))
      && (movie.poster_url === null || typeof movie.poster_url === "string")
      && typeof movie.recommendation_source === "string" && typeof movie.engine_mode === "string"
      && [movie.rating, movie.content_score, movie.svd_score, movie.final_score].every(score => score === null || (typeof score === "number" && Number.isFinite(score)))
      && movie.watch_providers && typeof movie.watch_providers.country_code === "string"
      && (movie.watch_providers.link === null || typeof movie.watch_providers.link === "string")
      && [movie.watch_providers.stream, movie.watch_providers.free, movie.watch_providers.ads, movie.watch_providers.rent, movie.watch_providers.buy].every(providers =>
        Array.isArray(providers) && providers.every(provider => provider && typeof provider.name === "string"
          && Number.isInteger(provider.provider_id) && (provider.logo_url === null || typeof provider.logo_url === "string"))))
    && Array.isArray(data.clarifications) && data.clarifications.every(item =>
      item && item.type === "movie" && typeof item.query === "string" && Array.isArray(item.options)
      && item.options.every(option => option && Number.isInteger(option.tmdb_id) && typeof option.title === "string" && (option.year === null || Number.isInteger(option.year))));
}

export default function MovieChat({ apiUrl, countryCode }: { apiUrl: string; countryCode: string }) {
  const [messages, setMessages] = useState<Message[]>([{ role: "assistant", text: "Tell me what you're in the mood for." }]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const busy = useRef(false);
  const request = useRef<AbortController | null>(null);
  const latest = useRef<HTMLDivElement>(null);
  const composer = useRef<HTMLTextAreaElement>(null);

  useEffect(() => () => request.current?.abort(), []);
  useEffect(() => {
    if (messages.length > 1) latest.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [messages, loading]);

  async function send(text: string) {
    const message = text.trim();
    if (!message || busy.current) return;
    busy.current = true;
    setLoading(true);
    setInput("");
    setMessages(previous => [...previous, { role: "user", text: message }]);
    const controller = new AbortController();
    request.current = controller;
    const timeout = setTimeout(() => controller.abort(), 120000);
    try {
      const response = await fetch(`${apiUrl}/movie-chat`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, country_code: countryCode }), signal: controller.signal,
      });
      const data: unknown = await response.json();
      if (!validResponse(data) || (!response.ok && data.status !== "error")) throw new Error("Invalid response");
      setMessages(previous => [...previous, { role: "assistant", text: data.status === "error" && !data.mock_mode ? UNAVAILABLE : data.assistant_message, result: data }]);
    } catch {
      setMessages(previous => [...previous, { role: "assistant", text: UNAVAILABLE }]);
    } finally {
      clearTimeout(timeout);
      busy.current = false;
      setLoading(false);
      request.current = null;
      setTimeout(() => composer.current?.focus(), 0);
    }
  }

  return (
    <section className="movieChat" aria-labelledby="movie-chat-title">
      <div className="movieChatHeading">
        <div><span className="eyebrow">IN YOUR OWN WORDS</span><h2 id="movie-chat-title">Movie AI</h2><p>Ask for a movie in your own words. Each request starts fresh.</p></div>
        <span className="watchRegion">{countryCode}</span>
      </div>
      <div className="chatMessages" aria-busy={loading}>
        {messages.map((message, index) => (
          <div className={`chatMessage ${message.role}`} key={index}>
            <span className="chatRole">{message.role === "user" ? "You" : "Movie AI"}</span>
            <p className="chatBubble" role={message.result?.status === "error" ? "alert" : undefined}>{message.text}</p>
            {message.result?.mock_mode && <span className="chatMockLabel">Development mock mode</span>}
            {message.result?.clarifications.map(item => (
              <div className="chatChoices" key={item.query} aria-label={`Choose ${item.query}`}>
                {item.options.map(option => (
                  <button type="button" disabled={loading || option.year === null} key={option.tmdb_id}
                    onClick={() => void send(`Something like ${option.title} (${option.year})`)}>
                    {option.title} {option.year !== null && `(${option.year})`}
                  </button>
                ))}
              </div>
            ))}
            {!!message.result?.recommendations.length && <div className="movieGrid">
              {message.result.recommendations.map((movie, rank) => <MovieCard key={`${movie.tmdb_id}-${rank}`} movie={movie} index={rank} />)}
            </div>}
          </div>
        ))}
        <div ref={latest} className={loading ? "chatLoading" : "chatInputLabel"} aria-live="polite" role="status">{loading ? "Finding your movies…" : messages.at(-1)?.role === "assistant" && messages.length > 1 ? messages.at(-1)?.text : ""}</div>
      </div>
      <form className="chatComposer" onSubmit={event => { event.preventDefault(); void send(input); }}>
        <label className="chatInputLabel" htmlFor="movie-chat-input">Ask Movie AI</label>
        <textarea id="movie-chat-input" ref={composer} placeholder="Ask for a movie..." rows={2}
          maxLength={10000} value={input} disabled={loading} onChange={event => setInput(event.target.value)}
          onKeyDown={event => {
            if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); void send(input); }
          }} />
        <button type="submit" className="chatSend" disabled={loading || !input.trim()}>{loading ? "Working…" : "Send"}</button>
      </form>
    </section>
  );
}
