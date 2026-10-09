"use client";

import { useEffect, useLayoutEffect, useRef, useState } from "react";
import MovieCard from "./MovieCard";
import type { Recommendation } from "./movie-types";

type MovieOption = { tmdb_id: number; title: string; year: number | null };
type ChatResponse = {
  status: "success" | "needs_clarification" | "unresolved" | "needs_preferences" | "unsupported_intent" | "error";
  assistant_message: string;
  recommendations: Recommendation[];
  clarifications: { type: "movie"; query: string; options: MovieOption[] }[];
  mock_mode: boolean;
  parsed_intent?: { intent: string } | null;
  sources?: { title: string; url: string }[];
  citations?: { start_index: number; end_index: number; source_index: number }[];
  collection_choices?: { collection_id: number; name: string }[];
  tool_results?: { tool: string; result: { watch_providers?: Recommendation["watch_providers"] } }[];
};
type Message = { role: "user" | "assistant"; text: string; result?: ChatResponse; requestText?: string };
const UNAVAILABLE = "Movie AI is temporarily unavailable. Please try again later.";

function recommendationText(result: ChatResponse) {
  return "Here are some movies you might enjoy:\n\n" + result.recommendations
    .map((movie, index) => `${index + 1}. ${movie.title}`).join("\n");
}

function validResponse(value: unknown): value is ChatResponse {
  if (!value || typeof value !== "object") return false;
  const data = value as ChatResponse;
  if (data.sources !== undefined && (!Array.isArray(data.sources) || data.sources.length > 5 || !data.sources.every(source =>
    source && typeof source.title === "string" && typeof source.url === "string" && /^https?:\/\//i.test(source.url)))) return false;
  if (data.citations !== undefined && (!Array.isArray(data.citations) || data.citations.length > 30 || !data.citations.every((citation, index, all) =>
    citation && Number.isInteger(citation.start_index) && Number.isInteger(citation.end_index) && Number.isInteger(citation.source_index)
    && citation.start_index >= 0 && citation.end_index > citation.start_index && typeof data.assistant_message === "string"
    && citation.end_index <= Array.from(data.assistant_message).length && citation.source_index >= 0
    && citation.source_index < (data.sources?.length ?? 0) && (index === 0 || all[index - 1].end_index <= citation.start_index)))) return false;
  if (data.collection_choices !== undefined && (!Array.isArray(data.collection_choices) || !data.collection_choices.every(item =>
    item && Number.isInteger(item.collection_id) && typeof item.name === "string"))) return false;
  if (data.tool_results !== undefined && (!Array.isArray(data.tool_results) || !data.tool_results.every(item => {
    if (!item || typeof item.tool !== "string" || !item.result || typeof item.result !== "object") return false;
    if (item.tool !== "get_watch_providers") return true;
    const providers = item.result.watch_providers;
    return !!providers && typeof providers.country_code === "string"
      && (providers.link === null || typeof providers.link === "string")
      && [providers.stream, providers.free, providers.ads, providers.rent, providers.buy].every(category =>
        Array.isArray(category) && category.every(provider => provider && (provider.name === null || typeof provider.name === "string")));
  }))) return false;
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

function citedText(message: Message) {
  // Citation offsets belong to the original backend prose, not our title list.
  if (message.result?.status === "success" && message.result.recommendations.length) return message.text;
  const citations = message.result?.citations;
  const sources = message.result?.sources;
  if (!citations?.length || !sources?.length) return message.text;
  const chars = Array.from(message.text);
  const parts = [];
  let cursor = 0;
  for (const [index, citation] of citations.entries()) {
    parts.push(chars.slice(cursor, citation.start_index).join(""));
    parts.push(<a key={index} href={sources[citation.source_index].url} target="_blank" rel="noopener noreferrer"
      title={sources[citation.source_index].title}>{chars.slice(citation.start_index, citation.end_index).join("")}</a>);
    cursor = citation.end_index;
  }
  parts.push(chars.slice(cursor).join(""));
  return parts;
}

export default function MovieChat({ apiUrl, countryCode }: { apiUrl: string; countryCode: string }) {
  const [messages, setMessages] = useState<Message[]>([{ role: "assistant", text: "Tell me what you're in the mood for." }]);
  const experimentalEnabled = process.env.NODE_ENV === "development" && process.env.NEXT_PUBLIC_MOVIE_EXPERIMENTAL_RANKING === "true";
  const [rankingMode, setRankingMode] = useState("baseline");
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const busy = useRef(false);
  const sessionId = useRef<string | null>(null);
  const request = useRef<AbortController | null>(null);
  const messageArea = useRef<HTMLDivElement>(null);
  const latestAssistant = useRef<HTMLDivElement>(null);
  const latestUser = useRef<HTMLDivElement>(null);
  const renderedCount = useRef(messages.length);
  const userReading = useRef(false);
  const manualScroll = useRef(false);
  const composer = useRef<HTMLTextAreaElement>(null);

  useEffect(() => () => request.current?.abort(), []);
  useLayoutEffect(() => {
    if (renderedCount.current === messages.length) return;
    renderedCount.current = messages.length;
    const area = messageArea.current;
    const assistantReply = messages.at(-1)?.role === "assistant";
    const message = assistantReply ? latestAssistant.current : latestUser.current;
    if (!area || !message || !area.clientHeight || userReading.current) return;
    const areaBounds = area.getBoundingClientRect();
    const messageBounds = message.getBoundingClientRect();
    const top = area.scrollTop + messageBounds.top - areaBounds.top - area.clientTop;
    const nextTop = assistantReply ? top : Math.max(area.scrollTop,
      top + messageBounds.height - area.clientHeight + 2);
    // Scroll only this container, instantly. Clamp to its real content bounds.
    manualScroll.current = false;
    area.scrollTo({ top: Math.max(0, Math.min(nextTop, area.scrollHeight - area.clientHeight)), behavior: "instant" });
  }, [messages]);

  async function send(text: string) {
    const message = text.trim();
    if (!message || busy.current) return;
    busy.current = true;
    userReading.current = false;
    manualScroll.current = false;
    setLoading(true);
    setInput("");
    setMessages(previous => [...previous, { role: "user", text: message }]);
    const controller = new AbortController();
    request.current = controller;
    const timeout = setTimeout(() => controller.abort(), 120000);
    try {
      if (!sessionId.current) {
        if (typeof crypto.randomUUID === "function") sessionId.current = crypto.randomUUID();
        else {
          const bytes = crypto.getRandomValues(new Uint8Array(16));
          bytes[6] = (bytes[6] & 0x0f) | 0x40;
          bytes[8] = (bytes[8] & 0x3f) | 0x80;
          const hex = Array.from(bytes, byte => byte.toString(16).padStart(2, "0")).join("");
          sessionId.current = `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
        }
      }
      const response = await fetch(`${apiUrl}/movie-chat`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, country_code: countryCode, session_id: sessionId.current, ranking_mode: experimentalEnabled ? rankingMode : "baseline" }), signal: controller.signal,
      });
      const data: unknown = await response.json();
      if (!validResponse(data) || (!response.ok && data.status !== "error")) throw new Error("Invalid response");
      const displayedText = data.status === "success" && data.recommendations.length
        ? recommendationText(data)
        : data.status === "error" && !data.mock_mode && response.status !== 422 ? UNAVAILABLE : data.assistant_message;
      setMessages(previous => [...previous, { role: "assistant", text: displayedText, result: data, requestText: message }]);
    } catch {
      setMessages(previous => [...previous, { role: "assistant", text: UNAVAILABLE }]);
    } finally {
      clearTimeout(timeout);
      busy.current = false;
      setLoading(false);
      request.current = null;
      setTimeout(() => {
        if (composer.current?.getClientRects().length) composer.current.focus({ preventScroll: true });
      }, 0);
    }
  }

  return (
    <section className="movieChat" aria-labelledby="movie-chat-title">
      <div className="movieChatHeading">
        <div><span className="eyebrow">IN YOUR OWN WORDS</span><h2 id="movie-chat-title">Movie AI</h2><p>Ask for a movie in your own words, then keep the conversation going.</p></div>
      {experimentalEnabled && <label className="rankingSelector"><span>Active ranking</span><select aria-label="Ranking mode" value={rankingMode} disabled={loading} onChange={event => setRankingMode(event.target.value)}><option value="baseline">Original</option><option value="genre_votes">Experimental (Genre + Votes)</option></select></label>}
        <span className="watchRegion">{countryCode}</span>
      </div>
      <div className="chatMessages" ref={messageArea} aria-busy={loading} tabIndex={0} aria-label="Conversation messages"
        onWheel={() => { manualScroll.current = true; }}
        onTouchStart={() => { manualScroll.current = true; }}
        onPointerDown={() => { manualScroll.current = true; }}
        onKeyDown={event => {
          if (["ArrowUp", "ArrowDown", "PageUp", "PageDown", "Home", "End", " "].includes(event.key)) manualScroll.current = true;
        }}
        onScroll={event => {
          // Browser layout changes and our own scrollTo calls are not evidence
          // that the user chose to read history. Only user-initiated scrolling is.
          if (!manualScroll.current) return;
          const area = event.currentTarget;
          userReading.current = area.scrollHeight - area.clientHeight - area.scrollTop > 24;
        }}>
        {messages.map((message, index) => (
          <div className={`chatMessage ${message.role}`} key={index}
            ref={index === messages.length - 1 ? (message.role === "assistant" ? latestAssistant : latestUser) : undefined}>
            <span className="chatRole">{message.role === "user" ? "You" : "Movie AI"}</span>
            <p className="chatBubble" role={message.result?.status === "error" ? "alert" : undefined}>{citedText(message)}</p>
            {!!message.result?.sources?.length && <div className="watchSection compactWatch" aria-label="Research sources">
              <span>Sources: </span>{message.result.sources.map((source, sourceIndex) =>
                <a className="watchLink" href={source.url} key={source.url} target="_blank" rel="noopener noreferrer">{sourceIndex > 0 && " · "}{source.title}</a>)}
            </div>}
            {message.result?.mock_mode && <span className="chatMockLabel">Development mock mode</span>}
            {!!message.result?.collection_choices?.length && <div className="chatChoices" aria-label="Choose collection">
              {message.result.collection_choices.map(collection => <button type="button" disabled={loading} key={collection.collection_id}
                onClick={() => void send(`Give me the ${collection.name} in release order.`)}>{collection.name}</button>)}
            </div>}
            {message.result?.clarifications.map(item => (
              <div className="chatChoices" key={item.query} aria-label={`Choose ${item.query}`}>
                {item.options.map(option => (
                  <button type="button" disabled={loading || option.year === null} key={option.tmdb_id}
                    onClick={() => {
                      const original = message.requestText || `Tell me about ${item.query}`;
                      const position = original.toLocaleLowerCase().indexOf(item.query.toLocaleLowerCase());
                      const qualified = `${option.title} (${option.year})`;
                      if (index === messages.length - 1 && message.result?.parsed_intent?.intent === "recommend") {
                        void send(qualified);
                        return;
                      }
                      void send(position < 0 ? `Tell me about ${qualified}` : original.slice(0, position) + qualified + original.slice(position + item.query.length));
                    }}>
                    {option.title} {option.year !== null && `(${option.year})`}
                  </button>
                ))}
              </div>
            ))}
            {message.result?.tool_results?.filter(item => item.tool === "get_watch_providers").map((item, providerIndex) => {
              const providers = item.result.watch_providers;
              if (!providers) return null;
              return <div className="watchSection compactWatch" key={providerIndex}>
                <span className="watchRegion">{providers.country_code}</span>
                {(["stream", "free", "ads", "rent", "buy"] as const).map(category =>
                  !!providers[category]?.length && <p key={category}>{category}: {providers[category].map(provider => provider.name).join(", ")}</p>)}
                {providers.link && /^https:\/\//.test(providers.link) && <a className="watchLink" href={providers.link} target="_blank" rel="noopener noreferrer">View all watch options →</a>}
                <div className="justWatchCredit">Availability powered by JustWatch</div>
              </div>;
            })}
            {!!message.result?.recommendations.length && <div className="movieGrid">
              {message.result.recommendations.map((movie, rank) => <MovieCard key={`${movie.tmdb_id}-${rank}`} movie={movie} index={rank} />)}
            </div>}
          </div>
        ))}
      </div>
      <div className={loading ? "chatLoading" : "chatInputLabel"} aria-live="polite" role="status">{loading ? "Finding your movies…" : messages.at(-1)?.role === "assistant" && messages.length > 1 ? messages.at(-1)?.text : ""}</div>
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
