"use client";

import { useEffect, useRef, useState } from "react";
import styles from "./evaluation.module.css";

type Side = "A" | "B";
type Movie = { tmdb_id: number; title: string; year: number | null; genres: string; overview: string; poster_url: string | null };
type Scenario = { scenario_id: string; name: string; favorites: Movie[]; lists: Record<Side, Movie[]>; submitted: boolean };
type Session = { evaluation_id: string; scenarios: Scenario[]; completed: boolean };
type Draft = { ratings: Record<string, string>; preference: string; notes: string };
const API = "http://127.0.0.1:8013";
const SESSION_KEY = "movie-human-review-session-v1";
const emptyDraft = (): Draft => ({ ratings: {}, preference: "", notes: "" });

async function request(path: string, body?: unknown): Promise<Session> {
  const response = await fetch(API + path, { cache: "no-store", ...(body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  }) });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Could not save this evaluation.");
  return data;
}

export default function EvaluationReview() {
  const [session, setSession] = useState<Session | null>(null);
  const [index, setIndex] = useState(0);
  const [draft, setDraft] = useState<Draft>(emptyDraft);
  const [busy, setBusy] = useState(false);
  const [restoring, setRestoring] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const inFlight = useRef(false);
  const heading = useRef<HTMLHeadingElement>(null);
  const scenario = session?.scenarios[index];
  const completedCount = session?.scenarios.filter(row => row.submitted).length || 0;
  const draftKey = session && scenario ? `${SESSION_KEY}:${session.evaluation_id}:${scenario.scenario_id}` : "";

  useEffect(() => {
    let active = true;
    const restore = async () => {
      try {
        const identity = localStorage.getItem(SESSION_KEY);
        if (identity) {
          const restored = await request(`/sessions/${identity}`);
          if (active) {
            setSession(restored);
            const next = restored.scenarios.findIndex(row => !row.submitted);
            setIndex(next < 0 ? 0 : next);
          }
        }
      } catch (reason) {
        if (active) setError(`${reason instanceof Error ? reason.message : "Could not resume."} Start the local service, then reload, or start a new review.`);
      } finally { if (active) setRestoring(false); }
    };
    void restore();
    return () => { active = false; };
  }, []);

  const saveDraft = (next: Draft) => {
    setDraft(next);
    if (draftKey) {
      try { localStorage.setItem(draftKey, JSON.stringify(next)); }
      catch { setNotice("Draft storage is unavailable. Keep this page open until you submit."); }
    }
  };

  const openScenario = (next: number, current: Session) => {
    setIndex(next);
    setError("");
    const key = `${SESSION_KEY}:${current.evaluation_id}:${current.scenarios[next].scenario_id}`;
    try { setDraft(JSON.parse(localStorage.getItem(key) || "null") || emptyDraft()); }
    catch { setDraft(emptyDraft()); }
    heading.current?.focus();
    window.scrollTo({ top: 0, behavior: "instant" });
  };

  // Restore a scenario draft only when the session initially arrives.
  const restoredId = useRef("");
  useEffect(() => {
    if (session && restoredId.current !== session.evaluation_id) {
      restoredId.current = session.evaluation_id;
      const key = `${SESSION_KEY}:${session.evaluation_id}:${session.scenarios[index].scenario_id}`;
      try { setDraft(JSON.parse(localStorage.getItem(key) || "null") || emptyDraft()); }
      catch { setDraft(emptyDraft()); }
    }
  }, [session, index]);

  const start = async () => {
    if (inFlight.current) return;
    inFlight.current = true;
    setBusy(true); setError(""); setNotice("");
    try {
      const next = await request("/sessions", {});
      try { localStorage.setItem(SESSION_KEY, next.evaluation_id); }
      catch { setNotice("Browser storage is unavailable. Keep this page open until you finish."); }
      setSession(next); setIndex(0); setDraft(emptyDraft());
    } catch (reason) {
      setError(`${reason instanceof Error ? reason.message : "Could not start."} Run the local evaluation service on port 8013.`);
    } finally { inFlight.current = false; setBusy(false); }
  };

  const submit = async () => {
    if (!session || !scenario || scenario.submitted || inFlight.current) return;
    const ratings = { A: [] as { tmdb_id: number; rating: number | null }[], B: [] as { tmdb_id: number; rating: number | null }[] };
    for (const side of ["A", "B"] as const) {
      for (const movie of scenario.lists[side]) {
        const value = draft.ratings[`${side}:${movie.tmdb_id}`];
        if (!value) { setError("Rate each movie or choose ‘Not seen / skip’, then choose your preferred list."); return; }
        ratings[side].push({ tmdb_id: movie.tmdb_id, rating: value === "skip" ? null : Number(value) });
      }
    }
    if (!["A", "B", "tie"].includes(draft.preference)) { setError("Choose List A, List B or about equal."); return; }
    inFlight.current = true; setBusy(true); setError("");
    try {
      const next = await request(`/sessions/${session.evaluation_id}/submit`, {
        scenario_id: scenario.scenario_id, ratings, preference: draft.preference, notes: draft.notes,
      });
      setSession(next);
      setNotice("Scenario saved locally. You can submit each scenario once.");
      const nextIndex = next.scenarios.findIndex(row => !row.submitted);
      if (nextIndex >= 0) openScenario(nextIndex, next);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not save.");
      // A retry after a lost response must not submit the same scenario twice.
      try { setSession(await request(`/sessions/${session.evaluation_id}`)); } catch { /* Keep the draft for retry. */ }
    } finally { inFlight.current = false; setBusy(false); }
  };

  const download = async (format: "json" | "csv") => {
    if (!session?.completed || inFlight.current) return;
    inFlight.current = true; setBusy(true); setError("");
    try {
      const response = await fetch(`${API}/sessions/${session.evaluation_id}/export?format=${format}`, { cache: "no-store" });
      if (!response.ok) throw new Error((await response.json()).error);
      const url = URL.createObjectURL(await response.blob());
      const anchor = document.createElement("a");
      anchor.href = url; anchor.download = `movie-review-${session.evaluation_id}.${format}`;
      anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Could not export."); }
    finally { inFlight.current = false; setBusy(false); }
  };

  return <main className={styles.page}>
    <header className={styles.header}>
      <p className={styles.eyebrow}>Local human evaluation</p>
      <h1>Compare recommendation lists</h1>
      <p>Consider genre, tone, audience, overall fit and variety. Rate matches from 1 (very poor) to 5 (excellent). Skip movies you haven’t seen.</p>
      <p>No name or email is collected. Submitted reviews are saved on this computer. Drafts and an anonymous review ID are saved in this browser.</p>
    </header>
    {error && <p className={styles.error} role="alert">{error}</p>}
    {notice && <p role="status" className={styles.notice}>{notice}</p>}
    {!session ? <button type="button" disabled={busy || restoring} onClick={() => void start()}>
      {restoring ? "Resuming…" : busy ? "Starting…" : "Start anonymous review"}
    </button> : <>
      <p className={styles.progress}>{completedCount} of {session.scenarios.length} scenarios saved</p>
      <nav className={styles.scenarios} aria-label="Evaluation scenarios">
        {session.scenarios.map((row, position) => <button type="button" key={row.scenario_id} disabled={busy}
          aria-current={position === index ? "step" : undefined} onClick={() => openScenario(position, session)}>
          {position + 1}. {row.name}{row.submitted ? " ✓" : ""}
        </button>)}
      </nav>
      {scenario && <section key={scenario.scenario_id} aria-labelledby="scenario-heading">
        <h2 id="scenario-heading" tabIndex={-1} ref={heading}>{scenario.name}</h2>
        <p>Favorites: {scenario.favorites.map(movie => `${movie.title} (${movie.year ?? "Year unavailable"})`).join(" · ")}</p>
        <div className={styles.lists}>
          {(["A", "B"] as const).map(side => <section key={side} className={styles.list} aria-labelledby={`list-${side}`}>
            <h3 id={`list-${side}`}>List {side}</h3>
            <ol>
              {scenario.lists[side].map(movie => <li key={movie.tmdb_id} data-movie-id={movie.tmdb_id} data-list-side={side}>
                <div className={styles.movie}>
                  {movie.poster_url ? <img src={movie.poster_url} alt={`Poster for ${movie.title}`} width={72} height={108} loading="lazy" />
                    : <span className={styles.noPoster}>No poster</span>}
                  <div className={styles.movieDetails}><h4>{movie.title} <span>({movie.year ?? "Year unavailable"})</span></h4>
                    <p>{movie.genres || "Genres unavailable"}</p>
                    {movie.overview && <details><summary>Overview</summary><p>{movie.overview}</p></details>}
                  </div>
                </div>
                <label className={styles.rating}>Match rating
                  <select aria-label={`List ${side}: ${movie.title} match rating`} disabled={busy || scenario.submitted}
                    value={draft.ratings[`${side}:${movie.tmdb_id}`] || ""}
                    onChange={event => saveDraft({ ...draft, ratings: { ...draft.ratings, [`${side}:${movie.tmdb_id}`]: event.target.value } })}>
                    <option value="">Choose rating</option>
                    <option value="1">1 — Very poor match</option><option value="2">2 — Poor match</option>
                    <option value="3">3 — Acceptable</option><option value="4">4 — Good match</option>
                    <option value="5">5 — Excellent match</option><option value="skip">Not seen / skip</option>
                  </select>
                </label>
              </li>)}
            </ol>
          </section>)}
        </div>
        <fieldset className={styles.preference} disabled={busy || scenario.submitted}>
          <legend>Which list would you prefer overall?</legend>
          {[["A", "Prefer List A"], ["B", "Prefer List B"], ["tie", "About equal"]].map(([value, label]) =>
            <label key={value}><input type="radio" name="preference" value={value} checked={draft.preference === value}
              onChange={() => saveDraft({ ...draft, preference: value })} />{label}</label>)}
        </fieldset>
        <label className={styles.notes}>Notes (optional)
          <textarea maxLength={2000} rows={3} disabled={busy || scenario.submitted} value={draft.notes}
            placeholder="What makes a list better or worse? Please avoid names or other personal details."
            onChange={event => saveDraft({ ...draft, notes: event.target.value })} />
        </label>
        {scenario.submitted ? <p className={styles.notice}>This scenario is already saved.</p>
          : <button type="button" disabled={busy} onClick={() => void submit()}>{busy ? "Saving…" : "Save scenario and continue"}</button>}
      </section>}
      {session.completed && <section className={styles.export}>
        <h2>Review complete</h2>
        <p>Export your ratings for offline analysis. Completed exports include the comparison identities and blind assignments.</p>
        <div className={styles.actions}><button disabled={busy} onClick={() => void download("json")}>Export JSON</button>
          <button disabled={busy} onClick={() => void download("csv")}>Export CSV</button>
          <button disabled={busy} onClick={() => void start()}>Start another anonymous review</button></div>
      </section>}
    </>}
  </main>;
}
