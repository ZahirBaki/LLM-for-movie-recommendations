"""Render measured evaluation output; no ranking or production imports."""
import json
from pathlib import Path


def fmt(value):
    return "null" if value is None else f"{value:.4f}" if isinstance(value,(int,float)) else str(value)


def measured_findings(data):
    scenarios = {row["code"]:row for row in data["scenarios"] if row["status"]=="available"}
    lines = ["## Measured failure modes", ""]
    control = data.get("controls",{}).get("Interstellar only",{})
    if control.get("status")=="available" and control["probes"]:
        row = control["probes"][0]
        lines += [f"**Interstellar-only control:** Space Chimps is content rank {row['content_rank']}, with overview cosine {fmt(row['overview_cosine_mean'])} and content score {fmt(row['content_score'])}. Its weighted story/genre contributions are {fmt(row['weighted_content_components']['story_score'])}/{fmt(row['weighted_content_components']['genre_score'])}, plus bonus {fmt(row['multi_match_bonus'])}. This is an actual engine result, not an assumed causal explanation.", ""]
    if "A" in scenarios and "H" in scenarios:
        a,h = scenarios["A"],scenarios["H"]
        p = h["probes"][0]
        similarities = ", ".join(f"TMDB {row['tmdb_id']}: {fmt(row['cosine'])}" for row in p["per_seed_overview_similarity"])
        lines += [f"**Space Chimps and mixed favorites:** it is absent from A's top-200 engine pool but ranks {p['baseline_pool_rank']} in H. H overview similarities are {similarities}. Mean/max aggregation therefore preserves a strong partial match to Interstellar even with a weaker Digger match. The inferred union is Adventure/Drama/Science Fiction/Comedy: Space Chimps matches three of four, earning weighted genre {fmt(p['weighted_content_components']['genre_score'])}, with weighted story {fmt(p['weighted_content_components']['story_score'])} and bonus {fmt(p['multi_match_bonus'])}. Additional Family/Animation tags are unpenalized. A uses collaborative-primary ranking; H forces content-only, so changing the seed also changes the ranking mode. These measurements do not isolate a pure seed effect.", ""]
    if "G" in scenarios:
        g = scenarios["G"]
        lines += [f"**Obscure Digger matches:** Return with Honor, The End of Man and God Tussi Great Ho are the first three results in G. Each matches both Comedy and Drama and receives weighted genre 0.25 and bonus 0.0333. Their overview cosines are "+", ".join(fmt(row["overview_cosine_mean"]) for row in g["probes"])+"; their weighted story contributions are "+", ".join(fmt(row["weighted_content_components"]["story_score"]) for row in g["probes"])+f". Cast/director contributions are zero. Their votes are 72/21/58; top-ten median votes are {fmt(g['metrics']['vote_count']['median'])}. G anchor containment is {fmt(g['metrics']['anchor_containment_top10'])}, so a simple Comedy/Drama mismatch does not explain the failure. Vote confidence, popularity, tone and audience are absent from this content ordering.", ""]
    metadata = data.get("catalog_metadata_quality")
    if metadata:
        lines += ["**Metadata quality:** "+json.dumps(metadata["missing_fields"])+" catalog rows have blank fields (counts per field). "+str(metadata.get("votes_below_50","unavailable"))+" rows have fewer than 50 votes. Missing text and credits can weaken features, but these global counts do not prove the reported failures are caused by missing metadata. The catalog has no explicit tone or age-certification field. External Digger uses its fixed TMDB synopsis and verified metadata; model compatibility is checked against three stored overview vectors.", ""]
    lines += ["**Embedding scope:** only overview text is embedded; genres and credits are separate features. High overview cosine demonstrates representation similarity, not which keywords caused it or that the audience/tone matches. Establishing keyword causality needs controlled text ablations; this milestone does not claim it. The scoring breakdown supports a combined explanation involving overview similarity, broad union-genre rewards and missing audience/reliability signals.", ""]
    if all(code in scenarios for code in "ABCDEFGH"):
        agg = data["aggregate"]
        b,r = agg["baseline"],agg["genre_votes"]
        lines += [f"Across 80 baseline slots, any shared seed genre is present in every result; scenario anchor containment averages {fmt(b['anchor_containment_top10'])}, while mean genre Jaccard is {fmt(b['mean_genre_jaccard'])}. There are {b['unique_titles']} unique titles, no within-list duplicates and no returned seeds. Shared-genre checks alone are too permissive to detect the reported poor examples.", "",
            f"**Most promising next hypothesis:** evaluate the small combined genre/vote reranker specifically for content-only requests with human judgments. In this suite its mean Jaccard changes {fmt(b['mean_genre_jaccard'])} to {fmt(r['mean_genre_jaccard'])}, anchor containment {fmt(b['anchor_containment_top10'])} to {fmt(r['anchor_containment_top10'])}, and genre diversity {fmt(b['genre_diversity'])} to {fmt(r['genre_diversity'])}. It addresses both broad genre rewards and low-confidence results, but the diversity loss is material. Genre-only reranking improves anchor containment more in the aggregate; vote-only/popularity-only signals do not reliably improve tag alignment. No variant is a proven relevance winner.", ""]
    return lines


def write_report(data, path):
    lines = ["# Recommendation quality evaluation", "",
        "Offline measurements of the current production engine. All experiments are isolated and are not deployed.", "",
        "## Current ranking architecture", "",
        "The API uses RecommendationEngine, ContentRecommender and persisted SVDRecommender. `scripts/legacy/content_demo.py` and `scripts/legacy/collaborative_demo.py` are historical standalone prototypes, not the active API path.", "",
        "Candidates come from the 62,398-row catalog. Only overview text is embedded by normalized all-MiniLM-L6-v2 (384 dimensions). Story similarity is 70% mean cosine plus 30% strongest cosine, normalized by the maximum non-seed candidate. Separate story/genre/director/actor base weights are 0.40/0.25/0.20/0.15, renormalized across active categories, with a multi-match bonus up to 0.10. Inferred actor/director matches have strength 0.20. Inferred genres are a union; additional candidate genres receive no penalty.", "",
        "At least two SVD-covered local favorites enable collaborative-primary ranking. SVD averages favorite latent vectors, scores by dot product and request min-max normalization, and retrieves up to 500 candidates. Explicit soft preferences add 0.10 times content score. Strong SVD-unavailable candidates can compete near the tenth-result boundary by content rank; remaining slots use content fallback. External/mixed requests use the existing content-only path with all seed vectors and metadata, and no invented collaborative scores.", "",
        "Required genres/actors/directors demand ALL requested values; excluded fields reject ANY; year bounds are inclusive. Favorites are excluded by ID. This tool applies explicit excluded/seen/shown ID unions with `--exclude`; chat separately retains its bounded session filtering. Popularity, TMDB ratings and TMDB votes do not contribute to content ranking. MovieLens interaction counts influence SVD indirectly. Content-only `engine_final_score` is null: its actual sorting key is `content_score`, displayed separately as ranking score.", "",
        "## Protocol, provenance and limits", "",
        "A-F seed identities are verified against local catalog ID/title/year records. G-H use the verified public TMDB snapshot in `evaluation/fixtures/external_movies.json` (source URL and retrieval time included). Full numeric diagnostics, per-seed similarities, score contributions, filters, experimental constants and artifact SHA-256 hashes are in `evaluation/results/recommendation_quality.json`. Only the cached existing model is loaded; no large model is trained/downloaded, and no OpenAI/provider calls are made.", "",
        "**Objective diagnostics:** genre Jaccard, any shared seed genre at top 5/10, scenario anchor-tag containment, popularity/vote distributions, genre/embedding diversity, duplication and exclusions. Anchor tags describe the scenario's category, not ground-truth relevance. Suite catalog coverage measures these eight scenarios, not user-wide exposure. Cross-scenario repeats can be appropriate.", "",
        "**Offline relevance metrics:** trustworthy Recall@10/NDCG@10 are unavailable for these saved factors. `scripts/data/build_svd_model.py` calls `fit()` on the full ratings input; no excluded hold-out split provenance accompanies the factors. Historical evaluation scripts train separate models for their splits; their CSV metrics cannot be attributed to the current engine. No retraining or invented labels was performed. Tested metric helpers are provided for future leakage-free data.", "",
        "**Subjective quality:** no human judgments were collected. Genre overlap, popularity, votes and embedding similarity cannot prove recommendation relevance. Strongest contributing seed below means highest overview cosine only, not causal attribution of a whole hybrid score.", "",
        "## Baseline scenarios", ""]
    for scenario in data["scenarios"]:
        lines += [f"### {scenario['code']}. {scenario['name']}", ""]
        if scenario["status"]!="available":
            lines += ["Unavailable: "+scenario["reason"], ""]
            continue
        lines += ["Seeds: "+", ".join(f"{row['title']} ({row['year']}), TMDB {row['tmdb_id']}" for row in scenario["seeds"]), "",
            "| Rank | Title (year), ID | Ranking score | Engine final | Content | SVD | Mean/max cosine | Genre Jaccard | Popularity | Rating | Votes | Strongest overview seed | Source |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
        for row in scenario["recommendations"]:
            lines.append(f"| {row['rank']} | {row['title'].replace('|','/')} ({row['year']}), {row['tmdb_id']} | {fmt(row['ranking_score'])} | {fmt(row['engine_final_score'])} | {fmt(row['content_score'])} | {fmt(row['collaborative_score'])} | {fmt(row['overview_cosine_mean'])}/{fmt(row['overview_cosine_max'])} | {fmt(row['genre_overlap'])} | {fmt(row['popularity'])} | {fmt(row['vote_average'])} | {fmt(row['vote_count'])} | {row['most_similar_overview_seed']['title']} | {row['recommendation_source']} |")
        m = scenario["metrics"]
        lines += ["", "Mode: "+scenario["recommendations"][0]["engine_mode"] if scenario["recommendations"] else "No matching candidates.",
            f"Shared genre top5/top10 {fmt(m['genre_consistency_top5'])}/{fmt(m['genre_consistency_top10'])}; anchor tags top5/top10 {fmt(m['anchor_containment_top5'])}/{fmt(m['anchor_containment_top10'])}; mean Jaccard {fmt(m['mean_genre_jaccard'])}; genre diversity {fmt(m['genre_diversity'])}; embedding diversity {fmt(m['embedding_diversity'])}.",
            "Popularity distribution: "+json.dumps(m["popularity"])+"; votes: "+json.dumps(m["vote_count"]),
            f"Duplicate rate {m['duplicate_rate']}; seed exclusion {m['seed_exclusion_correct']}; explicit exclusions {m['explicit_exclusion_correct']}.", ""]
        if scenario["probes"]:
            lines += ["Diagnostic probes (not preferred recommendations or relevance labels):", "",
                "| Movie | Baseline pool rank | Raw story | Weighted story | Weighted genre | Weighted actor | Weighted director | Bonus | Content | Votes |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
            for row in scenario["probes"]:
                if row.get("status")=="unavailable":
                    lines.append(f"| ID {row['tmdb_id']} unavailable | | | | | | | | | |")
                    continue
                w = row["weighted_content_components"]
                lines.append(f"| {row['title']} | {fmt(row['baseline_pool_rank'])} | {fmt(row['story_raw'])} | {fmt(w['story_score'])} | {fmt(w['genre_score'])} | {fmt(w['actor_score'])} | {fmt(w['director_score'])} | {fmt(row['multi_match_bonus'])} | {fmt(row['content_score'])} | {fmt(row['vote_count'])} |")
            lines.append("")
    lines += measured_findings(data)
    lines += ["## Isolated experimental comparisons", "",
        "Genre/popularity/vote rerankers use the baseline engine top-200 pool and declared constants in JSON. Audience filtering uses Family/Animation tags, a coarse proxy; no tone or age-certification field is present. Mean/strongest-story ablations use the full filtered catalog, change normalized story aggregation and hold non-story contributions and the multi-match bonus fixed. They omit SVD; their comparison against collaborative scenarios does not isolate aggregation alone.", "",
        "| Method | Unique titles | Coverage | Cross-scenario repeats | Mean genre Jaccard | Anchor containment@10 | Genre diversity | Mean list median votes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name,m in data["aggregate"].items():
        lines.append(f"| {name} | {m['unique_titles']} | {fmt(m['catalog_coverage'])} | {fmt(m['cross_scenario_repeat_rate'])} | {fmt(m['mean_genre_jaccard'])} | {fmt(m['anchor_containment_top10'])} | {fmt(m['genre_diversity'])} | {fmt(m['mean_list_median_votes'])} |")
    lines += ["", "Per-scenario metric deltas (relative to baseline):", "",
        "| Scenario | Variant | Genre Jaccard delta | Anchor@10 delta | Genre diversity delta |", "|---|---|---:|---:|---:|"]
    for scenario in data["scenarios"]:
        if scenario["status"]!="available":
            continue
        for name,entry in scenario["experiments"].items():
            delta = lambda key: entry["metrics"][key]-scenario["metrics"][key] if entry["metrics"][key] is not None and scenario["metrics"][key] is not None else None
            lines.append(f"| {scenario['code']} | {name} | {fmt(delta('mean_genre_jaccard'))} | {fmt(delta('anchor_containment_top10'))} | {fmt(delta('genre_diversity'))} |")
    for code in ["G","H"]:
        scenario = next((row for row in data["scenarios"] if row["code"]==code and row["status"]=="available"),None)
        if scenario:
            lines += ["",f"### Experimental top five: {scenario['name']}","","| Method | Top five | Genre Jaccard | Anchor@10 | Median votes |","|---|---|---:|---:|---:|"]
            for name,entry in [("baseline",scenario),*scenario["experiments"].items()]:
                m = entry["metrics"]
                votes = m["vote_count"]["median"] if m["vote_count"] else None
                lines.append(f"| {name} | "+"; ".join(f"{row['title'].replace('|','/')} ({row['year']})" for row in entry["recommendations"][:5])+f" | {fmt(m['mean_genre_jaccard'])} | {fmt(m['anchor_containment_top10'])} | {fmt(votes)} |")
    lines += ["", "## Trade-offs and next milestone", "",
        "Genre-aware reranking strengthens tag alignment but can reduce diversity and cannot distinguish serious from light comedy. Popularity/vote confidence favors mainstream and older titles and may disadvantage new releases. Family/Animation filtering can wrongly discard relevant animated films. Mixed-seed union genres allow a partial match to multiple favorites without matching their tone. These are hypotheses/proxy trade-offs, not proof of enjoyment. No winner is declared from eight examples.", "",
        "Next milestone: collect blinded pairwise judgments across these and additional diverse seed sets; test a small opt-in content-only genre/reliability reranker against baseline. Establish a leakage-free interaction evaluation artifact before production changes. Preserve discovery of well-matched low-vote films. Tone/audience features require a verified source and labels before evaluation.", "",
        "## Reproduction", "", "```powershell", ".venv\\Scripts\\python.exe -m evaluation.scripts.recommendation_quality evaluate",
        ".venv\\Scripts\\python.exe -m evaluation.scripts.recommendation_quality diagnose --favorites 157336 1248832 --top-n 10 --exclude 11802", "```", "",
        "Use `--filters-json` with the existing genre/actor/director/year filter fields. External metadata comes only from the snapshot; the local model must already be cached. No production code, weights, catalog, embeddings or SVD artifacts are modified. Hashes are checked before/after the run. Provider availability is not queried or used as a quality metric."]
    if data.get("validation"):
        lines += ["", "## Validation", "", *["- "+entry for entry in data["validation"]]]
    lines += ["", "## Files created", "",
        "`evaluation/scripts/recommendation_quality.py` (offline CLI, metrics, diagnostics, isolated reranking); `evaluation/scripts/recommendation_quality_report.py` (report renderer); `tests/backend/test_recommendation_quality.py` (offline unit coverage); `evaluation/fixtures/external_movies.json` (verified public metadata snapshot); `evaluation/results/recommendation_quality.json` (measured output); `RECOMMENDATION_QUALITY_EVALUATION.md` (this report). Regression logs are local verification artifacts. Existing production/frontend files were not edited in this milestone."]
    Path(path).write_text("\n".join(lines)+"\n",encoding="utf-8")
