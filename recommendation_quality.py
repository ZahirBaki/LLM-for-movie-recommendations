"""Read-only offline diagnostics and isolated experiments; never imported by API."""
import argparse
import ast
import contextlib
import hashlib
import inspect
import io
import json
import math
import sys
from pathlib import Path

from experimental_ranking import genre_votes_scores
import numpy as np
import pandas as pd
from chat_service import MovieIntent
from chat_recommender import ENGINE_FIELDS
from content_recommender import ContentRecommender
from movie_tools import MovieDetails
from runtime_movie_embeddings import RuntimeMovieEmbeddings

SCENARIOS = [
    ("A", "Serious science fiction", [(157336,"Interstellar",2014),(329865,"Arrival",2016)], ["Drama","Science Fiction"]),
    ("B", "Animated family", [(354912,"Coco",2017),(150540,"Inside Out",2015),(269149,"Zootopia",2016)], ["Animation","Family"]),
    ("C", "Crime drama", [(238,"The Godfather",1972),(769,"GoodFellas",1990)], ["Crime","Drama"]),
    ("D", "Psychological thriller", [(807,"Se7en",1995),(11324,"Shutter Island",2010)], ["Mystery","Thriller"]),
    ("E", "Action/superhero", [(36657,"X-Men",2000),(24428,"The Avengers",2012)], ["Action","Science Fiction"]),
    ("F", "Romantic drama", [(11036,"The Notebook",2004),(313369,"La La Land",2016)], ["Romance","Drama"]),
    ("G", "External seed", [(1248832,"Digger",2026)], ["Comedy","Drama"]),
    ("H", "Mixed local/external", [(157336,"Interstellar",2014),(1248832,"Digger",2026)], ["Drama","Science Fiction"]),
]
ARTIFACTS = ["data/tmdb_movies.csv","data/movie_embeddings.npy","data/svd_movie_factors.npy",
    "data/svd_movie_index.csv","data/movie_mapping.csv","recommendation_engine.py","content_recommender.py",
    "svd_recommender.py","runtime_movie_embeddings.py","evaluation_data/external_movies.json",
    "recommendation_quality.py","recommendation_quality_report.py"]


def artifact_hashes(paths=ARTIFACTS):
    result = {}
    for name in paths:
        path = Path(name)
        if path.exists():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024*1024), b""):
                    digest.update(chunk)
            result[name] = digest.hexdigest()
    return result


def number(value):
    return float(value) if value is not None and pd.notna(value) and np.isfinite(value) else None


def tags(value):
    return {part.strip().casefold() for part in str(value or "").split(",") if part.strip()}


def jaccard(left, right):
    return len(left & right)/len(left | right) if left | right else None


def ranking_metrics(recommended, relevant, k=10):
    """Requires actual labels; synthetic ground truth is used only in tests."""
    if k < 1:
        raise ValueError("k must be positive")
    relevant = set(relevant)
    if not relevant:
        return {"recall": None, "ndcg": None}
    seen = set()
    hits = []
    for identity in list(recommended)[:k]:
        hits.append(int(identity in relevant and identity not in seen))
        seen.add(identity)
    dcg = sum(hit/math.log2(rank+2) for rank, hit in enumerate(hits))
    ideal = sum(1/math.log2(rank+2) for rank in range(min(k, len(relevant))))
    return {"recall": sum(hits)/len(relevant), "ndcg": dcg/ideal}


def distribution(values):
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return dict(zip(["min","p25","median","p75","max"], map(float, np.percentile(clean,[0,25,50,75,100]))))


def objective_metrics(records, seed_ids, seed_genres, anchors=(), excluded=()):
    ids = [row["tmdb_id"] for row in records]
    genres = [tags(row["genres"]) for row in records]
    def fraction(k, predicate):
        chosen = genres[:k]
        return sum(predicate(values) for values in chosen)/len(chosen) if chosen else None
    pairs = [1-jaccard(a,b) for i,a in enumerate(genres) for b in genres[i+1:] if a | b]
    overlaps = [jaccard(values,seed_genres) for values in genres]
    overlaps = [value for value in overlaps if value is not None]
    return {
        "count": len(ids), "mean_genre_jaccard": float(np.mean(overlaps)) if overlaps else None,
        "genre_consistency_top5": fraction(5,lambda values: bool(values & seed_genres)),
        "genre_consistency_top10": fraction(10,lambda values: bool(values & seed_genres)),
        "anchor_containment_top5": fraction(5,lambda values: set(anchors)<=values) if anchors else None,
        "anchor_containment_top10": fraction(10,lambda values: set(anchors)<=values) if anchors else None,
        "popularity": distribution([row["popularity"] for row in records]),
        "vote_count": distribution([row["vote_count"] for row in records]),
        "genre_diversity": float(np.mean(pairs)) if pairs else None,
        "duplicate_rate": (len(ids)-len(set(ids)))/len(ids) if ids else 0,
        "seed_exclusion_correct": not bool(set(ids)&set(seed_ids)),
        "explicit_exclusion_correct": not bool(set(ids)&set(excluded)),
    }


def production_content_weights():
    tree = ast.parse(inspect.getsource(ContentRecommender))
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=="base_weights" for target in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("Production weights unavailable; do not infer them")


class QualityDiagnostics:
    def __init__(self, engine, external_movies=(), embedding_cache=None):
        self.engine = engine
        self.catalog = engine.content.movies
        self.positions = {int(row.tmdb_id): index for index,row in self.catalog.iterrows()}
        self.external = {movie.tmdb_id: movie for movie in external_movies}
        if embedding_cache is None:
            indices = [0,len(self.catalog)//2,len(self.catalog)-1]
            embedding_cache = RuntimeMovieEmbeddings(engine.content.overview_embeddings.shape[1], compatibility_samples=[
                (str(self.catalog.iloc[index]["overview"]),engine.content.overview_embeddings[index]) for index in indices])
        self.embedding_cache = embedding_cache

    def prepare(self, ids, filters):
        if not ids or len(ids)>20 or any(type(identity) is not int or identity<=0 for identity in ids):
            raise ValueError("Provide one to twenty positive TMDB IDs")
        allowed = set(ENGINE_FIELDS)-{"top_n"}
        if set(filters)-allowed:
            raise ValueError("Unknown filters: "+", ".join(sorted(set(filters)-allowed)))
        intent = MovieIntent(intent="recommend", **filters)
        local, external, seeds = [], [], []
        for identity in dict.fromkeys(ids):
            if identity in self.positions:
                row = self.catalog.iloc[self.positions[identity]]
                year = int(row.release_year) if pd.notna(row.release_year) else None
                movie = dict(tmdb_id=identity,title=str(row.title),year=year,genres=str(row.genres),actors=str(row.actors),director=str(row.director))
                local.append({key:movie[key] for key in ["tmdb_id","title","year"]})
            elif identity in self.external:
                query = self.embedding_cache.get(self.external[identity])
                external.append(query)
                movie = dict(tmdb_id=identity,title=query.title,year=query.year,genres=query.genres,actors=query.actors,director=query.director)
            else:
                raise ValueError(f"TMDB ID {identity} is unavailable locally and has no verified external snapshot")
            seeds.append(movie)
        inputs = {field:getattr(intent,field) for field in ENGINE_FIELDS if field!="top_n"}
        inputs["favorite_movies"] = local
        if external:
            inputs["external_queries"] = external
        indices = self.catalog.index[self.catalog.tmdb_id.isin([movie["tmdb_id"] for movie in local])].tolist()
        vectors = [self.engine.content.overview_embeddings[index] for index in indices]+[query.embedding for query in external]
        vector_ids = [int(self.catalog.iloc[index].tmdb_id) for index in indices]+[query.tmdb_id for query in external]
        cosine = np.stack(vectors) @ self.engine.content.overview_embeddings.T
        _,genres,actors,directors = self.engine.content.validate_preferences([],inputs["favorite_genres"],inputs["favorite_actors"],inputs["favorite_directors"])
        active = {"story_score":True,"genre_score":bool(genres or any(tags(movie["genres"]) for movie in seeds)),
            "actor_score":bool(actors or any(tags(movie["actors"]) for movie in seeds)),
            "director_score":bool(directors or any(movie["director"].strip() for movie in seeds))}
        weights = production_content_weights()
        total = sum(value for key,value in weights.items() if active[key])
        weights = {key:value/total if active[key] else 0 for key,value in weights.items()}
        content = self.engine.content.get_scores(favorite_tmdb_ids=[movie["tmdb_id"] for movie in local],
            favorite_genres=genres,favorite_actors=actors,favorite_directors=directors,external_queries=external)
        components = {int(row.tmdb_id):row for _,row in content.iterrows()}
        return inputs,seeds,cosine,vector_ids,weights,components

    def describe(self, frame, seeds, cosine, vector_ids, weights, components):
        records = []
        content_ranks = {identity: rank for rank, identity in enumerate(
            sorted(components, key=lambda identity: -float(components[identity].content_score)), 1)}
        seed_lookup = {movie["tmdb_id"]:movie for movie in seeds}
        union = set().union(*(tags(movie["genres"]) for movie in seeds))
        for rank,(_,row) in enumerate(frame.iterrows(),1):
            identity = int(row.tmdb_id)
            if identity not in self.positions:
                raise ValueError("Engine returned a candidate outside the catalog")
            position = self.positions[identity]
            metadata = self.catalog.iloc[position]
            similarities = cosine[:,position]
            best = int(np.argmax(similarities))
            component = components.get(identity)
            contributions = {key:number(component[key])*weight for key,weight in weights.items()} if component is not None else None
            content_score = number(component.content_score) if component is not None else None
            bonus = number(component.multi_match_bonus) if component is not None else None
            if contributions is not None and not np.isclose(sum(contributions.values())+bonus,content_score,atol=1e-6):
                raise ValueError("Score reconstruction differs from production; diagnostic assumptions require review")
            final = number(row.get("final_score"))
            source = str(row.get("recommendation_source","unknown"))
            rank_score = final if final is not None else content_score if source in {"content","content_fallback"} else None
            record = {"rank":rank,"content_rank":content_ranks.get(identity),"tmdb_id":identity,"title":str(metadata.title),"year":int(metadata.release_year) if pd.notna(metadata.release_year) else None,
                "engine_title":str(row.title),"engine_final_score":final,"ranking_score":rank_score,
                "ranking_score_source":"final_score" if final is not None else "content_score" if rank_score is not None else None,
                "content_score":content_score,"collaborative_score":number(row.get("svd_score")),
                "overview_cosine_mean":float(similarities.mean()),"overview_cosine_max":float(similarities.max()),
                "story_raw":float(.7*similarities.mean()+.3*similarities.max()),
                "content_components":{key:number(component[key]) for key in weights} if component is not None else None,
                "weighted_content_components":contributions,"multi_match_bonus":bonus,
                "genre_overlap":jaccard(tags(metadata.genres),union),"genres":str(metadata.genres),
                "popularity":number(metadata.get("popularity")),"vote_average":number(metadata.get("vote_average")),"vote_count":number(metadata.get("vote_count")),
                "most_similar_overview_seed":seed_lookup[vector_ids[best]],
                "per_seed_overview_similarity":[{"tmdb_id":seed_id,"cosine":float(value)} for seed_id,value in zip(vector_ids,similarities)],
                "recommendation_source":source,"engine_mode":str(row.get("engine_mode","unavailable")),
                "ranking_kind":"content_only" if source in {"content","content_fallback"} else "hybrid_cold" if source=="content_cold" else "collaborative_primary"}
            if "experiment_score" in row:
                record["experiment_score"] = number(row.experiment_score)
            records.append(record)
        return records

    def diagnose(self, ids, top_n=10, filters=None, excluded=(), anchors=(), experiments=False, probes=()):
        if not 1<=top_n<=100:
            raise ValueError("top_n must be between 1 and 100")
        filters = filters or {}
        excluded = set(excluded)
        with contextlib.redirect_stdout(io.StringIO()):
            inputs,seeds,cosine,vector_ids,weights,components = self.prepare(ids,filters)
            wanted = max(200,top_n) if experiments else top_n
            frame = self.engine.recommend(**inputs,top_n=min(len(self.catalog),wanted+len(excluded)))
        if frame.empty:
            frame = pd.DataFrame(columns=["tmdb_id","title"])
        frame = frame.loc[~frame.tmdb_id.isin(excluded)].head(wanted).copy()
        records = self.describe(frame.head(top_n),seeds,cosine,vector_ids,weights,components)
        union = set().union(*(tags(movie["genres"]) for movie in seeds))
        result = {"status":"available","seeds":seeds,"top_n":top_n,"filters":filters,"excluded_ids":sorted(excluded),
            "content_weights":weights,"recommendations":records,
            "metrics":objective_metrics(records,ids,union,{value.casefold() for value in anchors},excluded),
            "heldout_ranking_metrics":{"status":"unavailable","reason":"Saved SVD was built from full ratings; no leakage-free held-out artifact or split provenance is available. Existing evaluation CSVs describe separately trained models."},
            "seed_attribution_scope":"Strongest seed means highest overview cosine, not causal attribution of a collaborative/hybrid final score."}
        if len(records)>1:
            vectors = self.engine.content.overview_embeddings[[self.positions[row["tmdb_id"]] for row in records]]
            similarities = vectors @ vectors.T
            result["metrics"]["embedding_diversity"] = float(np.mean((1-np.clip(similarities,-1,1))[np.triu_indices(len(records),1)]))
        else:
            result["metrics"]["embedding_diversity"] = None
        result["probes"] = []
        for identity in probes:
            if identity not in self.positions:
                result["probes"].append({"tmdb_id":identity,"status":"unavailable"})
                continue
            probe = pd.DataFrame([dict(tmdb_id=identity,title=str(self.catalog.iloc[self.positions[identity]].title),recommendation_source="content",engine_mode="content diagnostic probe")])
            detail = self.describe(probe,seeds,cosine,vector_ids,weights,components)[0]
            detail["baseline_pool_rank"] = int(np.flatnonzero(frame.tmdb_id.to_numpy()==identity)[0])+1 if identity in set(frame.tmdb_id) else None
            detail["rank"] = None
            result["probes"].append(detail)
        if experiments:
            result["experiments"] = self.experiments(frame,top_n,seeds,cosine,vector_ids,weights,components,ids,union,anchors,excluded,filters)
        return result

    def experiments(self, frame, top_n, seeds, cosine, vector_ids, weights, components, ids, union, anchors, excluded, filters):
        if frame.empty:
            return {}
        records = self.describe(frame,seeds,cosine,vector_ids,weights,components)
        data = frame.copy().reset_index(drop=True)
        base = 1-np.arange(len(data))/max(len(data)-1,1)
        genre = np.array([row["genre_overlap"] or 0 for row in records])
        votes = np.array([max(row["vote_count"] or 0,0) for row in records])
        popular = np.log1p([max(row["popularity"] or 0,0) for row in records])
        popular = np.clip(popular/max(float(np.percentile(popular,95)),1e-12),0,1)
        reliability = votes/(votes+500)
        scores = {"genre_aware":.8*base+.2*genre,"popularity_confidence":.9*base+.1*popular,
            "vote_reliability":.8*base+.2*reliability,"genre_votes":genre_votes_scores(base, genre, votes, [row["popularity"] or 0 for row in records])}
        variants = {name:(data.assign(experiment_score=score),"baseline engine top-200 pool") for name,score in scores.items()}
        audience = data.copy()
        if not union & {"family","animation"}:
            audience = audience[[not bool(tags(row["genres"]) & {"family","animation"}) for row in records]].copy()
        audience["experiment_score"] = base[audience.index]
        variants["audience_proxy_filter"] = (audience,"baseline top-200; remove Family/Animation when absent from seeds (genre proxy, not age certification)")
        eligible = ~self.catalog.tmdb_id.isin(ids).to_numpy()
        all_components = pd.DataFrame([row.to_dict() for row in components.values()])
        metadata = self.catalog.set_index("tmdb_id")
        for field in ["genres","actors","director"]:
            all_components[field] = all_components.tmdb_id.map(metadata[field])
        positions = np.array([self.positions[int(identity)] for identity in all_components.tmdb_id])
        for name,raw in [("mean_story",cosine.mean(axis=0)),("strongest_story",cosine.max(axis=0))]:
            normalizer = max(float(np.maximum(raw[eligible],0).max()),1e-12)
            normalized = np.clip(raw/normalizer,0,1)
            ablation = all_components.copy()
            ablation["experiment_score"] = ablation.content_score+weights["story_score"]*(normalized[positions]-ablation.story_score)
            ablation["recommendation_source"] = "content"
            ablation["engine_mode"] = "isolated content aggregation experiment"
            hard = {key:filters.get(key) for key in ENGINE_FIELDS if key.startswith(("required_","exclude_")) or key in {"min_year","max_year"}}
            ablation = self.engine.apply_filters(ablation,**hard)
            ablation = ablation[~ablation.tmdb_id.isin(excluded)]
            variants[name] = (ablation,"all eligible catalog; change normalized story aggregation only, fixed other features/bonus; no SVD")
        outputs = {}
        for name,(candidates,scope) in variants.items():
            selected = candidates.sort_values(["experiment_score","tmdb_id"],ascending=[False,True],kind="stable").head(top_n)
            described = self.describe(selected,seeds,cosine,vector_ids,weights,components)
            outputs[name] = {"scope":scope,"recommendations":described,
                "metrics":objective_metrics(described,ids,union,{value.casefold() for value in anchors},excluded)}
        return outputs


def load_snapshots(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [MovieDetails.model_validate(movie) for movie in raw["movies"]]


def evaluate(diagnostic):
    results = []
    for code,name,expected,anchors in SCENARIOS:
        print(f"Evaluating {code}: {name}",file=sys.stderr,flush=True)
        try:
            data = diagnostic.diagnose([movie[0] for movie in expected],anchors=anchors,experiments=True,
                probes=[11802] if code in {"A","H"} else [36827,51733,26153] if code=="G" else [])
            if [(movie["tmdb_id"],movie["title"],movie["year"]) for movie in data["seeds"]]!=expected:
                raise ValueError("Scenario identity differs from verified title/year")
            data.update(code=code,name=name)
        except Exception as exc:
            data = {"code":code,"name":name,"status":"unavailable","reason":str(exc),"expected_seeds":expected}
        results.append(data)
    available = [row for row in results if row["status"]=="available"]
    methods = ["baseline",*next((list(row["experiments"]) for row in available),[])]
    aggregate = {}
    for method in methods:
        exposure = [movie["tmdb_id"] for row in available for movie in (row["recommendations"] if method=="baseline" else row["experiments"][method]["recommendations"])]
        metrics = [row["metrics"] if method=="baseline" else row["experiments"][method]["metrics"] for row in available]
        def mean(key):
            values = [data[key] for data in metrics if data[key] is not None]
            return float(np.mean(values)) if values else None
        medians = [m["vote_count"]["median"] for m in metrics if m["vote_count"]]
        aggregate[method] = {"unique_titles":len(set(exposure)),"catalog_coverage":len(set(exposure))/len(diagnostic.catalog),
            "cross_scenario_repeat_rate":(len(exposure)-len(set(exposure)))/len(exposure) if exposure else None,
            "mean_genre_jaccard":mean("mean_genre_jaccard"),"anchor_containment_top10":mean("anchor_containment_top10"),
            "genre_diversity":mean("genre_diversity"),"mean_list_median_votes":float(np.mean(medians)) if medians else None}
    controls = {}
    try:
        controls["Interstellar only"] = diagnostic.diagnose([157336], probes=[11802])
    except Exception as exc:
        controls["Interstellar only"] = {"status":"unavailable","reason":str(exc)}
    metadata_quality = {"missing_fields":{field:int(diagnostic.catalog[field].fillna("").astype(str).str.strip().eq("").sum())
        for field in ["overview","genres","actors","director"] if field in diagnostic.catalog}}
    if "vote_count" in diagnostic.catalog:
        votes = pd.to_numeric(diagnostic.catalog.vote_count,errors="coerce")
        metadata_quality["votes_below_50"] = int(votes.lt(50).sum())
        metadata_quality["missing_vote_count"] = int(votes.isna().sum())
    metadata_quality["available_columns"] = list(diagnostic.catalog.columns)
    return {"schema_version":1,"catalog_size":len(diagnostic.catalog),"artifact_sha256":artifact_hashes(),"controls":controls,
        "catalog_metadata_quality":metadata_quality,
        "scenarios":results,"aggregate":aggregate,"experimental_parameters":{"pool_size":200,"vote_reliability_prior":500,
        "genre_aware":"0.8 rank percentile + 0.2 genre Jaccard","popularity_confidence":"0.9 rank percentile + 0.1 log popularity (pool p95 normalized)",
        "vote_reliability":"0.8 rank percentile + 0.2 votes/(votes+500)","genre_votes":"0.65 rank + 0.2 genre + 0.1 vote reliability + 0.05 popularity"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["diagnose","evaluate"])
    parser.add_argument("--favorites",type=int,nargs="+")
    parser.add_argument("--top-n",type=int,default=10)
    parser.add_argument("--exclude",type=int,nargs="*",default=[])
    parser.add_argument("--filters-json",default="{}")
    parser.add_argument("--snapshot",default="evaluation_data/external_movies.json")
    parser.add_argument("--output")
    parser.add_argument("--report",default="RECOMMENDATION_QUALITY_EVALUATION.md")
    args = parser.parse_args()
    from recommendation_engine import RecommendationEngine
    before = artifact_hashes()
    engine = RecommendationEngine()
    diagnostic = QualityDiagnostics(engine,load_snapshots(args.snapshot))
    filters = json.loads(args.filters_json)
    if args.command=="evaluate":
        data = evaluate(diagnostic)
        output = args.output or "evaluation_results/recommendation_quality.json"
        from recommendation_quality_report import write_report
        write_report(data,args.report)
    else:
        if not args.favorites:
            parser.error("diagnose requires --favorites")
        data = diagnostic.diagnose(args.favorites,args.top_n,filters,args.exclude)
        output = args.output
    if artifact_hashes()!=before:
        raise RuntimeError("An input artifact changed during evaluation")
    payload = json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)
    if output:
        Path(output).parent.mkdir(parents=True,exist_ok=True)
        Path(output).write_text(payload+"\n",encoding="utf-8")
        print("Wrote "+output)
    else:
        print(payload)


if __name__=="__main__":
    main()
