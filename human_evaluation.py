"""Local blind review of saved evaluation outputs. No engine or API calls."""
import argparse
import copy
import csv
import hashlib
import io
import json
import math
import random
import re
import secrets
import statistics
import threading
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from uuid import uuid4

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "evaluation_results/recommendation_quality.json"
DEFAULT_STORE = ROOT / "evaluation_results/human"
SIDES = ("A", "B")
PREFERENCES = ("A", "B", "tie")


class EvaluationError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def blind_assignments(scenario_ids, algorithms, seed):
    if len(algorithms)!=2 or algorithms[0]==algorithms[1]:
        raise EvaluationError("Choose two different ranking approaches.")
    rng = random.Random(seed)
    orientations = [i % 2 for i in range(len(scenario_ids))]
    rng.shuffle(orientations)
    return {identity:dict(zip(SIDES, algorithms[orientation:]+algorithms[:orientation]))
            for identity,orientation in zip(scenario_ids,orientations)}


def load_scenarios(source, catalog, snapshot, algorithms):
    output = read_json(source)
    if output.get("artifact_sha256",{}).get("data/tmdb_movies.csv") != digest(catalog):
        raise EvaluationError("Catalog differs from the measured output. Regenerate the existing evaluation CLI output first.")
    expected_snapshot = output.get("artifact_sha256",{}).get("evaluation_data/external_movies.json")
    if expected_snapshot and expected_snapshot!=digest(snapshot):
        raise EvaluationError("External metadata differs from the measured output.")
    needed = {movie["tmdb_id"] for scenario in output["scenarios"] if scenario["status"]=="available"
              for entry in [scenario,*scenario.get("experiments",{}).values()]
              for movie in entry["recommendations"]}
    needed.update(seed["tmdb_id"] for scenario in output["scenarios"] if scenario["status"]=="available" for seed in scenario["seeds"])
    metadata = {}
    with Path(catalog).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            identity = int(row["tmdb_id"])
            if identity in needed:
                metadata[identity] = row
    for row in read_json(snapshot)["movies"]:
        metadata.setdefault(row["tmdb_id"],row)

    def movie(record):
        identity = record["tmdb_id"]
        if identity not in metadata:
            raise EvaluationError(f"Metadata unavailable for TMDB {identity}.")
        row = metadata[identity]
        year = int(float(row["release_year"])) if row.get("release_year") else None
        if row["title"] != record["title"] or year != record["year"]:
            raise EvaluationError(f"Identity mismatch for TMDB {identity}.")
        poster = row.get("poster_path")
        return {"tmdb_id":identity,"title":row["title"],"year":year,"genres":row.get("genres") or "",
                "overview":row.get("overview") or "",
                "poster_url":"https://image.tmdb.org/t/p/w342"+poster if poster and poster.startswith("/") else None}

    scenarios = []
    for row in output["scenarios"]:
        if row["status"]!="available":
            raise EvaluationError(f"Scenario {row['code']} unavailable: {row.get('reason','unknown reason')}")
        variants = {"baseline":row,**row["experiments"]}
        lists = {}
        for algorithm in algorithms:
            if algorithm not in variants:
                raise EvaluationError(f"Approach {algorithm} unavailable for scenario {row['code']}.")
            entry = variants[algorithm]
            movies = [movie(record) for record in entry["recommendations"]]
            if not movies or len(movies)>10 or len({m["tmdb_id"] for m in movies})!=len(movies):
                raise EvaluationError("Each measured list must contain one to ten unique movie IDs.")
            lists[algorithm] = {"movies":movies,"metrics":{key:entry["metrics"].get(key) for key in
                                  ("mean_genre_jaccard","genre_diversity","anchor_containment_top10")}}
        scenarios.append({"scenario_id":row["code"],"name":row["name"],"favorites":[movie(seed) for seed in row["seeds"]],"variants":lists})
    if len(scenarios)!=8 or len({s["scenario_id"] for s in scenarios})!=8:
        raise EvaluationError("The existing eight evaluation scenarios are required.")
    return scenarios


class EvaluationStore:
    def __init__(self, scenarios, directory=DEFAULT_STORE, seed=None, source_sha256="", algorithms=("baseline","genre_votes")):
        self.scenarios, self.directory = scenarios, Path(directory)
        self.seed, self.source_sha256, self.algorithms = seed, source_sha256, list(algorithms)
        self.lock = threading.RLock()

    def path(self, identity):
        if not isinstance(identity,str) or not re.fullmatch(r"[0-9a-f]{32}",identity):
            raise EvaluationError("Invalid anonymous evaluation ID.")
        return self.directory / "sessions" / (identity+".json")

    def load(self, identity):
        path = self.path(identity)
        if not path.exists():
            raise EvaluationError("Evaluation not found.",404)
        return read_json(path)

    def create(self):
        with self.lock:
            identity = uuid4().hex
            seed = self.seed if self.seed is not None else secrets.randbits(64)
            assignments = blind_assignments([s["scenario_id"] for s in self.scenarios],self.algorithms,seed)
            session = {"schema_version":1,"evaluation_id":identity,"assignment_seed":seed,"source_sha256":self.source_sha256,
                       "scenarios":copy.deepcopy(self.scenarios),"blind_assignments":assignments,"submissions":{}}
            atomic_json(self.path(identity),session)
            return self.public(session)

    @staticmethod
    def public(session):
        scenarios = []
        for row in session["scenarios"]:
            identity = row["scenario_id"]
            assignment = session["blind_assignments"][identity]
            scenarios.append({"scenario_id":identity,"name":row["name"],"favorites":row["favorites"],
                "lists":{side:row["variants"][assignment[side]]["movies"] for side in SIDES},
                "submitted":identity in session["submissions"]})
        return {"evaluation_id":session["evaluation_id"],"scenarios":scenarios,
                "completed":len(session["submissions"])==len(scenarios)}

    def get(self, identity):
        with self.lock:
            return self.public(self.load(identity))

    def submit(self, identity, payload):
        with self.lock:
            session = self.load(identity)
            if not isinstance(payload,dict) or set(payload)!={"scenario_id","ratings","preference","notes"}:
                raise EvaluationError("Expected scenario, movie ratings, list preference and optional notes.")
            scenario = next((s for s in session["scenarios"] if s["scenario_id"]==payload["scenario_id"]),None)
            if scenario is None:
                raise EvaluationError("Unknown scenario.")
            scenario_id = scenario["scenario_id"]
            if scenario_id in session["submissions"]:
                raise EvaluationError("This scenario has already been submitted.",409)
            if payload["preference"] not in PREFERENCES:
                raise EvaluationError("Choose List A, List B or about equal.")
            if not isinstance(payload["notes"],str) or len(payload["notes"])>2000:
                raise EvaluationError("Notes must be text of at most 2,000 characters.")
            if not isinstance(payload["ratings"],dict) or set(payload["ratings"])!=set(SIDES):
                raise EvaluationError("Both lists need movie ratings or explicit skips.")
            assignment = session["blind_assignments"][scenario_id]
            validated = {}
            for side in SIDES:
                expected = scenario["variants"][assignment[side]]["movies"]
                ratings = payload["ratings"][side]
                if not isinstance(ratings,list) or len(ratings)!=len(expected):
                    raise EvaluationError("Rate or skip every movie in both lists.")
                for item,movie in zip(ratings,expected):
                    if not isinstance(item,dict) or set(item)!={"tmdb_id","rating"} or type(item["tmdb_id"]) is not int or item["tmdb_id"]!=movie["tmdb_id"]:
                        raise EvaluationError("Movie IDs/order differ from the assigned list.")
                    rating = item["rating"]
                    if rating is not None and (type(rating) is not int or not 1<=rating<=5):
                        raise EvaluationError("Ratings must be integers 1-5, or null for an explicit skip.")
                validated[side] = copy.deepcopy(ratings)
            session["submissions"][scenario_id] = {"ratings":validated,"preference":payload["preference"],"notes":payload["notes"]}
            atomic_json(self.path(identity),session)
            return self.public(session)

    def export(self, identity):
        with self.lock:
            session = self.load(identity)
            if len(session["submissions"])!=len(session["scenarios"]):
                raise EvaluationError("Complete all scenarios before exporting the blind mapping.",403)
            records = []
            for scenario in session["scenarios"]:
                code = scenario["scenario_id"]
                assignment = session["blind_assignments"][code]
                submitted = session["submissions"][code]
                lists = {}
                for side in SIDES:
                    variant = scenario["variants"][assignment[side]]
                    lists[side] = {"algorithm_id":assignment[side],"metrics":variant["metrics"],
                        "movies":[{"tmdb_id":movie["tmdb_id"],"rank":rank,"rating":rating["rating"],"skipped":rating["rating"] is None}
                                  for rank,(movie,rating) in enumerate(zip(variant["movies"],submitted["ratings"][side]),1)]}
                records.append({"scenario_id":code,"evaluation_id":identity,"source_sha256":session["source_sha256"],"blind_assignment":assignment,
                    "lists":lists,"preference":submitted["preference"],"notes":submitted["notes"]})
            result = {"schema_version":1,"completed":True,"evaluation_id":identity,"assignment_seed":session["assignment_seed"],
                      "source_sha256":session["source_sha256"],"records":records}
            atomic_json(self.directory / "completed" / (identity+".json"),result)
            return result


CSV_FIELDS = ["evaluation_id","scenario_id","source_sha256","assignment_seed","list_side","algorithm_id","algorithm_a","algorithm_b",
              "tmdb_id","rank","rating","skipped","preference","notes","mean_genre_jaccard","genre_diversity","anchor_containment_top10"]


def export_csv(data):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output,fieldnames=CSV_FIELDS)
    writer.writeheader()
    for record in data["records"]:
        for side in SIDES:
            entry = record["lists"][side]
            for movie in entry["movies"]:
                writer.writerow({"evaluation_id":data["evaluation_id"],"scenario_id":record["scenario_id"],"source_sha256":data["source_sha256"],
                    "assignment_seed":data["assignment_seed"],"list_side":side,"algorithm_id":entry["algorithm_id"],
                    "algorithm_a":record["blind_assignment"]["A"],"algorithm_b":record["blind_assignment"]["B"],
                    **movie,"preference":record["preference"],"notes":record["notes"],**entry["metrics"]})
    return output.getvalue()


def load_csv(path):
    grouped = {}
    with Path(path).open(encoding="utf-8-sig",newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames!=CSV_FIELDS:
            raise EvaluationError("Unexpected CSV export columns.")
        for row in reader:
            key = (row["evaluation_id"],row["scenario_id"])
            assignment = {"A":row["algorithm_a"],"B":row["algorithm_b"]}
            record = grouped.setdefault(key,{"scenario_id":key[1],"evaluation_id":key[0],"blind_assignment":assignment,
                "preference":row["preference"],"notes":row["notes"],"source_sha256":row["source_sha256"],"lists":{}})
            if any(record[k]!=v for k,v in [("blind_assignment",assignment),("preference",row["preference"]),("notes",row["notes"]),("source_sha256",row["source_sha256"])]):
                raise EvaluationError("Inconsistent CSV comparison metadata.")
            side = row["list_side"]
            if side not in SIDES or assignment[side]!=row["algorithm_id"]:
                raise EvaluationError("Invalid CSV blind assignment.")
            metrics = {key:float(row[key]) if row[key] else None for key in ["mean_genre_jaccard","genre_diversity","anchor_containment_top10"]}
            entry = record["lists"].setdefault(side,{"algorithm_id":row["algorithm_id"],"metrics":metrics,"movies":[]})
            if entry["metrics"]!=metrics or row["skipped"] not in {"True","False"}:
                raise EvaluationError("Invalid CSV rating metadata.")
            entry["movies"].append({"tmdb_id":int(row["tmdb_id"]),"rank":int(row["rank"]),"rating":int(row["rating"]) if row["rating"] else None,"skipped":row["skipped"]=="True"})
    for record in grouped.values():
        for entry in record["lists"].values():
            entry["movies"].sort(key=lambda movie:movie["rank"])
    return list(grouped.values())


def analyze(records):
    """Descriptive, skip-aware summaries. A comparison is the unit for win rates."""
    seen, algorithms, scenarios = set(), defaultdict(list), defaultdict(lambda:defaultdict(list))
    outcomes = defaultdict(lambda:{"comparisons":0,"wins":0,"ties":0})
    agreement = {"genre":[],"diversity":[]}
    if len({record.get("source_sha256") for record in records})>1:
        raise EvaluationError("Analyze different measured-output versions separately.")
    for record in records:
        key = (record["evaluation_id"],record["scenario_id"])
        if key in seen:
            raise EvaluationError("Duplicate evaluation/scenario in analysis inputs.")
        seen.add(key)
        if record["preference"] not in PREFERENCES or set(record["lists"])!=set(SIDES):
            raise EvaluationError("Invalid comparison record.")
        if set(record["blind_assignment"])!=set(SIDES) or len(set(record["blind_assignment"].values()))!=2:
            raise EvaluationError("Invalid algorithm pair.")
        for side in SIDES:
            entry = record["lists"][side]
            algorithm = entry["algorithm_id"]
            if record["blind_assignment"][side]!=algorithm:
                raise EvaluationError("Algorithm mapping differs from blind assignment.")
            for value in entry["metrics"].values():
                if value is not None and (type(value) not in {int,float} or not math.isfinite(value) or not 0<=value<=1):
                    raise EvaluationError("Invalid genre diagnostic metric.")
            movie_ids = set()
            for rank,movie in enumerate(entry["movies"],1):
                value = movie["rating"]
                if (type(movie["tmdb_id"]) is not int or movie["tmdb_id"]<=0 or movie["tmdb_id"] in movie_ids or movie["rank"]!=rank
                    or movie["skipped"] is not (value is None) or value is not None and (type(value) is not int or not 1<=value<=5)):
                    raise EvaluationError("Invalid movie identity, rank, rating or skip flag.")
                movie_ids.add(movie["tmdb_id"])
            if not movie_ids:
                raise EvaluationError("Empty recommendation list.")
            algorithms[algorithm].append(entry)
            scenarios[record["scenario_id"]][algorithm].append(entry)
            outcome = outcomes[algorithm]
            outcome["comparisons"]+=1
            outcome["wins"]+=int(record["preference"]==side)
            outcome["ties"]+=int(record["preference"]=="tie")
        for key,metric in [("genre","mean_genre_jaccard"),("diversity","genre_diversity")]:
            a,b = (record["lists"][side]["metrics"].get(metric) for side in SIDES)
            if a is None or b is None:
                continue
            if abs(a-b)<1e-12:
                agreement[key].append("equal_proxy")
            elif record["preference"]=="tie":
                agreement[key].append("human_tie")
            else:
                agreement[key].append("agrees" if record["preference"]==("A" if a>b else "B") else "disagrees")

    def summary(entries):
        ratings = [movie["rating"] for entry in entries for movie in entry["movies"] if movie["rating"] is not None]
        means = [statistics.mean(values) for entry in entries if (values := [m["rating"] for m in entry["movies"] if m["rating"] is not None])]
        return {"rated_movies":len(ratings),"skipped_movies":sum(m["rating"] is None for e in entries for m in e["movies"]),
                "mean_rating":statistics.mean(ratings) if ratings else None,"median_rating":statistics.median(ratings) if ratings else None,
                "mean_list_rating":statistics.mean(means) if means else None,
                "mean_genre_overlap":statistics.mean(values) if (values := [e["metrics"]["mean_genre_jaccard"] for e in entries if e["metrics"].get("mean_genre_jaccard") is not None]) else None,
                "mean_genre_diversity":statistics.mean(values) if (values := [e["metrics"]["genre_diversity"] for e in entries if e["metrics"].get("genre_diversity") is not None]) else None}
    result = {algorithm:summary(entries) for algorithm,entries in algorithms.items()}
    for algorithm,entry in result.items():
        outcome = outcomes[algorithm]
        entry.update(outcome,win_rate=outcome["wins"]/outcome["comparisons"],tie_rate=outcome["ties"]/outcome["comparisons"])
    by_scenario = {}
    for code,entries in scenarios.items():
        by_scenario[code] = {}
        subset = [r for r in records if r["scenario_id"]==code]
        for algorithm,lists in entries.items():
            own = [r for r in subset if algorithm in r["blind_assignment"].values()]
            wins = sum(r["preference"] in SIDES and r["blind_assignment"][r["preference"]]==algorithm for r in own)
            ties = sum(r["preference"]=="tie" for r in own)
            by_scenario[code][algorithm] = dict(summary(lists),comparisons=len(own),wins=wins,win_rate=wins/len(own),tie_rate=ties/len(own))
    return {"comparison_count":len(seen),"anonymous_evaluations":len({r["evaluation_id"] for r in records}),"algorithms":result,"by_scenario":by_scenario,
            "proxy_preference_agreement":{key:{label:values.count(label) for label in ["agrees","disagrees","human_tie","equal_proxy"]} for key,values in agreement.items()},
            "limitations":["Descriptive summaries only; no statistical significance is claimed.",
                "Skips are excluded, never imputed. Movie-weighted and equally weighted list means are both reported; missingness can bias comparisons.",
                "Win rate uses all comparisons, including ties. Repeated slots/evaluations are not independent observations.",
                "Genre agreement is agreement of list preference with the higher tag-overlap proxy, not proof of relevance. Diversity is a genre proxy, not human-rated variety."]}


def create_server(store, port=8013, origin="http://localhost:3000"):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # No client IPs or evaluation IDs in access logs.

        def reply(self, status, data, content_type="application/json"):
            body = json.dumps(data,ensure_ascii=False,allow_nan=False).encode() if content_type=="application/json" else data.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type",content_type+"; charset=utf-8")
            self.send_header("Cache-Control","no-store")
            self.send_header("Content-Length",str(len(body)))
            if self.headers.get("Origin")==origin:
                self.send_header("Access-Control-Allow-Origin",origin)
            self.end_headers()
            self.wfile.write(body)

        def allowed(self):
            if self.headers.get("Host") not in {f"localhost:{server.server_port}",f"127.0.0.1:{server.server_port}"} or self.headers.get("Origin") not in {None,origin}:
                raise EvaluationError("Only the configured local frontend may access this service.",403)

        def do_OPTIONS(self):
            try:
                self.allowed()
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin",origin)
                self.send_header("Access-Control-Allow-Methods","GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers","Content-Type")
                self.end_headers()
            except EvaluationError as exc:
                self.reply(exc.status,{"error":str(exc)})

        def handle_request(self):
            try:
                self.allowed()
                parsed = urlparse(self.path)
                parts = parsed.path.strip("/").split("/")
                if self.command=="POST":
                    size = int(self.headers.get("Content-Length","0"))
                    if not 0<size<=100000:
                        raise EvaluationError("Invalid request size.")
                    payload = json.loads(self.rfile.read(size))
                    if parts==["sessions"] and payload=={}:
                        self.reply(201,store.create())
                        return
                    if len(parts)==3 and parts[0]=="sessions" and parts[2]=="submit":
                        self.reply(200,store.submit(parts[1],payload))
                        return
                elif self.command=="GET" and parts[0]=="sessions":
                    if len(parts)==2:
                        self.reply(200,store.get(parts[1]))
                        return
                    if len(parts)==3 and parts[2]=="export":
                        kind = parse_qs(parsed.query).get("format",["json"])[0]
                        if kind not in {"json","csv"}:
                            raise EvaluationError("Export format must be JSON or CSV.")
                        data = store.export(parts[1])
                        self.reply(200,data if kind=="json" else export_csv(data),"application/json" if kind=="json" else "text/csv")
                        return
                raise EvaluationError("Not found.",404)
            except EvaluationError as exc:
                self.reply(exc.status,{"error":str(exc)})
            except (ValueError,KeyError,TypeError):
                self.reply(400,{"error":"Invalid evaluation data."})

        do_GET = handle_request
        do_POST = handle_request

    server = ThreadingHTTPServer(("127.0.0.1",port),Handler)
    return server


def serve(store, port=8013, origin="http://localhost:3000"):
    server = create_server(store,port,origin)
    print(f"Local blind evaluation service: http://127.0.0.1:{server.server_port} (no ranking/API calls)",flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command",required=True)
    run = commands.add_parser("serve")
    run.add_argument("--source",type=Path,default=DEFAULT_SOURCE)
    run.add_argument("--catalog",type=Path,default=ROOT/"data/tmdb_movies.csv")
    run.add_argument("--snapshot",type=Path,default=ROOT/"evaluation_data/external_movies.json")
    run.add_argument("--store",type=Path,default=DEFAULT_STORE)
    run.add_argument("--challenger",default="genre_votes",choices=["genre_aware","popularity_confidence","vote_reliability","genre_votes","audience_proxy_filter","mean_story","strongest_story"])
    run.add_argument("--seed",type=int)
    run.add_argument("--port",type=int,default=8013)
    run.add_argument("--origin",default="http://localhost:3000")
    analysis = commands.add_parser("analyze")
    analysis.add_argument("files",type=Path,nargs="+")
    analysis.add_argument("--output",type=Path)
    args = parser.parse_args()
    try:
        if args.command=="serve":
            algorithms = ["baseline",args.challenger]
            scenarios = load_scenarios(args.source,args.catalog,args.snapshot,algorithms)
            serve(EvaluationStore(scenarios,args.store,args.seed,digest(args.source),algorithms),args.port,args.origin)
        else:
            records = []
            for path in args.files:
                if path.suffix.lower()==".csv":
                    records.extend(load_csv(path))
                else:
                    data = read_json(path)
                    if data.get("completed") is not True:
                        raise EvaluationError("Only completed exports can be analyzed.")
                    records.extend(data["records"])
            result = analyze(records)
            if args.output:
                atomic_json(args.output,result)
            print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
    except (ValueError,KeyError,TypeError) as exc:
        parser.error(str(exc))


if __name__=="__main__":
    main()
