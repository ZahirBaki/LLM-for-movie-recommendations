"""Synthetic, offline blind review/export/analysis tests; no engine or paid calls."""
import copy
import csv
import json
from pathlib import Path
import shutil
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from uuid import uuid4
from human_evaluation import (EvaluationError, EvaluationStore, analyze, atomic_json,
    blind_assignments, create_server, digest, export_csv, load_csv, load_scenarios)


def synthetic_scenarios():
    return [{"scenario_id":str(i),"name":f"Fixture {i}","favorites":[{"tmdb_id":100+i,"title":"Favorite","year":2020}],
        "variants":{algorithm:{"movies":[{"tmdb_id":i*10+j,"title":f"Movie {i*10+j}","year":2020,"genres":"Drama","overview":"Test overview","poster_url":None} for j in [1,2]],
            "metrics":{"mean_genre_jaccard":genre,"genre_diversity":diversity,"anchor_containment_top10":1}}
            for algorithm,genre,diversity in [("baseline",.3,.8),("genre_votes",.7,.2)]}} for i in range(8)]


def submission(scenario, preference="A", ratings=(1,5)):
    return {"scenario_id":scenario["scenario_id"],"preference":preference,"notes":"A note, with commas\nand newlines.",
        "ratings":{side:[{"tmdb_id":movie["tmdb_id"],"rating":ratings[index]} for index,movie in enumerate(scenario["lists"][side])]
                   for side in ["A","B"]}}


class HumanEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(__file__).parent / (".human-test-"+uuid4().hex)
        self.directory.mkdir()
        self.addCleanup(shutil.rmtree,self.directory)
        self.store = EvaluationStore(synthetic_scenarios(),self.directory,seed=42,source_sha256="fixture")
        self.session = self.store.create()
        self.identity = self.session["evaluation_id"]

    def complete(self):
        for scenario in self.session["scenarios"]:
            self.store.submit(self.identity,submission(scenario))
        return self.store.export(self.identity)

    def test_reproducible_balanced_assignment(self):
        ids = [str(i) for i in range(8)]
        first = blind_assignments(ids,["baseline","genre_votes"],42)
        self.assertEqual(first,blind_assignments(ids,["baseline","genre_votes"],42))
        self.assertNotEqual(first,blind_assignments(ids,["baseline","genre_votes"],99))
        self.assertEqual(sum(value["A"]=="baseline" for value in first.values()),4)
        self.assertTrue(all(set(value.values())=={"baseline","genre_votes"} for value in first.values()))

    def test_public_payload_has_no_mapping_metrics_or_algorithm_identifiers(self):
        text = json.dumps(self.session)
        for forbidden in ["baseline","genre_votes","algorithm","assignment_seed","metrics","blind_assignment","recommendation_source"]:
            self.assertNotIn(forbidden,text)
        self.assertEqual(len(self.session["scenarios"]),8)

    def test_assignments_and_metadata_persist_across_store_restart(self):
        restarted = EvaluationStore([],self.directory,seed=99)
        self.assertEqual(restarted.get(self.identity),self.session)
        self.assertEqual(restarted.load(self.identity)["blind_assignments"],self.store.load(self.identity)["blind_assignments"])

    def test_rating_validation(self):
        for value in [0,6,True,1.5,"5"]:
            payload = submission(self.session["scenarios"][0])
            payload["ratings"]["A"][0]["rating"] = value
            with self.assertRaisesRegex(EvaluationError,"integers"):
                self.store.submit(self.identity,payload)

    def test_wrong_movie_id_side_order_or_missing_movie_is_rejected(self):
        for change in ["id","order","missing"]:
            payload = submission(self.session["scenarios"][0])
            if change=="id": payload["ratings"]["A"][0]["tmdb_id"]=9999
            elif change=="order": payload["ratings"]["A"].reverse()
            else: payload["ratings"]["B"].pop()
            with self.assertRaises(EvaluationError): self.store.submit(self.identity,payload)

    def test_skip_is_explicit_null_not_missing_rating(self):
        payload = submission(self.session["scenarios"][0],ratings=(None,5))
        result = self.store.submit(self.identity,payload)
        self.assertTrue(result["scenarios"][0]["submitted"])
        self.assertIsNone(self.store.load(self.identity)["submissions"]["0"]["ratings"]["A"][0]["rating"])
        payload = submission(self.session["scenarios"][1])
        del payload["ratings"]["B"][0]["rating"]
        with self.assertRaises(EvaluationError): self.store.submit(self.identity,payload)

    def test_duplicate_submission_and_concurrent_double_click(self):
        payload = submission(self.session["scenarios"][0])
        def attempt():
            try: self.store.submit(self.identity,payload); return 200
            except EvaluationError as exc: return exc.status
        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(sorted(pool.map(lambda _:attempt(),range(2))),[200,409])
        self.assertEqual(len(self.store.load(self.identity)["submissions"]),1)

    def test_no_export_before_completion(self):
        with self.assertRaises(EvaluationError) as error: self.store.export(self.identity)
        self.assertEqual(error.exception.status,403)
        self.assertFalse((self.directory/"completed").exists())

    def test_export_mapping_json_and_anonymous_record_fields(self):
        result = self.complete()
        self.assertTrue(result["completed"])
        self.assertEqual(len(result["records"]),8)
        self.assertEqual(json.loads((self.directory/"completed"/(self.identity+".json")).read_text(encoding="utf-8")),result)
        private = self.store.load(self.identity)
        for record in result["records"]:
            self.assertEqual(record["blind_assignment"],private["blind_assignments"][record["scenario_id"]])
            for side in ["A","B"]:
                self.assertEqual(record["lists"][side]["algorithm_id"],record["blind_assignment"][side])
        for identifier in ["email","name","ip_address"]:
            self.assertNotIn(identifier,json.dumps(result))

    def test_csv_round_trip_and_json_analysis_agree(self):
        data = self.complete()
        path = self.directory/"result.csv"
        path.write_text(export_csv(data),encoding="utf-8",newline="")
        records = load_csv(path)
        self.assertEqual(analyze(records),analyze(data["records"]))
        self.assertEqual(records[0]["notes"],data["records"][0]["notes"])

    def test_analysis_mean_median_win_tie_and_scenario_counts(self):
        data = self.complete()
        records = data["records"]
        for record in records:
            record["preference"]="tie" if record["scenario_id"]=="0" else next(side for side in ["A","B"] if record["blind_assignment"][side]=="genre_votes")
            for entry in record["lists"].values():
                for movie in entry["movies"]: movie["rating"]=5 if entry["algorithm_id"]=="genre_votes" else 1
        result = analyze(records)
        self.assertEqual(result["algorithms"]["genre_votes"]["mean_rating"],5)
        self.assertEqual(result["algorithms"]["baseline"]["median_rating"],1)
        self.assertEqual(result["algorithms"]["genre_votes"]["win_rate"],7/8)
        self.assertEqual(result["algorithms"]["baseline"]["tie_rate"],1/8)
        self.assertEqual(result["by_scenario"]["0"]["baseline"]["tie_rate"],1)
        self.assertEqual(result["proxy_preference_agreement"]["genre"]["agrees"],7)
        self.assertEqual(result["proxy_preference_agreement"]["diversity"]["disagrees"],7)

    def test_missing_ratings_are_not_imputed(self):
        data = self.complete()
        for record in data["records"]:
            for entry in record["lists"].values():
                for movie in entry["movies"]: movie.update(rating=None,skipped=True)
        result = analyze(data["records"])
        self.assertIsNone(result["algorithms"]["baseline"]["mean_rating"])
        self.assertIsNone(result["algorithms"]["genre_votes"]["median_rating"])
        self.assertEqual(result["algorithms"]["baseline"]["skipped_movies"],16)

    def test_analysis_rejects_duplicate_inputs(self):
        records = self.complete()["records"]
        with self.assertRaisesRegex(EvaluationError,"Duplicate"):
            analyze(records+records)

    def test_validation_preference_notes_and_unknown_fields(self):
        for field,value in [("preference","other"),("notes","x"*2001),("scenario_id","missing"),("email","not-collected")]:
            payload = submission(self.session["scenarios"][0]); payload[field]=value
            with self.assertRaises(EvaluationError): self.store.submit(self.identity,payload)

    def test_invalid_path_and_session_isolation(self):
        with self.assertRaises(EvaluationError): self.store.get("../outside")
        other = self.store.create()
        self.store.submit(self.identity,submission(self.session["scenarios"][0]))
        self.assertFalse(self.store.get(other["evaluation_id"])["scenarios"][0]["submitted"])

    def test_http_blinding_cors_exports_and_duplicate_status(self):
        server = create_server(self.store,0)
        thread = threading.Thread(target=server.serve_forever,daemon=True); thread.start()
        self.addCleanup(server.server_close); self.addCleanup(server.shutdown)
        base = f"http://127.0.0.1:{server.server_port}"
        request = Request(base+"/sessions/"+self.identity,headers={"Origin":"http://localhost:3000"})
        with urlopen(request) as response:
            self.assertEqual(response.headers["Access-Control-Allow-Origin"],"http://localhost:3000")
            self.assertNotIn("genre_votes",response.read().decode())
        for path,origin,status in [(f"/sessions/{self.identity}/export","http://localhost:3000",403),
                                   (f"/sessions/{self.identity}","https://untrusted.example",403),
                                   ("/mapping","http://localhost:3000",404)]:
            with self.assertRaises(HTTPError) as error: urlopen(Request(base+path,headers={"Origin":origin}))
            self.assertEqual(error.exception.code,status)

    def test_load_measured_outputs_without_hardcoded_rankings(self):
        source,catalog,snapshot = self.directory/"source.json",self.directory/"catalog.csv",self.directory/"external.json"
        scenarios = synthetic_scenarios()
        with catalog.open("w",encoding="utf-8",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=["tmdb_id","title","release_year","genres","overview","poster_path"]); writer.writeheader()
            for scenario in scenarios:
                for movie in [*scenario["favorites"],*scenario["variants"]["baseline"]["movies"]]:
                    writer.writerow(dict(tmdb_id=movie["tmdb_id"],title=movie["title"],release_year=movie["year"],genres="Drama",overview="Metadata",poster_path="/poster.jpg"))
        atomic_json(snapshot,{"movies":[]})
        measured=[]
        for scenario in scenarios:
            entry = {"code":scenario["scenario_id"],"name":scenario["name"],"status":"available","seeds":scenario["favorites"],
                     "recommendations":scenario["variants"]["baseline"]["movies"],"metrics":scenario["variants"]["baseline"]["metrics"],
                     "experiments":{"genre_votes":{"recommendations":list(reversed(scenario["variants"]["genre_votes"]["movies"])),"metrics":scenario["variants"]["genre_votes"]["metrics"]}}}
            measured.append(entry)
        atomic_json(source,{"artifact_sha256":{"data/tmdb_movies.csv":digest(catalog)},"scenarios":measured})
        loaded=load_scenarios(source,catalog,snapshot,["baseline","genre_votes"])
        self.assertEqual([m["tmdb_id"] for m in loaded[0]["variants"]["genre_votes"]["movies"]],[2,1])
        self.assertEqual(loaded[0]["variants"]["baseline"]["movies"][0]["overview"],"Metadata")
        catalog.write_text("changed",encoding="utf-8")
        with self.assertRaisesRegex(EvaluationError,"Catalog differs"):
            load_scenarios(source,catalog,snapshot,["baseline","genre_votes"])
