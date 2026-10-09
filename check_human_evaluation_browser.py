"""Opt-in local browser checks; use isolated --store test storage on port 8013.

Requires next dev at localhost:3000 and human_evaluation.py serve at port 8013.
Ratings are synthetic test fixtures, never genuine human evaluation results.
"""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright, expect


def check():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome",headless=True)
        context = browser.new_context(viewport={"width":1360,"height":1000},accept_downloads=True)
        page = context.new_page()
        errors, responses, submissions = [], [], []
        page.on("pageerror",lambda error:errors.append(str(error)))
        page.route("http://127.0.0.1:8000/**",lambda route:route.fulfill(json=[]))
        page.route("https://image.tmdb.org/**",lambda route:route.fulfill(content_type="image/svg+xml",
            body='<svg xmlns="http://www.w3.org/2000/svg" width="72" height="108"><rect width="72" height="108" fill="#333"/></svg>'))
        def collect(response):
            if response.url.startswith("http://127.0.0.1:8013/sessions") and "/export" not in response.url and response.status<300:
                responses.append(response.json())
        page.on("response",collect)
        page.on("request",lambda request:submissions.append(request.post_data_json) if request.url.endswith("/submit") else None)
        page.goto("http://localhost:3000/evaluation",wait_until="networkidle")
        expect(page.get_by_role("heading",name="Compare recommendation lists")).to_be_visible()
        assert page.get_by_role("navigation",name="Main navigation").get_by_role("link",name="Evaluation",exact=False).count()==0
        page.get_by_role("button",name="Start anonymous review").click()
        expect(page.get_by_text("0 of 8 scenarios saved",exact=True)).to_be_visible()
        a,b = page.get_by_role("heading",name="List A",exact=True),page.get_by_role("heading",name="List B",exact=True)
        assert a.bounding_box()["x"]<b.bounding_box()["x"]
        assert abs(a.bounding_box()["y"]-b.bounding_box()["y"])<10
        for word in ["baseline","genre_votes","collaborative","algorithm"]:
            assert word not in page.locator("main:visible").inner_text().lower()
        session = responses[-1]
        identity = session["evaluation_id"]
        assert len(session["scenarios"])==8
        for scenario in session["scenarios"]:
            assert set(scenario["lists"])=={"A","B"}
        incomplete = context.request.get(f"http://127.0.0.1:8013/sessions/{identity}/export")
        assert incomplete.status==403
        assert page.get_by_role("button",name="Export JSON").count()==0
        page.get_by_role("button",name="Save scenario and continue").click()
        expect(page.locator('main:visible [role="alert"]')).to_contain_text("Rate each movie")
        assert len(submissions)==0
        first = page.get_by_role("combobox").first
        first.select_option("5")
        page.get_by_role("textbox",name="Notes (optional)").fill("Synthetic browser fixture, not human feedback.")
        page.reload(wait_until="networkidle")
        expect(page.get_by_role("combobox").first).to_have_value("5")
        expect(page.get_by_role("textbox",name="Notes (optional)")).to_have_value("Synthetic browser fixture, not human feedback.")
        print("PASS: desktop blind layout, no navigation link, hidden mappings/exports, validation and draft/session recovery")
        for count in range(8):
            scenario = session["scenarios"][count]
            expect(page.get_by_role("heading",name=scenario["name"],exact=True)).to_be_visible()
            for side,movies in scenario["lists"].items():
                for position,movie in enumerate(movies):
                    selector = page.locator(f'li[data-list-side="{side}"][data-movie-id="{movie["tmdb_id"]}"] select')
                    selector.select_option("skip" if position==0 else "4" if side=="A" else "3")
            page.get_by_role("radio",name="Prefer List A",exact=True).check()
            page.get_by_role("button",name="Save scenario and continue").click(click_count=2)
            expect(page.get_by_text(f"{count+1} of 8 scenarios saved",exact=True)).to_be_visible()
        assert len(submissions)==8
        expect(page.get_by_role("heading",name="Review complete")).to_be_visible()
        replay = context.request.post(f"http://127.0.0.1:8013/sessions/{identity}/submit",data=submissions[0])
        assert replay.status==409
        with page.expect_download() as download_event:
            page.get_by_role("button",name="Export JSON").click()
        downloaded = download_event.value
        data = json.loads(Path(downloaded.path()).read_text(encoding="utf-8"))
        assert data["completed"] and len(data["records"])==8
        assert data["evaluation_id"]==identity
        for record in data["records"]:
            expected = next(s for s in session["scenarios"] if s["scenario_id"]==record["scenario_id"])
            for side,entry in record["lists"].items():
                assert [m["tmdb_id"] for m in entry["movies"]]==[m["tmdb_id"] for m in expected["lists"][side]]
                assert entry["movies"][0]["rating"] is None and entry["movies"][0]["skipped"]
                assert entry["algorithm_id"]==record["blind_assignment"][side]
        with page.expect_download() as csv_event:
            page.get_by_role("button",name="Export CSV").click()
        from human_evaluation import analyze, load_csv
        assert analyze(load_csv(csv_event.value.path()))==analyze(data["records"])
        for response in responses:
            assert "genre_votes" not in json.dumps(response) and "baseline" not in json.dumps(response)
        page.get_by_role("button",name="Start another anonymous review").click()
        expect(page.get_by_text("0 of 8 scenarios saved",exact=True)).to_be_visible()
        assert page.evaluate("localStorage.getItem('movie-human-review-session-v1')")!=identity
        print("PASS: eight synthetic submissions, movie-ID/side order, skips, duplicate prevention and mapped JSON/CSV exports")
        mobile = browser.new_context(viewport={"width":390,"height":844})
        phone = mobile.new_page()
        phone.on("pageerror",lambda error:errors.append(str(error)))
        phone.route("http://127.0.0.1:8000/**",lambda route:route.fulfill(json=[]))
        phone.route("https://image.tmdb.org/**",lambda route:route.fulfill(status=404))
        phone.goto("http://localhost:3000/evaluation",wait_until="networkidle")
        phone.get_by_role("button",name="Start anonymous review").click()
        expect(phone.get_by_text("0 of 8 scenarios saved",exact=True)).to_be_visible()
        a,b = phone.get_by_role("heading",name="List A",exact=True),phone.get_by_role("heading",name="List B",exact=True)
        assert b.bounding_box()["y"]>a.bounding_box()["y"]
        for width in [390,320,844]:
            phone.set_viewport_size({"width":width,"height":568 if width in [320,844] else 844})
            assert phone.evaluate("document.documentElement.scrollWidth <= innerWidth")
        phone.set_viewport_size({"width":390,"height":844})
        phone.get_by_role("combobox").first.select_option("skip")
        phone.reload(wait_until="networkidle")
        expect(phone.get_by_role("combobox").first).to_have_value("skip")
        expect(phone.get_by_role("textbox",name="Notes (optional)")).to_be_editable()
        assert len(errors)==0,errors
        print("PASS: independent mobile session, stacked lists, 320/390/844 widths, draft recovery and no runtime errors")
        browser.close()


if __name__=="__main__":
    check()
