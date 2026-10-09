"""Development selector test with offline API fixtures; run with flagged next dev."""
from playwright.sync_api import sync_playwright, expect
import json

def check():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
        for width in (1360,390):
            page = browser.new_page(viewport={"width":width,"height":900})
            requests = []
            def reply(route):
                requests.append(route.request.post_data_json)
                route.fulfill(content_type="application/json",body=json.dumps({"status":"success","assistant_message":"Saved conversation", "recommendations":[],"clarifications":[],"mock_mode":True}))
            page.route("**/movie-chat", reply)
            page.goto("http://localhost:3000",wait_until="networkidle")
            selector = page.get_by_role("combobox",name="Ranking mode")
            expect(selector).to_have_value("baseline")
            composer = page.get_by_placeholder("Ask for a movie...")
            def send(text):
                composer.fill(text)
                page.get_by_role("button",name="Send",exact=True).click()
                expect(composer).to_be_enabled()
                expect(page.get_by_role("button",name="Send",exact=True)).to_be_disabled()
            send("Interstellar")
            expect(page.locator(".chatMessages").get_by_text("Saved conversation",exact=True)).to_have_count(1)
            selector.select_option("genre_votes")
            expect(page.locator(".chatMessages").get_by_text("Saved conversation",exact=True)).to_have_count(1)
            send("Give me five more")
            expect(page.locator(".chatMessages").get_by_text("Saved conversation",exact=True)).to_have_count(2)
            selector.select_option("baseline")
            send("Something from the 1990s")
            expect(page.locator(".chatMessages").get_by_text("Saved conversation",exact=True)).to_have_count(3)
            assert [request["ranking_mode"] for request in requests] == ["baseline","genre_votes","baseline"]
            assert len({request["session_id"] for request in requests}) == 1
            page.close()
        browser.close()
    print("Ranking selector desktop/mobile checks passed")

if __name__ == "__main__":
    check()
