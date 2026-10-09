"""Real engine/chat API checks with the existing offline fixture server on 8012."""
from playwright.sync_api import sync_playwright, expect

def check():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome",headless=True)
        for mode in ("baseline","genre_votes"):
            for width in (1360,390):
                page = browser.new_page(viewport={"width":width,"height":1000})
                errors = []
                page.on("pageerror",lambda error:errors.append(str(error)))
                page.route("http://127.0.0.1:8000/**",lambda route:route.fulfill(response=route.fetch(url=route.request.url.replace(":8000",":8012"))))
                page.route("https://image.tmdb.org/**",lambda route:route.fulfill(content_type="image/svg+xml",body="<svg xmlns='http://www.w3.org/2000/svg' width='300' height='450'/>") )
                page.goto("http://localhost:3000",wait_until="networkidle")
                page.get_by_role("combobox",name="Ranking mode").select_option(mode)
                composer = page.get_by_placeholder("Ask for a movie...")
                def send(text):
                    composer.fill(text)
                    with page.expect_response(lambda response:response.url.endswith("/movie-chat"),timeout=120000) as captured:
                        composer.press("Enter")
                    response = captured.value
                    payload = response.json()
                    assert payload["status"] == "success",payload
                    expect(composer).to_be_enabled(timeout=120000)
                    assert response.request.post_data_json["ranking_mode"] == mode
                    last = page.locator(".chatMessage.assistant").last
                    expect(last.locator(".movieCard")).to_have_count(len(payload["recommendations"]))
                    expect(last.locator(".movieCard h3")).to_have_text([movie["title"] for movie in payload["recommendations"]])
                    if payload["recommendations"]:
                        expect(last.locator(".chatBubble")).to_have_text("Here are some movies you might enjoy:\n\n" + "\n".join(f"{index}. {movie['title']}" for index,movie in enumerate(payload["recommendations"],1)))
                    return payload,response.request.post_data_json["session_id"]
                initial,session = send("Something like Interstellar")
                assert len(initial["recommendations"]) == 10
                more,same_session = send("I've already seen the first three. Give me five more.")
                assert session == same_session
                assert len(more["recommendations"]) == 5
                old = {movie["tmdb_id"] for movie in initial["recommendations"]}
                new = [movie["tmdb_id"] for movie in more["recommendations"]]
                assert len(set(new)) == 5 and not old & set(new) and 157336 not in new
                assert not errors,errors
                page.close()
                print(f"PASS: {mode} width {width}: ten -> word-count follow-up -> five new ordered cards")
        browser.close()

if __name__ == "__main__":
    check()
