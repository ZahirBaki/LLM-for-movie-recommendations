"""Opt-in browser checks with real ranking and test-only provider fixtures.

Run --serve in one terminal, then run without arguments in another. Playwright
and an installed Chrome browser are required only for this development check.
The website's API calls are routed to port 8012, leaving existing servers alone.
"""

import sys


def serve():
    import os
    os.environ["MOVIE_CHAT_MOCK"] = "true"
    os.environ["ENVIRONMENT"] = "development"
    os.environ["APP_ENV"] = "development"
    os.environ["NODE_ENV"] = "development"
    import api
    import uvicorn
    from unittest.mock import patch

    def providers(tmdb_id, country_code="SE"):
        logo = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='32' height='32'%3E%3Crect width='32' height='32' fill='%23e63946'/%3E%3C/svg%3E"
        item = {"provider_id": 1, "name": "Test streaming service", "logo_url": logo, "display_priority": 1}
        return {"country_code": country_code, "stream": [item], "free": [], "ads": [], "rent": [item], "buy": [item], "link": "https://www.justwatch.com/"}

    with patch("api.get_watch_providers", side_effect=providers), patch("requests.sessions.Session.request", side_effect=AssertionError("Offline browser fixture server")):
        uvicorn.run(api.app, host="127.0.0.1", port=8012)


def check():
    from playwright.sync_api import sync_playwright, expect
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1360, "height": 1000})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.route("http://127.0.0.1:8000/**", lambda route: route.fulfill(response=route.fetch(url=route.request.url.replace(":8000", ":8012"))))
        page.route("https://image.tmdb.org/**", lambda route: route.fulfill(content_type="image/svg+xml", body="<svg xmlns='http://www.w3.org/2000/svg' width='300' height='450'><rect width='300' height='450' fill='#19191e'/></svg>"))
        page.goto("http://localhost:3000", wait_until="domcontentloaded")
        expect(page.locator(".movieChat")).to_be_visible(timeout=60000)
        chat = page.locator(".movieChat")
        text = page.get_by_placeholder("Ask for a movie...")
        expect(chat.get_by_role("button", name="Send", exact=True)).to_be_disabled()
        text.fill("   ")
        expect(chat.get_by_role("button", name="Send", exact=True)).to_be_disabled()

        # Existing search, preference and country controls remain usable.
        page.get_by_placeholder("Search for Interstellar, The Godfather...").fill("Interstellar")
        movie = page.locator(".searchDropdown button").filter(has_text="Interstellar").first
        expect(movie).to_be_visible()
        movie.click()
        page.get_by_placeholder("Try: Bale").fill("Bale")
        actor = page.locator(".preferenceDropdown button").filter(has_text="Christian Bale").first
        expect(actor).to_be_visible()
        actor.click()
        page.get_by_placeholder("Try: Nolan").fill("Nolan")
        director = page.locator(".preferenceDropdown button").filter(has_text="Christopher Nolan").first
        expect(director).to_be_visible()
        director.click()
        page.get_by_placeholder("Select genres...").click()
        genre = page.locator(".preferenceDropdown button").filter(has_text="Science Fiction").first
        expect(genre).to_be_visible()
        genre.click()
        country = page.get_by_placeholder("Search country...")
        country.click()
        expect(page.locator(".countryDropdown")).to_be_visible()
        page.locator(".hero h1").click()
        expect(page.locator(".countryDropdown")).to_have_count(0)
        country.fill("United States")
        page.locator(".countryDropdown button").filter(has_text="United States").click()
        expect(chat.locator(".movieChatHeading .watchRegion")).to_have_text("US")
        with page.expect_response(lambda response: response.url.endswith("/recommend"), timeout=120000) as manual:
            page.get_by_role("button", name="Find movies for me", exact=True).click()
        assert manual.value.status == 200
        expect(page.locator(".resultsSection .movieCard").first).to_be_visible()
        manual_count = page.locator(".resultsSection .movieCard").count()
        print("PASS: existing search, actor/director, genre, country/outside click and manual recommendations")

        def send(phrase, expected_status):
            text.fill(phrase)
            with page.expect_response(lambda response: response.url.endswith("/movie-chat"), timeout=120000) as response:
                text.press("Enter")
                if expected_status == "success":
                    expect(text).to_be_disabled()
            data = response.value.json()
            assert data["status"] == expected_status, data["status"]
            expect(text).to_be_enabled(timeout=120000)
            expect(chat.locator(".chatMessage.assistant").last.locator(".chatBubble")).to_have_text(data["assistant_message"])
            return data

        first = send("Something like Interstellar", "success")
        assert first["resolved_movies"][0]["tmdb_id"] == 157336
        expect(chat.locator(".movieCard").first).to_be_visible()
        expect(chat.locator(".providerLogo").first).to_be_visible()
        assert chat.locator(".providerLogo").first.evaluate("image => image.complete && image.naturalWidth > 0")
        expect(chat.locator(".purchaseOptions").first).to_contain_text("Rent")
        expect(chat.locator(".purchaseOptions").first).to_contain_text("Buy")
        expect(chat.locator(".watchLink").first).to_have_attribute("href", "https://www.justwatch.com/")
        expect(chat.locator(".justWatchCredit").first).to_contain_text("JustWatch")
        print("PASS: Interstellar cards, provider logos, rent/buy, watch link and attribution")
        animated = send("Something like Coco, Inside Out and Zootopia", "success")
        assert [(movie["tmdb_id"], movie["year"]) for movie in animated["resolved_movies"]] == [(354912, 2017), (150540, 2015), (269149, 2016)]
        print("PASS: Coco / Inside Out / Zootopia browser regression")
        filtered = send("Something like Interstellar with Ryan Gosling after 2015", "success")
        assert filtered["parsed_intent"]["required_actors"] == ["Ryan Gosling"]
        assert filtered["parsed_intent"]["min_year"] == 2016
        print("PASS: Ryan Gosling/year mock request and real recommendations")
        send("Something like Crash", "needs_clarification")
        expect(chat.get_by_role("button", name="Crash (1996)", exact=True)).to_be_visible()
        expect(chat.get_by_role("button", name="Crash (2005)", exact=True)).to_be_visible()
        with page.expect_response(lambda response: response.url.endswith("/movie-chat"), timeout=120000) as chosen:
            chat.get_by_role("button", name="Crash (1996)", exact=True).click()
        assert chosen.value.json()["resolved_movies"][0]["tmdb_id"] == 884
        expect(text).to_be_enabled(timeout=120000)
        print("PASS: Crash choices and independent selection request")
        send("Something like Nonexistent Movie xyz987654", "unresolved")
        send("Recommend a comedy", "needs_preferences")
        send("Who directed Interstellar?", "unsupported_intent")
        send("Unavailable mock phrase", "error")
        assert page.locator(".resultsSection .movieCard").count() == manual_count
        print("PASS: unresolved, needs preferences, unsupported and unavailable phrase; manual results preserved")
        # Enter/Shift+Enter and in-flight duplicate suppression.
        text.fill("Something like")
        text.press("Shift+Enter")
        text.press("End")
        page.keyboard.insert_text("Interstellar")
        assert "\n" in text.input_value()
        sent = []
        page.on("request", lambda request: sent.append(request.url) if request.url.endswith("/movie-chat") else None)
        with page.expect_response(lambda response: response.url.endswith("/movie-chat"), timeout=120000):
            text.press("Enter")
            text.press("Enter")
        expect(text).to_be_enabled(timeout=120000)
        assert len(sent) == 1
        # Malformed JSON and network failures must produce safe messages.
        page.route("**/movie-chat", lambda route: route.fulfill(json={"status": "success", "assistant_message": "internal", "recommendations": [{}], "clarifications": [], "mock_mode": False}))
        text.fill("Example")
        text.press("Enter")
        expect(chat.locator(".chatMessage.assistant").last.locator(".chatBubble")).to_contain_text("temporarily unavailable")
        expect(text).to_be_enabled()
        page.route("**/movie-chat", lambda route: route.abort())
        text.fill("Example")
        text.press("Enter")
        expect(chat.locator(".chatMessage.assistant").last.locator(".chatBubble")).to_contain_text("temporarily unavailable")
        print("PASS: Shift+Enter, duplicate suppression, malformed response and network error")
        page.set_viewport_size({"width": 390, "height": 844})
        chat.scroll_into_view_if_needed()
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        page.screenshot(path="frontend/.next/movie-chat-browser.png")
        assert not errors, errors
        page.reload(wait_until="domcontentloaded")
        expect(page.locator(".movieChat .chatMessage")).to_have_count(1)
        print("PASS: mobile width, no browser runtime errors and refresh resets chat")
        browser.close()


if __name__ == "__main__":
    serve() if "--serve" in sys.argv else check()
