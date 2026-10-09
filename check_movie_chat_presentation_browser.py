"""Offline presentation/scroll checks. Requires localhost:3000 and Chrome.

Every API response and poster is mocked; no backend or OpenAI calls are made.
Run with .venv/Scripts/python.exe check_movie_chat_presentation_browser.py.
"""
from copy import deepcopy


def check():
    from playwright.sync_api import sync_playwright, expect

    titles = ["Up", "WALL·E", "Ratatouille", "Big Hero 6", "Monsters, Inc.",
              "The Iron Giant", "The Incredibles", "Finding Nemo", "Toy Story"]
    movies = [{"tmdb_id": index + 1, "title": title, "year": 2000 + index,
        "poster_url": f"https://image.tmdb.org/t/p/w500/presentation-{index}.png",
        "rating": 8.1, "genres": "Animation", "overview": f"Description for {title}.",
        "recommendation_source": "content", "engine_mode": "CONTENT", "content_score": 0.8,
        "svd_score": None, "final_score": 0.8, "watch_providers": {"country_code": "SE",
            "stream": [{"provider_id": 1, "name": "Fixture streaming", "logo_url": None}],
            "free": [], "ads": [], "rent": [], "buy": [], "link": "https://example.org/watch"}}
        for index, title in enumerate(titles)]
    recommendations = {"status": "success", "assistant_message":
        "Invented movie (9.9) is on Unverified Netflix: https://example.org/wrong. "
        + " ".join(movie["overview"] for movie in movies),
        "recommendations": movies, "clarifications": [], "mock_mode": False,
        "sources": [{"title": "Old prose source", "url": "https://example.org/source"}],
        "citations": [{"start_index": 0, "end_index": 8, "source_index": 0}]}
    short = {"status": "success", "assistant_message": "A short conversational answer.",
             "recommendations": [], "clarifications": [], "mock_mode": False}

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context(viewport={"width": 1360, "height": 1000}, reduced_motion="reduce")
        page = context.new_page()
        errors, pending_images, pending_answers, requests = [], [], [], []
        page.on("pageerror", lambda error: errors.append(str(error)))

        def respond(route):
            data = route.request.post_data_json
            requests.append(data)
            if data["message"] in {"Delayed reply", "Delayed recommendations"}:
                pending_answers.append(route)
            elif data["message"] == "Long explanation":
                route.fulfill(json={**short, "assistant_message": "\n\n".join(
                    f"Paragraph {index}: " + "An explicitly requested explanation. " * 15 for index in range(30))})
            elif data["message"] == "Recommend five movies":
                route.fulfill(json={**deepcopy(recommendations), "recommendations": deepcopy(movies[:5])})
            elif data["message"] == "Recommend external fixture movies":
                route.fulfill(json={**short, "status": "needs_preferences",
                    "assistant_message": "I found the movie on TMDB, but the recommendation engine cannot score movies outside the local catalog. No recommendations were generated from that movie.",
                    "resolved_movies": [{"tmdb_id": 1248832, "title": "Digger", "year": 2026}]})
            elif data["message"] == "Recommend fixture movies":
                route.fulfill(json=deepcopy(recommendations))
            elif data["message"] == "Tell me more about the second movie.":
                route.fulfill(json={**short, "assistant_message": "WALL·E details from the existing movie tools."})
            else:
                route.fulfill(json=short)

        page.route("http://127.0.0.1:8000/**", respond)
        page.route("https://image.tmdb.org/**", lambda route: pending_images.append(route))
        page.goto("http://localhost:3000", wait_until="networkidle")
        text = page.get_by_placeholder("Ask for a movie...")
        area = page.locator(".chatMessages")
        last = page.locator(".chatMessage.assistant").last

        def settle():
            page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")

        def send(message):
            text.fill(message)
            text.press("Enter")
            expect(text).to_be_enabled(timeout=10000)
            settle()

        def visible_layout():
            for selector in [".siteNavigation", ".chatComposer"]:
                bounds = page.locator(selector).bounding_box()
                assert bounds and bounds["y"] >= -1 and bounds["y"] + bounds["height"] <= page.evaluate("innerHeight") + 1, (selector, bounds)
            assert page.evaluate("window.scrollY") == 0
            assert page.evaluate("document.documentElement.scrollHeight <= innerHeight + 1")
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")

        def start_visible():
            bounds = last.locator(".chatRole").bounding_box()
            container = area.bounding_box()
            assert bounds and container and container["y"] - 1 <= bounds["y"] <= container["y"] + 4, (bounds, container)
            visible_layout()

        assert area.evaluate("node => node.clientHeight") >= 600, "Desktop messages need more room"
        send("Recommend fixture movies")
        expected = "Here are some movies you might enjoy:\n\n" + "\n".join(
            f"{index + 1}. {title}" for index, title in enumerate(titles))
        expect(last.locator(".chatBubble")).to_have_text(expected)
        expect(last.locator(".movieCard h3")).to_have_text(titles)
        expect(last.locator(".chatBubble a")).to_have_count(0)
        assert "Description" not in last.locator(".chatBubble").inner_text()
        assert "Netflix" not in last.locator(".chatBubble").inner_text()
        start_visible()
        print("PASS: exact backend title/order list; no duplicated descriptions, ratings, links, invented titles or citation offsets")

        def scroll_up():
            area.hover()
            page.mouse.wheel(0, -100000)
            page.wait_for_function("document.querySelector('.chatMessages').scrollTop <= 1")
            settle()

        # Manually read earlier content, then release delayed posters.
        scroll_up()
        settle()
        before = area.evaluate("node => node.scrollTop")
        for route in pending_images:
            route.fulfill(content_type="image/svg+xml", body="<svg xmlns='http://www.w3.org/2000/svg' width='300' height='450'><rect width='300' height='450' fill='#19191e'/></svg>")
        pending_images.clear()
        page.wait_for_function("Array.from(document.querySelectorAll('.posterContainer img')).every(image => image.complete)")
        settle()
        assert abs(area.evaluate("node => node.scrollTop") - before) <= 1
        text.fill("Unsent draft")  # An unrelated rerender must also preserve reading position.
        settle()
        assert abs(area.evaluate("node => node.scrollTop") - before) <= 1
        visible_layout()
        print("PASS: delayed poster loads and input rerenders respect manual scrolling; document stays still")

        send("What is machine learning?")
        expect(last.locator(".chatBubble")).to_have_text(short["assistant_message"])
        bubble = last.locator(".chatBubble").bounding_box()
        container = area.bounding_box()
        assert bubble["y"] >= container["y"] and bubble["y"] + bubble["height"] <= container["y"] + container["height"] + 1
        visible_layout()
        send("Tell me more about the second movie.")
        expect(last.locator(".chatBubble")).to_have_text("WALL·E details from the existing movie tools.")
        print("PASS: general conversation and explicit detail text remain unchanged; short responses visible")

        send("Long explanation")
        expect(last.locator(".chatBubble")).to_contain_text("Paragraph 0")
        start_visible()
        print("PASS: long explanatory answers start at readable beginning, with navigation and input visible")

        # Scrolling upward while a response is pending must not be undone.
        text.fill("Delayed reply")
        text.press("Enter")
        expect(text).to_be_disabled()
        expect(page.get_by_role("status")).to_have_text("Finding your movies…")
        settle()
        user_bounds = page.locator(".chatMessage.user").last.bounding_box()
        container = area.bounding_box()
        assert user_bounds["y"] >= container["y"] and user_bounds["y"] + user_bounds["height"] <= container["y"] + container["height"] + 1
        scroll_up()
        assert pending_answers
        pending_answers.pop().fulfill(json=short)
        expect(text).to_be_enabled()
        settle()
        assert area.evaluate("node => node.scrollTop") == 0
        visible_layout()
        print("PASS: user message/loading visible on send; scrolling up during pending response is respected")

        # Existing card controls remain intact and all cards can be reached.
        summary = page.locator(".movieCard summary").first
        summary.click()
        expect(page.locator(".movieCard details").first).to_have_attribute("open", "")
        expect(page.locator(".movieCard details").first).to_contain_text("CONTENT")
        expect(page.locator(".movieCard .overview").first).to_have_text("Description for Up.")
        expect(page.locator(".movieCard .providerChip").first).to_contain_text("Fixture streaming")
        expect(page.locator(".movieCard .watchLink").first).to_have_attribute("href", "https://example.org/watch")
        print("PASS: existing card descriptions, providers, links and explanation controls unchanged")

        # A browser/programmatic scroll while loading is not a manual history read.
        text.fill("Delayed recommendations")
        text.press("Enter")
        expect(text).to_be_disabled()
        settle()
        area.evaluate("node => node.scrollTop = Math.max(0, node.scrollTop - 40)")
        settle()
        assert pending_answers
        pending_answers.pop().fulfill(json=deepcopy(recommendations))
        expect(text).to_be_enabled()
        settle()
        start_visible()
        print("PASS: layout-generated scroll while loading does not suppress assistant-start positioning")
        send("Recommend external fixture movies")
        expect(last.locator(".chatBubble")).to_contain_text("cannot score movies outside the local catalog")
        expect(last.locator(".movieCard")).to_have_count(0)
        print("PASS: verified external movie capability limitation displayed without fabricated cards")

        for width, height in [(1360, 1000), (1360, 768), (390, 844), (320, 568), (844, 390)]:
            page.set_viewport_size({"width": width, "height": height})
            page.unroute("https://image.tmdb.org/**")
            page.route("https://image.tmdb.org/**", lambda route: route.fulfill(content_type="image/svg+xml", body="<svg xmlns='http://www.w3.org/2000/svg' width='300' height='450'/>") )
            for phrase, count in [("Recommend five movies", 5), ("Recommend fixture movies", 9)]:
                send(phrase)
                expect(last.locator(".movieCard")).to_have_count(count)
                expect(last.locator(".movieCard h3")).to_have_text(titles[:count])
                # If a reply is shorter than the area, clamping leaves its start
                # visible without adding artificial blank space below it.
                bounds = last.locator(".chatRole").bounding_box()
                container = area.bounding_box()
                assert container["y"] - 1 <= bounds["y"] < container["y"] + container["height"]
                assert last.locator(".chatBubble").bounding_box()["y"] < container["y"] + container["height"], (width, height, phrase, container, last.locator(".chatBubble").bounding_box())
                if last.bounding_box()["height"] > container["height"]:
                    start_visible()
                visible_layout()
                area.evaluate("node => node.scrollTop = node.scrollHeight")
                settle()
                card = last.locator(".movieCard").last.bounding_box()
                container = area.bounding_box()
                assert card["y"] + card["height"] <= container["y"] + container["height"] + 1
                assert area.evaluate("node => node.scrollTop <= node.scrollHeight - node.clientHeight + 1")
                visible_layout()
            send("Long explanation")
            start_visible()
            send("Short answer")
            expect(last.locator(".chatBubble")).to_have_text(short["assistant_message"])
            bubble = last.locator(".chatBubble").bounding_box()
            container = area.bounding_box()
            assert bubble["y"] >= container["y"] and bubble["y"] + bubble["height"] <= container["y"] + container["height"] + 1
            visible_layout()
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path="frontend/.next/movie-chat-presentation-browser.png")
        assert len({request["session_id"] for request in requests}) == 1
        assert not errors, errors
        print("PASS: short/long answers, five/nine cards and consecutive messages on desktop/mobile; no empty spacer; stable session ID")
        browser.close()


if __name__ == "__main__":
    check()
