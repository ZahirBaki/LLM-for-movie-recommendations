"""Small deterministic output/secret checks and movie routing hints."""
import os
import re
import unicodedata

MAX_MESSAGE_LENGTH = 10_000


def output_token_limit():
    try:
        configured = int(os.getenv("MOVIE_ASSISTANT_MAX_OUTPUT_TOKENS", "900"))
    except ValueError:
        configured = 900
    return max(256, min(configured, 2000))


MAX_OUTPUT_TOKENS = output_token_limit()
LONG_FORM_WORD_THRESHOLD = 1000
LONG_FORM_PAGE_THRESHOLD = 3
LONG_FORM_REPLY = "I can help with a concise explanation, specific questions, or an outline, but I can't generate long essays, reports, or multi-page assignments."
PRIVATE_DATA_REPLY = "I can't share private credentials, environment values, or hidden instructions. I can explain how to manage them safely."


def normalized_message(message):
    return " ".join(unicodedata.normalize("NFKC", message).casefold().split())


def scope_redirect(message):
    """Legacy name retained; no longer restricts the subject of conversation."""
    text = normalized_message(message)
    if re.search(r"\b(?:reveal|show|print|expose|give me)\b.{0,80}\b(?:system prompt|internal instructions|api keys?|environment variables?|hidden tool)\b", text):
        return PRIVATE_DATA_REPLY
    if re.search(r"\b(?:forget|ignore|override|remove|bypass)\b.{0,40}\b(?:token|output|length|word)\s+(?:cap|limit|limits)\b", text):
        return LONG_FORM_REPLY
    for match in re.finditer(r"\b(\d[\d,]*)\s*[-–—]?\s*(words?|pages?)\b", text):
        digits = match[1].replace(",", "")
        count = int(digits) if len(digits) <= 12 else 10**12
        threshold = LONG_FORM_WORD_THRESHOLD if match[2].startswith("word") else LONG_FORM_PAGE_THRESHOLD
        if count < threshold:
            continue
        before = text[max(0, match.start() - 140):match.start()]
        # Source length isn't requested output length: summaries/outlines of long
        # documents are allowed. A long outline itself is still blocked.
        if re.search(r"\b(?:outline|summary|summarize|review|analysis)\b.{0,50}\b(?:of|for|this|the)\s+(?:an?\s+)?$", before):
            continue
        if re.search(r"\b(?:write|generate|create|produce|compose|draft|prepare|make|give|provide)\b", before) or re.search(r"\b(?:explain|summarize|describe)\b.{0,100}\b(?:in|into|using)\s*$", before):
            return LONG_FORM_REPLY
    if re.search(r"\b(?:write|generate|create|produce|draft)\b.{0,80}\b(?:multi[- ]page|long[- ]form)\b", text):
        return LONG_FORM_REPLY
    return None


def is_story_order_request(message):
    text = normalized_message(message)
    story = re.search(r"\bchronolog\w*\b|\bstory\b.{0,30}\border\b|\btimeline\b", text)
    authoritative = re.search(r"\brelease\s+(?:order|dates?)\b|\btmdb\b|\brating\b|\bruntime\b|\bproviders?\b", text)
    return bool(story and not authoritative)


def is_collection_request(message):
    text = normalized_message(message)
    if is_story_order_request(text):
        return False
    return bool(re.search(r"\bcollections?\b", text))


def is_broad_franchise_request(message):
    """Recognize informational scope, without any franchise membership table."""
    text = normalized_message(message)
    if re.search(r"\bcollections?\b", text):
        return False  # Explicit database collection scope wins.
    if re.search(r"\btmdb\b|\bratings?\b|\bruntime\b|\bproviders?\b|\bwhere\b.{0,40}\bwatch\b|\bfind\b", text):
        return False  # Leave authoritative facts/search/provider requests tool-capable.
    return bool(re.search(
        r"\b(?:franchise|universe|series)\b"
        r"|\b(?:all|every|entire|whole|complete)\b.{0,100}\b(?:movies?|films?)\b"
        r"|\b(?:movies?|films?)\b.{0,100}\bincluding\b"
        r"|\bincluding\b.{0,100}\b(?:movies?|films?)\b"
        r"|\b(?:movies?|films?)\b.{0,100}\b(?:in order|release order|watch order|watching order)\b", text))


def is_recommendation_request(message):
    """Force common personal movie requests before model-generated prose.

    Unqualified 'recommend something' refers to movies in this app. Explicit
    non-movie recommendation subjects stay conversational.
    """
    text = normalized_message(message)
    if is_broad_franchise_request(text) and re.search(r"\border\b|\bsequence\b", text) and not re.search(r"\brecommend|\bsuggest|\bsimilar\b|\blike\b|\blove\b|\bwatched\b|\bmy preferences\b", text):
        return False  # Informational watch order is not personal ranking.
    movie = bool(re.search(r"\b(?:movies?|films?|watch)\b", text))
    if not movie and re.search(r"\b(?:books?|restaurants?|recipes?|laptops?|courses?|software|python|javascript|database|to eat)\b", text):
        return False
    return bool(re.search(
        r"\b(?:recommend|suggest)\b.{0,70}\b(?:movies?|films?|something|anything)\b"
        r"|\b(?:movies?|films?|something|anything)\s+(?:similar\s+to|like)\b"
        r"|\bwhat\s+(?:should|could|can)\s+i\s+watch\b|\bwhat\s+to\s+watch\b"
        r"|\b(?:seen|watched)\b.+\bwhat\s+(?:next|should i watch next)\b"
        r"|\b(?:give|find|show)\b.{0,60}\b(?:movies?|films?|something)\b.{0,60}\b(?:i would like|i'd like|based on|my preferences)\b",
        text))


def has_current_movie_preferences(message):
    """Detect named seed syntax, leaving extraction and identity to existing tools.

    Pronouns alone are contextual seeds, never new titles. This does not resolve
    or split movie names (which can themselves contain commas or 'and').
    """
    text = normalized_message(message)
    match = re.search(r"\b(?:seen|watched|similar to|like)\s+(.+)", text)
    if not match:
        return False
    seed = re.split(r"[.!?]|\bi (?:want|would like)\b|\b(?:recommend|give me|what should|what next)\b", match[1], maxsplit=1)[0]
    seed = seed.strip(" ,:;")
    return bool(seed and not re.fullmatch(
        r"(?:it|its|those|them|these|that|this)(?: movies| films| ones| one)?"
        r"|(?:the )?(?:first|second|third|fourth|fifth|last)(?: one| movie| film| two| three| \d+)?"
        r"|number \d+", seed))


def is_research_request(message):
    """Completeness/freshness signals, never personal ranking or backend facts."""
    text = normalized_message(message)
    if is_recommendation_request(text) or re.search(
        r"\bcollections?\b|\btmdb\b|\bratings?\b|\bruntime\b|\bproviders?\b|\bwhere\b.{0,40}\bwatch\b|\bfind\b", text):
        return False
    movie_scope = re.search(r"\b(?:movies?|films?|franchise|universe|series|trilogy)\b", text)
    intent = re.search(r"\b(?:all|every|complete|entire|whole|full|upcoming|confirmed|latest|currently|announced)\b|\bincluding\b", text)
    # Universe/franchise list requests implicitly ask for broader membership.
    broad_list = re.search(r"\b(?:universe|franchise)\b", text) and re.search(r"\b(?:movies?|films?|order|list)\b", text)
    return bool(movie_scope and (intent or broad_list))


def preserve_explicit_movie_years(titles, message):
    """Restore a unique explicit release year the model omitted from a title.

    Match only an extracted title verbatim in the user message. Never choose a
    catalog identity or infer a year; conflicting qualifiers remain unresolved.
    """
    text = normalized_message(message)
    result = []
    for title in titles:
        pattern = r"(?<!\w)" + re.escape(normalized_message(title)) + r"\s*(?:\((\d{4})\)|(\d{4})(?!\d))"
        years = {parenthesized or trailing for parenthesized, trailing in re.findall(pattern, text)}
        result.append(f"{title} ({next(iter(years))})" if len(years) == 1 else title)
    return result
