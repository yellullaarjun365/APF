"""Free, no-API-key web fallback for knowledge questions the local RAG
knowledge base can't answer. Uses Wikipedia's public REST API only --
no key, no signup, no cost.

Used strictly as a fallback: src/knowledge/retrieve.py (the local
knowledge base) is always tried first in rag_answer.py; this only runs
when that returns zero chunks. Local knowledge stays authoritative.
"""
import requests

_SEARCH_URL = "https://en.wikipedia.org/w/api.php"
_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
_TIMEOUT_S = 8
_HEADERS = {"User-Agent": "AquaPredictAI/1.0 (educational aquaculture project)"}


def fetch_wikipedia_summary(query: str):
    """Search Wikipedia for `query`, return a dict with the best-matching
    page's title/extract/url, or None if nothing reasonable is found.

    Synchronous and fast (two small HTTP calls) -- callers in async code
    should dispatch this via asyncio.to_thread, the same pattern already
    used for get_species_images in species_images.py, rather than
    awaiting it directly.
    """
    try:
        search_resp = requests.get(
            _SEARCH_URL,
            params={
                "action": "query", "list": "search", "srsearch": query,
                "format": "json", "srlimit": 1,
            },
            headers=_HEADERS,
            timeout=_TIMEOUT_S,
        )
        search_resp.raise_for_status()
        results = search_resp.json().get("query", {}).get("search", [])
        if not results:
            return None
        title = results[0]["title"]

        summary_resp = requests.get(
            _SUMMARY_URL.format(title=title.replace(" ", "_")),
            headers=_HEADERS,
            timeout=_TIMEOUT_S,
        )
        summary_resp.raise_for_status()
        data = summary_resp.json()
        extract = data.get("extract", "").strip()
        if not extract:
            return None

        return {
            "title": data.get("title", title),
            "extract": extract,
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
        }
    except Exception as e:
        print(f"[web_fallback] Wikipedia lookup failed for {query!r}: {e}")
        return None
