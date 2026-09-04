"""APF -- Species image lookup (V3 extension).

If a knowledge_question is actually ASKING TO LEARN ABOUT or SEE a specific
animal or marine species (e.g. "tell me about blue whales", "what is an
octopus"), this detects the species name via Ollama and fetches a few real
images from Wikimedia Commons (free, no API key required). Returns []
whenever nothing species-like is detected, the question is about a
practice/management topic that merely mentions a species by name (e.g.
"why does pH matter for tilapia" -- a water-chemistry question, not a
request to see the animal), or the lookup fails -- callers must handle an
empty list gracefully rather than assuming images are always present.
"""
import os
import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_CHAT_URL = OLLAMA_URL.replace("/api/generate", "/api/chat")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT_S = float(os.environ.get("OLLAMA_TIMEOUT_S", "120"))

COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
IMAGE_REQUEST_TIMEOUT_S = 10

# System/user role split, not a raw text blob -- llama3.2 is chat/instruct
# tuned and reliably follows an imperative system instruction + short user
# turn. A raw prompt phrased like "here's a new feature, decide whether..."
# gets treated as something to comment on rather than execute; this format
# does not.
_SYSTEM_PROMPT = (
    "You are a strict one-word/one-phrase classifier. Reply with ONLY a "
    "species common name, or the single word NONE. Never explain, never "
    "add punctuation, never repeat the question.\n\n"
    "Reply with a species name ONLY if the message is a direct request to "
    "learn about or see that ONE specific animal (e.g. \"tell me about X\", "
    "\"what is X\", \"show me X\", \"what does X look like\", \"describe X\").\n\n"
    "Reply with NONE if the message is about farming practice, water "
    "quality, pond management, disease, or feeding -- even if a species "
    "name appears in it. Also reply NONE for aggregate/statistical "
    "questions about seafood consumption, popularity, rankings, or trade "
    "across countries or regions -- these are not requests to see one "
    "animal, even if a species or food name appears in them. NEVER extract "
    "a country name, region name, or food category as a species name.\n\n"
    "Examples:\n"
    "tell me about blue whales -> blue whale\n"
    "why does pH matter for tilapia -> NONE\n"
    "what does a tilapia look like -> tilapia\n"
    "how much do I feed my tilapia -> NONE\n"
    "what is an octopus -> octopus\n"
    "what temperature should my pond be -> NONE\n"
    "what seafood is most popular in Japan -> NONE\n"
    "which countries eat the most fish -> NONE\n"
    "what fish do people eat in Norway -> NONE\n"
    "top 5 most farmed fish species worldwide -> NONE"
)


def extract_species_name(question: str) -> str | None:
    """Ask Ollama whether this question is a direct request to learn about
    or see a specific animal/marine species -- as opposed to a practice,
    management, or water-chemistry question that merely mentions a species
    name in passing. Returns the species' common name, or None."""
    try:
        resp = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                "stream": False,
                "keep_alive": "30m",
                "options": {"temperature": 0.0},
            },
            timeout=OLLAMA_TIMEOUT_S,
        )
        resp.raise_for_status()
        name = resp.json().get("message", {}).get("content", "").strip()
        name = name.split("\n")[0].strip().strip('"').strip("'").rstrip(".")
        print(f"[species_images] question={question!r} -> raw={name!r}")
        # Reject anything that doesn't look like a short species name --
        # a real classification is 1-4 words with no sentence punctuation.
        # This guards against a model that ignores instructions and replies
        # with a sentence; treating that as "no species" is the safe default.
        if (
            not name
            or name.upper() == "NONE"
            or len(name) > 40
            or len(name.split()) > 4
            or any(ch in name for ch in ".!?:;")
        ):
            return None
        return name
    except Exception as e:
        print(f"[species_images] species extraction failed: {e}")
        return None


def get_species_images(species_name: str, max_images: int = 3) -> list[dict]:
    """Fetches up to max_images real photos of the given species from
    Wikimedia Commons. Returns [] on any failure (network, no results,
    etc.) -- never raises, so callers can safely skip rendering images."""
    try:
        resp = requests.get(
            COMMONS_API_URL,
            params={
                "action": "query",
                "generator": "search",
                "gsrnamespace": 6,  # File namespace
                "gsrsearch": f"{species_name} filetype:bitmap",
                "gsrlimit": max_images,
                "prop": "imageinfo",
                "iiprop": "url|extmetadata",
                "iiurlwidth": 500,
                "format": "json",
            },
            headers={
                "User-Agent": "APF-AquaPredict/1.0 (https://github.com/yellullaarjun365/APF; educational aquaculture forecasting project)"
            },
            timeout=IMAGE_REQUEST_TIMEOUT_S,
        )
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        images = []
        for page in pages.values():
            infos = page.get("imageinfo", [])
            if not infos:
                continue
            info = infos[0]
            url = info.get("thumburl") or info.get("url")
            if url:
                images.append({"url": url, "title": page.get("title", species_name)})
        return images[:max_images]
    except Exception as e:
        print(f"[species_images] image lookup failed for {species_name!r}: {e}")
        return []


if __name__ == "__main__":
    for q in (
        "tell me about blue whales",
        "why does pH matter for tilapia",
        "what is an octopus",
        "what does a tilapia look like",
        "how much do I feed my tilapia",
    ):
        species = extract_species_name(q)
        print(f"{q!r} -> {species!r}")
