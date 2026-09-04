"""APF -- Species image lookup (V3 extension).

If a knowledge_question looks like it's asking about a specific animal or
marine species (e.g. "blue whale", "great white shark", "octopus"), this
detects the species name via Ollama and fetches a few real images from
Wikimedia Commons (free, no API key required). Returns [] whenever nothing
species-like is detected or the lookup fails -- callers must handle an
empty list gracefully rather than assuming images are always present.
"""
import os
import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT_S = float(os.environ.get("OLLAMA_TIMEOUT_S", "120"))

COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
IMAGE_REQUEST_TIMEOUT_S = 10


def extract_species_name(question: str) -> str | None:
    """Ask Ollama whether this question is about a specific animal/marine
    species. Returns the species' common name, or None if it isn't asking
    about a specific species (e.g. a water-quality or pond-management
    question shouldn't trigger image lookup)."""
    prompt = (
        "Does this question ask about a specific animal or marine species "
        "(e.g. \"blue whale\", \"great white shark\", \"octopus\", \"tilapia\")? "
        "If yes, reply with ONLY the common species name, nothing else. "
        "If no (e.g. it's about water quality, pond management, farming "
        "technique, or isn't about a specific species), reply with exactly: NONE\n\n"
        f"Question: {question}"
    )
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "keep_alive": "30m"},
            timeout=OLLAMA_TIMEOUT_S,
        )
        resp.raise_for_status()
        name = resp.json().get("response", "").strip()
        if not name or name.upper() == "NONE" or len(name) > 60:
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
    species = extract_species_name("tell me about blue whales")
    print("Detected species:", species)
    if species:
        imgs = get_species_images(species)
        for img in imgs:
            print(img["url"])
