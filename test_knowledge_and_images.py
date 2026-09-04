"""
APF -- batch test for the knowledge base + species image lookup feature.

Run with the API already up (uvicorn src.api.main:app --reload --port 8000):

    python test_knowledge_and_images.py

Prints a compact table: question | intent-relevant status | #images | sources | reply preview.
Nothing here modifies your data -- read-only /chat calls.
"""
import requests
import json

API = "http://localhost:8000/chat"

TESTS = [
    ("tell me about blue whales",             "species, expect images"),
    ("what is a dolphin",                     "species, expect images"),
    ("describe an octopus",                   "species, expect images"),
    ("what does a great white shark look like","species, expect images"),
    ("show me a tilapia",                     "species, expect images"),
    ("why does pH matter for tilapia",        "practice, expect NO images"),
    ("how much do I feed my tilapia",         "practice, expect NO images"),
    ("what dissolved oxygen level is dangerous","practice, expect NO images"),
    ("how many fish should I stock per hectare","practice, expect NO images"),
    ("what seafood is most popular in Japan", "culture/seafood, expect NO images"),
    ("which countries eat the most fish",     "culture/seafood, expect NO images"),
    ("tell me about narwhals",                "possibly out-of-KB, watch reply honesty"),
    ("what's the best pizza topping",         "out-of-KB, should NOT hallucinate"),
    ("fish",                                  "ambiguous, should not crash"),
    ("hello",                                 "small talk, should hit chat intent"),
]

def run_one(text):
    try:
        resp = requests.post(
            API,
            json={"farmer_text": text, "known_fields": {}, "history": []},
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}

def main():
    print(f"{'#':<3} {'STATUS':<16} {'IMGS':<5} {'SOURCES':<45} QUESTION")
    print("-" * 110)
    results = []
    for i, (text, note) in enumerate(TESTS, 1):
        data = run_one(text)
        status = data.get("status", data.get("error", "ERROR"))
        images = len(data.get("images", []) or [])
        sources = ",".join(data.get("sources", []) or [])[:44]
        print(f"{i:<3} {status:<16} {images:<5} {sources:<45} {text}")
        results.append((text, note, data))

    print("\n" + "=" * 110)
    print("FULL REPLIES (for manual review -- check honesty on out-of-KB cases, no hallucination)")
    print("=" * 110)
    for text, note, data in results:
        print(f"\nQ: {text}   [{note}]")
        reply = data.get("reply") or data.get("error") or json.dumps(data)[:200]
        print(f"A: {reply}")
        if data.get("images"):
            print(f"   images: {[img['url'] for img in data['images']]}")

if __name__ == "__main__":
    main()
