"""APF -- LLM-based explanation layer (V1).

Turns the forecasting model's structured output into a farmer-facing
natural-language explanation. Per PROJECT_MANUAL.md §3, this LLM never
produces or adjusts the forecast number itself -- it only translates the
already-computed point estimate, range, and top factors into plain
language. If any future change lets the LLM guess at the number instead
of using the trained model's output, treat that as a regression.

Uses a local Ollama model: free, no API key, no account, runs on-device.
Falls back to a deterministic template if Ollama is unreachable or times
out, so a farmer-facing request never hard-fails because a local LLM
daemon isn't running.

Setup (one-time):
    1. Install Ollama: https://ollama.com/download
    2. Pull a model:  ollama pull llama3.2
    3. Ollama runs its own local server automatically on
       http://localhost:11434 once installed.

Config (override via environment variables if needed):
    OLLAMA_URL        default http://localhost:11434/api/generate
    OLLAMA_MODEL       default llama3.2
    OLLAMA_TIMEOUT_S   default 20
"""
import os
import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT_S = float(os.environ.get("OLLAMA_TIMEOUT_S", "120"))


def _template_explanation(params: dict, result: dict) -> str:
    """Deterministic fallback -- same shape as the old hardcoded
    placeholder in main.py. Used whenever Ollama isn't reachable, so the
    farmer still gets a usable answer instead of an error."""
    pe = result["point_estimate_kg"]
    lb = result["lower_bound_kg"]
    ub = result["upper_bound_kg"]
    area = params.get("pond_area_ha", 0.5)
    density = params.get("stocking_count", 3000) / area if area > 0 else 0

    explanation = (
        f"Based on your pond parameters, I estimate a harvest of **{pe:.0f} kg** "
        f"({lb:.0f}--{ub:.0f} kg at 90% confidence). "
        f"With {params.get('stocking_count', 0):,} fish in {area:.1f} ha "
        f"(density ~{density:,.0f} fish/ha) over {params.get('culture_days', 0)} days, "
        f"this yield is consistent with {params.get('intensity', 'semi-intensive')} "
        f"Nile tilapia culture under {params.get('mean_temperature_c', 28)}C conditions."
    )
    do = params.get("mean_do_mg_l", 7.5)
    temp = params.get("mean_temperature_c", 28)
    if do < 4:
        explanation += " Note: Your DO levels are low -- consider aeration to avoid mortality."
    elif temp > 32:
        explanation += " Note: High temperatures increase stress risk -- monitor DO closely."
    return explanation


def _build_prompt(params: dict, result: dict) -> str:
    top_factors = result.get("top_factors", [])[:4]
    factors_str = ", ".join(
        f['feature'].replace('_', ' ') for f in top_factors
    ) or "not available"

    return f"""You are explaining a Nile tilapia production forecast to a farmer in plain, friendly language.

Use ONLY the numbers given below -- never invent, round differently, or change any figure.

Pond parameters:
- Pond area: {params.get('pond_area_ha')} ha
- Stocking count: {params.get('stocking_count')} fish
- Culture duration: {params.get('culture_days')} days
- Mean temperature: {params.get('mean_temperature_c')} C
- Mean dissolved oxygen: {params.get('mean_do_mg_l')} mg/L
- Mean pH: {params.get('mean_ph')}
- Season: {params.get('season')}
- Intensity: {params.get('intensity')}

Model prediction (already computed -- do not recalculate or alter):
- Point estimate: {result.get('point_estimate_kg')} kg
- Range (90% confidence): {result.get('lower_bound_kg')}-{result.get('upper_bound_kg')} kg
- Top factors driving this prediction: {factors_str}

Write a 3-5 sentence explanation for the farmer. Mention the estimate, the range, and what's driving it, in simple terms. If dissolved oxygen is below 4 mg/L or temperature is above 32C, add a short practical caution. Do not use markdown formatting -- plain sentences only."""


def generate_explanation(params: dict, result: dict) -> str:
    """Main entry point. Called by src/api/main.py in place of the old
    hardcoded template. Tries the local Ollama model first; falls back to
    the deterministic template on any failure (timeout, Ollama not
    running, malformed response) so this never raises."""
    prompt = _build_prompt(params, result)
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "keep_alive": "30m"},
            timeout=OLLAMA_TIMEOUT_S,
        )
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
        if text:
            return text
    except Exception as e:
        print(f"[llm_explain] Ollama call failed, using template fallback: {e}")
    return _template_explanation(params, result)


if __name__ == "__main__":
    # Quick manual test: python -m src.explain.llm_explain
    test_params = {
        "pond_area_ha": 0.5, "stocking_count": 3000, "culture_days": 120,
        "mean_temperature_c": 28.0, "mean_do_mg_l": 7.5, "mean_ph": 7.5,
        "season": "summer", "intensity": "semi-intensive",
    }
    test_result = {
        "point_estimate_kg": 848.6, "lower_bound_kg": 163.0, "upper_bound_kg": 1534.0,
        "top_factors": [{"feature": "num__stocking_count", "importance": 0.31}],
    }
    print(generate_explanation(test_params, test_result))
