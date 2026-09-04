"""Async LLM explanation with caching."""
from llm.async_client import ollama_generate


def _template_explanation(params: dict, result: dict) -> str:
    pe = result["point_estimate_kg"]
    lb = result["lower_bound_kg"]
    ub = result["upper_bound_kg"]

    area = params.get("pond_area_ha", 0.5)

    density = (
        params.get("stocking_count", 3000) / area
        if area > 0
        else 0
    )

    explanation = (
        f"Based on your pond parameters, I estimate a harvest of "
        f"{pe:.0f} kg ({lb:.0f} to {ub:.0f} kg at 90 percent confidence). "
        f"With {params.get('stocking_count', 0):,} fish in {area:.1f} ha "
        f"(density about {density:,.0f} fish per ha) over "
        f"{params.get('culture_days', 0)} days, this yield is consistent "
        f"with {params.get('intensity', 'semi-intensive')} Nile tilapia "
        f"culture under {params.get('mean_temperature_c', 28)} degree conditions."
    )

    do = params.get("mean_do_mg_l", 7.5)
    temp = params.get("mean_temperature_c", 28)

    if do < 4:
        explanation += (
            " Note: Your DO levels are low -- "
            "consider running an aerator at dawn."
        )

    elif temp > 32:
        explanation += (
            " Note: High temperatures increase stress risk -- "
            "monitor DO closely."
        )

    return explanation


def _build_prompt(params: dict, result: dict) -> str:
    top_factors = result.get("top_factors", [])[:4]

    factors_str = ", ".join(
        f["feature"].replace("_", " ")
        for f in top_factors
    ) or "not available"

    do_tip = ""

    if params.get("mean_do_mg_l", 7.5) < 4:
        do_tip = (
            "Add a practical tip: run aerator at dawn."
        )

    if params.get("mean_temperature_c", 28) > 32:
        do_tip = (
            "Add a practical tip: monitor DO closely, "
            "consider shading."
        )

    return f"""You are a friendly aquaculture extension officer speaking to a small-scale farmer in simple language.
Use ONLY the numbers below -- never invent or change any figure.

Pond: {params.get('pond_area_ha')} ha,
{params.get('stocking_count')} fish,
{params.get('culture_days')} days,
{params.get('mean_temperature_c')}C,
DO {params.get('mean_do_mg_l')} mg/L,
pH {params.get('mean_ph')}.

Forecast: {result.get('point_estimate_kg')} kg
(range {result.get('lower_bound_kg')}-{result.get('upper_bound_kg')} kg).

Top drivers: {factors_str}

Write 3-4 sentences. Mention estimate, range, and main factor.
{do_tip}

Plain text only, no markdown."""


async def generate_explanation(
    params: dict,
    result: dict
) -> str:

    try:
        text = await ollama_generate(
            _build_prompt(params, result),
            temperature=0.3
        )

        if text:
            return text

    except Exception as e:
        print(
            f"[llm_explain] Ollama failed: {e}"
        )

    return _template_explanation(
        params,
        result
    )
