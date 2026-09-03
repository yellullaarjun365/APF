"""APF V1 -- Explanation layer (M3 placeholder).
Generates natural-language explanations of model predictions.

Current implementation: rule-based template (no external LLM needed).
Future: swap _generate_explanation() for a call to Claude API or
a local LLM, keeping the same input/output signature.
"""
from typing import Any

def explain(prediction: dict, params: dict, lang: str = "en") -> str:
    """Generate a farmer-friendly explanation of the forecast.

    Args:
        prediction: dict from /predict endpoint (point_estimate_kg,
                    lower_bound_kg, upper_bound_kg, top_factors, ...)
        params: the structured input parameters used for the prediction
        lang: "en" for English, "te" for Telugu (placeholder)

    Returns:
        A plain-text explanation string.
    """
    if lang != "en":
        # Telugu placeholder — requires a real translation layer or LLM
        return _explain_te(prediction, params)

    point = prediction["point_estimate_kg"]
    low = prediction["lower_bound_kg"]
    high = prediction["upper_bound_kg"]
    factors = prediction["top_factors"]

    # Build factor sentences
    factor_sentences = []
    for f in factors[:3]:
        name = _prettify_feature(f["feature"])
        direction = "increases" if f["impact_kg"] > 0 else "reduces"
        factor_sentences.append(
            f"- **{name}** {direction} the forecast by about {abs(f['impact_kg']):.0f} kg."
        )

    # Water quality advisory
    wq_advice = []
    if params.get("min_do_mg_l", 10) < 3.0:
        wq_advice.append("Low dissolved oxygen is a major risk — consider aeration or reducing stocking density.")
    if params.get("max_temp_c", 25) > 32:
        wq_advice.append("High temperatures increase disease risk — monitor for streptococcosis.")
    if params.get("min_ph", 7) < 6.0:
        wq_advice.append("Acidic conditions stress the fish — check for acid runoff or algal crash.")

    advice_block = "
".join(wq_advice) if wq_advice else "Water quality parameters look acceptable for the planned cycle."

    return (
        f"## Harvest Forecast

"
        f"Expected harvest: **{point:.0f} kg** of tilapia.

"
        f"The model is 90% confident the actual harvest will fall between "
        f"**{low:.0f} kg** and **{high:.0f} kg**.

"
        f"### What Drives This Forecast

"
        f"The three biggest factors are:

"
        + "
".join(factor_sentences) +
        f"

### Water Quality Check

"
        f"{advice_block}

"
        f"*This forecast is based on a synthetic-data-trained model. "
        f"Use it for planning, not as a guarantee.*"
    )

def _prettify_feature(raw: str) -> str:
    """Convert raw feature names to human-readable labels."""
    mapping = {
        "stocking_count": "number of fish stocked",
        "initial_weight_g": "initial fish size",
        "culture_days": "culture duration",
        "mean_temperature_c": "average water temperature",
        "pond_area_ha": "pond size",
        "stocking_density_fish_ha": "stocking density",
        "mean_do_mg_l": "dissolved oxygen level",
        "min_do_mg_l": "minimum dissolved oxygen",
        "do_temp_interaction": "oxygen-temperature interaction",
        "temp_stress_degdays": "heat stress accumulation",
        "stress_days": "number of stressful days",
        "density_x_do_deficit": "density vs oxygen shortage",
        "max_temp_c": "peak temperature",
        "min_temp_c": "lowest temperature",
        "temp_range": "daily temperature swing",
        "feed_protein_pct": "feed protein content",
    }
    # Strip sklearn prefix if present
    clean = raw.replace("num__", "").replace("cat__", "")
    return mapping.get(clean, clean.replace("_", " "))

def _explain_te(prediction: dict, params: dict) -> str:
    """Telugu placeholder — returns English with a note."""
    en = explain(prediction, params, lang="en")
    return f"[Telugu translation pending]

{en}"
