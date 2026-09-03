"""APF V1 -- Explanation generation layer (M5).

Converts structured prediction results into natural language explanations.
V1 uses template-based generation. V2+ should integrate an LLM.

Usage:
    from src.explain.explain import generate_explanation
    explanation = generate_explanation(params, prediction_result)
"""
from typing import Dict, List, Any


def generate_explanation(params: dict, result: dict) -> str:
    """Generate a natural language explanation of the prediction.

    Args:
        params: The pond parameters used for prediction
        result: The prediction result dict with point_estimate_kg, etc.

    Returns:
        A human-readable explanation string
    """
    pe = result.get("point_estimate_kg", 0)
    lb = result.get("lower_bound_kg", 0)
    ub = result.get("upper_bound_kg", 0)

    area = params.get("pond_area_ha", 0.5)
    stocking = params.get("stocking_count", 0)
    density = stocking / area if area > 0 else 0
    days = params.get("culture_days", 0)
    temp = params.get("mean_temperature_c", 28)
    do = params.get("mean_do_mg_l", 7.5)
    ph = params.get("mean_ph", 7.5)
    intensity = params.get("intensity", "semi-intensive")

    # Main prediction statement
    explanation = (
        f"Based on your pond parameters, I estimate a harvest of **{pe:.0f} kg** "
        f"of Nile tilapia. The 90% confidence interval is {lb:.0f}--{ub:.0f} kg.\n\n"
    )

    # Context about the operation
    explanation += (
        f"You have {stocking:,} fish stocked in {area:.1f} hectares "
        f"(stocking density of approximately {density:,.0f} fish/ha), "
        f"with a planned culture period of {days} days. "
        f"This is classified as a {intensity} system.\n\n"
    )

    # Water quality commentary
    wq_comments = []
    if do < 3:
        wq_comments.append(
            "Your dissolved oxygen is critically low. "
            "Immediate aeration is recommended to prevent mass mortality."
        )
    elif do < 4:
        wq_comments.append(
            "Your dissolved oxygen is below the stress threshold. "
            "Consider increasing water exchange or aeration."
        )
    elif do < 5:
        wq_comments.append(
            "Your dissolved oxygen is on the lower side. "
            "Monitor closely and consider mild aeration during peak hours."
        )

    if temp > 35:
        wq_comments.append(
            "Water temperature is near the lethal maximum. "
            "Shade the pond and increase water exchange immediately."
        )
    elif temp > 32:
        wq_comments.append(
            "High temperature increases metabolic stress and disease risk. "
            "Ensure adequate DO and consider partial shading."
        )
    elif temp < 20:
        wq_comments.append(
            "Temperature is below optimal. Growth rate will be reduced. "
            "Consider delaying stocking or using greenhouses."
        )

    if ph < 6.0:
        wq_comments.append(
            "pH is acidic. This can stress fish and damage gills. "
            "Apply agricultural lime to raise pH."
        )
    elif ph > 9.0:
        wq_comments.append(
            "pH is alkaline. Ammonia toxicity increases at high pH. "
            "Check total ammonia nitrogen levels."
        )

    if wq_comments:
        explanation += "**Water Quality Notes:**\n"
        for comment in wq_comments:
            explanation += f"- {comment}\n"
        explanation += "\n"

    # Top factors from model
    top_factors = result.get("top_factors", [])
    if top_factors:
        explanation += "**Key factors driving this forecast:**\n"
        for i, factor in enumerate(top_factors[:4], 1):
            feat_name = factor["feature"].replace("num__", "").replace("cat__", "").replace("_", " ").title()
            explanation += f"{i}. {feat_name}\n"
        explanation += "\n"

    # General recommendation
    explanation += (
        "**Recommendation:** Continue monitoring water quality daily. "
        "If DO drops below 3 mg/L or temperature exceeds 35C, "
        "take immediate corrective action. Regular feeding based on "
        "the estimated biomass will help achieve the forecasted yield."
    )

    return explanation


def generate_brief_explanation(params: dict, result: dict) -> str:
    """Generate a shorter explanation for chat responses."""
    pe = result.get("point_estimate_kg", 0)
    lb = result.get("lower_bound_kg", 0)
    ub = result.get("upper_bound_kg", 0)
    area = params.get("pond_area_ha", 0.5)
    stocking = params.get("stocking_count", 0)
    density = stocking / area if area > 0 else 0
    days = params.get("culture_days", 0)
    temp = params.get("mean_temperature_c", 28)

    return (
        f"I estimate a harvest of **{pe:.0f} kg** ({lb:.0f}--{ub:.0f} kg at 90% confidence). "
        f"With {stocking:,} fish in {area:.1f} ha (density ~{density:,.0f} fish/ha) "
        f"over {days} days at {temp}C, this yield is consistent with your system parameters."
    )


if __name__ == "__main__":
    test_params = {
        "pond_area_ha": 0.5,
        "stocking_count": 3000,
        "initial_weight_g": 15.0,
        "culture_days": 120,
        "mean_temperature_c": 28.0,
        "season": "summer",
        "intensity": "semi-intensive",
        "mean_do_mg_l": 7.5,
        "min_do_mg_l": 5.0,
        "mean_ph": 7.5,
        "min_ph": 6.8,
        "max_temp_c": 32.0,
        "min_temp_c": 24.0,
    }
    test_result = {
        "point_estimate_kg": 1500.0,
        "lower_bound_kg": 1200.0,
        "upper_bound_kg": 1800.0,
        "top_factors": [
            {"feature": "stocking_count", "importance": 0.35},
            {"feature": "culture_days", "importance": 0.25},
            {"feature": "mean_do_mg_l", "importance": 0.15},
        ],
    }
    print(generate_explanation(test_params, test_result))
