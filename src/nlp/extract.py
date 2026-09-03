"""APF V1 -- NLP extraction module (M3).
Converts free-text farmer descriptions into structured JSON suitable
for the forecasting model. Uses a lightweight rule-based parser with
regex fallback. Designed to be swapped for an LLM (Claude / local model)
without changing the interface.

Interface contract (never change without updating the API):
  extract_farmer_input(text: str) -> dict with keys:
    - "structured":  dict of extracted field → value
    - "missing_fields": list of required fields still missing
    - "confidence": "high" | "medium" | "low"
"""
import re
from typing import Any

# ------------------------------------------------------------------
# Required fields for a complete prediction
# ------------------------------------------------------------------
REQUIRED_FIELDS = [
    "pond_area_ha", "stocking_count", "initial_weight_g", "culture_days",
    "mean_temperature_c", "season", "intensity", "feed_protein_pct",
    "mean_do_mg_l", "min_do_mg_l", "mean_ph", "min_ph",
    "max_temp_c", "min_temp_c",
]

# ------------------------------------------------------------------
# Regex patterns for each field
# ------------------------------------------------------------------
_PATTERNS = {
    "pond_area_ha": [
        r"(\d+(?:\.\d+)?)\s*(?:ha|hectare|hectares|acre|acres)",
        r"pond\s*(?:area|size)?\s*(?:of|is)?\s*(\d+(?:\.\d+)?)\s*(?:ha|hectare|hectares|acre|acres)",
    ],
    "stocking_count": [
        r"(\d{3,6})\s*(?:fish|fingerlings|tilapia|stocked|seeds)",
        r"stocked\s*(\d{3,6})",
        r"stocking\s*(?:count|density)?\s*(?:of|is)?\s*(\d{3,6})",
    ],
    "initial_weight_g": [
        r"initial\s*weight\s*(?:of|is)?\s*(\d+(?:\.\d+)?)\s*(?:g|grams|gram)",
        r"(\d+(?:\.\d+)?)\s*(?:g|grams|gram)\s*(?:fingerlings|seeds)",
    ],
    "culture_days": [
        r"(\d{2,3})\s*(?:days|day)",
        r"culture\s*(?:period|duration|days)?\s*(?:of|is)?\s*(\d{2,3})",
        r"grow\s*(?:them|the fish)?\s*for\s*(\d{2,3})\s*(?:days|day)",
    ],
    "mean_temperature_c": [
        r"(?:temperature|temp)\s*(?:is|of|around|about)?\s*(\d{2}(?:\.\d+)?)\s*(?:°?C|degrees?)",
        r"(\d{2}(?:\.\d+)?)\s*(?:°?C|degrees?)\s*(?:temperature|temp)",
    ],
    "max_temp_c": [
        r"max(?:imum)?\s*(?:temperature|temp)\s*(?:is|of|around)?\s*(\d{2}(?:\.\d+)?)\s*(?:°?C|degrees?)",
        r"(?:temperature|temp)\s*(?:goes|reaches|peaks)\s*(?:up\s*to|at)\s*(\d{2}(?:\.\d+)?)",
    ],
    "min_temp_c": [
        r"min(?:imum)?\s*(?:temperature|temp)\s*(?:is|of|around)?\s*(\d{2}(?:\.\d+)?)\s*(?:°?C|degrees?)",
        r"(?:temperature|temp)\s*(?:drops\s*to|goes\s*down\s*to)\s*(\d{2}(?:\.\d+)?)",
    ],
    "mean_do_mg_l": [
        r"(?:do|dissolved\s*oxygen)\s*(?:is|of|around)?\s*(\d+(?:\.\d+)?)\s*(?:mg/?l|mg\s*per\s*litre|mg\s*per\s*liter)",
        r"oxygen\s*(?:level|concentration)?\s*(?:is|of)?\s*(\d+(?:\.\d+)?)",
    ],
    "min_do_mg_l": [
        r"(?:lowest|minimum|min)\s*(?:do|dissolved\s*oxygen)\s*(?:is|of|drops\s*to)?\s*(\d+(?:\.\d+)?)",
    ],
    "mean_ph": [
        r"(?:ph|pH)\s*(?:is|of|around)?\s*(\d+(?:\.\d+)?)",
    ],
    "min_ph": [
        r"(?:lowest|minimum|min)\s*(?:ph|pH)\s*(?:is|of|drops\s*to)?\s*(\d+(?:\.\d+)?)",
    ],
    "feed_protein_pct": [
        r"(?:protein|feed\s*protein)\s*(?:is|of|around)?\s*(\d{2})\s*%",
        r"(\d{2})\s*%\s*(?:protein|feed\s*protein)",
    ],
    "season": [
        r"\b(summer|winter|monsoon|rabi|kharif)\b",
    ],
    "intensity": [
        r"\b(extensive|semi[- ]?intensive|intensive)\b",
    ],
}

# ------------------------------------------------------------------
# Post-processors
# ------------------------------------------------------------------
_CONVERTERS = {
    "pond_area_ha": lambda v: float(v) * 0.4047 if "acre" in str(v).lower() else float(v),
    "stocking_count": int,
    "initial_weight_g": float,
    "culture_days": int,
    "mean_temperature_c": float,
    "max_temp_c": float,
    "min_temp_c": float,
    "mean_do_mg_l": float,
    "min_do_mg_l": float,
    "mean_ph": float,
    "min_ph": float,
    "feed_protein_pct": int,
    "season": lambda v: v.lower().replace("rabi", "winter").replace("kharif", "summer").replace("monsoon", "monsoon"),
    "intensity": lambda v: v.lower().replace("semi intensive", "semi-intensive").replace("semiintensive", "semi-intensive"),
}

def _extract_field(text: str, field: str) -> Any | None:
    """Try every regex pattern for a field until one matches."""
    text_lower = text.lower()
    for pat in _PATTERNS.get(field, []):
        m = re.search(pat, text_lower, re.IGNORECASE)
        if m:
            raw = m.group(1)
            try:
                return _CONVERTERS[field](raw)
            except Exception:
                return None
    return None

def extract_farmer_input(text: str) -> dict:
    """Main entry point. Extracts structured data from free text."""
    structured = {}
    for field in REQUIRED_FIELDS:
        val = _extract_field(text, field)
        if val is not None:
            structured[field] = val

    missing = [f for f in REQUIRED_FIELDS if f not in structured]

    # Infer min_temp_c / max_temp_c from mean if only mean is given
    if "mean_temperature_c" in structured and "max_temp_c" not in structured:
        structured["max_temp_c"] = round(structured["mean_temperature_c"] + 3.0, 1)
    if "mean_temperature_c" in structured and "min_temp_c" not in structured:
        structured["min_temp_c"] = round(structured["mean_temperature_c"] - 3.0, 1)

    # Infer min_do from mean if only mean is given
    if "mean_do_mg_l" in structured and "min_do_mg_l" not in structured:
        structured["min_do_mg_l"] = round(structured["mean_do_mg_l"] - 1.5, 2)

    # Infer min_ph from mean if only mean is given
    if "mean_ph" in structured and "min_ph" not in structured:
        structured["min_ph"] = round(structured["mean_ph"] - 0.3, 2)

    # Recompute missing after inferences
    missing = [f for f in REQUIRED_FIELDS if f not in structured]

    confidence = "high" if len(missing) == 0 else ("medium" if len(missing) <= 3 else "low")

    return {
        "structured": structured,
        "missing_fields": missing,
        "confidence": confidence,
    }
