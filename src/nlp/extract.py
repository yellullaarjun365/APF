"""APF V1 -- Information extraction from free-text farmer input (M4).

Converts natural language pond descriptions into structured parameters.
V1 uses rule-based extraction. V2+ should integrate an LLM.

Usage:
    from src.nlp.extract import extract_pond_params
    params = extract_pond_params("I have a 0.5 ha pond with 3000 fish...")
"""
import re
from typing import Optional

# Required fields for a complete prediction
REQUIRED_FIELDS = [
    "pond_area_ha",
    "stocking_count", 
    "culture_days",
    "mean_temperature_c",
]

# Default values for optional fields
DEFAULTS = {
    "initial_weight_g": 15.0,
    "season": "summer",
    "intensity": "semi-intensive",
    "feed_protein_pct": 30.0,
    "mean_do_mg_l": 7.5,
    "min_do_mg_l": 5.0,
    "mean_ph": 7.5,
    "min_ph": 6.8,
    "max_temp_c": 32.0,
    "min_temp_c": 24.0,
}

SEASONS = ["summer", "winter", "monsoon"]
INTENSITIES = ["extensive", "semi-intensive", "intensive"]


def extract_pond_params(text: str) -> dict:
    """Extract structured pond parameters from free text.

    Returns a dict with extracted values + missing field info.
    """
    text_lower = text.lower()
    extracted = {}

    # Pond area
    area = _extract_area(text_lower)
    if area is not None:
        extracted["pond_area_ha"] = area

    # Stocking count
    count = _extract_stocking_count(text_lower)
    if count is not None:
        extracted["stocking_count"] = count

    # Culture duration
    days = _extract_duration(text_lower)
    if days is not None:
        extracted["culture_days"] = days

    # Temperature
    temp = _extract_temperature(text_lower)
    if temp is not None:
        extracted["mean_temperature_c"] = temp

    # DO
    do = _extract_do(text_lower)
    if do is not None:
        extracted["mean_do_mg_l"] = do

    # pH
    ph = _extract_ph(text_lower)
    if ph is not None:
        extracted["mean_ph"] = ph

    # Initial weight
    weight = _extract_weight(text_lower)
    if weight is not None:
        extracted["initial_weight_g"] = weight

    # Season
    season = _extract_season(text_lower)
    if season is not None:
        extracted["season"] = season

    # Intensity
    intensity = _extract_intensity(text_lower)
    if intensity is not None:
        extracted["intensity"] = intensity

    # Derive min/max from means
    _derive_bounds(extracted)

    # Check completeness
    missing = [f for f in REQUIRED_FIELDS if f not in extracted or extracted[f] is None]

    # Fill defaults for optional fields
    for k, v in DEFAULTS.items():
        if k not in extracted or extracted[k] is None:
            extracted[k] = v

    return {
        "extracted": extracted,
        "missing_fields": missing,
        "is_complete": len(missing) == 0,
    }


def _extract_area(text: str) -> Optional[float]:
    """Extract pond area in hectares."""
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:ha|hectare|hectares|acre|acres)', text)
    if m:
        val = float(m.group(1))
        if 'acre' in text:
            val *= 0.4047
        return round(val, 2)
    return None


def _extract_stocking_count(text: str) -> Optional[int]:
    """Extract number of fish stocked."""
    patterns = [
        r'(\d{1,3}(?:,\d{3})+)\s*(?:fish|fingerlings|stocked|stocking|tilapia)',
        r'(\d+)\s*(?:fish|fingerlings|stocked|stocking|tilapia)',
        r'stocked\s*(\d{1,3}(?:,\d{3})+)',
        r'stocking\s*(\d{1,3}(?:,\d{3})+)',
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return int(m.group(1).replace(',', ''))
    return None


def _extract_duration(text: str) -> Optional[int]:
    """Extract culture duration in days."""
    m = re.search(r'(\d+)\s*days?', text)
    if m:
        return int(m.group(1))

    m = re.search(r'(\d+)\s*weeks?', text)
    if m:
        return int(m.group(1)) * 7

    m = re.search(r'(\d+)\s*months?', text)
    if m:
        return int(m.group(1)) * 30

    return None


def _extract_temperature(text: str) -> Optional[float]:
    """Extract water temperature in Celsius."""
    patterns = [
        r'(\d+(?:\.\d+)?)\s*(?:\u00b0c|c|degrees?\s*c)',
        r'temperature\s*(?:is|of|=)?\s*(\d+(?:\.\d+)?)',
        r'temp\s*(?:is|of|=)?\s*(\d+(?:\.\d+)?)',
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return float(m.group(1))
    return None


def _extract_do(text: str) -> Optional[float]:
    """Extract dissolved oxygen in mg/L."""
    patterns = [
        r'(\d+(?:\.\d+)?)\s*(?:mg/l|mg\s*/\s*l|do|dissolved\s*oxygen)',
        r'do\s*(?:is|of|=)?\s*(\d+(?:\.\d+)?)',
        r'oxygen\s*(?:is|of|=)?\s*(\d+(?:\.\d+)?)',
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            val = float(m.group(1))
            if 0.5 <= val <= 20:
                return val
    return None


def _extract_ph(text: str) -> Optional[float]:
    """Extract pH value."""
    patterns = [
        r'ph\s*(?:is|of|=)?\s*(\d+(?:\.\d+)?)',
        r'pH\s*(?:is|of|=)?\s*(\d+(?:\.\d+)?)',
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            val = float(m.group(1))
            if 4 <= val <= 11:
                return val
    return None


def _extract_weight(text: str) -> Optional[float]:
    """Extract initial fish weight in grams."""
    patterns = [
        r'(\d+(?:\.\d+)?)\s*(?:g|grams?)',
        r'weight\s*(?:is|of|=)?\s*(\d+(?:\.\d+)?)\s*(?:g|grams?)',
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            val = float(m.group(1))
            if val < 5:
                val *= 1000
            return val
    return None


def _extract_season(text: str) -> Optional[str]:
    """Extract season from text."""
    for season in SEASONS:
        if season in text:
            return season
    return None


def _extract_intensity(text: str) -> Optional[str]:
    """Extract culture intensity from text."""
    for intensity in INTENSITIES:
        if intensity in text:
            return intensity
    return None


def _derive_bounds(extracted: dict):
    """Derive min/max values from mean values."""
    if "mean_do_mg_l" in extracted and "min_do_mg_l" not in extracted:
        extracted["min_do_mg_l"] = extracted["mean_do_mg_l"] - 1.5
    if "mean_ph" in extracted and "min_ph" not in extracted:
        extracted["min_ph"] = extracted["mean_ph"] - 0.3
    if "mean_temperature_c" in extracted:
        if "max_temp_c" not in extracted:
            extracted["max_temp_c"] = extracted["mean_temperature_c"] + 3
        if "min_temp_c" not in extracted:
            extracted["min_temp_c"] = extracted["mean_temperature_c"] - 3


def generate_followup_question(missing_fields: list) -> str:
    """Generate a natural follow-up question for missing fields."""
    field_names = {
        "pond_area_ha": "pond area",
        "stocking_count": "number of fish stocked",
        "culture_days": "culture duration",
        "mean_temperature_c": "water temperature",
    }
    names = [field_names.get(f, f.replace("_", " ")) for f in missing_fields]

    if len(names) == 1:
        return f"I need your {names[0]} to make a forecast. Could you provide that?"
    elif len(names) == 2:
        return f"I need your {names[0]} and {names[1]} to make a forecast. Could you provide these?"
    else:
        return f"I need a few more details: {', '.join(names)}. Could you provide these?"


if __name__ == "__main__":
    test_inputs = [
        "I have a 0.5 ha pond with 3000 fish for 120 days at 28C",
        "My pond is 2 acres with 5000 tilapia, temp is 30 degrees",
        "Stocked 10000 fish in 1 hectare, 4 months culture",
    ]
    for text in test_inputs:
        result = extract_pond_params(text)
        print(f"Input: {text}")
        print(f"  Extracted: {result['extracted']}")
        print(f"  Missing: {result['missing_fields']}")
        print(f"  Complete: {result['is_complete']}")
        print()
