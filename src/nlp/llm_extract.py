"""LLM-based parameter extraction with regex fallback."""
import json
import re

from llm.async_client import ollama_generate


_EXTRACTION_SYSTEM = """You are a precise parameter extractor for aquaculture data.
Extract ONLY fields mentioned. Return STRICT JSON with these exact keys. Use null for missing.
Keys: pond_area_ha, stocking_count, initial_weight_g, culture_days, mean_temperature_c, season, intensity, feed_protein_pct, mean_do_mg_l, min_do_mg_l, mean_ph, min_ph, max_temp_c, min_temp_c
Rules: Convert acres to ha (x0.4047), weeks to days (x7), months to days (x30). Infer mins from means."""


async def extract_params_llm(text: str) -> dict:
    prompt = f"Extract parameters from this farmer message:\n\n{text}\n\nJSON:"

    raw = await ollama_generate(
        prompt,
        system=_EXTRACTION_SYSTEM,
        temperature=0.0
    )

    if not raw:
        return {}

    raw = raw.strip().strip("```json").strip("```").strip()

    try:
        parsed = json.loads(raw)

        valid = {
            "pond_area_ha",
            "stocking_count",
            "initial_weight_g",
            "culture_days",
            "mean_temperature_c",
            "season",
            "intensity",
            "feed_protein_pct",
            "mean_do_mg_l",
            "min_do_mg_l",
            "mean_ph",
            "min_ph",
            "max_temp_c",
            "min_temp_c"
        }

        return {
            k: v
            for k, v in parsed.items()
            if k in valid and v is not None
        }

    except Exception:
        return {}


def extract_params_regex(text: str) -> dict:
    text_lower = text.lower()
    extracted = {}

    m = re.search(
        r'(\d+(?:\.\d+)?)\s*(?:ha|hectare|hectares|acre|acres)',
        text_lower
    )

    if m:
        val = float(m.group(1))

        if 'acre' in text_lower:
            val *= 0.4047

        extracted["pond_area_ha"] = round(val, 2)

    m = re.search(
        r'(\d+(?:,\d+)*)\s*(?:tilapia\s+)?'
        r'(?:fish|fingerlings|stocked|stocking)',
        text_lower
    )

    if not m:
        m = re.search(
            r'(\d+(?:,\d+)*)\s*tilapia',
            text_lower
        )

    if m:
        extracted["stocking_count"] = int(
            m.group(1).replace(',', '')
        )

    m = re.search(
        r'(\d+)\s*(?:day|days|week|weeks|month|months)',
        text_lower
    )

    if m:
        val = int(m.group(1))

        if 'week' in text_lower:
            val *= 7
        elif 'month' in text_lower:
            val *= 30

        extracted["culture_days"] = val

    m = re.search(
        r'(\d+(?:\.\d+)?)\s*(?:°c|degrees?\s*c)\b',
        text_lower
    )

    if not m:
        m = re.search(
            r'(\d+(?:\.\d+)?)\s*c(?![a-z])',
            text_lower
        )

    if m:
        extracted["mean_temperature_c"] = float(
            m.group(1)
        )

    m = re.search(
        r'(\d+(?:\.\d+)?)\s*'
        r'(?:mg/l|mg\s*/\s*l|do|dissolved\s*oxygen)',
        text_lower
    )

    if m:
        extracted["mean_do_mg_l"] = float(
            m.group(1)
        )

    m = re.search(
        r'ph\s*(?:is|of|=)?\s*(\d+(?:\.\d+)?)',
        text_lower
    )

    if m:
        extracted["mean_ph"] = float(
            m.group(1)
        )

    m = re.search(
        r'(\d+(?:\.\d+)?)\s*(?:g|grams?)',
        text_lower
    )

    if m:
        val = float(m.group(1))

        if val < 5:
            val *= 1000

        extracted["initial_weight_g"] = val

    for s in ["summer", "winter", "monsoon"]:
        if s in text_lower:
            extracted["season"] = s
            break

    for i in ["extensive", "semi-intensive", "intensive"]:
        if i in text_lower:
            extracted["intensity"] = i
            break

    if (
        "mean_do_mg_l" in extracted
        and "min_do_mg_l" not in extracted
    ):
        extracted["min_do_mg_l"] = (
            extracted["mean_do_mg_l"] - 1.5
        )

    if (
        "mean_ph" in extracted
        and "min_ph" not in extracted
    ):
        extracted["min_ph"] = (
            extracted["mean_ph"] - 0.3
        )

    if "mean_temperature_c" in extracted:

        if "max_temp_c" not in extracted:
            extracted["max_temp_c"] = (
                extracted["mean_temperature_c"] + 3
            )

        if "min_temp_c" not in extracted:
            extracted["min_temp_c"] = (
                extracted["mean_temperature_c"] - 3
            )

    return extracted


async def extract_params(text: str) -> dict:
    result = await extract_params_llm(text)

    return (
        result
        if result
        else extract_params_regex(text)
    )
