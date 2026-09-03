# Nile Tilapia Growth / FCR / Mortality — Research Notes

Companion to `config/tilapia_biology_params.yaml`. This file holds the
narrative context, caveats, and full source list; the YAML holds only the
numbers the simulation code should actually read. Compiled 2026-09-03.

Manual reference: this fills in the §4.2 requirement ("parameterize from
FAO species fact sheets and published tilapia growth-trial papers, not
from memory... don't hard-code numeric thresholds until they're sourced").

---

## 1. Growth vs. temperature

Optimal growth window is consistently reported as roughly 25-32°C, with
several sources converging on 27-30°C as the core optimum. One controlled
trial found 28°C paired with a 25% protein diet gave the best measured
growth outcome. Growth degrades sharply outside the window: 21-22°C is
explicitly called too low for a species that thrives at 28-32°C. On the
high end, 32°C is also the temperature associated with masculinizing
sex-reversal effects in juveniles — a confound worth excluding from a
pure-growth training range, or at least flagging.

**Best lead for an actual SGR curve (not just a band):** a factorial study
tested temperature (20-34°C) crossed with dietary protein (25-50%) and
measured SGR, feed efficiency, and IGF-I directly via response-surface
methodology (ScienceDirect, *Aquaculture*, 2012). This is the strongest
candidate for extracting real regression coefficients rather than an
optimum band — full text not yet fetched.

## 2. Feed conversion ratio (FCR)

| System | FCR range | Note |
|---|---|---|
| Commercial feed, overall | 1.1–2.5 (avg ~1.8) | Pelleted 1.5–2.5, extruded 1.1–2.5 |
| Semi-intensive pond | 1.70–1.90 | Varies by feed type (sinking/floating) |
| Intensive tank | 2.23–2.86 | Worse than semi-intensive pond in this dataset |
| Cage (density 20→50 fish/m³) | improves with density | Opposite direction from the tank-density result |

**Caveat worth resolving before coding:** the tank study and the cage study
disagree on which way FCR moves with stocking density. Treat FCR-vs-density
as system-type-dependent rather than a single universal relationship until
this is reconciled — possibly by fetching both full papers and checking
whether density ranges/methods actually overlap.

General drivers, consistent across sources: higher crude protein, DO, and
pH all decrease FCR and increase thermal growth coefficient; higher
stocking weight improves both FCR and survival (Mengistu et al. 2020,
*Reviews in Aquaculture*, systematic yield-gap review).

## 3. Water quality safe ranges (general operating band)

- DO: minimum safe ≥ 5 mg/L
- pH: 6.4–9.0 (nitrifying bacteria prefer narrower 7.0–8.0)
- Temperature: 25–30°C
- Ammonia: optimum ceiling 0.05 mg/L; absolute max generally cited at 0.1 mg/L

Source: *Performance of Nile Tilapia Fingerlings I: Effect of pH* (2009).

## 4. Mortality thresholds

### Dissolved oxygen (duration-dependent, not a hard cutoff)
- Below 3 mg/L: feed intake and growth decline
- 0.8 mg/L at 26°C: critical respiration threshold (Duy et al. 2008)
- Below 2 mg/L for 2-3+ consecutive days: mortality risk rises (Chervinski 1982)
- Below 1 mg/L: short-term survivable via air-gulping at the surface

A separate factorial trial tested three DO bands directly against growth
and innate immunity: low (1.0–1.5 mg/L), medium (2.5–3.0 mg/L), normal
(6.0–6.5 mg/L) — useful for a dose-response curve rather than a single
threshold.

### pH (life-stage-split mortality curve)
| pH | Juvenile mortality | Adult mortality |
|---|---|---|
| 3 | 100% | 100% |
| 4 | 44% | 34% |
| 5 | 22% | 18% |
| 6 | 12% | 14% |

Juveniles are more sensitive at pH 4–5; adults are slightly more sensitive
at pH 6. Directly usable as a survival-penalty function keyed on pH
deviation and life stage.

### Open gap
No literature-sourced **high-temperature** mortality threshold has been
gathered yet — only low-DO and low-pH thresholds above. Heat-stress
mortality still needs a source before the stress/mortality module (§4.2,
third bullet) is complete.

---

## Source list (as retrieved via web search, 2026-09-03)

- ResearchGate: growth curves of Nile tilapia strains at different temperatures (Scielo/Acta Scientiarum, 22/28/30°C trial)
- ResearchGate: Specific growth of Nile tilapia SGRw figure (temp range 21-22°C vs 28-32°C optimum)
- PMC11571909: Temperature and feeding frequency interactions, juvenile Nile tilapia
- PMC6375642 / PLOS ONE: Temperature preference & spontaneous sex reversal in juveniles
- ScienceDirect (2022): Capacity for thermal adaptation in Nile tilapia — oxygen uptake and ventilation
- ScienceDirect (2012): Growth and IGF-I response to temperature and dietary protein
- AquaHoy / Hamed et al. 2024, BMC Vet Res: temperature x protein optimization
- ResearchGate: Average FCR of Nile tilapia fed commercial diets (El-Sayed 2013)
- ScienceDirect (2001): caged Nile tilapia biomass/aeration, cage-cum-pond system
- ScienceDirect (2018): protein:energy ratio, semi-intensive pond aquaculture
- academia.edu: Sinking feed efficiency, FCR at different stocking densities (Bangladesh)
- ResearchGate figure: FCR of tilapia at different feed types, semi-intensive
- Wiley (2026 & 2025): Musa et al., feeding strategies & pond fertilization in semi-intensive systems
- Wiley (2020): Mengistu et al., systematic review of yield-gap factors (Reviews in Aquaculture)
- ResearchGate figure: BMP water reuse/floating feed FCR comparison
- ResearchGate (2009): Performance of Nile Tilapia Fingerlings I — Effect of pH
- ScienceDirect (2023): combined temp/salinity/DO effect on fry survival during transport
- Springer (2015): water de-stratification, DO and ammonia in Thailand tilapia ponds
- ResearchGate / Journal of Applied Aquaculture (2014/2026): DO level x stocking density factorial trial
- Springer (2018): tolerance of Nile tilapia life stages to low pH and acidified waters

Note: the original FAO species fact sheet URL (aquaculture/CulturedSpecies
Nile tilapia page) referenced in the prior session is confirmed dead
(404) as of this session. The FAO Fisheries Circular "Aquaculture of
tilapias" document (production-yield/history focus) was used as a partial
substitute but does not cover growth-physiology parameters — the sources
above were used instead for that.
