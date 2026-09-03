# APF — Project Status (living document)

Read `PROJECT_MANUAL.md` first for the full plan and rules. This file is the
source of truth for **where things actually stand right now**.

**Current version target:** V1 (MVP) — see manual §2
**Current milestone:** V1-M6 — End-to-end web UI integration

---

## Session Log (most recent first)

## Session 6 — 2026-09-04 (Tool: Kimi via kimi.moonshot.cn)
**Milestone:** V1-M6 – UI/API integration fixes (file corruption repair)
**Completed this session:**
- Diagnosed root cause of Streamlit crash: `app/app.py` and `PROJECT_STATUS.md` on GitHub were **file-writer scripts** (Python code that writes files) instead of the actual files. A previous session generated `with open("/mnt/agents/output/app.py", "w")` which fails on Windows.
- Rewrote `app/app.py` as real Streamlit code (same fixes as Session 5: `json=` body, `importance` field, no auth endpoints).
- Rewrote `PROJECT_STATUS.md` as actual markdown (was also a writer script).
- Cleaned `requirements.txt`: removed duplicates, removed unused auth packages (`authlib`, `httpx`, `python-jose`, `passlib`, `sqlalchemy`) — no auth in V1.
**Decisions made:**
- None new; confirmed Session 5 decisions (no auth in V1, browser-native voice input).
**Files changed:**
- `app/app.py` — replaced writer-script with real Streamlit app.
- `PROJECT_STATUS.md` — replaced writer-script with real markdown.
- `requirements.txt` — deduplicated, removed auth deps.
**Blockers / open questions for Arjun:**
- Run the full stack and verify both pages work end-to-end.
- Should V1 include TTS audio output for chat responses? (Manual says "text or speech" — currently text-only.)
**Next session should:**
- Run API (:8000) + Streamlit (:8501), test chat extraction and structured forecast.
- If working, tag V1-M6 complete and decide: TTS in V1, or move to V2 (disease detection).

## Session 5 — 2026-09-04 (Tool: Kimi via kimi.moonshot.cn)
**Milestone:** V1-M6 – UI/API integration fixes
**Completed this session:**
- Diagnosed PowerShell `curl` syntax issue (`Invoke-WebRequest` vs `curl.exe`).
- Identified three bugs in `app/app.py`:
  1. `_api_predict_text` used `params=` instead of `json=` for POST body.
  2. `format_assistant_response` expected `impact_kg` but API returns `importance`.
  3. UI referenced non-existent auth/chat endpoints (`/auth/google`, `/chat/history`, `/upload`).
- Generated fixed `app.py` that only calls existing API endpoints (`/health`, `/predict`, `/predict/extract`).
- Cleaned duplicate entries from `requirements.txt`.
**Decisions made:**
- Auth, chat history persistence, and file upload are **deferred to V3+**.
  The V1 UI is chat + structured forecast only, no user accounts.
- Voice input uses browser-native `webkitSpeechRecognition` (English-only for V1).
**Files changed:**
- `app/app.py` — fixed API integration, removed auth dependencies, fixed feature-importance field name.
- `requirements.txt` — deduplicated.
**Blockers / open questions for Arjun:**
- Need to test end-to-end: run API + Streamlit simultaneously, verify chat text extraction and structured forecast pages both work.
**Next session should:**
- Run the full stack (API on :8000 + Streamlit on :8501) and verify both pages work.

## Session 4 — 2026-09-03 (Tool: Claude Sonnet 5 via claude.ai chat)
**Milestone:** V1-M3/M4/M5 – FastAPI, NLP extraction, explanation layer
**Completed this session:**
- Generated `src/api/main.py` with `/health`, `/predict`, `/predict/extract`, `/predict/structured`.
- Generated `src/nlp/extract.py` with rule-based text-to-parameter extraction.
- Generated `src/explain/explain.py` with template-based prediction explanation.
- Generated placeholder STT/TTS modules in `src/speech/`.
- Committed M3-M5 backend code.
**Files changed:**
- Added `src/api/`, `src/nlp/`, `src/explain/`, `src/speech/`.
- Updated `requirements.txt` with `fastapi`, `uvicorn[standard]`.

## Session 3 — 2026-09-03 (Tool: Claude Sonnet 5 via claude.ai chat)
**Milestone:** V1-M2 – Baseline model training (v1.1.0)
**Completed this session:**
- Fixed `total_feed_kg` outcome leakage in `src/features/build_features.py`.
- Improved synthetic generator (`scripts/generate_synthetic_data.py` v1.1.0) with wider DO/pH variance, hypoxia events, heat waves.
- Added 7 new engineered features (`do_stress_severity`, `ph_stress_total`, etc.).
- Trained LightGBM baseline: MAE=211 kg, R²=0.947, 6/15 top features are water-quality related.
- Saved artifacts: `model.pkl`, `interval.json`, `metrics.json`.
**Files changed:**
- `scripts/generate_synthetic_data.py`
- `src/features/build_features.py`
- `src/models/train_baseline.py`
- `src/models/artifacts/*`

## Session 2 — 2026-09-03 (Tool: Claude Sonnet 5 via claude.ai chat)
**Milestone:** V1-M2 – Baseline model training (attempt, failed due to leakage)
**Completed this session:**
- Attempted to train baseline model but `total_feed_kg` was not dropped from features, causing 40.5% importance on leaked feature.
- Identified `survival_rate` as additional minor leakage source.
**Decisions made:**
- Must drop all outcome-derived columns (`total_yield_kg`, `yield_kg_per_ha`, `final_survival_count`, `final_weight_g`, `fcr_effective`, `total_feed_kg`, `survival_rate`) before training.

## Session 1 — 2026-09-03 (Tool: Claude Sonnet 5 via claude.ai chat)
**Milestone:** V1-M1 – Synthetic data generation
**Completed this session:**
- Built mechanistic synthetic data generator for Nile Tilapia.
- Sourced growth kinetics, FCR, and mortality parameters from FAO species fact sheets and published literature.
- Generated v1.0.2 training set with three-tier validation (marginal distributions, relationship checks, outcome plausibility).
- All validation checks passed.
**Files changed:**
- `scripts/generate_synthetic_data.py`
- `config/tilapia_biology_params.yaml`
- `data/validation/v1_train_validation.json`
- `data/synthetic/v1_train.parquet`

## Session 0 — 2026-09-03 (Tool: Claude Sonnet 5 via claude.ai chat)
**Milestone:** V1-M0 – Planning
**Completed this session:**
- Reviewed Arjun's 27-stage roadmap and rewrote into phased V1/V2/V3+ plan.
- Researched public data landscape; confirmed no usable farm-level production dataset exists.
- Designed three-tier synthetic data validation protocol.
- Wrote `PROJECT_MANUAL.md` and this file.
**Decisions made:**
- FastAPI backend + Streamlit frontend, tree-based model baseline, English-only for V1.

---

<!-- New sessions: copy the template below, fill it in, and add it above
     this comment, keeping most-recent-first order. -->
