# APF — Project Status (living document)

Read `PROJECT_MANUAL.md` first for the full plan and rules. This file is the
source of truth for **where things actually stand right now**.

**Current version target:** V1 (MVP) — see manual §2
**Current milestone:** V1-M0 — planning complete, implementation not started

---

## Session Log (most recent first)

## Session 0 — 2026-09-03 (Tool: Claude Sonnet 5 via claude.ai chat)
**Milestone:** V1-M0 – Planning
**Completed this session:**
- Reviewed Arjun's pasted 27-stage roadmap draft and rewrote it into a
  phased V1/V2/V3+ plan matching the exact V1 scope he specified (text/speech
  input → forecast → LLM explanation, text/speech out, single web interface).
- Researched the actual public-data landscape for Nile tilapia
  production/water-quality/disease datasets (see PROJECT_MANUAL.md §4.1)
  to confirm the "no usable production-target dataset exists" premise
  before committing to a synthetic-data approach.
- Designed a three-tier validation protocol for the synthetic data generator
  (marginal distributions / relationship checks / outcome plausibility) so
  the synthetic approach is defensible rather than arbitrary.
- Wrote `PROJECT_MANUAL.md` (evergreen spec) and this file (living status/
  session log).
**Decisions made:**
- V1 tech stack recommendation: FastAPI backend + Streamlit frontend,
  tree-based model (XGBoost/LightGBM) baseline before any deep sequence
  model, Whisper for STT, Claude API (or a local LLM) for extraction/
  explanation. Arjun has not yet confirmed or overridden this.
- V1 defaults to English-only text/speech input; Telugu/code-mixed speech
  support deferred pending Arjun's decision (flagged in manual §3).
**Files changed:**
- Added `PROJECT_MANUAL.md`, `PROJECT_STATUS.md` (not yet pushed to the
  repo — delivered to Arjun to add himself; repo's current contents were
  not verifiable from this session, see note below).
**Blockers / open questions for Arjun:**
- Repo contents at github.com/yellullaarjun365/APF could not be checked
  from this session (not reachable/indexed via available tools) — the next
  session should `git pull` / inspect the actual repo state rather than
  assume it's empty.
- Confirm the V1 tech stack choices above, or redirect them.
- Decide on Telugu/code-mixed speech support timing.
**Next session should:**
- Verify actual repo state first (§0 of the manual).
- Start V1-M1: pin down the Nile tilapia growth/FCR/mortality parameters
  from cited literature (manual §4.2) before writing the synthetic
  generator — don't hard-code placeholder numbers.
- Stand up the repo skeleton from manual §5 and commit it.

---

<!-- New sessions: copy the template below, fill it in, and add it above
     this comment, keeping most-recent-first order. -->

<!--
## Session N — YYYY-MM-DD (Tool: ...)
**Milestone:**
**Completed this session:**
-
**Decisions made:**
-
**Files changed:**
-
**Blockers / open questions for Arjun:**
-
**Next session should:**
-
-->
