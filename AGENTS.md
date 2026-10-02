# AGENTS.md — Persistent Rules for upay ActivateAI

## Project Identity

- **Product:** upay ActivateAI (Dormant-to-Active Lifecycle, Uplift & Gemini Copilot Engine)
- **Track:** 04 — Growth & Campaign Intelligence (AI DEV FEST 2026, DIU CPC × upay)
- **Goal:** Increase Monthly Active Users (MAU) by identifying dormant upay users, predicting which solo-utility offer will trigger a 30-day repeat habit, and determining whether the user will activate *specifically because of the offer* (incremental uplift) — all under a fixed marketing budget.

## Environment & Stack

| Layer | Technology |
|---|---|
| OS / Shell | macOS with zsh |
| Backend runtime | Python 3 (venv/pip), source in `backend/` |
| Backend framework | FastAPI, Uvicorn, Pydantic |
| ML & Data | Pandas, NumPy, Scikit-learn, LightGBM, SHAP, SciPy |
| GenAI | `google-genai` SDK (`gemini-2.5-flash`, structured outputs) with deterministic offline fallback |
| Frontend | Vite + React + Tailwind CSS + Recharts + Lucide Icons, source in `frontend/` |
| Testing | pytest (backend), Vite dev server (frontend) |
| Data | 100% synthetic, stored in `data/`, git-ignored |
| Model artifacts | Stored in `models/`, git-ignored |

## Strict Architecture Guardrails

1. **Separate data preparation from model inference.** The data generator (`backend/data_generator.py`) must have zero imports from ML or API modules. Model training and prediction live in `backend/ml_engine.py`. API serving lives in `backend/main.py`.

2. **Keep deterministic business rules separate from ML predictions.** Budget caps, fatigue suppression thresholds, eligibility filters, and uplift quadrant classification cutoffs belong in `backend/config.py` and `backend/optimizer.py` — never inside the ML model itself.

3. **Never put financial or targeting decisions inside a free-form LLM prompt.** The Gemini API (`backend/genai_service.py`) is used *strictly* to translate structured SHAP outputs into plain-language explanations, bilingual SMS copy (Bangla + English), and experiment summaries. All targeting, scoring, and budget allocation logic must be deterministic and auditable.

4. **Use SHAP for explainability.** Every user-level recommendation must be traceable to the top 3 SHAP feature attributions that drove the uplift score. Pre-compute SHAP values at training time for fast API serving.

5. **Require human-in-the-loop campaign approval.** No campaign transitions from DRAFT to APPROVED without an explicit `POST /api/approve-campaign` call that records reviewer name, timestamp, and notes. The frontend must enforce this gate.

6. **Gemini must degrade gracefully.** If `GEMINI_API_KEY` is missing or the API call fails, every GenAI function must return a valid deterministic fallback response (template-based) with `generation_source: "Offline Deterministic Fallback"`. The system must never crash or return an error due to a missing API key.

7. **Train/test isolation is sacred.** `data/train.csv` and `data/test.csv` must have zero `customer_id` overlap. A pytest test must enforce this. The model is trained only on `train.csv`; all API-served predictions come from pre-computed `test.csv` evaluations.

## File Structure Conventions

- Backend Python modules live flat under `backend/` (no nested `backend/app/` subpackage).
- Frontend React source lives under `frontend/src/` following standard Vite conventions.
- Documentation lives under `docs/`.
- Generated data files (`*.csv`) are git-ignored via `.gitignore`.
- Generated model artifacts (`*.joblib`, `*.json`, `*.csv` in `models/`) are git-ignored.
- No plan files (`*plan*.md`, `*PLAN*.md`) are tracked in Git.
