# upay ActivateAI - Status

| Component | Files | Status |
|---|---|---|
| Synthetic data generator | `backend/data_generator.py` | Done. 10,000 users, seed 42, 8,000 train / 2,000 test |
| ML engine (T-Learner, SHAP) | `backend/ml_engine.py` | Done. Writes predictions, fairness audit, overview stats to `models/` |
| Budget optimizer | `backend/optimizer.py` | Done. Deterministic, budget-capped |
| Gemini service + fallback | `backend/genai_service.py` | Done. Offline template fallback |
| FastAPI backend | `backend/main.py`, `api/index.py` | Done. Includes human approval endpoint |
| React dashboard | `frontend/src/` | Done. 4 tabs |
| Tests | `backend/test_app.py` | 25 passing (run `python backend/data_generator.py` first) |
| Deployment | `Dockerfile`, `docker-compose.yml`, `vercel.json` | Done |

Known limits: all results are on synthetic data with planted effects. The lifecycle-stage fairness ratio (0.04) is flagged by design and reviewed by a human.
