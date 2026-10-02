# AI Usage Log (judges may request this)
| Date/time | Tool/model | What for | Notes |
|---|---|---|---|
| 2026-10-02 22:46 | Google Antigravity (Claude Opus 4.6) | Full project build: data generation, ML models, backend API, frontend UI, docs, Docker | Agentic coding assistant used for code generation, testing, and documentation. All code was reviewed and validated by the developer. |
| 2026-10-02 22:46 | LightGBM 4.x | T-learner uplift model: one classifier per treatment arm | Trained on 100% synthetic data. No real PII. |
| 2026-10-02 22:46 | XGBoost (legacy) | Original T-learner implementation (retained for comparison) | Replaced by LightGBM as primary model. |
| 2026-10-02 22:46 | SHAP (via LightGBM built-in) | Feature contribution explanations for uplift predictions | Used to generate plain-language reasons for each recommendation. |
| 2026-10-02 22:46 | scikit-learn | Train/test split, baseline response model | Standard ML utilities. |
| 2026-10-02 22:46 | SciPy | Statistical tests for experiment comparison (t-test, confidence intervals) | Used in /experiments/compare endpoint. |
