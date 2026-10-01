"""Generic API skeleton. Keep data prep, model inference and business rules separate."""
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="AI Hackathon API")


class PredictRequest(BaseModel):
    features: dict


def apply_business_rules(features: dict) -> list[str]:
    """Deterministic rules live here, NOT inside an LLM prompt."""
    return []


def model_predict(features: dict) -> float:
    """Replace with a real model call (load a trained artifact)."""
    return 0.0


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    score = model_predict(req.features)
    rules = apply_business_rules(req.features)
    return {
        "score": score,
        "rules_triggered": rules,
        "reasons": [],  # SHAP / rule trace goes here
        "needs_human_review": score > 0.8,
    }
