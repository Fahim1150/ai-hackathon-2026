"""CampaignIQ API. Model scores -> business rules -> human approval gate."""
import json, pathlib, joblib, numpy as np, pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .config import OFFERS, FEATURES
from .rules import segment, apply_rules

app = FastAPI(title="CampaignIQ - Uplift & Next-Best-Offer", version="1.0.0")
MODELS = pathlib.Path("models")
_state = {}

def _load():
    if "engine" not in _state:
        if not (MODELS / "engine.joblib").exists():
            raise HTTPException(503, "Model not trained. Run: python -m backend.app.train")
        _state["engine"] = joblib.load(MODELS / "engine.joblib")
        _state["pop"] = pd.read_csv(MODELS / "population.csv")
    return _state["engine"], _state["pop"]

class UserProfile(BaseModel):
    user_id: str = "SYN_TEST"
    monthly_trans_count: int = Field(ge=0)
    avg_trans_amount: float = Field(ge=0)
    days_since_last_trans: int = Field(ge=0)
    tenure_months: int = Field(ge=0)
    cashout_ratio: float = Field(ge=0, le=1)
    offers_last_30d: int = Field(ge=0)

class PlanRequest(BaseModel):
    budget_bdt: float = Field(gt=0)

def _recommend(eng, df: pd.DataFrame):
    s = eng.score(df).iloc[0]
    offer = int(s["best_offer"])
    allowed, blocked_by = apply_rules(int(df["offers_last_30d"].iloc[0]), s["expected_gain"], s["best_uplift"])
    return {
        "segment": segment(s["baseline_prob"], s["best_uplift"]),
        "baseline_conversion_prob": round(float(s["baseline_prob"]), 3),
        "best_offer_if_sent": OFFERS[offer]["name"],
        "uplift": round(float(s["best_uplift"]), 3),
        "expected_profit_gain_bdt": round(float(s["expected_gain"]), 2),
        "decision": OFFERS[offer]["name"] if allowed else "Do not send an offer",
        "rules_blocking": blocked_by,
        "reasons": eng.explain(df, offer),
        "reason_note": "Approximate drivers (SHAP difference between offer and control models); model output, not a rule.",
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    p = MODELS / "metrics.json"
    if not p.exists():
        raise HTTPException(503, "metrics.json missing. Train first.")
    return json.loads(p.read_text())

@app.post("/recommend")
def recommend(profile: UserProfile):
    eng, _ = _load()
    df = pd.DataFrame([profile.model_dump()])
    out = _recommend(eng, df)
    out["user_id"] = profile.user_id
    out["human_review_required"] = False   # single-user suggestion; launching a campaign needs /plan approval
    return out

@app.post("/plan")
def plan(req: PlanRequest):
    """Pick users for a campaign under an expected-spend budget. Always requires human approval."""
    eng, pop = _load()
    s = eng.score(pop)
    keep = [apply_rules(int(o), g, u)[0] for o, g, u in zip(pop["offers_last_30d"], s["expected_gain"], s["best_uplift"])]
    s = s[keep].copy()
    s["roi"] = s["expected_gain"] / s["expected_cost"].clip(lower=1e-6)
    s = s.sort_values("roi", ascending=False)
    chosen = s[s["expected_cost"].cumsum() <= req.budget_bdt]
    by_offer = chosen["best_offer"].map(lambda k: OFFERS[int(k)]["name"]).value_counts().to_dict()
    return {
        "eligible_users": int(len(s)), "selected_users": int(len(chosen)),
        "population_size": int(len(pop)),
        "expected_spend_bdt": round(float(chosen["expected_cost"].sum()), 2),
        "expected_incremental_profit_bdt": round(float(chosen["expected_gain"].sum()), 2),
        "offers_assigned": by_offer,
        "status": "PENDING_HUMAN_APPROVAL",
        "note": "No campaign is launched automatically; a growth manager must approve.",
    }
