"""
upay ActivateAI — FastAPI Backend.

Serves pre-computed uplift predictions, budget simulation, Gemini-powered
nudges, and human-in-the-loop campaign approval.

All targeting/scoring logic is deterministic and lives in optimizer.py
and ml_engine.py. The Gemini service only translates structured outputs
into human-readable text (AGENTS.md guardrails #2, #3, #5).
"""

from __future__ import annotations

import json
import pathlib
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.optimizer import optimize_reactivation_budget
from backend.genai_service import generate_customer_nudge, generate_experiment_insights

load_dotenv()

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="upay ActivateAI",
    description="Dormant-to-Active Lifecycle, Uplift & Gemini Copilot Engine",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Data loading (lazy singleton)
# ---------------------------------------------------------------------------
MODELS_DIR = pathlib.Path("models")
_cache: dict[str, Any] = {}


def _load_predictions() -> pd.DataFrame:
    if "predictions" not in _cache:
        p = MODELS_DIR / "test_predictions.csv"
        if not p.exists():
            raise HTTPException(503, "test_predictions.csv missing. Run: python backend/ml_engine.py")
        _cache["predictions"] = pd.read_csv(p)
    return _cache["predictions"]


def _load_json(filename: str) -> dict:
    key = f"json_{filename}"
    if key not in _cache:
        p = MODELS_DIR / filename
        if not p.exists():
            raise HTTPException(503, f"{filename} missing. Run: python backend/ml_engine.py")
        _cache[key] = json.loads(p.read_text())
    return _cache[key]


# ---------------------------------------------------------------------------
# In-memory campaign approval log (AGENTS.md guardrail #5)
# ---------------------------------------------------------------------------
_approvals: list[dict] = []


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class SimulateRequest(BaseModel):
    budget_bdt: float = Field(gt=0, description="Total reactivation budget in BDT")
    max_fatigue_cap: int = Field(default=70, ge=0, le=100, description="Max fatigue score allowed")
    min_uplift_cutoff: float = Field(default=0.02, ge=0, le=1, description="Min uplift score threshold")
    lifecycle_stage: str | None = Field(default=None, description="Optional lifecycle stage filter")


class ApprovalRequest(BaseModel):
    reviewer_name: str = Field(min_length=1, description="Name of the human reviewer")
    budget_bdt: float = Field(gt=0, description="Approved budget in BDT")
    notes: str = Field(default="", description="Optional reviewer notes")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    """Health check."""
    return {"status": "ok"}


@app.get("/api/overview")
def overview():
    """Dormancy funnel, uplift metrics, and fairness audit summary."""
    overview_stats = _load_json("overview_stats.json")
    fairness = _load_json("fairness_audit.json")

    # Merge fairness into overview (it may already be there, but ensure fresh)
    overview_stats["fairness"] = fairness
    return overview_stats


@app.post("/api/simulate-mau-growth")
def simulate_mau_growth(req: SimulateRequest):
    """Run budget optimizer and generate Gemini experiment insights."""
    predictions = _load_predictions()

    # Run deterministic optimizer
    result = optimize_reactivation_budget(
        predictions_df=predictions,
        budget_bdt=req.budget_bdt,
        max_fatigue_cap=req.max_fatigue_cap,
        min_uplift_cutoff=req.min_uplift_cutoff,
        target_stage=req.lifecycle_stage,
    )

    # Generate experiment insights (Gemini or fallback)
    insights = generate_experiment_insights(result)
    result["experiment_insights"] = insights

    return result


@app.get("/api/customer/{customer_id}")
def get_customer(customer_id: str):
    """Single customer 360: profile, uplift, SHAP, fatigue, bilingual SMS."""
    predictions = _load_predictions()

    # Find customer
    match = predictions[predictions["customer_id"] == customer_id]
    if match.empty:
        raise HTTPException(404, f"Customer {customer_id} not found in test set")

    row = match.iloc[0]

    # Parse SHAP top-3
    try:
        shap_drivers = json.loads(row["shap_top3"])
    except (json.JSONDecodeError, TypeError):
        shap_drivers = []

    # Build profile dict for GenAI
    profile = {
        "customer_id": row["customer_id"],
        "lifecycle_stage": row["lifecycle_stage"],
        "wallet_type": row["wallet_type"],
        "days_inactive": int(row["days_inactive"]),
        "historical_tx_count": int(row["historical_tx_count"]),
        "monthly_inflow_bdt": float(row["monthly_inflow_bdt"]),
        "cashout_ratio": float(row["cashout_ratio"]),
        "top_affinity_domain": row["top_affinity_domain"],
        "channel_type": row["channel_type"],
        "promos_sent_30d": int(row["promos_sent_30d"]),
        "promo_ignore_streak": int(row["promo_ignore_streak"]),
        "prob_active_treated": float(row["prob_active_treated"]),
        "prob_active_control": float(row["prob_active_control"]),
        "uplift_score": float(row["uplift_score"]),
        "uplift_quadrant": row["uplift_quadrant"],
        "offer_fatigue_score": int(row["offer_fatigue_score"]),
        "assigned_offer": row["assigned_offer"],
        "assigned_offer_cost_bdt": float(row["assigned_offer_cost_bdt"]),
    }

    # Generate bilingual nudge (Gemini or fallback)
    nudge = generate_customer_nudge(profile, shap_drivers)

    return {
        **profile,
        "shap_top3": shap_drivers,
        "nudge": nudge,
    }


@app.get("/api/customers")
def list_customers(
    page: int = 1,
    limit: int = 20,
    search: str = "",
    quadrant: str = "",
    stage: str = "",
):
    """Paginated customer list with optional filters."""
    predictions = _load_predictions()
    df = predictions.copy()

    # Filters
    if search:
        df = df[df["customer_id"].str.contains(search, case=False)]
    if quadrant:
        df = df[df["uplift_quadrant"] == quadrant]
    if stage:
        df = df[df["lifecycle_stage"] == stage]

    total = len(df)

    # Sort by uplift descending
    df = df.sort_values("uplift_score", ascending=False)

    # Paginate
    start = (page - 1) * limit
    page_df = df.iloc[start: start + limit]

    customers = []
    for _, row in page_df.iterrows():
        customers.append({
            "customer_id": row["customer_id"],
            "lifecycle_stage": row["lifecycle_stage"],
            "wallet_type": row["wallet_type"],
            "uplift_score": float(row["uplift_score"]),
            "uplift_quadrant": row["uplift_quadrant"],
            "offer_fatigue_score": int(row["offer_fatigue_score"]),
            "assigned_offer": row["assigned_offer"],
            "days_inactive": int(row["days_inactive"]),
        })

    return {
        "customers": customers,
        "total": total,
        "page": page,
        "limit": limit,
        "pages": (total + limit - 1) // limit,
    }


@app.post("/api/approve-campaign")
def approve_campaign(req: ApprovalRequest):
    """Human-in-the-loop campaign approval (AGENTS.md guardrail #5)."""
    approval = {
        "approved": True,
        "reviewer_name": req.reviewer_name,
        "budget_bdt": req.budget_bdt,
        "notes": req.notes,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "campaign_id": f"CAMP_{len(_approvals) + 1:04d}",
    }
    _approvals.append(approval)
    return approval


@app.get("/api/approvals")
def list_approvals():
    """List all campaign approvals."""
    return {"approvals": _approvals}
