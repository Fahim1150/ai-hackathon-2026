"""
upay ActivateAI — FastAPI Backend.

Serves pre-computed uplift predictions, budget simulation, Gemini-powered
nudges, and human-in-the-loop campaign approval.

All targeting/scoring logic is deterministic and lives in optimizer.py
and ml_engine.py. The Gemini service only translates structured outputs
into human-readable text (AGENTS.md guardrails #2, #3, #5).

Security hardening:
  - CORS restricted to explicit dev/prod origins (no wildcard with credentials)
  - API key RBAC via X-API-Key header on protected endpoints
  - Rate limiting via slowapi (60 req/min per client IP)
  - Campaign approvals persisted to SQLite (backend/database.py)
"""



import json
import os
import pathlib
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from backend.optimizer import optimize_reactivation_budget
from backend.genai_service import generate_customer_nudge, generate_experiment_insights
from backend.database import init_db, insert_approval, list_approvals, count_approvals

load_dotenv()

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

app = FastAPI(
    title="upay ActivateAI",
    description="Dormant-to-Active Lifecycle, Uplift & Gemini Copilot Engine",
    version="2.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS: explicit origins only — never use "*" with allow_credentials=True
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# RBAC — API Key authentication
# ---------------------------------------------------------------------------
# In production this would be a per-user token validated against a DB.
# For the hackathon we use a single configurable key (default provided for dev).
ACTIVATE_AI_API_KEY = os.environ.get("ACTIVATE_AI_API_KEY", "upay-activate-ai-dev-key-2026")

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(
    api_key: str | None = Security(_api_key_header),
) -> str:
    """Dependency that enforces a valid X-API-Key header."""
    if api_key is None or api_key != ACTIVATE_AI_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key

# ---------------------------------------------------------------------------
# Initialise persistent storage on startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
def _startup():
    init_db()

# ---------------------------------------------------------------------------
# Data loading (lazy singleton)
# ---------------------------------------------------------------------------
BASE_DIR = pathlib.Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
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
# Request / Response models
# ---------------------------------------------------------------------------
class SimulateRequest(BaseModel):
    budget_bdt: float = Field(gt=0, description="Total reactivation budget in BDT")
    max_fatigue_cap: int = Field(default=70, ge=0, le=100, description="Max fatigue score allowed")
    min_uplift_cutoff: float = Field(default=0.02, ge=0, le=1, description="Min uplift score threshold")
    lifecycle_stage: str | None = Field(default=None, description="Optional lifecycle stage filter")


class ApprovalRequest(BaseModel):
    admin_user: str = Field(min_length=1, description="Name of the human reviewer / admin")
    allocated_budget: float = Field(gt=0, description="Approved budget in BDT")
    notes: str = Field(default="", description="Optional reviewer notes")


class BatchPredictRequest(BaseModel):
    customers: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Data loading helpers (lazy singleton)
# ---------------------------------------------------------------------------
def _load_engine() -> Any:
    if "engine" not in _cache:
        from backend.ml_engine import ActivateAIEngine
        _cache["engine"] = ActivateAIEngine.load(str(MODELS_DIR))
    return _cache["engine"]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/predict/batch", dependencies=[Depends(require_api_key)])
@limiter.limit("1000/minute")
def predict_batch(req: BatchPredictRequest, request: Request):
    """Dynamically run LightGBM inference + SHAP on a batch of raw customer features."""
    try:
        engine = _load_engine()
    except ImportError:
        # Fallback for lightweight serverless environments (e.g. Vercel)
        # where LightGBM/SHAP exceed deployment size limits.
        return {"predictions": [], "error": "ML engine dependencies (LightGBM/SHAP) are not available in this environment."}
        
    df = pd.DataFrame(req.customers)
    if df.empty:
        return {"predictions": []}
        
    scored_df = engine.predict_batch(df)
    
    # Parse the json strings back to dicts for API response
    scored_df["shap_top3"] = scored_df["shap_top3"].apply(json.loads)
    
    return {"predictions": scored_df.to_dict(orient="records")}

import hashlib

@app.post("/api/ingest/governed-data", dependencies=[Depends(require_api_key)])
@limiter.limit("60/minute")
def ingest_governed_data(req: BatchPredictRequest, request: Request):
    """
    Data Governance Gateway.
    Simulates ingesting raw production data, stripping PII, hashing identifiers,
    and returning a sanitized schema ready for the batch inference pipeline.
    """
    raw_df = pd.DataFrame(req.customers)
    if raw_df.empty:
        return {"status": "success", "sanitized_records": 0, "data": []}
        
    # PII Stripping Guardrails
    pii_columns = ["customer_name", "phone_number", "national_id", "email", "address", "gps_location"]
    sanitized_df = raw_df.drop(columns=[c for c in pii_columns if c in raw_df.columns], errors='ignore')
    
    # One-way hashing for identifiers
    if "customer_id" in sanitized_df.columns:
        sanitized_df["customer_id"] = sanitized_df["customer_id"].apply(
            lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:16]
        )
        
    # Forward the sanitized payload to the batch prediction logic internally if needed,
    # or just return the clean governed payload to demonstrate the ETL pipeline.
    
    return {
        "status": "success", 
        "sanitized_records": len(sanitized_df),
        "data": sanitized_df.to_dict(orient="records")
    }

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


@app.post("/api/simulate-mau-growth", dependencies=[Depends(require_api_key)])
@limiter.limit("60/minute")
def simulate_mau_growth(req: SimulateRequest, request: Request):
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


@app.get("/api/customer/{customer_id}", dependencies=[Depends(require_api_key)])
@limiter.limit("60/minute")
def get_customer(customer_id: str, request: Request):
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


@app.post("/api/approve-campaign", dependencies=[Depends(require_api_key)])
@limiter.limit("60/minute")
def approve_campaign(req: ApprovalRequest, request: Request):
    """Human-in-the-loop campaign approval (AGENTS.md guardrail #5).

    Now persisted to SQLite — survives server restarts.
    """
    next_id = count_approvals() + 1
    campaign_id = f"CAMP_{next_id:04d}"

    record = insert_approval(
        campaign_id=campaign_id,
        admin_user=req.admin_user,
        allocated_budget=req.allocated_budget,
        notes=req.notes,
    )
    return record


@app.get("/api/approvals", dependencies=[Depends(require_api_key)])
@limiter.limit("60/minute")
def get_approvals(request: Request):
    """List all campaign approvals from persistent ledger."""
    return {"approvals": list_approvals()}

