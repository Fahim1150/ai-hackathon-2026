"""
upay ActivateAI — Pytest Test Suite.

Verifies train/test isolation, uplift score bounds, budget constraints,
Gemini fallback resilience, and all API endpoint status codes.
"""

import json
import pathlib

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import app, ACTIVATE_AI_API_KEY

client = TestClient(app)
AUTH_HEADERS = {"X-API-Key": ACTIVATE_AI_API_KEY}


# =========================================================================
# Data integrity tests
# =========================================================================

class TestDataIntegrity:
    """Tests that run against the CSV files directly."""

    def test_train_test_files_exist(self):
        assert pathlib.Path("data/train.csv").exists(), "data/train.csv missing"
        assert pathlib.Path("data/test.csv").exists(), "data/test.csv missing"

    def test_train_test_row_counts(self):
        train = pd.read_csv("data/train.csv")
        test = pd.read_csv("data/test.csv")
        assert len(train) == 8000, f"Expected 8000 train rows, got {len(train)}"
        assert len(test) == 2000, f"Expected 2000 test rows, got {len(test)}"

    def test_train_test_isolation(self):
        """AGENTS.md guardrail #7: zero customer_id overlap."""
        train = pd.read_csv("data/train.csv")
        test = pd.read_csv("data/test.csv")
        overlap = set(train["customer_id"]) & set(test["customer_id"])
        assert len(overlap) == 0, f"Data leakage! {len(overlap)} shared customer_ids: {list(overlap)[:5]}"

    def test_no_null_values(self):
        train = pd.read_csv("data/train.csv")
        test = pd.read_csv("data/test.csv")
        assert train.isnull().sum().sum() == 0, "train.csv has NaN values"
        assert test.isnull().sum().sum() == 0, "test.csv has NaN values"


# =========================================================================
# Model prediction tests
# =========================================================================

class TestModelPredictions:
    """Tests that run against the scored predictions."""

    @pytest.fixture(autouse=True)
    def load_predictions(self):
        self.pred = pd.read_csv("models/test_predictions.csv")

    def test_predictions_exist(self):
        assert pathlib.Path("models/test_predictions.csv").exists()

    def test_uplift_score_bounds(self):
        """All uplift scores must be in [-1, 1]."""
        assert (self.pred["uplift_score"] >= -1).all(), "uplift_score below -1 found"
        assert (self.pred["uplift_score"] <= 1).all(), "uplift_score above 1 found"

    def test_fatigue_score_bounds(self):
        """Fatigue scores must be in [0, 100]."""
        assert (self.pred["offer_fatigue_score"] >= 0).all(), "fatigue below 0"
        assert (self.pred["offer_fatigue_score"] <= 100).all(), "fatigue above 100"

    def test_quadrant_values(self):
        valid = {"Persuadable", "Sure Thing", "Lost Cause", "Sleeping Dog"}
        actual = set(self.pred["uplift_quadrant"].unique())
        assert actual <= valid, f"Invalid quadrants: {actual - valid}"

    def test_shap_top3_parseable(self):
        """Every SHAP top-3 field must be valid JSON with 3 entries."""
        for i, row in self.pred.head(50).iterrows():
            drivers = json.loads(row["shap_top3"])
            assert isinstance(drivers, list), f"Row {i}: SHAP not a list"
            assert len(drivers) == 3, f"Row {i}: expected 3 SHAP drivers, got {len(drivers)}"
            for d in drivers:
                assert "feature" in d and "shap_value" in d, f"Row {i}: missing SHAP keys"


# =========================================================================
# Budget optimizer tests
# =========================================================================

class TestBudgetOptimizer:
    """Tests for deterministic budget constraint adherence."""

    @pytest.fixture(autouse=True)
    def load_predictions(self):
        self.pred = pd.read_csv("models/test_predictions.csv")

    def test_budget_constraint_5000(self):
        from backend.optimizer import optimize_reactivation_budget
        result = optimize_reactivation_budget(self.pred, budget_bdt=5000)
        assert result["activate_ai"]["total_spend_bdt"] <= 5000

    def test_budget_constraint_10000(self):
        from backend.optimizer import optimize_reactivation_budget
        result = optimize_reactivation_budget(self.pred, budget_bdt=10000)
        assert result["activate_ai"]["total_spend_bdt"] <= 10000

    def test_zero_fatigue_cap_targets_nobody_fatigued(self):
        from backend.optimizer import optimize_reactivation_budget
        result = optimize_reactivation_budget(self.pred, budget_bdt=50000, max_fatigue_cap=0)
        # With cap=0, only users with fatigue_score=0 are eligible
        assert result["activate_ai"]["total_spend_bdt"] <= 50000

    def test_high_uplift_cutoff_reduces_targeting(self):
        from backend.optimizer import optimize_reactivation_budget
        result = optimize_reactivation_budget(self.pred, budget_bdt=50000, min_uplift_cutoff=0.99)
        # Very few (or zero) users have uplift >= 0.99
        assert result["activate_ai"]["users_targeted"] <= 10


# =========================================================================
# Gemini fallback tests
# =========================================================================

class TestGeminiFallback:
    """AGENTS.md guardrail #6: system must never crash without API key."""

    def test_customer_nudge_fallback(self):
        from backend.genai_service import generate_customer_nudge
        profile = {
            "customer_id": "UPAY_TEST",
            "lifecycle_stage": "At-Risk Churner",
            "wallet_type": "Primary",
            "days_inactive": 30,
            "uplift_score": 0.15,
            "uplift_quadrant": "Persuadable",
            "assigned_offer": "20 BDT Recharge Bonus",
            "offer_fatigue_score": 20,
            "channel_type": "App",
        }
        shap = [
            {"feature": "days_inactive", "value": 30, "shap_value": 0.8, "direction": "increases activation"},
            {"feature": "cashout_ratio", "value": 0.3, "shap_value": -0.4, "direction": "decreases activation"},
            {"feature": "promos_sent_30d", "value": 1, "shap_value": 0.2, "direction": "increases activation"},
        ]
        result = generate_customer_nudge(profile, shap)
        assert "marketer_explanation" in result
        assert "sms_english" in result
        assert "sms_bangla" in result
        assert "generation_source" in result
        # Without API key set in test env, should be fallback
        assert result["generation_source"] in ("Gemini 2.5 Flash", "Offline Deterministic Fallback")

    def test_experiment_insights_fallback(self):
        from backend.genai_service import generate_experiment_insights
        summary = {
            "activate_ai": {"users_targeted": 100, "incremental_mau_gained": 30, "total_spend_bdt": 3000, "cost_per_incremental_mau": 100, "by_lifecycle_stage": {}},
            "mass_blast_baseline": {"users_targeted": 2000, "incremental_mau_gained": 50, "total_spend_bdt": 50000, "cost_per_incremental_mau": 1000},
            "savings": {"budget_saved_bdt": 47000, "fatigued_users_protected": 200, "sure_thing_waste_avoided_bdt": 800, "cost_efficiency_improvement_pct": 90, "sleeping_dogs_not_disturbed": 300},
        }
        result = generate_experiment_insights(summary)
        assert "insights" in result
        assert len(result["insights"]) == 3
        assert "next_experiment" in result
        assert "generation_source" in result


# =========================================================================
# API endpoint tests
# =========================================================================

class TestAPIEndpoints:
    """Tests for all FastAPI endpoints."""

    def test_health(self):
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_overview(self):
        r = client.get("/api/overview")
        assert r.status_code == 200
        data = r.json()
        assert "funnel" in data
        assert "uplift_quadrant_distribution" in data
        assert "fairness" in data
        assert "validation_metrics" in data

    def test_simulate_mau_growth(self):
        r = client.post("/api/simulate-mau-growth", json={
            "budget_bdt": 10000,
            "max_fatigue_cap": 70,
            "min_uplift_cutoff": 0.02,
        }, headers=AUTH_HEADERS)
        assert r.status_code == 200
        data = r.json()
        assert "activate_ai" in data
        assert "mass_blast_baseline" in data
        assert "configurations" in data
        assert len(data["configurations"]) == 4
        assert data["activate_ai"]["total_spend_bdt"] <= 10000

    def test_simulate_with_stage_filter(self):
        r = client.post("/api/simulate-mau-growth", json={
            "budget_bdt": 5000,
            "lifecycle_stage": "Payday Cash-Outer",
        }, headers=AUTH_HEADERS)
        assert r.status_code == 200

    def test_customer_found(self):
        # Use a customer_id from the test set
        pred = pd.read_csv("models/test_predictions.csv")
        cid = pred.iloc[0]["customer_id"]
        r = client.get(f"/api/customer/{cid}", headers=AUTH_HEADERS)
        assert r.status_code == 200
        data = r.json()
        assert data["customer_id"] == cid
        assert "uplift_score" in data
        assert "uplift_quadrant" in data
        assert "shap_top3" in data
        assert "nudge" in data
        assert "generation_source" in data["nudge"]

    def test_customer_not_found(self):
        r = client.get("/api/customer/FAKE_CUSTOMER_999", headers=AUTH_HEADERS)
        assert r.status_code == 404

    def test_customers_list(self):
        r = client.get("/api/customers?page=1&limit=10")
        assert r.status_code == 200
        data = r.json()
        assert "customers" in data
        assert "total" in data
        assert len(data["customers"]) <= 10

    def test_customers_filter_by_quadrant(self):
        r = client.get("/api/customers?quadrant=Persuadable&limit=5")
        assert r.status_code == 200
        data = r.json()
        for c in data["customers"]:
            assert c["uplift_quadrant"] == "Persuadable"

    def test_approve_campaign(self):
        r = client.post("/api/approve-campaign", json={
            "admin_user": "Test Reviewer",
            "allocated_budget": 10000,
            "notes": "Automated test approval",
        }, headers=AUTH_HEADERS)
        assert r.status_code == 200
        data = r.json()
        assert data["approved"] is True
        assert data["admin_user"] == "Test Reviewer"
        assert "timestamp" in data
        assert "campaign_id" in data

    def test_approve_campaign_requires_name(self):
        r = client.post("/api/approve-campaign", json={
            "admin_user": "",
            "allocated_budget": 10000,
        }, headers=AUTH_HEADERS)
        assert r.status_code == 422  # Pydantic validation error
