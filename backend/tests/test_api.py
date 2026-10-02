import subprocess, sys, pathlib
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.rules import segment, apply_rules

client = TestClient(app)

GOOD = {
    "user_id": "SYN_100000",
    "monthly_txn_count": 12, 
    "avg_txn_amount": 300, 
    "last_active_days": 25,
    "tenure_months": 12, 
    "cashout_share": 0.3, 
    "bill_pay_history": 2,
    "offers_last_30d": 0
}

def test_health():
    assert client.get("/health").json() == {"status": "ok"}

def test_segments():
    assert segment(0.7, 0.0).startswith("Sure Thing")
    assert segment(0.2, 0.2).startswith("Persuadable")
    assert segment(0.2, -0.05).startswith("Sleeping Dog")
    assert segment(0.1, 0.0).startswith("Lost Cause")

def test_fatigue_rule_blocks():
    ok, why = apply_rules(5, 50.0, 0.2)
    assert not ok and any("fatigue" in w.lower() for w in why)

def test_recommend():
    if not pathlib.Path("models/engine.joblib").exists():
        return  # skip if not trained
    r = client.post("/recommend", json=GOOD).json()
    assert {"segment", "decision", "reasons", "uplift"} <= r.keys()

def test_plan_and_approve():
    if not pathlib.Path("models/engine.joblib").exists():
        return
    p = client.post("/plan", json={"budget_bdt": 5000}).json()
    assert p["status"] == "PENDING_HUMAN_APPROVAL" and p["expected_spend_bdt"] <= 5000
    camp_id = p.get("campaign_id")
    if camp_id:
        appr = client.post(f"/campaigns/{camp_id}/approve").json()
        assert appr["status"] == "APPROVED"

def test_optimize():
    if not pathlib.Path("models/engine.joblib").exists():
        return
    r = client.post("/budget/optimize", json={"budget_bdt": 10000, "max_per_user_cost": 25.0}).json()
    assert r["expected_spend_bdt"] <= 10000
    assert "baseline_comparison" in r

def test_customer_offers():
    if not pathlib.Path("models/engine.joblib").exists():
        return
    r = client.get("/customers/SYN_100000/offers")
    assert r.status_code in [200, 404]  # 404 if user not in sample

def test_fatigue_endpoint():
    if not pathlib.Path("models/engine.joblib").exists():
        return
    r = client.get("/campaigns/fatigue").json()
    assert "fatigue_percentage" in r

def test_experiments_compare():
    r = client.post("/experiments/compare", json={
        "variant_a_users": 1000, "variant_a_conversions": 100,
        "variant_b_users": 1000, "variant_b_conversions": 120
    }).json()
    assert "relative_lift" in r
    assert r["relative_lift"] > 0
    assert "p_value" in r

def test_fairness():
    if not pathlib.Path("models/engine.joblib").exists():
        return
    r = client.get("/fairness").json()
    assert "region" in r
    assert "age_band" in r

def test_segments_endpoint():
    if not pathlib.Path("models/engine.joblib").exists():
        return
    r = client.get("/uplift/segments").json()
    assert len(r) > 0
