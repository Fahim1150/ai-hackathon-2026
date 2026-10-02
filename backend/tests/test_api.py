import subprocess, sys, pathlib
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.rules import segment, apply_rules

client = TestClient(app)
GOOD = {"monthly_trans_count": 12, "avg_trans_amount": 300, "days_since_last_trans": 25,
        "tenure_months": 12, "cashout_ratio": 0.3, "offers_last_30d": 0}

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

def test_recommend_and_plan():
    if not pathlib.Path("models/engine.joblib").exists():
        return  # run data generation + training first
    r = client.post("/recommend", json=GOOD).json()
    assert {"segment", "decision", "reasons", "uplift"} <= r.keys()
    p = client.post("/plan", json={"budget_bdt": 5000}).json()
    assert p["status"] == "PENDING_HUMAN_APPROVAL" and p["expected_spend_bdt"] <= 5000
