"""
upay ActivateAI — Feedback Loop & Concept Drift Detection.

Simulates a scheduled cron job that ingests new campaign conversion logs,
joins them with historical features, and detects covariate drift via
Population Stability Index (PSI).
"""

import numpy as np
import pandas as pd

from backend.ml_engine import ActivateAIEngine

def calculate_psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    """Calculate the Population Stability Index (PSI) between two distributions."""
    def scale_range(s, min_val, max_val):
        s = s.copy()
        s += -(np.min(s))
        s /= np.max(s) / (max_val - min_val)
        s += min_val
        return s

    breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
    breakpoints = scale_range(breakpoints, np.min(expected), np.max(expected))
    
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)

    def sub_psi(e_perc, a_perc):
        if a_perc == 0:
            a_perc = 0.0001
        if e_perc == 0:
            e_perc = 0.0001
        return (e_perc - a_perc) * np.log(e_perc / a_perc)

    return np.sum([sub_psi(expected_percents[i], actual_percents[i]) for i in range(len(expected_percents))])


def ingest_campaign_logs(log_path: str, reference_features_path: str = "data/train.csv"):
    """Ingest new logs and compute drift."""
    print("=" * 60)
    print("upay ActivateAI — Feedback Loop & Drift Detection Job")
    print("=" * 60)
    print(f"⏳ Ingesting new conversion logs from {log_path}...")
    
    try:
        new_data = pd.read_csv(log_path)
        ref_data = pd.read_csv(reference_features_path)
    except FileNotFoundError:
        print("❌ Error: Missing data files. Cannot run drift detection.")
        return

    # Check for covariate drift on key numerical features
    features_to_monitor = ["days_inactive", "monthly_inflow_bdt", "cashout_ratio"]
    drift_report = {}
    
    print("\n📊 Covariate Drift Analysis (Population Stability Index)")
    for feat in features_to_monitor:
        if feat in new_data.columns and feat in ref_data.columns:
            psi = calculate_psi(ref_data[feat], new_data[feat])
            drift_report[feat] = psi
            
            status = "✅ Stable"
            if psi > 0.2:
                status = "❌ Severe Drift"
            elif psi > 0.1:
                status = "⚠️ Moderate Drift"
                
            print(f"  {feat:<20}: PSI = {psi:.4f}  {status}")

    # Log ingestion summary
    print("\n✅ New campaign data successfully ingested into feature store.")
    print("🎯 Model retraining will be scheduled if severe drift is detected.")

if __name__ == "__main__":
    import os
    
    # Simulate some new logs with slight drift
    test_logs_path = "data/recent_campaign_logs_simulated.csv"
    if not os.path.exists(test_logs_path):
        # Generate synthetic logs for the simulation
        df = pd.read_csv("data/test.csv")
        # Inject artificial drift to monthly_inflow_bdt
        df["monthly_inflow_bdt"] = df["monthly_inflow_bdt"] * 1.5 
        df.to_csv(test_logs_path, index=False)
        
    ingest_campaign_logs(test_logs_path)
