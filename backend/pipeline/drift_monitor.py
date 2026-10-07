"""
upay ActivateAI — Model Registry & Drift Monitoring

Compares incoming batch inference data against the original train.csv
distributions using Population Stability Index (PSI).
Logs drift metrics and alerts into the persistent model_registry_logs SQLite table.
"""

import numpy as np
import pandas as pd
from backend.database import log_model_drift

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


def run_drift_check(new_data_path: str, reference_data_path: str = "data/train.csv", model_version: str = "v2.0.0"):
    """Run drift check on incoming batch data and log to registry."""
    print("=" * 60)
    print(f"upay ActivateAI — Drift Monitor (Version: {model_version})")
    print("=" * 60)
    
    try:
        new_data = pd.read_csv(new_data_path)
        ref_data = pd.read_csv(reference_data_path)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    features_to_monitor = ["days_inactive", "monthly_inflow_bdt", "cashout_ratio", "historical_tx_count"]
    
    for feat in features_to_monitor:
        if feat in new_data.columns and feat in ref_data.columns:
            psi = float(calculate_psi(ref_data[feat], new_data[feat]))
            
            status = "Stable"
            if psi > 0.2:
                status = "Severe Drift"
            elif psi > 0.1:
                status = "Moderate Drift"
                
            # Log to persistent SQLite ledger
            log_model_drift(
                model_version=model_version,
                feature_name=feat,
                drift_score=psi,
                status=status
            )
            
            icon = "✅" if status == "Stable" else ("❌" if status == "Severe Drift" else "⚠️")
            print(f"  {feat:<22}: PSI = {psi:.4f}  {icon} {status}")
            
    print("\n✅ Drift monitoring complete. Results logged to model_registry_logs table.")

if __name__ == "__main__":
    # Test script with simulated drifted data
    import os
    test_logs_path = "data/recent_campaign_logs_simulated.csv"
    if os.path.exists(test_logs_path):
        run_drift_check(test_logs_path)
    else:
        print("Run feedback_loop.py first to generate simulated data.")
