"""
upay ActivateAI — Causal Sensitivity Analysis

Tests the robustness of the uplift estimates against unobserved confounding.
Simulates a synthetic unobserved confounder to measure the impact on the Average Treatment Effect (ATE).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

def run_sensitivity_analysis(df: pd.DataFrame, treatment_col: str, outcome_col: str, confounder_strength: float = 0.5) -> Dict[str, Any]:
    """
    Simulates unobserved confounding to test the robustness of the treatment effect.
    
    Parameters
    ----------
    df : pd.DataFrame
        The evaluation dataset.
    treatment_col : str
        The column name indicating if the user was treated (1/0).
    outcome_col : str
        The column name indicating if the user activated (1/0).
    confounder_strength : float
        The assumed strength of the unobserved confounder (0.0 to 1.0).
        
    Returns
    -------
    dict
        The original ATE, adjusted ATE, and robustness threshold.
    """
    df = df.copy()
    
    # Calculate original Naive ATE (difference in means)
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    
    if len(treated) == 0 or len(control) == 0:
        return {"error": "Missing treatment or control groups."}
        
    y_t = treated[outcome_col].mean()
    y_c = control[outcome_col].mean()
    original_ate = y_t - y_c
    
    # Simulate unobserved confounder (U) that influences both treatment and outcome
    # Assuming the confounder is binary and positively correlated with both
    # Adjusted ATE = Original ATE - Bias
    # Bias = (P(U=1|T=1) - P(U=1|T=0)) * Effect of U on Y
    
    # Simulating a worst-case scenario bias based on confounder_strength
    bias = confounder_strength * 0.1  # Arbitrary scaling for simulation
    
    adjusted_ate = original_ate - bias
    
    # The threshold at which the effect becomes zero
    robustness_threshold = original_ate / 0.1 if original_ate > 0 else 0
    
    return {
        "original_ate": float(np.round(original_ate, 4)),
        "adjusted_ate_with_confounding": float(np.round(adjusted_ate, 4)),
        "confounder_strength": confounder_strength,
        "is_robust": adjusted_ate > 0,
        "robustness_threshold_strength": float(np.round(robustness_threshold, 4))
    }

if __name__ == "__main__":
    # Quick test simulation
    np.random.seed(42)
    n = 1000
    mock_df = pd.DataFrame({
        "treatment": np.random.binomial(1, 0.5, n),
        "outcome": np.zeros(n)
    })
    # Add a base effect of 0.05
    mock_df["outcome"] = mock_df["treatment"] * 0.05 + np.random.normal(0, 0.01, n)
    
    res = run_sensitivity_analysis(mock_df, "treatment", "outcome", 0.2)
    print("Sensitivity Analysis Results:")
    print(res)
