"""
upay ActivateAI — Deterministic MAU Budget Optimizer.

Pure business logic — no ML, no LLM. Ranks dormant users by incremental
MAU efficiency (uplift / cost), greedily allocates a fixed BDT budget,
and compares results against a naive "Mass SMS Blast" baseline.

Per AGENTS.md guardrail #2: budget caps, fatigue suppression, and eligibility
filters are deterministic and auditable — never inside the ML model.
"""

from __future__ import annotations

import pandas as pd


def optimize_reactivation_budget(
    predictions_df: pd.DataFrame,
    budget_bdt: float,
    max_fatigue_cap: int = 70,
    min_uplift_cutoff: float = 0.02,
    target_stage: str | None = None,
) -> dict:
    """
    Simulate 4 distinct targeting configurations for a given budget:
    1. Mass Blast Baseline
    2. Propensity-Only
    3. Causal Uplift Only
    4. ActivateAI Full Stack
    """
    df = predictions_df.copy()

    # Pre-compute propensity score column (use base_propensity if available)
    if "base_propensity" not in df.columns:
        df["base_propensity"] = df["prob_active_control"]

    if target_stage:
        df = df[df["lifecycle_stage"] == target_stage]

    # Helper to calculate standardized metrics for a targeted subset
    def _calc_metrics(targeted: pd.DataFrame, name: str) -> dict:
        total_spend = float(targeted["assigned_offer_cost_bdt"].sum())
        total_uplift = float(targeted["uplift_score"].sum())
        cost_per_mau = total_spend / max(total_uplift, 1e-9)

        # Waste = money spent on Sure Things (would have activated anyway) + Sleeping Dogs (negative impact)
        waste_mask = targeted["uplift_quadrant"].isin(["Sure Thing", "Sleeping Dog", "Lost Cause"])
        cashback_waste = float(targeted.loc[waste_mask, "assigned_offer_cost_bdt"].sum())

        fatigue_count = (targeted["offer_fatigue_score"] > max_fatigue_cap).sum()
        fatigue_rate = float(fatigue_count / max(len(targeted), 1))

        return {
            "name": name,
            "users_targeted": int(len(targeted)),
            "incremental_mau_gained": round(total_uplift, 2),
            "total_spend_bdt": round(total_spend, 2),
            "cost_per_incremental_mau": round(cost_per_mau, 2),
            "cashback_waste_bdt": round(cashback_waste, 2),
            "opt_out_fatigue_rate": round(fatigue_rate, 4),
        }

    # 1. Mass Blast Baseline (target everyone until budget runs out, usually random or all)
    # Since Mass Blast just targets everyone, if budget is limited, it targets a random subset.
    # To be fair, let's just take a random sample until budget runs out, or we can just say 
    # it targets the first N users.
    # Actually, previous mass blast didn't respect budget. Let's make it respect budget by random order.
    df_shuffled = df.sample(frac=1.0, random_state=42).copy()
    blast_cum_cost = df_shuffled["assigned_offer_cost_bdt"].cumsum()
    blast_targeted = df_shuffled[blast_cum_cost <= budget_bdt]
    blast = _calc_metrics(blast_targeted, "Mass SMS Blast")

    # 2. Propensity-Only Targeting
    propensity = df.sort_values("base_propensity", ascending=False).copy()
    prop_cum_cost = propensity["assigned_offer_cost_bdt"].cumsum()
    prop_targeted = propensity[prop_cum_cost <= budget_bdt]
    propensity_metrics = _calc_metrics(prop_targeted, "Propensity-Only")

    # 3. Causal Uplift Only (ignores fatigue and efficiency)
    uplift = df.sort_values("uplift_score", ascending=False).copy()
    uplift_cum_cost = uplift["assigned_offer_cost_bdt"].cumsum()
    uplift_targeted = uplift[uplift_cum_cost <= budget_bdt]
    uplift_metrics = _calc_metrics(uplift_targeted, "Causal Uplift Only")

    # 4. ActivateAI Full Stack (Uplift + Fatigue Guardrail + Efficiency Knapsack)
    eligible = df.copy()
    eligible = eligible[eligible["offer_fatigue_score"] <= max_fatigue_cap]
    eligible = eligible[eligible["uplift_score"] >= min_uplift_cutoff]
    
    # Dynamic Offer Optimization (Pricing Markdown)
    # Calculate the minimum incentive tier (10, 15, 20 BDT) needed to maintain min_uplift_cutoff
    # Assuming uplift scales roughly linearly with incentive value
    def optimize_cost(row):
        u = row["uplift_score"]
        c = row["assigned_offer_cost_bdt"]
        if u <= 0 or c <= 0:
            return c
        # Minimum fraction of the offer needed to just cross the cutoff
        min_fraction = min_uplift_cutoff / u
        min_cost = c * min_fraction
        
        # Round up to nearest valid tier (10, 15, 20)
        if min_cost <= 10:
            return 10.0
        elif min_cost <= 15:
            return 15.0
        return float(c)

    eligible["assigned_offer_cost_bdt"] = eligible.apply(optimize_cost, axis=1)
    
    # Rank by incremental MAU efficiency
    eligible["efficiency"] = eligible["uplift_score"] / eligible["assigned_offer_cost_bdt"].clip(lower=1.0)
    eligible = eligible.sort_values("efficiency", ascending=False)
    
    ai_cum_cost = eligible["assigned_offer_cost_bdt"].cumsum()
    ai_targeted = eligible[ai_cum_cost <= budget_bdt]
    ai_metrics = _calc_metrics(ai_targeted, "ActivateAI Full Stack")

    return {
        "activate_ai": ai_metrics,  # Keep this for backward compatibility
        "mass_blast_baseline": blast,  # Keep this for backward compatibility
        "configurations": [
            blast,
            propensity_metrics,
            uplift_metrics,
            ai_metrics,
        ],
        "parameters": {
            "budget_bdt": budget_bdt,
            "max_fatigue_cap": max_fatigue_cap,
            "min_uplift_cutoff": min_uplift_cutoff,
            "target_stage": target_stage,
        },
    }


# ---------------------------------------------------------------------------
# CLI quick-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    df = pd.read_csv("models/test_predictions.csv")
    print("=" * 60)
    print("upay ActivateAI — Ablation Study Quick Test")
    print("=" * 60)

    for budget in [5_000, 10_000, 25_000, 50_000]:
        result = optimize_reactivation_budget(df, budget_bdt=budget)
        print(f"\n--- Budget: {budget:,} BDT ---")
        for c in result["configurations"]:
            print(f"  {c['name']:<25}: {c['users_targeted']:>4} users, "
                  f"{c['incremental_mau_gained']:>6.1f} MAU, "
                  f"{c['total_spend_bdt']:>5.0f} ৳ spend, "
                  f"{c['cost_per_incremental_mau']:>4.0f} ৳/MAU, "
                  f"{c['cashback_waste_bdt']:>5.0f} ৳ waste, "
                  f"{c['opt_out_fatigue_rate']*100:>4.1f}% fatigue")

    print("\n✅ All budget constraints respected.")
    print("🎯 Done.")
