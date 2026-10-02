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
    Rank eligible dormant users by incremental MAU efficiency and greedily
    allocate budget. Returns ActivateAI vs Mass Blast comparison.

    Parameters
    ----------
    predictions_df : DataFrame with columns from ml_engine.predict()
    budget_bdt     : Total marketing budget in BDT
    max_fatigue_cap: Exclude users with offer_fatigue_score > this
    min_uplift_cutoff: Exclude users with uplift_score < this
    target_stage   : Optional lifecycle_stage filter
    """
    df = predictions_df.copy()

    # ------------------------------------------------------------------
    # 1. Mass Blast Baseline (target everyone, no filtering)
    # ------------------------------------------------------------------
    blast = _compute_blast_baseline(df)

    # ------------------------------------------------------------------
    # 2. ActivateAI: filter → rank → greedily allocate
    # ------------------------------------------------------------------
    eligible = df.copy()

    # Apply filters
    eligible = eligible[eligible["offer_fatigue_score"] <= max_fatigue_cap]
    eligible = eligible[eligible["uplift_score"] >= min_uplift_cutoff]
    if target_stage:
        eligible = eligible[eligible["lifecycle_stage"] == target_stage]

    # Rank by incremental MAU efficiency
    eligible = eligible.copy()
    eligible["efficiency"] = eligible["uplift_score"] / eligible["assigned_offer_cost_bdt"].clip(lower=1.0)
    eligible = eligible.sort_values("efficiency", ascending=False)

    # Greedy allocation under budget constraint
    cumulative_cost = eligible["assigned_offer_cost_bdt"].cumsum()
    selected = eligible[cumulative_cost <= budget_bdt]

    activate_ai = _compute_activate_ai_metrics(selected, df)

    # ------------------------------------------------------------------
    # 3. Savings comparison
    # ------------------------------------------------------------------
    savings = _compute_savings(activate_ai, blast, df, max_fatigue_cap)

    return {
        "activate_ai": activate_ai,
        "mass_blast_baseline": blast,
        "savings": savings,
        "parameters": {
            "budget_bdt": budget_bdt,
            "max_fatigue_cap": max_fatigue_cap,
            "min_uplift_cutoff": min_uplift_cutoff,
            "target_stage": target_stage,
        },
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_blast_baseline(df: pd.DataFrame) -> dict:
    """Mass blast: send every user their matched offer, no filtering."""
    total_spend = float(df["assigned_offer_cost_bdt"].sum())
    total_uplift = float(df["uplift_score"].sum())
    n = len(df)

    # Waste on Sure Things
    sure_things = df[df["uplift_quadrant"] == "Sure Thing"]
    waste_sure_things = float(sure_things["assigned_offer_cost_bdt"].sum())

    # Sleeping Dogs spammed
    sleeping_dogs = df[df["uplift_quadrant"] == "Sleeping Dog"]
    fatigued_spammed = int(len(sleeping_dogs))

    cost_per_mau = total_spend / max(total_uplift, 1e-9)

    return {
        "users_targeted": n,
        "incremental_mau_gained": round(total_uplift, 2),
        "total_spend_bdt": round(total_spend, 2),
        "cost_per_incremental_mau": round(cost_per_mau, 2),
        "waste_on_sure_things_bdt": round(waste_sure_things, 2),
        "fatigued_users_spammed": fatigued_spammed,
        "negative_uplift_users": int((df["uplift_score"] < 0).sum()),
        "by_lifecycle_stage": _breakdown_by(df, "lifecycle_stage"),
        "by_offer_type": _breakdown_by(df, "assigned_offer"),
    }


def _compute_activate_ai_metrics(selected: pd.DataFrame, full_df: pd.DataFrame) -> dict:
    """Metrics for ActivateAI's optimized targeting."""
    if len(selected) == 0:
        return {
            "users_targeted": 0,
            "incremental_mau_gained": 0.0,
            "total_spend_bdt": 0.0,
            "cost_per_incremental_mau": 0.0,
            "by_lifecycle_stage": {},
            "by_offer_type": {},
        }

    total_spend = float(selected["assigned_offer_cost_bdt"].sum())
    total_uplift = float(selected["uplift_score"].sum())
    cost_per_mau = total_spend / max(total_uplift, 1e-9)

    return {
        "users_targeted": int(len(selected)),
        "incremental_mau_gained": round(total_uplift, 2),
        "total_spend_bdt": round(total_spend, 2),
        "cost_per_incremental_mau": round(cost_per_mau, 2),
        "by_lifecycle_stage": _breakdown_by(selected, "lifecycle_stage"),
        "by_offer_type": _breakdown_by(selected, "assigned_offer"),
    }


def _compute_savings(ai: dict, blast: dict, df: pd.DataFrame, fatigue_cap: int) -> dict:
    """Delta metrics: what ActivateAI saves vs mass blast."""
    budget_saved = blast["total_spend_bdt"] - ai["total_spend_bdt"]

    # Fatigued users that ActivateAI did NOT target
    fatigued_protected = int((df["offer_fatigue_score"] > fatigue_cap).sum())

    # Sure Thing waste avoided
    sure_thing_waste = blast["waste_on_sure_things_bdt"]

    # MAU efficiency improvement
    blast_mau = blast["incremental_mau_gained"]
    ai_mau = ai["incremental_mau_gained"]
    if blast_mau > 0:
        mau_lift_pct = round((ai_mau - blast_mau) / blast_mau * 100, 1)
    else:
        mau_lift_pct = 0.0

    # Cost efficiency improvement
    if blast["cost_per_incremental_mau"] > 0 and ai["cost_per_incremental_mau"] > 0:
        cost_reduction_pct = round(
            (1 - ai["cost_per_incremental_mau"] / blast["cost_per_incremental_mau"]) * 100, 1
        )
    else:
        cost_reduction_pct = 0.0

    return {
        "budget_saved_bdt": round(budget_saved, 2),
        "fatigued_users_protected": fatigued_protected,
        "sure_thing_waste_avoided_bdt": round(sure_thing_waste, 2),
        "mau_lift_improvement_pct": mau_lift_pct,
        "cost_efficiency_improvement_pct": cost_reduction_pct,
        "sleeping_dogs_not_disturbed": blast["fatigued_users_spammed"],
    }


def _breakdown_by(df: pd.DataFrame, col: str) -> dict:
    """Group-level metrics for a given column."""
    result = {}
    for val, g in df.groupby(col):
        result[val] = {
            "count": int(len(g)),
            "total_spend_bdt": round(float(g["assigned_offer_cost_bdt"].sum()), 2),
            "incremental_mau": round(float(g["uplift_score"].sum()), 2),
            "mean_uplift": round(float(g["uplift_score"].mean()), 4),
        }
    return result


# ---------------------------------------------------------------------------
# CLI quick-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    df = pd.read_csv("models/test_predictions.csv")
    print("=" * 60)
    print("upay ActivateAI — Budget Optimizer Quick Test")
    print("=" * 60)

    for budget in [5_000, 10_000, 25_000, 50_000]:
        result = optimize_reactivation_budget(df, budget_bdt=budget)
        ai = result["activate_ai"]
        bl = result["mass_blast_baseline"]
        sv = result["savings"]
        print(f"\n--- Budget: {budget:,} BDT ---")
        print(f"  ActivateAI:  {ai['users_targeted']} users, "
              f"{ai['incremental_mau_gained']:.1f} incremental MAU, "
              f"{ai['total_spend_bdt']:.0f} BDT spend, "
              f"{ai['cost_per_incremental_mau']:.0f} BDT/MAU")
        print(f"  Mass Blast:  {bl['users_targeted']} users, "
              f"{bl['incremental_mau_gained']:.1f} incremental MAU, "
              f"{bl['total_spend_bdt']:.0f} BDT spend, "
              f"{bl['cost_per_incremental_mau']:.0f} BDT/MAU")
        print(f"  Savings:     {sv['budget_saved_bdt']:.0f} BDT saved, "
              f"{sv['fatigued_users_protected']} fatigued protected, "
              f"{sv['cost_efficiency_improvement_pct']:.0f}% cost improvement")
        assert ai["total_spend_bdt"] <= budget, f"Budget violated! {ai['total_spend_bdt']} > {budget}"

    print("\n✅ All budget constraints respected.")
    print("🎯 Done. Ready for Phase 5 (GenAI Service).")
