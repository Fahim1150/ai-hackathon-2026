"""
upay ActivateAI — Grounded Gemini API Service.

Translates structured ML outputs into plain-language explanations and
bilingual (Bangla + English) SMS nudges using Gemini 3.8 Flash with
Pydantic Structured Outputs.

Per AGENTS.md guardrail #3: the LLM NEVER makes targeting, scoring, or
budget decisions. It only translates pre-computed SHAP attributions into
human-readable text.

Per AGENTS.md guardrail #6: every function degrades gracefully to a
deterministic template if GEMINI_API_KEY is missing or the call fails.
"""

from __future__ import annotations

import os
import json
import traceback
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Pydantic response schemas (used for Gemini structured output)
# ---------------------------------------------------------------------------

class CustomerNudge(BaseModel):
    """Structured response for a single customer nudge."""
    marketer_explanation: str = Field(
        description="2-3 sentence plain-language summary for the growth manager explaining "
                    "why this user is a good reactivation target and what drives the prediction."
    )
    sms_english: str = Field(
        description="SMS notification text in English, max 160 characters."
    )
    sms_bangla: str = Field(
        description="SMS notification text in Bangla (Unicode), max 160 characters."
    )
    compliance_note: str = Field(
        description="One-line compliance reminder (e.g., opt-out instructions)."
    )
    generation_source: str = Field(
        description="Either 'Gemini 3.8 Flash' or 'Offline Deterministic Fallback'."
    )


class ExperimentInsights(BaseModel):
    """Structured response for campaign experiment intelligence."""
    insights: list[str] = Field(
        description="Exactly 3 concise bullet points explaining cohort performance."
    )
    next_experiment: str = Field(
        description="Recommendation for the next A/B campaign experiment to run."
    )
    generation_source: str = Field(
        description="Either 'Gemini 3.8 Flash' or 'Offline Deterministic Fallback'."
    )


# ---------------------------------------------------------------------------
# Gemini client (lazy-initialized)
# ---------------------------------------------------------------------------
_client = None


def _get_client():
    """Lazy-initialize the Gemini client. Returns None if no API key."""
    global _client
    if _client is not None:
        return _client

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return None

    try:
        from google import genai
        _client = genai.Client(api_key=api_key)
        return _client
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Customer Nudge
# ---------------------------------------------------------------------------

def generate_customer_nudge(
    user_profile: dict[str, Any],
    shap_drivers: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Generate a personalized marketer explanation and bilingual SMS nudge.

    Args:
        user_profile: Dict with customer_id, lifecycle_stage, uplift_score,
                      uplift_quadrant, assigned_offer, offer_fatigue_score, etc.
        shap_drivers: List of top-3 SHAP dicts with feature, value, shap_value, direction.

    Returns:
        Dict matching CustomerNudge schema.
    """
    # Try Gemini first
    client = _get_client()
    if client is not None:
        try:
            return _gemini_customer_nudge(client, user_profile, shap_drivers)
        except Exception as e:
            print(f"⚠️  Gemini call failed, using fallback: {e}")

    # Deterministic fallback
    return _fallback_customer_nudge(user_profile, shap_drivers)


def _gemini_customer_nudge(
    client, user_profile: dict, shap_drivers: list[dict]
) -> dict[str, Any]:
    """Call Gemini 3.8 Flash with structured output."""
    from google.genai import types

    # Build grounded prompt from structured data only
    shap_text = "\n".join(
        f"  - {d['feature']} = {d['value']} → {d['direction']} (SHAP magnitude: {d.get('shap_value', 'N/A')})"
        for d in shap_drivers
    )

    prompt = f"""You are a bilingual (English/Bangla) MFS growth copywriter for upay Bangladesh.

Given the following structured ML output for a dormant upay user, produce:
1. A 2-3 sentence marketer explanation summarizing why this user is a reactivation target.
2. An SMS in English (≤160 chars) promoting the recommended offer.
3. An SMS in Bangla/Unicode (≤160 chars) promoting the same offer.
4. A one-line compliance note (include opt-out mention).

USER PROFILE:
  Customer ID: {user_profile.get('customer_id', 'N/A')}
  Lifecycle Stage: {user_profile.get('lifecycle_stage', 'N/A')}
  Wallet Type: {user_profile.get('wallet_type', 'N/A')}
  Days Inactive: {user_profile.get('days_inactive', 'N/A')}
  Uplift Score: {user_profile.get('uplift_score', 'N/A')}
  Uplift Quadrant: {user_profile.get('uplift_quadrant', 'N/A')}
  Recommended Offer: {user_profile.get('assigned_offer', 'N/A')}
  Offer Fatigue Score: {user_profile.get('offer_fatigue_score', 'N/A')}/100
  Channel: {user_profile.get('channel_type', 'N/A')}

TOP 3 SHAP DRIVERS:
{shap_text}

IMPORTANT: Use ONLY the data above. Do not fabricate statistics or features not listed."""

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=CustomerNudge,
            temperature=0.3,
        ),
    )

    result = json.loads(response.text)
    result["generation_source"] = "Gemini 3.8 Flash"
    return result


def _fallback_customer_nudge(
    user_profile: dict, shap_drivers: list[dict]
) -> dict[str, Any]:
    """Deterministic template-based fallback."""
    offer = user_profile.get("assigned_offer", "a special offer")
    stage = user_profile.get("lifecycle_stage", "dormant user")
    quadrant = user_profile.get("uplift_quadrant", "unknown")
    uplift = user_profile.get("uplift_score", 0)
    days = user_profile.get("days_inactive", "N/A")

    # Top SHAP driver
    top_driver = shap_drivers[0]["feature"] if shap_drivers else "recent activity"
    top_direction = shap_drivers[0].get("direction", "") if shap_drivers else ""

    explanation = (
        f"This {stage} user (inactive for {days} days) falls in the '{quadrant}' uplift quadrant "
        f"with an incremental uplift score of {uplift:.3f}. "
        f"The primary driver is '{top_driver}' which {top_direction}. "
        f"Sending '{offer}' is predicted to reactivate this user into a 30-day habit."
    )

    sms_en = f"upay: {offer}! You haven't used upay in {days} days. Tap to activate & save today."
    if len(sms_en) > 160:
        sms_en = sms_en[:157] + "..."

    sms_bn = f"upay: {offer}! আপনি {days} দিন ধরে upay ব্যবহার করেননি। এখনই এক্টিভেট করুন!"
    if len(sms_bn) > 160:
        sms_bn = sms_bn[:157] + "..."

    return {
        "marketer_explanation": explanation,
        "sms_english": sms_en,
        "sms_bangla": sms_bn,
        "compliance_note": "Reply STOP to opt out. upay is licensed by Bangladesh Bank.",
        "generation_source": "Offline Deterministic Fallback",
    }


# ---------------------------------------------------------------------------
# Experiment Insights
# ---------------------------------------------------------------------------

def generate_experiment_insights(
    simulation_summary: dict[str, Any],
) -> dict[str, Any]:
    """
    Generate 3 actionable experiment takeaways from budget simulation results.

    Args:
        simulation_summary: Dict from optimize_reactivation_budget() output.

    Returns:
        Dict matching ExperimentInsights schema.
    """
    client = _get_client()
    if client is not None:
        try:
            return _gemini_experiment_insights(client, simulation_summary)
        except Exception as e:
            print(f"⚠️  Gemini call failed, using fallback: {e}")

    return _fallback_experiment_insights(simulation_summary)


def _gemini_experiment_insights(client, summary: dict) -> dict[str, Any]:
    """Call Gemini 3.8 Flash for experiment intelligence."""
    from google.genai import types

    prompt = f"""You are an MFS growth analyst for upay Bangladesh.

Given the following campaign simulation results comparing ActivateAI (uplift-targeted) vs Mass SMS Blast, produce:
1. Exactly 3 concise bullet points (1-2 sentences each) explaining key cohort performance differences.
2. One recommendation for the next A/B campaign experiment to run.

SIMULATION RESULTS:
{json.dumps(summary, indent=2, default=str)}

Focus on actionable insights: which lifecycle stages benefit most, where budget is wasted, and what the fatigue/sleeping-dog data suggests."""

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ExperimentInsights,
            temperature=0.4,
        ),
    )

    result = json.loads(response.text)
    result["generation_source"] = "Gemini 3.8 Flash"
    return result


def _fallback_experiment_insights(summary: dict) -> dict[str, Any]:
    """Deterministic template-based fallback for experiment insights."""
    ai = summary.get("activate_ai", {})
    blast = summary.get("mass_blast_baseline", {})
    savings = summary.get("savings", {})

    ai_mau = ai.get("incremental_mau_gained", 0)
    blast_mau = blast.get("incremental_mau_gained", 0)
    ai_cost_per = ai.get("cost_per_incremental_mau", 0)
    blast_cost_per = blast.get("cost_per_incremental_mau", 0)
    fatigued_protected = savings.get("fatigued_users_protected", 0)
    waste_avoided = savings.get("sure_thing_waste_avoided_bdt", 0)
    sleeping_dogs = savings.get("sleeping_dogs_not_disturbed", 0)

    # Find best lifecycle stage
    best_stage = "unknown"
    best_uplift = -1
    by_stage = ai.get("by_lifecycle_stage", {})
    for stage, data in by_stage.items():
        mu = data.get("mean_uplift", 0)
        if mu > best_uplift:
            best_uplift = mu
            best_stage = stage

    insights = [
        f"ActivateAI generated {ai_mau:.1f} incremental MAU at {ai_cost_per:.0f} BDT/user "
        f"vs the mass blast's {blast_mau:.1f} MAU at {blast_cost_per:.0f} BDT/user — "
        f"a {savings.get('cost_efficiency_improvement_pct', 0):.0f}% cost efficiency gain.",

        f"'{best_stage}' users showed the highest uplift responsiveness. "
        f"{waste_avoided:.0f} BDT of cashback waste on 'Sure Things' was avoided by "
        f"not targeting users who would have reactivated anyway.",

        f"{fatigued_protected} fatigued users were protected from spam, and "
        f"{sleeping_dogs} 'Sleeping Dogs' (negative uplift) were not disturbed — "
        f"preventing churn risk from notification overload.",
    ]

    next_exp = (
        f"Run an A/B test isolating '{best_stage}' users: "
        f"compare the top-recommended solo-utility offer against a generic 'Welcome Back' "
        f"message to quantify the incremental value of affinity-matched offers vs generic nudges."
    )

    return {
        "insights": insights,
        "next_experiment": next_exp,
        "generation_source": "Offline Deterministic Fallback",
    }


# ---------------------------------------------------------------------------
# CLI quick-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 60)
    print("upay ActivateAI — GenAI Service Quick Test")
    print("=" * 60)

    # Test customer nudge
    test_profile = {
        "customer_id": "UPAY_108500",
        "lifecycle_stage": "Payday Cash-Outer",
        "wallet_type": "Salary",
        "days_inactive": 5,
        "uplift_score": 0.312,
        "uplift_quadrant": "Persuadable",
        "assigned_offer": "Zero-Fee Bill Pay + 30 BDT Cash Reward",
        "offer_fatigue_score": 12,
        "channel_type": "App",
    }
    test_shap = [
        {"feature": "cashout_ratio", "value": 0.95, "shap_value": 1.82, "direction": "increases activation"},
        {"feature": "monthly_inflow_bdt", "value": 35000, "shap_value": 0.94, "direction": "increases activation"},
        {"feature": "days_inactive", "value": 5, "shap_value": -0.45, "direction": "decreases activation"},
    ]

    print("\n--- Customer Nudge ---")
    nudge = generate_customer_nudge(test_profile, test_shap)
    print(json.dumps(nudge, indent=2, ensure_ascii=False))

    # Test experiment insights
    test_summary = {
        "activate_ai": {
            "users_targeted": 350,
            "incremental_mau_gained": 85.3,
            "total_spend_bdt": 8750,
            "cost_per_incremental_mau": 103,
            "by_lifecycle_stage": {
                "Payday Cash-Outer": {"count": 180, "mean_uplift": 0.28},
                "At-Risk Churner": {"count": 120, "mean_uplift": 0.15},
                "One-Hit Wonder": {"count": 50, "mean_uplift": 0.05},
            },
        },
        "mass_blast_baseline": {
            "users_targeted": 2000,
            "incremental_mau_gained": 45.2,
            "total_spend_bdt": 54000,
            "cost_per_incremental_mau": 1195,
        },
        "savings": {
            "budget_saved_bdt": 45250,
            "fatigued_users_protected": 280,
            "sure_thing_waste_avoided_bdt": 1050,
            "cost_efficiency_improvement_pct": 91,
            "sleeping_dogs_not_disturbed": 368,
        },
    }

    print("\n--- Experiment Insights ---")
    insights = generate_experiment_insights(test_summary)
    print(json.dumps(insights, indent=2, ensure_ascii=False))

    print(f"\n✅ generation_source (nudge): {nudge['generation_source']}")
    print(f"✅ generation_source (insights): {insights['generation_source']}")
    print("🎯 Done.")
