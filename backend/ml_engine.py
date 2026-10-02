"""
upay ActivateAI — ML Engine (T-Learner Uplift + SHAP + Fatigue + Fairness).

Trains two LightGBM models (treated / control), computes uplift scores,
SHAP explanations, fatigue scores, and fairness audits on the held-out test set.

All deterministic business logic (quadrant thresholds, fatigue formula, offer
mapping) is kept separate from the ML model per AGENTS.md guardrail #2.
"""

import json
import pathlib
import warnings
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Constants — deterministic business rules (AGENTS.md guardrail #2)
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "days_inactive",
    "historical_tx_count",
    "monthly_inflow_bdt",
    "cashout_ratio",
    "promos_sent_30d",
    "promo_ignore_streak",
]

CATEGORICAL_COLS = [
    "lifecycle_stage",
    "wallet_type",
    "top_affinity_domain",
    "channel_type",
]

ALL_FEATURES = FEATURE_COLS + CATEGORICAL_COLS

OFFER_MAP = {
    "Utility_Bill":    ("Zero-Fee Bill Pay + 30 BDT Cash Reward", 30),
    "Mobile_Recharge": ("20 BDT Recharge Bonus", 20),
    "Merchant_QR":     ("10% Super Shop QR Cash Reward", 15),
    "Toll_Transit":    ("Zero-Fee UCB ATM + Toll Pass Offer", 50),
}

# Uplift quadrant thresholds
PERSUADABLE_MIN_UPLIFT = 0.05
SLEEPING_DOG_MAX_UPLIFT = -0.02
SURE_THING_CONTROL_MIN = 0.25

# LightGBM hyperparameters
LGB_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_samples=20,
    random_state=42,
    verbose=-1,
    force_col_wise=True,
)

MODELS_DIR = pathlib.Path("models")


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categoricals for LightGBM (native category support)."""
    X = df[ALL_FEATURES].copy()
    for col in CATEGORICAL_COLS:
        X[col] = X[col].astype("category")
    return X


# ---------------------------------------------------------------------------
# Uplift quadrant classification (deterministic — no ML)
# ---------------------------------------------------------------------------
def classify_quadrant(
    uplift: float, prob_control: float
) -> str:
    """Classify a user into one of 4 uplift quadrants."""
    if uplift >= PERSUADABLE_MIN_UPLIFT:
        return "Persuadable"
    if uplift < SLEEPING_DOG_MAX_UPLIFT:
        return "Sleeping Dog"
    if prob_control >= SURE_THING_CONTROL_MIN:
        return "Sure Thing"
    return "Lost Cause"


# ---------------------------------------------------------------------------
# Fatigue score (deterministic — no ML)
# ---------------------------------------------------------------------------
def compute_fatigue_score(promos_sent_30d: int, promo_ignore_streak: int) -> int:
    """Offer fatigue score 0–100. Higher = more fatigued."""
    return int(min(100, promos_sent_30d * 8 + promo_ignore_streak * 6))


# ---------------------------------------------------------------------------
# Offer assignment (deterministic — no ML)
# ---------------------------------------------------------------------------
def assign_offer(affinity_domain: str) -> tuple[str, float]:
    """Map top_affinity_domain → (offer_name, cost_bdt)."""
    return OFFER_MAP.get(affinity_domain, ("Unknown Offer", 0.0))


# ---------------------------------------------------------------------------
# ActivateAIEngine
# ---------------------------------------------------------------------------
class ActivateAIEngine:
    """T-Learner uplift engine with SHAP explainability."""

    def __init__(self):
        self.model_t1: lgb.LGBMClassifier | None = None  # treated
        self.model_t0: lgb.LGBMClassifier | None = None  # control

    # ----- Training -----
    def train(self, train_path: str = "data/train.csv") -> "ActivateAIEngine":
        """Train T-Learner: two LightGBM models split by treatment."""
        df = pd.read_csv(train_path)
        y = df["activated_30d"]
        t = df["treatment"]

        X = _prepare_features(df)

        # Model T1: trained on treated users only
        mask_t1 = t == 1
        self.model_t1 = lgb.LGBMClassifier(**LGB_PARAMS)
        self.model_t1.fit(X[mask_t1], y[mask_t1])

        # Model T0: trained on control users only
        mask_t0 = t == 0
        self.model_t0 = lgb.LGBMClassifier(**LGB_PARAMS)
        self.model_t0.fit(X[mask_t0], y[mask_t0])

        print(f"✅ Trained T1 on {mask_t1.sum()} treated rows")
        print(f"✅ Trained T0 on {mask_t0.sum()} control rows")
        return self

    # ----- Prediction -----
    def predict(self, test_path: str = "data/test.csv") -> pd.DataFrame:
        """Score test set: uplift, quadrant, fatigue, offer, SHAP top-3."""
        df = pd.read_csv(test_path)
        X = _prepare_features(df)

        # --- Uplift scores ---
        prob_t1 = self.model_t1.predict_proba(X)[:, 1]
        prob_t0 = self.model_t0.predict_proba(X)[:, 1]
        uplift = prob_t1 - prob_t0

        df["prob_active_treated"] = np.round(prob_t1, 4)
        df["prob_active_control"] = np.round(prob_t0, 4)
        df["uplift_score"] = np.round(uplift, 4)

        # --- Quadrant classification ---
        df["uplift_quadrant"] = [
            classify_quadrant(u, c)
            for u, c in zip(uplift, prob_t0)
        ]

        # --- Fatigue score ---
        df["offer_fatigue_score"] = [
            compute_fatigue_score(p, i)
            for p, i in zip(df["promos_sent_30d"], df["promo_ignore_streak"])
        ]

        # --- Best offer assignment ---
        offers = [assign_offer(a) for a in df["top_affinity_domain"]]
        df["assigned_offer"] = [o[0] for o in offers]
        df["assigned_offer_cost_bdt"] = [o[1] for o in offers]

        # --- SHAP top-3 ---
        print("⏳ Computing SHAP values for all test users...")
        shap_top3 = self._compute_shap_top3(X)
        df["shap_top3"] = shap_top3

        print(f"✅ Scored {len(df)} test users")
        return df

    # ----- SHAP -----
    def _compute_shap_top3(self, X: pd.DataFrame) -> list[str]:
        """Compute SHAP for the treated model and extract top 3 drivers per row."""
        explainer = shap.TreeExplainer(self.model_t1)
        shap_values = explainer.shap_values(X)

        # For binary classification, shap_values may be a list [class0, class1]
        if isinstance(shap_values, list):
            sv = shap_values[1]  # class 1 (activated)
        else:
            sv = shap_values

        feature_names = list(X.columns)
        results = []
        for i in range(len(X)):
            row_vals = sv[i]
            # Get indices of top 3 by absolute magnitude
            top_idx = np.argsort(-np.abs(row_vals))[:3]
            top3 = []
            for idx in top_idx:
                feat_name = feature_names[idx]
                feat_value = X.iloc[i, idx]
                shap_val = float(row_vals[idx])
                top3.append({
                    "feature": feat_name,
                    "value": feat_value if not isinstance(feat_value, (np.integer, np.floating)) else float(feat_value),
                    "shap_value": round(shap_val, 4),
                    "direction": "increases activation" if shap_val > 0 else "decreases activation",
                })
            results.append(json.dumps(top3))
        return results

    # ----- Fairness audit -----
    @staticmethod
    def fairness_audit(scored_df: pd.DataFrame) -> dict[str, Any]:
        """Audit contact rates and uplift across lifecycle_stage and wallet_type."""
        audit = {}
        for group_col in ["lifecycle_stage", "wallet_type"]:
            groups = {}
            for val, g in scored_df.groupby(group_col):
                persuadable_rate = (g["uplift_quadrant"] == "Persuadable").mean()
                groups[val] = {
                    "count": int(len(g)),
                    "mean_uplift": round(float(g["uplift_score"].mean()), 4),
                    "persuadable_rate": round(float(persuadable_rate), 4),
                    "mean_fatigue": round(float(g["offer_fatigue_score"].mean()), 1),
                }
            rates = [v["persuadable_rate"] for v in groups.values()]
            max_rate = max(rates) if max(rates) > 0 else 1e-9
            audit[group_col] = {
                "groups": groups,
                "min_max_persuadable_ratio": round(min(rates) / max_rate, 3),
            }
        return audit

    # ----- Save / Load -----
    def save(self, models_dir: str = "models") -> None:
        """Save trained models to disk."""
        out = pathlib.Path(models_dir)
        out.mkdir(exist_ok=True)
        joblib.dump(self.model_t1, out / "model_t1.joblib")
        joblib.dump(self.model_t0, out / "model_t0.joblib")
        print(f"✅ Models saved to {out}/")

    @classmethod
    def load(cls, models_dir: str = "models") -> "ActivateAIEngine":
        """Load trained models from disk."""
        d = pathlib.Path(models_dir)
        engine = cls()
        engine.model_t1 = joblib.load(d / "model_t1.joblib")
        engine.model_t0 = joblib.load(d / "model_t0.joblib")
        return engine


# ---------------------------------------------------------------------------
# Validation metrics
# ---------------------------------------------------------------------------
def compute_validation_metrics(scored_df: pd.DataFrame) -> dict:
    """Compute Uplift@K and AUUC on the test set."""
    df = scored_df.copy()
    df = df.sort_values("uplift_score", ascending=False).reset_index(drop=True)
    n = len(df)

    # Cumulative uplift curve
    cum_treated_activated = 0
    cum_treated_total = 0
    cum_control_activated = 0
    cum_control_total = 0
    uplift_curve = []

    for _, row in df.iterrows():
        if row["treatment"] == 1:
            cum_treated_total += 1
            cum_treated_activated += row["activated_30d"]
        else:
            cum_control_total += 1
            cum_control_activated += row["activated_30d"]

        if cum_treated_total > 0 and cum_control_total > 0:
            rate_t = cum_treated_activated / cum_treated_total
            rate_c = cum_control_activated / cum_control_total
            uplift_curve.append(rate_t - rate_c)
        else:
            uplift_curve.append(0.0)

    uplift_curve = np.array(uplift_curve)

    # AUUC: area under uplift curve (normalized)
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auuc = float(_trapz(uplift_curve, dx=1.0 / n))

    # Uplift@K: uplift in top K% of ranked users
    metrics = {"auuc": round(auuc, 4)}
    for k_pct in [10, 20, 30, 50]:
        k = int(n * k_pct / 100)
        top_k = df.head(k)
        t_mask = top_k["treatment"] == 1
        c_mask = top_k["treatment"] == 0
        if t_mask.sum() > 0 and c_mask.sum() > 0:
            rate_t = top_k.loc[t_mask, "activated_30d"].mean()
            rate_c = top_k.loc[c_mask, "activated_30d"].mean()
            metrics[f"uplift_at_{k_pct}pct"] = round(float(rate_t - rate_c), 4)
        else:
            metrics[f"uplift_at_{k_pct}pct"] = 0.0

    return metrics


# ---------------------------------------------------------------------------
# Overview stats for API
# ---------------------------------------------------------------------------
def build_overview_stats(scored_df: pd.DataFrame, fairness: dict, val_metrics: dict) -> dict:
    """Aggregate stats for GET /api/overview."""
    funnel = scored_df["lifecycle_stage"].value_counts().to_dict()
    quadrant_dist = scored_df["uplift_quadrant"].value_counts().to_dict()
    offer_dist = scored_df["assigned_offer"].value_counts().to_dict()

    return {
        "total_test_users": int(len(scored_df)),
        "funnel": funnel,
        "uplift_quadrant_distribution": quadrant_dist,
        "offer_distribution": offer_dist,
        "validation_metrics": val_metrics,
        "fairness": fairness,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("upay ActivateAI — ML Engine")
    print("=" * 60)

    # 1. Train
    engine = ActivateAIEngine()
    engine.train("data/train.csv")

    # 2. Score test set
    scored = engine.predict("data/test.csv")

    # 3. Save models
    engine.save("models")

    # 4. Save scored predictions
    MODELS_DIR.mkdir(exist_ok=True)
    scored.to_csv(MODELS_DIR / "test_predictions.csv", index=False)
    print(f"✅ Predictions saved to {MODELS_DIR / 'test_predictions.csv'}")

    # 5. Fairness audit
    fairness = ActivateAIEngine.fairness_audit(scored)
    with open(MODELS_DIR / "fairness_audit.json", "w") as f:
        json.dump(fairness, f, indent=2)
    print(f"✅ Fairness audit saved to {MODELS_DIR / 'fairness_audit.json'}")

    # 6. Validation metrics
    val_metrics = compute_validation_metrics(scored)

    # 7. Overview stats
    overview = build_overview_stats(scored, fairness, val_metrics)
    with open(MODELS_DIR / "overview_stats.json", "w") as f:
        json.dump(overview, f, indent=2)
    print(f"✅ Overview stats saved to {MODELS_DIR / 'overview_stats.json'}")

    # 8. Print summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION METRICS")
    print("=" * 60)
    for k, v in val_metrics.items():
        print(f"  {k}: {v}")

    print("\n📊 UPLIFT QUADRANT DISTRIBUTION")
    for q, c in scored["uplift_quadrant"].value_counts().items():
        pct = c / len(scored) * 100
        print(f"  {q}: {c} ({pct:.1f}%)")

    print("\n📊 MEAN UPLIFT BY QUADRANT")
    for q, g in scored.groupby("uplift_quadrant"):
        print(f"  {q}: mean_uplift={g['uplift_score'].mean():.4f}, n={len(g)}")

    print("\n📊 FAIRNESS AUDIT")
    for group_col, data in fairness.items():
        print(f"\n  {group_col} (min/max ratio: {data['min_max_persuadable_ratio']}):")
        for val, stats in data["groups"].items():
            print(f"    {val}: n={stats['count']}, mean_uplift={stats['mean_uplift']}, "
                  f"persuadable_rate={stats['persuadable_rate']}, mean_fatigue={stats['mean_fatigue']}")

    print("\n🎯 Done. Ready for Phase 4 (Optimizer).")


if __name__ == "__main__":
    main()
