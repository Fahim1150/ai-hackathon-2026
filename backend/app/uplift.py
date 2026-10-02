"""Multi-arm T-learner uplift engine: one LightGBM model per arm (control + offers).
Also includes a plain ResponseModel baseline for comparison."""
import numpy as np, pandas as pd, lightgbm as lgb
from .config import OFFERS, MARGIN, FEATURES

COSTS = np.array([OFFERS[k]["cost"] for k in sorted(OFFERS)])

# ── plain-language templates for SHAP explanations ──────────────────────
_TEMPLATES = {
    "monthly_txn_count": {
        "high": "High transaction frequency ({val}/month) suggests active engagement",
        "low": "Low transaction count ({val}/month) limits offer impact",
    },
    "avg_txn_amount": {
        "high": "Higher average transaction amount (৳{val}) indicates valuable user",
        "low": "Low average transaction (৳{val}) may limit campaign ROI",
    },
    "last_active_days": {
        "high": "Inactive for {val} days — a well-timed offer could re-engage",
        "low": "Recently active ({val} days ago) — less urgency for re-engagement",
    },
    "tenure_months": {
        "high": "Long tenure ({val} months) shows loyalty and responsiveness",
        "low": "Short tenure ({val} months) — still building engagement",
    },
    "cashout_share": {
        "high": "High cash-out share ({val:.0%}) — responds well to fee waivers",
        "low": "Low cash-out share ({val:.0%}) — fee waiver offers less relevant",
    },
    "bill_pay_history": {
        "high": "Active bill payer ({val} bills) — utility cashback may appeal",
        "low": "Minimal bill pay history ({val}) — utility offers less compelling",
    },
    "offers_last_30d": {
        "high": "Already received {val} offers recently — fatigue risk",
        "low": "Fresh audience ({val} recent offers) — good time to engage",
    },
}


def _plain_reason(feature: str, value: float, effect: str) -> str:
    """Turn a SHAP contribution into a human-readable sentence."""
    tpl = _TEMPLATES.get(feature)
    if tpl is None:
        return f"{feature}={value} {effect}"
    bucket = "high" if effect == "raises uplift" else "low"
    try:
        return tpl[bucket].format(val=value)
    except (KeyError, ValueError):
        return f"{feature}={value} {effect}"


class UpliftEngine:
    """T-learner: one LightGBM classifier per treatment arm."""

    def __init__(self):
        self.models = {}

    def fit(self, X, arm, y):
        for k in OFFERS:
            mask = arm == k
            if mask.sum() < 10:
                continue
            m = lgb.LGBMClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.85, colsample_bytree=0.9,
                min_child_samples=20, random_state=42,
                verbose=-1,
            )
            m.fit(X.loc[mask, FEATURES], y[mask])
            self.models[k] = m
        return self

    def arm_probs(self, X):
        """P(conversion | arm k, x) for every arm -> shape (n, n_arms)."""
        return np.column_stack([
            self.models[k].predict_proba(X[FEATURES])[:, 1]
            for k in sorted(self.models)
        ])

    def score(self, X):
        """Score every user: best offer, uplift, expected gain, expected cost."""
        P = self.arm_probs(X)
        n_offers = P.shape[1] - 1  # exclude control
        p0 = P[:, 0]
        uplift = P[:, 1:] - p0[:, None]
        gain = P[:, 1:] * (MARGIN - COSTS[1:n_offers+1]) - (p0 * MARGIN)[:, None]
        best = gain.argmax(axis=1)
        idx = np.arange(len(X))
        return pd.DataFrame({
            "baseline_prob": p0,
            "best_offer": best + 1,
            "best_uplift": uplift[idx, best],
            "expected_gain": gain[idx, best],
            "expected_cost": P[idx, best + 1] * COSTS[best + 1],
        }, index=X.index)

    def score_all_offers(self, X):
        """Return uplift, gain, cost for EVERY offer (not just best)."""
        P = self.arm_probs(X)
        n_offers = P.shape[1] - 1
        p0 = P[:, 0]
        rows = []
        for k in range(1, n_offers + 1):
            rows.append({
                "offer_id": k,
                "offer_name": OFFERS[k]["name"],
                "prob_treated": float(P[0, k]),
                "prob_control": float(p0[0]),
                "uplift": float(P[0, k] - p0[0]),
                "expected_gain": float(P[0, k] * (MARGIN - COSTS[k]) - p0[0] * MARGIN),
                "cost": float(COSTS[k]),
            })
        return sorted(rows, key=lambda r: r["uplift"], reverse=True)

    def explain(self, row: pd.DataFrame, offer: int, top: int = 3):
        """SHAP contributions: offer model minus control model -> plain language."""
        X_feat = row[FEATURES]

        # LightGBM SHAP values (leaf-based, fast)
        ct = self.models[offer].predict_proba(X_feat, pred_contrib=True)[0]
        cc = self.models[0].predict_proba(X_feat, pred_contrib=True)[0]

        # pred_contrib returns [n_features + 1] per class for binary;
        # for LightGBM binary classifier it returns shape (n_classes, n_features+1)
        # We want the positive-class contributions
        if ct.ndim == 2:
            ct = ct[1, :-1]
            cc = cc[1, :-1]
        else:
            ct = ct[:-1]
            cc = cc[:-1]

        diff = ct - cc
        order = np.argsort(-np.abs(diff))[:top]
        results = []
        for i in order:
            feat = FEATURES[i]
            val = float(row[feat].iloc[0])
            effect = "raises uplift" if diff[i] > 0 else "lowers uplift"
            results.append({
                "feature": feat,
                "value": val,
                "effect": effect,
                "magnitude": round(float(abs(diff[i])), 4),
                "reason": _plain_reason(feat, val, effect),
            })
        return results


class ResponseModel:
    """Plain response model baseline: ignores treatment, just predicts P(convert)."""

    def __init__(self):
        self.model = None

    def fit(self, X, y):
        self.model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.85, random_state=42, verbose=-1,
        )
        self.model.fit(X[FEATURES], y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X[FEATURES])[:, 1]
