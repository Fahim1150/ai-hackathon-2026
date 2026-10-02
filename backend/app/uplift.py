"""Multi-arm T-learner uplift engine: one XGBoost model per arm (control + offers)."""
import numpy as np, pandas as pd, xgboost as xgb
from .config import OFFERS, MARGIN, FEATURES

COSTS = np.array([OFFERS[k]["cost"] for k in sorted(OFFERS)])

class UpliftEngine:
    def __init__(self):
        self.models = {}

    def fit(self, X, arm, y):
        for k in OFFERS:
            m = xgb.XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.08,
                                  subsample=0.9, random_state=42, eval_metric="logloss")
            m.fit(X[arm == k], y[arm == k])
            self.models[k] = m
        return self

    def arm_probs(self, X):
        """P(conversion | arm k, x) for every arm -> shape (n, 4)."""
        return np.column_stack([self.models[k].predict_proba(X[FEATURES])[:, 1] for k in sorted(OFFERS)])

    def score(self, X):
        P = self.arm_probs(X)
        p0 = P[:, 0]
        uplift = P[:, 1:] - p0[:, None]                                   # (n, 3)
        gain = P[:, 1:] * (MARGIN - COSTS[1:]) - (p0 * MARGIN)[:, None]   # expected profit gain vs no offer
        best = gain.argmax(1)
        idx = np.arange(len(X))
        return pd.DataFrame({
            "baseline_prob": p0,
            "best_offer": best + 1,
            "best_uplift": uplift[idx, best],
            "expected_gain": gain[idx, best],
            "expected_cost": P[idx, best + 1] * COSTS[best + 1],
        }, index=X.index)

    def explain(self, row: pd.DataFrame, offer: int, top=3):
        """Approximate drivers: SHAP contributions of offer model minus control model (log-odds)."""
        d = xgb.DMatrix(row[FEATURES])
        ct = self.models[offer].get_booster().predict(d, pred_contribs=True)[0][:-1]
        cc = self.models[0].get_booster().predict(d, pred_contribs=True)[0][:-1]
        diff = ct - cc
        order = np.argsort(-np.abs(diff))[:top]
        return [{"feature": FEATURES[i], "value": float(row[FEATURES[i]].iloc[0]),
                 "effect": "raises uplift" if diff[i] > 0 else "lowers uplift",
                 "magnitude": round(float(abs(diff[i])), 3)} for i in order]
