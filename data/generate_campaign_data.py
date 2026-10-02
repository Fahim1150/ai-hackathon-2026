"""Synthetic randomized-campaign data (4 arms: control + 3 offers). 100% synthetic.
Assumptions documented in docs/SYNTHETIC_ASSUMPTIONS.md. Run from repo root."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from backend.app.config import OFFERS, MARGIN

def generate(n=30000, seed=42, path="data/campaign_data.csv"):
    rng = np.random.default_rng(seed)
    region = rng.choice(["urban", "rural"], n, p=[0.6, 0.4])
    age_band = rng.choice(["18-25", "26-40", "41+"], n, p=[0.3, 0.45, 0.25])
    rural = region == "rural"
    count = rng.poisson(np.where(rural, 9, 14))
    avg_amt = rng.exponential(450, n) + 50
    days = np.clip(rng.exponential(18, n).astype(int) + 1, 1, 60)
    tenure = rng.integers(1, 48, n)
    cashout = np.clip(rng.beta(2, 3, n) + np.where(rural, 0.15, 0), 0, 1)
    offers30 = rng.poisson(1.3, n)

    p0 = 1 / (1 + np.exp(-(-1.4 + 0.06 * count - 0.04 * days + 0.01 * tenure)))
    noise = rng.normal(0, 0.03, (n, 3))
    fatigue = -0.04 * offers30 - 0.06 * (offers30 >= 4)   # fatigued users can react negatively
    lift = np.zeros((n, 4))
    lift[:, 1] = np.where((days > 14) & (count > 5), 0.25, 0.03) + noise[:, 0] + fatigue
    lift[:, 2] = np.where((age_band == "18-25") & (avg_amt < 400), 0.20, 0.02) + noise[:, 1] + fatigue
    lift[:, 3] = np.where(cashout > 0.6, 0.22, 0.01) + noise[:, 2] + fatigue

    arm = rng.integers(0, 4, n)                           # random assignment (RCT)
    p = np.clip(p0 + lift[np.arange(n), arm], 0.001, 0.999)
    conv = rng.binomial(1, p)
    cost = np.array([OFFERS[a]["cost"] for a in arm])
    profit = conv * MARGIN - (arm > 0) * conv * cost      # offer cost paid on redemption

    df = pd.DataFrame({
        "user_id": [f"SYN_{100000 + i}" for i in range(n)],
        "region": region, "age_band": age_band,
        "monthly_trans_count": count, "avg_trans_amount": avg_amt.round(2),
        "days_since_last_trans": days, "tenure_months": tenure,
        "cashout_ratio": cashout.round(3), "offers_last_30d": offers30,
        "arm": arm, "converted": conv, "profit": profit,
    })
    df.to_csv(path, index=False)
    print(f"wrote {path}: {len(df)} rows, arms={np.bincount(arm).tolist()}")

if __name__ == "__main__":
    generate()
