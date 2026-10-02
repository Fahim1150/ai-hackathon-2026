"""Synthetic randomized-campaign data (6 arms: control + 5 offers). 100 % synthetic.
~50 000 customers with KNOWN planted uplift effects so the model can be validated.
Assumptions documented in docs/SYNTHETIC_ASSUMPTIONS.md.  Run from repo root.

Hidden segments baked in:
  * Persuadables  – certain feature combos get +15-25 pp uplift
  * Sure things   – high baseline, near-zero uplift
  * Lost causes   – low baseline, near-zero uplift
  * Sleeping dogs – offer HURTS (negative uplift from fatigue / over-contact)
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from backend.app.config import OFFERS, MARGIN

N_OFFERS = len(OFFERS)  # 6 (0=control, 1-5=offers)


def generate(n: int = 50_000, seed: int = 42,
             csv_path: str = "data/synthetic_customers.csv",
             parquet_path: str = "data/synthetic_customers.parquet"):
    rng = np.random.default_rng(seed)

    # ── demographics ─────────────────────────────────────────────────────
    region = rng.choice(
        ["Dhaka", "Chittagong", "Rajshahi", "Sylhet", "Rural"],
        n, p=[0.30, 0.20, 0.15, 0.10, 0.25],
    )
    age_band = rng.choice(
        ["18-25", "26-35", "36-50", "51+"],
        n, p=[0.25, 0.35, 0.25, 0.15],
    )
    is_rural = (region == "Rural").astype(float)
    is_young = np.isin(age_band, ["18-25"]).astype(float)
    is_senior = np.isin(age_band, ["51+"]).astype(float)

    # ── behavioural features ─────────────────────────────────────────────
    tenure_months = rng.integers(1, 60, n)
    monthly_txn_count = np.clip(
        rng.poisson(np.where(is_rural, 8, 14), n), 0, 80
    )
    avg_txn_amount = np.clip(rng.exponential(400, n) + 50, 10, 10_000).round(2)
    cashout_share = np.clip(
        rng.beta(2, 3, n) + is_rural * 0.15, 0, 1
    ).round(3)
    bill_pay_history = rng.poisson(np.where(is_young, 2, 4), n).clip(0, 10)
    last_active_days = np.clip(
        rng.exponential(18, n).astype(int) + 1, 1, 90
    )
    offers_last_30d = rng.poisson(1.3, n)

    # ── baseline conversion probability (control arm) ────────────────────
    logit_base = (
        -1.5
        + 0.06 * monthly_txn_count
        - 0.03 * last_active_days
        + 0.012 * tenure_months
        + 0.05 * bill_pay_history
        - 0.3 * is_rural
    )
    p0 = 1 / (1 + np.exp(-logit_base))

    # ── planted uplift effects per offer ─────────────────────────────────
    noise = rng.normal(0, 0.025, (n, 5))
    fatigue = -0.04 * offers_last_30d - 0.08 * (offers_last_30d >= 4)

    lift = np.zeros((n, N_OFFERS))  # col 0 = control (always 0)

    # Offer 1: 10 % cashback on utility bill
    #   Persuadable when lapsing-but-formerly-active (last_active > 14 & txn > 5)
    lift[:, 1] = (
        np.where((last_active_days > 14) & (monthly_txn_count > 5), 0.25, 0.03)
        + noise[:, 0] + fatigue
    )

    # Offer 2: Free data bundle
    #   Persuadable for young users with low avg amount
    lift[:, 2] = (
        np.where((is_young == 1) & (avg_txn_amount < 400), 0.22, 0.02)
        + noise[:, 1] + fatigue
    )

    # Offer 3: Cash-out fee waiver
    #   Persuadable when cashout_share > 0.6
    lift[:, 3] = (
        np.where(cashout_share > 0.6, 0.24, 0.01)
        + noise[:, 2] + fatigue
    )

    # Offer 4: Referral bonus
    #   Persuadable for long-tenure users with moderate activity
    lift[:, 4] = (
        np.where((tenure_months > 24) & (monthly_txn_count >= 8), 0.20, 0.02)
        + noise[:, 3] + fatigue
    )

    # Offer 5: P2P fee waiver
    #   Persuadable for high-txn-count users with bill-pay history
    lift[:, 5] = (
        np.where((monthly_txn_count > 12) & (bill_pay_history >= 3), 0.18, 0.015)
        + noise[:, 4] + fatigue
    )

    # ── random treatment assignment (RCT) ────────────────────────────────
    arm = rng.integers(0, N_OFFERS, n)
    p_treated = np.clip(p0 + lift[np.arange(n), arm], 0.001, 0.999)
    converted = rng.binomial(1, p_treated)

    cost = np.array([OFFERS[a]["cost"] for a in arm])
    profit = converted * MARGIN - (arm > 0) * converted * cost

    # ── assemble dataframe ───────────────────────────────────────────────
    df = pd.DataFrame({
        "user_id":            [f"SYN_{100_000 + i}" for i in range(n)],
        "region":             region,
        "age_band":           age_band,
        "tenure_months":      tenure_months,
        "monthly_txn_count":  monthly_txn_count,
        "avg_txn_amount":     avg_txn_amount,
        "cashout_share":      cashout_share,
        "bill_pay_history":   bill_pay_history,
        "last_active_days":   last_active_days,
        "offers_last_30d":    offers_last_30d,
        "arm":                arm,
        "converted":          converted,
        "profit":             profit,
    })

    pathlib.Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    # ── summary ──────────────────────────────────────────────────────────
    print(f"Wrote {csv_path} and {parquet_path}: {len(df):,} rows")
    print(f"Arms distribution: {np.bincount(arm).tolist()}")
    print(f"Overall conversion rate: {converted.mean():.3f}")
    for k in range(N_OFFERS):
        mask = arm == k
        print(f"  Arm {k} ({OFFERS[k]['name']}): n={mask.sum()}, "
              f"conv={converted[mask].mean():.3f}")


if __name__ == "__main__":
    generate()
