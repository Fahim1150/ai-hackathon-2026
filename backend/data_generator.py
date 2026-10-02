"""
Synthetic upay MFS Dormancy & Lifecycle Dataset Generator.

Generates 10,000 synthetic user records with planted causal uplift patterns
and saves data/train.csv (8,000 rows) and data/test.csv (2,000 rows).

100% standalone — zero imports from backend ML or API modules.
Assumptions documented in docs/LOGIC_CHAIN.md §3.
"""

import pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Constants (duplicated here intentionally — data_generator must be standalone)
# ---------------------------------------------------------------------------
SEED = 42
N_USERS = 10_000
TRAIN_FRAC = 0.8  # 8,000 train / 2,000 test

LIFECYCLE_STAGES = [
    "Sign-Up Drop-off",
    "Payday Cash-Outer",
    "One-Hit Wonder",
    "At-Risk Churner",
]
WALLET_TYPES = ["Primary", "Salary", "Remittance"]
AFFINITY_DOMAINS = ["Utility_Bill", "Mobile_Recharge", "Merchant_QR", "Toll_Transit"]
CHANNEL_TYPES = ["App", "USSD_268"]

OFFER_MAP = {
    "Utility_Bill":    ("Zero-Fee Bill Pay + 30 BDT Cash Reward", 30),
    "Mobile_Recharge": ("20 BDT Recharge Bonus", 20),
    "Merchant_QR":     ("10% Super Shop QR Cash Reward", 15),
    "Toll_Transit":    ("Zero-Fee UCB ATM + Toll Pass Offer", 50),
}

STAGE_WEIGHTS = [0.20, 0.25, 0.20, 0.35]  # stage distribution


# ---------------------------------------------------------------------------
# Helper: generate features conditioned on lifecycle stage
# ---------------------------------------------------------------------------
def _generate_stage_features(rng: np.random.Generator, stage: str, n: int) -> dict:
    """Return a dict of arrays for stage-specific feature distributions."""

    if stage == "Sign-Up Drop-off":
        days_inactive = rng.integers(30, 181, size=n)
        historical_tx = np.zeros(n, dtype=int)
        cashout_ratio = rng.beta(2, 8, size=n)  # low — they never transacted
        monthly_inflow = rng.lognormal(mean=8.5, sigma=0.8, size=n).clip(500, 80_000)

    elif stage == "Payday Cash-Outer":
        days_inactive = rng.integers(1, 15, size=n)
        historical_tx = rng.integers(5, 31, size=n)
        cashout_ratio = rng.uniform(0.91, 1.0, size=n)  # >90% cash-out
        monthly_inflow = rng.lognormal(mean=9.8, sigma=0.5, size=n).clip(8_000, 80_000)

    elif stage == "One-Hit Wonder":
        days_inactive = rng.integers(14, 91, size=n)
        historical_tx = np.ones(n, dtype=int)
        cashout_ratio = rng.beta(3, 4, size=n)
        monthly_inflow = rng.lognormal(mean=8.8, sigma=0.7, size=n).clip(500, 40_000)

    else:  # At-Risk Churner
        days_inactive = rng.integers(21, 121, size=n)
        historical_tx = rng.integers(2, 51, size=n)
        cashout_ratio = rng.beta(4, 5, size=n)
        monthly_inflow = rng.lognormal(mean=9.2, sigma=0.6, size=n).clip(1_000, 60_000)

    return {
        "days_inactive": days_inactive,
        "historical_tx_count": historical_tx,
        "cashout_ratio": np.round(cashout_ratio, 3),
        "monthly_inflow_bdt": np.round(monthly_inflow, 2),
    }


# ---------------------------------------------------------------------------
# Helper: assign wallet type with stage-aware probabilities
# ---------------------------------------------------------------------------
def _assign_wallet(rng: np.random.Generator, stage: str, n: int) -> np.ndarray:
    if stage == "Payday Cash-Outer":
        # Cash-outers are predominantly salary wallets
        probs = [0.15, 0.70, 0.15]
    elif stage == "Sign-Up Drop-off":
        probs = [0.60, 0.20, 0.20]
    else:
        probs = [0.50, 0.25, 0.25]
    return rng.choice(WALLET_TYPES, size=n, p=probs)


# ---------------------------------------------------------------------------
# Helper: assign affinity domain with wallet-aware bias
# ---------------------------------------------------------------------------
def _assign_affinity(rng: np.random.Generator, wallets: np.ndarray) -> np.ndarray:
    result = []
    for w in wallets:
        if w == "Salary":
            probs = [0.40, 0.20, 0.25, 0.15]
        elif w == "Remittance":
            probs = [0.25, 0.35, 0.15, 0.25]
        else:
            probs = [0.25, 0.30, 0.25, 0.20]
        result.append(rng.choice(AFFINITY_DOMAINS, p=probs))
    return np.array(result)


# ---------------------------------------------------------------------------
# Helper: assign channel type
# ---------------------------------------------------------------------------
def _assign_channel(rng: np.random.Generator, wallets: np.ndarray) -> np.ndarray:
    result = []
    for w in wallets:
        p_app = 0.55 if w == "Remittance" else 0.75
        result.append(rng.choice(CHANNEL_TYPES, p=[p_app, 1 - p_app]))
    return np.array(result)


# ---------------------------------------------------------------------------
# Helper: generate promo fatigue features
# ---------------------------------------------------------------------------
def _generate_promo_features(rng: np.random.Generator, n: int) -> tuple:
    """Return (promos_sent_30d, promo_ignore_streak)."""
    # ~15% of users are heavily fatigued
    is_fatigued = rng.random(n) < 0.15
    promos = np.where(
        is_fatigued,
        rng.integers(8, 16, size=n),
        rng.poisson(lam=2.0, size=n).clip(0, 7),
    )
    # Ignore streak correlates with promos sent
    ignore = np.where(
        is_fatigued,
        rng.integers(5, 13, size=n),
        rng.poisson(lam=0.8, size=n).clip(0, 4),
    )
    return promos, ignore


# ---------------------------------------------------------------------------
# Core: compute activation probability with planted causal patterns
# ---------------------------------------------------------------------------
def _activation_probability(
    treatment: np.ndarray,
    days_inactive: np.ndarray,
    historical_tx: np.ndarray,
    cashout_ratio: np.ndarray,
    monthly_inflow: np.ndarray,
    promos_sent: np.ndarray,
    promo_ignore: np.ndarray,
    affinity: np.ndarray,
    offer_name: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Planted ground-truth causal activation probabilities.

    Four uplift quadrants:
    - Persuadables: moderate inactivity + some history + treated → high activation
    - Sure Things: low inactivity + high history → high activation regardless
    - Lost Causes: deeply inactive + no history → very low regardless
    - Sleeping Dogs: heavily fatigued → treatment HURTS activation
    """
    n = len(treatment)
    base_prob = np.full(n, 0.05)  # low default

    # --- Baseline (control) probability ---
    # More history and less inactivity → higher baseline
    base_prob += np.clip(historical_tx / 80, 0, 0.15)
    base_prob -= np.clip(days_inactive / 400, 0, 0.10)
    base_prob += np.clip(monthly_inflow / 200_000, 0, 0.08)
    base_prob -= cashout_ratio * 0.05

    # Sure Things: very recently active users with strong transaction history
    is_sure_thing = (days_inactive < 7) & (historical_tx > 10)
    base_prob = np.where(is_sure_thing, rng.uniform(0.35, 0.50, n), base_prob)

    # Lost Causes: deeply inactive, near-zero history
    is_lost_cause = (days_inactive > 90) & (historical_tx <= 1)
    base_prob = np.where(is_lost_cause, rng.uniform(0.01, 0.05, n), base_prob)

    # --- Treatment effect (uplift) ---
    uplift = np.zeros(n)

    # Persuadables: moderate inactivity, some history, gets TREATED
    is_persuadable_candidate = (
        (days_inactive >= 7) & (days_inactive <= 45)
        & (historical_tx > 2)
        & (~is_sure_thing)
        & (~is_lost_cause)
    )

    # Affinity match bonus — matched offer gives stronger lift
    affinity_matched = np.array([
        OFFER_MAP.get(a, ("", 0))[0] == o
        for a, o in zip(affinity, offer_name)
    ])
    matched_lift = np.where(affinity_matched, 0.27, 0.10)
    uplift = np.where(is_persuadable_candidate, matched_lift, uplift)

    # Sure Things: near-zero uplift (they activate anyway)
    uplift = np.where(is_sure_thing, rng.uniform(-0.02, 0.02, n), uplift)

    # Lost Causes: near-zero uplift (nothing works)
    uplift = np.where(is_lost_cause, rng.uniform(-0.01, 0.01, n), uplift)

    # Sleeping Dogs: fatigue causes NEGATIVE uplift
    is_sleeping_dog = (promos_sent > 7) & (promo_ignore > 5)
    fatigue_penalty = -(0.03 + promo_ignore * 0.005)
    uplift = np.where(is_sleeping_dog, fatigue_penalty, uplift)

    # --- Noise ---
    noise = rng.normal(0, 0.02, n)

    # --- Final activation probability ---
    # Control group: base only; Treated group: base + uplift
    p = base_prob + treatment * uplift + noise
    p = np.clip(p, 0.001, 0.999)
    return p


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------
def generate(n: int = N_USERS, seed: int = SEED) -> pd.DataFrame:
    """Generate the full synthetic dataset."""
    rng = np.random.default_rng(seed)

    # --- Lifecycle stage assignment ---
    stages = rng.choice(LIFECYCLE_STAGES, size=n, p=STAGE_WEIGHTS)

    # --- Build feature arrays per stage ---
    customer_ids = [f"UPAY_{100_000 + i}" for i in range(n)]
    all_features = {
        "days_inactive": np.empty(n, dtype=int),
        "historical_tx_count": np.empty(n, dtype=int),
        "cashout_ratio": np.empty(n, dtype=float),
        "monthly_inflow_bdt": np.empty(n, dtype=float),
    }

    for stage in LIFECYCLE_STAGES:
        mask = stages == stage
        count = mask.sum()
        if count == 0:
            continue
        feats = _generate_stage_features(rng, stage, count)
        for key in all_features:
            all_features[key][mask] = feats[key]

    # --- Wallet, affinity, channel ---
    wallets = np.empty(n, dtype=object)
    for stage in LIFECYCLE_STAGES:
        mask = stages == stage
        wallets[mask] = _assign_wallet(rng, stage, mask.sum())

    affinity = _assign_affinity(rng, wallets)
    channel = _assign_channel(rng, wallets)

    # --- Promo fatigue ---
    promos_sent, promo_ignore = _generate_promo_features(rng, n)

    # --- Offer assignment (based on affinity) ---
    offer_names = np.array([OFFER_MAP[a][0] for a in affinity])
    offer_costs = np.array([OFFER_MAP[a][1] for a in affinity], dtype=float)

    # --- Treatment assignment (randomized 50/50 simulated RCT) ---
    treatment = rng.integers(0, 2, size=n)

    # --- Activation outcome ---
    p_activate = _activation_probability(
        treatment=treatment,
        days_inactive=all_features["days_inactive"],
        historical_tx=all_features["historical_tx_count"],
        cashout_ratio=all_features["cashout_ratio"],
        monthly_inflow=all_features["monthly_inflow_bdt"],
        promos_sent=promos_sent,
        promo_ignore=promo_ignore,
        affinity=affinity,
        offer_name=offer_names,
        rng=rng,
    )
    activated = rng.binomial(1, p_activate)

    # --- Assemble DataFrame ---
    df = pd.DataFrame({
        "customer_id": customer_ids,
        "lifecycle_stage": stages,
        "wallet_type": wallets,
        "days_inactive": all_features["days_inactive"],
        "historical_tx_count": all_features["historical_tx_count"],
        "monthly_inflow_bdt": all_features["monthly_inflow_bdt"],
        "cashout_ratio": all_features["cashout_ratio"],
        "top_affinity_domain": affinity,
        "promos_sent_30d": promos_sent,
        "promo_ignore_streak": promo_ignore,
        "channel_type": channel,
        "recommended_solo_offer": offer_names,
        "offer_cost_bdt": offer_costs,
        "treatment": treatment,
        "activated_30d": activated,
    })
    return df


# ---------------------------------------------------------------------------
# Train / test split and save
# ---------------------------------------------------------------------------
def save_splits(df: pd.DataFrame, data_dir: str = "data") -> None:
    """Stratified train/test split and save to CSV."""
    out = pathlib.Path(data_dir)
    out.mkdir(exist_ok=True)

    # Stratify by treatment × lifecycle_stage for balanced splits
    strat_col = df["treatment"].astype(str) + "_" + df["lifecycle_stage"]

    train, test = train_test_split(
        df,
        test_size=1 - TRAIN_FRAC,
        random_state=SEED,
        stratify=strat_col,
    )

    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)

    print(f"✅ Saved {len(train)} rows → {out / 'train.csv'}")
    print(f"✅ Saved {len(test)} rows  → {out / 'test.csv'}")

    # --- Quick sanity checks ---
    overlap = set(train["customer_id"]) & set(test["customer_id"])
    assert len(overlap) == 0, f"❌ Data leakage! {len(overlap)} shared customer_ids"
    print("✅ Zero customer_id overlap (no data leakage)")

    assert len(train) + len(test) == len(df), "❌ Row count mismatch"
    print(f"✅ Total: {len(train)} + {len(test)} = {len(df)}")

    # Treatment balance
    for name, split in [("train", train), ("test", test)]:
        t_rate = split["treatment"].mean()
        print(f"   {name}: treatment rate = {t_rate:.3f}")

    # Lifecycle stage distribution
    print("\n📊 Lifecycle stage distribution:")
    print(df["lifecycle_stage"].value_counts().to_string())

    # Activation rates by treatment (planted pattern check)
    print("\n📊 Activation rate by treatment:")
    print(df.groupby("treatment")["activated_30d"].mean().to_string())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("🔧 Generating 10,000 synthetic upay user records...")
    df = generate()
    save_splits(df)
    print("\n🎯 Done. Ready for Phase 3 (ML Engine).")
