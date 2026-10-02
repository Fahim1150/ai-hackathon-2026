"""Shared constants. Business assumptions live here so they are visible and documented."""
OFFERS = {
    0: {"name": "No offer", "cost": 0.0},
    1: {"name": "10% cashback on utility bill", "cost": 30.0},
    2: {"name": "Free data bundle", "cost": 15.0},
    3: {"name": "Cash-out fee waiver", "cost": 20.0},
}
MARGIN = 100.0            # BDT value of one incremental transaction (assumption)
FEATURES = [
    "monthly_trans_count", "avg_trans_amount", "days_since_last_trans",
    "tenure_months", "cashout_ratio", "offers_last_30d",
]
GROUP_COLS = ["region", "age_band"]   # NOT model features; used only for fairness checks
PROPENSITY = 1 / len(OFFERS)          # randomized experiment: equal chance per arm

# Business rules (deterministic, kept separate from the ML model)
FATIGUE_CAP = 3          # no offer if user got >= 3 offers in last 30 days
MIN_GAIN_BDT = 1.0       # require at least this expected profit gain per user
BASE_HIGH = 0.40         # baseline conversion above this = "Sure Thing" if uplift small
UPLIFT_MIN = 0.03        # minimum meaningful uplift
