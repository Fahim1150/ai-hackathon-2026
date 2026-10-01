"""Generic synthetic MFS data generator. Clearly synthetic, seeded, with injected patterns.
Document every assumption in docs/SYNTHETIC_ASSUMPTIONS.md."""
import numpy as np, pandas as pd

rng = np.random.default_rng(42)
N_USERS, N_TX = 2000, 60000

users = pd.DataFrame({
    "user_id": range(N_USERS),
    "segment": rng.choice(["low", "mid", "high"], N_USERS, p=[.5, .35, .15]),
    "is_agent": rng.random(N_USERS) < 0.03,
})
tx = pd.DataFrame({
    "tx_id": range(N_TX),
    "user_id": rng.integers(0, N_USERS, N_TX),
    "type": rng.choice(["send", "cashout", "payment", "topup"], N_TX, p=[.35, .25, .25, .15]),
    "amount": np.round(rng.lognormal(6.5, 1.0, N_TX), 2),
    "ts": pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, 90 * 86400, N_TX), unit="s"),
})
tx["hour"] = tx.ts.dt.hour
tx["month_end"] = tx.ts.dt.day >= 25

# Injected known pattern (ground-truth label for testing): rare very large amounts
tx["is_anomaly"] = False
idx = rng.choice(N_TX, int(N_TX * 0.01), replace=False)
tx.loc[idx, "amount"] = tx.loc[idx, "amount"] * 8
tx.loc[idx, "is_anomaly"] = True

users.to_csv("data/users.csv", index=False)
tx.to_csv("data/transactions.csv", index=False)
print("done:", len(users), "users,", len(tx), "transactions")
