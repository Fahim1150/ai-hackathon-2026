"""Train, evaluate on a held-out RCT test set, save artifacts. Run: python -m backend.app.train"""
import json, pathlib, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from .config import OFFERS, MARGIN, FEATURES, PROPENSITY, FATIGUE_CAP, MIN_GAIN_BDT, UPLIFT_MIN, GROUP_COLS
from .uplift import UpliftEngine, COSTS

OUT = pathlib.Path("models"); OUT.mkdir(exist_ok=True)

def policy_value(test, chosen):
    """Unbiased profit/user of a policy via inverse-propensity weighting (valid: random assignment)."""
    match = (test["arm"].values == chosen)
    return float((match * test["profit"].values / PROPENSITY).mean())

def qini_auc(score, treat, y):
    """Average gap (in incremental conversions) between the Qini curve and random targeting."""
    o = np.argsort(-score); t, yy = treat[o], y[o]
    nt, nc = np.cumsum(t), np.cumsum(1 - t)
    yt, yc = np.cumsum(yy * t), np.cumsum(yy * (1 - t))
    q = yt - yc * np.divide(nt, nc, out=np.zeros_like(nt, dtype=float), where=nc > 0)
    rand = np.linspace(0, q[-1], len(q))
    return float((q - rand).mean())

def main():
    df = pd.read_csv("data/campaign_data.csv")
    train, test = train_test_split(df, test_size=0.3, random_state=42, stratify=df["arm"])
    eng = UpliftEngine().fit(train[FEATURES], train["arm"].values, train["converted"].values)
    s = eng.score(test)
    P = eng.arm_probs(test)

    # ---- policies (offer id per user; 0 = no offer) ----
    allowed = np.array([(o < FATIGUE_CAP) and g >= MIN_GAIN_BDT and u >= UPLIFT_MIN for o, g, u in
                        zip(test["offers_last_30d"], s["expected_gain"], s["best_uplift"])])
    pol = {
        "No campaign": np.zeros(len(test), int),
        "Blast offer 1 to everyone": np.ones(len(test), int),
        "Blast offer 2 to everyone": np.full(len(test), 2),
        "Blast offer 3 to everyone": np.full(len(test), 3),
        "Random offer": np.random.default_rng(0).integers(0, 4, len(test)),
        "Response model (ignores baseline)": P[:, 1:].argmax(1) + 1,
        "Uplift model": np.where(s["expected_gain"] > 0, s["best_offer"], 0),
        "Uplift model + business rules": np.where(allowed, s["best_offer"], 0),
    }
    res = {}
    for name, ch in pol.items():
        res[name] = {"profit_per_user_bdt": round(policy_value(test, ch), 2),
                     "contact_rate": round(float((ch > 0).mean()), 3)}
    base = res["No campaign"]["profit_per_user_bdt"]
    for r in res.values():
        r["incremental_vs_no_campaign"] = round(r["profit_per_user_bdt"] - base, 2)

    # ---- Qini per offer (control vs that arm) vs random score ----
    qini = {}
    rng = np.random.default_rng(1)
    for k in (1, 2, 3):
        m = test["arm"].isin([0, k]).values
        t = (test["arm"].values[m] == k).astype(int); y = test["converted"].values[m]
        qini[OFFERS[k]["name"]] = {"model": round(qini_auc((P[m, k] - P[m, 0]), t, y), 3),
                                   "random": round(qini_auc(rng.random(m.sum()), t, y), 3)}

    # ---- fairness: do groups get contacted / benefit differently? ----
    fair = {}
    tmp = test.assign(chosen=pol["Uplift model + business rules"])
    for g in GROUP_COLS:
        rows = {}
        for val, d in tmp.groupby(g):
            rows[val] = {"contact_rate": round(float((d["chosen"] > 0).mean()), 3),
                         "profit_per_user_bdt": round(policy_value(d, d["chosen"].values), 2),
                         "n": int(len(d))}
        rates = [v["contact_rate"] for v in rows.values()]
        fair[g] = {"groups": rows, "contact_rate_ratio_min_over_max": round(min(rates) / max(max(rates), 1e-9), 3)}

    metrics = {"policies": res, "qini_auc": qini, "fairness": fair,
               "test_rows": int(len(test)),
               "note": "Synthetic randomized data; effects were planted by the generator. Values are IPW estimates."}
    json_path = OUT / "metrics.json"; json_path.write_text(json.dumps(metrics, indent=2))
    joblib.dump(eng, OUT / "engine.joblib")
    test[["user_id", "region", "age_band"] + FEATURES].to_csv(OUT / "population.csv", index=False)
    print(json.dumps({k: v["incremental_vs_no_campaign"] for k, v in res.items()}, indent=2))
    print("qini:", qini); print("fairness ratios:", {g: f["contact_rate_ratio_min_over_max"] for g, f in fair.items()})

if __name__ == "__main__":
    main()
