"""Train, evaluate on a held-out RCT test set, save artifacts.
Run: python -m backend.app.train
"""
import json, pathlib, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from .config import (OFFERS, MARGIN, FEATURES, PROPENSITY,
                     FATIGUE_CAP, MIN_GAIN_BDT, UPLIFT_MIN, GROUP_COLS)
from .uplift import UpliftEngine, ResponseModel, COSTS

OUT = pathlib.Path("models")
OUT.mkdir(exist_ok=True)


# ── evaluation helpers ──────────────────────────────────────────────────

def policy_value(test, chosen):
    """Unbiased profit/user of a policy via inverse-propensity weighting
    (valid because of random assignment in the synthetic RCT)."""
    match = (test["arm"].values == chosen)
    return float((match * test["profit"].values / PROPENSITY).mean())


def qini_auc(score, treat, y):
    """Average gap (incremental conversions) between Qini curve and random."""
    o = np.argsort(-score)
    t, yy = treat[o], y[o]
    nt = np.cumsum(t)
    nc = np.cumsum(1 - t)
    yt = np.cumsum(yy * t)
    yc = np.cumsum(yy * (1 - t))
    q = yt - yc * np.divide(nt, nc, out=np.zeros_like(nt, dtype=float), where=nc > 0)
    rand = np.linspace(0, q[-1], len(q))
    return float((q - rand).mean())


def auuc(score, treat, y):
    """Area Under the Uplift Curve (normalised AUUC).
    Fraction of the Qini AUC relative to the perfect model's Qini AUC."""
    model_qini = qini_auc(score, treat, y)
    # Perfect model: score = true uplift = y_t - E[y_c]
    # Approximate perfect by sorting treated-converted first
    perfect_score = treat * y  # 1 for treated converters, 0 otherwise
    perfect_qini = qini_auc(perfect_score.astype(float), treat, y)
    if abs(perfect_qini) < 1e-9:
        return 0.0
    return round(model_qini / perfect_qini, 4)


# ── main training pipeline ──────────────────────────────────────────────

def main():
    # Load synthetic data
    data_path = pathlib.Path("data/synthetic_customers.csv")
    if not data_path.exists():
        raise FileNotFoundError(
            "Run `python data/generate_synthetic.py` first to create the dataset."
        )
    df = pd.read_csv(data_path)

    # 70/30 stratified split — test set is NEVER used for training
    train, test = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df["arm"]
    )
    print(f"Train: {len(train):,} rows | Test: {len(test):,} rows (held out)")

    # ── fit uplift engine (T-learner, LightGBM) ─────────────────────────
    train_cols = list(dict.fromkeys(FEATURES + GROUP_COLS + ["user_id"]))
    eng = UpliftEngine().fit(
        train[train_cols],
        train["arm"].values,
        train["converted"].values,
    )

    # ── fit response model baseline ──────────────────────────────────────
    resp = ResponseModel().fit(train, train["converted"].values)

    # ── score test set ───────────────────────────────────────────────────
    test_cols = list(dict.fromkeys(FEATURES + GROUP_COLS + ["user_id"]))
    test_input = test[test_cols].copy()
    test_input.index = test.index
    s = eng.score(test_input)
    P = eng.arm_probs(test_input)
    resp_probs = resp.predict_proba(test_input)

    # ── define policies ──────────────────────────────────────────────────
    n_offers = len(OFFERS) - 1  # exclude control
    allowed = np.array([
        (o < FATIGUE_CAP) and g >= MIN_GAIN_BDT and u >= UPLIFT_MIN
        for o, g, u in zip(test["offers_last_30d"], s["expected_gain"], s["best_uplift"])
    ])

    pol = {
        "No campaign":
            np.zeros(len(test), int),
        "Blast offer 1 to everyone":
            np.ones(len(test), int),
        "Blast offer 2 to everyone":
            np.full(len(test), 2),
        "Blast offer 3 to everyone":
            np.full(len(test), 3),
        "Random offer":
            np.random.default_rng(0).integers(0, len(OFFERS), len(test)),
        "Equal-split (round-robin)":
            np.array([(i % n_offers) + 1 for i in range(len(test))]),
        "Response model (ignores baseline)":
            P[:, 1:].argmax(1) + 1,
        "Uplift model":
            np.where(s["expected_gain"] > 0, s["best_offer"], 0),
        "Uplift model + business rules":
            np.where(allowed, s["best_offer"], 0),
    }

    res = {}
    for name, ch in pol.items():
        pv = policy_value(test, ch)
        res[name] = {
            "profit_per_user_bdt": round(pv, 2),
            "contact_rate": round(float((ch > 0).mean()), 3),
        }
    base = res["No campaign"]["profit_per_user_bdt"]
    for r in res.values():
        r["incremental_vs_no_campaign"] = round(r["profit_per_user_bdt"] - base, 2)

    # ── Qini AUC & AUUC per offer ───────────────────────────────────────
    qini = {}
    rng = np.random.default_rng(1)
    for k in range(1, len(OFFERS)):
        m = test["arm"].isin([0, k]).values
        t = (test["arm"].values[m] == k).astype(int)
        y = test["converted"].values[m]
        model_score = P[m, k] - P[m, 0]
        random_score = rng.random(m.sum())

        model_qini = round(qini_auc(model_score, t, y), 3)
        random_qini = round(qini_auc(random_score, t, y), 3)
        model_auuc = auuc(model_score, t, y)

        qini[OFFERS[k]["name"]] = {
            "qini_auc_model": model_qini,
            "qini_auc_random": random_qini,
            "auuc": model_auuc,
        }

    # ── fairness check ───────────────────────────────────────────────────
    fair = {}
    tmp = test.assign(chosen=pol["Uplift model + business rules"])
    for g in GROUP_COLS:
        rows = {}
        for val, d in tmp.groupby(g):
            rows[str(val)] = {
                "contact_rate": round(float((d["chosen"] > 0).mean()), 3),
                "avg_uplift": round(float(
                    s.loc[d.index, "best_uplift"].mean()
                ), 4),
                "profit_per_user_bdt": round(
                    policy_value(d, d["chosen"].values), 2
                ),
                "n": int(len(d)),
            }
        rates = [v["contact_rate"] for v in rows.values()]
        fair[g] = {
            "groups": rows,
            "contact_rate_ratio_min_over_max":
                round(min(rates) / max(max(rates), 1e-9), 3),
        }

    # ── segment distribution ─────────────────────────────────────────────
    from .rules import segment
    seg_labels = [
        segment(bp, bu) for bp, bu
        in zip(s["baseline_prob"], s["best_uplift"])
    ]
    seg_counts = pd.Series(seg_labels).value_counts().to_dict()

    # ── assemble metrics ─────────────────────────────────────────────────
    metrics = {
        "policies": res,
        "qini_auc": qini,
        "fairness": fair,
        "segments": seg_counts,
        "test_rows": int(len(test)),
        "train_rows": int(len(train)),
        "n_offers": n_offers,
        "note": "Synthetic randomized data; effects were planted by the generator. "
                "Values are IPW estimates.",
    }

    # ── save ─────────────────────────────────────────────────────────────
    json_path = OUT / "metrics.json"
    json_path.write_text(json.dumps(metrics, indent=2))
    joblib.dump(eng, OUT / "engine.joblib")
    joblib.dump(resp, OUT / "response_model.joblib")
    out_cols = list(dict.fromkeys(["user_id", "region", "age_band"] + FEATURES))
    test[out_cols].to_csv(
        OUT / "population.csv", index=False
    )
    print(f"\nSaved: {json_path}, engine.joblib, response_model.joblib, population.csv")

    # ── print summary ────────────────────────────────────────────────────
    print("\n── Policy comparison (incremental BDT vs no campaign) ──")
    for name, v in res.items():
        print(f"  {name:45s} {v['incremental_vs_no_campaign']:+.2f} BDT  "
              f"(contact {v['contact_rate']:.1%})")
    print("\n── Qini AUC / AUUC per offer ──")
    for name, v in qini.items():
        print(f"  {name:35s}  Qini={v['qini_auc_model']:8.3f}  "
              f"Random={v['qini_auc_random']:8.3f}  AUUC={v['auuc']:.4f}")
    print("\n── Fairness ──")
    for g, f in fair.items():
        print(f"  {g}: min/max contact-rate ratio = "
              f"{f['contact_rate_ratio_min_over_max']}")
    print("\n── Segments ──")
    for seg, cnt in seg_counts.items():
        print(f"  {seg}: {cnt}")


if __name__ == "__main__":
    main()
