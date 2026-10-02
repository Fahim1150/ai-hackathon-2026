import os, requests, pandas as pd, streamlit as st

API = os.getenv("API_URL", st.secrets.get("API_URL", "http://localhost:8000") if hasattr(st, "secrets") else "http://localhost:8000")
st.set_page_config(page_title="CampaignIQ", layout="wide")
st.title("CampaignIQ - Uplift & Next-Best-Offer")
st.caption("Synthetic data only. Estimates come from a randomized synthetic experiment; effects were planted by the data generator.")

def call(method, path, **kw):
    try:
        r = requests.request(method, f"{API}{path}", timeout=90, **kw)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e} (free hosts may take ~60s to wake up; retry)")
        return None

tab1, tab2, tab3 = st.tabs(["1. Does it beat blasting?", "2. Plan a campaign", "3. Single user"])

with tab1:
    m = call("GET", "/metrics")
    if m:
        pol = pd.DataFrame(m["policies"]).T
        st.subheader("Profit per user vs. no campaign (BDT, held-out test set)")
        st.bar_chart(pol["incremental_vs_no_campaign"])
        st.dataframe(pol)
        st.subheader("Fairness check: contact rate by group")
        for g, v in m["fairness"].items():
            st.write(f"**{g}** (min/max contact-rate ratio: {v['contact_rate_ratio_min_over_max']})")
            st.dataframe(pd.DataFrame(v["groups"]).T)
        st.caption(m["note"])

with tab2:
    budget = st.slider("Expected promo budget (BDT)", 1000, 100000, 20000, 1000)
    if st.button("Build campaign plan"):
        st.session_state["plan"] = call("POST", "/plan", json={"budget_bdt": budget})
    plan = st.session_state.get("plan")
    if plan:
        c1, c2, c3 = st.columns(3)
        c1.metric("Users selected", plan["selected_users"], f"of {plan['population_size']}")
        c2.metric("Expected spend (BDT)", plan["expected_spend_bdt"])
        c3.metric("Expected incremental profit (BDT)", plan["expected_incremental_profit_bdt"])
        st.bar_chart(pd.Series(plan["offers_assigned"]))
        st.warning("Status: " + plan["status"])
        if st.button("Approve campaign (human sign-off)"):
            st.success("Approved by growth manager. (Demo: no messages are actually sent.)")

with tab3:
    c = st.columns(3)
    f = {
        "monthly_trans_count": c[0].number_input("Transactions / month", 0, 100, 12),
        "avg_trans_amount": c[1].number_input("Avg amount (BDT)", 0.0, 50000.0, 300.0),
        "days_since_last_trans": c[2].number_input("Days since last txn", 0, 90, 25),
        "tenure_months": c[0].number_input("Tenure (months)", 0, 120, 12),
        "cashout_ratio": c[1].slider("Cash-out ratio", 0.0, 1.0, 0.3),
        "offers_last_30d": c[2].number_input("Offers received, last 30d", 0, 20, 0),
    }
    if st.button("Recommend"):
        r = call("POST", "/recommend", json=f)
        if r:
            st.subheader(r["decision"])
            st.write("Segment:", r["segment"])
            st.write(f"Baseline conversion: {r['baseline_conversion_prob']} | Uplift: {r['uplift']} | Expected gain: {r['expected_profit_gain_bdt']} BDT")
            if r["rules_blocking"]:
                st.info("Business rules blocked sending: " + "; ".join(r["rules_blocking"]))
            st.write("Main drivers:")
            st.dataframe(pd.DataFrame(r["reasons"]))
            st.caption(r["reason_note"])
