import os
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")

st.title("AI Hackathon Prototype")
st.caption("Synthetic data only")

amount = st.number_input("Amount", value=1000.0)
if st.button("Analyze"):
    r = requests.post(f"{API}/predict", json={"features": {"amount": amount}}, timeout=60)
    out = r.json()
    st.metric("Score", out["score"])
    st.write("Reasons:", out["reasons"])
    if out["needs_human_review"]:
        st.warning("Flagged for human review")