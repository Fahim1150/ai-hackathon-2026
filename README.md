# upay ActivateAI 🚀

**Dormant-to-Active Lifecycle, Incremental Uplift & Gemini Copilot Engine**  
*Built for AI DEV FEST 2026 – AI Hackathon (Track 04: Growth & Campaign Intelligence / DIU CPC × upay)*

---

## 📌 1. Project Overview

**The Business Problem:**  
In Bangladesh's MFS (Mobile Financial Services) sector, competitors heavily dominate the peer-to-peer (P2P) "Send Money" market due to network effects. Consequently, many registered **upay** users become dormant or cash out 100% of their salary on payday. 

**The Strategy:**  
"Solo-utility" transactions (Utility Bill Pay, Mobile Recharge, Super Shop QR, DPS savings) do **not** require the recipient to be an active upay user. By building habits around these specific features, we can bypass the competitor's network effect.

**The Solution:**  
**upay ActivateAI** is an end-to-end intelligence engine that maximizes Monthly Active Users (MAU) under a fixed marketing budget. Instead of predicting *who will transact* (which wastes budget on "Sure Things"), it uses **Incremental Uplift Modeling** (T-Learner LightGBM) to predict *who will activate SPECIFICALLY because of the offer* (Persuadables). It then uses **Gemini 2.5 Flash** to generate highly personalized, bilingual SMS nudges grounded in SHAP explainability drivers.

---

## ✨ 2. Core Features (The 4-Tab Dashboard)

1. **📊 Dormancy Funnel & MAU Growth Simulator**  
   Interactive sandbox for Growth Managers. Adjust the Reactivation Budget and Fatigue Caps to instantly simulate expected MAU growth. Compares the AI's uplift-targeted allocation against a traditional "Mass Promo Blast" baseline. Includes **Gemini-generated Experiment Intelligence** to recommend next A/B tests.

2. **📈 Solo-Utility Habit & Uplift Explorer**  
   Visualizes the 4 Uplift Quadrants (*Persuadables*, *Sure Things*, *Lost Causes*, *Sleeping Dogs*). Maps the optimal solo-utility offer (Recharge, Bill Pay, etc.) based on historical affinity and analyzes uplift responsiveness across different lifecycle stages (e.g., *Payday Cash-Outer* vs *One-Hit Wonder*).

3. **🎯 Customer 360, SHAP & Gemini Copywriter**  
   Drill down into individual synthetic profiles. Displays the precise **SHAP (SHapley Additive exPlanations)** drivers for why a user scored a specific uplift. Leverages **Gemini 2.5 Flash** to instantly draft English and Bangla SMS copy customized to the user's dormancy stage and SHAP drivers.

4. **🛡️ Responsible AI & Human Oversight**  
   Enforces strict guardrails. Includes a Fairness Audit across wallet types (Salary/Primary/Remittance) and lifecycle stages. Enforces a **Human Reviewer** sign-off gate before any campaign can transition from draft to approved.

---

## 🛠️ 3. Technology Stack

*   **Data Science & ML:** Python 3.13, Pandas, Scikit-learn, LightGBM, SHAP, SciPy.
*   **Backend API:** FastAPI, Uvicorn, Pydantic, Pytest.
*   **Generative AI:** Official `google-genai` SDK (Gemini 2.5 Flash) with strict JSON Structured Outputs.
*   **Frontend UI:** React 18, Vite, Tailwind CSS v4, Recharts, Lucide-React.
*   **Data:** 100% Synthetic data generated in-memory (No real PII).

---

## 🚦 4. Strict Architecture Guardrails

This project strictly adheres to 7 core Responsible AI guardrails (enforced via `AGENTS.md`):
1. **Separation of Concerns:** Data prep, ML inference, and API serving are strictly isolated.
2. **Deterministic Targeting:** Budget caps, offer eligibility, and uplift cutoffs are hard-coded business rules (never LLM-driven).
3. **No Financial Decisions in LLMs:** Gemini is used *strictly* as a translation and copywriting engine based on deterministic SHAP inputs.
4. **Traceability:** Every recommendation requires top-3 SHAP attributions.
5. **Human-in-the-Loop:** No campaign launches without human approval logs (Reviewer Name, Timestamp, Budget).
6. **Deterministic Fallback:** If the Gemini API key is missing or the network is offline, the system gracefully degrades to a template-based fallback (marked as `Offline Deterministic Fallback`) without crashing.
7. **Train/Test Isolation:** Strictly enforced zero `customer_id` overlap between training and inference sets.

---

## ⚙️ 5. Installation & Setup

### Prerequisites
*   macOS / Linux (zsh/bash)
*   Python 3.13+
*   Node.js 20+

### Step 1: Clone & Environment Setup
```bash
git clone https://github.com/Fahim1150/ai-hackathon-2026.git
cd ai-hackathon-2026

# Setup Python Virtual Environment
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn lightgbm shap scipy fastapi "uvicorn[standard]" pytest httpx google-genai pydantic python-dotenv
```

### Step 2: Configure Gemini (Optional but Recommended)
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY. 
# If left empty, the app uses the built-in Offline Deterministic Fallback.
```

### Step 3: Generate Data & Train Models
Because no real user data is tracked in Git, you must generate the synthetic dataset and train the ML models locally:
```bash
# 1. Generate 10k synthetic upay users (train & test sets)
python backend/data_generator.py

# 2. Train LightGBM T-Learner, compute SHAP, and score users
python backend/ml_engine.py
```

### Step 4: Run the Backend & Frontend Servers
Open two separate terminal windows:

**Terminal 1 (FastAPI Backend):**
```bash
source .venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

**Terminal 2 (React Frontend):**
```bash
cd frontend
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```
👉 Access the UI at: **http://localhost:5173**

---

## 🧪 6. Testing

The backend includes a comprehensive 25-test Pytest suite that verifies Train/Test isolation, uplift score bounds, budget constraint adherence, Gemini fallback resilience, and endpoint health.

```bash
source .venv/bin/activate
python -m pytest backend/test_app.py -v
```

---

*Designed for the upay Growth & Campaign Intelligence Track.*
