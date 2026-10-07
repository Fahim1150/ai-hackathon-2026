# upay ActivateAI: Dormant to Active Lifecycle & Incremental Uplift Engine

## Executive Summary
While competitors dominate P2P transfers, upay ActivateAI leverages causal Machine Learning to reactivate dormant upay users through solo utility transactions (e.g., bill pay, mobile recharge, toll payments, and super shop purchases). By shifting from reactive mass blasts to precision causal targeting, this engine identifies the exact users who will adopt a 30-day repeat habit solely because of an incentive, maximizing Monthly Active Users (MAU) under a fixed marketing budget.

## The Problem
Mobile financial services face three critical growth hurdles:
* **High User Dormancy:** A significant portion of the user base downloads the app but fails to form a lasting habit.
* **Immediate Cash Out Behavior:** Users frequently cash out their entire balance on payday, leaving no float for utility transactions.
* **Wasted Campaign Budgets:** Traditional mass SMS blasts suffer from heavy cashback waste, rewarding users who would have transacted anyway (Sure Things) while annoying those who will never convert (Lost Causes).

## The Solution
ActivateAI is a full stack Machine Learning pipeline that solves the dormancy crisis through causal inference. The platform identifies "Persuadables" (users who activate strictly because of a promotion), strictly limits offer fatigue, and dynamically optimizes budget allocation. Once a campaign is optimized, the platform utilizes the Gemini 2.5 Flash API to generate SHAP grounded, bilingual SMS nudges (Bangla and English) tailored to the unique behavioral drivers of each user.

## Key Innovations
Our platform integrates rigorous statistical validation, strict data governance, and scalable MLOps, directly addressing advanced hackathon criteria:

* **Causal ML Rigor:** Implements advanced T-Learner and S-Learner meta-learners alongside Propensity Baselines. Robustness is verified through 100-iteration bootstrap confidence intervals, Doubly Robust Off-Policy Evaluation (OPE), and unobserved confounder Sensitivity Analysis.
* **Deterministic Knapsack Optimizer:** Replaces static allocations with a dynamic pricing algorithm. It dynamically markdowns offer costs to the minimum required tier to cross the uplift threshold, maximizing Incremental MAU per budget dollar.
* **Scalability & MLOps:** Production ready architecture featuring a dedicated Model Registry, Population Stability Index (PSI) Data Drift Monitoring, and a governed Data Ingestion API that hashes identifiers and strips PII before batch inference.
* **Security & Responsible AI:** Enforces human-in-the-loop campaign oversight via a persistent SQLite Approval Ledger. The backend is hardened with Role-Based Access Control (RBAC), strict CORS origins, and rate limiting to prevent API abuse.

## Architecture Stack
* **Backend:** FastAPI, Python 3, LightGBM, SHAP, `google-genai`, SQLite, Pytest
* **Frontend:** React, Vite, Tailwind CSS, Recharts, Lucide Icons

## Local Setup Instructions
The following instructions are tailored for a macOS zsh environment.

### 1. Clone & Environment Setup
```zsh
# Navigate to your workspace
cd /path/to/ai-hackathon-2026

# Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install backend dependencies
pip install -r backend/requirements.txt
```

### 2. Environment Variables
Create a `.env` file in the root directory and add your Gemini API key:
```zsh
echo "GEMINI_API_KEY=your_api_key_here" > .env
```
*(Note: The system features a deterministic offline fallback if the API key is omitted.)*

### 3. Run Test Suite
Verify the backend integrity and security constraints:
```zsh
# Run pytest on the backend
export PYTHONPATH=.
python -m pytest
```

### 4. Start Development Servers
You will need two terminal windows to run the frontend and backend concurrently.

**Terminal 1 (Backend):**
```zsh
source .venv/bin/activate
export PYTHONPATH=.
uvicorn backend.main:app --reload --port 8000
```

**Terminal 2 (Frontend):**
```zsh
cd frontend
npm install
npm run dev
```
Navigate to `http://localhost:5173` in your browser to view the application.

## Compliance Disclaimer
**Note:** All data used in the training, evaluation, and demonstration of upay ActivateAI is strictly synthetic. No real customer data, financial records, or Personally Identifiable Information (PII) is utilized in this repository, ensuring full adherence to hackathon privacy guidelines and ethical AI standards.
