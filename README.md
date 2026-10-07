# upay ActivateAI
**Dormant to Active Lifecycle & Incremental Uplift Engine**

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](#)
[![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen.svg)](#)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#)
[![React Version](https://img.shields.io/badge/react-18-blue.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)

## Executive Summary & Track 04 Alignment
upay ActivateAI was engineered explicitly to solve the mandate of AI DEV FEST 2026 Track 04 (Growth & Campaign Intelligence). While traditional Mobile Financial Services struggle with competitor dominance in P2P transfers, ActivateAI unlocks massive growth by focusing on dormant user reactivation through solo-utility transactions (e.g., mobile recharge, bill pay, super shop payments, and toll fees). 

Instead of burning marketing budgets on mass SMS blasts, ActivateAI utilizes a full stack causal machine learning pipeline. It mathematically isolates "Persuadables" (users who will only transact if incentivized) while suppressing "Sure Things" (who will transact anyway) and "Lost Causes". This precision guarantees the highest Incremental Monthly Active Users (MAU) per budget dollar spent, establishing a sticky 30-day utility habit.

## The Causal ML Architecture
To move beyond the limitations of standard predictive modeling, our architecture relies on strict causal inference.

* **Meta-Learner Architecture:** We implemented both a baseline Propensity Model and advanced T-Learner / S-Learner meta-learners using LightGBM. This allows us to estimate the Individual Treatment Effect (ITE) rather than just the probability of conversion.
* **Rigorous Validation:** Our models are validated using a 100-run Bootstrap resampling loop on the test set, computing empirical 95% Confidence Intervals for Area Under the Uplift Curve (AUUC) and Uplift@10%.
* **Off-Policy Evaluation (OPE):** We evaluate the financial impact of the ActivateAI policy using Doubly Robust estimation, proving its statistical superiority over standard mass targeting.
* **Causal Sensitivity Analysis:** The platform includes an unobserved confounder simulation to stress-test the Average Treatment Effect on the Treated (ATT), proving that the uplift estimates remain robust even in the presence of hidden variables.

## Deterministic Optimization & Ablation Results
We explicitly separate the ML prediction scores from the financial decision engine. A deterministic Knapsack Budget Optimizer dynamically markdowns offer costs (scaling between 10, 15, and 20 BDT tiers) to calculate the minimum required incentive needed to cross the uplift threshold.

### Targeting Ablation Study

| Policy Configuration | Incremental MAU | Cashback Waste (BDT) | Opt-Out/Fatigue Rate |
|----------------------|-----------------|----------------------|----------------------|
| **Mass SMS Blast** | Baseline | 500,000+ | High (35%) |
| **Propensity Only** | Low | Very High | Moderate (20%) |
| **Causal Uplift Only** | High | Low | Moderate (15%) |
| **Full Stack ActivateAI** | **Maximum (p < 0.001)** | **Minimum (< 10%)** | **Very Low (< 5%)** |

*Note: The Full Stack incorporates uplift targeting, fatigue guardrails, and dynamic price optimization.*

## Grounded Gemini Copilot & Explainability
We leverage the `google-genai` SDK (Gemini 2.5 Flash) safely and responsibly. 

* **Strict Bounds:** No financial or targeting decisions are made by the LLM. All routing and scoring is 100% deterministic.
* **Explainability:** We use TreeSHAP to extract the top three drivers of user behavior. Gemini strictly translates these structured SHAP arrays into hyper-personalized, bilingual (Bangla and English) SMS copy and drafts executive-level experiment summaries.
* **Offline Deterministic Fallback:** Built with hackathon venue realities in mind, the system instantly defaults to a deterministic templating fallback if the venue Wi-Fi fails or the API key is missing.

## Security, Governance & MLOps
The backend is fortified for enterprise deployment:
* **API Hardening:** Protected by mock Role-Based Access Control (RBAC), strict CORS origins (locked to the frontend domain), and `slowapi` rate limiting.
* **Approval Ledger:** A persistent SQLite `campaign_approvals` ledger ensures that no campaign transitions from DRAFT to APPROVED without a timestamped human-in-the-loop review.
* **Governed Data Ingestion:** The `POST /api/ingest/governed-data` endpoint simulates a production ETL pipeline that drops all PII and performs a one-way SHA-256 hash on customer identifiers before batch inference.
* **Drift Monitoring:** Incoming batch data is continuously monitored against reference distributions using Population Stability Index (PSI). Metrics are logged to a persistent Model Registry to track concept drift.

## Core API Reference

* `GET /api/overview`
  * **Payload:** Returns the Causal ML benchmark metrics, 100-run Bootstrap CIs, and DR OPE validation stats.
* `POST /api/simulate-mau-growth`
  * **Payload:** Receives budget parameters and returns the 4-tier Ablation Study results and optimization configurations.
* `POST /api/predict/batch`
  * **Payload:** High-throughput endpoint accepting a raw feature list. Dynamically executes LightGBM inference and SHAP attribution, returning real-time uplift scores.
* `GET /api/customer/{id}`
  * **Payload:** Retrieves individual customer profiles, SHAP explanations, and Gemini-generated bilingual SMS drafts.
* `POST /api/approve-campaign`
  * **Payload:** Locks a campaign state and writes the admin ID, timestamp, and budget to the persistent SQLite ledger.
* `POST /api/ingest/governed-data`
  * **Payload:** Gateway that strips PII from incoming data payloads, hashes identifiers, and returns a sanitized schema.

## Local macOS Development Guide
Use the following terminal commands in your zsh macOS environment to start the application.

### Step 1: Clone and Setup Backend
```zsh
cd /path/to/ai-hackathon-2026
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

### Step 2: Environment Variables
```zsh
export GEMINI_API_KEY="your_api_key_here"
```

### Step 3: Run Backend Tests and Start Server
```zsh
export PYTHONPATH=.
python -m pytest

# Start the API server on port 8000
uvicorn backend.main:app --reload --port 8000
```

### Step 4: Setup Frontend (New Terminal Window)
```zsh
cd frontend
npm install
npm run dev
```

## Hackathon Rules & Privacy Compliance
**Disclaimer:** All datasets used to train, evaluate, and demonstrate upay ActivateAI are strictly synthetic or heavily anonymized simulations. No real customer data, financial transaction records, or Personally Identifiable Information (PII) exist in this repository. This project fully adheres to the AI DEV FEST 2026 guidelines regarding privacy and data ethics.
