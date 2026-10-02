# CampaignIQ - Uplift & Next-Best-Offer Engine

> One-line pitch. Built for AI DEV FEST 2026 – AI Hackathon (DIU CPC × upay).

## 1. Project Overview
- **User:** <customer / merchant / agent / ops persona>
- **Problem:** <what is difficult, costly or risky + baseline>
- **Solution:** <what we built>
- **Purpose / impact metric:** <metric and target>

Problem statement: For [user], [problem] causes [consequence]. We built [product] that uses [data] to [action], measured by [metric].

## 2. Features
| Feature | How AI is used |
|---|---|
| <feature> | <model / LLM / method> |

## 3. Technology Stack
Languages, frameworks, AI models, APIs, libraries, services (with versions).

## 4. Requirements
Python 3.11+, Node 20+ (if frontend), <other>.

## 5. Installation & Setup
```bash
git clone <repo-url> && cd <repo>
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
cp .env.example .env   # then fill in values
python data/generate_synthetic.py
```

## 6. Environment Variables
| Name | Purpose | How to get it |
|---|---|---|
| `LLM_API_KEY` | LLM provider key | <link/instructions> (placeholder only, never commit real values) |

## 7. Run & Build
```bash
uvicorn backend.app.main:app --reload --port 8000
# build (if applicable): <command>
```

## 8. Live Deployment
URL: <https://...>

## 9. Testing
```bash
pytest backend/tests
```
Manual verification steps: <steps>

## 10. Other Configuration
Extra files, access requirements, demo credentials (non-sensitive).

## Data & Responsible AI
- Data is 100% synthetic: see `docs/SYNTHETIC_ASSUMPTIONS.md`.
- Explainability, fairness check, human oversight: <describe>
- AI tools used: see `docs/AI_USAGE_LOG.md`.
