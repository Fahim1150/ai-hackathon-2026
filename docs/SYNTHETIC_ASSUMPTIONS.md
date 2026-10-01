# Synthetic Data Assumptions
- Seed: 42. All data is generated; no real PII used.
- Amounts ~ lognormal(6.5, 1.0) (assumption, not real upay data).
- 1% anomalies injected as 8x amounts (ground truth for testing).
- Segment split 50/35/15. 3% of users are agents.
- Clean test set held out, never used for training.
