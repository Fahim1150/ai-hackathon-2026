"""Deterministic business rules + segmentation. Separate from the ML model on purpose."""
from .config import FATIGUE_CAP, MIN_GAIN_BDT, BASE_HIGH, UPLIFT_MIN

def segment(baseline_prob: float, uplift: float) -> str:
    if uplift < -0.01:
        return "Sleeping Dog (offer may hurt - do not contact)"
    if uplift >= UPLIFT_MIN:
        return "Persuadable (converts because of the offer)"
    if baseline_prob >= BASE_HIGH:
        return "Sure Thing (converts anyway - save the budget)"
    return "Lost Cause (offer does not move them)"

def apply_rules(offers_last_30d: int, expected_gain: float, uplift: float):
    """Return (allowed, reasons)."""
    reasons = []
    if offers_last_30d >= FATIGUE_CAP:
        reasons.append(f"Offer-fatigue cap: {offers_last_30d} offers in last 30 days")
    if expected_gain < MIN_GAIN_BDT:
        reasons.append(f"Expected gain below {MIN_GAIN_BDT} BDT threshold")
    if uplift < UPLIFT_MIN:
        reasons.append("Uplift below minimum meaningful effect")
    return (len(reasons) == 0), reasons
