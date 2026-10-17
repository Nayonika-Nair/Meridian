"""Two deliberately separate assessments.

AI Adoption Evidence Score (0-100): how strongly the collected evidence indicates the VENDOR adopts AI at all.
Meridian AI Exposure Assessment: what the evidence says about whether MERIDIAN DATA reaches AI. AI adoption alone
never raises this above "no flow evidence" - only explicit processing disclosures and analyst-reviewed observations do.

ALL WEIGHTS BELOW ARE CONFIGURABLE ASSUMPTIONS, not measured quantities. They are shown in the UI.
"""
from __future__ import annotations

ADOPTION_WEIGHTS = {
    "class": {"marketing_claim": 0.15, "technical_indication": 0.45, "corroborated_claim": 0.55,
              "explicit_processing_disclosure": 0.80, "observed_flow": 0.90, "verified_flow": 0.95},
    "specificity": {"provider": 1.0, "endpoint": 1.0, "capability": 0.8, "vague": 0.5},
    "reliability": {1: 1.0, 2: 0.85, 3: 0.7},
    "max_items_per_domain": 2,           # many pages on one domain must not inflate the score
}
LEVELS = ["NO_EVIDENCE", "NO_FLOW_EVIDENCE", "DISCLOSED_PROCESSING", "FLOW_OBSERVED"]
LEVEL_TEXT = {
    "NO_EVIDENCE": "No AI-related evidence collected. This is not proof the vendor does not use AI.",
    "NO_FLOW_EVIDENCE": "The vendor appears to use AI, but nothing shows Meridian data reaching it.",
    "DISCLOSED_PROCESSING": "A vendor document ties AI processing to customer data or lists an AI provider as a sub-processor. Not an observed flow.",
    "FLOW_OBSERVED": "An analyst recorded an authorised observation of Meridian data reaching an AI service.",
}


def _item_w(e, W):
    return (W["class"].get(e["effective_class"], 0.0) * W["specificity"].get(e.get("specificity"), 0.5) * W["reliability"].get(e.get("reliability_tier"), 0.7))


def adoption_score(evidence: list[dict], weights: dict | None = None) -> dict:
    """1 - prod(1 - w_i) over counted items. Rejected, negated and fixture items are excluded; at most N items per domain."""
    W = weights or ADOPTION_WEIGHTS
    counted, per_dom, excluded = [], {}, {"rejected": 0, "negated": 0, "fixture": 0, "domain_cap": 0}
    for e in sorted(evidence, key=lambda x: -_item_w(x, W)):
        if e.get("review_status") == "rejected":
            excluded["rejected"] += 1
        elif e.get("negated"):
            excluded["negated"] += 1
        elif e.get("origin") == "fixture":
            excluded["fixture"] += 1
        elif per_dom.get(e["domain"], 0) >= W["max_items_per_domain"]:
            excluded["domain_cap"] += 1
        else:
            per_dom[e["domain"]] = per_dom.get(e["domain"], 0) + 1
            counted.append(e)
    prod = 1.0
    for e in counted:
        prod *= 1 - _item_w(e, W)
    n = len(counted)
    domains = len({e["domain"] for e in counted})
    conf = "insufficient" if n < 2 or domains < 2 else "moderate" if domains < 3 else "reasonable"
    return {"score": round(100 * (1 - prod), 1), "items_counted": n, "domains": domains, "excluded": excluded, "evidence_breadth": conf,
            "weights_are": "configurable assumptions", "weights": W}


def score_pair(evidence: list[dict], weights=None) -> dict:
    """Score with every non-rejected item (a hypothesis) and with analyst-accepted items only (what the Twin may use)."""
    return {"hypothesis_all_unrejected": adoption_score(evidence, weights),
            "reviewed_only": adoption_score([e for e in evidence if e.get("review_status") == "accepted"], weights)}


def exposure_assessment(vendor: dict, evidence: list[dict], p_ai: float | None = None, flow_discount: float | None = None) -> dict:
    live = [e for e in evidence if e.get("origin") != "fixture" and e.get("review_status") != "rejected" and not e.get("negated")]
    acc = [e for e in live if e.get("review_status") == "accepted"]
    observed = [e for e in acc if e["effective_class"] in ("observed_flow", "verified_flow")]
    disclosed = [e for e in live if e["effective_class"] == "explicit_processing_disclosure"]
    ai = [e for e in live if e["effective_class"] != "explicit_processing_disclosure"]
    if observed:
        level = "FLOW_OBSERVED"
    elif disclosed:
        level = "DISCLOSED_PROCESSING"
    elif ai:
        level = "NO_FLOW_EVIDENCE"
    else:
        level = "NO_EVIDENCE"
    reasons = []
    if observed:
        reasons.append(f"{len(observed)} accepted observation(s) with an authorised reference.")
    if disclosed:
        n_acc = sum(1 for e in disclosed if e.get("review_status") == "accepted")
        reasons.append(f"{len(disclosed)} disclosure(s) in vendor documents ({n_acc} analyst-accepted, {len(disclosed) - n_acc} still hypothesis).")
    if ai and not (observed or disclosed):
        reasons.append("AI wording found, but none of it concerns customer data or a sub-processing relationship.")
    reasons.append(f"Modelled P_ai uses reviewed evidence only: {p_ai:.1%}." if p_ai is not None else "Modelled P_ai not supplied.")
    if flow_discount is not None and flow_discount >= 1.0 and any(e["effective_class"] not in ("observed_flow", "verified_flow") for e in acc):
        reasons.append("The AI-use discount is 1.0, so accepted AI-use-only evidence is NOT discounted in the Twin. Lower it to treat AI use as weaker than a proven flow.")
    return {"level": level, "summary": LEVEL_TEXT[level], "reasons": reasons, "p_ai": p_ai,
            "counts": {"observed_or_verified": len(observed), "disclosures": len(disclosed), "adoption_only": len(ai),
                       "reviewed": len(acc), "pending": sum(1 for e in live if e.get("review_status") in ("pending", "needs_review"))},
            "never_implied": "AI use alone is never treated as a confirmed Meridian data flow."}
