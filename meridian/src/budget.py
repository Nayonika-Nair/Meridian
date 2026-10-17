"""Adaptive leak budget, document pricing and the privacy brake.

allowance = BASE x (1 - P_ai) x criticality_factor x history_factor
history_factor = 1 - HISTORY_MAX_CUT x min(1, privacy_debt / (DEBT_CRITICAL_MULT x BASE))
With zero privacy debt this is exactly the original BASE x (1-P_ai) x criticality.
"""
from .config import (BASE, WEIGHTS, BRAKE, HISTORY_MAX_CUT, DEBT_CRITICAL_MULT, debt_tier, p_ai_tier)


def allowance(base, p_ai, criticality_factor):
    return base * (1 - p_ai) * criticality_factor


def history_factor(debt, base=BASE):
    if not base:
        return 1.0
    return 1 - HISTORY_MAX_CUT * min(1.0, max(0.0, debt) / (DEBT_CRITICAL_MULT * base))


def adaptive_allowance(base, p_ai, criticality_factor, debt=0.0):
    return allowance(base, p_ai, criticality_factor) * history_factor(debt, base)


def allowance_breakdown(base, p_ai, criticality, criticality_factor, debt=0.0):
    """Never hide the calculation: every term that moves the budget, in order."""
    hf = history_factor(debt, base)
    final = base * (1 - p_ai) * criticality_factor * hf
    lines = [
        {'term': 'BASE', 'value': f'{base:,.0f}', 'effect': 'starting allowance per window'},
        {'term': 'AI exposure', 'value': f'{p_ai:.1%}', 'effect': f'x {1 - p_ai:.3f}  (1 - P_ai, model estimate)'},
        {'term': 'Criticality', 'value': criticality.upper(), 'effect': f'x {criticality_factor:.2f}'},
        {'term': 'Historical exposure', 'value': f'{debt:,.0f} debt pts ({debt_tier(debt, base)})',
         'effect': f'x {hf:.3f}  (cut {1 - hf:.1%}, max {HISTORY_MAX_CUT:.0%})'},
    ]
    return {'lines': lines, 'allowance': final, 'history_factor': hf, 'p_ai_factor': 1 - p_ai,
            'criticality_factor': criticality_factor, 'debt': debt, 'base': base, 'p_ai': p_ai,
            'criticality': criticality}


def destination_multiplier(p_ai):
    return 1 + p_ai


def residual_risk(entities, undetected_risk=0.0, ocr_penalty=0.0, unknown_regions=0):
    """Weighted probability-of-leak summed over entities (confidence = chance it was removed)."""
    r = sum(WEIGHTS.get(e.get('type', 'UNKNOWN'), 1) * (1 - float(e.get('confidence', 0))) for e in entities)
    return r + undetected_risk + ocr_penalty + 2 * unknown_regions


def doc_cost(entities, p_ai, undetected_risk=0.0, ocr_penalty=0.0, unknown_regions=0):
    return residual_risk(entities, undetected_risk, ocr_penalty, unknown_regions) * destination_multiplier(p_ai)


_ORDER = ['AUTO', 'STRICT', 'REVIEW', 'BLOCK']


def decision(balance, initial, cost, has_unresolved_id=False, min_state=None):
    """Privacy brake. GOV_ID hard-block overrides everything.
    `min_state` lets an approved policy force a stricter floor (e.g. 'REVIEW')."""
    if has_unresolved_id:
        return 'BLOCK'
    r = (balance - cost) / initial if initial else 0
    if r > BRAKE['AUTO']:
        d = 'AUTO'
    elif r > BRAKE['STRICT']:
        d = 'STRICT'
    elif r > BRAKE['REVIEW']:
        d = 'REVIEW'
    else:
        d = 'BLOCK'
    if min_state and _ORDER.index(min_state) > _ORDER.index(d):
        d = min_state
    return d


def decision_explanation(a):
    return {'AUTO': 'Green: normal automated approval.',
            'STRICT': 'Amber: approve only after stricter redaction/validation.',
            'REVIEW': 'Human review required; budget is nearing exhaustion or policy demands review.',
            'BLOCK': 'Red: block the transfer and alert the reviewer.'}[a]
