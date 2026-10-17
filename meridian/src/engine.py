"""Control-plane engine: live vendor state + document gate.

vendor_state()  - everything the dashboard needs for one vendor (twin -> P_ai -> adaptive budget -> debt -> TTF)
evaluate_document() / commit() - destination-aware redaction, brake decision, explanation, receipt, ledger.
"""
import time
from .config import BASE, AI_USE_FLOW_DISCOUNT, p_ai_tier, debt_tier, worst
from .demo import vendor_assessment
from .budget import decision
from .debt import privacy_debt, window_spend, time_to_failure
from .redaction import plan_redaction
from .explain import decision_reasons
from .receipt import make_receipt
from .dna import data_dna

SPEND_STATES = ('AUTO', 'STRICT')
STATE_TIER = {'AUTO': 'LOW', 'STRICT': 'MODERATE', 'REVIEW': 'HIGH', 'BLOCK': 'CRITICAL'}


def policy_floor(policy):
    return 'REVIEW' if policy and policy.get('require_review') else None


def describe_policy(policy):
    if not policy:
        return []
    out = []
    if policy.get('extra_redact'):
        out.append('Always redact ' + ', '.join(policy['extra_redact']))
    if policy.get('require_review'):
        out.append('Human review required')
    out.append(f"Approved by {policy.get('approved_by', 'analyst')}")
    return out


def vendor_state(vendor, evidence_rows, ledger=None, base=BASE, trials=5000, flow_discount=AI_USE_FLOW_DISCOUNT,
                 now=None, policy=None, docs_per_day=None, avg_doc_cost=None, volume_multiplier=1.0,
                 p_ai_override=None, records=None):
    now = time.time() if now is None else now
    recs = records if records is not None else (ledger.records(vendor['vendor_name']) if ledger else [])
    debt = privacy_debt(recs)
    a = vendor_assessment(vendor, evidence_rows, base, trials, flow_discount, debt, p_ai_override)
    spent = window_spend(recs, now) * volume_multiplier
    allow = a['allowance']
    balance = max(0.0, allow - spent)
    state = decision(balance, allow, 0.0, False, policy_floor(policy))
    ttf = time_to_failure(balance, allow, recs, now, docs_per_day, avg_doc_cost, volume_multiplier)
    d_tier = debt_tier(debt, base)
    p_tier = p_ai_tier(a['p_ai'])
    return {**a, 'vendor_id': vendor['vendor_id'], 'name': vendor['vendor_name'], 'base': base, 'debt': debt,
            'debt_tier': d_tier, 'p_ai_tier': p_tier, 'spent_window': spent, 'balance': balance,
            'remaining_frac': balance / allow if allow else 0.0, 'state': state, 'state_tier': STATE_TIER[state],
            'ttf': ttf, 'overall_tier': worst(p_tier, d_tier, STATE_TIER[state]), 'policy': policy,
            'records': recs, 'now': now}


def evaluate_document(state, doc_name, doc, task_key, policy=None, undetected_risk=0.0, ocr_penalty=0.0,
                      unknown_regions=0, human_approved=False, policy_doc=None):
    """Plan redaction for this destination, apply the brake, explain it, and build the receipt. Writes nothing."""
    extra = tuple(policy.get('extra_redact', [])) if policy else ()
    plan = plan_redaction(doc, task_key, state['p_ai'], policy_doc, extra, undetected_risk, ocr_penalty, unknown_regions)
    floor = policy_floor(policy)
    d = decision(state['balance'], state['allowance'], plan['cost'], plan['unresolved_gov_id'], floor)
    why = decision_reasons(d, state['balance'], state['allowance'], plan['cost'], plan['unresolved_gov_id'],
                           state['p_ai'], forced_review=bool(floor))
    spends = d in SPEND_STATES or (d == 'REVIEW' and human_approved)
    after = state['balance'] - plan['cost'] if spends else state['balance']
    receipt = make_receipt(state['name'], doc_name, plan['task'], plan, state['p_ai'], state['balance'], after, d, why,
                           approved_policy=describe_policy(policy))
    return {'plan': plan, 'decision': d, 'why': why, 'receipt': receipt, 'cost': plan['cost'], 'spends': spends,
            'balance_after': after, 'remaining_after': after / state['allowance'] if state['allowance'] else 0.0,
            'dna_before': data_dna(doc['entities']), 'dna_after': data_dna(plan['entities'], exposure_key='exposure')}


def commit(ledger, state, result, doc_name, synthetic=False):
    """Append the decision to the tamper-evident ledger. BLOCK/unapproved REVIEW are logged with zero spend."""
    cost = result['cost'] if result['spends'] else 0.0
    meta = {'decision': result['decision'], 'receipt_hash': result['receipt']['receipt_hash'],
            'proposed_cost': round(result['cost'], 4), 'p_ai': round(state['p_ai'], 4), 'synthetic': synthetic}
    return ledger.append(state['name'], doc_name, cost, state['balance'], state['balance'] - cost, meta)
