"""What-if simulator and Privacy Fire Drill.

SIMULATION ONLY: every scenario works on deep copies and never reads or writes the ledger,
evidence overlay or approved policies. Callers pass the current ledger *records* in; nothing is written back.

Risk index = projected 30-day exposure points / leak budget. "Risk increase" is the relative change in that index.
"""
import copy, math
from .config import WEIGHTS, BRAKE
from .twin import FLOW_PROVEN
from .engine import vendor_state, evaluate_document

KINDS = {
    'add_llm': 'Vendor adds an LLM',
    'add_subprocessor': 'A new sub-processor appears',
    'raise_p_ai': 'Chance of AI goes up',
    'remove_gov_id': 'Government ID is removed from the file',
    'detector_miss': 'We miss a personal field',
    'volume': 'We send more (or fewer) files',
    'ocr_drop': 'The scan is harder to read',
    'stale': 'The evidence is old',
    'ai_reachable': 'An unknown destination turns out to be AI',
}
SIM_TAG = 'SIMULATED (what-if) - not real evidence'


def _row(parent, target, kind, tag, label):
    return {'parent': parent, 'target': target, 'target_kind': kind, 'tag': tag, 'age_days': 1,
            'claim_type': FLOW_PROVEN, 'source': 'What-if simulator', 'url': 'n/a', 'date': 'n/a',
            'evidence_type': SIM_TAG, 'excerpt': label, 'confidence': 'n/a', 'verified': False}


def apply_scenario(scn, vendor, rows, doc):
    """-> (rows, doc, ctx) copies with the scenario applied."""
    rows, doc = copy.deepcopy(rows), copy.deepcopy(doc)
    ctx = {'undetected': 0.0, 'ocr': 0.0, 'regions': 0, 'volume': 1.0, 'p_override': None}
    k, vn = scn['kind'], vendor['vendor_name']
    tag = scn.get('tag', 'Technical proof (docs/SDK/job posts)')
    if k == 'add_llm':
        rows.append(_row(vn, 'Simulated LLM provider', 'ai_provider', tag, 'Vendor adds an LLM provider.'))
    elif k == 'add_subprocessor':
        rows.append(_row(vn, 'Simulated subprocessor', 'service', scn.get('sub_tag', 'Corroborated (2+ independent sources)'), 'New subprocessor.'))
        if scn.get('with_ai', True):
            rows.append(_row('Simulated subprocessor', 'Simulated subprocessor AI', 'ai', tag, 'Subprocessor runs an AI service.'))
    elif k == 'raise_p_ai':
        ctx['p_override'] = float(scn.get('value', 0.6))
    elif k == 'remove_gov_id':
        doc['entities'] = [e for e in doc['entities'] if e['type'] != 'GOV_ID']
        doc['fields_present'] = [f for f in doc['fields_present'] if f != 'gov_id']
    elif k == 'detector_miss':
        t, n = scn.get('entity_type', 'BANK'), int(scn.get('count', 1))
        ctx['undetected'] = WEIGHTS.get(t, 1) * n          # a missed entity is fully exposed
    elif k == 'volume':
        ctx['volume'] = float(scn.get('multiplier', 2.0))
    elif k == 'ocr_drop':
        ctx['ocr'] = float(scn.get('penalty', 5.0))
        ctx['regions'] = int(scn.get('regions', 2))
    elif k == 'stale':
        for r in rows:
            r['age_days'] = r.get('age_days', 0) + int(scn.get('extra_days', 400))
    elif k == 'ai_reachable':
        for r in rows:
            if r.get('target_kind') == 'unknown':
                r['target_kind'] = 'ai_provider'
                r['tag'] = scn.get('tag', 'Corroborated (2+ independent sources)')
                r['claim_type'] = FLOW_PROVEN
                r['age_days'] = 1
    else:
        raise ValueError(f'unknown scenario kind {k}')
    return rows, doc, ctx


def _metrics(vendor, rows, doc, ctx, task_key, records, base, trials, flow_discount, policy, now, docs_per_day):
    st = vendor_state(vendor, rows, records=records, base=base, trials=trials, flow_discount=flow_discount, now=now,
                      policy=policy, docs_per_day=docs_per_day, volume_multiplier=ctx['volume'], p_ai_override=ctx['p_override'])
    ev = evaluate_document(st, 'simulated_document', doc, task_key, policy, ctx['undetected'], ctx['ocr'], ctx['regions'])
    floor = BRAKE['REVIEW'] * st['allowance']
    docs_left = max(0.0, (st['balance'] - floor) / ev['cost']) if ev['cost'] > 0 else math.inf
    per_day = (docs_per_day or 10) * ctx['volume']
    exposure30 = ev['cost'] * per_day * 30
    idx = exposure30 / st['allowance'] if st['allowance'] > 0 else math.inf
    return {'p_ai': st['p_ai'], 'tier': st['p_ai_tier'], 'allowance': st['allowance'], 'balance': st['balance'],
            'doc_cost': ev['cost'], 'decision': ev['decision'], 'reasons': ev['why']['reasons'], 'docs_until_block': docs_left,
            'exposure_30d': exposure30, 'risk_index': idx, 'ttf_days': st['ttf']['days'], 'blast': st['blast'],
            'unresolved_gov_id': ev['plan']['unresolved_gov_id']}


MITIGATION = {
    'add_llm': 'Tighten redaction (BANK/GOV_ID always redacted), require human review, request AI contract clauses and a sub-processor list.',
    'add_subprocessor': 'Request the sub-processor register, re-run the twin with the new evidence, and approve the proposed stricter policy.',
    'raise_p_ai': 'Re-verify the evidence behind the increase; until confirmed, apply the proposed stricter policy.',
    'remove_gov_id': 'Data minimisation: stop sending government IDs where the task does not need them.',
    'detector_miss': 'Add a second-pass detector and regex hard-checks for ID/account patterns; hard-block on any pattern hit.',
    'volume': 'Rate-limit transfers, batch for review, or raise the budget only after the AI-exposure evidence is re-checked.',
    'ocr_drop': 'Route low-quality scans to manual review; re-OCR before release.',
    'stale': 'Re-verify the evidence. NOTE: aging lowers P_ai in the model, which loosens the budget without any real reduction in risk.',
    'ai_reachable': 'Ask the vendor to name the provider; treat as an AI destination (strict redaction) until resolved.',
}


def run_scenario(vendor, rows, scn, doc, task_key, records=(), base=1000.0, trials=3000, flow_discount=1.0,
                 policy=None, now=None, docs_per_day=10):
    records = list(records)
    cur = _metrics(vendor, rows, doc, {'undetected': 0.0, 'ocr': 0.0, 'regions': 0, 'volume': 1.0, 'p_override': None},
                   task_key, records, base, trials, flow_discount, policy, now, docs_per_day)
    r2, d2, ctx = apply_scenario(scn, vendor, rows, doc)
    new = _metrics(vendor, r2, d2, ctx, task_key, records, base, trials, flow_discount, policy, now, docs_per_day)
    ci, ni = cur['risk_index'], new['risk_index']
    inc = (ni / ci - 1) if ci not in (0, math.inf) and ni != math.inf else (math.inf if ni == math.inf and ci != math.inf else 0.0)
    return {'scenario': scn, 'label': KINDS[scn['kind']], 'current': cur, 'simulated': new, 'risk_increase': inc,
            'budget_change_pct': (new['allowance'] / cur['allowance'] - 1) if cur['allowance'] else 0.0,
            'decision_changed': cur['decision'] != new['decision'],
            'documents_affected': max(0.0, (cur['docs_until_block'] if cur['docs_until_block'] != math.inf else 0) -
                                      (new['docs_until_block'] if new['docs_until_block'] != math.inf else 0)),
            'exposure_delta': new['exposure_30d'] - cur['exposure_30d'], 'mitigation': MITIGATION[scn['kind']],
            'false_assurance': scn['kind'] == 'stale' and new['allowance'] > cur['allowance']}


FIRE_DRILL = [
    {'kind': 'detector_miss', 'entity_type': 'BANK', 'count': 1},
    {'kind': 'add_subprocessor', 'with_ai': True},
    {'kind': 'ai_reachable'},
    {'kind': 'ocr_drop', 'penalty': 5.0, 'regions': 2},
    {'kind': 'volume', 'multiplier': 3.0},
    {'kind': 'stale', 'extra_days': 400},
]


def fire_drill(vendor, rows, doc, task_key, records=(), **kw):
    return [run_scenario(vendor, rows, s, doc, task_key, records, **kw) for s in FIRE_DRILL]
